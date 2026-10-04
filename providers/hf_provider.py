import time

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException

from core import cache_db
from core.artifacts import hf_artifact_metadata
from core.errors import ProviderError
from core.scoring import enrich_result_with_scores
from core.utils import (
    calculate_fit,
    determine_use_case,
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    format_likes,
    infer_quant_from_name,
    parse_retry_after_seconds,
)
from providers.base import SearchResult
from providers.capabilities import get_provider_capabilities

_HF_MAX_ATTEMPTS = 3
_HF_BACKOFF_BASE_SECONDS = 0.5
_HF_BACKOFF_MAX_SECONDS = 4.0


class HuggingFaceProvider:
    """Class adapter wrapping the module-level :func:`search_hf_models`.

    The free function is preserved for backward compatibility; the
    class form lets the orchestrator treat HF the same as every other
    :class:`BaseProvider` subclass. ``refresh_installed()`` is a no-op
    since HF has no local-install concept; ``enrich()`` is the
    :func:`enrich_hf_model_details` function.
    """

    slug = "huggingface"
    display_name = "Hugging Face"

    def __init__(self, model_info_cache: dict | None = None):
        self.model_info_cache = model_info_cache if model_info_cache is not None else {}

    def detect(self) -> bool:
        return True

    def search(
        self,
        query: str,
        specs: dict,
        limit: int = 15,
        *,
        page: int = 0,
        hf_token: str | None = None,
        **kwargs,
    ) -> SearchResult:
        """
        Search Hugging Face models matching the query and hardware specifications.

        Parameters:
            page (int): Zero-based result page to retrieve.

        Returns:
            SearchResult: Matching models, any search errors, pagination information,
            and additive structured diagnostics.
        """
        offset = page * limit
        structured_errors: list[ProviderError] = []
        results, errors, has_more_pages = search_hf_models(
            query,
            specs,
            self.model_info_cache,
            limit=limit,
            offset=offset,
            hf_token=hf_token,
            lookahead=1,
            return_page_info=True,
            _structured_error_sink=structured_errors.append,
        )
        return SearchResult(
            results=results[:limit],
            errors=list(errors),
            has_more_pages=has_more_pages,
            structured_errors=structured_errors,
        )

    def list_installed(self) -> list[str]:
        return []

    def enrich(self, model: dict, specs: dict) -> dict:
        return enrich_hf_model_details(model, specs, self.model_info_cache)


def _repo_id_from_model(model):
    """Return the HuggingFace repository id string from a model object."""
    return getattr(model, "modelId", None) or getattr(model, "id", None)


def _select_preferred_gguf(siblings):
    """Choose the preferred GGUF file from a list of repo sibling objects.

    Preference order: Q4_K_M → Q4_0 → Q5_K_M → any ``.gguf``.
    Returns ``None`` if no GGUF file is found.
    """
    filenames = [item.rfilename for item in siblings if getattr(item, "rfilename", None)]
    priorities = ["Q4_K_M.gguf", "Q4_0.gguf", "Q5_K_M.gguf"]

    for suffix in priorities:
        target = next((name for name in filenames if suffix in name), None)
        if target:
            return target

    return next((name for name in filenames if name.endswith(".gguf")), None)


def classify_hidden_gem(downloads, likes):
    """Classify a model as a hidden gem based on its download and like counts.

    A hidden gem has significant downloads but low public visibility (few
    likes relative to download volume).

    Returns:
        ``(is_gem: bool, gem_score: float)`` — score is higher for more
        gem-like models.
    """
    if downloads < 5000:
        return False, 0.0
    if likes > 200:
        return False, 0.0
    like_ratio = likes / max(downloads, 1)
    if like_ratio > 0.005:
        return False, 0.0
    score = downloads / (likes + 1)
    return True, score


def _get_status_code(exc):
    """Extract the HTTP status code from a HuggingFace exception, or ``None``."""
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) is not None:
        return response.status_code
    return getattr(exc, "status_code", None)


def _get_retry_after_seconds(exc):
    """Extract the ``Retry-After`` delay in seconds from a HuggingFace exception."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    return parse_retry_after_seconds(headers.get("Retry-After"))


def _format_hf_http_error(exc):
    """Return a user-friendly error message for a HuggingFace HTTP error."""
    status = _get_status_code(exc)
    retry_after = _get_retry_after_seconds(exc)
    if status == 429:
        if retry_after is not None:
            return f"Hugging Face rate-limited (429). Retry in {retry_after}s."
        return "Hugging Face rate-limited (429). Retry shortly."
    if status is not None:
        return f"Hugging Face request failed (HTTP {status})."
    return f"Hugging Face request failed: {exc}"


def _is_retryable_hf_http_error(exc):
    """Return whether a Hugging Face HTTP error is transient and safe to retry."""
    status = _get_status_code(exc)
    return status == 429 or (status is not None and 500 <= status <= 599)


def _hf_retry_delay_seconds(exc, attempt):
    """Return Retry-After or bounded exponential delay before the next attempt."""
    if isinstance(exc, HfHubHTTPError):
        retry_after = _get_retry_after_seconds(exc)
        if retry_after is not None:
            return max(float(retry_after), 0.0)
    return min(
        _HF_BACKOFF_BASE_SECONDS * (2 ** max(attempt - 1, 0)),
        _HF_BACKOFF_MAX_SECONDS,
    )


def _list_hf_models_with_retry(api, **kwargs):
    """List Hugging Face models with bounded retries for transient failures only."""
    for attempt in range(1, _HF_MAX_ATTEMPTS + 1):
        try:
            return list(api.list_models(**kwargs))
        except HfHubHTTPError as exc:
            if attempt >= _HF_MAX_ATTEMPTS or not _is_retryable_hf_http_error(exc):
                raise
            time.sleep(_hf_retry_delay_seconds(exc, attempt))
        except RequestException as exc:
            if attempt >= _HF_MAX_ATTEMPTS:
                raise
            time.sleep(_hf_retry_delay_seconds(exc, attempt))

    raise RuntimeError("Hugging Face retry loop exhausted unexpectedly")


def search_hf_models(
    query,
    specs,
    model_info_cache,
    limit=15,
    offset=0,
    hf_token=None,
    lookahead=0,
    return_page_info=False,
    _structured_error_sink=None,
):
    """
    Search Hugging Face for GGUF models matching a query.

    Args:
        query: Free-text search string.
        specs: Hardware specification dictionary used to determine model fit.
        model_info_cache: Shared cache for Hugging Face repository metadata.
        limit: Maximum number of models to process.
        offset: Number of raw search results to skip.
        hf_token: Optional Hugging Face API token.
        lookahead: Number of additional raw results to fetch for page-boundary detection.
        return_page_info: Whether to include the page-availability flag in the return value.
        _structured_error_sink: Optional internal callback receiving ``ProviderError`` values.

    Returns:
        A tuple containing the parsed model results and parsing or request errors.
        When `return_page_info` is true, includes a third boolean indicating whether
        additional raw results are available. Structured diagnostics are additive and
        do not change this public tuple shape.
    """
    results = []
    errors = []
    found_keys = set()
    window_size = limit + max(int(lookahead), 0)

    def _record_error(
        message,
        *,
        code,
        retryable=False,
        status_code=None,
        retry_after_seconds=None,
    ):
        """Append the legacy string and optionally emit a structured diagnostic."""
        errors.append(message)
        if _structured_error_sink is not None:
            _structured_error_sink(
                ProviderError(
                    provider=HuggingFaceProvider.slug,
                    code=code,
                    message=message,
                    retryable=retryable,
                    status_code=status_code,
                    retry_after_seconds=retry_after_seconds,
                )
            )

    def _return(has_more_pages=False):
        """
        Package search results with optional pagination information.

        Parameters:
            has_more_pages (bool): Whether additional result pages are available.

        Returns:
            tuple: Search results and errors, optionally followed by a pagination flag.
        """
        if return_page_info:
            return results, errors, has_more_pages
        return results, errors

    try:
        api = HfApi(token=hf_token) if hf_token else HfApi()
        raw_models = _list_hf_models_with_retry(
            api,
            search=query,
            sort="downloads",
            limit=offset + window_size,
            filter="gguf",
            expand=["likes", "siblings", "downloads", "sha", "cardData"],
        )
        raw_window = raw_models[offset : offset + window_size]
    except HfHubHTTPError as exc:
        status = _get_status_code(exc)
        retry_after = _get_retry_after_seconds(exc)
        message = _format_hf_http_error(exc)
        _record_error(
            message,
            code="rate_limited" if status == 429 else "http_error",
            retryable=_is_retryable_hf_http_error(exc),
            status_code=status,
            retry_after_seconds=float(retry_after) if retry_after is not None else None,
        )
        return _return()
    except RequestException as exc:
        message = f"Hugging Face search failed: {exc}"
        _record_error(message, code="transport_error", retryable=True)
        return _return()
    except ValueError as exc:
        message = f"Hugging Face search failed: {exc}"
        _record_error(message, code="search_error", retryable=False)
        return _return()
    except OSError as exc:
        message = f"Hugging Face search failed: {exc}"
        _record_error(message, code="transport_error", retryable=True)
        return _return()

    has_more_pages = len(raw_window) > limit
    page_models = raw_window[:limit]

    hf_lists_installed = get_provider_capabilities(HuggingFaceProvider.slug).lists_installed

    for model in page_models:
        try:
            repo_id = _repo_id_from_model(model)
            if not repo_id:
                raise ValueError("missing model repository id")

            publisher = repo_id.split("/")[0]
            provider = publisher[:15]
            name = repo_id.split("/")[-1]
            unique_key = f"Hugging Face:{repo_id.lower()}"
            if unique_key in found_keys:
                continue
            found_keys.add(unique_key)

            params = extract_params(name)
            use_case = determine_use_case(name)
            use_case_key = determine_use_case_key(name)
            likes = getattr(model, "likes", 0)
            downloads = int(getattr(model, "downloads", 0) or 0)
            is_hidden_gem, gem_score = classify_hidden_gem(downloads, likes)
            score_str = f"[red]❤️ {format_likes(likes)}[/red]" if likes > 0 else "[grey50]-[/grey50]"
            if is_hidden_gem:
                score_str = f"{score_str} [yellow]💎[/yellow]"

            quant = infer_quant_from_name(name, default="GGUF")
            size = estimate_model_size_gb(name)
            siblings = getattr(model, "siblings", None) or []
            target = _select_preferred_gguf(siblings)

            if target:
                quant = target.split(".")[-2] if len(target.split(".")) > 2 else "GGUF"
                if "gguf" in quant.lower():
                    quant = "GGUF"

            fit_str, mode_str, _ = calculate_fit(size, specs)
            result_dict = {
                "inst": "[grey37]-[/grey37]"
                if hf_lists_installed
                else "[grey37]N/A[/grey37]",
                "source": "Hugging Face",
                "provider": provider,
                "publisher": publisher,
                "id": repo_id,
                "name": name,
                "params": params,
                "use_case": use_case,
                "use_case_key": use_case_key,
                "score": score_str,
                "likes": likes,
                "downloads": downloads,
                "is_hidden_gem": is_hidden_gem,
                "gem_score": gem_score,
                "quant": quant,
                "target_file": target,
                "artifact_metadata": hf_artifact_metadata(repo_id, target, model),
                "resolved_revision": getattr(model, "sha", None),
                "size_source": "estimated",
                "mode": mode_str,
                "fit": fit_str,
                "size": f"~{size:.1f} GB",
                "_size_gb": size,
            }
            enrich_result_with_scores(result_dict, specs)
            results.append(result_dict)
        except (
            AttributeError,
            TypeError,
            ValueError,
            HfHubHTTPError,
            RequestException,
            OSError,
        ) as exc:
            message = f"Hugging Face model parse failed: {exc}"
            _record_error(message, code="parse_error", retryable=False)

    return _return(has_more_pages)


def enrich_hf_model_details(model, specs, model_info_cache):
    """
    Enrich a Hugging Face model result with exact GGUF file details.

    Parameters:
        model (dict): Model result to update in place.
        specs: Hardware specifications used to calculate fit and operating mode.
        model_info_cache: Optional cache for Hugging Face repository metadata.

    Returns:
        dict: The updated model result, or the original result when repository or file metadata is unavailable.
    """
    repo_id = model.get("id")
    if not repo_id:
        return model

    target = model.get("target_file")

    cached = cache_db.get_model_cache("huggingface", repo_id)
    if cached is not None and (not target or target == cached.get("target_file")):
        model["resolved_revision"] = cached.get("resolved_revision")
        model["artifact_metadata"] = cached.get("artifact_metadata")
        size = cached.get("size_gb")
        if size is not None:
            fit_str, mode_str, _ = calculate_fit(size, specs)
            model["size"] = f"{size:.1f} GB"
            model["fit"] = fit_str
            model["mode"] = mode_str
            model["size_source"] = "exact"
        if cached.get("target_file"):
            target = cached.get("target_file")
            if target:
                quant = target.split(".")[-2] if len(target.split(".")) > 2 else "GGUF"
                if "gguf" in quant.lower():
                    quant = "GGUF"
                model["quant"] = quant
                model["target_file"] = target
        return model

    try:
        api = HfApi()
        info = api.model_info(repo_id, files_metadata=True)
        if model_info_cache is not None:
            model_info_cache[repo_id] = info

        siblings = info.siblings or []
        target = target or _select_preferred_gguf(siblings)
        if not target:
            return model

        metadata = next((item for item in siblings if item.rfilename == target), None)
        model["resolved_revision"] = getattr(info, "sha", None) if metadata else None
        model["artifact_metadata"] = hf_artifact_metadata(repo_id, target, info)
        size = None
        if metadata and metadata.size:
            size = metadata.size / (1024**3)
            fit_str, mode_str, _ = calculate_fit(size, specs)
            model["size"] = f"{size:.1f} GB"
            model["fit"] = fit_str
            model["mode"] = mode_str
            model["size_source"] = "exact"

        quant = target.split(".")[-2] if len(target.split(".")) > 2 else "GGUF"
        if "gguf" in quant.lower():
            quant = "GGUF"
        model["quant"] = quant
        model["target_file"] = target

        cache_db.set_model_cache(
            "huggingface",
            repo_id,
            {
                "size_gb": size,
                "target_file": target,
                "resolved_revision": model.get("resolved_revision"),
                "artifact_metadata": model.get("artifact_metadata"),
            },
        )
    except (HfHubHTTPError, RequestException, OSError, ValueError, TypeError):
        return model

    return model
