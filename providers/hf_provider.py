import requests
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

import cache_db
from scoring import enrich_result_with_scores
from utils import (
    calculate_fit,
    determine_use_case,
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    format_likes,
    infer_quant_from_name,
    parse_retry_after_seconds,
)


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


def search_hf_models(
    query,
    specs,
    model_info_cache,
    limit=15,
    offset=0,
    hf_token=None,
):
    """Search Hugging Face for GGUF models matching *query*.

    Args:
        query: Free-text search string.
        specs: Hardware specification dict.
        model_info_cache: Shared ``{repo_id: HfApi.model_info}`` cache dict.
        limit: Maximum number of results to return per page.
        offset: Number of results to skip (for pagination).
        hf_token: Optional HuggingFace API token for higher rate limits.

    Returns:
        ``(results: list[dict], errors: list[str], total_count: int)``.
    """
    results = []
    errors = []
    found_keys = set()

    try:
        api = HfApi(token=hf_token) if hf_token else HfApi()
        hf_models_iter = api.list_models(
            search=query,
            sort="downloads",
            limit=limit * 10,
            filter="gguf",
            expand=["likes", "siblings", "downloads"],
        )
        hf_models = list(hf_models_iter)[offset : offset + limit]
    except HfHubHTTPError as exc:
        return results, [_format_hf_http_error(exc)]
    except (requests.RequestException, ValueError, OSError) as exc:
        return results, [f"Hugging Face search failed: {exc}"]

    for model in hf_models:
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
                "inst": "[grey37]-[/grey37]",
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
            requests.RequestException,
            OSError,
        ) as exc:
            errors.append(f"Hugging Face model parse failed: {exc}")

    return results, errors


def enrich_hf_model_details(model, specs, model_info_cache):
    """Enrich a search result *model* dict with exact file size from the HF API.

    Fetches ``model_info`` (with ``files_metadata=True``) to resolve the
    target GGUF file size and update ``size``, ``fit``, ``mode``, and
    ``quant`` fields in-place.  Returns the (possibly unchanged) dict.
    """
    repo_id = model.get("id")
    if not repo_id:
        return model

    target = model.get("target_file")

    cached = cache_db.get_model_cache("huggingface", repo_id)
    if cached is not None:
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
            {"size_gb": size, "target_file": target},
        )
    except (HfHubHTTPError, requests.RequestException, OSError, ValueError, TypeError):
        return model

    return model
