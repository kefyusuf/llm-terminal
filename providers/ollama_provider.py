import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from bs4 import BeautifulSoup
from requests.exceptions import ConnectionError, RequestException, Timeout

import config
from core import cache_db
from core.errors import ProviderError
from core.http_client import create_session, get_session
from core.scoring import enrich_result_with_scores
from core.utils import (
    calculate_fit,
    determine_use_case,
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    infer_quant_from_name,
    parse_retry_after_seconds,
)
from providers.base import SearchResult


class OllamaProvider:
    """Class adapter wrapping the module-level :func:`search_ollama_models`.

    Holds the local-models set so it can be passed into
    :func:`search_ollama_models` from the polymorphic
    :meth:`search_with_installed` path. ``refresh_installed()`` is
    called by the orchestrator at the start of a search.
    """

    slug = "ollama"
    display_name = "Ollama"
    default_host = "http://localhost:11434"

    def __init__(self):
        self.installed: list[str] = []

    def detect(self) -> bool:
        """Return whether the configured Ollama API is reachable."""
        try:
            response = get_session().get(
                f"{config.settings.ollama_api_base}/api/tags",
                timeout=1,
            )
            return response.status_code == 200
        except RequestException:
            return False

    def refresh_installed(self) -> None:
        self.installed = get_installed_ollama_models()

    def search(
        self,
        query: str,
        specs: dict,
        limit: int = 20,
        *,
        page: int = 0,
        **kwargs,
    ) -> SearchResult:
        structured_errors: list[ProviderError] = []
        results, errors, has_more = search_ollama_models(
            query,
            specs,
            self.installed,
            page=page,
            page_size=limit,
            _structured_error_sink=structured_errors.append,
        )
        return SearchResult(
            results=results,
            errors=errors,
            has_more_pages=has_more,
            structured_errors=structured_errors,
        )

    def list_installed(self) -> list[str]:
        return self.installed

    def search_with_installed(
        self,
        query: str,
        specs: dict,
        limit: int = 20,
        *,
        page: int = 0,
        **kwargs,
    ) -> SearchResult:
        if not self.installed:
            self.refresh_installed()
        return self.search(query, specs, limit=limit, page=page, **kwargs)


_ollama_meta_cache_lock = threading.Lock()
_metadata_http = threading.local()


@contextmanager
def _metadata_batch(model_names):
    """Overlap at most four detail requests using search-owned worker clients."""
    sessions = []
    lock = threading.Lock()

    def initialize_worker():
        session = create_session()
        _metadata_http.session = session
        with lock:
            sessions.append(session)

    try:
        with ThreadPoolExecutor(max_workers=4, initializer=initialize_worker) as executor:
            yield executor.map(get_ollama_model_metadata, model_names)
    finally:
        for session in sessions:
            session.close()


def _init_ollama_cache():
    cache_db.init_db()


_init_ollama_cache()


def get_installed_ollama_models():
    """Return a list of locally installed Ollama model name prefixes (lowercase).

    Queries the configured Ollama REST API.
    Returns an empty list if Ollama is not running or the request fails.
    """
    try:
        response = get_session().get(f"{config.settings.ollama_api_base}/api/tags", timeout=1)
        if response.status_code == 200:
            return [
                model["name"].split(":")[0].lower() for model in response.json().get("models", [])
            ]
    except (RequestException, ValueError):
        return []
    return []


def _retry_after_from_response(response):
    """Return the ``Retry-After`` delay in seconds from an HTTP response, or ``None``."""
    return parse_retry_after_seconds(response.headers.get("Retry-After"))


def _parse_size_gb(size_text):
    """Parse a human-readable size string (e.g. ``"4.7GB"`` or ``"780 MB"``) into GB.

    Returns a ``float``, or ``None`` when the string cannot be parsed.
    """
    text = (size_text or "").strip().upper().replace(" ", "")
    match = re.search(r"(\d+(?:\.\d+)?)(GB|MB)", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "GB":
        return value
    if unit == "MB":
        return value / 1024.0
    return None


def _extract_models_table_rows(html_text, model_name=None):
    """Extract model-variant rows from the Ollama library HTML page.

    Parses table rows first; falls back to card-style anchor links when no
    suitable table is found.  Returns a list of dicts with keys
    ``name``, ``size_text``, ``size_gb``.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "name" not in headers or "size" not in headers:
            continue

        name_index = headers.index("name")
        size_index = headers.index("size")
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            if len(cells) <= max(name_index, size_index):
                continue
            model_variant = cells[name_index].get_text(strip=True)
            size_text = cells[size_index].get_text(strip=True)
            rows.append(
                {
                    "name": model_variant,
                    "size_text": size_text,
                    "size_gb": _parse_size_gb(size_text),
                }
            )
        if rows:
            return rows

    if model_name:
        model_rows = []
        model_prefix = f"/library/{model_name.lower()}:"
        for anchor in soup.find_all("a", href=True):
            raw_href = anchor.get("href")
            if not isinstance(raw_href, str):
                continue
            href = raw_href.strip()
            if not href.lower().startswith(model_prefix):
                continue
            variant_name = href.split("/library/", maxsplit=1)[-1]
            anchor_text = anchor.get_text(" ", strip=True)
            size_match = re.search(r"(\d+(?:\.\d+)?)\s*GB", anchor_text, re.IGNORECASE)
            size_text = f"{size_match.group(1)}GB" if size_match else ""
            model_rows.append(
                {
                    "name": variant_name,
                    "size_text": size_text,
                    "size_gb": _parse_size_gb(size_text),
                }
            )
        if model_rows:
            return model_rows
    return []


def _select_preferred_model_variant(model_name, rows):
    """Select the best model variant from *rows*, preferring ``:latest`` tags.

    Returns the first matching row dict with a valid ``size_gb``, or
    ``None`` when no suitable variant exists.
    """
    preferred_exact = f"{model_name}:latest"
    for row in rows:
        if row["name"].lower() == preferred_exact.lower() and row.get("size_gb"):
            return row

    for row in rows:
        if row["name"].lower().endswith(":latest") and row.get("size_gb"):
            return row

    for row in rows:
        if row.get("size_gb"):
            return row

    return None


def get_ollama_model_metadata(model_name):
    """Fetch size and quantisation metadata for *model_name* from ollama.com.

    Results are cached in SQLite for 24 hours.
    Returns a dict with keys ``size_gb``, ``size_text``, ``variant``,
    ``quant``, and ``params``, or ``None`` on failure.
    """
    cache_key = model_name.lower()

    cached = cache_db.get_model_cache("ollama", cache_key)
    if cached is not None:
        return cached

    metadata = None
    try:
        detail_url = f"https://ollama.com/library/{model_name}"
        session = getattr(_metadata_http, "session", None) or get_session()
        detail_response = session.get(
            detail_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=config.settings.ollama_timeout,
        )
        if detail_response.status_code == 200:
            rows = _extract_models_table_rows(detail_response.text, model_name=model_name)
            chosen = _select_preferred_model_variant(model_name, rows)
            if chosen and chosen.get("size_gb"):
                variant_name = chosen.get("name", model_name)
                metadata = {
                    "size_gb": chosen["size_gb"],
                    "size_text": chosen.get("size_text", ""),
                    "variant": variant_name,
                    "quant": infer_quant_from_name(variant_name, default="GGUF"),
                    "params": extract_params(variant_name),
                }
    except RequestException:
        metadata = None

    if metadata is not None:
        with _ollama_meta_cache_lock:
            cache_db.set_model_cache("ollama", cache_key, metadata)

    return metadata


def search_ollama_models(
    query,
    specs,
    local_models,
    page=0,
    page_size=15,
    _structured_error_sink=None,
):
    """Search the Ollama model registry for models matching *query*.

    Scrapes ``ollama.com/search``.  Returns
    ``(results: list[dict], errors: list[str], has_more_pages: bool)``.

    Note: Ollama uses htmx infinite scroll, not traditional pagination.
    The page parameter is ignored - we always fetch all results.

    Args:
        query: Free-text search string.
        specs: Hardware specification dict.
        local_models: List of locally installed models.
        page: Page number (ignored).
        page_size: Results per page (used for slicing).
        _structured_error_sink: Optional internal callback receiving ``ProviderError`` values.
    """
    results = []
    errors = []
    found_keys = set()
    html_text = ""

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
                    provider=OllamaProvider.slug,
                    code=code,
                    message=message,
                    retryable=retryable,
                    status_code=status_code,
                    retry_after_seconds=retry_after_seconds,
                )
            )

    try:
        # Ollama doesn't support page-based pagination via URL
        # Always fetch from page 1 and get all results
        url = f"https://ollama.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = get_session().get(url, headers=headers, timeout=config.settings.ollama_timeout)

        if response.status_code == 429:
            retry_after = _retry_after_from_response(response)
            if retry_after is not None:
                message = f"Ollama registry rate-limited (429). Retry in {retry_after}s."
            else:
                message = "Ollama registry rate-limited (429). Retry shortly."
            _record_error(
                message,
                code="rate_limited",
                retryable=True,
                status_code=429,
                retry_after_seconds=float(retry_after) if retry_after is not None else None,
            )
            return results, errors, False
        if response.status_code >= 500:
            message = f"Ollama registry unavailable (HTTP {response.status_code})."
            _record_error(
                message,
                code="http_error",
                retryable=True,
                status_code=response.status_code,
            )
            return results, errors, False
        if response.status_code != 200:
            message = f"Ollama registry request failed (HTTP {response.status_code})."
            _record_error(
                message,
                code="http_error",
                retryable=False,
                status_code=response.status_code,
            )
            return results, errors, False

        html_text = response.text
        soup = BeautifulSoup(html_text, "html.parser")
        candidates = []
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            if not isinstance(href, str):
                continue
            if not href.startswith("/library/") or "/blog/" in href or "/tags" in href:
                continue

            model_name = href.replace("/library/", "").strip()
            unique_key = f"Ollama:{model_name}"
            if unique_key in found_keys:
                continue
            found_keys.add(unique_key)

            candidates.append((anchor, model_name))

        with _metadata_batch([name for _, name in candidates]) as metadata_results:
            for (anchor, model_name), metadata in zip(candidates, metadata_results, strict=True):
                full_text = anchor.get_text(" ", strip=True)
                pulls = re.search(r"(\d+(?:\.\d+)?[KM]?)\s*Pulls", full_text, re.IGNORECASE)
                if not pulls:
                    parent = anchor.find_parent("li")
                    if parent:
                        pulls = re.search(
                            r"(\d+(?:\.\d+)?[KM]?)\s*Pulls",
                            parent.get_text(" ", strip=True),
                            re.IGNORECASE,
                        )

                score_str = f"[cyan]📥 {pulls.group(1)}[/cyan]" if pulls else "[grey50]-[/grey50]"
                params = extract_params(model_name)
                inst = (
                    "[green]✔[/green]" if model_name.lower() in local_models else "[grey37]-[/grey37]"
                )
                use_case = determine_use_case(model_name)
                use_case_key = determine_use_case_key(model_name)
                size_gb = estimate_model_size_gb(model_name)
                quant = infer_quant_from_name(model_name, default="GGUF")
                size_source = "estimated"

                if metadata and metadata.get("size_gb"):
                    size_gb = metadata["size_gb"]
                    size_source = "exact"
                    quant = metadata.get("quant", quant)
                    meta_params = metadata.get("params", "-")
                    if params == "-" and meta_params != "-":
                        params = meta_params

                fit_str, mode_str, _ = calculate_fit(size_gb, specs)

                result_dict = {
                    "inst": inst,
                    "source": "Ollama",
                    "provider": "Ollama Registry",
                    "publisher": "ollama",
                    "id": model_name,
                    "name": model_name,
                    "params": params,
                    "use_case": use_case,
                    "use_case_key": use_case_key,
                    "score": score_str,
                    "likes": 0,
                    "downloads": 0,
                    "is_hidden_gem": False,
                    "gem_score": 0.0,
                    "quant": quant,
                    "size_source": size_source,
                    "mode": mode_str,
                    "fit": fit_str,
                    "size": (f"{size_gb:.1f} GB" if size_source == "exact" else f"~{size_gb:.1f} GB"),
                    "_size_gb": size_gb,
                }
                enrich_result_with_scores(result_dict, specs)
                results.append(result_dict)

        if not results and "No models found." not in soup.get_text(" ", strip=True):
            _record_error(
                "Ollama registry parse failed: unsupported search page shape.",
                code="parse_error",
                retryable=False,
            )
    except Timeout:
        _record_error("Ollama registry request timed out.", code="timeout", retryable=True)
    except ConnectionError:
        _record_error(
            "Ollama registry unreachable. Check network connectivity.",
            code="transport_error",
            retryable=True,
        )
    except RequestException as exc:
        _record_error(f"Ollama search failed: {exc}", code="transport_error", retryable=True)
    except (ValueError, AttributeError) as exc:
        _record_error(f"Ollama parse failed: {exc}", code="parse_error", retryable=False)

    # Ollama returns all results at once (no page-offset support).
    # Keep the result cap for consistency, but never advertise a next page.
    limited_results = results[:page_size] if len(results) > page_size else results

    return limited_results, errors, False
