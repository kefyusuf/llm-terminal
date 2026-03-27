import re
import threading

import requests
from bs4 import BeautifulSoup

import cache_db
import config
from scoring import enrich_result_with_scores
from utils import (
    calculate_fit,
    determine_use_case,
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    infer_quant_from_name,
    parse_retry_after_seconds,
)

_ollama_meta_cache_lock = threading.Lock()


def _init_ollama_cache():
    cache_db.init_db()


_init_ollama_cache()


def get_installed_ollama_models():
    """Return a list of locally installed Ollama model name prefixes (lowercase).

    Queries the Ollama REST API at ``http://localhost:11434``.
    Returns an empty list if Ollama is not running or the request fails.
    """
    try:
        response = requests.get(f"{config.settings.ollama_api_base}/api/tags", timeout=1)
        if response.status_code == 200:
            return [
                model["name"].split(":")[0].lower() for model in response.json().get("models", [])
            ]
    except (requests.RequestException, ValueError):
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
        detail_response = requests.get(
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
    except requests.RequestException:
        metadata = None

    if metadata is not None:
        with _ollama_meta_cache_lock:
            cache_db.set_model_cache("ollama", cache_key, metadata)

    return metadata


def search_ollama_models(query, specs, local_models, page=0, page_size=15):
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
    """
    results = []
    errors = []
    found_keys = set()
    html_text = ""

    try:
        # Ollama doesn't support page-based pagination via URL
        # Always fetch from page 1 and get all results
        url = f"https://ollama.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=config.settings.ollama_timeout)

        if response.status_code == 429:
            retry_after = _retry_after_from_response(response)
            if retry_after is not None:
                errors.append(f"Ollama registry rate-limited (429). Retry in {retry_after}s.")
            else:
                errors.append("Ollama registry rate-limited (429). Retry shortly.")
            return results, errors
        if response.status_code >= 500:
            errors.append(f"Ollama registry unavailable (HTTP {response.status_code}).")
            return results, errors
        if response.status_code != 200:
            errors.append(f"Ollama registry request failed (HTTP {response.status_code}).")
            return results, errors

        html_text = response.text
        soup = BeautifulSoup(html_text, "html.parser")
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

            full_text = anchor.get_text(strip=True)
            pulls = re.search(r"(\d+(?:\.\d+)?[KM]?)\s*Pulls", full_text, re.IGNORECASE)
            if not pulls:
                parent = anchor.find_parent("li")
                if parent:
                    pulls = re.search(
                        r"(\d+(?:\.\d+)?[KM]?)\s*Pulls",
                        parent.get_text(strip=True),
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

            metadata = get_ollama_model_metadata(model_name)
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
    except requests.Timeout:
        errors.append("Ollama registry request timed out.")
    except requests.ConnectionError:
        errors.append("Ollama registry unreachable. Check network connectivity.")
    except requests.RequestException as exc:
        errors.append(f"Ollama search failed: {exc}")
    except (ValueError, AttributeError) as exc:
        errors.append(f"Ollama parse failed: {exc}")

    # Ollama returns all results at once (no pagination support)
    # Slice to page_size for consistency with HF pagination
    limited_results = results[:page_size] if len(results) > page_size else results

    # has_more_pages is True if we have more results than page_size
    has_more_pages = len(results) > page_size

    return limited_results, errors, has_more_pages
