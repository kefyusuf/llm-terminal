import re
import time

import requests
from bs4 import BeautifulSoup

from utils import (
    calculate_fit,
    determine_use_case,
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    infer_quant_from_name,
)


OLLAMA_MODEL_META_CACHE = {}
OLLAMA_MODEL_META_TTL_SECONDS = 900


def get_installed_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            return [
                model["name"].split(":")[0].lower()
                for model in response.json().get("models", [])
            ]
    except (requests.RequestException, ValueError):
        return []
    return []


def _retry_after_from_response(response):
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None


def _parse_size_gb(size_text):
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
            href = (anchor.get("href") or "").strip()
            if not isinstance(href, str):
                continue
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
    cache_key = model_name.lower()
    now = time.time()
    cached = OLLAMA_MODEL_META_CACHE.get(cache_key)
    if cached and (now - cached["timestamp"]) <= OLLAMA_MODEL_META_TTL_SECONDS:
        return cached["value"]

    metadata = None
    try:
        detail_url = f"https://ollama.com/library/{model_name}"
        detail_response = requests.get(
            detail_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        if detail_response.status_code == 200:
            rows = _extract_models_table_rows(
                detail_response.text, model_name=model_name
            )
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

    OLLAMA_MODEL_META_CACHE[cache_key] = {"timestamp": now, "value": metadata}
    return metadata
    try:
        return int(retry_after)
    except (TypeError, ValueError):
        return None


def search_ollama_models(query, specs, local_models):
    results = []
    errors = []
    found_keys = set()

    try:
        url = f"https://ollama.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 429:
            retry_after = _retry_after_from_response(response)
            if retry_after is not None:
                errors.append(
                    f"Ollama registry rate-limited (429). Retry in {retry_after}s."
                )
            else:
                errors.append("Ollama registry rate-limited (429). Retry shortly.")
            return results, errors
        if response.status_code >= 500:
            errors.append(f"Ollama registry unavailable (HTTP {response.status_code}).")
            return results, errors
        if response.status_code != 200:
            errors.append(
                f"Ollama registry request failed (HTTP {response.status_code})."
            )
            return results, errors

        soup = BeautifulSoup(response.text, "html.parser")
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

            score_str = (
                f"[cyan]📥 {pulls.group(1)}[/cyan]" if pulls else "[grey50]-[/grey50]"
            )
            params = extract_params(model_name)
            inst = (
                "[green]✔[/green]"
                if model_name.lower() in local_models
                else "[grey37]-[/grey37]"
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

            results.append(
                {
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
                    "size": (
                        f"{size_gb:.1f} GB"
                        if size_source == "exact"
                        else f"~{size_gb:.1f} GB"
                    ),
                }
            )
    except requests.Timeout:
        errors.append("Ollama registry request timed out.")
    except requests.ConnectionError:
        errors.append("Ollama registry unreachable. Check network connectivity.")
    except requests.RequestException as exc:
        errors.append(f"Ollama search failed: {exc}")
    except (ValueError, AttributeError) as exc:
        errors.append(f"Ollama parse failed: {exc}")

    return results, errors
