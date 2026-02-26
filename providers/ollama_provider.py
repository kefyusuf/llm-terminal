import re

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
            fit_str, mode_str, _ = calculate_fit(size_gb, specs)
            quant = infer_quant_from_name(model_name, default="GGUF")

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
                    "size_source": "estimated",
                    "mode": mode_str,
                    "fit": fit_str,
                    "size": f"~{size_gb} GB",
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
