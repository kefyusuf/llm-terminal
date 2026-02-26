import re

import requests
from bs4 import BeautifulSoup

from utils import (
    calculate_fit,
    determine_use_case,
    estimate_model_size_gb,
    extract_params,
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


def search_ollama_models(query, specs, local_models):
    results = []
    errors = []
    found_keys = set()

    try:
        url = f"https://ollama.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
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
            size_gb = estimate_model_size_gb(model_name)
            fit_str, mode_str, _ = calculate_fit(size_gb, specs)

            results.append(
                {
                    "inst": inst,
                    "source": "Ollama",
                    "provider": "Ollama Registry",
                    "id": model_name,
                    "name": model_name,
                    "params": params,
                    "use_case": use_case,
                    "score": score_str,
                    "quant": "Q4_0",
                    "mode": mode_str,
                    "fit": fit_str,
                    "size": f"~{size_gb} GB",
                }
            )
    except requests.RequestException as exc:
        errors.append(f"Ollama search failed: {exc}")
    except (ValueError, AttributeError) as exc:
        errors.append(f"Ollama parse failed: {exc}")

    return results, errors
