"""LM Studio provider — search and download models via LM Studio's local API.

LM Studio runs a local server on ``localhost:1234`` (configurable via
``LMSTUDIO_HOST`` env var) that exposes an OpenAI-compatible API.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from providers import BaseProvider
from scoring import enrich_result_with_scores
from utils import (
    calculate_fit,
    determine_use_case,
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    infer_quant_from_name,
)


class LMStudioProvider(BaseProvider):
    """Provider for LM Studio local model server."""

    slug = "lmstudio"
    display_name = "LM Studio"
    default_host = "http://localhost:1234"

    def __init__(self):
        self.host = os.environ.get("LMSTUDIO_HOST", self.default_host)

    def detect(self) -> bool:
        """Check if LM Studio server is running."""
        try:
            resp = requests.get(f"{self.host}/v1/models", timeout=2)
            return resp.status_code == 200
        except (requests.RequestException, requests.ConnectionError):
            return False

    def search(
        self,
        query: str,
        specs: dict[str, Any],
        limit: int = 15,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Search LM Studio's installed models."""
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            resp = requests.get(f"{self.host}/v1/models", timeout=5)
            if resp.status_code != 200:
                errors.append(f"LM Studio API error (HTTP {resp.status_code})")
                return results, errors

            data = resp.json()
            models = data.get("data", [])

            for model in models:
                model_id = model.get("id", "")
                if query and query != "*" and query.lower() not in model_id.lower():
                    continue

                name = model_id.split("/")[-1] if "/" in model_id else model_id
                publisher = model_id.split("/")[0] if "/" in model_id else "lmstudio"
                params = extract_params(name)
                use_case = determine_use_case(name)
                use_case_key = determine_use_case_key(name)
                quant = infer_quant_from_name(name, default="GGUF")
                size_gb = estimate_model_size_gb(name)

                fit_str, mode_str, _ = calculate_fit(size_gb, specs)

                result = {
                    "inst": "[green]✔[/green]",
                    "source": "LM Studio",
                    "provider": "LM Studio",
                    "publisher": publisher,
                    "id": model_id,
                    "name": name,
                    "params": params,
                    "use_case": use_case,
                    "use_case_key": use_case_key,
                    "score": "[grey50]-[/grey50]",
                    "likes": 0,
                    "downloads": 0,
                    "is_hidden_gem": False,
                    "gem_score": 0.0,
                    "quant": quant,
                    "size_source": "estimated",
                    "mode": mode_str,
                    "fit": fit_str,
                    "size": f"~{size_gb:.1f} GB",
                    "_size_gb": size_gb,
                }
                enrich_result_with_scores(result, specs)
                results.append(result)

                if len(results) >= limit:
                    break

        except requests.ConnectionError:
            errors.append("LM Studio not reachable. Is LM Studio running?")
        except requests.RequestException as exc:
            errors.append(f"LM Studio request failed: {exc}")

        return results, errors

    def list_installed(self) -> list[str]:
        """List installed LM Studio models."""
        try:
            resp = requests.get(f"{self.host}/v1/models", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("id", "").lower() for m in data.get("data", [])]
        except (requests.RequestException, ValueError):
            pass
        return []
