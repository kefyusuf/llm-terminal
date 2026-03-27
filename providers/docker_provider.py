"""Docker Model Runner provider — search and manage models via Docker Desktop.

Docker Desktop (4.30+) includes a Model Runner that serves models on
``localhost:12434`` (configurable via ``DOCKER_MODEL_RUNNER_HOST``).
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


class DockerProvider(BaseProvider):
    """Provider for Docker Desktop Model Runner."""

    slug = "docker"
    display_name = "Docker"
    default_host = "http://localhost:12434"

    def __init__(self):
        self.host = os.environ.get("DOCKER_MODEL_RUNNER_HOST", self.default_host)

    def detect(self) -> bool:
        """Check if Docker Model Runner is available."""
        try:
            resp = requests.get(f"{self.host}/models", timeout=2)
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
        """Search Docker Model Runner's available/installed models."""
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            resp = requests.get(f"{self.host}/models", timeout=5)
            if resp.status_code != 200:
                errors.append(f"Docker Model Runner API error (HTTP {resp.status_code})")
                return results, errors

            data = resp.json()
            # Docker Model Runner returns a list or {"models": [...]}
            models = data if isinstance(data, list) else data.get("models", data.get("data", []))

            for model in models:
                if isinstance(model, str):
                    model_id = model
                else:
                    model_id = model.get("id", model.get("name", ""))

                if not model_id:
                    continue
                if query and query != "*" and query.lower() not in model_id.lower():
                    continue

                name = model_id.split("/")[-1] if "/" in model_id else model_id
                publisher = model_id.split("/")[0] if "/" in model_id else "docker"
                params = extract_params(name)
                use_case = determine_use_case(name)
                use_case_key = determine_use_case_key(name)
                quant = infer_quant_from_name(name, default="GGUF")
                size_gb = estimate_model_size_gb(name)

                fit_str, mode_str, _ = calculate_fit(size_gb, specs)

                result = {
                    "inst": "[green]✔[/green]",
                    "source": "Docker",
                    "provider": "Docker",
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
            errors.append("Docker Model Runner not reachable. Is Docker Desktop running?")
        except requests.RequestException as exc:
            errors.append(f"Docker Model Runner request failed: {exc}")

        return results, errors

    def list_installed(self) -> list[str]:
        """List installed Docker models."""
        try:
            resp = requests.get(f"{self.host}/models", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                models = (
                    data if isinstance(data, list) else data.get("models", data.get("data", []))
                )
                ids = []
                for m in models:
                    if isinstance(m, str):
                        ids.append(m.lower())
                    else:
                        ids.append(m.get("id", m.get("name", "")).lower())
                return ids
        except (requests.RequestException, ValueError):
            pass
        return []
