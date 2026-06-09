"""LM Studio provider — search and download models via LM Studio's local API.

LM Studio runs a local server on ``localhost:1234`` (configurable via
``LMSTUDIO_HOST`` env var) that exposes an OpenAI-compatible API.
"""

from __future__ import annotations

import os
from typing import Any

from requests.exceptions import RequestException

from core.http_client import get_session
from providers import BaseProvider
from core.scoring import enrich_result_with_scores
from core.utils import (
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

    def __init__(self, host: str | None = None):
        self.host = host or os.getenv("LMSTUDIO_HOST", "http://localhost:1234")

    def _parse_models(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse the LM Studio /v1/models response into model result dicts.

        Args:
            data: The JSON response from ``/v1/models`` containing a ``"data"``
                  list of model objects.

        Returns:
            A list of model dictionaries with name, publisher, and source populated.
        """
        results: list[dict[str, Any]] = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            results.append(
                {
                    "name": model_id,
                    "publisher": "LM Studio",
                    "source": "LM Studio",
                    "params": "",
                    "quant": "",
                    "size": "",
                    "fit": "",
                    "mode": "",
                }
            )
        return results

    def detect(self) -> bool:
        """Check if LM Studio's local API is reachable.

        Returns:
            ``True`` if the API responds with a 200 status code.
        """
        try:
            resp = get_session().get(f"{self.host}/v1/models", timeout=2)
            return resp.status_code == 200
        except RequestException:
            return False

    def search(self, query: str, specs: dict, page_size: int = 20) -> tuple[list[dict], list[str]]:
        """Search for models in LM Studio.

        LM Studio serves whatever models the user has loaded locally, so
        search is a client-side filter over the full model list.

        Returns:
            A ``(results, errors)`` tuple.
        """
        try:
            resp = get_session().get(f"{self.host}/v1/models", timeout=5)
            if resp.status_code != 200:
                return [], [f"LM Studio API returned status {resp.status_code}"]

            data = resp.json()
        except RequestException as exc:
            return [], [f"LM Studio search failed: {exc}"]

        results = self._parse_models(data)

        # Client-side filter
        q = query.lower()
        results = [r for r in results if q in r["name"].lower()]

        # Enrich with scores
        enriched = []
        for r in results[:page_size]:
            try:
                enriched.append(enrich_result_with_scores(r, specs))
            except Exception:
                enriched.append(r)
        return enriched, []

    def list_installed(self) -> list[str]:
        """Return list of model names loaded in LM Studio."""
        return [m["name"] for m in self._parse_models({})]

    def search_with_installed(
        self, query: str, specs: dict, installed: list[str], page_size: int = 20
    ) -> tuple[list[dict], list[str]]:
        """Search across LM Studio with installed-model awareness.

        Delegates to :meth:`search` since LM Studio manages its own model set.
        """
        return self.search(query, specs, page_size=page_size)

    def get_metadata(self, model_name: str) -> dict | None:
        """Fetch metadata for a specific model.

        LM Studio's API does not provide per-model metadata endpoints,
        so this falls back to listing all models and searching by name.

        Args:
            model_name: The model name to look up.

        Returns:
            A model dict or ``None`` if not found.
        """
        try:
            resp = get_session().get(f"{self.host}/v1/models", timeout=2)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except RequestException:
            return None

        for model in data.get("data", []):
            if model.get("id", "").lower() == model_name.lower():
                return {
                    "name": model["id"],
                    "publisher": "LM Studio",
                    "source": "LM Studio",
                }
        return None
