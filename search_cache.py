"""In-memory cache for search result pages with hardware-aware validation."""

from __future__ import annotations

import time
from typing import Any


class SearchCache:
    """Store cached search pages and invalidate by age or hardware drift."""

    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_entries: int,
        ram_threshold_gb: float,
        vram_threshold_gb: float,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.ram_threshold_gb = ram_threshold_gb
        self.vram_threshold_gb = vram_threshold_gb
        self._entries: dict[str, dict[str, Any]] = {}

    def get(self, query_key: str, current_specs: dict[str, Any]) -> dict[str, Any] | None:
        entry = self._entries.get(query_key)
        if not entry:
            return None

        age = time.monotonic() - entry["timestamp"]
        if age > self.ttl_seconds:
            self._entries.pop(query_key, None)
            return None

        if not self._is_cache_compatible(current_specs, entry.get("specs")):
            self._entries.pop(query_key, None)
            return None
        return entry

    def set(
        self,
        query_key: str,
        *,
        results: list[dict[str, Any]],
        error: str,
        has_more_pages: bool,
        specs: dict[str, Any],
    ) -> None:
        self._entries[query_key] = {
            "timestamp": time.monotonic(),
            "results": [item.copy() for item in results],
            "error": error,
            "has_more_pages": has_more_pages,
            "specs": {
                "has_gpu": specs.get("has_gpu", False),
                "ram_free": specs.get("ram_free", 0.0),
                "vram_free": specs.get("vram_free", 0.0),
            },
        }

        if len(self._entries) <= self.max_entries:
            return

        oldest_key = min(self._entries, key=lambda key: self._entries[key]["timestamp"])
        self._entries.pop(oldest_key, None)

    def _is_cache_compatible(
        self,
        current_specs: dict[str, Any],
        cached_specs: dict[str, Any] | None,
    ) -> bool:
        if not cached_specs:
            return True
        if current_specs.get("has_gpu") != cached_specs.get("has_gpu"):
            return False

        ram_delta = abs(current_specs.get("ram_free", 0.0) - cached_specs.get("ram_free", 0.0))
        if ram_delta > self.ram_threshold_gb:
            return False

        if current_specs.get("has_gpu"):
            vram_delta = abs(
                current_specs.get("vram_free", 0.0) - cached_specs.get("vram_free", 0.0)
            )
            if vram_delta > self.vram_threshold_gb:
                return False

        return True
