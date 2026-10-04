"""Nonblocking, single-flight availability snapshots for local API consumers."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from loguru import logger


class ProviderDiscovery:
    """Retain the last availability result while a daemon performs slow probes."""

    def __init__(
        self,
        detector: Callable[[], dict[str, bool]],
        *,
        ttl_seconds: float = 30,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._detector = detector
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._availability: dict[str, bool] = {"huggingface": True}
        self._status = "pending"
        self._error: str | None = None
        self._last_attempt: float | None = None
        self._last_success: float | None = None
        self._refreshing = False

    def snapshot(self) -> dict:
        """Return immediately, starting at most one refresh after the TTL expires."""
        with self._lock:
            now = self._clock()
            stale = self._last_success is None or now - self._last_success >= self._ttl_seconds
            retry_due = self._last_attempt is None or now - self._last_attempt >= self._ttl_seconds
            if retry_due and not self._refreshing:
                self._refreshing = True
                threading.Thread(
                    target=self._refresh, daemon=True, name="provider-discovery"
                ).start()
            return {
                "availability": self._availability.copy(),
                "status": self._status,
                "refreshing": self._refreshing,
                "stale": stale,
                "error": self._error,
            }

    def _refresh(self) -> None:
        try:
            availability = dict(self._detector())
        except Exception as exc:
            # Retain stale data; exception text may contain secrets or private paths.
            error = type(exc).__name__
            logger.warning("Provider availability refresh failed ({})", error)
            with self._lock:
                self._status = "error"
                self._error = error
                self._last_attempt = self._clock()
                self._refreshing = False
            return
        with self._lock:
            self._availability = availability
            self._status = "ready"
            self._error = None
            self._last_attempt = self._clock()
            self._last_success = self._last_attempt
            self._refreshing = False
