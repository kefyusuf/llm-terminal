"""Shared HTTP client with connection pooling + retry for provider API calls."""

from __future__ import annotations

import requests
from urllib3.util.retry import Retry

_session: requests.Session | None = None


def _build_adapter() -> requests.adapters.HTTPAdapter:
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
    )
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=retry_strategy,
    )
    return adapter


def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = _build_adapter()
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


def close_session() -> None:
    global _session
    if _session is not None:
        _session.close()
        _session = None
