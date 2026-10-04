"""Availability freshness, single-flight refresh and failed-probe recovery."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from providers.discovery import ProviderDiscovery


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def wait_for(discovery, predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = discovery.snapshot()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"discovery did not reach expected state: {snapshot}")


def test_expired_snapshot_retains_results_and_coalesces_parallel_requests():
    clock = Clock()
    release = threading.Event()
    entered = threading.Event()
    attempts = []

    def detect():
        attempts.append(True)
        if len(attempts) == 1:
            return {"huggingface": True, "lmstudio": True}
        entered.set()
        release.wait(5)
        return {"huggingface": True, "lmstudio": False}

    discovery = ProviderDiscovery(detect, ttl_seconds=30, clock=clock)
    ready = wait_for(discovery, lambda s: s["status"] == "ready")
    assert ready["stale"] is False
    ready["availability"]["lmstudio"] = False
    assert discovery.snapshot()["availability"]["lmstudio"] is True
    clock.now = 31
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            snapshots = list(pool.map(lambda _: discovery.snapshot(), range(20)))
        assert entered.wait(1)
        assert all(s["refreshing"] and s["stale"] for s in snapshots)
        assert all(s["availability"]["lmstudio"] is True for s in snapshots)
        release.set()
        refreshed = wait_for(discovery, lambda s: not s["refreshing"])
        assert refreshed["availability"]["lmstudio"] is False
        assert refreshed["stale"] is False
        assert len(attempts) == 2
    finally:
        release.set()


def test_failed_refresh_retains_stale_data_and_recovers_after_retry_ttl():
    clock = Clock()
    secret = "API_TOKEN=private"
    attempts = []

    def detect():
        attempts.append(True)
        if len(attempts) == 2:
            raise RuntimeError(secret)
        return {"huggingface": True, "docker": len(attempts) == 1}

    discovery = ProviderDiscovery(detect, ttl_seconds=30, clock=clock)
    wait_for(discovery, lambda s: s["status"] == "ready")
    clock.now = 31
    failed = wait_for(discovery, lambda s: s["status"] == "error")
    assert failed["availability"]["docker"] is True
    assert failed["stale"] is True
    assert failed["refreshing"] is False
    assert failed["error"] == "RuntimeError"
    assert secret not in str(failed)
    assert discovery.snapshot()["status"] == "error"
    assert len(attempts) == 2
    clock.now = 62
    recovered = wait_for(discovery, lambda s: s["status"] == "ready" and not s["refreshing"])
    assert recovered["availability"]["docker"] is False
    assert recovered["error"] is None
    assert recovered["stale"] is False
