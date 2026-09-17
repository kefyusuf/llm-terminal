"""Background HTTP service for asynchronous model downloads.

This module is the entry point (``python -m downloads.download_service``).
It owns the service's lifecycle: starting the worker thread, binding the
HTTP socket, and shutting down cleanly on ``/shutdown`` or Ctrl-C.

The implementation is split across three modules:

- :mod:`downloads.store` — SQLite persistence (``DownloadStore``).
- :mod:`downloads.runner` — subprocess runners (HF, streamed).
- :mod:`downloads.api` — :class:`BaseHTTPRequestHandler` dispatch.

This module wires them together: the ``STATE`` singleton holds the
store + the worker bookkeeping, ``worker_loop`` is the claim/dispatch
loop, and ``main`` runs the HTTP server.

This split is the result of architecture-review candidate #5. The
pre-split file was 823 lines mixing all four concerns; the post-split
files are 200-300 lines each with a single, testable responsibility.
"""

from __future__ import annotations

import ipaddress
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

import config
from core.http_server import LocalThreadingHTTPServer
from downloads.runner import process_job
from downloads.store import DownloadStore

SERVICE_VERSION = "1.8"


def download_db_path():
    """Return the configured download-service SQLite path."""
    return config.settings.download_db_path


def service_bind_address() -> tuple[str, int]:
    """Return the configured download-service bind address."""
    return (
        config.settings.download_service_host,
        config.settings.download_service_port,
    )


def _is_loopback_host(host: str) -> bool:
    """Return whether *host* is an explicit loopback address or localhost."""
    normalized = str(host).strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_service_auth_boundary() -> None:
    """Reject non-loopback binds until authenticated TLS transport is supported."""
    host, _ = service_bind_address()
    if _is_loopback_host(host):
        return
    raise RuntimeError(
        "Non-loopback download-service binds are disabled until authenticated TLS transport is supported"
    )


def smoke_mode_enabled() -> bool:
    """Return whether the service should run its bounded smoke check."""
    return os.getenv("AIMODEL_SMOKE") == "1"


def ensure_data_dir() -> None:
    """Create the configured download database directory if needed."""
    download_db_path().parent.mkdir(parents=True, exist_ok=True)


class DownloadServiceState:
    """Process-wide singleton holding the store + worker bookkeeping.

    ``store`` — the SQLite-backed ``DownloadStore``.
    ``running_processes`` — map of target_id -> subprocess.Popen for
        the cancel-via-terminate path.
    ``stop_event`` — set by ``/shutdown`` to ask the worker to exit.
    ``server`` — set by ``main`` so ``/shutdown`` can call
        ``server.shutdown()`` from a background thread.
    ``worker_thread`` — set by ``main`` so the debug endpoint can
        report worker liveness.
    """

    def __init__(self):
        self.store = DownloadStore(download_db_path())
        self.store.normalize_target_ids()
        self.store.migrate_legacy_hf_commands()
        self.store.recover_orphaned_running_jobs()
        self.running_processes = {}
        self.running_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.server: Any = None
        self.worker_thread: Any = None

    def get_process(self, target_id):
        with self.running_lock:
            return self.running_processes.get(target_id)

    def set_process(self, target_id, process):
        with self.running_lock:
            self.running_processes[target_id] = process

    def clear_process(self, target_id):
        with self.running_lock:
            self.running_processes.pop(target_id, None)

    def snapshot_active_targets(self):
        with self.running_lock:
            return list(self.running_processes.keys())

    def request_shutdown(self):
        self.stop_event.set()
        server = self.server
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()


STATE = DownloadServiceState()


def _has_duplicates(values):
    return len(values) != len(set(values))


def _max_workers():
    try:
        import config as _cfg

        return getattr(_cfg.settings, "download_max_workers", 2)
    except Exception:
        return 2


def worker_loop():
    """Claim queued jobs and dispatch them to a thread pool.

    Runs in a single daemon thread spawned by :func:`main`. The
    claim/dispatch dance is the most algorithmically dense code in
    the service, hence the dedicated function. The :class:`ThreadPoolExecutor`
    is bounded by ``_max_workers()`` (default 2) and shuts down
    naturally when ``stop_event`` is set.
    """

    def _claim():
        while not STATE.stop_event.is_set():
            try:
                job = STATE.store.claim_next_queued()
            except Exception:
                logger.warning("Failed to claim next queued job, retrying")
                time.sleep(0.5)
                continue

            if job is None:
                time.sleep(0.25)
                continue

            target_id = job["target_id"]
            latest = STATE.store.get_job_by_target(target_id)
            if latest and latest.get("cancel_requested"):
                STATE.store.update_job(target_id, status="cancelled", detail="Canceled", progress="")
                continue

            return target_id

        return None

    with ThreadPoolExecutor(max_workers=_max_workers()) as pool:
        while not STATE.stop_event.is_set():
            target_id = _claim()
            if target_id is None:
                continue
            pool.submit(process_job, STATE, target_id)


def main():
    """Run the configured download service or its smoke check."""
    validate_service_auth_boundary()
    if smoke_mode_enabled():
        return run_smoke_check()

    from downloads.api import Handler

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()
    STATE.worker_thread = worker

    server = LocalThreadingHTTPServer(service_bind_address(), Handler)
    STATE.server = server
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        STATE.stop_event.set()
        server.server_close()

    return 0


def run_smoke_check() -> int:
    """Run bounded health and authenticated jobs checks against an ephemeral server."""
    from downloads.api import Handler

    STATE.stop_event.clear()

    worker = threading.Thread(target=worker_loop, daemon=True, name="download-service-smoke-worker")
    worker.start()
    STATE.worker_thread = worker

    host, _ = service_bind_address()
    server = LocalThreadingHTTPServer((host, 0), Handler)
    STATE.server = server
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.1},
        daemon=True,
        name="download-service-smoke-server",
    )
    thread.start()
    port = int(server.server_address[1])

    try:
        for path, expected_key in (("/health", "ok"), ("/jobs", "jobs")):
            request = urllib.request.Request(f"http://{host}:{port}{path}")
            token = config.settings.download_service_token
            if token:
                request.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if expected_key not in payload:
                raise SystemExit(f"[smoke] download service check failed for {path}")
    finally:
        STATE.request_shutdown()
        thread.join(timeout=5)
        worker.join(timeout=5)
        server.server_close()

    print("[smoke] download service ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
