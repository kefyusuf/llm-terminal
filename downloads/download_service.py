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
import sqlite3
import sys
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
from typing import Any

from loguru import logger

import config
from core.http_server import LocalThreadingHTTPServer
from downloads.runner import process_job
from downloads.store import DownloadStore

SERVICE_VERSION = "2.1"


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
        self._store: Any = None
        self.running_processes = {}
        self.running_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.server: Any = None
        self.worker_thread: Any = None

    @property
    def store(self):
        """Create persistence lazily so importing the service does not mutate it."""
        if self._store is None:
            self._store = DownloadStore(download_db_path())
        return self._store

    @store.setter
    def store(self, value):
        self._store = value

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

    def request_worker_stop(self):
        """Stop claims and cancel owned children without touching the HTTP loop."""
        self.stop_event.set()
        for target_id in self.snapshot_active_targets():
            self.store.mark_cancel_requested(target_id)

    def request_shutdown(self):
        self.request_worker_stop()
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
                # A visible terminal record may be retried before its previous
                # future finishes clearing the registered child process.
                excluded = {target for future, target in active.items() if not future.done()}
                job = STATE.store.claim_next_queued(exclude_targets=excluded)
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

    max_workers = _max_workers()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        active = {}
        while not STATE.stop_event.is_set():
            active = {future: target for future, target in active.items() if not future.done()}
            if len(active) >= max_workers:
                wait(active, timeout=0.25, return_when=FIRST_COMPLETED)
                continue
            target_id = _claim()
            if target_id is None:
                continue
            active[pool.submit(process_job, STATE, target_id)] = target_id


@contextmanager
def _service_server(address, handler):
    """Own the database and socket before recovery; hold ownership through shutdown."""
    database = download_db_path().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    # A separate SQLite file provides a process lock without blocking job writes.
    # Keep this file: unlinking it could let a third process lock a different inode.
    with closing(sqlite3.connect(str(database) + ".service-lock", timeout=0)) as owner:
        try:
            owner.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            raise RuntimeError("Download database ownership is unavailable") from exc
        server = LocalThreadingHTTPServer(address, handler)
        STATE.server = server
        try:
            STATE.store.normalize_target_ids()
            STATE.store.migrate_legacy_hf_commands()
            STATE.store.recover_orphaned_running_jobs()
            STATE.stop_event.clear()
            yield server
        finally:
            server.server_close()
            STATE.server = None


def main():
    """Run the configured download service or its smoke check."""
    validate_service_auth_boundary()
    if smoke_mode_enabled():
        return run_smoke_check()

    from downloads.api import Handler

    with _service_server(service_bind_address(), Handler) as server:
        worker = threading.Thread(target=worker_loop, daemon=True)
        STATE.worker_thread = worker
        worker_started = False
        try:
            worker.start()
            worker_started = True
            server.serve_forever(poll_interval=0.5)
        finally:
            try:
                STATE.request_worker_stop()
            finally:
                if worker_started:
                    worker.join()

    return 0


def run_smoke_check() -> int:
    """Run bounded health and authenticated jobs checks against an ephemeral server."""
    from downloads.api import Handler

    host, _ = service_bind_address()
    with _service_server((host, 0), Handler) as server:
        worker = threading.Thread(target=worker_loop, daemon=True, name="download-service-smoke-worker")
        STATE.worker_thread = worker
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.1},
            daemon=True,
            name="download-service-smoke-server",
        )
        port = int(server.server_address[1])
        worker_started = server_started = False
        try:
            worker.start()
            worker_started = True
            thread.start()
            server_started = True
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
            try:
                STATE.request_worker_stop()
            finally:
                try:
                    if server_started:
                        server.shutdown()
                        thread.join()
                finally:
                    if worker_started:
                        worker.join()

    print("[smoke] download service ok")
    return 0


if __name__ == "__main__":
    # Relative API imports must share this process's state when launched with -m.
    sys.modules["downloads.download_service"] = sys.modules[__name__]
    raise SystemExit(main())
