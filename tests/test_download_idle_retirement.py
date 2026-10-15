"""Idle retirement must be atomic with independent HTTP job admission."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from core.http_server import LocalThreadingHTTPServer
from downloads import service_client
from downloads.api import _make_handler
from downloads.download_service import DownloadServiceState
from downloads.store import DownloadStore


def request(port, path, body=None):
    req = Request(f"http://127.0.0.1:{port}{path}",
                  data=json.dumps(body).encode() if body is not None else None,
                  headers={"Authorization": "Bearer idle-retirement-only"})
    with build_opener(ProxyHandler({})).open(req, timeout=5) as response:
        return json.load(response)


@contextmanager
def running_service(tmp_path, monkeypatch, *, pause_snapshot=False, unsupported=False, lose_response=False):
    state = DownloadServiceState()
    state.store = DownloadStore(tmp_path / "jobs.db")
    snapshot = threading.Event()
    release = threading.Event()
    base = _make_handler(state, "idle-retirement-only")

    class Handler(base):
        def do_POST(self):
            if unsupported and self.path == "/shutdown-if-idle":
                self._json_response(404, {"error": "unsupported"})
                return
            super().do_POST()

        def _json_response(self, status, payload, extra_headers=None):
            if lose_response and self.path == "/shutdown-if-idle":
                raise BrokenPipeError("injected response loss")
            if pause_snapshot and self.command == "GET" and self.path == "/jobs?limit=1000":
                snapshot.set()
                assert release.wait(10), "Snapshot gate timed out"
            super()._json_response(status, payload, extra_headers)

    server = LocalThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.server = server
    monkeypatch.setattr(service_client.config.settings, "download_db_path", tmp_path / "jobs.db")
    monkeypatch.setattr(service_client.config.settings, "download_service_host", "127.0.0.1")
    monkeypatch.setattr(service_client.config.settings, "download_service_port", server.server_port)
    monkeypatch.setattr(service_client.config.settings, "download_service_token", "idle-retirement-only")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, server.server_port, snapshot, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_job_accepted_after_idle_snapshot_prevents_automatic_retirement(tmp_path, monkeypatch):
    with running_service(tmp_path, monkeypatch, pause_snapshot=True) as (state, port, snapshot, release):
        # Simulate a future client floor; all HTTP responses and store writes
        # remain real. No replacement process may start over an accepted job.
        monkeypatch.setattr(service_client, "MIN_SERVICE_VERSION", "99.0")

        def forbidden_start():
            raise AssertionError("Accepted job was overtaken by automatic replacement")

        monkeypatch.setattr(service_client, "_start_service_process", forbidden_start)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(service_client.ensure_service_running)
            try:
                assert snapshot.wait(5)
                created = request(port, "/jobs", {"model": {"source": "Ollama", "name": "late-job"}})
            finally:
                release.set()
            assert future.result(timeout=10) is False
        assert not state.stop_event.is_set()
        assert request(port, "/health")["ok"]
        assert request(port, "/jobs")["jobs"] == [created["job"]]


def test_idle_retirement_fences_late_job_admission(tmp_path, monkeypatch):
    with running_service(tmp_path, monkeypatch) as (state, port, _, _):
        entered = threading.Event()
        release = threading.Event()
        shutdown = state.request_shutdown

        def gated_shutdown():
            entered.set()
            assert release.wait(5)
            shutdown()

        monkeypatch.setattr(state, "request_shutdown", gated_shutdown)
        try:
            assert request(port, "/shutdown-if-idle", {}) == {"ok": True}
            assert entered.wait(5)
            try:
                request(port, "/jobs", {"model": {"source": "Ollama", "name": "too-late"}})
            except HTTPError as error:
                assert error.code == 503
            else:
                raise AssertionError("Retiring service admitted a late job")
            assert request(port, "/jobs")["jobs"] == []
        finally:
            release.set()


def test_idle_retirement_still_stops_after_response_loss(tmp_path, monkeypatch):
    with running_service(tmp_path, monkeypatch, lose_response=True) as (state, port, _, _):
        try:
            request(port, "/shutdown-if-idle", {})
        except (OSError, HTTPException):
            pass
        else:
            raise AssertionError("Injected response loss unexpectedly returned a response")
        assert state.stop_event.wait(5), "Response loss left the service permanently draining"


def test_unsupported_idle_retirement_never_falls_back_to_forced_shutdown(tmp_path, monkeypatch):
    with running_service(tmp_path, monkeypatch, unsupported=True) as (state, port, _, _):
        monkeypatch.setattr(service_client, "MIN_SERVICE_VERSION", "99.0")

        def forbidden_start():
            raise AssertionError("Unsupported retirement caused a replacement")

        monkeypatch.setattr(service_client, "_start_service_process", forbidden_start)
        assert service_client.ensure_service_running() is False
        assert not state.stop_event.is_set()
        assert request(port, "/health")["ok"]
