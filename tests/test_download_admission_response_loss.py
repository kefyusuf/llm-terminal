"""An accepted job remains discoverable when its admission response is lost."""
from __future__ import annotations

import socket
import threading
from contextlib import contextmanager

import pytest

from app.download_manager import DownloadManager
from core.http_server import LocalThreadingHTTPServer
from downloads import service_client
from downloads.api import _make_handler
from downloads.download_service import DownloadServiceState
from downloads.store import DownloadStore


@contextmanager
def accepted_job_service(monkeypatch, tmp_path, mode):
    state = DownloadServiceState()
    state.store = DownloadStore(tmp_path / "jobs.db")
    submissions = []
    release = threading.Event()
    finished = threading.Event()
    base = _make_handler(state, "response-loss-only")

    class Handler(base):
        def _json_response(self, status, payload, extra_headers=None):
            if self.command != "POST" or self.path != "/jobs" or status != 200:
                return super()._json_response(status, payload, extra_headers)
            submissions.append(payload["job"]["target_id"])
            try:
                if mode == "timeout":
                    assert release.wait(10), "Client did not finish within the response gate"
                elif mode in {"truncated", "invalid-json"}:
                    self.send_response(200)
                    self.send_header("Content-Length", "100" if mode == "truncated" else "1")
                    self.end_headers()
                    self.wfile.write(b"{")
                    self.wfile.flush()
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
            finally:
                finished.set()

    server = LocalThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.server = server
    for name, value in {
        "download_db_path": tmp_path / "jobs.db",
        "download_service_host": "127.0.0.1",
        "download_service_port": server.server_port,
        "download_service_token": "response-loss-only",
    }.items():
        monkeypatch.setattr(service_client.config.settings, name, value)

    def unexpected_process(*args, **kwargs):
        raise AssertionError("Response-loss tests must not start service processes")

    monkeypatch.setattr(service_client.subprocess, "Popen", unexpected_process)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield submissions
    finally:
        release.set()
        try:
            if submissions:
                assert finished.wait(5), "Admission response handler did not finish"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("mode", ["disconnected", "truncated", "invalid-json", "timeout"])
def test_accepted_job_with_lost_response_reports_unknown_and_is_recovered_from_history(monkeypatch, tmp_path, mode):
    manager = DownloadManager(
        update_status=lambda message: None,
        refresh_table=lambda: None,
        refresh_download_history_table=lambda: None,
        request_download_history_refresh=lambda force: None,
        render_download_debug=lambda *args: None,
        find_model_by_target_id=lambda target: None,
    )
    manager.download_registry = {"ollama:existing": {"state": "downloading"}}
    manager.active_downloads = {"ollama:existing"}
    model = {"source": "Ollama", "name": "accepted-model"}
    with accepted_job_service(monkeypatch, tmp_path, mode) as submissions:
        ok, message = manager.start_download(model)

        assert ok is False
        assert message == (
            "Download request outcome is unknown. "
            "Check download history before retrying."
        )
        assert submissions == ["ollama:accepted-model"]
        assert manager.download_registry == {"ollama:existing": {"state": "downloading"}}
        assert manager.active_downloads == {"ollama:existing"}
        assert "download_state" not in model

        jobs = service_client.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["target_id"] == "ollama:accepted-model"
        assert jobs[0]["status"] == "queued"
        assert manager.sync_jobs(force=True) is True
        assert manager.download_registry["ollama:accepted-model"]["state"] == "queued"
        assert submissions == ["ollama:accepted-model"]
