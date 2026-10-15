"""Rejected download requests give actionable feedback without losing local state."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler

import pytest

from app.download_manager import DownloadManager
from core.http_server import LocalThreadingHTTPServer
from downloads import service_client


@pytest.fixture(autouse=True)
def forbid_service_process_start(monkeypatch):
    def unexpected_process(*args, **kwargs):
        raise AssertionError("Feedback tests must not launch service processes")

    monkeypatch.setattr(service_client.subprocess, "Popen", unexpected_process)


@contextmanager
def controlled_service(monkeypatch, tmp_path, *, version="2.1", jobs=None, statuses=None):
    requests = []
    responses = [503, 200] if statuses is None else list(statuses)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            requests.append(("GET", self.path))
            if self.path == "/health":
                self.respond(200, {"ok": True, "version": version})
            elif self.path.startswith("/jobs?"):
                self.respond(200, {"jobs": jobs or []})
            else:
                self.respond(404, {})

        def do_POST(self):
            requests.append(("POST", self.path))
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path != "/jobs":
                self.respond(404, {})
                return
            status = responses.pop(0)
            self.respond(status, {"queued": True} if status == 200 else {"error": "Request refused"})

    server = LocalThreadingHTTPServer(("127.0.0.1", 0), Handler)
    monkeypatch.setattr(service_client.config.settings, "download_db_path", tmp_path / "jobs.db")
    monkeypatch.setattr(service_client.config.settings, "download_service_host", "127.0.0.1")
    monkeypatch.setattr(service_client.config.settings, "download_service_port", server.server_port)
    monkeypatch.setattr(service_client.config.settings, "download_service_token", "feedback-only")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def make_manager():
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
    return manager


def test_startup_configuration_refusal_is_reported_without_losing_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(service_client.config.settings, "download_db_path", tmp_path / "invalid\0.db")
    manager = make_manager()

    ok, message = manager.start_download({"source": "Ollama", "name": "new-model"})

    assert ok is False
    assert message == (
        "Download service is unavailable or cannot be upgraded safely. "
        "Let active downloads finish, then retry."
    )
    assert manager.download_registry == {"ollama:existing": {"state": "downloading"}}
    assert manager.active_downloads == {"ollama:existing"}


def test_retiring_service_rejection_is_retryable_without_automatic_resubmission(monkeypatch, tmp_path):
    manager = make_manager()
    model = {"source": "Ollama", "name": "new-model"}
    with controlled_service(monkeypatch, tmp_path) as requests:
        ok, message = manager.start_download(model)

        assert ok is False
        assert message == (
            "Download service is temporarily unavailable (HTTP 503). "
            "Download was not queued; retry shortly."
        )
        assert [r for r in requests if r[0] == "POST"] == [("POST", "/jobs")]
        assert manager.download_registry == {"ollama:existing": {"state": "downloading"}}
        assert manager.active_downloads == {"ollama:existing"}
        assert "download_state" not in model

        ok, message = manager.start_download(model)

        assert ok is True
        assert message == "Download queued: new-model"
        assert [r for r in requests if r[0] == "POST"] == [("POST", "/jobs"), ("POST", "/jobs")]


@pytest.mark.parametrize("status", ["queued", "running"])
def test_busy_legacy_service_reports_safe_retry_without_submitting_or_stopping(monkeypatch, tmp_path, status):
    manager = make_manager()
    with controlled_service(monkeypatch, tmp_path, version="1.7", jobs=[{"status": status}]) as requests:
        ok, message = manager.start_download({"source": "Ollama", "name": "new-model"})

        assert ok is False
        assert message == (
            "Download service is unavailable or cannot be upgraded safely. "
            "Let active downloads finish, then retry."
        )
        assert not any(method == "POST" for method, _ in requests)
        assert manager.download_registry == {"ollama:existing": {"state": "downloading"}}
        assert manager.active_downloads == {"ollama:existing"}


def test_other_http_errors_are_not_reported_as_temporary_retirement(monkeypatch, tmp_path):
    manager = make_manager()
    with controlled_service(monkeypatch, tmp_path, statuses=[401]) as requests:
        ok, message = manager.start_download({"source": "Ollama", "name": "new-model"})

        assert ok is False
        assert message == "Failed to queue download: HTTP Error 401: Unauthorized"
        assert [r for r in requests if r[0] == "POST"] == [("POST", "/jobs")]
