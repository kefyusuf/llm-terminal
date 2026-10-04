"""Regression coverage for the download-service authentication boundary."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import config
from downloads import download_service, service_client
from downloads.api import _make_handler


class _StoreStub:
    """Provide the minimal store surface used by the authenticated jobs endpoint."""

    def list_jobs(self, limit=50):
        """Return deterministic job data for HTTP boundary tests."""
        return [{"target_id": "model-1", "limit": limit}]


class _StateStub:
    """Provide the minimal state surface required by the HTTP handler."""

    def __init__(self):
        """Initialize deterministic state and worker metadata."""
        self.store = _StoreStub()
        self.worker_thread = None

    def snapshot_active_targets(self):
        """Return no active targets for diagnostic requests."""
        return []


def test_non_loopback_bind_is_rejected_without_tls(monkeypatch):
    """Remote binds must remain disabled until authenticated TLS transport exists."""
    monkeypatch.setattr(config.settings, "download_service_host", "0.0.0.0")
    monkeypatch.setattr(config.settings, "download_service_token", "secret-token")

    with pytest.raises(RuntimeError, match="authenticated TLS transport"):
        download_service.validate_service_auth_boundary()


def test_loopback_bind_remains_token_optional(monkeypatch):
    """Default loopback usage must remain compatible without a bearer token."""
    monkeypatch.setattr(config.settings, "download_service_host", "127.0.0.1")
    monkeypatch.setattr(config.settings, "download_service_token", None)

    download_service.validate_service_auth_boundary()


def test_service_client_rejects_non_loopback_plaintext_target(monkeypatch):
    """The client must reject remote HTTP targets before opening a connection."""
    called = False

    def fake_open(request, timeout):
        """Record any unexpected network attempt."""
        nonlocal called
        called = True
        raise AssertionError("opener must not be called for non-loopback HTTP")

    monkeypatch.setattr(config.settings, "download_service_host", "192.0.2.10")
    monkeypatch.setattr(config.settings, "download_service_token", "secret-token")
    monkeypatch.setattr(service_client._NO_PROXY_OPENER, "open", fake_open)

    with pytest.raises(RuntimeError, match="authenticated TLS transport"):
        service_client.get_service_health(timeout=1.25)
    assert called is False


def test_service_client_forwards_configured_bearer_token(monkeypatch):
    """The client must authenticate only the initial service request."""
    captured = {}

    class _ResponseStub:
        """Act as a minimal opener response context manager."""

        def __enter__(self):
            """Return the response stub for context-manager use."""
            return self

        def __exit__(self, *args):
            """Leave the context manager without suppressing exceptions."""
            return False

        def read(self):
            """Return a deterministic JSON response body."""
            return b'{"ok": true}'

    def fake_open(request, timeout):
        """Capture initial-request and redirectable Authorization state."""
        captured["authorization"] = request.get_header("Authorization")
        captured["unredirected"] = request.unredirected_hdrs.get("Authorization")
        captured["redirectable"] = request.headers.get("Authorization")
        captured["timeout"] = timeout
        return _ResponseStub()

    monkeypatch.setattr(config.settings, "download_service_host", "127.0.0.1")
    monkeypatch.setattr(config.settings, "download_service_token", "secret-token")
    monkeypatch.setattr(service_client._NO_PROXY_OPENER, "open", fake_open)

    assert service_client.get_service_health(timeout=1.25) == {"ok": True}
    assert captured == {
        "authorization": "Bearer secret-token",
        "unredirected": "Bearer secret-token",
        "redirectable": None,
        "timeout": 1.25,
    }


def test_service_client_formats_ipv6_loopback_url(monkeypatch):
    """IPv6 loopback literals must be bracketed in HTTP URLs."""
    monkeypatch.setattr(config.settings, "download_service_host", "::1")
    monkeypatch.setattr(config.settings, "download_service_port", 8765)

    assert service_client.service_base_url() == "http://[::1]:8765"


def test_service_client_bypasses_environment_proxies(monkeypatch):
    """Loopback service requests must ignore configured HTTP proxy variables."""
    handler = _make_handler(_StateStub())
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setattr(config.settings, "download_service_host", host)
    monkeypatch.setattr(config.settings, "download_service_port", port)
    monkeypatch.setattr(config.settings, "download_service_token", None)

    try:
        assert service_client.get_service_health(timeout=2) == {
            "ok": True,
            "version": "2.0",
        }
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_handler_keeps_health_public_and_protects_jobs():
    """A configured token must protect job data while leaving health probes public."""
    handler = _make_handler(_StateStub(), auth_token="secret-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    try:
        with urlopen(f"http://{host}:{port}/health", timeout=2) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["ok"] is True

        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"http://{host}:{port}/jobs", timeout=2)
        assert exc_info.value.code == 401

        request = Request(f"http://{host}:{port}/jobs")
        request.add_header("Authorization", "Bearer secret-token")
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["jobs"][0]["target_id"] == "model-1"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
