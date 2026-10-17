"""Regression tests for configurable local runtime endpoints."""

from __future__ import annotations

from requests.exceptions import ConnectionError


def test_ollama_detect_uses_configured_api_health(monkeypatch):
    import config
    from providers import ollama_provider

    class ResponseStub:
        status_code = 200

    class SessionStub:
        def get(self, url, timeout):
            assert url == "http://ollama-host.test:11434/api/tags"
            assert timeout == 1
            return ResponseStub()

    monkeypatch.setattr(config.settings, "ollama_api_base", "http://ollama-host.test:11434")
    monkeypatch.setattr(ollama_provider, "get_session", lambda: SessionStub())

    assert ollama_provider.OllamaProvider().detect() is True


def test_ollama_detect_returns_false_when_api_unreachable(monkeypatch):
    from providers import ollama_provider

    class SessionStub:
        def get(self, url, timeout):
            raise ConnectionError("offline")

    monkeypatch.setattr(ollama_provider, "get_session", lambda: SessionStub())

    assert ollama_provider.OllamaProvider().detect() is False
