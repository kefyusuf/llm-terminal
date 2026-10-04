"""Deterministic lifecycle coverage for downloads.service_client."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from downloads import service_client


def test_is_service_running_distinguishes_healthy_and_failed_requests(monkeypatch):
    """Reachability should require an explicit truthy health flag and contain request failures."""
    monkeypatch.setattr(service_client, "_request", lambda *_args, **_kwargs: {"ok": True})
    assert service_client.is_service_running() is True

    monkeypatch.setattr(service_client, "_request", lambda *_args, **_kwargs: {"ok": False})
    assert service_client.is_service_running() is False

    def _raise(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr(service_client, "_request", _raise)
    assert service_client.is_service_running() is False


def test_start_service_process_uses_platform_specific_detached_launch(monkeypatch):
    """Windows should use pythonw while POSIX should start a new session."""
    calls = []

    def _fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return object()

    monkeypatch.setattr(service_client.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(service_client.sys, "executable", "C:/Python/python.exe")
    monkeypatch.setattr(service_client.sys, "platform", "win32")

    service_client._start_service_process()

    assert calls == [
        (
            ["C:/Python/pythonw.exe", "-m", "downloads.download_service"],
            {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL},
        )
    ]

    calls.clear()
    monkeypatch.setattr(service_client.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(service_client.sys, "platform", "linux")

    service_client._start_service_process()

    assert calls == [
        (
            ["/usr/bin/python3", "-m", "downloads.download_service"],
            {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "start_new_session": True,
            },
        )
    ]


def test_wait_for_service_retries_until_healthy_compatible_response(monkeypatch):
    """Transient health failures and incompatible versions should be retried within the deadline."""
    health_calls = []
    responses = [URLError("booting"), {"ok": True, "version": "1.7"}, {"ok": True, "version": "2.0"}]

    def _health():
        health_calls.append(True)
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(service_client, "get_service_health", _health)
    monkeypatch.setattr(service_client.time, "time", lambda: 0.0)
    monkeypatch.setattr(service_client.time, "sleep", lambda _seconds: None)

    assert service_client._wait_for_service(deadline_seconds=1.0) is True
    assert len(health_calls) == 3


def test_wait_for_service_returns_false_after_deadline(monkeypatch):
    """The startup poll must terminate when its deadline expires."""
    time_values = iter([10.0, 12.0])
    monkeypatch.setattr(service_client.time, "time", lambda: next(time_values))

    assert service_client._wait_for_service(deadline_seconds=1.0) is False


def test_stop_service_returns_true_after_graceful_shutdown(monkeypatch):
    """A successful shutdown request followed by a dead health probe is a clean stop."""
    requests = []

    def _request(method, path, payload=None, timeout=2.0):
        requests.append((method, path, payload, timeout))
        return {"ok": True}

    monkeypatch.setattr(service_client, "_request", _request)
    monkeypatch.setattr(service_client, "psutil", None)
    monkeypatch.setattr(service_client, "is_service_running", lambda: False)

    assert service_client.stop_service() is True
    assert requests == [("POST", "/shutdown", {}, 1.0)]


def test_stop_service_falls_back_to_owned_process_kill(monkeypatch):
    """If graceful shutdown fails, the known service process should be killed as fallback."""
    process = SimpleNamespace(info={"cmdline": ["python", "-m", "downloads.download_service"]})
    killed = []
    process.kill = lambda: killed.append(True)
    process.poll = lambda: None
    monkeypatch.setattr(service_client, "_owned_service_process", process)

    fake_psutil = SimpleNamespace(
        process_iter=lambda _attrs: [process],
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
        ZombieProcess=type("ZombieProcess", (Exception,), {}),
    )

    def _shutdown_failure(*_args, **_kwargs):
        raise URLError("service unreachable")

    monkeypatch.setattr(service_client, "_request", _shutdown_failure)
    monkeypatch.setattr(service_client, "psutil", fake_psutil)
    monkeypatch.setattr(service_client, "is_service_running", lambda: False)

    assert service_client.stop_service() is True
    assert killed == [True]


def test_stop_service_returns_false_when_nothing_was_stopped(monkeypatch):
    """No reachable service and no matching process must report that nothing stopped."""
    def _shutdown_failure(*_args, **_kwargs):
        raise URLError("service unreachable")

    monkeypatch.setattr(service_client, "_request", _shutdown_failure)
    monkeypatch.setattr(service_client, "psutil", None)
    monkeypatch.setattr(service_client, "_owned_service_process", None)

    assert service_client.stop_service() is False


def test_ensure_service_running_reuses_compatible_service(monkeypatch):
    """A healthy compatible service should be reused without restart."""
    actions = []
    monkeypatch.setattr(service_client, "is_service_running", lambda: True)
    monkeypatch.setattr(service_client, "get_service_health", lambda: {"ok": True, "version": "2.0"})
    monkeypatch.setattr(service_client, "stop_service", lambda: actions.append("stop"))
    monkeypatch.setattr(service_client, "_start_service_process", lambda: actions.append("start"))
    monkeypatch.setattr(service_client, "_wait_for_service", lambda **_kwargs: actions.append("wait"))

    assert service_client.ensure_service_running() is True
    assert actions == []


def test_ensure_service_running_restarts_incompatible_service(monkeypatch):
    """A reachable but stale protocol version must be stopped and replaced."""
    actions = []
    monkeypatch.setattr(service_client, "is_service_running", lambda: True)
    monkeypatch.setattr(service_client, "get_service_health", lambda: {"ok": True, "version": "1.7"})
    monkeypatch.setattr(service_client, "list_jobs", lambda **kwargs: [])
    monkeypatch.setattr(service_client, "stop_service", lambda: actions.append("stop") or True)
    monkeypatch.setattr(service_client, "_start_service_process", lambda: actions.append("start"))
    monkeypatch.setattr(
        service_client,
        "_wait_for_service",
        lambda deadline_seconds: actions.append(("wait", deadline_seconds)) or True,
    )

    assert service_client.ensure_service_running() is True
    assert actions == ["stop", "start", ("wait", 6.0)]


def test_ensure_service_running_preserves_service_on_health_error(monkeypatch):
    """Unknown live service state must not trigger automatic termination."""
    actions = []

    def _health_error():
        raise ValueError("malformed health")

    monkeypatch.setattr(service_client, "is_service_running", lambda: True)
    monkeypatch.setattr(service_client, "get_service_health", _health_error)
    monkeypatch.setattr(service_client, "stop_service", lambda: actions.append("stop") or True)
    monkeypatch.setattr(service_client, "_start_service_process", lambda: actions.append("start"))
    monkeypatch.setattr(
        service_client,
        "_wait_for_service",
        lambda deadline_seconds: actions.append(("wait", deadline_seconds)) or False,
    )

    assert service_client.ensure_service_running() is False
    assert actions == []


def test_request_wrappers_preserve_methods_paths_payloads_and_timeouts(monkeypatch):
    """Public client helpers should remain thin, stable request-contract adapters."""
    calls = []

    def _request(method, path, payload=None, timeout=2.0):
        calls.append((method, path, payload, timeout))
        if path.startswith("/jobs?limit="):
            return {"jobs": [{"target_id": "job-1"}]}
        return {"ok": True}

    monkeypatch.setattr(service_client, "_request", _request)

    assert service_client.list_jobs(limit=7, timeout=1.25) == [{"target_id": "job-1"}]
    assert service_client.get_active_download_debug(timeout=1.5) == {"ok": True}
    assert service_client.create_job({"name": "model"}) == {"ok": True}
    assert service_client.cancel_job("job-1") == {"ok": True}

    assert calls == [
        ("GET", "/jobs?limit=7", None, 1.25),
        ("GET", "/debug/active", None, 1.5),
        ("POST", "/jobs", {"model": {"name": "model"}}, 3.0),
        ("POST", "/jobs/cancel", {"target_id": "job-1"}, 2.0),
    ]


def _http_404():
    return HTTPError("http://localhost/jobs/delete", 404, "not found", None, None)


def test_delete_job_restarts_and_retries_once_after_404(monkeypatch):
    """A stale service missing the delete endpoint should restart and retry exactly once."""
    calls = []

    def _request(method, path, payload=None, timeout=2.0):
        calls.append((method, path, payload, timeout))
        if len(calls) == 1:
            raise _http_404()
        return {"ok": True}

    monkeypatch.setattr(service_client, "_request", _request)
    monkeypatch.setattr(service_client, "ensure_service_running", lambda: True)

    assert service_client.delete_job("job-1") == {"ok": True}
    assert calls == [
        ("POST", "/jobs/delete", {"target_id": "job-1"}, 2.0),
        ("POST", "/jobs/delete", {"target_id": "job-1"}, 2.0),
    ]


def test_delete_job_reraises_404_when_restart_is_unavailable(monkeypatch):
    """Delete retry must fail closed if a compatible replacement service cannot be started."""
    monkeypatch.setattr(service_client, "_request", lambda *_args, **_kwargs: (_ for _ in ()).throw(_http_404()))
    monkeypatch.setattr(service_client, "ensure_service_running", lambda: False)

    with pytest.raises(HTTPError) as exc_info:
        service_client.delete_job("job-1")

    assert exc_info.value.code == 404
