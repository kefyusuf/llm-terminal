"""Behavioral contracts for bounded, shareable diagnostics."""

import json
import threading
import time

import requests
from click.testing import CliRunner

import cli
import config
from core import diagnostics


def check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


def test_offline_cli_json_is_shareable_and_never_probes(monkeypatch, tmp_path):
    secret = "private-token-value"
    monkeypatch.setattr(config.settings, "hf_token", secret)
    monkeypatch.setattr(config.settings, "hf_models_dir", tmp_path / secret)
    monkeypatch.setenv("LMSTUDIO_HOST", f"http://user:{secret}@localhost:1234/private")
    monkeypatch.setattr(diagnostics, "probe_endpoint", lambda *args: pytest_fail())
    result = CliRunner().invoke(cli.cli, ["doctor", "--offline", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert secret not in result.output
    assert str(tmp_path) not in result.output
    assert check(report, "huggingface")["status"] == "skipped"
    assert check(report, "hf_token")["configured"] is True
    assert check(report, "models")["exists"] is False
    assert check(report, "models")["free_bytes"] > 0


def pytest_fail():
    raise AssertionError("Offline diagnostics must not access the network")


def test_deadline_returns_without_waiting_for_blocked_network(monkeypatch):
    release = threading.Event()

    def blocked(*args):
        release.wait(3)
        return {"status": "ok", "code": "reachable"}

    monkeypatch.setattr(diagnostics, "probe_endpoint", blocked)
    start = time.monotonic()
    try:
        report = diagnostics.collect_diagnostics(config.settings, timeout=0.05)
        assert time.monotonic() - start < 0.8
        assert check(report, "ollama")["code"] == "deadline"
        assert check(report, "huggingface")["status"] == "error"
    finally:
        release.set()


def test_tls_errors_are_actionable_without_exception_secrets(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.exceptions.SSLError("token SECRET at C:/private/person.pem")

    monkeypatch.setattr(requests.Session, "get", fail)
    result = diagnostics.probe_endpoint("https://huggingface.co", 0.1, True)
    assert result["code"] == "tls"
    assert "REQUESTS_CA_BUNDLE" in result["action"]
    assert "SECRET" not in json.dumps(result)
    assert "private" not in json.dumps(result)


def test_probe_preserves_tls_and_disables_credentials_redirects(monkeypatch):
    observed = {}

    class Response:
        status_code = 200

        def iter_content(self, **kwargs):
            yield b'{"version": "0.12.1", "secret": "never-export"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def get(session, url, **kwargs):
        observed.update(kwargs)
        observed["auth"] = session.auth
        return Response()

    monkeypatch.setattr(requests.Session, "get", get)
    result = diagnostics.probe_endpoint("http://localhost:11434/api/version", 0.1, False)
    assert result["version"] == "0.12.1"
    assert observed["verify"] is True
    assert observed["allow_redirects"] is False
    assert callable(observed["auth"])
    assert "never-export" not in json.dumps(result)


def test_invalid_ca_file_is_error_even_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "missing-secret.pem"))
    report = diagnostics.collect_diagnostics(config.settings, offline=True)
    assert report["status"] == "error"
    assert check(report, "tls_ca")["code"] == "invalid_ca"
    assert "missing-secret" not in json.dumps(report)


def test_invalid_endpoint_is_rejected_without_network(monkeypatch):
    monkeypatch.setattr(requests.Session, "get", lambda *args, **kwargs: pytest_fail())
    for url in ("file:///private", "http://user:secret@localhost", "http://localhost/?token=x"):
        result = diagnostics.probe_endpoint(url, 0.1, False)
        assert result["status"] == "error"
        assert result["code"] == "invalid_endpoint"


def test_remote_source_requires_https(monkeypatch):
    monkeypatch.setattr(requests.Session, "get", lambda *args, **kwargs: pytest_fail())
    result = diagnostics.probe_endpoint("http://huggingface.co", 0.1, True)
    assert result["code"] == "invalid_endpoint"


def test_offline_reports_service_and_mlx_without_starting_them():
    report = diagnostics.collect_diagnostics(config.settings, offline=True)
    assert check(report, "download_service")["status"] == "skipped"
    assert check(report, "mlx")["code"] in {"installed", "not_installed", "unsupported_platform"}


def test_optional_runtime_absence_is_warning(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("private endpoint")

    monkeypatch.setattr(requests.Session, "get", fail)
    assert diagnostics.probe_endpoint("http://localhost", 0.1, False)["status"] == "warning"
    assert diagnostics.probe_endpoint("https://huggingface.co", 0.1, True)["status"] == "error"


def test_cli_error_exit_keeps_json_parseable(monkeypatch):
    monkeypatch.setattr(
        diagnostics,
        "collect_diagnostics",
        lambda *args, **kwargs: {"schema_version": 1, "status": "error", "checks": []},
    )
    result = CliRunner().invoke(cli.cli, ["doctor", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.output)["status"] == "error"
