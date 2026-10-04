"""Contracts that prevent checkout imports from masking packaging failures."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def load_verifier():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_installed_package.py"
    spec = importlib.util.spec_from_file_location("package_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rejects_checkout_and_non_distribution_imports(tmp_path):
    verifier = load_verifier()
    checkout = tmp_path / "checkout"
    prefix = tmp_path / "environment"
    installed = prefix / "Lib" / "site-packages" / "main.py"
    verifier.assert_installed_origin(installed, prefix, checkout, {installed})
    for origin, recorded in (
        (checkout / "main.py", {checkout / "main.py"}),
        (installed, set()),
        (tmp_path / "external.py", {tmp_path / "external.py"}),
    ):
        with pytest.raises(ValueError, match="installed distribution"):
            verifier.assert_installed_origin(origin, prefix, checkout, recorded)


def test_launcher_accepts_venv_interpreter_without_resolving_its_symlink(tmp_path, monkeypatch):
    verifier = load_verifier()

    def unexpected_resolve(*args, **kwargs):
        raise AssertionError(
            "POSIX venv interpreter symlinks must not be resolved to the base Python"
        )

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)
    expected = tmp_path / "environment" / "bin" / "python"
    verifier.assert_launcher_interpreter(expected, expected)
    with pytest.raises(ValueError, match="installed interpreter"):
        verifier.assert_launcher_interpreter(tmp_path / "base-python", expected)


def test_environment_removes_source_paths_and_user_configuration(tmp_path, monkeypatch):
    verifier = load_verifier()
    monkeypatch.setenv("PYTHONPATH", "/checkout")
    monkeypatch.setenv("PYTHONHOME", "/another-python")
    monkeypatch.setenv("AIMODEL_HF_TOKEN", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("LMSTUDIO_HOST", "http://private")
    monkeypatch.setenv("PIP_TARGET", str(tmp_path / "user-owned"))
    monkeypatch.setenv("PIP_USER", "1")
    env = verifier.isolated_environment(tmp_path)
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert "HF_TOKEN" not in env and "AIMODEL_HF_TOKEN" not in env
    assert env["AIMODEL_SMOKE"] == "1"
    assert Path(env["AIMODEL_CACHE_DB_PATH"]).parent == tmp_path
    assert env["AIMODEL_DOWNLOAD_SERVICE_HOST"] == "127.0.0.1"
    assert env["LMSTUDIO_HOST"] != "http://private"
    assert "PIP_TARGET" not in env and "PIP_USER" not in env


def test_select_artifact_requires_exactly_one_candidate(tmp_path):
    verifier = load_verifier()
    with pytest.raises(ValueError, match="exactly one"):
        verifier.select_artifact(tmp_path, "wheel")
    wheel = tmp_path / "package-1.0-py3-none-any.whl"
    wheel.touch()
    assert verifier.select_artifact(tmp_path, "wheel") == wheel
    (tmp_path / "package-0.9-py3-none-any.whl").touch()
    with pytest.raises(ValueError, match="exactly one"):
        verifier.select_artifact(tmp_path, "wheel")


def test_checkout_origin_rejection_runs_in_a_real_isolated_child(tmp_path):
    verifier = load_verifier()
    # Execute the same origin validator used inside the installed-import probe.
    code = (
        verifier.ORIGIN_VALIDATOR
        + "\nassert_installed_origin(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), {Path(sys.argv[1])})"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(tmp_path / "checkout" / "main.py"),
            str(tmp_path / "env"),
            str(tmp_path / "checkout"),
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "installed distribution" in result.stderr


def test_failure_and_timeout_are_not_reported_as_success(tmp_path):
    verifier = load_verifier()
    for code, timeout in (("raise SystemExit(3)", 10), ("import time; time.sleep(5)", 0.05)):
        with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            verifier.run_step(
                "intentional failure",
                [sys.executable, "-I", "-c", code],
                tmp_path,
                os.environ.copy(),
                timeout,
            )


def test_failed_rerun_replaces_stale_pass_report(tmp_path, monkeypatch):
    import json

    verifier = load_verifier()
    artifact = tmp_path / "package-1.0-py3-none-any.whl"
    artifact.touch()
    report = tmp_path / "report.json"
    report.write_text('{"status": "passed"}', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--dist-dir", str(tmp_path), "--kind", "wheel", "--report", str(report)],
    )

    def fail(*args):
        raise ValueError("intentional failure")

    monkeypatch.setattr(verifier, "verify_package", fail)
    with pytest.raises(ValueError):
        verifier.main()
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "failed"
