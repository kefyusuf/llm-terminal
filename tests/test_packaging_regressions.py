"""Regression tests for packaging and installed CLI behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_dev_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dev.py"
    spec = importlib.util.spec_from_file_location("dev_script_packaging", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pyproject_uses_supported_setuptools_backend():
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["build-system"]["build-backend"] == "setuptools.build_meta"


def test_pyproject_uses_pep639_license_metadata():
    """License metadata must use the supported SPDX/PEP 639 representation."""
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["project"]["license"] == "MIT"
    assert data["project"]["license-files"] == ["LICENSE"]
    assert "setuptools>=77.0.3" in data["build-system"]["requires"]


def test_bootstrap_installs_project_after_platform_lock(tmp_path, monkeypatch):
    dev = _load_dev_module()
    (tmp_path / "requirements-dev-linux.txt").write_text("pytest==8.4.2\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = ['setuptools>=68']\nbuild-backend = 'setuptools.build_meta'\n",
        encoding="utf-8",
    )

    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append(cmd)

        class _Result:
            returncode = 0
            stdout = "pip 26.1.1"
            stderr = ""

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    assert dev.bootstrap() == 0
    assert calls[-1] == [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-e",
        ".",
    ]


def test_cli_version_comes_from_distribution_metadata(monkeypatch):
    import cli

    monkeypatch.setattr(cli, "package_version", lambda name: "9.8.7", raising=False)

    assert cli.get_version() == "9.8.7"
