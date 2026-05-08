from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_dev_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dev.py"
    assert module_path.exists(), "scripts/dev.py must exist"

    spec = importlib.util.spec_from_file_location("dev_script", module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_rejects_unsupported_python():
    dev = _load_dev_module()

    with pytest.raises(SystemExit, match=r"requires Python 3\.10-3\.12"):
        dev.ensure_supported_python((3, 14, 0))


def test_bootstrap_selects_windows_dev_lock():
    dev = _load_dev_module()

    assert dev.select_dev_lock("Windows") == "requirements-dev-windows.txt"


def test_bootstrap_selects_linux_dev_lock():
    dev = _load_dev_module()

    assert dev.select_dev_lock("Linux") == "requirements-dev-linux.txt"


def test_bootstrap_creates_missing_venv_and_installs_platform_lock(tmp_path, monkeypatch):
    dev = _load_dev_module()
    (tmp_path / "requirements-dev-windows.txt").write_text("pytest==8.4.2\n", encoding="utf-8")

    calls = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append((cmd, cwd, kwargs))

        class _Result:
            returncode = 0
            stdout = "pip 26.1.1"
            stderr = ""

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dev.sys, "executable", "C:/Python312/python.exe")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    assert dev.bootstrap() == 0
    assert calls[0][0] == ["C:/Python312/python.exe", "-m", "venv", str(tmp_path / ".venv")]
    assert calls[-1][0] == [
        str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pip",
        "install",
        "-r",
        str(tmp_path / "requirements-dev-windows.txt"),
    ]


def test_bootstrap_reuses_existing_venv(tmp_path, monkeypatch):
    dev = _load_dev_module()
    (tmp_path / "requirements-dev-linux.txt").write_text("pytest==8.4.2\n", encoding="utf-8")
    existing_python = tmp_path / ".venv" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    existing_python.write_text("", encoding="utf-8")

    calls = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append((cmd, cwd, kwargs))

        class _Result:
            returncode = 0
            stdout = "pip 26.1.1"
            stderr = ""

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    assert dev.bootstrap() == 0
    assert all(command[:3] != [dev.sys.executable, "-m", "venv"] for command, _cwd, _kwargs in calls)
    assert calls[-1][0] == [
        str(tmp_path / ".venv" / "bin" / "python"),
        "-m",
        "pip",
        "install",
        "-r",
        str(tmp_path / "requirements-dev-linux.txt"),
    ]


def test_bootstrap_recreates_incompatible_existing_venv(tmp_path, monkeypatch):
    dev = _load_dev_module()
    (tmp_path / "requirements-dev-linux.txt").write_text("pytest==8.4.2\n", encoding="utf-8")
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)

    calls = []
    removed = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append((cmd, cwd, kwargs))

        class _Result:
            returncode = 0
            stdout = "pip 26.1.1"
            stderr = ""

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)
    real_rmtree = dev.shutil.rmtree

    def _recording_rmtree(path):
        removed.append(Path(path))
        real_rmtree(path)

    monkeypatch.setattr(dev.shutil, "rmtree", _recording_rmtree)

    assert dev.bootstrap() == 0
    assert removed == [tmp_path / ".venv"]
    assert any(command[:3] == [dev.sys.executable, "-m", "venv"] for command, _cwd, _kwargs in calls)


def test_lock_check_fails_when_generated_lock_missing(tmp_path, monkeypatch):
    dev = _load_dev_module()
    (tmp_path / "requirements.in").write_text("click>=8\n", encoding="utf-8")
    (tmp_path / "requirements-dev.in").write_text("-r requirements.in\npytest>=8\n", encoding="utf-8")
    (tmp_path / "requirements-windows.txt").write_text("click==8.3.3\n", encoding="utf-8")

    def _fake_compile_lock(source_file, output_file, root):
        output_file.write_text((tmp_path / "requirements-windows.txt").read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dev, "compile_lock", _fake_compile_lock)

    with pytest.raises(SystemExit, match=r"Missing committed lock file: requirements-dev-windows\.txt"):
        dev.lock(check=True)


def test_lock_check_uses_platform_specific_targets():
    dev = _load_dev_module()

    assert dev.select_lock_targets("Windows") == [
        ("requirements.in", "requirements-windows.txt"),
        ("requirements-dev.in", "requirements-dev-windows.txt"),
    ]
    assert dev.select_lock_targets("Linux") == [
        ("requirements.in", "requirements-linux.txt"),
        ("requirements-dev.in", "requirements-dev-linux.txt"),
    ]


def test_verify_runs_expected_commands_in_order(tmp_path, monkeypatch):
    dev = _load_dev_module()
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    calls = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append((cmd, cwd, kwargs))

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    assert dev.verify() == 0
    assert [command for command, _cwd, _kwargs in calls] == [
        [str(tmp_path / ".venv" / "Scripts" / "python.exe"), "-m", "pytest", "-q"],
        [
            str(tmp_path / ".venv" / "Scripts" / "python.exe"),
            "-c",
            "import main, app; print('import-ok')",
        ],
        [str(tmp_path / ".venv" / "Scripts" / "python.exe"), "-m", "ruff", "check", "."],
    ]


def test_verify_reports_failed_step(tmp_path, monkeypatch):
    dev = _load_dev_module()
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    def _fake_run(cmd, check, cwd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit, match=r"\[verify\] pytest failed"):
        dev.verify()


def test_smoke_runs_cli_system_check(tmp_path, monkeypatch):
    dev = _load_dev_module()
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    calls = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append((cmd, cwd, kwargs))

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    assert dev.smoke() == 0
    assert calls[0][0] == [str(venv_python), "-m", "cli", "system"]
    assert calls[0][2]["timeout"] == 10


def test_smoke_runs_api_health_check(tmp_path, monkeypatch):
    dev = _load_dev_module()
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    calls = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append((cmd, cwd, kwargs))

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    assert dev.smoke() == 0
    assert calls[1][0] == [str(venv_python), "api_server.py"]
    assert calls[1][2]["timeout"] == 15
    assert calls[1][2]["env"]["AIMODEL_SMOKE"] == "1"


def test_smoke_runs_tui_smoke_check(tmp_path, monkeypatch):
    dev = _load_dev_module()
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    calls = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append((cmd, cwd, kwargs))

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    assert dev.smoke() == 0
    assert calls[2][0] == [str(venv_python), "main.py"]
    assert calls[2][2]["timeout"] == 45
    assert calls[2][2]["env"]["AIMODEL_SMOKE"] == "1"


def test_smoke_runs_download_service_check(tmp_path, monkeypatch):
    dev = _load_dev_module()
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    calls = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append((cmd, cwd, kwargs))

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    assert dev.smoke() == 0
    assert calls[3][0] == [str(venv_python), "download_service.py"]
    assert calls[3][2]["timeout"] == 15
    assert calls[3][2]["env"]["AIMODEL_SMOKE"] == "1"
