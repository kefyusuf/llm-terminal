from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_dev_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dev.py"
    spec = importlib.util.spec_from_file_location("dev_script_lock_platform_safety", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_all_rejects_cross_platform_generation_before_compiling(tmp_path, monkeypatch):
    """One host interpreter must never generate another OS's committed lock files."""
    dev = _load_dev_module()
    compile_calls: list[str] = []

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Windows")
    monkeypatch.setattr(dev.sys, "version_info", (3, 12, 9, "final", 0))
    monkeypatch.setenv("AIMODEL_LOCK_ALL", "1")
    monkeypatch.setattr(
        dev,
        "compile_lock",
        lambda _source, output, _root: compile_calls.append(output.name),
    )

    with pytest.raises(SystemExit, match=r"cross-platform lock generation is unsupported"):
        dev.lock(check=False)

    assert compile_calls == []


def test_linux_locks_do_not_contain_windows_only_win32_setctime():
    """Linux locks must not be contaminated by Windows-only Loguru dependencies."""
    requirements_dir = Path(__file__).resolve().parents[1] / "requirements"

    for lock_name in ("requirements-linux.txt", "requirements-dev-linux.txt"):
        lock_text = (requirements_dir / lock_name).read_text(encoding="utf-8")
        assert "\nwin32-setctime==" not in f"\n{lock_text}"
