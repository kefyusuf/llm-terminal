from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_dev_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dev.py"
    spec = importlib.util.spec_from_file_location("dev_script_smoke_timeout", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_macos_api_smoke_timeout_covers_bounded_platform_probes(tmp_path, monkeypatch):
    """Darwin needs room for system_profiler plus health and shutdown deadlines."""
    dev = _load_dev_module()
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    calls = []

    def _fake_run(cmd, check, cwd, **kwargs):
        calls.append((cmd, kwargs))

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(dev, "project_root", lambda: tmp_path)
    monkeypatch.setattr(dev.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(dev.subprocess, "run", _fake_run)

    assert dev.smoke() == 0

    api_command, api_kwargs = calls[1]
    assert api_command == [str(venv_python), "-m", "api_server"]
    assert api_kwargs["timeout"] == 35
