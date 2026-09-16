from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_dev_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dev.py"
    spec = importlib.util.spec_from_file_location("dev_script_macos", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_macos_bootstrap_uses_generic_dev_requirements_wrapper():
    dev = _load_dev_module()

    assert dev.select_dev_lock("Darwin") == "requirements-dev.txt"
