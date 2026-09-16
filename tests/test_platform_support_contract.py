from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dev_module():
    module_path = _root() / "scripts" / "dev.py"
    spec = importlib.util.spec_from_file_location("dev_script_platform_contract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_macos_bootstrap_selects_compatibility_dev_requirements():
    """Darwin bootstrap must have an install source even before native lock files exist."""
    dev = _load_dev_module()

    assert dev.select_dev_lock("Darwin") == "requirements-dev.txt"


def test_macos_lock_management_remains_explicitly_unsupported_without_native_locks():
    """Do not pretend Linux/Windows locks are valid macOS lock artifacts."""
    dev = _load_dev_module()

    with pytest.raises(SystemExit, match="Unsupported platform for lock management: Darwin"):
        dev.select_lock_targets("Darwin")


def test_package_metadata_matches_supported_python_range():
    """pip must reject Python versions that scripts/dev.py intentionally does not support."""
    metadata = tomllib.loads((_root() / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["requires-python"] == ">=3.10,<3.15"
