"""Regression coverage for package, CLI, and changelog version alignment."""


def _root():
    """Return the repository root for metadata fixtures used by this test module."""
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def _project_version() -> str:
    """Read the declared project version from pyproject.toml."""
    for line in (_root() / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith('version = "'):
            return line.split('"', 2)[1]
    raise AssertionError("project version not found")


def _requires_python() -> str:
    """Read the declared Python support range from pyproject.toml."""
    for line in (_root() / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith('requires-python = "'):
            return line.split('"', 2)[1]
    raise AssertionError("requires-python not found")


def _latest_changelog_version() -> str:
    """Read the latest release version heading from CHANGELOG.md."""
    for line in (_root() / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            return line.removeprefix("## ").split(" - ", 1)[0].strip()
    raise AssertionError("changelog version not found")


def test_package_cli_and_changelog_versions_match():
    """Published package metadata, CLI output, and latest release notes must stay aligned."""
    from cli import get_version

    project_version = _project_version()
    assert get_version() == project_version
    assert _latest_changelog_version() == project_version


def test_package_python_range_matches_runtime_support_policy():
    """Wheel metadata must not advertise Python versions rejected by scripts/dev.py."""
    assert _requires_python() == ">=3.10,<3.15"
