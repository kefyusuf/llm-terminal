"""Regression coverage for package, CLI, and changelog version alignment."""


def _root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def _project_version() -> str:
    for line in (_root() / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith('version = "'):
            return line.split('"', 2)[1]
    raise AssertionError("project version not found")


def _latest_changelog_version() -> str:
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
