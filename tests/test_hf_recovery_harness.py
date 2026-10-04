"""Bounded live-harness input checks, without network or model files."""

import pytest

from scripts import verify_hf_recovery as harness


@pytest.mark.parametrize("size", [None, -1, 0, True, 101])
def test_live_sample_requires_known_positive_bounded_size(size):
    metadata = {"size_bytes": size, "resolved_revision": "a" * 40, "sha256": "b" * 64}
    with pytest.raises(ValueError, match="size"):
        harness.validate_sample(metadata, max_bytes=100)


def test_live_sample_requires_pin_and_digest():
    with pytest.raises(ValueError, match="identity"):
        harness.validate_sample({"size_bytes": 10}, max_bytes=100)


def test_harness_never_reuses_nonempty_workspace(tmp_path):
    original = tmp_path / "existing.db"
    original.write_bytes(b"user data")
    with pytest.raises(ValueError, match="empty"):
        harness.prepare_workspace(tmp_path)
    assert original.read_bytes() == b"user data"


def test_harness_retains_empty_owned_workspace(tmp_path):
    root = tmp_path / "trial"
    assert harness.prepare_workspace(root) == root.resolve()
    assert root.is_dir()
