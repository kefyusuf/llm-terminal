"""Compatibility coverage for the download-service client protocol floor."""

from downloads.service_client import is_service_compatible


def test_service_compatibility_true_for_current_version():
    """The artifact-plan 2.0 service protocol must be accepted."""
    assert is_service_compatible({"version": "2.0"}) is True


def test_service_compatibility_false_for_pre_auth_version():
    """The pre-auth 1.7 service must be restarted rather than reused."""
    assert is_service_compatible({"version": "1.7"}) is False


def test_service_compatibility_false_for_legacy_version():
    """Older legacy service versions must remain incompatible."""
    assert is_service_compatible({"version": "1.6"}) is False


def test_service_compatibility_false_for_missing_version():
    """A health response without a version must be incompatible."""
    assert is_service_compatible({}) is False
