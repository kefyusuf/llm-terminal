from service_client import is_service_compatible


def test_service_compatibility_true_for_current_version():
    assert is_service_compatible({"version": "1.5"}) is True


def test_service_compatibility_false_for_legacy_version():
    assert is_service_compatible({"version": "1.4"}) is False


def test_service_compatibility_false_for_missing_version():
    assert is_service_compatible({}) is False
