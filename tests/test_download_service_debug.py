from download_service import _can_terminate_process, _has_duplicates


def test_has_duplicates_detects_duplicate_values():
    assert _has_duplicates(["a", "b", "a"]) is True


def test_has_duplicates_handles_unique_values():
    assert _has_duplicates(["a", "b", "c"]) is False


class _DummyProcess:
    def terminate(self):
        return None


def test_can_terminate_process_true_for_real_like_process():
    assert _can_terminate_process(_DummyProcess()) is True


def test_can_terminate_process_false_for_placeholder_object():
    assert _can_terminate_process(object()) is False
