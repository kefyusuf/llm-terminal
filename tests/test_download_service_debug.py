from download_service import (
    _can_terminate_process,
    _has_duplicates,
    _is_hf_api_command,
    _repo_id_from_hf_command,
)


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


def test_is_hf_api_command_true_for_valid_payload():
    assert _is_hf_api_command(["hf_api_download", "Qwen/Foo"]) is True


def test_is_hf_api_command_false_for_non_hf_payload():
    assert _is_hf_api_command(["ollama", "pull", "qwen2.5"]) is False


def test_repo_id_from_hf_command_returns_empty_when_missing():
    assert _repo_id_from_hf_command(["hf_api_download"]) == ""
