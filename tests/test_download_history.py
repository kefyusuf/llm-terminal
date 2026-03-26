from download_history import (
    action_label_for_entry,
    cancel_model_payload,
    fallback_entry_from_target,
    is_external_entry,
)


def test_fallback_entry_from_target_splits_source_and_name():
    entry = fallback_entry_from_target("ollama:qwen3")
    assert entry["source"] == "ollama"
    assert entry["name"] == "qwen3"
    assert entry["state"] == "idle"


def test_is_external_entry_true_for_idle_unknown_detail():
    entry = {"state": "idle", "detail": "manually synced outside app"}
    assert is_external_entry(entry) is True


def test_is_external_entry_false_for_managed_detail():
    entry = {"state": "idle", "detail": "Completed"}
    assert is_external_entry(entry) is False


def test_action_label_for_entry_cancel_for_active_state():
    entry = {"state": "downloading", "detail": "40%"}
    assert action_label_for_entry(entry) == "Cancel"


def test_action_label_for_entry_external_for_unmanaged_idle():
    entry = {"state": "idle", "detail": "external artifact"}
    assert action_label_for_entry(entry) == "External"


def test_cancel_model_payload_extracts_id_from_target():
    payload = cancel_model_payload(
        "hugging face:org/repo", {"source": "Hugging Face", "name": "repo"}
    )
    assert payload == {"source": "Hugging Face", "name": "repo", "id": "org/repo"}
