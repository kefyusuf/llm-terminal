from download_lifecycle import (
    cancel_error_detail_from_http_error,
    delete_error_detail_from_http_error,
    entry_identity_keys,
    reset_results_download_state,
    should_cancel_before_delete,
    should_delete_ollama_data,
    trim_download_registry,
    upsert_download_registry_entry,
)


class _HttpErrorStub:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload


def test_cancel_error_detail_uses_error_payload_message():
    err = _HttpErrorStub(b'{"error": "job not found"}')
    assert cancel_error_detail_from_http_error(err) == "Cancel failed: job not found"


def test_delete_error_detail_maps_active_job_message():
    err = _HttpErrorStub(b'{"error": "cannot delete active job"}')
    assert delete_error_detail_from_http_error(err) == "Cannot delete active job. Cancel it first."


def test_should_cancel_before_delete_only_for_active_states():
    assert should_cancel_before_delete(True, "downloading") is True
    assert should_cancel_before_delete(True, "completed") is False
    assert should_cancel_before_delete(False, "downloading") is False


def test_should_delete_ollama_data_requires_ollama_and_model_name():
    assert should_delete_ollama_data(True, "ollama", "qwen") is True
    assert should_delete_ollama_data(True, "hugging face", "repo") is False
    assert should_delete_ollama_data(True, "ollama", "") is False


def test_entry_identity_keys_normalizes_values():
    source_key, name_key = entry_identity_keys({"source": " Ollama ", "name": " Qwen "})
    assert source_key == "ollama"
    assert name_key == "qwen"


def test_reset_results_download_state_updates_direct_and_identity_matches():
    all_results = [
        {
            "source": "Ollama",
            "name": "qwen",
            "id": "qwen",
            "download_state": "downloading",
            "download_label": "Downloading",
            "download_detail": "40%",
        },
        {
            "source": "Ollama",
            "name": "qwen",
            "id": "qwen:alt",
            "download_state": "queued",
            "download_label": "Queued",
            "download_detail": "",
        },
    ]

    def _target_id_for_item(item):
        return f"{item['source'].lower()}:{item['id']}"

    updated = reset_results_download_state(
        all_results,
        target_id="ollama:qwen",
        source_key="ollama",
        name_key="qwen",
        target_id_for_item=_target_id_for_item,
    )

    assert updated == 2
    assert all(item["download_state"] == "idle" for item in all_results)
    assert all(item["download_label"] == "Idle" for item in all_results)


def test_upsert_download_registry_entry_and_trim_behaviour():
    registry = {}
    upsert_download_registry_entry(
        registry,
        target_id="a",
        model={"source": "Ollama", "publisher": "meta", "name": "qwen"},
        state="queued",
        label="Queued",
        detail="",
        now=10.0,
    )
    upsert_download_registry_entry(
        registry,
        target_id="b",
        model={"source": "Ollama", "publisher": "meta", "name": "llama"},
        state="completed",
        label="Completed",
        detail="",
        now=20.0,
    )
    upsert_download_registry_entry(
        registry,
        target_id="c",
        model={"source": "Ollama", "publisher": "meta", "name": "mistral"},
        state="failed",
        label="Failed",
        detail="err",
        now=30.0,
    )

    trim_download_registry(registry, 2)

    assert "a" not in registry
    assert "b" in registry and "c" in registry
