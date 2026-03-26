import pytest

from download_manager import (
    build_download_command,
    download_target_id,
    normalize_target_id,
)


def test_build_download_command_hf():
    model = {"source": "Hugging Face", "id": "unsloth/Qwen3-Coder-Next-GGUF"}
    command = build_download_command(model)
    assert command == ["hf_api_download", "unsloth/Qwen3-Coder-Next-GGUF"]


def test_build_download_command_ollama():
    model = {"source": "Ollama", "name": "llama3.2"}
    assert build_download_command(model) == ["ollama", "pull", "llama3.2"]


def test_build_download_command_rejects_unknown_source():
    with pytest.raises(ValueError):
        build_download_command({"source": "Other"})


def test_download_target_id_prefers_id():
    assert download_target_id({"source": "Hugging Face", "id": "foo/bar"}) == "hugging face:foo/bar"


def test_download_target_id_normalizes_case_and_space():
    assert (
        download_target_id({"source": " Ollama ", "name": " Qwen3-Coder-Next "})
        == "ollama:qwen3-coder-next"
    )


def test_normalize_target_id_handles_missing_source_prefix():
    assert normalize_target_id("Qwen3") == "unknown:qwen3"
