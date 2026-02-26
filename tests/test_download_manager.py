import sys

import pytest

from download_manager import build_download_command, download_target_id


def test_build_download_command_hf():
    model = {"source": "Hugging Face", "id": "unsloth/Qwen3-Coder-Next-GGUF"}
    command = build_download_command(model)
    assert command[:3] == [
        sys.executable,
        "-m",
        "huggingface_hub.commands.huggingface_cli",
    ]
    assert command[3:] == [
        "download",
        "unsloth/Qwen3-Coder-Next-GGUF",
        "--include",
        "*.gguf",
    ]


def test_build_download_command_ollama():
    model = {"source": "Ollama", "name": "llama3.2"}
    assert build_download_command(model) == ["ollama", "pull", "llama3.2"]


def test_build_download_command_rejects_unknown_source():
    with pytest.raises(ValueError):
        build_download_command({"source": "Other"})


def test_download_target_id_prefers_id():
    assert (
        download_target_id({"source": "Hugging Face", "id": "foo/bar"})
        == "Hugging Face:foo/bar"
    )
