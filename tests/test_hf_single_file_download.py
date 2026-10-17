"""Regression tests for targeted Hugging Face GGUF downloads."""

from __future__ import annotations

import pytest

from downloads.download_manager import build_download_command


def test_hf_download_command_carries_selected_target_file():
    model = {
        "source": "Hugging Face",
        "id": "Qwen/example-GGUF",
        "target_file": "example-Q4_K_M.gguf",
    }

    assert build_download_command(model) == [
        "hf_api_download",
        "Qwen/example-GGUF",
        "example-Q4_K_M.gguf",
    ]


def test_hf_download_command_rejects_missing_target_file():
    model = {"source": "Hugging Face", "id": "Qwen/example-GGUF"}

    with pytest.raises(ValueError, match="missing Hugging Face target file"):
        build_download_command(model)
