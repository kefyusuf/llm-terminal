import pytest

from providers.hf_provider import search_hf_models
from providers.ollama_provider import get_installed_ollama_models, search_ollama_models


TEST_SPECS = {
    "has_gpu": False,
    "vram_free": 0.0,
    "ram_free": 64.0,
}


@pytest.mark.live
def test_live_hugging_face_search_returns_models():
    results, errors = search_hf_models(
        query="qwen",
        specs=TEST_SPECS,
        model_info_cache={},
        limit=5,
    )

    assert not errors, f"HF live search returned errors: {errors}"
    assert results, "HF live search returned no models"
    assert all(item["source"] == "Hugging Face" for item in results)
    assert all("publisher" in item and item["publisher"] for item in results)


@pytest.mark.live
def test_live_ollama_search_returns_models():
    local_models = get_installed_ollama_models()
    results, errors = search_ollama_models(
        query="llama",
        specs=TEST_SPECS,
        local_models=local_models,
    )

    assert not errors, f"Ollama live search returned errors: {errors}"
    assert results, "Ollama live search returned no models"
    assert all(item["source"] == "Ollama" for item in results)
