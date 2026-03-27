from __future__ import annotations

from types import SimpleNamespace

from providers.hf_provider import search_hf_models
from providers.ollama_provider import search_ollama_models

TEST_SPECS = {
    "has_gpu": False,
    "vram_free": 0.0,
    "ram_free": 64.0,
}


def test_hugging_face_search_returns_models_with_mocked_api(monkeypatch):
    class FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def list_models(self, **kwargs):
            return [
                SimpleNamespace(
                    modelId="Qwen/Qwen2.5-7B-Instruct-GGUF",
                    likes=120,
                    downloads=35000,
                    siblings=[SimpleNamespace(rfilename="qwen2.5-7b-instruct-q4_k_m.gguf")],
                )
            ]

    monkeypatch.setattr("providers.hf_provider.HfApi", FakeHfApi)

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


def test_ollama_search_returns_models_with_mocked_html(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text
            self.headers = {}

        def json(self):
            return {}

    html = """
    <html><body>
      <a href="/library/llama3">llama3 12.5M Pulls</a>
    </body></html>
    """

    def fake_get(url, headers=None, timeout=0):
        return FakeResponse(status_code=200, text=html)

    monkeypatch.setattr("providers.ollama_provider.requests.get", fake_get)
    monkeypatch.setattr("providers.ollama_provider.get_ollama_model_metadata", lambda _name: None)

    search_output = search_ollama_models(
        query="llama",
        specs=TEST_SPECS,
        local_models=[],
    )

    if len(search_output) == 3:
        results, errors, _has_more_pages = search_output
    else:
        results, errors = search_output

    assert not errors, f"Ollama live search returned errors: {errors}"
    assert results, "Ollama live search returned no models"
    assert all(item["source"] == "Ollama" for item in results)
