"""Mock-based tests for Ollama provider HTML scraping."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from requests.exceptions import RequestException

import pytest

SEARCH_HTML = """
<html><body>
<a href="/library/llama3">llama3 1.2M Pulls</a>
<a href="/library/qwen">qwen2 500K Pulls</a>
<a href="/library/codellama">codellama 300K Pulls</a>
</body></html>
"""

DETAIL_HTML = """
<html><body>
<table>
<thead><tr><th>Name</th><th>Size</th></tr></thead>
<tbody>
<tr><td>llama3:7b-q4</td><td>4.2 GB</td></tr>
<tr><td>llama3:13b-q4</td><td>7.8 GB</td></tr>
</tbody>
</table>
</body></html>
"""


class FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text
        self.headers = {}

    def json(self):
        return {}


class TestOllamaSearch:
    @patch("providers.ollama_provider.get_session")
    @patch("providers.ollama_provider.get_ollama_model_metadata", return_value=None)
    def test_search_parses_library_links(self, mock_meta, mock_session):
        mock_session.return_value.get.return_value = FakeResponse(text=SEARCH_HTML)
        from providers.ollama_provider import search_ollama_models

        specs = {
            "vram_total": 24, "vram_free": 20, "ram_total": 32, "ram_free": 28,
            "gpu_name": "RTX 4090", "has_gpu": True,
        }
        results, errors, has_more = search_ollama_models("*", specs, [], page=0, page_size=20)
        assert len(results) >= 2
        assert any("llama3" in r["name"] for r in results)

    @patch("providers.ollama_provider.get_session")
    @patch("providers.ollama_provider.get_ollama_model_metadata", return_value=None)
    def test_search_handles_empty_results(self, mock_meta, mock_session):
        mock_session.return_value.get.return_value = FakeResponse(text="<html></html>")
        from providers.ollama_provider import search_ollama_models

        specs = {"vram_total": 0, "vram_free": 0, "ram_total": 0, "ram_free": 0, "gpu_name": "", "has_gpu": False}
        results, errors, has_more = search_ollama_models("*", specs, [], page=0, page_size=20)
        assert results == []

    @patch("providers.ollama_provider.get_session")
    def test_search_network_error(self, mock_session):
        mock_session.return_value.get.side_effect = RequestException("Connection failed")
        from providers.ollama_provider import search_ollama_models

        specs = {"vram_total": 0, "vram_free": 0, "ram_total": 0, "ram_free": 0, "gpu_name": "", "has_gpu": False}
        results, errors, has_more = search_ollama_models("test", specs, [], page=0, page_size=20)
        assert results == []


class TestOllamaMetadata:
    @patch("providers.ollama_provider.get_session")
    def test_fetches_model_detail(self, mock_session):
        mock_session.return_value.get.return_value = FakeResponse(text=DETAIL_HTML)
        from providers.ollama_provider import get_ollama_model_metadata

        meta = get_ollama_model_metadata("llama3")
        assert meta is not None

    @patch("providers.ollama_provider.get_session")
    def test_handles_http_error(self, mock_session):
        mock_session.return_value.get.return_value = FakeResponse(status_code=404)
        from providers.ollama_provider import get_ollama_model_metadata

        meta = get_ollama_model_metadata("nonexistent")
        assert meta is None

    @patch("providers.ollama_provider.get_session")
    def test_handles_network_error(self, mock_session):
        mock_session.return_value.get.side_effect = RequestException("Network error")
        from providers.ollama_provider import get_ollama_model_metadata

        meta = get_ollama_model_metadata("test")
        assert meta is None
