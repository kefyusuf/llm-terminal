"""Tests for the REST API server."""

import json
import threading
import time
import urllib.request
from unittest.mock import patch

import pytest

from api_server import DEFAULT_HOST, create_server


@pytest.fixture(scope="module")
def api_server():
    """Start API server on a test port."""
    fake_ollama_results = [
        {
            "name": "llama3",
            "source": "Ollama",
            "publisher": "ollama",
            "params": "8B",
            "quant": "Q4_K_M",
            "size": "~4.8 GB",
            "use_case_key": "general",
            "fit": "[bold green]Perfect[/bold green]",
            "mode": "[green]GPU[/green]",
            "score_quality": 72,
            "score_speed": 60,
            "score_fit": 80,
            "score_context": 55,
            "score_composite": 67,
            "estimated_tok_s": 42.0,
            "is_moe": False,
            "total_experts": 0,
            "active_experts": 0,
        }
    ]
    fake_hf_results = [
        {
            "name": "Qwen2.5-7B-Instruct-GGUF",
            "source": "Hugging Face",
            "publisher": "Qwen",
            "params": "7B",
            "quant": "Q4_K_M",
            "size": "~4.8 GB",
            "use_case_key": "chat",
            "fit": "[bold yellow]Partial[/bold yellow]",
            "mode": "[yellow]GPU+CPU[/yellow]",
            "score_quality": 68,
            "score_speed": 45,
            "score_fit": 58,
            "score_context": 52,
            "score_composite": 56,
            "estimated_tok_s": 25.0,
            "is_moe": False,
            "total_experts": 0,
            "active_experts": 0,
        }
    ]

    def fake_search_ollama_models(*args, **kwargs):
        return fake_ollama_results, [], False

    def fake_search_hf_models(*args, **kwargs):
        return fake_hf_results, []

    patchers = [
        patch("api_server.search_ollama_models", side_effect=fake_search_ollama_models),
        patch("api_server.search_hf_models", side_effect=fake_search_hf_models),
        patch("api_server.get_installed_ollama_models", return_value=[]),
    ]
    for p in patchers:
        p.start()

    server = create_server(DEFAULT_HOST, 0)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)  # Allow server to start
    yield server, port
    server.shutdown()
    for p in reversed(patchers):
        p.stop()


def _get(path, port):
    url = f"http://{DEFAULT_HOST}:{port}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


class TestHealthEndpoint:
    def test_health_ok(self, api_server):
        _, port = api_server
        data = _get("/health", port)
        assert data["status"] == "ok"
        assert "api_version" in data

    def test_health_returns_service_name(self, api_server):
        _, port = api_server
        data = _get("/health", port)
        assert data["service"] == "ai-model-explorer-api"


class TestSystemEndpoint:
    def test_system_returns_hardware_info(self, api_server):
        _, port = api_server
        data = _get("/api/v1/system", port)
        assert "cpu_name" in data
        assert "ram_total_gb" in data
        assert "has_gpu" in data
        assert "backend" in data

    def test_system_ram_is_positive(self, api_server):
        _, port = api_server
        data = _get("/api/v1/system", port)
        assert data["ram_total_gb"] > 0


class TestModelsEndpoint:
    def test_models_returns_list(self, api_server):
        _, port = api_server
        data = _get("/api/v1/models?limit=3", port)
        assert "models" in data
        assert "total" in data
        assert isinstance(data["models"], list)

    def test_models_have_scores(self, api_server):
        _, port = api_server
        data = _get("/api/v1/models?limit=1&provider=ollama", port)
        if data["models"]:
            model = data["models"][0]
            assert "scores" in model
            assert "quality" in model["scores"]
            assert "composite" in model["scores"]


class TestPlanEndpoint:
    def test_plan_returns_hardware_requirements(self, api_server):
        _, port = api_server
        data = _get("/api/v1/models/llama-3-8b/plan", port)
        assert data["model"] == "llama-3-8b"
        assert "plans" in data
        assert len(data["plans"]) > 0

    def test_plan_with_custom_context(self, api_server):
        _, port = api_server
        data = _get("/api/v1/models/llama-3-8b/plan?context=32768", port)
        assert data["context_length"] == 32768


class TestScoresEndpoint:
    def test_scores_returns_breakdown(self, api_server):
        _, port = api_server
        data = _get("/api/v1/scores/llama-3-8b", port)
        assert "scores" in data
        scores = data["scores"]
        assert all(k in scores for k in ["quality", "speed", "fit", "context", "composite"])


class TestProvidersEndpoint:
    def test_providers_returns_list(self, api_server):
        _, port = api_server
        data = _get("/api/v1/providers", port)
        assert "providers" in data
        assert len(data["providers"]) >= 1
