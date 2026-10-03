"""Tests for the REST API server."""

import json
import threading
import time
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

from api_server import DEFAULT_HOST, create_server
from providers.capabilities import get_all_provider_capabilities


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
            "score_provenance": {"schema_version": 1, "kind": "heuristic", "measured": False},
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
        patch(
            "api_server.detect_available_providers",
            return_value={
                "huggingface": True,
                "ollama": False,
                "lmstudio": True,
                "docker": False,
                "mlx": False,
            },
        ),
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


def _get_error(path, port):
    """Return the JSON body and status code for an expected HTTP error."""
    url = f"http://{DEFAULT_HOST}:{port}{path}"
    req = urllib.request.Request(url)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    payload = json.loads(exc_info.value.read().decode())
    return exc_info.value.code, payload


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

    def test_models_preserve_provenance_and_missing_legacy_metadata(self, api_server):
        _, port = api_server
        data = _get("/api/v1/models?provider=all", port)
        by_source = {model["source"]: model for model in data["models"]}
        assert by_source["Ollama"]["score_provenance"]["measured"] is False
        assert by_source["Hugging Face"]["score_provenance"] is None

    @pytest.mark.parametrize("provider", ["unknown", "docker", "mlx"])
    def test_models_reject_unknown_provider(self, api_server, provider):
        _, port = api_server
        status, data = _get_error(f"/api/v1/models?provider={provider}", port)
        assert status == 400
        assert "provider" in data["error"].lower()

    @pytest.mark.parametrize("limit", ["0", "-1", "101"])
    def test_models_reject_out_of_range_limit(self, api_server, limit):
        _, port = api_server
        status, data = _get_error(f"/api/v1/models?limit={limit}", port)
        assert status == 400
        assert "limit" in data["error"].lower()


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

    @pytest.mark.parametrize("context", ["0", "-1"])
    def test_plan_rejects_non_positive_context(self, api_server, context):
        _, port = api_server
        status, data = _get_error(f"/api/v1/models/llama-3-8b/plan?context={context}", port)
        assert status == 400
        assert "context" in data["error"].lower()


class TestScoresEndpoint:
    def test_scores_returns_breakdown(self, api_server):
        _, port = api_server
        data = _get("/api/v1/scores/llama-3-8b", port)
        assert "scores" in data
        scores = data["scores"]
        assert all(k in scores for k in ["quality", "speed", "fit", "context", "composite"])
        assert data["score_provenance"]["context"]["basis"] == "model_size_proxy"
        assert data["score_provenance"]["size_source"] == "model_name_estimate"


class TestProvidersEndpoint:
    def test_providers_returns_list(self, api_server):
        """The providers endpoint should return a non-empty provider list."""
        _, port = api_server
        data = _get("/api/v1/providers", port)
        assert "providers" in data
        assert len(data["providers"]) >= 1

    def test_providers_cover_canonical_registry(self, api_server):
        """The API should expose every provider in the canonical capability registry."""
        _, port = api_server
        data = _get("/api/v1/providers", port)
        names = {provider["name"] for provider in data["providers"]}
        assert names == set(get_all_provider_capabilities())

    def test_providers_expose_canonical_capabilities(self, api_server):
        """Each API provider entry should mirror its canonical capability flags."""
        _, port = api_server
        data = _get("/api/v1/providers", port)
        by_name = {provider["name"]: provider for provider in data["providers"]}

        for slug, capabilities in get_all_provider_capabilities().items():
            assert by_name[slug]["display_name"] == capabilities.display_name
            assert by_name[slug]["capabilities"] == {
                "searchable": capabilities.searchable,
                "detectable": capabilities.detectable,
                "lists_installed": capabilities.lists_installed,
                "downloadable": capabilities.downloadable,
                "paginated": capabilities.paginated,
            }
