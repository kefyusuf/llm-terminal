"""Opt-in ready-result snapshots preserve the normal provider contract."""
import requests


def _transport(monkeypatch, tmp_path):
    from core import cache_db

    monkeypatch.setattr(cache_db, "_cache_db_path", tmp_path / "cache.db")
    cache_db.init_db()

    def fake_get(client, url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        if "/search?" in url:
            response._content = (
                b'<a href="/library/preview-first">first 1M Pulls</a>'
                b'<a href="/library/preview-second">second 1M Pulls</a>'
            )
        else:
            name = url.rsplit("/", 1)[-1]
            response._content = (
                '<table><tr><th>Name</th><th>Size</th></tr>'
                f'<tr><td>{name}:latest</td><td>3 GB</td></tr></table>'
            ).encode()
        return response

    monkeypatch.setattr(requests.Session, "get", fake_get)
    return {"has_gpu": False, "ram_total": 16, "ram_free": 12,
            "vram_total": 0, "vram_free": 0, "gpu_name": "No GPU"}


def test_preview_cap_and_consumer_mutation_do_not_change_final_scores(monkeypatch, tmp_path):
    from providers.ollama_provider import search_ollama_models

    specs = _transport(monkeypatch, tmp_path)
    published = []

    def preview(results):
        published.append([r["name"] for r in results])
        assert "score_composite" in results[0] and results[0]["size_source"] == "exact"
        results[0]["name"] = "consumer-change"
        results[0]["score_provenance"]["consumer-only"] = True

    results, errors, has_more = search_ollama_models(
        "preview", specs, [], page_size=1, _on_results=preview,
    )
    assert errors == [] and has_more is False
    assert published == [["preview-first"]]
    assert results[0]["name"] == "preview-first"
    assert "consumer-only" not in results[0]["score_provenance"]


def test_multi_provider_search_does_not_publish_ollama_only_prefix(monkeypatch, tmp_path):
    from providers import SearchResult
    from providers.ollama_provider import OllamaProvider
    from search.search_orchestrator import SearchOrchestrator

    specs = _transport(monkeypatch, tmp_path)

    class Monitor:
        def get_specs(self):
            return specs

    class HF:
        def search(self, *args, **kwargs):
            return SearchResult(results=[{"name": "hf-final"}])

    provider = OllamaProvider()
    provider.installed = ["preview-first"]
    previews = []
    orchestrator = SearchOrchestrator(
        monitor=Monitor(), hf_provider=HF(), ollama_provider=provider,
        on_progress=lambda *_: None, cancel_check=lambda: False,
        on_results=lambda *args: previews.append(args),
    )
    outcome = orchestrator.search(
        search_id=1, query="preview", providers=["ollama", "huggingface"],
        page=0, page_size=1,
    )
    assert previews == []
    assert [r["name"] for r in outcome.results] == ["preview-first", "hf-final"]
