"""Search keeps its result contract while overlapping bounded detail requests."""
from __future__ import annotations

import threading
import time
from urllib.parse import urlsplit

import pytest
import requests


def test_search_overlaps_details_without_reordering_or_sharing_sessions(monkeypatch, tmp_path):
    from core import cache_db
    from providers.ollama_provider import search_ollama_models

    monkeypatch.setattr(cache_db, "_cache_db_path", tmp_path / "cache.db")
    cache_db.init_db()
    names = [f"deepseek-pilot-{index}" for index in range(6)]
    search_html = "".join(f'<a href="/library/{name}">{name} 1M Pulls</a>' for name in names)
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = peak = started = 0
    clients = {}
    closed = set()
    search_client = None
    original_close = requests.Session.close

    def fake_get(client, url, **kwargs):
        nonlocal active, peak, started, search_client
        response = requests.Response()
        response.status_code = 200
        if urlsplit(url).path == "/search":
            search_client = id(client)
            response._content = search_html.encode()
            return response
        name = urlsplit(url).path.rsplit("/", 1)[-1]
        with lock:
            position = started
            started += 1
            active += 1
            peak = max(peak, active)
            owner = threading.get_ident()
            assert id(client) not in clients or clients[id(client)] == owner
            clients[id(client)] = owner
        try:
            if position < 4:
                barrier.wait(timeout=2)
            index = names.index(name)
            time.sleep((6 - index) * .01)
            response._content = (
                '<table><tr><th>Name</th><th>Size</th></tr>'
                f'<tr><td>{name}:latest</td><td>{index + 3} GB</td></tr></table>'
            ).encode()
            return response
        finally:
            with lock:
                active -= 1

    def record_close(client):
        closed.add(id(client))
        original_close(client)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(requests.Session, "close", record_close)
    results, errors, has_more = search_ollama_models(
        "deepsek", {"vram_total": 16, "vram_free": 14, "ram_total": 32,
                    "ram_free": 20, "has_gpu": True, "gpu_name": "RTX 4060 Ti"},
        [], page_size=6,
    )
    assert errors == [] and has_more is False
    assert [result["name"] for result in results] == names
    assert [result["_size_gb"] for result in results] == [3, 4, 5, 6, 7, 8]
    assert all(result["size_source"] == "exact" for result in results)
    assert peak == 4
    assert 1 < len(clients) <= 4
    assert search_client not in clients and search_client not in closed
    assert set(clients) <= closed


@pytest.mark.parametrize("failure", [requests.exceptions.Timeout, ValueError])
def test_search_preserves_metadata_fallback_and_partial_errors(monkeypatch, tmp_path, failure):
    from core import cache_db
    from providers.ollama_provider import search_ollama_models

    monkeypatch.setattr(cache_db, "_cache_db_path", tmp_path / "cache.db")
    cache_db.init_db()

    def fake_get(client, url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        if urlsplit(url).path == "/search":
            response._content = (
                b'<a href="/library/latency-good">good 1M Pulls</a>'
                b'<a href="/library/latency-failed">failed 2M Pulls</a>'
            )
        elif url.endswith("/latency-failed"):
            raise failure("fixture failure")
        else:
            response._content = (
                b'<table><tr><th>Name</th><th>Size</th></tr>'
                b'<tr><td>latency-good:latest</td><td>3 GB</td></tr></table>'
            )
        return response

    monkeypatch.setattr(requests.Session, "get", fake_get)
    results, errors, has_more = search_ollama_models(
        "latency", {"vram_total": 16, "vram_free": 14, "ram_total": 32,
                    "ram_free": 20, "has_gpu": True, "gpu_name": "RTX 4060 Ti"}, [],
    )
    assert has_more is False
    assert results[0]["name"] == "latency-good" and results[0]["_size_gb"] == 3
    if failure is ValueError:
        assert len(results) == 1
        assert errors == ["Ollama parse failed: fixture failure"]
    else:
        assert errors == []
        assert [result["name"] for result in results] == ["latency-good", "latency-failed"]
        assert results[1]["size_source"] == "estimated"
