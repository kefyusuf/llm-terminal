import time

from search_cache import SearchCache


def test_search_cache_returns_entry_when_fresh_and_compatible():
    cache = SearchCache(
        ttl_seconds=30,
        max_entries=5,
        ram_threshold_gb=1.0,
        vram_threshold_gb=1.0,
    )
    specs = {"has_gpu": False, "ram_free": 8.0, "vram_free": 0.0}
    cache.set(
        "ollama:qwen",
        results=[{"name": "qwen"}],
        error="",
        has_more_pages=False,
        specs=specs,
    )

    cached = cache.get("ollama:qwen", specs)
    assert cached is not None
    assert cached["results"][0]["name"] == "qwen"


def test_search_cache_expires_old_entry(monkeypatch):
    cache = SearchCache(
        ttl_seconds=5,
        max_entries=5,
        ram_threshold_gb=1.0,
        vram_threshold_gb=1.0,
    )
    specs = {"has_gpu": False, "ram_free": 8.0, "vram_free": 0.0}
    cache.set(
        "hf:llama:page0",
        results=[{"name": "llama"}],
        error="",
        has_more_pages=True,
        specs=specs,
    )

    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + 10)
    assert cache.get("hf:llama:page0", specs) is None


def test_search_cache_invalidates_when_ram_changes_beyond_threshold():
    cache = SearchCache(
        ttl_seconds=30,
        max_entries=5,
        ram_threshold_gb=0.5,
        vram_threshold_gb=1.0,
    )
    cache.set(
        "hf:qwen:page0",
        results=[{"name": "qwen"}],
        error="",
        has_more_pages=True,
        specs={"has_gpu": False, "ram_free": 8.0, "vram_free": 0.0},
    )

    current_specs = {"has_gpu": False, "ram_free": 6.0, "vram_free": 0.0}
    assert cache.get("hf:qwen:page0", current_specs) is None


def test_search_cache_evicts_oldest_entry_when_over_capacity():
    cache = SearchCache(
        ttl_seconds=30,
        max_entries=2,
        ram_threshold_gb=1.0,
        vram_threshold_gb=1.0,
    )
    specs = {"has_gpu": False, "ram_free": 8.0, "vram_free": 0.0}
    cache.set("a", results=[], error="", has_more_pages=False, specs=specs)
    cache.set("b", results=[], error="", has_more_pages=False, specs=specs)
    cache.set("c", results=[], error="", has_more_pages=False, specs=specs)

    assert cache.get("a", specs) is None
    assert cache.get("b", specs) is not None
    assert cache.get("c", specs) is not None
