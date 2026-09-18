from __future__ import annotations

import asyncio

from textual.widgets import DataTable

from app.viewer import AIModelViewer

STALE_MODEL = {
    "source": "Ollama",
    "publisher": "ollama",
    "name": "qwen2.5:7b",
    "id": "qwen2.5:7b",
    "params": "7B",
    "use_case": "Coding",
    "use_case_key": "coding",
    "score": 90,
    "quant": "Q4_K_M",
    "mode": "GPU",
    "fit": "Fit",
}


class _DummyMonitor:
    def __init__(self):
        self.cpu_name = "Test CPU"
        self.cpu_cores = 8
        self.gpu_name = "No GPU"
        self.nvidia_available = False

    def get_specs(self):
        return {
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "ram_free": 8.0,
            "ram_total": 16.0,
            "vram_free": 0.0,
            "vram_total": 0.0,
            "gpu_name": self.gpu_name,
            "has_gpu": False,
        }


class _FailedOutcome:
    def __init__(self, providers):
        self.results = []
        self.errors = ["Ollama search unavailable"]
        self.has_more_pages = False
        self.result_count = 0
        self.providers = list(providers)
        self.cancelled = False


class _SearchViewer(AIModelViewer):
    async def on_mount(self) -> None:
        self._apply_ui_mode()
        self._configure_results_table_columns(force=True)
        self.query_one("#results-table", DataTable).zebra_stripes = True

    def request_system_info_refresh(self, force=False) -> None:
        _ = force

    def request_download_poll(self, force=False) -> None:
        _ = force


def test_expired_cache_is_used_only_after_live_search_failure(monkeypatch):
    monkeypatch.setattr("tui_app.HardwareMonitor", _DummyMonitor)
    monkeypatch.setattr("app.viewer.get_provider_filter_labels", lambda: ["Ollama"])

    calls: list[tuple[int, str, tuple[str, ...]]] = []

    def _failed_search(self, *, search_id, query, providers, **_kwargs):
        _ = self
        calls.append((search_id, query, tuple(providers)))
        return _FailedOutcome(providers)

    monkeypatch.setattr("search.search_orchestrator.SearchOrchestrator.search", _failed_search)

    async def _run() -> None:
        app = _SearchViewer()
        async with app.run_test(size=(100, 30)) as pilot:
            specs = _DummyMonitor().get_specs()
            app.latest_specs = specs
            app.search_cache.ttl_seconds = -1
            app.search_cache.set(
                "ollama:qwen",
                results=[STALE_MODEL],
                error="",
                has_more_pages=False,
                specs=specs,
            )

            app.start_search("qwen")
            await pilot.pause(0.5)

            assert calls == [(1, "qwen", ("ollama",))]
            assert [model["name"] for model in app.all_results] == ["qwen2.5:7b"]
            assert app.last_search_error == "Offline — showing cached results"

    asyncio.run(_run())
