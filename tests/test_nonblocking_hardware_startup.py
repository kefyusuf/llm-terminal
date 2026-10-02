"""Hardware detection must not delay mounting or consume the UI thread."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from textual.widgets import DataTable, Input, Select

import app.startup_viewer as startup_module
import app.viewer as viewer_module
import tui_app
from app.viewer import AIModelViewer
from app.widgets import SystemInfoWidget
from providers import SearchResult

SPECS = {
    "cpu_name": "Detected CPU",
    "cpu_cores": 8,
    "ram_free": 8.0,
    "ram_total": 16.0,
    "vram_free": 4.0,
    "vram_total": 8.0,
    "gpu_name": "Detected GPU",
    "has_gpu": True,
}


def configure(monkeypatch, cached=None):
    monkeypatch.delenv("AIMODEL_SMOKE", raising=False)
    monkeypatch.setattr(
        viewer_module, "get_provider_filter_labels", lambda: ["Ollama", "Hugging Face"]
    )
    monkeypatch.setattr(startup_module, "ensure_service_running", lambda: False)
    monkeypatch.setattr(startup_module.cache_db, "get_hardware_snapshot", lambda: cached)
    monkeypatch.setattr(startup_module.cache_db, "set_hardware_snapshot", lambda specs: None)
    monkeypatch.setattr(AIModelViewer, "request_download_poll", lambda self, force=False: None)


@pytest.mark.parametrize("cached", [None, dict(SPECS, cpu_name="Cached CPU")])
def test_hardware_probe_leaves_mounted_input_and_cached_header_usable(monkeypatch, cached):
    configure(monkeypatch, cached)
    ui_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    searches = []

    class SlowMonitor:
        def __init__(self):
            assert threading.get_ident() != ui_thread, "hardware constructed on UI thread"
            entered.set()
            release.wait(5)

        def get_specs(self):
            return SPECS.copy()

    def check_running():
        assert threading.get_ident() != ui_thread, "process scan ran on UI thread"
        return True

    monkeypatch.setattr(tui_app, "HardwareMonitor", SlowMonitor)
    monkeypatch.setattr(tui_app, "check_ollama_running", check_running)

    async def run():
        app = AIModelViewer()

        def search(query, specs, **kwargs):
            searches.append((query, specs.copy()))
            return SearchResult.empty()

        monkeypatch.setattr(app.hf_provider, "search", search)
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(40):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            header = app.query_one(SystemInfoWidget)
            if cached:
                assert "Cached CPU" in str(header.render())
                assert "checking" in str(header.render())
                assert app._current_specs_for_search_ui()["cpu_name"] == "Cached CPU"
            else:
                assert "Detecting hardware" in str(header.render())
                assert app._current_specs_for_search_ui()["cpu_name"] == "Detecting hardware"
            search_input = app.query_one("#search-input", Input)
            search_input.focus()
            await pilot.press("x")
            assert search_input.value == "x"
            app.query_one("#provider-select", Select).value = "Hugging Face"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.search_counter == 1
            assert app.query_one("#results-table", DataTable).loading
            release.set()
            for _ in range(80):
                if (
                    app.latest_specs == SPECS
                    and app.ollama_running
                    and searches
                    and not app.query_one("#results-table", DataTable).loading
                ):
                    break
                await asyncio.sleep(0.025)
            assert app.latest_specs == SPECS
            assert app.ollama_running is True
            assert searches == [("x", SPECS)]
            await pilot.pause()
            assert not app.query_one("#results-table", DataTable).loading
            assert "Detected GPU" in str(header.render())

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_concurrent_workers_reuse_one_hardware_initialization(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    attempts = []

    class SlowMonitor:
        def __init__(self):
            attempts.append(True)
            entered.set()
            release.wait(5)

    monkeypatch.setattr(tui_app, "HardwareMonitor", SlowMonitor)
    app = AIModelViewer()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(app._get_hardware_monitor)
            assert entered.wait(1)
            second = pool.submit(app._get_hardware_monitor)
            # Both concurrent consumers must remain pending until detection completes.
            threading.Event().wait(0.1)
            assert not first.done() and not second.done()
            release.set()
            assert first.result(1) is second.result(1)
        assert len(attempts) == 1
    finally:
        release.set()


def test_failed_hardware_initialization_reports_pending_retry_and_recovers(monkeypatch):
    configure(monkeypatch)
    attempts = []
    recover = threading.Event()

    class RecoveringMonitor:
        def __init__(self):
            attempts.append(True)
            if not recover.is_set():
                raise OSError("probe unavailable")

        def get_specs(self):
            return SPECS.copy()

    monkeypatch.setattr(tui_app, "HardwareMonitor", RecoveringMonitor)
    monkeypatch.setattr(tui_app, "check_ollama_running", lambda: False)

    async def run():
        app = AIModelViewer()
        async with app.run_test(size=(120, 40)):
            for _ in range(80):
                if attempts and not app._system_info_refresh_running:
                    break
                await asyncio.sleep(0.01)
            assert app.latest_specs is None
            assert "unavailable" in str(app.query_one(SystemInfoWidget).render())
            prior_attempts = len(attempts)
            recover.set()
            app.request_system_info_refresh(force=True)
            for _ in range(80):
                if app.latest_specs == SPECS:
                    break
                await asyncio.sleep(0.01)
            assert app.latest_specs == SPECS
            assert len(attempts) == prior_attempts + 1
            assert "Detected GPU" in str(app.query_one(SystemInfoWidget).render())

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["search", "detail"])
def test_hardware_failure_in_user_workers_keeps_application_open(monkeypatch, operation):
    configure(monkeypatch)

    def unavailable():
        raise OSError("private probe path")

    monkeypatch.setattr(tui_app, "HardwareMonitor", unavailable)

    async def run():
        app = AIModelViewer()
        async with app.run_test(size=(120, 40)) as pilot:
            if operation == "search":
                app.start_search("example")
            else:
                app.open_hf_detail_worker({"id": "example/model"})
            await pilot.pause(0.4)
            assert "Hardware detection unavailable" in str(app.query_one("#status-bar").render())
            assert "private probe path" not in str(app.query_one("#status-bar").render())
            assert not app.query_one("#results-table", DataTable).loading
            search_input = app.query_one("#search-input", Input)
            search_input.focus()
            await pilot.press("x")
            assert search_input.value == "x"

    asyncio.run(run())
