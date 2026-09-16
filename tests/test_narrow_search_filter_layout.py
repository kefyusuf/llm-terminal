from __future__ import annotations

import asyncio

import app.viewer as viewer_module
import tui_app as base_app
from textual.widgets import Select


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
            "has_gpu": self.nvidia_available,
        }


def _configure_mount(monkeypatch) -> None:
    monkeypatch.delenv("AIMODEL_SMOKE", raising=False)
    monkeypatch.setattr(base_app, "HardwareMonitor", _DummyMonitor)
    monkeypatch.setattr(base_app, "ensure_service_running", lambda: True)
    monkeypatch.setattr(base_app.cache_db, "init_db", lambda: None)
    monkeypatch.setattr(base_app.cache_db, "cleanup_old_entries", lambda: None)
    monkeypatch.setattr(base_app.cache_db, "get_hardware_snapshot", lambda: None)
    monkeypatch.setattr(viewer_module, "get_provider_filter_labels", lambda: ("Ollama", "Hugging Face"))
    monkeypatch.setattr(
        base_app.AIModelViewer,
        "request_system_info_refresh",
        lambda self, force=False: None,
    )
    monkeypatch.setattr(
        base_app.AIModelViewer,
        "request_download_poll",
        lambda self, force=False: None,
    )


def test_use_case_selector_fits_and_stays_synced_at_80_columns(monkeypatch):
    """The compact use-case selector must fit and synchronize mouse/keyboard state."""
    _configure_mount(monkeypatch)
    app = viewer_module.AIModelViewer()
    app.ui_mode = "comfortable"
    app.compact_mode = False
    monkeypatch.setattr(app.dl, "sync_jobs", lambda force=False, jobs=None: True)

    async def _run() -> None:
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()

            panel = app.query_one("#use-case-panel")
            selector = app.query_one("#use-case-select", Select)
            panel_left = panel.region.x
            panel_right = panel.region.x + panel.region.width
            selector_right = selector.region.x + selector.region.width

            assert selector.region.width > 0
            assert selector.region.x >= panel_left
            assert selector_right <= panel_right
            assert selector_right <= app.size.width

            selector.value = "coding"
            await pilot.pause()
            assert app.use_case_filter == "coding"

            app.action_cycle_use_case()
            await pilot.pause()
            assert app.use_case_filter == "vision"
            assert selector.value == "vision"

    asyncio.run(_run())
