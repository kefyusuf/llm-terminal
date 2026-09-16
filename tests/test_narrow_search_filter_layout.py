from __future__ import annotations

import asyncio

import tui_app as base_app
from app.viewer import AIModelViewer
from textual.widgets import RadioButton, RadioSet


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


def test_use_case_controls_stay_inside_panel_at_80_columns(monkeypatch):
    """Every mouse-selectable use-case option must remain visible at 80 columns."""
    _configure_mount(monkeypatch)
    app = AIModelViewer()
    app.ui_mode = "comfortable"
    app.compact_mode = False
    monkeypatch.setattr(app.dl, "sync_jobs", lambda force=False, jobs=None: True)

    async def _run() -> None:
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()

            panel = app.query_one("#use-case-panel")
            radio_set = app.query_one("#use-case-filter", RadioSet)
            buttons = list(radio_set.query(RadioButton))

            panel_left = panel.region.x
            panel_right = panel.region.x + panel.region.width
            screen_right = app.size.width

            assert len(buttons) == 8
            for button in buttons:
                assert button.region.width > 0
                assert button.region.x >= panel_left
                assert button.region.x + button.region.width <= panel_right
                assert button.region.x + button.region.width <= screen_right

    asyncio.run(_run())
