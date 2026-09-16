from __future__ import annotations

import asyncio

import app.viewer as viewer_module
import tui_app as base_module
from app.modals import PlanModeModal
from textual.widgets import DataTable


MODEL = {
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


class _InteractionViewer(viewer_module.AIModelViewer):
    async def on_mount(self) -> None:
        """Mount only deterministic UI state needed by the interaction test."""
        self._apply_ui_mode()
        self._configure_results_table_columns(force=True)
        table = self.query_one("#results-table", DataTable)
        table.zebra_stripes = True
        self.all_results = [MODEL.copy()]
        self.refresh_table()
        table.move_cursor(row=0, animate=False, scroll=False)
        table.focus()

    def request_system_info_refresh(self, force=False) -> None:
        _ = force

    def request_download_poll(self, force=False) -> None:
        _ = force


def test_pilot_plan_and_compare_shortcuts_use_selected_result(monkeypatch):
    monkeypatch.setattr(base_module, "HardwareMonitor", _DummyMonitor)
    monkeypatch.setattr(viewer_module, "get_provider_filter_labels", lambda: ["Ollama"])

    async def _run() -> None:
        app = _InteractionViewer()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            table = app.query_one("#results-table", DataTable)
            assert table.row_count == 1
            assert table.cursor_row == 0

            await pilot.press("P")
            await pilot.pause()
            assert isinstance(app.screen, PlanModeModal)

            await pilot.click("#plan-close-btn")
            await pilot.pause()
            table.focus()

            await pilot.press("c")
            await pilot.pause()
            assert [model["name"] for model in app.comparison_set] == ["qwen2.5:7b"]

    asyncio.run(_run())
