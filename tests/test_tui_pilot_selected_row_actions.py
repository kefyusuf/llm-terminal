from __future__ import annotations

import asyncio

from textual.widgets import DataTable

from app.modals import PlanModeModal
from app.viewer import AIModelViewer
from results.results_view import result_unique_key


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


class _InteractionViewer(AIModelViewer):
    async def on_mount(self) -> None:
        """Mount deterministic widgets without starting services or polling."""
        self._apply_ui_mode()
        self._configure_results_table_columns(force=True)
        self.query_one("#results-table", DataTable).zebra_stripes = True

    def request_system_info_refresh(self, force=False) -> None:
        _ = force

    def request_download_poll(self, force=False) -> None:
        _ = force


def test_pilot_plan_and_compare_shortcuts_use_selected_result(monkeypatch):
    monkeypatch.setattr("tui_app.HardwareMonitor", _DummyMonitor)
    monkeypatch.setattr("app.viewer.get_provider_filter_labels", lambda: ["Ollama"])

    async def _run() -> None:
        app = _InteractionViewer()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            table = app.query_one("#results-table", DataTable)

            app.all_results = [MODEL.copy()]
            row_data = app._blank_result_row()
            row_data["source"] = "Ollama"
            row_data["name"] = MODEL["name"]
            table.add_row(
                *app._row_cells_for_current_layout(row_data),
                key=result_unique_key(app.all_results[0]),
            )
            table.move_cursor(row=0, animate=False, scroll=False)
            table.focus()
            await pilot.pause()

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
