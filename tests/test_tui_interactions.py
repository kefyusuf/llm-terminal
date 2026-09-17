from __future__ import annotations

import asyncio

from textual.widgets import DataTable

import app.viewer as viewer_module
import tui_app as tui_module
from app.modals import PlanModeModal
from app.viewer import AIModelViewer


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


def _model(name: str = "qwen2.5:7b") -> dict:
    return {
        "source": "Ollama",
        "publisher": "ollama",
        "name": name,
        "params": "7B",
        "use_case": "Coding",
        "use_case_key": "coding",
        "score": "91",
        "quant": "Q4_K_M",
        "mode": "GPU",
        "fit": "Perfect Fit",
        "downloads": 100,
        "likes": 10,
        "inst": "-",
        "download_state": "idle",
        "download_label": "Idle",
    }


def _configure_mounted_test_app(monkeypatch) -> None:
    monkeypatch.setattr(tui_module, "HardwareMonitor", _DummyMonitor)
    monkeypatch.setattr(
        viewer_module,
        "get_provider_filter_labels",
        lambda: ["Ollama", "Hugging Face"],
    )
    monkeypatch.setattr(tui_module.AIModelViewer, "_smoke_mode_enabled", lambda _self: True)
    monkeypatch.setattr(tui_module.AIModelViewer, "_finish_smoke_mode", lambda _self: None)


def test_compare_shortcut_uses_selected_datatable_row(monkeypatch):
    """The c shortcut must add the model represented by the current DataTable row key."""
    _configure_mounted_test_app(monkeypatch)

    async def _run() -> None:
        model = _model()
        app = AIModelViewer()

        async with app.run_test(size=(120, 40)) as pilot:
            app.all_results = [model]
            app.current_filter = "Ollama"
            app.refresh_table()

            table = app.query_one("#results-table", DataTable)
            table.focus()
            table.move_cursor(row=0, animate=False, scroll=False)
            await pilot.pause()

            await pilot.press("c")
            await pilot.pause()

            assert app.comparison_set == [model]

    asyncio.run(_run())


def test_plan_action_opens_modal_for_selected_datatable_row(monkeypatch):
    """Plan mode must resolve the selected model from the DataTable row key."""
    _configure_mounted_test_app(monkeypatch)

    async def _run() -> None:
        model = _model("llama3:8b")
        app = AIModelViewer()

        async with app.run_test(size=(120, 40)) as pilot:
            app.all_results = [model]
            app.current_filter = "Ollama"
            app.refresh_table()

            table = app.query_one("#results-table", DataTable)
            table.focus()
            table.move_cursor(row=0, animate=False, scroll=False)
            await pilot.pause()

            app.action_open_plan_mode()
            await pilot.pause()

            assert isinstance(app.screen, PlanModeModal)

    asyncio.run(_run())
