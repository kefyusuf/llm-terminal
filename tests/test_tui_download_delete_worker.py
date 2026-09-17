from __future__ import annotations

import asyncio
from types import SimpleNamespace

from textual.app import App

import app.viewer as viewer_module
import tui_app as tui_module
from app.viewer import AIModelViewer, DownloadJobModal


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


def _make_viewer(monkeypatch):
    monkeypatch.setattr(tui_module, "HardwareMonitor", _DummyMonitor)
    monkeypatch.setattr(viewer_module, "get_provider_filter_labels", lambda: ["Ollama"])
    return AIModelViewer()


def test_delete_download_entry_schedules_worker_before_blocking_manager_call(monkeypatch):
    """The public TUI action must schedule blocking deletion instead of executing it inline."""
    viewer = _make_viewer(monkeypatch)
    scheduled = []

    def _unexpected_inline_delete(*_args, **_kwargs):
        raise AssertionError("blocking DownloadManager.delete_entry ran on the UI action path")

    monkeypatch.setattr(viewer.dl, "delete_entry", _unexpected_inline_delete)
    monkeypatch.setattr(
        viewer,
        "_run_delete_download_entry_worker",
        lambda target_id, delete_data=False: scheduled.append((target_id, delete_data)),
    )

    viewer.delete_download_entry("ollama:qwen2.5:7b", delete_data=True)

    assert scheduled == [("ollama:qwen2.5:7b", True)]


def test_cancel_and_delete_delegates_single_combined_delete_operation():
    """The manager's delete-data path owns cancellation; the modal must not cancel twice."""

    class _HostApp(App):
        def __init__(self):
            super().__init__()
            self.cancel_calls = []
            self.delete_calls = []

        def cancel_model_download(self, model):
            self.cancel_calls.append(model)

        def delete_download_entry(self, target_id, delete_data=False):
            self.delete_calls.append((target_id, delete_data))

    async def _run() -> None:
        app = _HostApp()
        entry = {
            "target_id": "ollama:qwen2.5:7b",
            "source": "Ollama",
            "name": "qwen2.5:7b",
            "state": "downloading",
        }
        modal = DownloadJobModal(entry)

        async with app.run_test() as pilot:
            app.push_screen(modal)
            await pilot.pause()

            modal.cancel_and_delete()
            await pilot.pause()

            assert app.cancel_calls == []
            assert app.delete_calls == [("ollama:qwen2.5:7b", True)]

    asyncio.run(_run())


def test_download_history_detail_row_opens_runtime_modal(monkeypatch):
    """Download-history detail selection must use the runtime modal with single-operation delete."""
    viewer = _make_viewer(monkeypatch)
    target_id = "ollama:qwen2.5:7b"
    entry = {
        "target_id": target_id,
        "source": "Ollama",
        "name": "qwen2.5:7b",
        "state": "completed",
    }
    viewer.dl.download_registry[target_id] = entry
    pushed = []
    monkeypatch.setattr(viewer, "push_screen", pushed.append)

    event = SimpleNamespace(
        data_table=SimpleNamespace(id="download-history-table", cursor_column=0),
        row_key=SimpleNamespace(value=target_id),
    )

    viewer.on_data_table_row_selected(event)

    assert len(pushed) == 1
    assert isinstance(pushed[0], DownloadJobModal)
    assert pushed[0].entry is entry
