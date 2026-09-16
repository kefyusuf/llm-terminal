from __future__ import annotations

import asyncio

from textual.app import App

import tui_app as app_module
from app.modals import DownloadJobModal


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
    monkeypatch.setattr(app_module, "HardwareMonitor", _DummyMonitor)
    return app_module.AIModelViewer()


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
        raising=False,
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
