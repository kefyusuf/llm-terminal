from __future__ import annotations

import asyncio
import threading

import tui_app as app_module
from tui_app import AIModelViewer


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
    monkeypatch.setattr(app_module, "HardwareMonitor", _DummyMonitor)
    monkeypatch.setattr(app_module.cache_db, "init_db", lambda: None)
    monkeypatch.setattr(app_module.cache_db, "cleanup_old_entries", lambda: None)
    monkeypatch.setattr(app_module.cache_db, "get_hardware_snapshot", lambda: None)
    monkeypatch.setattr(AIModelViewer, "request_system_info_refresh", lambda self, force=False: None)


def test_mount_runs_service_readiness_and_initial_job_io_off_ui_thread(monkeypatch):
    """Mount must render without performing service startup/network I/O on Textual's UI thread."""
    _configure_mount(monkeypatch)
    ui_thread_id = threading.get_ident()
    applied = threading.Event()
    jobs = [
        {
            "target_id": "ollama:qwen2.5:7b",
            "source": "Ollama",
            "publisher": "ollama",
            "name": "qwen2.5:7b",
            "status": "completed",
            "detail": "Done",
            "progress": "",
        }
    ]

    def _ensure_service_running() -> bool:
        assert threading.get_ident() != ui_thread_id
        return True

    def _list_jobs(*, limit=50, timeout=2.0):
        _ = (limit, timeout)
        assert threading.get_ident() != ui_thread_id
        return jobs

    monkeypatch.setattr(app_module, "ensure_service_running", _ensure_service_running)
    monkeypatch.setattr(app_module, "list_jobs", _list_jobs, raising=False)

    app = AIModelViewer()

    def _sync_jobs(force=False, jobs=None):
        assert threading.get_ident() == ui_thread_id
        assert force is True
        assert jobs is not None
        assert jobs[0]["target_id"] == "ollama:qwen2.5:7b"
        applied.set()
        return True

    monkeypatch.setattr(app.dl, "sync_jobs", _sync_jobs)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await asyncio.wait_for(asyncio.to_thread(applied.wait), timeout=2.0)
            await pilot.pause()

    asyncio.run(_run())
