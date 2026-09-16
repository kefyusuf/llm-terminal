from __future__ import annotations

import asyncio
import threading

import app.startup_viewer as startup_module
import app.viewer as viewer_module
import tui_app as app_module
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


def _configure_mount(monkeypatch) -> None:
    monkeypatch.delenv("AIMODEL_SMOKE", raising=False)
    monkeypatch.setattr(app_module, "HardwareMonitor", _DummyMonitor)
    monkeypatch.setattr(viewer_module, "get_provider_filter_labels", lambda: ["Ollama"])
    monkeypatch.setattr(startup_module.cache_db, "init_db", lambda: None)
    monkeypatch.setattr(startup_module.cache_db, "cleanup_old_entries", lambda: None)
    monkeypatch.setattr(startup_module.cache_db, "get_hardware_snapshot", lambda: None)
    monkeypatch.setattr(AIModelViewer, "request_system_info_refresh", lambda self, force=False: None)
    monkeypatch.setattr(AIModelViewer, "request_download_poll", lambda self, force=False: None)


def test_mount_runs_service_readiness_and_initial_job_io_off_ui_thread(monkeypatch):
    """Mounted runtime startup must keep service readiness and first job I/O off the UI thread."""
    _configure_mount(monkeypatch)
    ui_thread_id = threading.get_ident()
    ensure_thread_ids: list[int] = []
    list_thread_ids: list[int] = []
    sync_calls: list[tuple[int, bool, object]] = []
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
        ensure_thread_ids.append(threading.get_ident())
        return True

    def _list_jobs(*, limit=50, timeout=2.0):
        _ = (limit, timeout)
        list_thread_ids.append(threading.get_ident())
        return jobs

    monkeypatch.setattr(startup_module, "ensure_service_running", _ensure_service_running)
    monkeypatch.setattr(startup_module, "list_jobs", _list_jobs)

    app = AIModelViewer()

    def _sync_jobs(force=False, jobs=None):
        sync_calls.append((threading.get_ident(), force, jobs))
        return True

    monkeypatch.setattr(app.dl, "sync_jobs", _sync_jobs)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(40):
                if sync_calls:
                    break
                await asyncio.sleep(0.05)
            await pilot.pause()

    asyncio.run(_run())

    assert ensure_thread_ids
    assert ensure_thread_ids[0] != ui_thread_id
    assert list_thread_ids
    assert list_thread_ids[0] != ui_thread_id
    assert sync_calls
    sync_thread_id, force, snapshot = sync_calls[0]
    assert sync_thread_id == ui_thread_id
    assert force is True
    assert snapshot is not None
    assert snapshot[0]["target_id"] == "ollama:qwen2.5:7b"
