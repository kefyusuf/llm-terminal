"""Slow external runtime probes must not block REST or mounted TUI input."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request

from textual.widgets import Input, Select

import api_server
import app.startup_viewer as startup_module
import app.viewer as viewer_module
import tui_app
from app.viewer import AIModelViewer


class Monitor:
    cpu_name = "Test CPU"
    cpu_cores = 8
    gpu_name = "No GPU"
    nvidia_available = False

    def get_specs(self):
        return {}


def test_rest_returns_pending_snapshot_while_runtime_probe_is_blocked(monkeypatch):
    release = threading.Event()
    entered = threading.Event()

    def detect():
        entered.set()
        release.wait(5)
        return {"huggingface": True, "ollama": False, "lmstudio": True}

    monkeypatch.setattr(api_server, "detect_available_providers", detect)
    server, thread = api_server.start_server_background(port=0, monitor=Monitor())
    url = f"http://127.0.0.1:{server.server_port}/api/v1/providers"
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            pending = json.load(response)
        assert entered.wait(1)
        assert pending["discovery"]["status"] == "pending"
        assert pending["discovery"]["refreshing"] is True
        assert {p["name"] for p in pending["providers"]} == {
            "huggingface", "ollama", "lmstudio", "docker", "mlx"
        }

        release.set()
        for _ in range(30):
            with urllib.request.urlopen(url, timeout=1) as response:
                ready = json.load(response)
            if ready["discovery"]["status"] == "ready":
                break
            threading.Event().wait(0.01)
        assert ready["discovery"]["status"] == "ready"
        by_name = {p["name"]: p for p in ready["providers"]}
        assert by_name["lmstudio"]["available"] is True
        assert by_name["ollama"]["available"] is False
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_slow_discovery_keeps_mounted_search_input_usable(monkeypatch):
    ui_thread = threading.get_ident()
    release = threading.Event()
    entered = threading.Event()

    def labels():
        assert threading.get_ident() != ui_thread, "provider probe ran on the UI thread"
        entered.set()
        release.wait(5)
        return ["Ollama", "Hugging Face", "LM Studio"]

    monkeypatch.delenv("AIMODEL_SMOKE", raising=False)
    monkeypatch.setattr(tui_app, "HardwareMonitor", Monitor)
    monkeypatch.setattr(viewer_module, "get_provider_filter_labels", labels)
    monkeypatch.setattr(startup_module, "ensure_service_running", lambda: False)
    monkeypatch.setattr(startup_module.cache_db, "get_hardware_snapshot", lambda: None)
    monkeypatch.setattr(AIModelViewer, "request_system_info_refresh", lambda self, force=False: None)
    monkeypatch.setattr(AIModelViewer, "request_download_poll", lambda self, force=False: None)
    monkeypatch.setattr(AIModelViewer, "run_search_worker", lambda self, *args: None)

    async def run():
        app = AIModelViewer()
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(30):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            selector = app.query_one("#provider-select", Select)
            selector.value = "Hugging Face"
            await pilot.pause()
            assert app.current_filter == "Hugging Face"
            search_input = app.query_one("#search-input", Input)
            search_input.focus()
            await pilot.press("x")
            assert search_input.value == "x"
            release.set()
            for _ in range(30):
                if "LM Studio" in app.provider_filter_labels:
                    break
                await asyncio.sleep(0.01)
            assert app.provider_filter_labels == ("Ollama", "Hugging Face", "LM Studio")
            await pilot.pause(0.2)
            assert app.current_filter == "Hugging Face"
            assert selector.value == "Hugging Face"
            assert app.search_counter == 0

    try:
        asyncio.run(run())
    finally:
        release.set()
