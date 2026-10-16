"""Participant search can show scored rows before the last detail completes."""
from __future__ import annotations

import asyncio
import threading
import time
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from textual.widgets import DataTable

from app.viewer import AIModelViewer


class _Monitor:
    cpu_name = "Test CPU"
    cpu_cores = 8
    gpu_name = "No GPU"
    nvidia_available = False

    def get_specs(self):
        return {"cpu_name": self.cpu_name, "cpu_cores": 8, "gpu_name": "No GPU",
                "has_gpu": False, "ram_total": 16, "ram_free": 12,
                "vram_total": 0, "vram_free": 0}


class _Viewer(AIModelViewer):
    async def on_mount(self, event):
        event.prevent_default()
        self._apply_ui_mode()
        self._configure_results_table_columns(force=True)

    def request_system_info_refresh(self, force=False):
        pass

    def request_download_poll(self, force=False):
        pass


async def _wait_for_ui(pilot, ready, description, *, timeout=2):
    deadline = time.monotonic() + timeout
    while not ready():
        assert time.monotonic() < deadline, f"Timed out waiting for {description}"
        await pilot.pause(.02)


@pytest.mark.parametrize("fast_delay", [0, 0.5])
@pytest.mark.parametrize("completion_delay", [0, 2.2])
def test_search_renders_a_scored_prefix_without_caching_or_completing(monkeypatch, tmp_path, fast_delay, completion_delay):
    from core import cache_db

    monkeypatch.setattr(cache_db, "_cache_db_path", tmp_path / "cache.db")
    cache_db.init_db()
    monkeypatch.setattr("tui_app.HardwareMonitor", _Monitor)
    monkeypatch.setattr("app.viewer.get_provider_filter_labels", lambda: ["Ollama"])
    release = threading.Event()
    slow_entered = threading.Event()

    def fake_get(client, url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        path = urlsplit(url).path
        if path == "/api/tags":
            response._content = b'{"models": []}'
        elif path == "/search":
            response._content = (
                b'<a href="/library/z-prefix-fast">fast 1M Pulls</a>'
                b'<a href="/library/a-prefix-slow">slow 1M Pulls</a>'
            )
        else:
            name = path.rsplit("/", 1)[-1]
            if name == "a-prefix-slow":
                slow_entered.set()
                assert release.wait(5), "test detail was not released"
                time.sleep(completion_delay)
            elif name == "z-prefix-fast":
                assert slow_entered.wait(2), "slow detail did not start"
                time.sleep(fast_delay)
            response._content = (
                '<table><tr><th>Name</th><th>Size</th></tr>'
                f'<tr><td>{name}:latest</td><td>3 GB</td></tr></table>'
            ).encode()
        return response

    monkeypatch.setattr(requests.Session, "get", fake_get)

    async def run():
        app = _Viewer()
        async with app.run_test(size=(110, 35)) as pilot:
            app.sort_mode = "name"
            app.start_search("preview")
            try:
                assert await asyncio.to_thread(slow_entered.wait, 2)
                table = app.query_one("#results-table", DataTable)
                await _wait_for_ui(
                    pilot, lambda: not table.loading and table.row_count == 1,
                    "the scored prefix to render",
                )
                assert [r["name"] for r in app.all_results] == ["z-prefix-fast"]
                assert app.all_results[0]["size_source"] == "exact"
                assert app.search_cache.get_stale("ollama:preview") is None
                assert app._search_inflight_signature is not None
                assert table.loading is False and table.row_count == 1
                selected = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            finally:
                release.set()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)
            await _wait_for_ui(
                pilot, lambda: app._search_inflight_signature is None,
                "the search to complete after releasing the slow detail",
            )
            assert [r["name"] for r in app.all_results] == ["z-prefix-fast", "a-prefix-slow"]
            assert table.coordinate_to_cell_key(table.cursor_coordinate).row_key == selected
            assert len(app.search_cache.get_stale("ollama:preview")["results"]) == 2
            assert app._search_inflight_signature is None
            app.on_search_results_ready(app.active_search_id, [{"name": "late-preview"}])
            assert [r["name"] for r in app.all_results] == ["z-prefix-fast", "a-prefix-slow"]

    asyncio.run(run())


@pytest.mark.parametrize("new_delay", [0, 1.5])
def test_new_search_ignores_late_rows_from_cancelled_search(monkeypatch, tmp_path, new_delay):
    from core import cache_db

    monkeypatch.setattr(cache_db, "_cache_db_path", tmp_path / "cache.db")
    cache_db.init_db()
    monkeypatch.setattr("tui_app.HardwareMonitor", _Monitor)
    monkeypatch.setattr("app.viewer.get_provider_filter_labels", lambda: ["Ollama"])
    release = threading.Event()
    entered = threading.Event()
    returned = threading.Event()

    def fake_get(client, url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        parsed = urlsplit(url)
        if parsed.path == "/api/tags":
            response._content = b'{"models": []}'
        elif parsed.path == "/search":
            name = "old-blocked" if parse_qs(parsed.query)["q"] == ["old"] else "new-ready"
            response._content = f'<a href="/library/{name}">{name} 1M Pulls</a>'.encode()
        else:
            name = parsed.path.rsplit("/", 1)[-1]
            if name == "old-blocked":
                entered.set()
                assert release.wait(10)
                returned.set()
            else:
                time.sleep(new_delay)
            response._content = (
                '<table><tr><th>Name</th><th>Size</th></tr>'
                f'<tr><td>{name}:latest</td><td>3 GB</td></tr></table>'
            ).encode()
        return response

    monkeypatch.setattr(requests.Session, "get", fake_get)

    async def run():
        app = _Viewer()
        async with app.run_test(size=(110, 35)) as pilot:
            app.start_search("old")
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                old_id = app.active_search_id
                app.start_search("new")
                await _wait_for_ui(
                    pilot,
                    lambda: app.active_search_id != old_id
                    and app._search_inflight_signature is None,
                    "the replacement search to complete",
                    timeout=5,
                )
                assert [r["name"] for r in app.all_results] == ["new-ready"]
                app.on_search_results_ready(old_id, [{"name": "old-late"}])
                release.set()
                assert await asyncio.to_thread(returned.wait, 1)
                await pilot.pause(.1)
                assert [r["name"] for r in app.all_results] == ["new-ready"]
                assert app.search_cache.get_stale("ollama:old") is None
                assert app.search_cache.get_stale("ollama:new")["results"][0]["name"] == "new-ready"
            finally:
                release.set()

    asyncio.run(run())
