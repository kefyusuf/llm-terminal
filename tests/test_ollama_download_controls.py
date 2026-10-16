"""Ollama controls through the runtime UI, real HTTP/store and a controlled CLI."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler

import pytest
from textual.widgets import DataTable

import config
from app.viewer import AIModelViewer
from core.http_server import LocalThreadingHTTPServer
from downloads.api import _make_handler
from downloads.download_service import DownloadServiceState
from downloads.runner import process_job
from downloads.store import DownloadStore


class ControlsViewer(AIModelViewer):
    def on_mount(self, event):
        event.prevent_default()
        self._configure_results_table_columns(force=True)
        self.query_one('#download-history-table', DataTable).add_columns(
            'Source', 'Publisher', 'Model', 'Status', 'Detail', 'Action')
        self._apply_ui_mode()
        self.set_interval(.05, self.request_download_poll)


@contextmanager
def controlled_cli_service(monkeypatch, tmp_path):
    script = tmp_path / 'pull.py'
    stage = tmp_path / 'advance'
    script.write_text(
        "import pathlib,sys,time\n"
        "print('pulling abcdef123456 17% 170 MB/1.0 GB 10 MB/s 1m23s',flush=True)\n"
        "while not pathlib.Path(sys.argv[1]).exists(): time.sleep(.01)\n"
        "print('pulling abcdef123456 42% 420 MB/1.0 GB 10 MB/s 58s',flush=True)\n"
        "time.sleep(30)\n", encoding='utf-8')
    monkeypatch.setattr('downloads.store.build_download_command',
                        lambda model: [sys.executable, '-u', str(script), str(stage)])
    native_cli = os.environ.get('AIMODEL_TEST_OLLAMA_BINARY')
    registry_stop = threading.Event()
    disconnected = threading.Event()

    class RegistryHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps({'version': '0.35.0'}).encode() if self.path == '/api/version' else b'Ollama is running'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            assert self.path == '/api/pull'
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert (request.get('model') or request.get('name')) == 'fixture:tiny'
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                self.wfile.write(b'{"status":"pulling manifest"}\n')
                self.wfile.flush()
                while not registry_stop.is_set():
                    progress = 42 if stage.exists() else 17
                    payload = {'status':'pulling '+('a'*12), 'digest':'sha256:'+('a'*64),
                               'total':1000000, 'completed':progress*10000}
                    self.wfile.write(json.dumps(payload).encode()+b'\n')
                    self.wfile.flush()
                    registry_stop.wait(.05)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                disconnected.set()

    registry = None
    registry_thread = None
    if native_cli:
        registry = LocalThreadingHTTPServer(('127.0.0.1', 0), RegistryHandler)
        registry_thread = threading.Thread(target=registry.serve_forever, daemon=True)
        registry_thread.start()
        monkeypatch.setenv('OLLAMA_HOST', f'http://127.0.0.1:{registry.server_port}')
        monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
        monkeypatch.setattr('downloads.store.build_download_command',
                            lambda model: [native_cli, 'pull', 'fixture:tiny'])
    state = DownloadServiceState()
    state.store = DownloadStore(tmp_path / 'jobs.db')
    server = LocalThreadingHTTPServer(('127.0.0.1', 0), _make_handler(state, 'controls-test'))
    state.server = server
    monkeypatch.setattr(config.settings, 'download_service_host', '127.0.0.1')
    monkeypatch.setattr(config.settings, 'download_service_port', server.server_port)
    monkeypatch.setattr(config.settings, 'download_service_token', 'controls-test')
    monkeypatch.setattr(config.settings, 'download_db_path', tmp_path / 'jobs.db')

    def work():
        while not state.stop_event.is_set():
            job = state.store.claim_next_queued()
            if job:
                process_job(state, job['target_id'])
            else:
                state.stop_event.wait(.01)

    worker = threading.Thread(target=work, daemon=True)
    http = threading.Thread(target=server.serve_forever, daemon=True)
    state.worker_thread = worker
    http.start()
    worker.start()
    completed = False
    try:
        yield state, stage
        completed = True
    finally:
        state.request_worker_stop()
        for target in state.snapshot_active_targets():
            child = state.get_process(target)
            if child is not None and child.poll() is None:
                child.kill()
        worker.join(timeout=5)
        server.shutdown()
        server.server_close()
        http.join(timeout=5)
        if registry is not None:
            if completed:
                disconnected.wait(timeout=2)
            registry_stop.set()
            registry.shutdown()
            registry.server_close()
            registry_thread.join(timeout=5)
            assert not registry_thread.is_alive()
            if completed:
                assert disconnected.is_set(), 'Cancel must disconnect the native CLI pull stream'
        assert not worker.is_alive()
        assert not http.is_alive()


async def wait_until(pilot, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'Visible download controls did not reach expected state'
        await pilot.pause(.03)


@pytest.mark.parametrize('width', [80, 120])
def test_ollama_start_live_percentage_and_cancel_through_runtime_ui(monkeypatch, tmp_path, width):
    from textual.widgets import ProgressBar

    monkeypatch.setattr('app.viewer.get_provider_filter_labels', lambda: ('Ollama',))
    with controlled_cli_service(monkeypatch, tmp_path) as (state, advance):
        async def run():
            app = ControlsViewer()
            model = {'source': 'Ollama', 'id': 'fixture:tiny', 'name': 'fixture:tiny',
                     'publisher': 'Fixture', 'score': '-', 'fit': 'Not assessed', 'size': '1 MB',
                     'download_state': 'idle'}
            async with app.run_test(size=(width, 32)) as pilot:
                app.current_filter = 'Ollama'
                app.all_results = [model]
                app.refresh_table()
                app.open_model_detail_modal(model)
                await pilot.pause()
                await pilot.click('#download-btn')
                table = app.query_one('#results-table', DataTable)
                key = 'Ollama:fixture:tiny'
                await wait_until(pilot, lambda: (state.store.get_job_by_target('ollama:fixture:tiny') or {}).get('progress') == '17%')
                await wait_until(pilot, lambda: '17%' in str(table.get_cell(key, 'download')))
                advance.touch()
                await wait_until(pilot, lambda: '42%' in str(table.get_cell(key, 'download')))
                panel = app.query_one('#active-download-progress')
                assert panel.query_one(ProgressBar).progress == 42
                assert panel.is_on_screen
                assert table.virtual_size.width <= table.scrollable_content_region.width

                table.focus()
                table.move_cursor(row=0, column=0)
                await pilot.press('enter')
                await pilot.pause()
                assert len(app.screen_stack) == 2, 'Selecting one download must open exactly one modal'
                live = app.screen.query_one('#model-download-progress')
                assert live.query_one(ProgressBar).progress == 42
                assert app.screen.query_one('#cancel-download-btn').is_on_screen
                await pilot.click('#cancel-download-btn')
                await wait_until(pilot, lambda: state.store.get_job_by_target('ollama:fixture:tiny')['status'] == 'cancelled')
                await wait_until(pilot, lambda: 'Canceled' in str(table.get_cell(key, 'download')))
                job = state.store.get_job_by_target('ollama:fixture:tiny')
                assert job['cancel_requested'] == 1
                assert not app.dl.active_downloads
                assert state.get_process('ollama:fixture:tiny') is None

        asyncio.run(run())


def test_late_download_snapshot_during_widget_removal_is_ignored(monkeypatch, tmp_path):
    """A worker may finish after the results table starts unmounting."""
    monkeypatch.delenv('AIMODEL_TEST_OLLAMA_BINARY', raising=False)
    with controlled_cli_service(monkeypatch, tmp_path):
        async def run():
            app = ControlsViewer()
            async with app.run_test(size=(80, 24)):
                await app.query_one('#results-table', DataTable).remove()
                app._apply_download_poll_snapshot([], None, None)

        asyncio.run(run())


@pytest.mark.parametrize('width', [80, 120])
@pytest.mark.parametrize('cancel_from', ['modal', 'panel'])
def test_live_panel_retains_metrics_and_updates_while_model_details_open(monkeypatch, tmp_path, width, cancel_from):
    from textual.geometry import Region
    from textual.widgets import ProgressBar, Static

    monkeypatch.delenv('AIMODEL_TEST_OLLAMA_BINARY', raising=False)
    with controlled_cli_service(monkeypatch, tmp_path) as (state, advance):
        async def run():
            app = ControlsViewer()
            model = {'source': 'Ollama', 'id': 'fixture:tiny', 'name': 'fixture:tiny',
                     'download_state': 'idle'}
            async with app.run_test(size=(width, 24)) as pilot:
                app.current_filter = 'Ollama'
                app.all_results = [model]
                app.refresh_table()
                app.open_model_detail_modal(model)
                await pilot.pause()
                await pilot.click('#download-btn')
                panel = app.query_one('#active-download-progress')
                summary = panel.query_one(Static)
                await wait_until(pilot, lambda: '170 MB' in str(summary.renderable))
                assert '1.0 GB' in str(summary.renderable)
                assert '10 MB/s' in str(summary.renderable)
                assert '1m23s' in str(summary.renderable)
                assert panel.query_one(ProgressBar).progress == 17
                assert panel.is_on_screen
                assert panel.query_one('#progress-cancel-btn').is_on_screen

                app.open_model_detail_modal(model)
                await pilot.pause()
                modal = app.screen
                live = modal.query_one('#model-download-progress')
                advance.touch()
                await wait_until(pilot, lambda: '420 MB' in str(live.query_one(Static).renderable))
                assert '58s' in str(live.query_one(Static).renderable)
                assert live.query_one(ProgressBar).progress == 42
                assert live.is_on_screen
                rendered = live.query_one(Static)
                visible = '\n'.join(strip.text for strip in rendered.render_lines(
                    Region(0, 0, rendered.content_size.width, rendered.content_size.height)))
                assert all(value in visible for value in ['42%', '420 MB', '10 MB/s', '58s'])
                if cancel_from == 'panel':
                    await pilot.click('#close-btn')
                    await wait_until(pilot, lambda: len(app.screen_stack) == 1)
                    await pilot.click(panel.query_one('#progress-cancel-btn'))
                else:
                    await pilot.click('#cancel-download-btn')
                await wait_until(pilot, lambda: state.store.get_job_by_target('ollama:fixture:tiny')['status'] == 'cancelled')
                await wait_until(pilot, lambda: 'Canceled' in str(summary.renderable))
                assert panel.query_one('#progress-cancel-btn').disabled

        asyncio.run(run())
