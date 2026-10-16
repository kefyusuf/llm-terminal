"""Local process/SQLite acceptance; no provider, model, or network calls."""

from __future__ import annotations

import sys
import threading
import time

import psutil
import pytest

from downloads import runner
from downloads.store import DownloadStore


class _State:
    def __init__(self, store):
        self.store = store
        self.process = None
        self.started_process = None
        self.launched = threading.Event()

    def set_process(self, _target, process):
        self.process = process
        self.started_process = process
        self.launched.set()

    def clear_process(self, _target):
        self.process = None


def _wait_until(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    assert condition(), "condition did not become true within the bound"


@pytest.mark.parametrize("cancel", [True, False])
def test_root_exit_finishes_while_descendant_keeps_stdout_open(tmp_path, cancel):
    release = tmp_path / "release"
    child_pid = tmp_path / "child.pid"
    script = (
        "import os, pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
        "deadline=time.monotonic()+20; "
        "exec(\"while not p.exists() and time.monotonic()<deadline: time.sleep(.02)\")', "
        "sys.argv[1]])\n"
        "path = pathlib.Path(sys.argv[2]); temporary = path.with_suffix('.tmp')\n"
        "temporary.write_text(str(child.pid)); temporary.replace(path)\n"
        "print('pulling 17%', flush=True)\n"
        "while not pathlib.Path(sys.argv[3]).exists(): time.sleep(.02)\n"
        "sys.exit(7)\n"
    )
    store = DownloadStore(tmp_path / "jobs.db")
    job, _ = store.upsert_job({"source": "Ollama", "name": "local-test"})
    target = job["target_id"]
    store.claim_next_queued()
    state = _State(store)
    errors = []

    def run():
        try:
            runner.run_streamed_command(state, target, [
                sys._base_executable, "-u", "-c", script, str(release), str(child_pid),
                str(tmp_path / "root-exit"),
            ])
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    process = descendant = None
    try:
        assert state.launched.wait(5)
        process = state.process
        _wait_until(lambda: child_pid.exists())
        descendant = psutil.Process(int(child_pid.read_text()))
        _wait_until(lambda: store.get_job_by_target(target)["progress"] == "17%")
        assert process.poll() is None
        if cancel:
            # The real HTTP handler persists cancellation, then terminates the root.
            store.mark_cancel_requested(target)
            process.terminate()
        else:
            (tmp_path / "root-exit").touch()
        worker.join(3)
        assert not worker.is_alive(), "cancel waited for inherited stdout EOF"
        assert not errors
        assert descendant.is_running(), "test must retain the inherited writer"
        final = store.get_job_by_target(target)
        assert final["status"] == ("cancelled" if cancel else "failed")
        assert final["cancel_requested"] == int(cancel)
        assert final["return_code"] != 0
        if not cancel:
            assert final["return_code"] == 7
        assert state.process is None
    finally:
        release.touch()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        if descendant is not None:
            try:
                descendant.wait(timeout=5)
            except psutil.TimeoutExpired:
                descendant.kill()
                descendant.wait(timeout=5)
        worker.join(5)
        assert not worker.is_alive(), "test-owned worker cleanup failed"


@pytest.mark.parametrize("separator", ["\r", ""])
def test_progress_parses_cr_split_utf8_and_final_unterminated_error(tmp_path, separator):
    advance = tmp_path / "advance"
    finish = tmp_path / "finish"
    script = (
        "import os,pathlib,sys,time\n"
        "def emit(value): os.write(sys.stdout.fileno(), value)\n"
        "def wait(path):\n"
        " deadline=time.monotonic()+15\n"
        " while not pathlib.Path(path).exists() and time.monotonic()<deadline: time.sleep(.02)\n"
        "emit(b'pulling \\xe2')\n"
        "time.sleep(.15)\n"
        "emit(b'\\x86\\x93 1')\n"
        "time.sleep(.15)\n"
        f"emit(b'7%' + {separator.encode()!r})\n"
        "wait(sys.argv[1])\n"
        f"emit(b'pulling 42%' + {separator.encode()!r})\n"
        "wait(sys.argv[2])\n"
        "emit('\\nregistry failed: café'.encode())\n"
        "sys.exit(3)\n"
    )
    store = DownloadStore(tmp_path / "jobs.db")
    job, _ = store.upsert_job({"source": "Ollama", "name": "progress-test"})
    target = job["target_id"]
    store.claim_next_queued()
    state = _State(store)
    errors = []

    def run():
        try:
            runner.run_streamed_command(state, target, [
                sys._base_executable, "-u", "-c", script, str(advance), str(finish),
            ])
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    process = None
    try:
        assert state.launched.wait(5)
        process = state.process
        _wait_until(lambda: store.get_job_by_target(target)["progress"] == "17%")
        assert process.poll() is None
        advance.touch()
        _wait_until(lambda: store.get_job_by_target(target)["progress"] == "42%")
        assert process.poll() is None
        finish.touch()
        worker.join(3)
        assert not worker.is_alive()
        assert not errors
        final = store.get_job_by_target(target)
        assert final["status"] == "failed"
        assert final["return_code"] == 3
        assert final["cancel_requested"] == 0
        assert final["detail"] == "registry failed: café"
    finally:
        advance.touch()
        finish.touch()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        worker.join(5)
        assert not worker.is_alive()


def test_pipe_read_error_stops_owned_process(tmp_path, monkeypatch):
    store = DownloadStore(tmp_path / "jobs.db")
    job, _ = store.upsert_job({"source": "Ollama", "name": "read-error-test"})
    target = job["target_id"]
    store.claim_next_queued()
    state = _State(store)
    actual_read = runner._read_stream_chunk

    def read_error(stream):
        chunk = actual_read(stream)
        if chunk:
            raise OSError("controlled pipe failure")
        return chunk

    monkeypatch.setattr(runner, "_read_stream_chunk", read_error)
    try:
        with pytest.raises(OSError, match="controlled pipe failure"):
            runner.run_streamed_command(state, target, [
                sys._base_executable, "-u", "-c",
                "import time; print('ready', flush=True); time.sleep(20)",
            ])
        assert state.started_process.poll() is not None, "pipe failure orphaned the owned CLI"
        assert state.process is None
    finally:
        process = state.started_process
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
