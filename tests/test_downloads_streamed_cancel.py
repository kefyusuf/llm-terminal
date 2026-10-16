"""Cancellation of real, silent, task-owned local processes."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from downloads import runner
from downloads.store import DownloadStore


@pytest.mark.parametrize("ignore_terminate", [False, True])
def test_streamed_cancel_stops_silent_process(tmp_path, ignore_terminate):
    if ignore_terminate and sys.platform == "win32":
        pytest.skip("Windows TerminateProcess cannot be ignored")
    ready = tmp_path / "ready"
    script = (
        "import pathlib,signal,sys,time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_terminate else "")
        + "pathlib.Path(sys.argv[1]).touch()\ntime.sleep(20)\n"
    )
    store = DownloadStore(tmp_path / "jobs.db")
    job, _ = store.upsert_job({"source": "Ollama", "name": "silent-test"})
    target = job["target_id"]
    store.claim_next_queued()
    processes = []
    cleared = []
    errors = []

    class State:
        def set_process(self, _target, process):
            processes.append(process)

        def clear_process(self, target):
            cleared.append(target)

    state = State()
    state.store = store

    def run():
        try:
            runner.run_streamed_command(state, target, [
                sys._base_executable, "-u", "-c", script, str(ready),
            ])
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        store.mark_cancel_requested(target)
        worker.join(4)
        assert not worker.is_alive(), "silent output blocked cancellation"
        assert not errors
        assert processes[0].poll() is not None
        final = store.get_job_by_target(target)
        assert final["status"] == "cancelled"
        assert final["return_code"] != 0
        if ignore_terminate:
            assert final["return_code"] == -9
        assert cleared == [target]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        worker.join(5)
        assert not worker.is_alive()
