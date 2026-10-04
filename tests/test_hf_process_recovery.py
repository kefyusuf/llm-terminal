"""Real local child-process regressions; no model acquisition or network."""

import subprocess
import threading
from types import SimpleNamespace

from downloads import runner
from downloads.store import DownloadStore


class ProcessState:
    def __init__(self, store):
        self.store = store
        self.process = None

    def set_process(self, target, process):
        self.process = process

    def clear_process(self, target):
        self.process = None


def test_hf_child_stderr_cannot_block_completion(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config.settings, "hf_models_dir", tmp_path / "models")
    monkeypatch.setattr(config.settings, "hf_token", None)
    launches = []
    actual_popen = subprocess.Popen

    def capture(command, **kwargs):
        launches.append((command, kwargs))
        return actual_popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", capture)
    # Exceeds the pipe buffer before exit; no SDK or model weights involved.
    monkeypatch.setattr(runner, "_hf_download_script", lambda: (
        "import sys; sys.stderr.write('x' * 262144 + '\\nlast diagnostic\\n'); "
        "sys.stderr.flush(); sys.exit(2)"
    ))
    store = DownloadStore(tmp_path / "jobs.db")
    job, _ = store.upsert_job({"source": "Hugging Face", "id": "owner/repo",
                               "target_file": "model.gguf"})
    state = ProcessState(store)
    target = job["target_id"]
    thread = threading.Thread(target=runner.process_job, args=(state, target), daemon=True)
    thread.start()
    thread.join(5)
    finished = not thread.is_alive()
    if not finished:
        store.mark_cancel_requested(target)
        if state.process is not None:
            state.process.kill()
        thread.join(5)
    assert finished, "child stalled because stderr was not consumed"
    final = store.get_job_by_target(target)
    assert final["status"] == "failed"
    assert final["detail"] == "last diagnostic"
    assert final["return_code"] == 2
    assert launches[0][0][-1] == "parent-stdin"
    assert launches[0][1]["stdin"] == subprocess.PIPE


def test_hf_stderr_tail_is_bounded():
    from io import StringIO

    output = SimpleNamespace(stderr=StringIO("x" * 1024 * 1024 + "\nlast diagnostic\n"))
    tail = runner._read_stderr_tail(output)
    assert len(tail) <= 4096
    assert tail.splitlines()[-1] == "last diagnostic"


def test_spawn_failure_becomes_terminal_job(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("unavailable")

    import config

    monkeypatch.setattr(config.settings, "hf_models_dir", tmp_path / "models")
    monkeypatch.setattr(subprocess, "Popen", fail)
    store = DownloadStore(tmp_path / "jobs.db")
    job, _ = store.upsert_job({"source": "Hugging Face", "id": "owner/repo",
                               "target_file": "model.gguf"})
    state = ProcessState(store)
    runner.process_job(state, job["target_id"])
    assert store.get_job_by_target(job["target_id"])["status"] == "failed"
    assert state.process is None
