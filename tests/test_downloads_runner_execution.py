"""Deterministic execution-path tests for downloads.runner.

These tests use fake process/state objects. They must not spawn real subprocesses,
perform network requests, or write outside pytest temporary directories.
"""

from __future__ import annotations

import os
import subprocess
import sys
from io import StringIO
from types import SimpleNamespace

import pytest

from downloads import runner


class _Store:
    def __init__(self, *, command=None, cancel_values=None):
        self.command = command
        self.updates = []
        self._cancel_values = list(cancel_values or [False])
        self._cancel_index = 0

    def update_job(self, target_id, **fields):
        self.updates.append((target_id, fields))

    def get_job_by_target(self, _target_id):
        index = min(self._cancel_index, len(self._cancel_values) - 1)
        value = self._cancel_values[index]
        self._cancel_index += 1
        return {"cancel_requested": value}

    def get_command(self, _target_id):
        return self.command


class _State:
    def __init__(self, store):
        self.store = store
        self.set_calls = []
        self.clear_calls = []

    def set_process(self, target_id, process):
        self.set_calls.append((target_id, process))

    def clear_process(self, target_id):
        self.clear_calls.append(target_id)


class _HFProcess:
    def __init__(self, poll_values, *, stderr=""):
        self._poll_values = list(poll_values)
        self._poll_index = 0
        self.stderr = StringIO(stderr)
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        index = min(self._poll_index, len(self._poll_values) - 1)
        value = self._poll_values[index]
        self._poll_index += 1
        return value

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


class _StreamProcess:
    def __init__(self, lines, *, return_code=0):
        self.stdout = list(lines) if lines is not None else None
        self.return_code = return_code
        self.terminate_calls = 0

    def terminate(self):
        self.terminate_calls += 1

    def wait(self):
        return self.return_code


def _install_fake_config(monkeypatch, models_dir):
    fake_config = SimpleNamespace(
        settings=SimpleNamespace(
            hf_models_dir=models_dir,
            hf_token=None,
        )
    )
    monkeypatch.setitem(sys.modules, "config", fake_config)


def _last_update(store):
    return store.updates[-1][1]


def test_finalize_terminal_pins_completed_failed_and_cancelled_states():
    store = _Store()
    state = _State(store)

    runner._finalize_terminal(state, "completed", 0, cancelled=False)
    runner._finalize_terminal(state, "failed", 2, cancelled=False, last_line="fatal error")
    runner._finalize_terminal(state, "cancelled", -15, cancelled=True)

    assert store.updates == [
        (
            "completed",
            {"status": "completed", "detail": "Completed", "progress": "", "return_code": 0},
        ),
        (
            "failed",
            {"status": "failed", "detail": "fatal error", "progress": "", "return_code": 2},
        ),
        (
            "cancelled",
            {"status": "cancelled", "detail": "Canceled", "progress": "", "return_code": -15},
        ),
    ]


def test_hf_download_rejects_missing_repo_and_target_file(tmp_path, monkeypatch):
    _install_fake_config(monkeypatch, tmp_path / "models")

    missing_repo_store = _Store()
    missing_repo_state = _State(missing_repo_store)
    runner.run_hf_download(missing_repo_state, "hf:missing-repo", ["hf_api_download"])

    assert _last_update(missing_repo_store) == {
        "status": "failed",
        "detail": "missing Hugging Face repository id",
        "return_code": 1,
    }
    assert missing_repo_state.set_calls == []

    missing_file_store = _Store()
    missing_file_state = _State(missing_file_store)
    runner.run_hf_download(
        missing_file_state,
        "hf:missing-file",
        ["hf_api_download", "owner/repo"],
    )

    assert _last_update(missing_file_store) == {
        "status": "failed",
        "detail": "missing Hugging Face target file",
        "return_code": 1,
    }
    assert missing_file_state.set_calls == []


def test_hf_download_success_sets_running_then_completed(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    _install_fake_config(monkeypatch, models_dir)
    process = _HFProcess([None, 0])
    store = _Store(cancel_values=[False])
    state = _State(store)

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    runner.run_hf_download(
        state,
        "hf:success",
        ["hf_api_download", "owner/repo", "model.gguf"],
    )

    assert models_dir.is_dir()
    assert store.updates[0][1] == {
        "status": "running",
        "detail": "Downloading",
        "progress": "",
    }
    assert _last_update(store) == {
        "status": "completed",
        "detail": "Completed",
        "progress": "",
        "return_code": 0,
    }
    assert state.set_calls == [("hf:success", process)]
    assert state.clear_calls == ["hf:success"]


def test_hf_download_passes_persisted_revision_to_child(tmp_path, monkeypatch):
    _install_fake_config(monkeypatch, tmp_path / "models")
    process = _HFProcess([0])
    captured = []
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda args, **kw: captured.append(args) or process
    )
    runner.run_hf_download(
        _State(_Store()), "hf:pinned", ["hf_api_download", "owner/repo", "model.gguf", "a" * 40]
    )
    assert captured[0][6] == "a" * 40


def test_hf_worker_rejects_unsafe_legacy_filename_before_mkdir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    _install_fake_config(monkeypatch, models_dir)
    spawned = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: spawned.append(a) or _HFProcess([0]))
    store = _Store()
    runner.run_hf_download(_State(store), "hf:unsafe", ["hf_api_download", "owner/repo", "../outside.gguf"])
    assert _last_update(store)["status"] == "failed"
    assert "target file" in _last_update(store)["detail"]
    assert not spawned
    assert not models_dir.exists()


def _directory_link(link, target):
    if os.name == "nt":
        env = os.environ.copy()
        env.update(AIMODEL_TEST_LINK=str(link), AIMODEL_TEST_TARGET=str(target))
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:AIMODEL_TEST_LINK -Target $env:AIMODEL_TEST_TARGET | Out-Null"],
            env=env, check=True, capture_output=True, timeout=30,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("inside", [False, True])
def test_hf_worker_checks_real_destination_through_directory_links(tmp_path, monkeypatch, inside):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target = models_dir / "contained" if inside else tmp_path / "outside"
    target.mkdir()
    _directory_link(models_dir / "nested", target)
    _install_fake_config(monkeypatch, models_dir)
    spawned = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: spawned.append(a) or _HFProcess([0]))
    store = _Store()
    runner.run_hf_download(_State(store), "hf:linked", ["hf_api_download", "owner/repo", "nested/model.gguf"])
    assert _last_update(store)["status"] == ("completed" if inside else "failed")
    assert bool(spawned) is inside


def test_hf_worker_reports_unusable_destination_without_spawning(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.write_text("not a directory", encoding="utf-8")
    _install_fake_config(monkeypatch, models_dir)
    spawned = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: spawned.append(a) or _HFProcess([0]))
    store = _Store()
    runner.run_hf_download(_State(store), "hf:blocked", ["hf_api_download", "owner/repo", "model.gguf"])
    assert _last_update(store)["status"] == "failed"
    assert not spawned


def test_hf_download_failure_uses_last_stderr_line(tmp_path, monkeypatch):
    _install_fake_config(monkeypatch, tmp_path / "models")
    process = _HFProcess([2], stderr="first line\nremote failed hard\n")
    store = _Store(cancel_values=[False])
    state = _State(store)

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)

    runner.run_hf_download(
        state,
        "hf:failed",
        ["hf_api_download", "owner/repo", "model.gguf"],
    )

    assert _last_update(store) == {
        "status": "failed",
        "detail": "remote failed hard",
        "progress": "",
        "return_code": 2,
    }
    assert state.clear_calls == ["hf:failed"]


def test_hf_download_cancel_escalates_from_terminate_to_kill(tmp_path, monkeypatch):
    _install_fake_config(monkeypatch, tmp_path / "models")
    process = _HFProcess([None, None, -9])
    store = _Store(cancel_values=[True])
    state = _State(store)
    monotonic_values = iter([10.0, 12.0])

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))

    runner.run_hf_download(
        state,
        "hf:cancel",
        ["hf_api_download", "owner/repo", "model.gguf"],
    )

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert _last_update(store) == {
        "status": "cancelled",
        "detail": "Canceled",
        "progress": "",
        "return_code": -9,
    }
    assert state.clear_calls == ["hf:cancel"]


def test_streamed_command_reports_progress_heartbeat_and_completion(monkeypatch):
    process = _StreamProcess(["downloading 42%\n", "waiting for data\n"], return_code=0)
    store = _Store(cancel_values=[False])
    state = _State(store)
    monotonic_values = iter([0.0, 0.1, 1.2])

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, "_service_popen_kwargs", lambda: {})
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))

    runner.run_streamed_command(state, "ollama:progress", ["ollama", "pull", "model"])

    assert store.updates[0][1] == {
        "status": "running",
        "detail": "Downloading",
        "progress": "42%",
    }
    assert store.updates[1][1] == {
        "status": "running",
        "detail": "Downloading",
        "progress": "1s",
    }
    assert _last_update(store) == {
        "status": "completed",
        "detail": "Completed",
        "progress": "",
        "return_code": 0,
    }
    assert state.clear_calls == ["ollama:progress"]


def test_streamed_command_cancel_terminates_and_finalizes_cancelled(monkeypatch):
    process = _StreamProcess(["cancel requested\n"], return_code=-15)
    store = _Store(cancel_values=[True])
    state = _State(store)
    monotonic_values = iter([0.0, 0.1])

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, "_service_popen_kwargs", lambda: {})
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))

    runner.run_streamed_command(state, "ollama:cancel", ["ollama", "pull", "model"])

    assert process.terminate_calls == 1
    assert _last_update(store) == {
        "status": "cancelled",
        "detail": "Canceled",
        "progress": "",
        "return_code": -15,
    }
    assert state.clear_calls == ["ollama:cancel"]


def test_streamed_command_failure_preserves_last_output_line(monkeypatch):
    process = _StreamProcess(["fatal registry failure\n"], return_code=3)
    store = _Store(cancel_values=[False])
    state = _State(store)
    monotonic_values = iter([0.0, 0.1])

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, "_service_popen_kwargs", lambda: {})
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))

    runner.run_streamed_command(state, "ollama:failed", ["ollama", "pull", "model"])

    assert _last_update(store) == {
        "status": "failed",
        "detail": "fatal registry failure",
        "progress": "",
        "return_code": 3,
    }


def test_process_job_dispatches_hf_and_streamed_commands(monkeypatch):
    hf_store = _Store(command=["hf_api_download", "owner/repo", "model.gguf"])
    hf_state = _State(hf_store)
    hf_calls = []
    streamed_calls = []

    monkeypatch.setattr(
        runner,
        "run_hf_download",
        lambda state, target_id, command: hf_calls.append((state, target_id, command)),
    )
    monkeypatch.setattr(
        runner,
        "run_streamed_command",
        lambda state, target_id, command: streamed_calls.append((state, target_id, command)),
    )

    runner.process_job(hf_state, "hf:dispatch")

    assert hf_calls == [
        (hf_state, "hf:dispatch", ["hf_api_download", "owner/repo", "model.gguf"])
    ]
    assert streamed_calls == []
    assert hf_state.clear_calls == ["hf:dispatch"]

    stream_store = _Store(command=["ollama", "pull", "model"])
    stream_state = _State(stream_store)
    runner.process_job(stream_state, "ollama:dispatch")

    assert streamed_calls == [
        (stream_state, "ollama:dispatch", ["ollama", "pull", "model"])
    ]
    assert stream_state.clear_calls == ["ollama:dispatch"]


def test_process_job_contains_missing_command_and_os_errors(monkeypatch):
    missing_store = _Store(command=None)
    missing_state = _State(missing_store)
    runner.process_job(missing_state, "missing")
    assert _last_update(missing_store) == {
        "status": "failed",
        "detail": "missing command",
        "return_code": 1,
    }

    for exc, expected_detail, expected_code in [
        (FileNotFoundError(), "required command not found", 127),
        (OSError("spawn denied"), "spawn denied", 1),
    ]:
        store = _Store(command=["ollama", "pull", "model"])
        state = _State(store)

        def _raise(*_args, _exc=exc, **_kwargs):
            raise _exc

        monkeypatch.setattr(runner, "run_streamed_command", _raise)
        runner.process_job(state, "error")

        assert _last_update(store) == {
            "status": "failed",
            "detail": expected_detail,
            "return_code": expected_code,
        }
        assert state.clear_calls == ["error"]
