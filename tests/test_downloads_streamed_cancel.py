from __future__ import annotations

import threading

from downloads import runner


class _Store:
    def __init__(self):
        self.updates = []

    def update_job(self, target_id, **fields):
        self.updates.append((target_id, fields))

    def get_job_by_target(self, _target_id):
        return {"cancel_requested": True}


class _State:
    def __init__(self, store):
        self.store = store
        self.set_calls = []
        self.clear_calls = []

    def set_process(self, target_id, process):
        self.set_calls.append((target_id, process))

    def clear_process(self, target_id):
        self.clear_calls.append(target_id)


class _SilentProcess:
    def __init__(self):
        self.stdout = self
        self._killed = threading.Event()
        self.terminate_calls = 0
        self.kill_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._killed.wait(timeout=0.5):
            raise StopIteration
        raise AssertionError("streamed runner did not cancel a silent subprocess")

    def poll(self):
        return -9 if self._killed.is_set() else None

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1
        self._killed.set()

    def wait(self):
        return -9 if self._killed.is_set() else 0


def test_streamed_cancel_checks_silent_process_and_escalates_to_kill(monkeypatch):
    process = _SilentProcess()
    store = _Store()
    state = _State(store)
    clock = {"value": 0.0}

    def _monotonic():
        value = clock["value"]
        clock["value"] += 1.0
        return value

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, "_service_popen_kwargs", lambda: {})
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runner.time, "monotonic", _monotonic)

    runner.run_streamed_command(
        state,
        "ollama:silent-cancel",
        ["ollama", "pull", "model"],
    )

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert store.updates[-1][1] == {
        "status": "cancelled",
        "detail": "Canceled",
        "progress": "",
        "return_code": -9,
    }
    assert state.clear_calls == ["ollama:silent-cancel"]
