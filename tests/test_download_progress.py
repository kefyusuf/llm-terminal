"""Provider telemetry survives redraws without inventing overall progress."""

from io import BytesIO
from types import SimpleNamespace

import pytest

from downloads import runner


def _run(monkeypatch, chunks, *, command=None, target="ollama:test"):
    remaining = list(chunks)
    updates = []
    clock = iter(range(1000))
    process = SimpleNamespace(
        stdout=BytesIO(), poll=lambda: None if remaining else 0,
    )
    store = SimpleNamespace(
        get_job_by_target=lambda _: {"cancel_requested": False},
        update_job=lambda _, **fields: updates.append(fields),
    )
    state = SimpleNamespace(
        store=store, set_process=lambda *_: None, clear_process=lambda *_: None,
    )
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(runner, "_service_popen_kwargs", lambda: {})
    monkeypatch.setattr(runner, "_read_stream_chunk", lambda _: remaining.pop(0) if remaining else b"")
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    runner.run_streamed_command(state, target, command or ["ollama", "pull", "test"])
    return [update for update in updates if update["status"] == "running"]


def test_ollama_redraw_preserves_provider_metrics_and_heartbeat(monkeypatch):
    updates = _run(monkeypatch, [
        b"\x1b[2K\rpulling abcd 42% 420 MB/1.0 GB 12 MB/s 48s\x1b[?25h\r",
        b"waiting for server\n",
    ])
    assert updates[-1]["detail"] == "pulling abcd 42% 420 MB/1.0 GB 12 MB/s 48s"
    assert updates[-1]["progress"] == "42%"


@pytest.mark.parametrize("phase", [
    "pulling efgh", "pulling manifest", "verifying sha256 digest", "writing manifest",
])
def test_new_layer_or_phase_clears_previous_transfer_metrics(monkeypatch, phase):
    updates = _run(monkeypatch, [
        b"pulling abcd 100% 1.0 GB/1.0 GB 12 MB/s 0s\r",
        (phase + "\r").encode(),
    ])
    assert updates[-1]["detail"] == phase
    assert updates[-1]["progress"] == ""


def test_identical_provider_redraw_does_not_repeat_store_write(monkeypatch):
    line = b"pulling abcd 42% 420 MB/1.0 GB 12 MB/s 48s\r"
    updates = _run(monkeypatch, [line, line, line])
    assert len(updates) == 1


def test_ollama_partial_metrics_are_replaced_when_full_chunk_arrives(monkeypatch):
    updates = _run(monkeypatch, [
        b"pulling abcd 42% 420 MB/1.0 GB", b" 12 MB/s 48s\r",
    ])
    assert updates[-1]["detail"].endswith("420 MB/1.0 GB 12 MB/s 48s")


def test_generic_percentage_behavior_remains_compatible(monkeypatch):
    updates = _run(
        monkeypatch, [b"working 42%\n"], command=["other", "download"], target="generic:test",
    )
    assert updates == [{"status": "running", "detail": "Downloading", "progress": "42%"}]


def test_canonical_ollama_target_retains_metrics_with_python_command(monkeypatch):
    updates = _run(
        monkeypatch, [b"pulling abcd 42% 420 MB/1.0 GB 12 MB/s 48s\r"],
        command=["python", "-c", "controlled fixture"],
    )
    assert updates == [{
        "status": "running", "detail": "pulling abcd 42% 420 MB/1.0 GB 12 MB/s 48s",
        "progress": "42%",
    }]


def test_transfer_detail_strips_terminal_controls_and_bounds_output():
    from downloads.progress import ollama_transfer_detail

    assert ollama_transfer_detail("\x1b]0;unsafe title\x07pulling abcd\x1b[2K\r") == "pulling abcd"
    assert len(ollama_transfer_detail("pulling abcd " + "x" * 9000)) <= 512
    assert ollama_transfer_detail("waiting for server") is None
