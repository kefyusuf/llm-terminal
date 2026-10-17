"""Behavior of the application-owned download queue without network access."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from downloads.manager import DownloadManager

MODEL = {"source": "Ollama", "name": "test:latest"}


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "download state did not settle"
        time.sleep(0.01)


def test_submit_progress_and_completion(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def backend(job, command, cancel, report):
        report("Downloading", "50% 2 MB/s ETA 00:01")
        started.set()
        assert release.wait(3)

    manager = DownloadManager(tmp_path / "jobs.db", backend=backend)
    try:
        response = manager.create_job(MODEL)
        assert response["queued"] is True
        assert started.wait(3)
        snapshot = manager.list_jobs()[0]
        assert snapshot["status"] == "running"
        assert snapshot["progress"] == "50% 2 MB/s ETA 00:01"
        snapshot["status"] = "tampered"
        assert manager.list_jobs()[0]["status"] == "running"
        release.set()
        wait_for(lambda: manager.list_jobs()[0]["status"] == "completed")
    finally:
        release.set()
        manager.shutdown()


def test_concurrent_same_target_is_scheduled_once(tmp_path):
    started = threading.Event()
    calls = []

    def backend(job, command, cancel, report):
        calls.append(job["target_id"])
        started.set()
        cancel.wait(3)

    manager = DownloadManager(tmp_path / "jobs.db", backend=backend)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda _: manager.create_job(MODEL), range(16)))
        assert sum(response["queued"] for response in responses) == 1
        assert started.wait(3)
        assert calls == ["ollama:test:latest"]
        manager.cancel_job(calls[0])
        wait_for(lambda: manager.list_jobs()[0]["status"] == "cancelled")
        assert manager.delete_job(calls[0])["ok"] is True
        assert manager.list_jobs() == []
    finally:
        manager.shutdown()


def test_shutdown_cancels_running_and_queued_and_joins(tmp_path):
    started = threading.Event()
    exited = threading.Event()
    calls = []

    def backend(job, command, cancel, report):
        calls.append(job["target_id"])
        started.set()
        assert cancel.wait(3)
        exited.set()

    manager = DownloadManager(tmp_path / "jobs.db", max_workers=1, backend=backend)
    manager.create_job(MODEL)
    assert started.wait(3)
    manager.create_job({"source": "Ollama", "name": "second"})
    manager.shutdown()
    assert exited.is_set()
    assert len(calls) == 1
    assert {job["status"] for job in manager.list_jobs()} == {"interrupted"}
    manager.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        manager.create_job(MODEL)


def test_startup_recovers_persisted_queue_without_executing_it(tmp_path):
    from downloads.store import DownloadStore

    path = tmp_path / "jobs.db"
    store = DownloadStore(path)
    job, _ = store.upsert_job(MODEL)
    store.upsert_job({"source": "Ollama", "name": "queued"})
    store.update_job(job["target_id"], status="running")
    manager = DownloadManager(path, backend=lambda *args: pytest.fail("must not auto resume"))
    try:
        assert {item["status"] for item in manager.list_jobs()} == {"interrupted"}
    finally:
        manager.shutdown()


def test_queued_cancel_can_be_deleted_and_requeued_without_old_worker_updates(tmp_path):
    started = threading.Event()
    release = threading.Event()
    seen = []

    def backend(job, command, cancel, report):
        seen.append(job["name"])
        if job["name"] == MODEL["name"]:
            started.set()
            assert release.wait(3)

    manager = DownloadManager(tmp_path / "jobs.db", max_workers=1, backend=backend)
    try:
        manager.create_job(MODEL)
        assert started.wait(3)
        second = {"source": "Ollama", "name": "second"}
        manager.create_job(second)
        manager.cancel_job("ollama:second")
        assert manager.delete_job("ollama:second")["ok"]
        assert manager.create_job(second)["queued"]
        release.set()
        wait_for(lambda: all(job["status"] == "completed" for job in manager.list_jobs()))
        assert seen == [MODEL["name"], "second"]
    finally:
        release.set()
        manager.shutdown()


def test_preview_and_blocked_submission_preserve_reason_without_queueing(tmp_path):
    manager = DownloadManager(tmp_path / "jobs.db", models_dir=tmp_path / "models")
    model = {"source": "Hugging Face", "id": "owner/repo", "target_file": "model.gguf"}
    try:
        assert manager.preview_job(model)["status"] == "unknown_size"
        assert manager.list_jobs() == []
        with pytest.raises(ValueError, match="unknown_size"):
            manager.create_job(model)
        assert manager.list_jobs() == []
        assert not (tmp_path / "models").exists()
    finally:
        manager.shutdown()


def test_backend_failure_is_persisted_and_target_can_be_retried(tmp_path):
    def backend(*args):
        raise ValueError("provider rejected the download")

    manager = DownloadManager(tmp_path / "jobs.db", backend=backend)
    try:
        manager.create_job(MODEL)
        wait_for(lambda: manager.list_jobs()[0]["status"] == "failed")
        assert manager.list_jobs()[0]["detail"] == "provider rejected the download"
        assert manager.create_job(MODEL)["queued"]
    finally:
        manager.shutdown()
