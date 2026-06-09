"""Tests for download_service worker functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from downloads.download_service import STATE, DownloadStore, worker_loop


def _make_store():
    """Create a store mock that causes the worker loop to exit after one iteration."""
    store = MagicMock()
    store.claim_next_queued.side_effect = _claim_and_stop
    store.get_job_by_target.return_value = {"cancel_requested": False}
    store.get_command.return_value = None
    return store


_claim_return = None


def set_claim_return(value):
    global _claim_return
    _claim_return = value


def _claim_and_stop():
    STATE.stop_event.set()
    return _claim_return


@pytest.fixture(autouse=True)
def reset_state():
    STATE.stop_event.clear()
    STATE.store = None
    STATE.processes = {}
    global _claim_return
    _claim_return = None
    yield
    STATE.stop_event.set()


@pytest.fixture
def store(tmp_path):
    return DownloadStore(tmp_path / "test_jobs.db")


class TestWorkerLoop:
    def test_loop_exits_when_stop_event_set(self):
        STATE.stop_event.set()
        STATE.store = MagicMock()
        worker_loop()

    def test_loop_skips_when_no_job(self):
        STATE.store = _make_store()
        set_claim_return(None)
        worker_loop()

    def test_loop_processes_job(self):
        store = _make_store()
        store.get_job_by_target.return_value = {"cancel_requested": False}
        store.get_command.return_value = ["echo", "hello"]
        STATE.store = store
        set_claim_return({"target_id": "test:job"})
        with patch("downloads.download_service._run_stream_download_job") as mock_run:
            worker_loop()
        mock_run.assert_called_once_with("test:job", ["echo", "hello"])

    def test_loop_skips_cancelled_job(self):
        store = _make_store()
        store.get_job_by_target.return_value = {"cancel_requested": True}
        STATE.store = store
        set_claim_return({"target_id": "test:cancel"})
        worker_loop()
        store.update_job.assert_called_with("test:cancel", status="cancelled", detail="Canceled", progress="")

    def test_loop_handles_missing_command(self):
        store = _make_store()
        store.get_job_by_target.return_value = {"cancel_requested": False}
        store.get_command.return_value = None
        STATE.store = store
        set_claim_return({"target_id": "test:nocmd"})
        worker_loop()
        store.update_job.assert_called()


class TestClaimNextQueued:
    def test_returns_queued_job(self, store):
        model = {"source": "Ollama", "name": "test-model", "publisher": "ollama"}
        job, _ = store.upsert_job(model)
        target_id = job["target_id"]

        claimed = store.claim_next_queued()
        assert claimed is not None
        assert claimed["target_id"] == target_id

    def test_returns_none_when_empty(self, store):
        claimed = store.claim_next_queued()
        assert claimed is None

    def test_skips_active_jobs(self, store):
        model = {"source": "Hugging Face", "name": "test/model", "publisher": "test", "id": "test/model"}
        job, _ = store.upsert_job(model)
        store.update_job(job["target_id"], status="running", detail="Downloading")

        claimed = store.claim_next_queued()
        assert claimed is None

    def test_prioritizes_oldest(self, store):
        store.upsert_job({"source": "Ollama", "name": "first", "publisher": "ollama"})
        store.upsert_job({"source": "Ollama", "name": "second", "publisher": "ollama"})

        first = store.claim_next_queued()
        second = store.claim_next_queued()
        assert first is not None
        assert second is not None
        assert first["target_id"] != second["target_id"]


class TestGetJobByTarget:
    def test_returns_job(self, store):
        model = {"source": "Ollama", "name": "find-me", "publisher": "ollama"}
        job, _ = store.upsert_job(model)
        found = store.get_job_by_target(job["target_id"])
        assert found is not None
        assert found["name"] == "find-me"

    def test_returns_none_for_missing(self, store):
        assert store.get_job_by_target("nonexistent") is None


class TestListJobs:
    def test_lists_all_jobs(self, store):
        store.upsert_job({"source": "Ollama", "name": "a", "publisher": "ollama"})
        store.upsert_job({"source": "Hugging Face", "name": "b", "publisher": "x", "id": "b"})
        jobs = store.list_jobs(limit=10)
        assert len(jobs) == 2

    def test_respects_limit(self, store):
        for i in range(5):
            store.upsert_job({"source": "Ollama", "name": f"m{i}", "publisher": "ollama"})
        jobs = store.list_jobs(limit=3)
        assert len(jobs) == 3

    def test_orders_by_created_at_desc(self, store):
        import time
        store.upsert_job({"source": "Ollama", "name": "first", "publisher": "ollama"})
        time.sleep(0.01)
        store.upsert_job({"source": "Ollama", "name": "second", "publisher": "ollama"})
        jobs = store.list_jobs(limit=10)
        assert jobs[0]["name"] == "second"
        assert jobs[1]["name"] == "first"


class TestUpdateJob:
    def test_updates_status(self, store):
        model = {"source": "Ollama", "name": "update-me", "publisher": "ollama"}
        job, _ = store.upsert_job(model)
        store.update_job(job["target_id"], status="running", detail="Working")
        updated = store.get_job_by_target(job["target_id"])
        assert updated["status"] == "running"
        assert updated["detail"] == "Working"

    def test_updates_progress(self, store):
        model = {"source": "Ollama", "name": "prog-test", "publisher": "x"}
        job, _ = store.upsert_job(model)
        store.update_job(job["target_id"], progress="75%")
        updated = store.get_job_by_target(job["target_id"])
        assert updated["progress"] == "75%"


class TestGetCommand:
    def test_returns_command_list(self, store):
        model = {"source": "Ollama", "name": "cmd-test", "publisher": "ollama"}
        job, _ = store.upsert_job(model)
        cmd = store.get_command(job["target_id"])
        assert isinstance(cmd, list)

    def test_returns_none_for_missing(self, store):
        cmd = store.get_command("nonexistent")
        assert cmd is None or cmd == []
