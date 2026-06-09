"""Tests for app.download_manager.DownloadManager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.download_manager import DownloadManager


@pytest.fixture
def manager():
    callbacks = {
        "update_status": MagicMock(),
        "refresh_table": MagicMock(),
        "refresh_download_history_table": MagicMock(),
        "request_download_history_refresh": MagicMock(),
        "render_download_debug": MagicMock(),
        "find_model_by_target_id": MagicMock(return_value=None),
    }
    return DownloadManager(**callbacks)


class TestInit:
    def test_creates_empty_registry(self, manager):
        assert manager.download_registry == {}

    def test_creates_empty_active_set(self, manager):
        assert manager.active_downloads == set()

    def test_poll_not_running(self, manager):
        assert manager._download_poll_running is False


class TestDownloadCellText:
    def test_idle_state_returns_markup(self, manager):
        text = manager.download_cell_text({"download_state": "idle"})
        assert isinstance(text, str)

    def test_active_state_returns_markup(self, manager):
        text = manager.download_cell_text({"download_state": "downloading", "download_label": "Downloading"})
        assert isinstance(text, str)

    def test_missing_state_defaults_idle(self, manager):
        text = manager.download_cell_text({})
        assert isinstance(text, str)


class TestStatusTextFromState:
    def test_known_state(self, manager):
        text = manager.status_text_from_state("completed", label="Completed")
        assert isinstance(text, str)

    def test_unknown_state(self, manager):
        text = manager.status_text_from_state("unknown")
        assert isinstance(text, str)


class TestCanPoll:
    def test_allows_poll_when_idle(self, manager):
        assert manager.can_poll(modal_poll_pause_count=0, force=False) is True

    def test_blocks_poll_when_modal_paused(self, manager):
        assert manager.can_poll(modal_poll_pause_count=1, force=False) is False

    def test_force_overrides_modal_pause(self, manager):
        assert manager.can_poll(modal_poll_pause_count=1, force=True) is True

    def test_blocks_poll_when_already_running(self, manager):
        manager.set_poll_running(True)
        assert manager.can_poll(modal_poll_pause_count=0, force=False) is False

    def test_force_overrides_running(self, manager):
        manager.set_poll_running(True)
        assert manager.can_poll(modal_poll_pause_count=0, force=True) is True

    def test_resume_after_set_false(self, manager):
        manager.set_poll_running(True)
        manager.set_poll_running(False)
        assert manager.can_poll(modal_poll_pause_count=0, force=False) is True


class TestRecordEntry:
    def test_adds_entry_to_registry(self, manager):
        manager._record_entry("test-id", state="downloading", label="Downloading", detail="50%")
        assert "test-id" in manager.download_registry

    def test_entry_has_expected_keys(self, manager):
        manager._record_entry("test-id", state="completed", label="Done")
        entry = manager.download_registry["test-id"]
        assert entry["state"] == "completed"
        assert entry["label"] == "Done"

    def test_trim_respects_limit(self, manager):
        manager.download_history_limit = 2
        for i in range(5):
            manager._record_entry(f"id-{i}", state="idle")
        assert len(manager.download_registry) <= 2


class TestEnsureDownloadFields:
    def test_adds_download_fields_from_registry(self, manager):
        manager.download_registry["test-id"] = {"state": "downloading", "label": "Active", "detail": "50%"}
        results = [{"source": "Ollama", "name": "llama3"}]
        with patch("app.download_manager.download_target_id", return_value="test-id"):
            changed = manager.ensure_download_fields(results)
        assert changed is True
        assert results[0]["download_state"] == "downloading"

    def test_no_change_when_already_synced(self, manager):
        manager.download_registry["test-id"] = {"state": "idle", "label": "Idle", "detail": ""}
        results = [{"download_state": "idle", "download_label": "Idle", "download_detail": ""}]
        with patch("app.download_manager.download_target_id", return_value="test-id"):
            changed = manager.ensure_download_fields(results)
        assert changed is False


class TestRefreshHistoryTable:
    def test_returns_sorted_entries(self, manager):
        manager.download_registry = {
            "a": {"target_id": "a", "state": "completed", "label": "Done", "source": "HF", "publisher": "meta", "name": "llama", "detail": "", "created_at": 1},
            "b": {"target_id": "b", "state": "downloading", "label": "Active", "source": "Ollama", "publisher": "ollama", "name": "qwen", "detail": "50%", "created_at": 2},
        }
        rows = manager.refresh_history_table(manager.download_registry)
        assert len(rows) == 2
        # Most recent first
        assert rows[0]["key"] == "b"

    def test_empty_registry(self, manager):
        rows = manager.refresh_history_table({})
        assert rows == []


class TestRequestHistoryRefresh:
    def test_returns_true_when_due(self, manager):
        manager.last_download_history_refresh_at = 0
        result = manager.request_history_refresh(force=False)
        assert result is True

    def test_returns_false_when_not_due(self, manager):
        import time
        manager.last_download_history_refresh_at = time.monotonic()
        result = manager.request_history_refresh(force=False)
        assert result is False

    def test_force_overrides_interval(self, manager):
        import time
        manager.last_download_history_refresh_at = time.monotonic()
        result = manager.request_history_refresh(force=True)
        assert result is True


class TestCancelDownload:
    @patch("app.download_manager.cancel_job")
    def test_success_returns_ok(self, mock_cancel, manager):
        mock_cancel.return_value = {"job": {}}
        ok, msg = manager.cancel_download({"source": "Ollama", "name": "llama3"})
        assert ok is True

    @patch("app.download_manager.cancel_job", side_effect=Exception("fail"))
    def test_failure_returns_error(self, mock_cancel, manager):
        ok, msg = manager.cancel_download({"source": "Ollama", "name": "llama3"})
        assert ok is False


class TestStartDownload:
    @patch("app.download_manager.ensure_service_running", return_value=True)
    @patch("app.download_manager.create_job")
    def test_queues_download(self, mock_create, mock_ensure, manager):
        mock_create.return_value = {"queued": True}
        ok, msg = manager.start_download({"source": "HF", "name": "qwen"})
        assert ok is True
        assert "queued" in msg.lower()

    @patch("app.download_manager.ensure_service_running", return_value=False)
    def test_service_unavailable(self, mock_ensure, manager):
        ok, msg = manager.start_download({})
        assert ok is False

    @patch("app.download_manager.ensure_service_running", return_value=True)
    @patch("app.download_manager.create_job", side_effect=Exception("fail"))
    def test_create_fails(self, mock_create, mock_ensure, manager):
        ok, msg = manager.start_download({})
        assert ok is False


class TestDeleteEntry:
    def test_deletes_idle_entry(self, manager):
        manager.download_registry["test-id"] = {"state": "completed", "source": "Ollama", "name": "llama3"}
        with patch("app.download_manager.delete_job"):
            ok, msg, keys, tid = manager.delete_entry("test-id", delete_data=False)
        assert ok is True
        assert "test-id" not in manager.download_registry

    def test_returns_none_keys_on_delete_failure(self, manager):
        manager.download_registry["test-id"] = {"state": "idle"}
        with patch("app.download_manager.delete_job", side_effect=Exception("fail")):
            ok, msg, keys, tid = manager.delete_entry("test-id")
        assert ok is False
        assert keys is None


class TestPollJobs:
    @patch("app.download_manager.list_jobs")
    @patch("app.download_manager.get_active_download_debug")
    def test_returns_jobs_and_debug(self, mock_debug, mock_jobs, manager):
        mock_jobs.return_value = [{"target_id": "x", "status": "running"}]
        mock_debug.return_value = {"count": 1}
        jobs, debug, health = manager.poll_jobs()
        assert jobs == [{"target_id": "x", "status": "running"}]
        assert debug == {"count": 1}


class TestSyncJobs:
    @patch("app.download_manager.list_jobs")
    @patch("app.download_manager.download_target_id", side_effect=lambda x: x.get("id", x.get("source", "?")))
    def test_sync_populates_registry(self, mock_dtid, mock_list, manager):
        mock_list.return_value = [
            {"target_id": "hf:test/model", "source": "HF", "publisher": "meta", "name": "llama", "status": "completed"},
        ]
        manager.sync_jobs(force=True)
        assert len(manager.download_registry) > 0
