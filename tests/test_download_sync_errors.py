"""Regression tests for action-triggered download job synchronization failures."""

from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from app.download_manager import DownloadManager
from downloads.manager import DownloadManager as BackendDownloadManager


def _manager() -> DownloadManager:
    return DownloadManager(
        update_status=MagicMock(),
        refresh_table=MagicMock(),
        refresh_download_history_table=MagicMock(),
        request_download_history_refresh=MagicMock(),
        render_download_debug=MagicMock(),
        find_model_by_target_id=MagicMock(return_value=None),
        backend_manager=BackendDownloadManager.__new__(BackendDownloadManager),
    )


def test_expected_sync_failure_preserves_state_and_surfaces_status():
    manager = _manager()
    manager.download_registry = {"hf:existing": {"state": "downloading"}}
    manager.active_downloads = {"hf:existing"}

    with patch("downloads.manager.DownloadManager.list_jobs", side_effect=URLError("offline")):
        synced = manager.sync_jobs(force=True)

    assert synced is False
    assert manager.download_registry == {"hf:existing": {"state": "downloading"}}
    assert manager.active_downloads == {"hf:existing"}
    manager._update_status.assert_called_once_with(
        "Download history is unavailable; showing last known download state."
    )


def test_unexpected_sync_failure_is_not_silently_swallowed():
    manager = _manager()

    with (
        patch("downloads.manager.DownloadManager.list_jobs", side_effect=AssertionError("programming bug")),
        pytest.raises(AssertionError, match="programming bug"),
    ):
        manager.sync_jobs(force=True)


def test_successful_self_fetch_returns_true():
    manager = _manager()

    with patch("downloads.manager.DownloadManager.list_jobs", return_value=[]):
        assert manager.sync_jobs(force=True) is True


def test_supplied_jobs_bypass_service_fetch():
    manager = _manager()

    with patch("downloads.manager.DownloadManager.list_jobs", side_effect=AssertionError("must not fetch")):
        assert manager.sync_jobs(jobs=[]) is True
