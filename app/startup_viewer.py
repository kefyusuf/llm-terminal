"""Responsive startup lifecycle for the runtime Textual viewer."""

from __future__ import annotations

import time
from contextlib import suppress

from textual import work
from textual.widgets import DataTable

import config
from app.widgets import SystemInfoWidget
from core import cache_db
from tui_app import AIModelViewer as BaseAIModelViewer

_EXPECTED_STARTUP_ERRORS = (OSError, ValueError, RuntimeError)


class AIModelViewer(BaseAIModelViewer):
    """Base viewer variant that restores download history off the UI thread."""

    def on_mount(self) -> None:
        """Mount the UI immediately and restore download history asynchronously."""
        self.title = "AI Model Explorer"
        self._apply_ui_mode()
        self._configure_results_table_columns(force=True)
        self.query_one("#results-table", DataTable).zebra_stripes = True
        download_table = self.query_one("#download-history-table", DataTable)
        download_table.add_columns(
            "Source",
            "Publisher",
            "Model",
            "Status",
            "Detail",
            "Action",
        )
        self.refresh_download_history_table()
        self._update_results_meta(0)

        if self._smoke_mode_enabled():
            self.call_after_refresh(self._finish_smoke_mode)
            self.set_timer(1, self._finish_smoke_mode)
            return

        cache_db.init_db()
        cache_db.cleanup_old_entries()

        cached_specs = cache_db.get_hardware_snapshot()
        if cached_specs is not None:
            self.latest_specs = cached_specs
            self.query_one(SystemInfoWidget).update_info(cached_specs, None)
        else:
            self.query_one(SystemInfoWidget).update("Detecting hardware...")

        self.last_download_history_refresh_at = time.monotonic()
        self.update_status(
            "Ready. Search defaults to Ollama. Select 'Hugging Face' filter for HF models."
        )
        self.request_system_info_refresh(force=True)
        if not self.ollama_running:
            self.update_status(
                "Ready. Ollama is not running; local install/runtime features disabled. Search HF for more models."
            )
        self.system_metrics_timer = self.set_interval(
            config.settings.hardware_poll_interval,
            self.request_system_info_refresh,
        )
        self._run_download_startup_worker()

    @work(thread=True)
    def _run_download_startup_worker(self) -> None:
        """Initialize the manager and fetch history away from the UI thread."""
        jobs = None
        with suppress(_EXPECTED_STARTUP_ERRORS):
            jobs = self.dl.backend.list_jobs(limit=self.dl.download_history_limit)
        self.call_from_thread(self._apply_download_startup, jobs)

    def _apply_download_startup(self, jobs) -> None:
        """Apply the startup snapshot and enable periodic download polling on the UI thread."""
        if jobs is not None:
            self.dl.sync_jobs(force=True, jobs=jobs)
        else:
            self.update_status(
                "Initial download history sync failed. Polling will retry."
            )

        if self.download_status_timer is None:
            self.download_status_timer = self.set_interval(
                config.settings.ui_download_poll_interval,
                self.request_download_poll,
            )
