"""Responsive startup lifecycle for the runtime Textual viewer."""

from __future__ import annotations

import time

from textual import work
from textual.widgets import DataTable

import config
from app.widgets import SystemInfoWidget
from core import cache_db
from core.hardware import check_ollama_running
from downloads.service_client import ensure_service_running, list_jobs
from tui_app import AIModelViewer as BaseAIModelViewer

_EXPECTED_STARTUP_ERRORS = (OSError, ValueError, RuntimeError)


class AIModelViewer(BaseAIModelViewer):
    """Base viewer variant that keeps download-service startup off the UI thread."""

    def on_mount(self) -> None:
        """Mount the UI immediately and initialize the download service asynchronously."""
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
            self.query_one(SystemInfoWidget).update_info(cached_specs, check_ollama_running())

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
        self._run_download_service_startup_worker()

    @work(thread=True)
    def _run_download_service_startup_worker(self) -> None:
        """Start/reuse the service and fetch the first job snapshot away from the UI thread."""
        service_ok = False
        jobs = None
        sync_failed = False

        try:
            service_ok = ensure_service_running()
        except _EXPECTED_STARTUP_ERRORS:
            service_ok = False

        if service_ok:
            try:
                jobs = list_jobs(
                    limit=self.dl.download_history_limit,
                    timeout=self.dl.download_poll_request_timeout,
                )
            except _EXPECTED_STARTUP_ERRORS:
                sync_failed = True

        self.call_from_thread(
            self._apply_download_service_startup,
            service_ok,
            jobs,
            sync_failed,
        )

    def _apply_download_service_startup(self, service_ok: bool, jobs, sync_failed: bool) -> None:
        """Apply the startup snapshot and enable periodic download polling on the UI thread."""
        if not service_ok:
            self.update_status(
                "Download service is unavailable or outdated. Restart the app/service."
            )
        elif jobs is not None:
            self.dl.sync_jobs(force=True, jobs=jobs)
        elif sync_failed:
            self.update_status(
                "Download service started; initial history sync failed. Polling will retry."
            )

        if self.download_status_timer is None:
            self.download_status_timer = self.set_interval(
                config.settings.ui_download_poll_interval,
                self.request_download_poll,
            )
