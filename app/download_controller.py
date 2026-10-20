"""Download actions, history table, polling, and progress-panel updates."""

from __future__ import annotations

from contextlib import suppress

from textual.css.query import NoMatches
from textual.widgets import DataTable, Static

from app.download_progress import DownloadProgressPanel
from app.modals import DownloadJobModal
from core.logging_ import get_logger
from downloads.download_history import (
    cancel_model_payload,
    fallback_entry_from_target,
    is_external_entry,
)
from downloads.download_lifecycle import reset_results_download_state
from downloads.download_manager import download_target_id
from downloads.download_status import is_active_state

logger = get_logger(__name__)

_EXPECTED_STARTUP_ERRORS = (OSError, ValueError, RuntimeError)
HISTORY_COLUMNS = ("Source", "Publisher", "Model", "Status", "Detail", "Action")
ACTION_COLUMN = HISTORY_COLUMNS.index("Action")


class DownloadController:
    """Connect the DownloadTracker to the history table, progress panel, and results."""

    def __init__(self, app, state, tracker):
        self.app = app
        self.state = state
        self.dl = tracker
        self._activity_status = None

    def _status(self, text) -> None:
        self.app.status.update(text)

    def find_model(self, target_id):
        return next(
            (item for item in self.state.results if download_target_id(item) == target_id),
            None,
        )

    # -- startup ----------------------------------------------------------

    def setup_history_table(self) -> None:
        self.app.query_one("#download-history-table", DataTable).add_columns(*HISTORY_COLUMNS)
        self.refresh_history_table()

    def start(self) -> None:
        """Initialize the backend and fetch history away from the UI thread."""
        self.app.run_worker(self._startup_worker, thread=True, group="downloads")

    def _startup_worker(self) -> None:
        jobs = None
        with suppress(_EXPECTED_STARTUP_ERRORS):
            jobs = self.dl.backend.list_jobs(limit=self.dl.download_history_limit)
        self.app.call_from_thread(self.apply_startup, jobs)

    def apply_startup(self, jobs) -> None:
        """Apply the startup snapshot, then let periodic polling take over."""
        if jobs is not None:
            self.dl.sync_jobs(force=True, jobs=jobs)
        else:
            self._status("Initial download history sync failed. Polling will retry.")
        self.app.start_download_polling()

    # -- actions ----------------------------------------------------------

    def start_download(self, model) -> None:
        ok, msg = self.dl.start_download(model)
        self._status(msg)
        if ok:
            self.dl.sync_jobs(force=True)
            self.refresh_history_table()

    def cancel_download(self, model) -> None:
        """Request cancellation of an in-progress or queued download for *model*."""
        ok, msg = self.dl.cancel_download(model)
        self._status(msg)
        if ok:
            self.dl.sync_jobs(force=True)
            self.refresh_history_table()
            self.app.table.refresh()

    def delete_entry(self, target_id, delete_data=False) -> None:
        """Schedule potentially blocking deletion outside Textual's UI thread."""
        target_id, delete_data = str(target_id), bool(delete_data)
        self.app.run_worker(
            lambda: self.app.call_from_thread(
                self.apply_delete_result, self.dl.delete_entry(target_id, delete_data=delete_data)
            ),
            thread=True,
            group="downloads",
        )

    def apply_delete_result(self, result) -> None:
        _ok, msg, keys, target_id = result
        if keys:
            source_key, name_key = keys
            reset_results_download_state(
                self.state.results,
                target_id=target_id,
                source_key=source_key,
                name_key=name_key,
                target_id_for_item=download_target_id,
            )
        self.app.table.refresh()
        self.refresh_history_table()
        self._status(msg)

    def mark_ollama_installed(self, model_name) -> None:
        for item in self.state.results:
            if item.get("source") == "Ollama" and item.get("name") == model_name:
                item["inst"] = "[green]✔[/green]"
        self.app.table.refresh()

    def on_history_row_selected(self, data_table, row_key) -> None:
        target_id = str(row_key)
        entry = self.dl.download_registry.get(target_id) or fallback_entry_from_target(target_id)
        if data_table.cursor_column != ACTION_COLUMN:
            self.app.push_screen(DownloadJobModal(entry))
            return
        if is_external_entry(entry):
            self._status("External download; management unavailable in this app.")
        elif is_active_state(entry.get("state", "idle")):
            self.cancel_download(cancel_model_payload(target_id, entry))
        else:
            self.delete_entry(target_id, delete_data=False)

    # -- history and progress ---------------------------------------------

    def refresh_history_table(self) -> None:
        self.refresh_progress_panel()
        try:
            table = self.app.query_one("#download-history-table", DataTable)
        except Exception:
            return
        table.clear()
        active_count = len(self.dl.active_downloads)
        if active_count > 0:
            self._activity_status = f"{active_count} download(s) active"
            self._status(self._activity_status)
        elif self.app.status.text() == self._activity_status:
            self._status("No active downloads")
        for row in self.dl.refresh_history_table(self.dl.download_registry):
            table.add_row(
                row["source"],
                row["publisher"],
                row["name"],
                self.dl.status_text_from_state(row["state"], row["label"]),
                row["detail"],
                row["action"],
                key=row["key"],
            )

    def request_history_refresh(self, force=False) -> None:
        if self.dl.request_history_refresh(force):
            self.refresh_history_table()

    def refresh_progress_panel(self) -> None:
        """Expose current transfer details even when compact history is hidden."""
        try:
            root_screen = self.app.screen_stack[0]
            panel = root_screen.query_one("#active-download-progress", DownloadProgressPanel)
            table = root_screen.query_one("#results-table", DataTable)
        except (NoMatches, IndexError):
            # Poll workers can finish after shutdown has removed these widgets.
            return
        entries = list(self.dl.download_registry.values())
        active = [entry for entry in entries if is_active_state(entry.get("state"))]
        entry = max(active or entries, key=lambda item: item.get("updated_at", 0), default=None)
        table.styles.min_height = 5 if entry else 15
        panel.show_entry(entry, other_active=max(0, len(active) - 1))

    def render_debug(self, debug=None, health=None) -> None:
        _ = health
        try:
            debug_widget = self.app.query_one("#downloads-debug", Static)
        except Exception:
            return
        if debug is None:
            debug_widget.update("Workers: unavailable | Duplicates: unknown")
            return
        dup_text = "yes" if debug.get("has_duplicates", False) else "no"
        worker_text = "up" if debug.get("worker_alive", True) else "down"
        debug_widget.update(
            f"Workers: {debug.get('count', 0)} ({worker_text}) | Duplicates: {dup_text}"
        )

    # -- polling ----------------------------------------------------------

    def request_poll(self, force=False) -> None:
        if not self.dl.can_poll(self.app.status.modal_poll_pause_count, force):
            return
        self.dl.set_poll_running(True)
        self.app.run_worker(self._poll_worker, thread=True, group="download-poll")

    def _poll_worker(self) -> None:
        jobs, debug, health = self.dl.poll_jobs()
        self.app.call_from_thread(self.apply_poll_snapshot, jobs, debug, health)

    def apply_poll_snapshot(self, jobs, debug, health) -> None:
        self.dl.set_poll_running(False)
        previous_active = set(self.dl.active_downloads)
        if jobs is not None:
            self.dl.sync_jobs(force=False, jobs=jobs)
            if self.dl.ensure_download_fields(self.state.results):
                self.app.table.refresh()
        self.render_debug(debug, health)
        self.refresh_progress_panel()
        self._refresh_history_after_poll(jobs, previous_active)
        status_message = self.dl.take_poll_status_message()
        if status_message:
            self._status(status_message)

    def _refresh_history_after_poll(self, jobs, previous_active) -> None:
        if jobs is not None and previous_active - self.dl.active_downloads:
            self.request_history_refresh(force=True)
            return
        if not self.dl.active_downloads:
            if self.dl.download_history_refresh_pending:
                self.request_history_refresh()
            return
        if any(
            is_active_state(self.dl.download_registry.get(target_id, {}).get("state"))
            for target_id in self.dl.active_downloads
        ):
            self.request_history_refresh()
