"""Download actions, history, polling, and progress-panel updates."""

from textual import work
from textual.css.query import NoMatches
from textual.widgets import (
    DataTable,
    Static,
)

from app.download_progress import DownloadProgressPanel
from core.logging_ import get_logger
from downloads.download_manager import download_target_id
from downloads.download_status import is_active_state

logger = get_logger(__name__)


class DownloadControllerMixin:
    """Download actions, history, polling, and progress-panel updates.

    Methods use the viewer's shared state and Textual APIs, preserving subclass hooks.
    """

    def _find_model_by_target_id(self, target_id):
        """Find a model in all_results by its target_id."""
        for item in self.all_results:
            if download_target_id(item) == target_id:
                return item
        return None

    def on_download_job_modal_action(self, result, target_id):
        _ = (result, target_id)

    def cancel_model_download(self, model):
        """Request cancellation of an in-progress or queued download for *model*."""
        ok, msg = self.dl.cancel_download(model)
        self.update_status(msg)
        if ok:
            self.dl.sync_jobs(force=True)
            self._refresh_download_history_table_ui()
            self.refresh_table()

    def delete_download_entry(self, target_id, delete_data=False):
        _ok, msg, keys, tid = self.dl.delete_entry(target_id, delete_data=delete_data)
        if keys:
            source_key, name_key = keys
            from downloads.download_lifecycle import reset_results_download_state
            reset_results_download_state(
                self.all_results,
                target_id=tid,
                source_key=source_key,
                name_key=name_key,
                target_id_for_item=download_target_id,
            )
        self.refresh_table()
        self._refresh_download_history_table_ui()
        self.update_status(msg)

    def start_model_download(self, model):
        ok, msg = self.dl.start_download(model)
        self.update_status(msg)
        if ok:
            self.dl.sync_jobs(force=True)
            self._refresh_download_history_table_ui()

    def _refresh_download_history_table_ui(self):
        self._refresh_download_progress_panel()
        try:
            table = self.query_one("#download-history-table", DataTable)
        except Exception:
            return
        table.clear()
        active_count = len(self.dl.active_downloads)
        if active_count > 0:
            self._download_activity_status = f"{active_count} download(s) active"
            self.update_status(self._download_activity_status)
        else:
            status = self.query_one("#status-bar", Static)
            if status.renderable == getattr(self, "_download_activity_status", None):
                self.update_status("No active downloads")
        rows = self.dl.refresh_history_table(self.dl.download_registry)
        for r in rows:
            table.add_row(
                r["source"], r["publisher"], r["name"],
                self.dl.status_text_from_state(r["state"], r["label"]),
                r["detail"], r["action"],
                key=r["key"],
            )

    def _request_download_history_refresh_ui(self, force=False):
        if self.dl.request_history_refresh(force):
            self._refresh_download_history_table_ui()

    def refresh_download_history_table(self):
        self._refresh_download_history_table_ui()

    def request_download_history_refresh(self, force=False):
        self._request_download_history_refresh_ui(force)

    def request_download_poll(self, force=False):
        if not self.dl.can_poll(self._modal_poll_pause_count, force):
            return
        self.dl.set_poll_running(True)
        self._run_download_poll_worker()

    @work(thread=True)
    def _run_download_poll_worker(self):
        jobs, debug, health = self.dl.poll_jobs()
        self.call_from_thread(self._apply_download_poll_snapshot, jobs, debug, health)

    def _apply_download_poll_snapshot(self, jobs, debug, health):
        self.dl.set_poll_running(False)
        previous_active = set(self.dl.active_downloads)
        if jobs is not None:
            self.dl.sync_jobs(force=False, jobs=jobs)
            changed = self.dl.ensure_download_fields(self.all_results)
            if changed:
                self.refresh_table()
        self._render_download_debug(debug, health)
        self._refresh_download_progress_panel()

        if jobs is not None and previous_active - self.dl.active_downloads:
            self._request_download_history_refresh_ui(force=True)
            return
        if not self.dl.active_downloads:
            if self.dl.download_history_refresh_pending:
                self._request_download_history_refresh_ui()
            return
        if any(
            is_active_state(self.dl.download_registry.get(target_id, {}).get("state"))
            for target_id in self.dl.active_downloads
        ):
            self._request_download_history_refresh_ui()

    def refresh_download_progress(self):
        self.request_download_poll()

    def _refresh_download_progress_panel(self):
        """Expose current transfer details even when compact history is hidden."""
        try:
            root_screen = self.screen_stack[0]
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

    def refresh_download_debug(self):
        self.request_download_poll(force=True)

    def _render_download_debug(self, debug=None, health=None):
        try:
            debug_widget = self.query_one("#downloads-debug", Static)
        except Exception:
            return

        if debug is not None:
            count = debug.get("count", 0)
            has_duplicates = bool(debug.get("has_duplicates", False))
            worker_alive = bool(debug.get("worker_alive", True))
            dup_text = "yes" if has_duplicates else "no"
            worker_text = "up" if worker_alive else "down"
            debug_widget.update(f"Workers: {count} ({worker_text}) | Duplicates: {dup_text}")
            return

        debug_widget.update("Workers: unavailable | Duplicates: unknown")

    def _download_cell_text(self, model):
        return self.dl.download_cell_text(model)

    def _mark_ollama_model_installed(self, model_name):
        for item in self.all_results:
            if item.get("source") == "Ollama" and item.get("name") == model_name:
                item["inst"] = "[green]✔[/green]"
        self.refresh_table()
