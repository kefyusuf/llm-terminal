"""Runtime TUI viewer extensions for provider selection and responsive download actions."""

from __future__ import annotations

from collections.abc import Sequence

from textual import on, work
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, DataTable, Input, Select

from app.modals import DownloadJobModal as BaseDownloadJobModal
from app.startup_viewer import AIModelViewer as BaseAIModelViewer
from downloads.download_history import cancel_model_payload, fallback_entry_from_target, is_external_entry
from downloads.download_lifecycle import reset_results_download_state
from downloads.download_manager import download_target_id
from downloads.download_status import is_active_state
from providers import get_provider_filter_labels
from results.results_view import result_unique_key


_PROVIDER_COMPACT_TAGS = {
    "Ollama": "OL",
    "Hugging Face": "HF",
    "LM Studio": "LM",
    "Docker": "DK",
    "MLX": "MLX",
}


def cycle_provider_label(labels: Sequence[str], current: str) -> str:
    """Return the next provider label, recovering unknown state to the first option."""
    if not labels:
        return current
    if current not in labels:
        return labels[0]
    return labels[(labels.index(current) + 1) % len(labels)]


def provider_compact_tag(label: str) -> str:
    """Return a short stable tag for a provider label in compact mode."""
    return _PROVIDER_COMPACT_TAGS.get(label, label[:3].upper() or "-")


class DownloadJobModal(BaseDownloadJobModal):
    """Runtime download modal that delegates cancel+delete to one manager operation."""

    @on(Button.Pressed, "#job-cancel-delete-btn")
    def cancel_and_delete(self) -> None:
        """Let the delete-data path own cancel-before-delete for active jobs."""
        delete_fn = getattr(self.app, "delete_download_entry", None)
        if callable(delete_fn):
            delete_fn(str(self.entry.get("target_id", "")), delete_data=True)
        self.dismiss()


class AIModelViewer(BaseAIModelViewer):
    """Run the main viewer with synchronized provider and download interactions."""

    def __init__(self):
        """Snapshot available provider labels for both mouse and keyboard selection."""
        super().__init__()
        labels = tuple(get_provider_filter_labels())
        self.provider_filter_labels = labels or ("Ollama",)
        if self.current_filter not in self.provider_filter_labels:
            self.current_filter = self.provider_filter_labels[0]

    async def on_mount(self) -> None:
        """Mount the compact provider selector after the base UI initializes."""
        super().on_mount()
        panel = self.query_one("#provider-panel", Vertical)
        await panel.remove_children()
        selector = Select(
            ((label, label) for label in self.provider_filter_labels),
            value=self.current_filter,
            allow_blank=False,
            id="provider-select",
        )
        selector.styles.width = "100%"
        selector.styles.height = 3
        await panel.mount(selector)

    def _get_selected_model(self) -> dict | None:
        """Resolve the selected result through Textual's stable DataTable row key."""
        table = self.query_one("#results-table", DataTable)
        cursor_row = table.cursor_row
        if cursor_row < 0 or cursor_row >= table.row_count:
            return None

        row_key, _column_key = table.coordinate_to_cell_key(table.cursor_coordinate)
        row_key_value = row_key.value
        if row_key_value is None:
            return None

        return next(
            (item for item in self.all_results if result_unique_key(item) == row_key_value),
            None,
        )

    def _apply_resize_reflow(self, generation: int) -> None:
        """Ignore a deferred resize callback after the results table has unmounted."""
        if generation != self._resize_reflow_generation:
            return
        try:
            self.query_one("#results-table", DataTable)
        except NoMatches:
            self._resize_reflow_timer = None
            return
        super()._apply_resize_reflow(generation)

    def _apply_provider_filter(self, label: str, *, sync_widget: bool) -> None:
        """Apply one provider label and keep the mounted selector synchronized."""
        if label not in self.provider_filter_labels or label == self.current_filter:
            return

        self.current_filter = label
        if sync_widget:
            try:
                selector = self.query_one("#provider-select", Select)
                if selector.value != label:
                    selector.value = label
            except NoMatches:
                pass

        current_query = self.query_one("#search-input", Input).value.strip()
        if current_query:
            self.start_search(current_query)
            self.update_status(f"Provider switched to {label}. Searching...")
        else:
            self.refresh_table()
            self.update_status(f"Provider filter set to {label}.")

    def action_cycle_provider(self) -> None:
        """Cycle through the exact provider labels displayed by the selector."""
        next_filter = cycle_provider_label(self.provider_filter_labels, self.current_filter)
        self._apply_provider_filter(next_filter, sync_widget=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Apply provider changes made directly through the mounted selector."""
        if event.select.id != "provider-select":
            return
        self._apply_provider_filter(str(event.value), sync_widget=False)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Use the responsive runtime modal for download-history detail rows."""
        data_table = getattr(event, "data_table", None)
        if data_table is not None and data_table.id == "download-history-table":
            target_id = str(event.row_key.value)
            entry = self.dl.download_registry.get(target_id)
            if not entry:
                entry = fallback_entry_from_target(target_id)

            if data_table.cursor_column == 5:
                state = entry.get("state", "idle")
                if is_external_entry(entry):
                    self.update_status("External download; management unavailable in this app.")
                    return
                if is_active_state(state):
                    self.cancel_model_download(cancel_model_payload(target_id, entry))
                else:
                    self.delete_download_entry(target_id, delete_data=False)
            else:
                self.push_screen(DownloadJobModal(entry))
            return

        super().on_data_table_row_selected(event)

    def _apply_download_poll_snapshot(self, jobs, debug, health):
        """Apply the base snapshot and surface deduplicated poll status transitions."""
        super()._apply_download_poll_snapshot(jobs, debug, health)
        status_message = self.dl.take_poll_status_message()
        if status_message:
            self.update_status(status_message)

    def delete_download_entry(self, target_id, delete_data=False):
        """Schedule potentially blocking deletion outside Textual's UI thread."""
        self._run_delete_download_entry_worker(str(target_id), delete_data=bool(delete_data))

    @work(thread=True)
    def _run_delete_download_entry_worker(self, target_id: str, delete_data: bool = False):
        """Perform service/subprocess deletion work and hand the result back to the UI thread."""
        result = self.dl.delete_entry(target_id, delete_data=delete_data)
        self.call_from_thread(self._apply_delete_download_entry_result, result)

    def _apply_delete_download_entry_result(self, result) -> None:
        """Apply one completed delete result on the Textual UI thread."""
        _ok, msg, keys, target_id = result
        if keys:
            source_key, name_key = keys
            reset_results_download_state(
                self.all_results,
                target_id=target_id,
                source_key=source_key,
                name_key=name_key,
                target_id_for_item=download_target_id,
            )
        self.refresh_table()
        self._refresh_download_history_table_ui()
        self.update_status(msg)

    def _compact_chip_text(self, shown_count: int, total: int) -> str:
        """Render compact search state using a provider-specific short tag."""
        provider_short = provider_compact_tag(self.current_filter)
        use_case_label = self._use_case_compact_tag(self.use_case_filter)
        sort_label = self._sort_compact_tag(self.sort_mode)
        fit_label = self._fit_compact_tag(self.fit_filter)
        gems_label = "ON" if self.hidden_gems_only else "OFF"
        page_label = str(self.current_page + 1) if self.current_filter == "Hugging Face" else "1"

        return (
            f"[#8ea3cf]M:[/#8ea3cf][#dbe7ff]{shown_count}/{total}[/#dbe7ff]  "
            f"[#8ea3cf]P:[/#8ea3cf][#9fe8ff]{provider_short}[/#9fe8ff]  "
            f"[#8ea3cf]U:[/#8ea3cf][#d1b3ff]{use_case_label}[/#d1b3ff]  "
            f"[#8ea3cf]S:[/#8ea3cf][#7edfff]{sort_label}[/#7edfff]  "
            f"[#8ea3cf]F:[/#8ea3cf][#f2c46d]{fit_label}[/#f2c46d]  "
            f"[#8ea3cf]G:[/#8ea3cf][#4fe08a]{gems_label}[/#4fe08a]  "
            f"[#8ea3cf]Pg:[/#8ea3cf][#dbe7ff]{page_label}[/#dbe7ff]"
        )
