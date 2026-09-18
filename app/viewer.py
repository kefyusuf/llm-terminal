"""Runtime TUI viewer extensions for compact filters and responsive download actions."""

from __future__ import annotations

import time
from collections.abc import Sequence

from textual import on, work
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, DataTable, Input, RadioSet, Select

from app.modals import DownloadJobModal as BaseDownloadJobModal
from app.responsive_modals import ModelDetailModal as ResponsiveModelDetailModal
from app.search_constants import USE_CASE_OPTIONS
from app.startup_viewer import AIModelViewer as BaseAIModelViewer
from downloads.download_history import cancel_model_payload, fallback_entry_from_target, is_external_entry
from downloads.download_lifecycle import reset_results_download_state
from downloads.download_manager import download_target_id
from downloads.download_status import is_active_state
from providers import get_provider_filter_labels
from results.results_view import filter_results_for_view, result_unique_key
from search.search_orchestration import build_query_key, cache_hit_suffix, provider_display_name


_PROVIDER_COMPACT_TAGS = {
    "Ollama": "OL",
    "Hugging Face": "HF",
    "LM Studio": "LM",
    "Docker": "DK",
    "MLX": "MLX",
}
_RENDERED_RESULT_FIELDS = (
    "inst",
    "source",
    "publisher",
    "name",
    "params",
    "use_case",
    "score",
    "quant",
    "mode",
    "fit",
)


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
    """Run the main viewer with synchronized compact filters and download interactions."""

    def __init__(self):
        """Snapshot selector choices for mouse and keyboard interaction."""
        super().__init__()
        labels = tuple(get_provider_filter_labels())
        self.provider_filter_labels = labels or ("Ollama",)
        self.use_case_filter_keys = tuple(key for key, _label in USE_CASE_OPTIONS)
        if self.current_filter not in self.provider_filter_labels:
            self.current_filter = self.provider_filter_labels[0]
        if self.use_case_filter not in self.use_case_filter_keys:
            self.search_state.set_use_case("all")
        self._results_table_render_signature = None
        self._live_first_stale_query_key: str | None = None

    async def on_mount(self) -> None:
        """Mount compact provider and use-case selectors after the base UI initializes."""
        super().on_mount()

        provider_panel = self.query_one("#provider-panel", Vertical)
        await provider_panel.remove_children()
        provider_selector = Select(
            ((label, label) for label in self.provider_filter_labels),
            value=self.current_filter,
            allow_blank=False,
            id="provider-select",
        )
        provider_selector.styles.width = "100%"
        provider_selector.styles.height = 3
        await provider_panel.mount(provider_selector)

        use_case_panel = self.query_one("#use-case-panel", Vertical)
        use_case_selector = Select(
            ((label, key) for key, label in USE_CASE_OPTIONS),
            value=self.use_case_filter,
            allow_blank=False,
            id="use-case-select",
        )
        use_case_selector.styles.width = "100%"
        use_case_selector.styles.height = 3
        await use_case_panel.mount(use_case_selector)
        self._hide_legacy_use_case_radios()

    def _hide_legacy_use_case_radios(self) -> None:
        """Keep the base RadioSet in the DOM for base layout code without rendering it."""
        try:
            radio_set = self.query_one("#use-case-filter", RadioSet)
        except NoMatches:
            return
        radio_set.styles.display = "none"

    def _apply_ui_mode(self) -> None:
        """Apply the base layout, then keep the runtime use-case selector as the visible control."""
        super()._apply_ui_mode()
        self._hide_legacy_use_case_radios()
        try:
            selector = self.query_one("#use-case-select", Select)
        except NoMatches:
            return
        selector.styles.display = "block"
        selector.styles.width = "100%"
        selector.styles.height = 3

    def _dispatch_debounced_search(self) -> None:
        """Use fresh cache immediately, but reserve stale data for post-live fallback."""
        self._search_debounce_timer = None
        payload = self._pending_search_payload
        self._pending_search_payload = None
        if payload is None:
            return

        query, providers, page, signature = payload
        self.current_page = page
        self._search_inflight_signature = signature
        self._search_inflight_started_at = time.monotonic()
        query_key = build_query_key(providers, query, self.current_page)

        current_specs = self._current_specs_for_search_ui()
        self.last_search_error = ""
        self.search_counter += 1
        self.active_search_id = self.search_counter
        table = self.query_one("#results-table", DataTable)
        table.clear()
        self._table_row_keys = set()
        table.loading = True
        self._search_progress_visible = False
        self._update_results_meta(0)

        provider_name = provider_display_name(providers)
        self.on_search_progress(self.active_search_id, f"Searching {provider_name}: {query}")

        self._live_first_stale_query_key = None
        cached = self.search_cache.get(query_key, current_specs)
        if cached:
            self.all_results = [item.copy() for item in cached["results"]]
            self.dl.ensure_download_fields(self.all_results)
            self.last_search_error = cached["error"]
            if "has_more_pages" in cached:
                self.has_more_pages = cached["has_more_pages"]
            self.on_search_completed(self.active_search_id)
            cache_msg = cache_hit_suffix(providers, self.current_page)
            self.update_status(f"Loaded{cache_msg}")
            return

        self._live_first_stale_query_key = query_key
        self.run_search_worker(query, query_key, self.active_search_id, providers)

    def on_search_completed(self, search_id: int) -> None:
        """Use retained stale data only when an attempted live search fails empty."""
        stale_query_key = self._live_first_stale_query_key
        should_try_stale = (
            search_id == self.active_search_id
            and stale_query_key is not None
            and not self.all_results
            and bool(self.last_search_error)
        )

        if should_try_stale:
            stale = self.search_cache.get_stale(stale_query_key)
            if stale and stale.get("results"):
                self.all_results = [item.copy() for item in stale["results"]]
                self.dl.ensure_download_fields(self.all_results)
                self.last_search_error = "Offline — showing cached results"
                if "has_more_pages" in stale:
                    self.has_more_pages = stale["has_more_pages"]
                self._live_first_stale_query_key = None
                super().on_search_completed(search_id)
                self.update_status("Offline mode — showing cached results")
                return

        self._live_first_stale_query_key = None
        super().on_search_completed(search_id)

    def _results_table_signature(self):
        """Capture ordered, non-download content that determines the rendered table."""
        filtered_results = filter_results_for_view(
            self.all_results,
            current_filter=self.current_filter,
            use_case_filter=self.use_case_filter,
            hidden_gems_only=self.hidden_gems_only,
            sort_mode=self.sort_mode,
            fit_filter=self.fit_filter,
        )

        seen_keys: set[str] = set()
        row_signature = []
        for result in filtered_results:
            unique_key = result_unique_key(result)
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)
            row_signature.append(
                (
                    unique_key,
                    *(str(result.get(field, "")) for field in _RENDERED_RESULT_FIELDS),
                )
            )

        column_signature = (
            tuple(self.results_column_keys),
            tuple(sorted(self.results_column_widths.items())),
        )
        return column_signature, tuple(row_signature)

    def refresh_table(self) -> None:
        """Keep the download-only fast path only while rendered table content is unchanged."""
        next_signature = self._results_table_signature()
        previous_signature = self._results_table_render_signature
        if previous_signature is not None and next_signature != previous_signature:
            self._table_row_keys = set()

        super().refresh_table()
        self._results_table_render_signature = self._results_table_signature()

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

    def _set_use_case_filter(self, key: str) -> None:
        """Apply one use-case key and synchronize the runtime selector when mounted."""
        if key not in self.use_case_filter_keys:
            return

        self.search_state.set_use_case(key)
        try:
            selector = self.query_one("#use-case-select", Select)
        except NoMatches:
            super()._set_use_case_filter(key)
            return
        if selector.value != key:
            selector.value = key

    def action_cycle_provider(self) -> None:
        """Cycle through the exact provider labels displayed by the selector."""
        next_filter = cycle_provider_label(self.provider_filter_labels, self.current_filter)
        self._apply_provider_filter(next_filter, sync_widget=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Apply provider or use-case changes made through compact selectors."""
        if event.select.id == "provider-select":
            self._apply_provider_filter(str(event.value), sync_widget=False)
            return

        if event.select.id != "use-case-select":
            return

        key = str(event.value)
        if key not in self.use_case_filter_keys or key == self.use_case_filter:
            return
        self._set_use_case_filter(key)
        self.refresh_table()
        self.update_status(f"Use Case filter set to {self._use_case_label(key)}.")

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

    def open_model_detail_modal(self, model):
        """Open the viewport-bounded runtime model detail modal."""
        self.push_screen(ResponsiveModelDetailModal(model))

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
