"""Search actions, debounce, caching, pagination, and worker callbacks."""

import time
from contextlib import suppress

from textual import work
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    RadioButton,
    RadioSet,
    Static,
)

import config
from app.search_constants import (
    FIT_COMPACT_TAGS,
    FIT_OPTIONS,
    SORT_COMPACT_TAGS,
    SORT_OPTIONS,
    USE_CASE_COMPACT_TAGS,
    USE_CASE_OPTIONS,
)
from core import cache_db
from core.logging_ import get_logger
from providers import get_provider_filter_labels
from search.search_orchestration import (
    build_query_key,
    cache_hit_suffix,
    is_hf_provider_selection,
    page_info_suffix,
    provider_display_name,
    provider_search_status,
    providers_from_filter,
    validate_page_request,
)

logger = get_logger(__name__)


class SearchControllerMixin:
    """Search actions, debounce, caching, pagination, and worker callbacks.

    Methods use the viewer's shared state and Textual APIs, preserving subclass hooks.
    """

    def _use_case_label(self, key: str) -> str:
        for option_key, option_label in USE_CASE_OPTIONS:
            if option_key == key:
                return option_label
        return "Any Use"

    def _sort_label(self, key: str) -> str:
        for option_key, option_label in SORT_OPTIONS:
            if option_key == key:
                return option_label
        return "Score"

    def _fit_label(self, key: str) -> str:
        for option_key, option_label in FIT_OPTIONS:
            if option_key == key:
                return option_label
        return "All"

    def _use_case_compact_tag(self, key: str) -> str:
        return USE_CASE_COMPACT_TAGS.get(key, "ALL")

    def _sort_compact_tag(self, key: str) -> str:
        return SORT_COMPACT_TAGS.get(key, "SCORE")

    def _fit_compact_tag(self, key: str) -> str:
        return FIT_COMPACT_TAGS.get(key, "ALL")

    def _compact_chip_text(self, shown_count: int, total: int) -> str:
        provider_short = "HF" if self.current_filter == "Hugging Face" else "OL"
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

    def _set_use_case_filter(self, key: str) -> None:
        self.use_case_filter = key
        radio_id = f"uc-{key}"
        try:
            radio_button = self.query_one(f"#{radio_id}", RadioButton)
            radio_button.value = True
        except Exception:
            logger.debug("Radio button #{} not found, skipping set", radio_id)

    def action_cycle_use_case(self) -> None:
        keys = [key for key, _label in USE_CASE_OPTIONS]
        next_key = self.search_state.cycle_use_case(keys)
        self._set_use_case_filter(next_key)
        self.refresh_table()
        self.update_status(f"Use Case filter set to {self._use_case_label(next_key)}.")

    def action_cycle_sort_mode(self) -> None:
        keys = [key for key, _label in SORT_OPTIONS]
        next_key = self.search_state.cycle_sort(keys)
        self.refresh_table()
        self.update_status(f"Sort set to {self._sort_label(next_key)}.")

    def action_cycle_fit_filter(self) -> None:
        keys = [key for key, _label in FIT_OPTIONS]
        next_key = self.search_state.cycle_fit(keys)
        self.refresh_table()
        self.update_status(f"Fit filter set to {self._fit_label(next_key)}.")

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()
        self.update_status("Search focused. Type query and press Enter.")

    def action_refresh_search(self) -> None:
        query = self.query_one("#search-input", Input).value.strip()
        if not query:
            self.update_status("Nothing to refresh. Enter a search query first.")
            return
        self.start_search(query)

    def action_prev_page(self) -> None:
        self._go_to_page(self.current_page - 1)

    def action_next_page(self) -> None:
        self._go_to_page(self.current_page + 1)

    def action_cycle_provider(self) -> None:
        cycle = get_provider_filter_labels()
        current = self.current_filter if self.current_filter in cycle else cycle[0]
        next_filter = cycle[(cycle.index(current) + 1) % len(cycle)]
        self.current_filter = next_filter

        current_query = self.query_one("#search-input", Input).value.strip()
        if current_query:
            self.start_search(current_query)
            self.update_status(f"Provider switched to {next_filter}. Searching...")
        else:
            self.refresh_table()
            self.update_status(f"Provider filter set to {next_filter}.")

    def action_toggle_hidden_gems(self) -> None:
        checkbox = self.query_one("#gem-toggle", Checkbox)
        checkbox.value = not checkbox.value
        self.hidden_gems_only = bool(checkbox.value)
        self.refresh_table()
        self.update_status(
            "Hidden gems filter: ON." if self.hidden_gems_only else "Hidden gems filter: OFF."
        )

    def _current_specs_for_search_ui(self):
        if self.latest_specs is not None:
            return self.latest_specs
        cached = cache_db.get_hardware_snapshot()
        if cached is not None:
            self.latest_specs = cached
            return cached
        return {
            "cpu_name": "Detecting hardware",
            "cpu_cores": 0,
            "ram_free": 0.0,
            "ram_total": 0.0,
            "vram_free": 0.0,
            "vram_total": 0.0,
            "gpu_name": "Detecting hardware",
            "has_gpu": False,
        }

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        self.start_search(query)

    def start_search(self, query: str) -> None:
        query = query.strip()
        if not query:
            return
        providers = self._get_search_providers()
        if providers == ["ollama"]:
            self.current_page = 0
        page = self.current_page
        signature = (tuple(providers), query.lower(), page)

        if (
            self._search_inflight_signature == signature
            and (time.monotonic() - self._search_inflight_started_at) < 1.0
        ):
            return

        self._pending_search_payload = (query, providers, page, signature)
        if self._search_debounce_timer is not None:
            with suppress(Exception):
                self._search_debounce_timer.stop()
        self._search_debounce_timer = self.set_timer(
            self._search_debounce_delay,
            self._dispatch_debounced_search,
        )

    def _dispatch_debounced_search(self) -> None:
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

        # Try stale cache as fallback (offline mode)
        stale = self.search_cache.get_stale(query_key)
        if stale:
            self.all_results = [item.copy() for item in stale["results"]]
            self.dl.ensure_download_fields(self.all_results)
            self.last_search_error = "Offline — showing cached results"
            if "has_more_pages" in stale:
                self.has_more_pages = stale["has_more_pages"]
            self.on_search_completed(self.active_search_id)
            self.update_status("Offline mode — showing cached results")
            return

        self.run_search_worker(query, query_key, self.active_search_id, providers)

    def on_search_progress(self, search_id: int, message: str) -> None:
        if search_id != self.active_search_id:
            return
        now = time.monotonic()
        last_search_id, last_message, last_ts = self._search_progress_stamp
        if search_id == last_search_id and message == last_message and (now - last_ts) < 0.2:
            return
        self._search_progress_stamp = (search_id, message, now)
        self.update_status(message)
        if self._search_progress_visible:
            return

        table = self.query_one("#results-table", DataTable)
        self._configure_results_table_columns()
        table.clear()
        row_data = self._blank_result_row()
        row_data["inst"] = "[cyan]...[/cyan]"
        row_data["source"] = "System"
        name_width = max(16, self.results_column_widths.get("name", 24))
        row_data["name"] = self._truncate_cell(f"Status: {message}", max(10, name_width - 1))
        row_data["download"] = self._download_cell_markup("Working")
        table.add_row(*self._row_cells_for_current_layout(row_data), key="search-progress")
        self._update_results_meta(0)
        self._search_progress_visible = True

    def _get_search_providers(self) -> list[str]:
        """Return list of providers to search based on current_filter."""
        return providers_from_filter(self.current_filter)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "use-case-filter":
            pid = event.pressed.id
            self._set_use_case_filter(pid.replace("uc-", "") if pid else "all")
            self.refresh_table()
            return

        pid = event.pressed.id
        if pid == "filter-ollama":
            self.current_filter = "Ollama"
        elif pid == "filter-hf":
            self.current_filter = "Hugging Face"

        current_query = self.query_one("#search-input", Input).value.strip()
        if current_query:
            self.start_search(current_query)
        else:
            self.refresh_table()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id != "gem-toggle":
            return
        self.hidden_gems_only = bool(event.value)
        providers = self._get_search_providers()
        if self.hidden_gems_only and providers == ["ollama"]:
            self.update_status(
                "Hidden gems only available for Hugging Face. Switch provider filter."
            )
        self.refresh_table()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "prev-page":
            self._go_to_page(self.current_page - 1)
        elif button_id == "next-page":
            self._go_to_page(self.current_page + 1)

    def _go_to_page(self, page_num: int) -> None:
        is_valid, message = validate_page_request(page_num, self.max_pages)
        if not is_valid:
            if message:
                self.update_status(message)
            return
        current_query = self.query_one("#search-input", Input).value.strip()
        if not current_query:
            return
        self.current_page = page_num
        self.start_search(current_query)

    def _update_pagination_controls(self) -> None:
        try:
            prev_btn = self.query_one("#prev-page", Button)
            next_btn = self.query_one("#next-page", Button)
            indicator = self.query_one("#page-indicator", Static)
        except Exception as e:
            self.update_status(f"Pagination UI error: {e}")
            return

        providers = self._get_search_providers()
        is_hf = is_hf_provider_selection(providers)

        # Only show pagination controls for HuggingFace (Ollama doesn't support it)
        if is_hf:
            prev_btn.disabled = self.current_page == 0
            next_btn.disabled = not self.has_more_pages
            indicator.update(f"Page {self.current_page + 1}")
            prev_btn.styles.display = "block"
            next_btn.styles.display = "block"
            indicator.styles.display = "block"
        else:
            # Hide pagination for Ollama
            prev_btn.styles.display = "none"
            next_btn.styles.display = "none"
            indicator.styles.display = "none"

        prev_btn.refresh()
        next_btn.refresh()
        indicator.refresh()

    @work(thread=True)
    def run_search_worker(
        self,
        query: str,
        query_key: str,
        search_id: int,
        providers: list[str] | None = None,
    ) -> None:
        from search.search_orchestrator import SearchOrchestrator

        if providers is None:
            providers = self._get_search_providers()

        if search_id != self.search_state.active_id:
            return

        try:
            monitor = self._get_hardware_monitor()
        except (OSError, ValueError, RuntimeError):
            self.call_from_thread(self._on_hardware_search_failed, search_id)
            return
        orchestrator = SearchOrchestrator(
            monitor=monitor,
            hf_provider=self.hf_provider,
            ollama_provider=self.ollama_provider,
            on_progress=lambda sid, msg: self.call_from_thread(
                self.on_search_progress, sid, msg
            ),
            cancel_check=lambda: search_id != self.search_state.active_id,
            on_results=lambda sid, results: self.call_from_thread(
                self.on_search_results_ready, sid, results
            ),
        )

        outcome = orchestrator.search(
            search_id=search_id,
            query=query,
            providers=providers,
            page=self.current_page,
            page_size=self.page_size,
            hf_token=config.settings.hf_token,
            ollama_page_size=config.settings.ollama_search_limit,
        )

        if outcome.cancelled or search_id != self.search_state.active_id:
            return

        self.call_from_thread(
            self._apply_search_outcome, search_id, query_key, outcome, monitor.get_specs()
        )

    def on_search_results_ready(self, search_id: int, results: list[dict]) -> None:
        """Render a ready prefix without completing or caching an active search."""
        if search_id != self.active_search_id or self._search_inflight_signature is None:
            return
        self.all_results = [item.copy() for item in results]
        self.dl.ensure_download_fields(self.all_results)
        self.has_more_pages = False
        self._search_progress_visible = True
        self.query_one("#results-table", DataTable).loading = False
        self._refresh_search_results()
        self.update_status(f"Searching Ollama — {len(results)} results ready...")

    def _apply_search_outcome(self, search_id, query_key, outcome, specs) -> None:
        """Commit only the current completed outcome on the UI thread."""
        if search_id != self.active_search_id:
            return
        self.on_search_progress(search_id, "Finalizing search results...")
        self.all_results = outcome.results
        self.dl.ensure_download_fields(self.all_results)
        self.last_search_error = " | ".join(outcome.errors[:2])
        self.has_more_pages = outcome.has_more_pages

        self.update_status(
            provider_search_status(
                outcome.providers,
                result_count=outcome.result_count,
                has_more_pages=outcome.has_more_pages,
                current_page=self.current_page,
            ),
        )

        self.search_cache.set(
            query_key,
            results=self.all_results,
            error=self.last_search_error,
            has_more_pages=self.has_more_pages,
            specs=specs,
        )
        self.on_search_completed(search_id)

    def _on_hardware_search_failed(self, search_id):
        """Finish the active search without terminating the app after a probe failure."""
        if search_id != self.active_search_id:
            return
        self.all_results = []
        self.has_more_pages = False
        self.last_search_error = "Hardware detection unavailable. Retry the search."
        self.on_search_completed(search_id)

    def on_search_completed(self, search_id: int) -> None:
        if search_id != self.active_search_id:
            return

        self._search_inflight_signature = None
        self._search_inflight_started_at = 0.0
        self._search_progress_visible = False

        table = self.query_one("#results-table", DataTable)
        table.loading = False
        self._refresh_search_results()

        if table.row_count > 0:
            shown = table.row_count
            page_info = page_info_suffix(self.current_page)
            self.update_status(f"{shown} results listed{page_info}.")
        elif self.last_search_error:
            self.update_status(self.last_search_error[:120])
        else:
            providers = self._get_search_providers()
            if providers == ["ollama"]:
                self.update_status("No Ollama results. Try 'Hugging Face' filter for more models.")
            else:
                self.update_status(
                    "No results found. Try a different query or disable Hidden gems."
                )

        if table.row_count == 0 and self.last_search_error:
            row_data = self._blank_result_row()
            row_data["inst"] = "[red]![/red]"
            row_data["source"] = "System"
            name_width = max(16, self.results_column_widths.get("name", 24))
            row_data["name"] = self._truncate_cell(
                f"Search error: {self.last_search_error}",
                max(10, name_width - 1),
            )
            table.add_row(*self._row_cells_for_current_layout(row_data))
        table.focus()
        self._update_pagination_controls()
