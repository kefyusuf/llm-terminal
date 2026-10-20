"""Search flow: filters, debounce, live-first caching, pagination, and workers."""

from __future__ import annotations

import time
from contextlib import suppress

from textual.css.query import NoMatches
from textual.widgets import Button, Input, Select, Static

import config
from app.search_constants import (
    FIT_OPTIONS,
    SORT_OPTIONS,
    USE_CASE_OPTIONS,
    cycle_provider_label,
    option_label,
)
from core.logging_ import get_logger
from providers import get_provider_display_names, get_provider_filter_labels
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

STALE_FALLBACK_ERROR = "Offline — showing cached results"


class SearchController:
    """Own the search lifecycle and filter selection for SearchResultsState."""

    def __init__(self, app, state, *, cache, hf_provider, ollama_provider, debounce_delay=0.12):
        self.app = app
        self.state = state
        self.cache = cache
        self.hf_provider = hf_provider
        self.ollama_provider = ollama_provider
        self.debounce_delay = debounce_delay
        self._debounce_timer = None
        self._stale_query_key: str | None = None
        names = get_provider_display_names()
        self.provider_labels = (names["ollama"], names["huggingface"])
        self._provider_refresh_running = False
        self.use_case_keys = tuple(key for key, _label in USE_CASE_OPTIONS)
        if state.current_filter not in self.provider_labels:
            state.current_filter = self.provider_labels[0]
        if state.use_case_filter not in self.use_case_keys:
            state.set_use_case("all")

    # -- helpers ----------------------------------------------------------

    def providers(self) -> list[str]:
        return providers_from_filter(self.state.current_filter)

    def query_text(self) -> str:
        return self.app.query_one("#search-input", Input).value.strip()

    def _status(self, text) -> None:
        self.app.status.update(text)

    def _set_results(self, items) -> None:
        self.state.results = items
        self.app.dl.ensure_download_fields(self.state.results)

    # -- provider selector ------------------------------------------------

    def request_provider_refresh(self) -> None:
        """Start one background probe without delaying search input or selection."""
        if self._provider_refresh_running:
            return
        self._provider_refresh_running = True
        self.app.run_worker(self._provider_refresh_worker, thread=True, group="providers")

    def _provider_refresh_worker(self) -> None:
        labels = tuple(get_provider_filter_labels())
        self.app.call_from_thread(self.apply_provider_labels, labels)

    def apply_provider_labels(self, labels: tuple[str, ...]) -> None:
        """Update options on the UI thread, preserving the user's active selection."""
        self._provider_refresh_running = False
        current = self.state.current_filter
        if current not in labels:
            labels = (*labels, current)
        if labels == self.provider_labels:
            return
        self.provider_labels = labels
        selector = self.app.query_one("#provider-select", Select)
        with selector.prevent(Select.Changed):
            selector.set_options((label, label) for label in labels)
            selector.value = current

    def set_provider(self, label: str, *, sync_widget: bool) -> None:
        if label not in self.provider_labels or label == self.state.current_filter:
            return
        self.state.current_filter = label
        if sync_widget:
            with suppress(NoMatches):
                selector = self.app.query_one("#provider-select", Select)
                if selector.value != label:
                    selector.value = label
        query = self.query_text()
        if query:
            self.start(query)
            self._status(f"Provider switched to {label}. Searching...")
        else:
            self.app.table.refresh()
            self._status(f"Provider filter set to {label}.")

    def cycle_provider(self) -> None:
        label = cycle_provider_label(self.provider_labels, self.state.current_filter)
        self.set_provider(label, sync_widget=True)

    # -- view filters -----------------------------------------------------

    def set_use_case(self, key: str, *, sync_widget: bool = True) -> None:
        if key not in self.use_case_keys:
            return
        self.state.set_use_case(key)
        if sync_widget:
            with suppress(NoMatches):
                selector = self.app.query_one("#use-case-select", Select)
                if selector.value != key:
                    selector.value = key
        self.app.table.refresh()
        self._status(f"Use Case filter set to {option_label(USE_CASE_OPTIONS, key, 'Any Use')}.")

    def cycle_use_case(self) -> None:
        keys = list(self.use_case_keys)
        current = self.state.use_case_filter if self.state.use_case_filter in keys else "all"
        self.set_use_case(keys[(keys.index(current) + 1) % len(keys)])

    def cycle_sort(self) -> None:
        key = self.state.cycle_sort([key for key, _label in SORT_OPTIONS])
        self.app.table.refresh()
        self._status(f"Sort set to {option_label(SORT_OPTIONS, key, 'Score')}.")

    def cycle_fit(self) -> None:
        key = self.state.cycle_fit([key for key, _label in FIT_OPTIONS])
        self.app.table.refresh()
        self._status(f"Fit filter set to {option_label(FIT_OPTIONS, key, 'All')}.")

    def set_hidden_gems(self, enabled: bool) -> None:
        self.state.hidden_gems_only = enabled
        if enabled and self.providers() == ["ollama"]:
            self._status("Hidden gems only available for Hugging Face. Switch provider filter.")
        self.app.table.refresh()

    # -- pagination -------------------------------------------------------

    def refresh_current(self) -> None:
        query = self.query_text()
        if not query:
            self._status("Nothing to refresh. Enter a search query first.")
            return
        self.start(query)

    def go_to_page(self, page_num: int) -> None:
        is_valid, message = validate_page_request(page_num, self.state.max_pages)
        if not is_valid:
            if message:
                self._status(message)
            return
        query = self.query_text()
        if not query:
            return
        self.state.current_page = page_num
        self.start(query)

    def update_pagination_controls(self) -> None:
        try:
            prev_btn = self.app.query_one("#prev-page", Button)
            next_btn = self.app.query_one("#next-page", Button)
            indicator = self.app.query_one("#page-indicator", Static)
        except Exception as exc:
            self._status(f"Pagination UI error: {exc}")
            return

        # Only Hugging Face supports paging.
        if is_hf_provider_selection(self.providers()):
            prev_btn.disabled = self.state.current_page == 0
            next_btn.disabled = not self.state.has_more_pages
            indicator.update(f"Page {self.state.current_page + 1}")
            display = "block"
        else:
            display = "none"
        for widget in (prev_btn, next_btn, indicator):
            widget.styles.display = display
            widget.refresh()

    # -- search lifecycle -------------------------------------------------

    def start(self, query: str) -> None:
        query = query.strip()
        if not query:
            return
        providers = self.providers()
        if providers == ["ollama"]:
            self.state.current_page = 0
        page = self.state.current_page
        signature = (tuple(providers), query.lower(), page)
        if self.state.is_stale_signature(signature, time.monotonic()):
            return

        self.state.pending_payload = (query, providers, page, signature)
        if self._debounce_timer is not None:
            with suppress(Exception):
                self._debounce_timer.stop()
        self._debounce_timer = self.app.set_timer(self.debounce_delay, self.dispatch_debounced)

    def dispatch_debounced(self) -> None:
        """Use fresh cache immediately, but reserve stale data for post-live fallback."""
        self._debounce_timer = None
        payload = self.state.pending_payload
        self.state.pending_payload = None
        if payload is None:
            return

        query, providers, page, signature = payload
        state = self.state
        state.current_page = page
        search_id = state.begin_inflight(signature, time.monotonic())
        query_key = build_query_key(providers, query, page)
        current_specs = self.app.status.specs_for_search()
        self.app.table.begin_loading()
        self.on_progress(search_id, f"Searching {provider_display_name(providers)}: {query}")

        self._stale_query_key = None
        cached = self.cache.get(query_key, current_specs)
        if cached:
            self._set_results([item.copy() for item in cached["results"]])
            state.last_error = cached["error"]
            if "has_more_pages" in cached:
                state.has_more_pages = cached["has_more_pages"]
            self.on_completed(search_id)
            self._status(f"Loaded{cache_hit_suffix(providers, page)}")
            return

        self._stale_query_key = query_key
        self.app.run_worker(
            lambda: self._search_worker(query, query_key, search_id, providers),
            thread=True,
            group="search",
        )

    def on_progress(self, search_id: int, message: str) -> None:
        state = self.state
        if search_id != state.active_id:
            return
        now = time.monotonic()
        last_id, last_message, last_ts = state.progress_stamp
        if search_id == last_id and message == last_message and (now - last_ts) < 0.2:
            return
        state.progress_stamp = (search_id, message, now)
        self._status(message)
        if state.progress_visible:
            return
        self.app.table.show_progress_row(message)
        state.progress_visible = True

    def _search_worker(self, query, query_key, search_id, providers) -> None:
        # Resolved at call time so tests can substitute the orchestrator.
        from search.search_orchestrator import SearchOrchestrator

        if search_id != self.state.active_id:
            return
        try:
            monitor = self.app.status.hardware_monitor()
        except (OSError, ValueError, RuntimeError):
            self.app.call_from_thread(self.on_hardware_failed, search_id)
            return
        orchestrator = SearchOrchestrator(
            monitor=monitor,
            hf_provider=self.hf_provider,
            ollama_provider=self.ollama_provider,
            on_progress=lambda sid, msg: self.app.call_from_thread(self.on_progress, sid, msg),
            cancel_check=lambda: search_id != self.state.active_id,
            on_results=lambda sid, results: self.app.call_from_thread(
                self.on_results_ready, sid, results
            ),
        )
        outcome = orchestrator.search(
            search_id=search_id,
            query=query,
            providers=providers,
            page=self.state.current_page,
            page_size=self.state.page_size,
            hf_token=config.settings.hf_token,
            ollama_page_size=config.settings.ollama_search_limit,
        )
        if outcome.cancelled or search_id != self.state.active_id:
            return
        self.app.call_from_thread(
            self.apply_outcome, search_id, query_key, outcome, monitor.get_specs()
        )

    def on_results_ready(self, search_id: int, results: list[dict]) -> None:
        """Render a ready prefix without completing or caching an active search."""
        state = self.state
        if search_id != state.active_id or state.inflight_signature is None:
            return
        self._set_results([item.copy() for item in results])
        state.has_more_pages = False
        state.progress_visible = True
        self.app.table.set_loading(False)
        self.app.table.refresh_preserving_selection()
        self._status(f"Searching Ollama — {len(results)} results ready...")

    def apply_outcome(self, search_id, query_key, outcome, specs) -> None:
        """Commit only the current completed outcome on the UI thread."""
        state = self.state
        if search_id != state.active_id:
            return
        self.on_progress(search_id, "Finalizing search results...")
        self._set_results(outcome.results)
        state.last_error = " | ".join(outcome.errors[:2])
        state.has_more_pages = outcome.has_more_pages
        self._status(
            provider_search_status(
                outcome.providers,
                result_count=outcome.result_count,
                has_more_pages=outcome.has_more_pages,
                current_page=state.current_page,
            ),
        )
        self.cache.set(
            query_key,
            results=state.results,
            error=state.last_error,
            has_more_pages=state.has_more_pages,
            specs=specs,
        )
        self.on_completed(search_id)

    def on_hardware_failed(self, search_id) -> None:
        """Finish the active search without terminating the app after a probe failure."""
        if search_id != self.state.active_id:
            return
        self.state.results = []
        self.state.has_more_pages = False
        self.state.last_error = "Hardware detection unavailable. Retry the search."
        self.on_completed(search_id)

    def _try_stale_fallback(self, search_id: int) -> bool:
        """Use retained stale data only when an attempted live search failed empty."""
        stale_key, self._stale_query_key = self._stale_query_key, None
        state = self.state
        if stale_key is None or state.results or not state.last_error:
            return False
        stale = self.cache.get_stale(stale_key)
        if not stale or not stale.get("results"):
            return False
        self._set_results([item.copy() for item in stale["results"]])
        state.last_error = STALE_FALLBACK_ERROR
        if "has_more_pages" in stale:
            state.has_more_pages = stale["has_more_pages"]
        self._finish(search_id)
        self._status("Offline mode — showing cached results")
        return True

    def on_completed(self, search_id: int) -> None:
        if search_id != self.state.active_id:
            return
        if self._try_stale_fallback(search_id):
            return
        self._finish(search_id)

    def _finish(self, search_id: int) -> None:
        state = self.state
        table = self.app.table
        state.end_inflight()
        table.set_loading(False)
        table.refresh_preserving_selection()

        shown = table.row_count()
        if shown > 0:
            self._status(f"{shown} results listed{page_info_suffix(state.current_page)}.")
        elif state.last_error:
            self._status(state.last_error[:120])
            table.show_error_row(state.last_error)
        elif self.providers() == ["ollama"]:
            self._status("No Ollama results. Try 'Hugging Face' filter for more models.")
        else:
            self._status("No results found. Try a different query or disable Hidden gems.")
        table.focus()
        self.update_pagination_controls()
