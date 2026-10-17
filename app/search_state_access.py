"""Legacy viewer attribute access backed by SearchResultsState."""

from typing import ClassVar


class SearchStateAccessMixin:
    """Keep existing viewer fields mapped to the shared search-state container."""

    _SEARCH_STATE_ATTRS = frozenset(
        {
            "all_results",
            "current_filter",
            "use_case_filter",
            "sort_mode",
            "fit_filter",
            "hidden_gems_only",
            "current_page",
            "page_size",
            "max_pages",
            "total_results",
            "has_more_pages",
            "results_column_keys",
            "results_column_widths",
            "_table_row_keys",
            "_search_inflight_signature",
            "_search_inflight_started_at",
            "_search_progress_stamp",
            "_search_progress_visible",
            "_pending_search_payload",
            "last_search_error",
            "search_counter",
            "active_search_id",
        }
    )

    _SEARCH_STATE_TO_FIELD: ClassVar[dict[str, str]] = {
        "all_results": "results",
        "results_column_keys": "column_keys",
        "results_column_widths": "column_widths",
        "_table_row_keys": "table_row_keys",
        "_search_inflight_signature": "inflight_signature",
        "_search_inflight_started_at": "inflight_started_at",
        "_search_progress_stamp": "progress_stamp",
        "_search_progress_visible": "progress_visible",
        "_pending_search_payload": "pending_payload",
        "last_search_error": "last_error",
        "search_counter": "counter",
        "active_search_id": "active_id",
    }

    _FIELD_TO_LEGACY: ClassVar[dict[str, str]] = {
        v: k for k, v in _SEARCH_STATE_TO_FIELD.items()
    }

    def __getattr__(self, name: str):
        if name in self._SEARCH_STATE_ATTRS:
            field = self._SEARCH_STATE_TO_FIELD.get(name, name)
            return getattr(self.search_state, field)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self._SEARCH_STATE_ATTRS:
            field = self._SEARCH_STATE_TO_FIELD.get(name, name)
            setattr(self.search_state, field, value)
            return
        super().__setattr__(name, value)
