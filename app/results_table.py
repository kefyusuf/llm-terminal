"""Results-table layout, cell presentation, and selection."""

from __future__ import annotations

from textual.widgets import DataTable, Static

from app.search_constants import (
    FIT_COMPACT_TAGS,
    RESULTS_COLUMN_LABELS_COMFORTABLE,
    RESULTS_COLUMN_LABELS_COMPACT,
    SORT_COMPACT_TAGS,
    USE_CASE_COMPACT_TAGS,
    provider_compact_tag,
)
from core.logging_ import get_logger
from results.results_layout import column_keys_for_width, compute_column_widths
from results.results_presenter import (
    download_cell_markup,
    fit_cell_markup,
    installed_cell_markup,
    mode_cell_markup,
    score_cell_markup,
    source_cell_markup,
    use_case_cell_markup,
)
from results.results_text import (
    align_plain_cell,
    blank_result_row,
    format_header_label,
    truncate_cell,
    truncate_plain_cell,
)
from results.results_view import filter_results_for_view, result_unique_key

logger = get_logger(__name__)

# Fields whose change requires a full row rebuild instead of a download-cell update.
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


class ResultsTableView:
    """Render SearchResultsState into the #results-table DataTable."""

    def __init__(self, app, state):
        self.app = app
        self.state = state
        self._render_signature = None

    # -- layout -----------------------------------------------------------

    def _table(self) -> DataTable:
        return self.app.query_one("#results-table", DataTable)

    def _width(self, key: str, default: int) -> int:
        return self.state.column_widths.get(key, default)

    def _column_labels(self):
        if self.app.layout.compact_mode:
            return RESULTS_COLUMN_LABELS_COMPACT
        return RESULTS_COLUMN_LABELS_COMFORTABLE

    def configure_columns(self, force: bool = False, refresh_rows: bool = False) -> None:
        table = self._table()
        compact = self.app.layout.compact_mode
        available_width = table.scrollable_content_region.width or max(1, self.app.size.width - 4)
        next_keys = column_keys_for_width(available_width, compact=compact)
        base_widths = compute_column_widths(next_keys, available_width, compact=compact)

        if (
            not force
            and next_keys == self.state.column_keys
            and base_widths == self.state.column_widths
        ):
            return
        if refresh_rows:
            self.refresh()
            return

        labels = self._column_labels()
        table.clear(columns=True)
        self.state.table_row_keys = set()
        for key in next_keys:
            width = base_widths[key]
            table.add_column(format_header_label(labels[key], width), width=width, key=key)
        self.state.column_keys = next_keys
        self.state.column_widths = {key: base_widths[key] for key in next_keys}

    # -- cells ------------------------------------------------------------

    def _markup(self, render, text, width):
        return render(
            text,
            width=width,
            truncate_plain=truncate_plain_cell,
            align_plain=align_plain_cell,
        )

    def download_cell_markup(self, download_text, width=None):
        if width is None:
            width = max(3, self._width("download", 4) - 1)
        return self._markup(download_cell_markup, download_text, width)

    def row_cells(self, row_data):
        return [row_data.get(key, "-") for key in self.state.column_keys]

    def _name_cell(self, text):
        name_width = max(16, self._width("name", 24))
        return truncate_cell(text, max(10, name_width - 1))

    def _result_row(self, result) -> dict:
        w = self._width
        publisher = truncate_plain_cell(result.get("publisher", "-"), max(5, w("publisher", 8) - 1))
        use_case = truncate_plain_cell(result.get("use_case", "-"), max(4, w("use_case", 5) - 1))
        params = truncate_plain_cell(result.get("params", "-"), max(4, w("params", 6) - 1))
        quant = truncate_plain_cell(result.get("quant", "-"), max(4, w("quant", 6) - 1))
        mode_plain = truncate_plain_cell(result.get("mode", "-"), max(5, w("mode", 8) - 1))
        fit_plain = truncate_plain_cell(result.get("fit", "-"), max(7, w("fit", 8)))
        params = align_plain_cell(params, max(4, w("params", 6) - 1), "left")
        quant = align_plain_cell(quant, max(4, w("quant", 6) - 1), "left")

        row = blank_result_row()
        row["inst"] = self._markup(
            installed_cell_markup, result.get("inst", "-"), max(3, w("inst", 7) - 2)
        )
        row["source"] = self._markup(
            source_cell_markup, result.get("source", "-"), max(10, w("source", 13) - 1)
        )
        publisher_cell = align_plain_cell(publisher, max(6, w("publisher", 8) - 1), "left")
        row["publisher"] = f"[#7d8bad]{publisher_cell}[/#7d8bad]"
        row["name"] = self._name_cell(result["name"])
        row["params"] = params
        row["use_case"] = self._markup(use_case_cell_markup, use_case, max(4, w("use_case", 6) - 1))
        row["score"] = self._markup(
            score_cell_markup, result.get("score", "-"), max(5, w("score", 8) - 1)
        )
        quant_cell = align_plain_cell(quant, max(6, w("quant", 9) - 1), "left")
        row["quant"] = f"[#7e90bf]{quant_cell}[/#7e90bf]"
        row["mode"] = self._markup(mode_cell_markup, mode_plain, max(5, w("mode", 8) - 1))
        row["fit"] = self._markup(fit_cell_markup, fit_plain, max(7, w("fit", 8)))
        row["download"] = self.download_cell_markup(self.app.dl.download_cell_text(result))
        return row

    # -- meta and status rows ---------------------------------------------

    def compact_chip_text(self, shown_count: int, total: int) -> str:
        state = self.state
        provider_short = provider_compact_tag(state.current_filter)
        use_case_label = USE_CASE_COMPACT_TAGS.get(state.use_case_filter, "ALL")
        sort_label = SORT_COMPACT_TAGS.get(state.sort_mode, "SCORE")
        fit_label = FIT_COMPACT_TAGS.get(state.fit_filter, "ALL")
        gems_label = "ON" if state.hidden_gems_only else "OFF"
        page_label = str(state.current_page + 1) if state.current_filter == "Hugging Face" else "1"
        return (
            f"[#8ea3cf]M:[/#8ea3cf][#dbe7ff]{shown_count}/{total}[/#dbe7ff]  "
            f"[#8ea3cf]P:[/#8ea3cf][#9fe8ff]{provider_short}[/#9fe8ff]  "
            f"[#8ea3cf]U:[/#8ea3cf][#d1b3ff]{use_case_label}[/#d1b3ff]  "
            f"[#8ea3cf]S:[/#8ea3cf][#7edfff]{sort_label}[/#7edfff]  "
            f"[#8ea3cf]F:[/#8ea3cf][#f2c46d]{fit_label}[/#f2c46d]  "
            f"[#8ea3cf]G:[/#8ea3cf][#4fe08a]{gems_label}[/#4fe08a]  "
            f"[#8ea3cf]Pg:[/#8ea3cf][#dbe7ff]{page_label}[/#dbe7ff]"
        )

    def update_meta(self, shown_count) -> None:
        total = len(self.state.results)
        try:
            results_meta = self.app.query_one("#results-meta", Static)
            compact_chipbar = self.app.query_one("#compact-chipbar", Static)
        except Exception:
            logger.debug("Results meta widgets not yet mounted, skipping update")
            return
        results_meta.update(f"Models ({shown_count} shown / {total} total)")
        if not self.app.layout.compact_mode:
            compact_chipbar.update("")
            return
        compact_chipbar.update(self.compact_chip_text(shown_count, total))

    def begin_loading(self) -> None:
        table = self._table()
        table.clear()
        self.state.table_row_keys = set()
        table.loading = True
        self.update_meta(0)

    def set_loading(self, loading: bool) -> None:
        self._table().loading = loading

    def show_progress_row(self, message: str) -> None:
        table = self._table()
        self.configure_columns()
        table.clear()
        row = blank_result_row()
        row["inst"] = "[cyan]...[/cyan]"
        row["source"] = "System"
        row["name"] = self._name_cell(f"Status: {message}")
        row["download"] = self.download_cell_markup("Working")
        table.add_row(*self.row_cells(row), key="search-progress")
        self.update_meta(0)

    def show_error_row(self, error: str) -> None:
        row = blank_result_row()
        row["inst"] = "[red]![/red]"
        row["source"] = "System"
        row["name"] = self._name_cell(f"Search error: {error}")
        self._table().add_row(*self.row_cells(row))

    def row_count(self) -> int:
        return self._table().row_count

    def focus(self) -> None:
        self._table().focus()

    # -- rendering --------------------------------------------------------

    def _filtered_results(self):
        state = self.state
        return filter_results_for_view(
            state.results,
            current_filter=state.current_filter,
            use_case_filter=state.use_case_filter,
            hidden_gems_only=state.hidden_gems_only,
            sort_mode=state.sort_mode,
            fit_filter=state.fit_filter,
        )

    def _signature(self):
        """Capture ordered, non-download content that determines the rendered table."""
        seen: set[str] = set()
        rows = []
        for result in self._filtered_results():
            key = result_unique_key(result)
            if key in seen:
                continue
            seen.add(key)
            rows.append((key, *(str(result.get(field, "")) for field in _RENDERED_RESULT_FIELDS)))
        columns = (
            tuple(self.state.column_keys),
            tuple(sorted(self.state.column_widths.items())),
        )
        return columns, tuple(rows)

    def refresh_preserving_selection(self) -> None:
        """Keep selection on the same model when newly ready rows sort ahead."""
        table = self._table()
        selected = None
        if 0 <= table.cursor_row < table.row_count:
            selected = table.ordered_rows[table.cursor_row].key
        self.refresh()
        if selected is not None and selected in table.rows:
            table.move_cursor(row=table.get_row_index(selected), animate=False, scroll=False)

    def refresh(self) -> None:
        """Rebuild rows, or update only download cells while row content is unchanged."""
        try:
            table = self._table()
            prev_cursor_row = table.cursor_row
            prev_scroll_x = table.scroll_x
            prev_scroll_y = table.scroll_y
            self.configure_columns()
        except Exception:
            return

        signature = self._signature()
        if self._render_signature is not None and signature != self._render_signature:
            self.state.table_row_keys = set()
        self._render_signature = signature

        filtered_results = self._filtered_results()
        if self.state.table_row_keys and self.state.results:
            new_keys = {result_unique_key(r) for r in filtered_results}
            if self.state.table_row_keys == new_keys:
                self._update_download_cells(table, filtered_results)
                return

        table.clear()
        added = set()
        for result in filtered_results:
            unique_key = result_unique_key(result)
            if unique_key in added:
                continue
            added.add(unique_key)
            table.add_row(*self.row_cells(self._result_row(result)), key=unique_key)

        self.state.table_row_keys = added
        self.update_meta(table.row_count)
        if table.row_count > 0:
            restored_row = max(0, min(prev_cursor_row, table.row_count - 1))
            table.move_cursor(row=restored_row, animate=False, scroll=False)
            table.scroll_to(x=prev_scroll_x, y=prev_scroll_y, animate=False, immediate=True)

    def _update_download_cells(self, table, results) -> None:
        for result in results:
            key = result_unique_key(result)
            markup = self.download_cell_markup(self.app.dl.download_cell_text(result))
            try:
                if table.get_cell(key, "download") != markup:
                    table.update_cell(key, "download", markup)
            except Exception:
                pass

    # -- selection --------------------------------------------------------

    def selected_model(self) -> dict | None:
        """Resolve the selected result through Textual's stable DataTable row key."""
        table = self._table()
        if table.cursor_row < 0 or table.cursor_row >= table.row_count:
            return None
        row_key, _column_key = table.coordinate_to_cell_key(table.cursor_coordinate)
        return self.model_for_key(row_key.value)

    def model_for_key(self, key) -> dict | None:
        if key is None:
            return None
        return next(
            (item for item in self.state.results if result_unique_key(item) == str(key)),
            None,
        )
