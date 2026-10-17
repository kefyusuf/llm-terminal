"""Results-table layout, cell presentation, filtering, and selection."""

from textual.widgets import (
    DataTable,
    Static,
)

from app.search_constants import (
    RESULTS_COLUMN_LABELS_COMFORTABLE,
    RESULTS_COLUMN_LABELS_COMPACT,
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


class ResultsTableMixin:
    """Results-table layout, cell presentation, filtering, and selection.

    Methods use the viewer's shared state and Textual APIs, preserving subclass hooks.
    """

    def _column_labels(self):
        if self.compact_mode:
            return RESULTS_COLUMN_LABELS_COMPACT
        return RESULTS_COLUMN_LABELS_COMFORTABLE

    def _get_selected_model(self) -> dict | None:
        """Get the model dict at the current cursor position."""
        table = self.query_one("#results-table", DataTable)
        cursor_row = table.cursor_row
        if cursor_row < 0 or cursor_row >= table.row_count:
            return None
        try:
            row_key = table.get_row_at(cursor_row)
            if not row_key:
                return None
            row_key_str = str(row_key[0])
            return next(
                (item for item in self.all_results if result_unique_key(item) == row_key_str),
                None,
            )
        except Exception:
            logger.debug("Failed to get selected model from cursor position")
            return None

    def _configure_results_table_columns(self, force: bool = False, refresh_rows: bool = False):
        table = self.query_one("#results-table", DataTable)
        available_width = table.scrollable_content_region.width or max(1, self.size.width - 4)
        next_keys = column_keys_for_width(available_width, compact=self.compact_mode)
        base_widths = compute_column_widths(
            next_keys,
            available_width,
            compact=self.compact_mode,
        )
        labels = self._column_labels()

        if (
            not force
            and next_keys == self.results_column_keys
            and base_widths == self.results_column_widths
        ):
            return

        if refresh_rows:
            self.refresh_table()
            return

        table.clear(columns=True)
        self._table_row_keys = set()
        for key in next_keys:
            width = base_widths[key]
            table.add_column(
                self._format_header_label(labels[key], width),
                width=width,
                key=key,
            )

        self.results_column_keys = next_keys
        self.results_column_widths = {key: base_widths[key] for key in next_keys}

    def _truncate_cell(self, value, max_len):
        return truncate_cell(value, max_len)

    def _truncate_plain_cell(self, value, max_len):
        return truncate_plain_cell(value, max_len)

    def _format_header_label(self, label, width):
        return format_header_label(label, width)

    def _align_plain_cell(self, value, width, align="left"):
        return align_plain_cell(value, width, align)

    def _blank_result_row(self):
        return blank_result_row()

    def _fit_cell_markup(self, fit_text):
        return fit_cell_markup(
            fit_text,
            width=max(7, self.results_column_widths.get("fit", 8)),
            truncate_plain=self._truncate_plain_cell,
            align_plain=self._align_plain_cell,
        )

    def _mode_cell_markup(self, mode_text):
        return mode_cell_markup(
            mode_text,
            width=max(5, self.results_column_widths.get("mode", 8) - 1),
            truncate_plain=self._truncate_plain_cell,
            align_plain=self._align_plain_cell,
        )

    def _installed_cell_markup(self, installed_text):
        return installed_cell_markup(
            installed_text,
            width=max(3, self.results_column_widths.get("inst", 7) - 2),
            truncate_plain=self._truncate_plain_cell,
            align_plain=self._align_plain_cell,
        )

    def _source_cell_markup(self, source_text):
        return source_cell_markup(
            source_text,
            width=max(10, self.results_column_widths.get("source", 13) - 1),
            truncate_plain=self._truncate_plain_cell,
            align_plain=self._align_plain_cell,
        )

    def _score_cell_markup(self, score_text):
        return score_cell_markup(
            score_text,
            width=max(5, self.results_column_widths.get("score", 8) - 1),
            truncate_plain=self._truncate_plain_cell,
            align_plain=self._align_plain_cell,
        )

    def _use_case_cell_markup(self, use_case_text):
        return use_case_cell_markup(
            use_case_text,
            width=max(4, self.results_column_widths.get("use_case", 6) - 1),
            truncate_plain=self._truncate_plain_cell,
            align_plain=self._align_plain_cell,
        )

    def _download_cell_markup(self, download_text, width=None):
        if width is None:
            width = max(3, self.results_column_widths.get("download", 4) - 1)
        return download_cell_markup(
            download_text,
            width=width,
            truncate_plain=self._truncate_plain_cell,
            align_plain=self._align_plain_cell,
        )

    def _row_cells_for_current_layout(self, row_data):
        return [row_data.get(key, "-") for key in self.results_column_keys]

    def _update_results_meta(self, shown_count):
        total = len(self.all_results)
        try:
            results_meta = self.query_one("#results-meta", Static)
            compact_chipbar = self.query_one("#compact-chipbar", Static)
        except Exception:
            logger.debug("Results meta widgets not yet mounted, skipping update")
            return

        results_meta.update(f"Models ({shown_count} shown / {total} total)")

        if not self.compact_mode:
            compact_chipbar.update("")
            return

        compact_chipbar.update(self._compact_chip_text(shown_count, total))

    def _refresh_search_results(self) -> None:
        """Keep selection on the same model when newly ready rows sort ahead."""
        table = self.query_one("#results-table", DataTable)
        selected = None
        if 0 <= table.cursor_row < table.row_count:
            selected = table.ordered_rows[table.cursor_row].key
        self.refresh_table()
        if selected is not None and selected in table.rows:
            table.move_cursor(row=table.get_row_index(selected), animate=False, scroll=False)

    def refresh_table(self):
        try:
            table = self.query_one("#results-table", DataTable)
            prev_cursor_row = table.cursor_row
            prev_scroll_x = table.scroll_x
            prev_scroll_y = table.scroll_y
            self._configure_results_table_columns()
        except Exception:
            return

        # Incremental path: only update download column when row keys match
        if self._table_row_keys and len(self.all_results) > 0:
            filtered_results = filter_results_for_view(
                self.all_results,
                current_filter=self.current_filter,
                use_case_filter=self.use_case_filter,
                hidden_gems_only=self.hidden_gems_only,
                sort_mode=self.sort_mode,
                fit_filter=self.fit_filter,
            )
            new_keys = {result_unique_key(r) for r in filtered_results}
            if self._table_row_keys == new_keys:
                for result in filtered_results:
                    key = result_unique_key(result)
                    download_text = self._download_cell_text(result)
                    dl_width = max(3, self.results_column_widths.get("download", 4) - 1)
                    download_markup = self._download_cell_markup(download_text, dl_width)
                    try:
                        if table.get_cell(key, "download") != download_markup:
                            table.update_cell(key, "download", download_markup)
                    except Exception:
                        pass
                return
            # Fall through to full rebuild

        table.clear()
        added = set()

        filtered_results = filter_results_for_view(
            self.all_results,
            current_filter=self.current_filter,
            use_case_filter=self.use_case_filter,
            hidden_gems_only=self.hidden_gems_only,
            sort_mode=self.sort_mode,
            fit_filter=self.fit_filter,
        )

        for result in filtered_results:
            display_name = result["name"]
            name_width = max(16, self.results_column_widths.get("name", 24))
            display_name = self._truncate_cell(display_name, max(10, name_width - 1))

            publisher = self._truncate_plain_cell(
                result.get("publisher", "-"),
                max(5, self.results_column_widths.get("publisher", 8) - 1),
            )
            use_case = self._truncate_plain_cell(
                result.get("use_case", "-"),
                max(4, self.results_column_widths.get("use_case", 5) - 1),
            )
            params = self._truncate_plain_cell(
                result.get("params", "-"),
                max(4, self.results_column_widths.get("params", 6) - 1),
            )
            quant = self._truncate_plain_cell(
                result.get("quant", "-"),
                max(4, self.results_column_widths.get("quant", 6) - 1),
            )
            mode_plain = self._truncate_plain_cell(
                result.get("mode", "-"),
                max(5, self.results_column_widths.get("mode", 8) - 1),
            )
            fit_plain = self._truncate_plain_cell(
                result.get("fit", "-"),
                max(7, self.results_column_widths.get("fit", 8)),
            )
            score = result.get("score", "-")
            download = self._download_cell_text(result)
            download_width = max(3, self.results_column_widths.get("download", 4) - 1)
            mode = self._mode_cell_markup(mode_plain)
            fit = self._fit_cell_markup(fit_plain)
            params = self._align_plain_cell(
                params,
                max(4, self.results_column_widths.get("params", 6) - 1),
                "left",
            )
            quant = self._align_plain_cell(
                quant,
                max(4, self.results_column_widths.get("quant", 6) - 1),
                "left",
            )
            download = self._download_cell_markup(download, download_width)

            unique_key = result_unique_key(result)
            if unique_key in added:
                continue
            added.add(unique_key)

            row_data = self._blank_result_row()
            row_data["inst"] = self._installed_cell_markup(result.get("inst", "-"))
            row_data["source"] = self._source_cell_markup(result.get("source", "-"))
            row_data["publisher"] = (
                f"[#7d8bad]{self._align_plain_cell(publisher, max(6, self.results_column_widths.get('publisher', 8) - 1), 'left')}[/#7d8bad]"
            )
            row_data["name"] = display_name
            row_data["params"] = params
            row_data["use_case"] = self._use_case_cell_markup(use_case)
            row_data["score"] = self._score_cell_markup(score)
            row_data["quant"] = (
                f"[#7e90bf]{self._align_plain_cell(quant, max(6, self.results_column_widths.get('quant', 9) - 1), 'left')}[/#7e90bf]"
            )
            row_data["mode"] = mode
            row_data["fit"] = fit
            row_data["download"] = download
            table.add_row(*self._row_cells_for_current_layout(row_data), key=unique_key)

        self._table_row_keys = added
        self._update_results_meta(table.row_count)

        if table.row_count > 0:
            restored_row = max(0, min(prev_cursor_row, table.row_count - 1))
            table.move_cursor(row=restored_row, animate=False, scroll=False)
            table.scroll_to(
                x=prev_scroll_x,
                y=prev_scroll_y,
                animate=False,
                immediate=True,
            )
