"""Viewer styling, view modes, themes, and debounced resize handling."""

from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Checkbox,
    DataTable,
    Footer,
    Input,
    Label,
    Static,
)

from core.logging_ import get_logger
from terminal_ui.themes import THEMES, next_theme, theme_css

logger = get_logger(__name__)


VIEWER_CSS = """
    Screen { layout: vertical; padding: 1; background: #090c14; }
    SystemInfoWidget {
        height: 3;
        border: round #3a4666;
        background: #101728;
        color: #d5def0;
        padding: 0 1;
        content-align: center middle;
        margin-bottom: 1;
    }
    Input {
        width: 100%;
        margin-bottom: 0;
        background: #1b2235;
        color: #dbe7ff;
        border: round #3a4666;
    }
    #search-filters-row { height: 3; margin-bottom: 1; }
    #search-panel { width: 34%; min-width: 28; margin-right: 1; height: 3; }
    #provider-panel { width: 22; margin-right: 1; height: 3; }
    #use-case-panel { width: 1fr; height: 3; }
    #search-input { height: 3; }
    #provider-select, #use-case-select { width: 100%; height: 3; }
    #gem-toggle {
        margin-top: 1;
        margin-bottom: 1;
        height: auto;
        padding: 0 1;
        background: transparent;
        border: none;
        color: #95a8d6;
    }
    #results-meta {
        margin-bottom: 1;
        color: #a9b7dc;
        text-style: bold;
    }
    #compact-chipbar {
        height: 1;
        margin-bottom: 1;
        color: #8ea3cf;
        text-style: bold;
    }
    #results-table {
        scrollbar-gutter: stable;
        height: 2fr;
        min-height: 15;
        border: round #3a4666;
        background: #161d31;
        color: #dde4f8;
        overflow-x: hidden;
    }
    #results-table .datatable--header {
        overflow-x: hidden;
    }
    #results-table .datatable--header:first-of-type {
        min-width: 8;
        padding-left: 1;
    }
    #pagination-controls {
        height: 3;
        margin-top: 1;
        margin-bottom: 1;
        align: center middle;
        width: 100%;
    }
    #pagination-controls Button {
        margin: 0 5;
        min-width: 15;
        width: auto;
    }
    #page-indicator {
        color: #9fe8ff;
        text-style: bold;
        padding: 0 5;
        min-width: 30;
        text-align: center;
        content-align: center middle;
        width: auto;
    }
    .hidden {
        display: none;
    }
    .datatable--header { background: #2b3754; color: #9fe8ff; text-style: bold; }
    .datatable--cursor { background: #425071; color: #ffffff; text-style: bold; }
    .datatable--fixed-cursor { background: #425071; color: #ffffff; text-style: bold; }
    .datatable--odd-row { background: #151c30; }
    .datatable--even-row { background: #1c2540; }
    #downloads-label { margin-top: 1; color: #c6d3f2; }
    #downloads-debug { color: #7f8fb8; margin-bottom: 1; }
    #download-history-table {
        height: 6;
        min-height: 4;
        border: round #3a4666;
        background: #161d31;
        color: #d3dcf4;
    }
    #download-history-table .datatable--cursor {
        background: #2d3748;
    }
    .action-cancel {
        color: #dd8448;
        text-style: bold;
    }
    .action-delete {
        color: #c05621;
        text-style: bold;
    }
    #status-bar { height: 1; color: #8ea3cf; margin-top: 1; }
    Footer { background: #22324d; color: #b9c9ec; }
    """


class LayoutController:
    """Own view mode, color theme, and debounced resize reflow."""

    def __init__(self, app, *, ui_mode: str, theme: str):
        self.app = app
        self.ui_mode = ui_mode
        self.compact_mode = ui_mode == "compact"
        self.color_theme = theme
        self._resize_timer = None
        self._resize_generation = 0

    def apply_ui_mode(self) -> None:
        app = self.app
        try:
            results_table = app.query_one("#results-table", DataTable)
            downloads_label = app.query_one("#downloads-label", Label)
            downloads_debug = app.query_one("#downloads-debug", Static)
            download_table = app.query_one("#download-history-table", DataTable)
            gem_toggle = app.query_one("#gem-toggle", Checkbox)
            compact_chipbar = app.query_one("#compact-chipbar", Static)
            search_input = app.query_one("#search-input", Input)
            search_row = app.query_one("#search-filters-row", Horizontal)
            search_panel = app.query_one("#search-panel", Vertical)
            pagination_controls = app.query_one("#pagination-controls", Horizontal)
            results_meta = app.query_one("#results-meta", Static)
            footer = app.query_one(Footer)
        except Exception:
            logger.debug("UI mode widgets not yet mounted, skipping apply")
            return

        compact = self.compact_mode
        comfortable_display = "none" if compact else "block"
        results_table.styles.height = "3fr" if compact else "2fr"
        gem_toggle.styles.display = comfortable_display
        compact_chipbar.styles.display = "block" if compact else "none"
        for widget in (
            results_meta,
            pagination_controls,
            downloads_label,
            downloads_debug,
            download_table,
            footer,
        ):
            widget.styles.display = comfortable_display
        search_row.styles.height = "3"
        search_panel.styles.width = "30%" if compact else "34%"
        search_panel.styles.height = "3"
        search_input.styles.height = "3"
        search_input.placeholder = (
            "Press / to search..."
            if compact
            else "Press / to search Ollama models (e.g., qwen, llama)"
        )
        app.table.update_meta(results_table.row_count)

    def toggle_view_mode(self) -> None:
        self.compact_mode = not self.compact_mode
        self.ui_mode = "compact" if self.compact_mode else "comfortable"
        self.apply_ui_mode()
        self.app.table.configure_columns(force=True, refresh_rows=True)
        if self.compact_mode:
            self.app.status.update(
                "View mode: compact. Keys: p provider, u use, s sort, f fit, h gems, [/] page."
            )
        else:
            self.app.status.update("View mode: comfortable")

    def schedule_reflow(self) -> None:
        self._resize_generation += 1
        generation = self._resize_generation
        if self._resize_timer is not None:
            try:
                self._resize_timer.stop()
            except Exception:
                logger.debug("Resize reflow timer already stopped")
        self._resize_timer = self.app.set_timer(0.16, lambda: self.apply_reflow(generation))

    def apply_reflow(self, generation: int) -> None:
        """Reflow columns unless a newer resize superseded this one or the table unmounted."""
        if generation != self._resize_generation:
            return
        self._resize_timer = None
        try:
            self.app.query_one("#results-table", DataTable)
        except NoMatches:
            return
        self.app.table.configure_columns(refresh_rows=True)

    def cycle_theme(self) -> None:
        self.color_theme = next_theme(self.color_theme)
        try:
            self.app.stylesheet.add_source(theme_css(THEMES[self.color_theme]))
        except Exception:
            logger.warning("Failed to inject theme CSS for {}", self.color_theme)
        self.app.status.update(f"Theme: {self.color_theme}")

