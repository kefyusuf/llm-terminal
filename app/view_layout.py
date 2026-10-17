"""Viewer styling, view modes, themes, and debounced resize handling."""

from textual import events
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Checkbox,
    DataTable,
    Footer,
    Input,
    Label,
    RadioSet,
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
    RadioSet {
        layout: horizontal;
        width: 100%;
        height: 3;
        border: round #2f3850;
        align: center middle;
        margin-bottom: 0;
        background: #11192b;
        color: #7f8fb8;
    }
    RadioButton { color: #9fb2df; }
    RadioButton.-selected { color: #4fe08a; text-style: bold; }
    #filter-set { margin-bottom: 0; }
    #use-case-filter { height: 3; margin-bottom: 0; }
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



class ViewLayoutMixin:
    """Viewer styling, view modes, themes, and debounced resize handling.

    Methods use the viewer's shared state and Textual APIs, preserving subclass hooks.
    """

    def _apply_ui_mode(self) -> None:
        try:
            results_table = self.query_one("#results-table", DataTable)
            downloads_label = self.query_one("#downloads-label", Label)
            downloads_debug = self.query_one("#downloads-debug", Static)
            download_table = self.query_one("#download-history-table", DataTable)
            use_case_filter = self.query_one("#use-case-filter", RadioSet)
            gem_toggle = self.query_one("#gem-toggle", Checkbox)
            compact_chipbar = self.query_one("#compact-chipbar", Static)
            search_input = self.query_one("#search-input", Input)
            search_row = self.query_one("#search-filters-row", Horizontal)
            search_panel = self.query_one("#search-panel", Vertical)
            provider_panel = self.query_one("#provider-panel", Vertical)
            use_case_panel = self.query_one("#use-case-panel", Vertical)
            pagination_controls = self.query_one("#pagination-controls", Horizontal)
            results_meta = self.query_one("#results-meta", Static)
            footer = self.query_one(Footer)
        except Exception:
            logger.debug("UI mode widgets not yet mounted, skipping apply")
            return

        if self.compact_mode:
            results_table.styles.height = "3fr"
            use_case_filter.styles.display = "block"
            gem_toggle.styles.display = "none"
            compact_chipbar.styles.display = "block"
            results_meta.styles.display = "none"
            pagination_controls.styles.display = "none"
            downloads_label.styles.display = "none"
            downloads_debug.styles.display = "none"
            download_table.styles.display = "none"
            footer.styles.display = "none"
            search_row.styles.height = "3"
            search_panel.styles.width = "30%"
            search_panel.styles.height = "3"
            provider_panel.styles.display = "block"
            provider_panel.styles.width = "22"
            provider_panel.styles.height = "3"
            use_case_panel.styles.display = "block"
            use_case_panel.styles.width = "1fr"
            use_case_panel.styles.height = "3"
            search_input.styles.height = "3"
            search_input.placeholder = "Press / to search..."
        else:
            results_table.styles.height = "2fr"
            use_case_filter.styles.display = "block"
            gem_toggle.styles.display = "block"
            compact_chipbar.styles.display = "none"
            results_meta.styles.display = "block"
            pagination_controls.styles.display = "block"
            downloads_label.styles.display = "block"
            downloads_debug.styles.display = "block"
            download_table.styles.display = "block"
            footer.styles.display = "block"
            search_row.styles.height = "3"
            search_panel.styles.width = "34%"
            search_panel.styles.height = "3"
            provider_panel.styles.display = "block"
            provider_panel.styles.width = "22"
            provider_panel.styles.height = "3"
            use_case_panel.styles.display = "block"
            use_case_panel.styles.width = "1fr"
            use_case_panel.styles.height = "3"
            search_input.styles.height = "3"
            search_input.placeholder = "Press / to search Ollama models (e.g., qwen, llama)"

        self._update_results_meta(self.query_one("#results-table", DataTable).row_count)

    def action_toggle_view_mode(self) -> None:
        self.compact_mode = not self.compact_mode
        self.ui_mode = "compact" if self.compact_mode else "comfortable"
        self._apply_ui_mode()
        self._configure_results_table_columns(force=True, refresh_rows=True)
        if self.compact_mode:
            self.update_status(
                "View mode: compact. Keys: p provider, u use, s sort, f fit, h gems, [/] page."
            )
        else:
            self.update_status("View mode: comfortable")

    def on_resize(self, event: events.Resize) -> None:
        _ = event
        self._resize_reflow_generation += 1
        generation = self._resize_reflow_generation
        if self._resize_reflow_timer is not None:
            try:
                self._resize_reflow_timer.stop()
            except Exception:
                logger.debug("Resize reflow timer already stopped")
        self._resize_reflow_timer = self.set_timer(
            0.16,
            lambda: self._apply_resize_reflow(generation),
        )

    def _apply_resize_reflow(self, generation: int) -> None:
        if generation != self._resize_reflow_generation:
            return
        self._resize_reflow_timer = None
        self._configure_results_table_columns(refresh_rows=True)

    def action_cycle_theme(self) -> None:
        """Cycle to the next color theme."""
        self._color_theme = next_theme(self._color_theme)
        theme = THEMES[self._color_theme]
        # Inject theme CSS
        try:
            self.stylesheet.add_source(theme_css(theme))
        except Exception:
            logger.warning("Failed to inject theme CSS for {}", self._color_theme)
        self.update_status(f"Theme: {self._color_theme}")
