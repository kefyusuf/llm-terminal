"""Textual application shell that composes the viewer controllers."""

import os

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Input,
    Label,
    Select,
    Static,
)

import config
from app.download_controller import DownloadController
from app.download_progress import DownloadProgressPanel
from app.download_tracker import DownloadTracker
from app.model_details import DetailsController
from app.results_table import ResultsTableView
from app.search_constants import USE_CASE_OPTIONS
from app.search_controller import SearchController
from app.search_results_state import SearchResultsState
from app.status_bar import StatusController
from app.view_layout import VIEWER_CSS, LayoutController
from app.widgets import SystemInfoWidget
from core import cache_db
from core.logging_ import get_logger
from providers.hf_provider import HuggingFaceProvider
from providers.ollama_provider import OllamaProvider
from search.search_cache import SearchCache

logger = get_logger(__name__)

PROVIDER_REFRESH_INTERVAL = 30


class AIModelViewer(App):
    """Discover, compare, and download local LLM models.

    The app owns widgets, bindings, and timers. Behaviour lives in controllers
    that share one SearchResultsState:

    - ``status``: status bar, hardware snapshot, Ollama liveness
    - ``layout``: view mode, theme, resize reflow
    - ``table``: results-table rendering and selection
    - ``search``: filters, debounce, live-first cache, pagination, workers
    - ``downloads``: history table, polling, progress panel, actions
    - ``details``: model detail, plan, and comparison modals
    """

    CSS = VIEWER_CSS

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("/", "focus_search", "Search"),
        ("r", "refresh_search", "Refresh"),
        ("p", "cycle_provider", "Providers"),
        ("P", "open_plan_mode", "Plan"),
        ("c", "toggle_comparison", "Compare"),
        ("C", "show_comparison", "View Cmp"),
        ("[", "prev_page", "Prev Page"),
        ("]", "next_page", "Next Page"),
        ("u", "cycle_use_case", "Use Case"),
        ("s", "cycle_sort_mode", "Sort"),
        ("f", "cycle_fit_filter", "Fit"),
        ("v", "toggle_view_mode", "View"),
        ("h", "toggle_hidden_gems", "Hidden Gems"),
        ("t", "cycle_theme", "Theme"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
    ]

    def __init__(self):
        super().__init__()
        settings = config.settings
        self.state = SearchResultsState(
            page_size=settings.hf_search_limit,
            max_pages=settings.hf_search_max_pages,
        )
        hf_model_info_cache = {}
        self.status = StatusController(self)
        self.layout = LayoutController(self, ui_mode=settings.ui_mode, theme=settings.theme)
        self.table = ResultsTableView(self, self.state)
        self.dl = DownloadTracker(
            update_status=self.update_status,
            refresh_table=lambda: self.table.refresh(),
            refresh_download_history_table=lambda: self.downloads.refresh_history_table(),
            request_download_history_refresh=lambda force=False: (
                self.downloads.request_history_refresh(force)
            ),
            render_download_debug=lambda debug=None, health=None: (
                self.downloads.render_debug(debug, health)
            ),
            find_model_by_target_id=lambda target_id: self.downloads.find_model(target_id),
            history_limit=settings.download_history_limit,
            history_refresh_interval=settings.download_history_refresh_interval,
        )
        self.downloads = DownloadController(self, self.state, self.dl)
        self.search = SearchController(
            self,
            self.state,
            cache=SearchCache(
                ttl_seconds=settings.search_cache_ttl_seconds,
                max_entries=settings.search_cache_max_entries,
                ram_threshold_gb=settings.search_cache_ram_threshold_gb,
                vram_threshold_gb=settings.search_cache_vram_threshold_gb,
            ),
            hf_provider=HuggingFaceProvider(model_info_cache=hf_model_info_cache),
            ollama_provider=OllamaProvider(),
        )
        self.details = DetailsController(
            self, self.state, hf_model_info_cache=hf_model_info_cache
        )
        self.system_metrics_timer = None
        self.download_status_timer = None

    def compose(self) -> ComposeResult:
        yield SystemInfoWidget(id="header")
        with Horizontal(id="search-filters-row"):
            with Vertical(id="search-panel"):
                yield Input(
                    placeholder="Press / to search Ollama models (e.g., qwen, llama)",
                    id="search-input",
                )
            with Vertical(id="provider-panel"):
                yield Select(
                    ((label, label) for label in self.search.provider_labels),
                    value=self.state.current_filter,
                    allow_blank=False,
                    id="provider-select",
                )
            with Vertical(id="use-case-panel"):
                yield Select(
                    ((label, key) for key, label in USE_CASE_OPTIONS),
                    value=self.state.use_case_filter,
                    allow_blank=False,
                    id="use-case-select",
                )
        yield Checkbox("Hidden gems only (Hugging Face)", id="gem-toggle")
        yield Static("", id="compact-chipbar")
        yield Static("Models (0 shown / 0 total)", id="results-meta")
        yield DataTable(id="results-table", cursor_type="row")
        with Horizontal(id="pagination-controls"):
            yield Button("◀ Prev", id="prev-page", variant="default", disabled=True)
            yield Static("Page 1", id="page-indicator")
            yield Button("Next ▶", id="next-page", variant="default", disabled=True)
        yield Label("Recent Downloads", id="downloads-label")
        yield Static("Workers: - | Duplicates: -", id="downloads-debug")
        yield DataTable(id="download-history-table", cursor_type="row")
        yield DownloadProgressPanel(id="active-download-progress")
        yield Static("", id="status-bar")
        yield Footer()

    # -- lifecycle --------------------------------------------------------

    def _smoke_mode_enabled(self) -> bool:
        return os.getenv("AIMODEL_SMOKE") == "1"

    def _finish_smoke_mode(self) -> None:
        if getattr(self, "_smoke_exit_requested", False):
            return
        self._smoke_exit_requested = True
        self.update_status("Smoke mode startup complete.")
        self.exit()

    def on_mount(self) -> None:
        """Mount the UI immediately; restore history and hardware off the UI thread."""
        self.title = "AI Model Explorer"
        self.layout.apply_ui_mode()
        self.table.configure_columns(force=True)
        self.query_one("#results-table", DataTable).zebra_stripes = True
        self.downloads.setup_history_table()
        self.table.update_meta(0)

        if self._smoke_mode_enabled():
            self.call_after_refresh(self._finish_smoke_mode)
            self.set_timer(1, self._finish_smoke_mode)
            return

        cache_db.init_db()
        cache_db.cleanup_old_entries()
        self.status.show_cached_specs()
        self.update_status(
            "Ready. Search defaults to Ollama. Select 'Hugging Face' filter for HF models."
        )
        self.status.request_system_info_refresh(force=True)
        if not self.status.ollama_running:
            self.update_status(
                "Ready. Ollama is not running; local install/runtime features disabled. "
                "Search HF for more models."
            )
        self.system_metrics_timer = self.set_interval(
            config.settings.hardware_poll_interval,
            self.status.request_system_info_refresh,
        )
        self.downloads.start()
        self.search.request_provider_refresh()
        self.set_interval(PROVIDER_REFRESH_INTERVAL, self.search.request_provider_refresh)

    def start_download_polling(self) -> None:
        if self.download_status_timer is None:
            self.download_status_timer = self.set_interval(
                config.settings.ui_download_poll_interval,
                self.request_download_poll,
            )

    def on_unmount(self) -> None:
        self.dl.shutdown()

    # -- API used by modals -----------------------------------------------

    def update_status(self, text) -> None:
        self.status.update(text)

    def _set_modal_poll_pause(self, enabled: bool) -> None:
        self.status.set_modal_poll_pause(enabled)

    def request_download_poll(self, force=False) -> None:
        self.downloads.request_poll(force)

    def start_model_download(self, model) -> None:
        self.downloads.start_download(model)

    def cancel_model_download(self, model) -> None:
        self.downloads.cancel_download(model)

    def delete_download_entry(self, target_id, delete_data=False) -> None:
        self.downloads.delete_entry(target_id, delete_data=delete_data)

    # -- events -----------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.search.start(event.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider-select":
            self.search.set_provider(str(event.value), sync_widget=False)
        elif event.select.id == "use-case-select":
            key = str(event.value)
            if key != self.state.use_case_filter:
                self.search.set_use_case(key, sync_widget=False)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "gem-toggle":
            self.search.set_hidden_gems(bool(event.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prev-page":
            self.search.go_to_page(self.state.current_page - 1)
        elif event.button.id == "next-page":
            self.search.go_to_page(self.state.current_page + 1)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.prevent_default()
        data_table = event.data_table
        if data_table.id == "download-history-table":
            self.downloads.on_history_row_selected(data_table, event.row_key.value)
        elif data_table.id == "results-table":
            self.details.on_result_selected(event.row_key.value)

    def on_resize(self, event) -> None:
        _ = event
        self.layout.schedule_reflow()

    # -- actions ----------------------------------------------------------

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()
        self.update_status("Search focused. Type query and press Enter.")

    def action_refresh_search(self) -> None:
        self.search.refresh_current()

    def action_cycle_provider(self) -> None:
        self.search.cycle_provider()

    def action_cycle_use_case(self) -> None:
        self.search.cycle_use_case()

    def action_cycle_sort_mode(self) -> None:
        self.search.cycle_sort()

    def action_cycle_fit_filter(self) -> None:
        self.search.cycle_fit()

    def action_toggle_hidden_gems(self) -> None:
        checkbox = self.query_one("#gem-toggle", Checkbox)
        checkbox.value = not checkbox.value
        self.state.hidden_gems_only = bool(checkbox.value)
        self.table.refresh()
        self.update_status(f"Hidden gems filter: {'ON' if checkbox.value else 'OFF'}.")

    def action_prev_page(self) -> None:
        self.search.go_to_page(self.state.current_page - 1)

    def action_next_page(self) -> None:
        self.search.go_to_page(self.state.current_page + 1)

    def action_open_plan_mode(self) -> None:
        self.details.open_plan_mode()

    def action_toggle_comparison(self) -> None:
        self.details.toggle_comparison()

    def action_show_comparison(self) -> None:
        self.details.show_comparison()

    def action_toggle_view_mode(self) -> None:
        self.layout.toggle_view_mode()

    def action_cycle_theme(self) -> None:
        self.layout.cycle_theme()
