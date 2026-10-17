import os
import threading
import time

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

import config
from app.download_controller import DownloadControllerMixin
from app.download_manager import DownloadManager
from app.download_progress import DownloadProgressPanel
from app.model_details import ModelDetailsMixin
from app.results_table import ResultsTableMixin
from app.search_controller import SearchControllerMixin
from app.search_results_state import SearchResultsState
from app.search_state_access import SearchStateAccessMixin
from app.status_bar import StatusBarMixin
from app.view_layout import VIEWER_CSS, ViewLayoutMixin
from app.widgets import SystemInfoWidget
from core import cache_db
from core.hardware import HardwareMonitor, check_ollama_running
from core.logging_ import get_logger
from providers.hf_provider import HuggingFaceProvider
from providers.ollama_provider import OllamaProvider
from search.search_cache import SearchCache

logger = get_logger(__name__)


class AIModelViewer(
    SearchStateAccessMixin,
    ResultsTableMixin,
    SearchControllerMixin,
    DownloadControllerMixin,
    ModelDetailsMixin,
    StatusBarMixin,
    ViewLayoutMixin,
    App,
):
    """Main TUI application for discovering, comparing, and downloading local LLM models.

    Owns an in-process download manager, polls hardware
    metrics via :class:`~hardware.HardwareMonitor`, and queries both the
    Ollama registry and Hugging Face for GGUF model results.
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
        self.monitor = None
        self._hardware_init_lock = threading.Lock()
        self.search_state = SearchResultsState(
            page_size=config.settings.hf_search_limit,
            max_pages=config.settings.hf_search_max_pages,
        )
        self.ui_mode = config.settings.ui_mode
        self.compact_mode = self.ui_mode == "compact"
        self.comparison_set: list[dict] = []
        self._color_theme = config.settings.theme
        self.ollama_running = False
        self.hf_model_info_cache = {}
        self.ollama_provider = OllamaProvider()
        self.hf_provider = HuggingFaceProvider(model_info_cache=self.hf_model_info_cache)
        self.search_cache = SearchCache(
            ttl_seconds=config.settings.search_cache_ttl_seconds,
            max_entries=config.settings.search_cache_max_entries,
            ram_threshold_gb=config.settings.search_cache_ram_threshold_gb,
            vram_threshold_gb=config.settings.search_cache_vram_threshold_gb,
        )
        self.system_metrics_timer = None
        self.ollama_status_timer = None
        self.download_status_timer = None
        self.latest_specs = None
        self._modal_poll_pause_count = 0
        self._system_info_refresh_running = False
        self._resize_reflow_timer = None
        self._resize_reflow_generation = 0
        self._search_debounce_timer = None
        self._search_debounce_delay = 0.12

        self.dl = DownloadManager(
            update_status=self.update_status,
            refresh_table=self.refresh_table,
            refresh_download_history_table=self._refresh_download_history_table_ui,
            request_download_history_refresh=self._request_download_history_refresh_ui,
            render_download_debug=self._render_download_debug,
            find_model_by_target_id=self._find_model_by_target_id,
            history_limit=config.settings.download_history_limit,
            history_refresh_interval=config.settings.download_history_refresh_interval,
        )

    def compose(self) -> ComposeResult:
        yield SystemInfoWidget(id="header")
        with Horizontal(id="search-filters-row"):
            with Vertical(id="search-panel"):
                yield Input(
                    placeholder="Press / to search Ollama models (e.g., qwen, llama)",
                    id="search-input",
                )
            with Vertical(id="provider-panel"), RadioSet(id="filter-set"):
                yield RadioButton("Ollama", value=True, id="filter-ollama")
                yield RadioButton("Hugging Face", id="filter-hf")
            with Vertical(id="use-case-panel"), RadioSet(id="use-case-filter"):
                yield RadioButton("Any Use", value=True, id="uc-all")
                yield RadioButton("Chat", id="uc-chat")
                yield RadioButton("Coding", id="uc-coding")
                yield RadioButton("Vision", id="uc-vision")
                yield RadioButton("Reason", id="uc-reasoning")
                yield RadioButton("Math", id="uc-math")
                yield RadioButton("Embed", id="uc-embedding")
                yield RadioButton("General", id="uc-general")
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

    def _smoke_mode_enabled(self) -> bool:
        return os.getenv("AIMODEL_SMOKE") == "1"

    def _finish_smoke_mode(self) -> None:
        if getattr(self, "_smoke_exit_requested", False):
            return
        self._smoke_exit_requested = True
        self.update_status("Smoke mode startup complete.")
        self.exit()

    def on_unmount(self) -> None:
        self.dl.shutdown()

    def on_mount(self) -> None:
        """Initialise the UI, restore download history, and set up polling timers."""
        self.title = "AI Model Explorer"
        self._apply_ui_mode()
        self._configure_results_table_columns(force=True)
        self.query_one("#results-table", DataTable).zebra_stripes = True
        download_table = self.query_one("#download-history-table", DataTable)
        download_table.add_columns(
            "Source",
            "Publisher",
            "Model",
            "Status",
            "Detail",
            "Action",
        )
        self.refresh_download_history_table()
        self._update_results_meta(0)

        if self._smoke_mode_enabled():
            self.call_after_refresh(self._finish_smoke_mode)
            self.set_timer(1, self._finish_smoke_mode)
            return

        self.dl.sync_jobs(force=True)

        cache_db.init_db()
        cache_db.cleanup_old_entries()

        cached_specs = cache_db.get_hardware_snapshot()
        if cached_specs is not None:
            self.latest_specs = cached_specs
            self.query_one(SystemInfoWidget).update_info(cached_specs, None)
        else:
            self.query_one(SystemInfoWidget).update("Detecting hardware...")

        self.last_download_history_refresh_at = time.monotonic()
        self.update_status(
            "Ready. Search defaults to Ollama. Select 'Hugging Face' filter for HF models."
        )
        self.request_system_info_refresh(force=True)
        if not self.ollama_running:
            self.update_status(
                "Ready. Ollama is not running; local install/runtime features disabled. Search HF for more models."
            )
        self.system_metrics_timer = self.set_interval(
            config.settings.hardware_poll_interval,
            self.request_system_info_refresh,
        )
        self.download_status_timer = self.set_interval(
            config.settings.ui_download_poll_interval,
            self.request_download_poll,
        )

    @work(thread=True)
    def _run_system_info_refresh_worker(self):
        specs = None
        running_now = self.ollama_running
        try:
            specs = self._get_hardware_monitor().get_specs()
            running_now = check_ollama_running()
            cache_db.set_hardware_snapshot(specs)
        except Exception as exc:
            logger.warning("System info refresh failed: {}", exc)
        self.call_from_thread(self._apply_system_info_refresh, specs, running_now)

    def _get_hardware_monitor(self):
        """Initialize once from workers; failed construction remains retryable."""
        with self._hardware_init_lock:
            if self.monitor is None:
                self.monitor = HardwareMonitor()
            return self.monitor
