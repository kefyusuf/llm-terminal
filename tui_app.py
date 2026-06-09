import os
import subprocess
import time
from urllib.error import HTTPError

from textual import events, on, work
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
from core import cache_db
from core.logging_ import get_logger

logger = get_logger(__name__)

from core.model_intelligence import plan_hardware_for_model
from providers import get_all_provider_classes, get_provider_filter_labels
from terminal_ui.themes import THEMES, next_theme, theme_css

from core.hardware import HardwareMonitor, check_ollama_running
from providers.hf_provider import enrich_hf_model_details, search_hf_models
from providers.ollama_provider import get_installed_ollama_models, search_ollama_models
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
from search.search_cache import SearchCache
from search.search_orchestration import (
    build_query_key,
    cache_hit_suffix,
    has_more_pages_for_results,
    is_hf_provider_selection,
    page_info_suffix,
    provider_display_name,
    provider_result_count,
    provider_search_status,
    providers_from_filter,
    validate_page_request,
)
from downloads.download_history import cancel_model_payload, fallback_entry_from_target, is_external_entry
from downloads.download_manager import download_target_id
from downloads.download_status import is_active_state
from downloads.service_client import ensure_service_running

from app.download_manager import DownloadManager
from app.modals import (
    ComparisonModal,
    DownloadJobModal,
    ModelDetailModal,
    PlanModeModal,
)
from app.widgets import SystemInfoWidget


class AIModelViewer(App):
    """Main TUI application for discovering, comparing, and downloading local LLM models.

    Connects to a background :mod:`downloads.download_service` over HTTP, polls hardware
    metrics via :class:`~hardware.HardwareMonitor`, and queries both the
    Ollama registry and Hugging Face for GGUF model results.
    """

    CSS = """
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

    RESULTS_COLUMN_LABELS_COMFORTABLE = {
        "inst": "Install",
        "source": "Source",
        "publisher": "Provider",
        "name": "Model",
        "params": "Scale",
        "use_case": "Use Case",
        "score": "Score",
        "quant": "Format",
        "mode": "Mode",
        "fit": "Fit",
        "download": "Download",
    }

    RESULTS_COLUMN_LABELS_COMPACT = {
        "inst": "In",
        "source": "Src",
        "publisher": "Prov",
        "name": "Model",
        "params": "Param",
        "use_case": "Use",
        "score": "Score",
        "quant": "Quant",
        "mode": "Mode",
        "fit": "Fit",
        "download": "D/L",
    }

    USE_CASE_OPTIONS = [
        ("all", "Any Use"),
        ("chat", "Chat"),
        ("coding", "Coding"),
        ("vision", "Vision"),
        ("reasoning", "Reason"),
        ("math", "Math"),
        ("embedding", "Embed"),
        ("general", "General"),
    ]

    SORT_OPTIONS = [
        ("score", "Score"),
        ("downloads", "Downloads"),
        ("name", "Name"),
    ]

    FIT_OPTIONS = [
        ("all", "All"),
        ("fit", "Fit"),
        ("partial", "Partial"),
        ("nofit", "No Fit"),
    ]

    def __init__(self):
        super().__init__()
        self.monitor = HardwareMonitor()
        self.all_results = []
        self.ui_mode = config.settings.ui_mode
        self.compact_mode = self.ui_mode == "compact"
        self.current_filter = "Ollama"
        self.use_case_filter = "all"
        self.sort_mode = "score"
        self.fit_filter = "all"
        self.hidden_gems_only = False
        self.comparison_set: list[dict] = []
        self._color_theme = config.settings.theme
        self.ollama_running = False
        self.last_search_error = ""
        self.search_counter = 0
        self.active_search_id = 0
        self.hf_model_info_cache = {}
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
        self.results_column_keys = []
        self.results_column_widths = {}
        self.current_page = 0
        self.page_size = config.settings.hf_search_limit
        self.max_pages = config.settings.hf_search_max_pages
        self.total_results = 0
        self.has_more_pages = True
        self._system_info_refresh_running = False
        self._resize_reflow_timer = None
        self._resize_reflow_generation = 0
        self._search_debounce_timer = None
        self._search_debounce_delay = 0.12
        self._pending_search_payload = None
        self._search_inflight_signature = None
        self._search_inflight_started_at = 0.0
        self._search_progress_stamp = (0, "", 0.0)
        self._search_progress_visible = False

        self.dl = DownloadManager(
            update_status=self.update_status,
            refresh_table=self.refresh_table,
            refresh_download_history_table=self._refresh_download_history_table_ui,
            request_download_history_refresh=self._request_download_history_refresh_ui,
            render_download_debug=self._render_download_debug,
            ensure_download_fields=None,
            find_model_by_target_id=self._find_model_by_target_id,
            history_limit=config.settings.download_history_limit,
            history_refresh_interval=config.settings.download_history_refresh_interval,
            poll_request_timeout=config.settings.download_poll_request_timeout,
        )

    def _set_modal_poll_pause(self, enabled: bool) -> None:
        if enabled:
            self._modal_poll_pause_count += 1
            return

        self._modal_poll_pause_count = max(0, self._modal_poll_pause_count - 1)
        if self._modal_poll_pause_count == 0:
            self.request_system_info_refresh(force=True)
            self.request_download_poll(force=True)

    def _column_labels(self):
        if self.compact_mode:
            return self.RESULTS_COLUMN_LABELS_COMPACT
        return self.RESULTS_COLUMN_LABELS_COMFORTABLE

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

    def _use_case_label(self, key: str) -> str:
        for option_key, option_label in self.USE_CASE_OPTIONS:
            if option_key == key:
                return option_label
        return "Any Use"

    def _sort_label(self, key: str) -> str:
        for option_key, option_label in self.SORT_OPTIONS:
            if option_key == key:
                return option_label
        return "Score"

    def _fit_label(self, key: str) -> str:
        for option_key, option_label in self.FIT_OPTIONS:
            if option_key == key:
                return option_label
        return "All"

    def _use_case_compact_tag(self, key: str) -> str:
        mapping = {
            "all": "ALL",
            "chat": "CHAT",
            "coding": "CODE",
            "vision": "VIS",
            "reasoning": "RSN",
            "math": "MATH",
            "embedding": "EMB",
            "general": "GEN",
        }
        return mapping.get(key, "ALL")

    def _sort_compact_tag(self, key: str) -> str:
        mapping = {
            "score": "SCORE",
            "downloads": "DL",
            "name": "NAME",
        }
        return mapping.get(key, "SCORE")

    def _fit_compact_tag(self, key: str) -> str:
        mapping = {
            "all": "ALL",
            "fit": "FIT",
            "partial": "PART",
            "nofit": "NO",
        }
        return mapping.get(key, "ALL")

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
        keys = [key for key, _label in self.USE_CASE_OPTIONS]
        current_key = self.use_case_filter if self.use_case_filter in keys else "all"
        next_key = keys[(keys.index(current_key) + 1) % len(keys)]
        self._set_use_case_filter(next_key)
        self.refresh_table()
        self.update_status(f"Use Case filter set to {self._use_case_label(next_key)}.")

    def action_cycle_sort_mode(self) -> None:
        keys = [key for key, _label in self.SORT_OPTIONS]
        current_key = self.sort_mode if self.sort_mode in keys else "score"
        next_key = keys[(keys.index(current_key) + 1) % len(keys)]
        self.sort_mode = next_key
        self.refresh_table()
        self.update_status(f"Sort set to {self._sort_label(next_key)}.")

    def action_cycle_fit_filter(self) -> None:
        keys = [key for key, _label in self.FIT_OPTIONS]
        current_key = self.fit_filter if self.fit_filter in keys else "all"
        next_key = keys[(keys.index(current_key) + 1) % len(keys)]
        self.fit_filter = next_key
        self.refresh_table()
        self.update_status(f"Fit filter set to {self._fit_label(next_key)}.")

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

    def on_mount(self) -> None:
        """Initialise the UI, start the download service, and set up polling timers."""
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

        service_ok = ensure_service_running()
        if not service_ok:
            self.update_status(
                "Download service is unavailable or outdated. Restart the app/service."
            )
        else:
            self.dl.sync_jobs(force=True)

        cache_db.init_db()
        cache_db.cleanup_old_entries()

        cached_specs = cache_db.get_hardware_snapshot()
        if cached_specs is not None:
            self.latest_specs = cached_specs
            self.query_one(SystemInfoWidget).update_info(cached_specs, check_ollama_running())

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

    def action_open_plan_mode(self) -> None:
        """Open plan mode to analyze hardware requirements for the selected model."""
        model = self._get_selected_model()
        if not model:
            self.update_status("No model selected for plan analysis.")
            return

        model_name = model.get("name", "")

        try:
            plans = plan_hardware_for_model(model_name)
            self.push_screen(PlanModeModal(model_name, plans))
        except Exception as exc:
            logger.warning("Plan analysis failed for {}: {}", model_name, exc)
            self.update_status(f"Plan analysis failed: {exc}")

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

    def action_toggle_comparison(self) -> None:
        """Toggle the selected model in/out of the comparison set (max 4)."""
        model = self._get_selected_model()
        if not model:
            self.update_status("No model selected for comparison.")
            return

        model_key = result_unique_key(model)

        # Check if already in comparison set
        for i, m in enumerate(self.comparison_set):
            if result_unique_key(m) == model_key:
                self.comparison_set.pop(i)
                count = len(self.comparison_set)
                self.update_status(f"Removed from comparison. ({count}/4 models)")
                return

        # Add to comparison set
        if len(self.comparison_set) >= 4:
            self.update_status(
                "Comparison set full (max 4). Press C to view or remove a model first."
            )
            return

        self.comparison_set.append(model)
        count = len(self.comparison_set)
        self.update_status(f"Added to comparison. ({count}/4 models) Press C to view.")

    def action_show_comparison(self) -> None:
        """Show the comparison modal for selected models."""
        if len(self.comparison_set) < 2:
            self.update_status(
                f"Need at least 2 models for comparison. ({len(self.comparison_set)}/4 selected)"
            )
            return

        self.push_screen(ComparisonModal(self.comparison_set))

    def _configure_results_table_columns(self, force: bool = False, refresh_rows: bool = False):
        table = self.query_one("#results-table", DataTable)
        available_width = max(table.size.width, self.size.width, 80)
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

        table.clear(columns=True)
        for key in next_keys:
            width = base_widths[key]
            table.add_column(
                self._format_header_label(labels[key], width),
                width=width,
                key=key,
            )

        self.results_column_keys = next_keys
        self.results_column_widths = {key: base_widths[key] for key in next_keys}

        if refresh_rows:
            self.refresh_table()

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

    def update_status(self, text):
        """Update the status bar at the bottom of the screen with *text*."""
        try:
            self.query_one("#status-bar", Static).update(text)
        except Exception:
            logger.debug("Status bar not yet mounted, skipping update")

    def update_system_info(self):
        """Compatibility wrapper that triggers an async system-info refresh."""
        self.request_system_info_refresh(force=True)

    def request_system_info_refresh(self, force=False):
        if self._modal_poll_pause_count > 0 and not force:
            return
        if self._system_info_refresh_running and not force:
            return
        if self._system_info_refresh_running:
            return
        self._system_info_refresh_running = True
        self._run_system_info_refresh_worker()

    @work(thread=True)
    def _run_system_info_refresh_worker(self):
        specs = None
        running_now = self.ollama_running
        try:
            specs = self.monitor.get_specs()
            running_now = check_ollama_running()
            cache_db.set_hardware_snapshot(specs)
        except Exception as exc:
            logger.warning("System info refresh failed: {}", exc)
        self.call_from_thread(self._apply_system_info_refresh, specs, running_now)

    def _apply_system_info_refresh(self, specs, running_now):
        self._system_info_refresh_running = False
        if specs is not None:
            self.latest_specs = specs
        specs_to_render = self.latest_specs
        if specs_to_render is not None:
            try:
                self.query_one(SystemInfoWidget).update_info(specs_to_render, running_now)
            except Exception:
                pass

        state_changed = running_now != self.ollama_running
        self.ollama_running = running_now
        if state_changed:
            if running_now:
                self.update_status("Ollama started. Local runtime features enabled.")
            else:
                self.update_status("Ollama stopped. Local runtime features disabled.")

    def _find_model_by_target_id(self, target_id):
        """Find a model in all_results by its target_id."""
        for item in self.all_results:
            if download_target_id(item) == target_id:
                return item
        return None

    def poll_ollama_status(self, refresh_only=False):
        self.request_system_info_refresh(force=not refresh_only)

    def _current_specs_for_search_ui(self):
        if self.latest_specs is not None:
            return self.latest_specs
        cached = cache_db.get_hardware_snapshot()
        if cached is not None:
            self.latest_specs = cached
            return cached
        return {
            "cpu_name": self.monitor.cpu_name,
            "cpu_cores": self.monitor.cpu_cores,
            "ram_free": 0.0,
            "ram_total": 0.0,
            "vram_free": 0.0,
            "vram_total": 0.0,
            "gpu_name": self.monitor.gpu_name,
            "has_gpu": self.monitor.nvidia_available,
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
            try:
                self._search_debounce_timer.stop()
            except Exception:
                pass
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

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        data_table = getattr(event, "data_table", None)
        if data_table is not None and data_table.id == "download-history-table":
            target_id = str(event.row_key.value)
            entry = self.dl.download_registry.get(target_id)

            if not entry:
                entry = fallback_entry_from_target(target_id)

            cursor_column = data_table.cursor_column
            if cursor_column == 5:
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

        if data_table is not None and data_table.id != "results-table":
            return
        row_key = str(event.row_key.value)
        selected_model = next(
            (item for item in self.all_results if result_unique_key(item) == row_key),
            None,
        )
        if selected_model:
            if selected_model["source"] == "Hugging Face":
                self.update_status("Loading detailed model metadata...")
                self.open_hf_detail_worker(selected_model)
            else:
                self.open_model_detail_modal(selected_model)

    def open_model_detail_modal(self, model):
        self.push_screen(ModelDetailModal(model))

    def on_download_job_modal_action(self, result, target_id):
        _ = (result, target_id)

    def cancel_model_download(self, model):
        """Request cancellation of an in-progress or queued download for *model*."""
        ok, msg = self.dl.cancel_download(model)
        self.update_status(msg)
        if ok:
            self.dl.sync_jobs(force=True)
            self._refresh_download_history_table_ui()
            self.refresh_table()

    def delete_download_entry(self, target_id, delete_data=False):
        ok, msg, keys, tid = self.dl.delete_entry(target_id, delete_data=delete_data)
        if keys:
            source_key, name_key = keys
            from downloads.download_lifecycle import reset_results_download_state
            reset_results_download_state(
                self.all_results,
                target_id=tid,
                source_key=source_key,
                name_key=name_key,
                target_id_for_item=download_target_id,
            )
        self.refresh_table()
        self._refresh_download_history_table_ui()
        self.update_status(msg)

    def start_model_download(self, model):
        ok, msg = self.dl.start_download(model)
        self.update_status(msg)
        if ok:
            self.dl.sync_jobs(force=True)
            self._refresh_download_history_table_ui()

    def _refresh_download_history_table_ui(self):
        try:
            table = self.query_one("#download-history-table", DataTable)
        except Exception:
            return
        table.clear()
        rows = self.dl.refresh_history_table(self.dl.download_registry)
        for r in rows:
            table.add_row(
                r["source"], r["publisher"], r["name"],
                self.dl.status_text_from_state(r["state"], r["label"]),
                r["detail"], r["action"],
                key=r["key"],
            )

    def _request_download_history_refresh_ui(self, force=False):
        if self.dl.request_history_refresh(force):
            self._refresh_download_history_table_ui()

    def refresh_download_history_table(self):
        self._refresh_download_history_table_ui()

    def request_download_history_refresh(self, force=False):
        self._request_download_history_refresh_ui(force)

    def request_download_poll(self, force=False):
        if not self.dl.can_poll(self._modal_poll_pause_count, force):
            return
        self.dl.set_poll_running(True)
        self._run_download_poll_worker()

    @work(thread=True)
    def _run_download_poll_worker(self):
        jobs, debug, health = self.dl.poll_jobs()
        self.call_from_thread(self._apply_download_poll_snapshot, jobs, debug, health)

    def _apply_download_poll_snapshot(self, jobs, debug, health):
        self.dl.set_poll_running(False)
        if jobs is not None:
            self.dl.sync_jobs(force=False, jobs=jobs)
            changed = self.dl.ensure_download_fields(self.all_results)
            if changed:
                self.refresh_table()
        self._render_download_debug(debug, health)

        if not self.dl.active_downloads:
            if self.dl.download_history_refresh_pending:
                self._request_download_history_refresh_ui()
            return
        if any(
            is_active_state(self.dl.download_registry.get(target_id, {}).get("state"))
            for target_id in self.dl.active_downloads
        ):
            self._request_download_history_refresh_ui()

    def refresh_download_progress(self):
        self.request_download_poll()

    def refresh_download_debug(self):
        self.request_download_poll(force=True)

    def _render_download_debug(self, debug=None, health=None):
        try:
            debug_widget = self.query_one("#downloads-debug", Static)
        except Exception:
            return

        if debug is not None:
            count = debug.get("count", 0)
            has_duplicates = bool(debug.get("has_duplicates", False))
            worker_alive = bool(debug.get("worker_alive", True))
            dup_text = "yes" if has_duplicates else "no"
            worker_text = "up" if worker_alive else "down"
            debug_widget.update(f"Workers: {count} ({worker_text}) | Duplicates: {dup_text}")
            return

        if health is not None:
            version = health.get("version", "unknown")
            debug_widget.update(f"Workers: legacy service ({version}) | Duplicates: unknown")
            return

        debug_widget.update("Workers: unavailable | Duplicates: unknown")

    def _download_cell_text(self, model):
        return self.dl.download_cell_text(model)

    def _mark_ollama_model_installed(self, model_name):
        for item in self.all_results:
            if item.get("source") == "Ollama" and item.get("name") == model_name:
                item["inst"] = "[green]✔[/green]"
        self.refresh_table()

    @work(thread=True)
    def open_hf_detail_worker(self, selected_model):
        specs = self.monitor.get_specs()
        enriched = enrich_hf_model_details(
            selected_model.copy(),
            specs,
            self.hf_model_info_cache,
        )
        self.call_from_thread(self.on_hf_detail_ready, enriched)

    def on_hf_detail_ready(self, enriched_model):
        model_id = enriched_model.get("id")
        if model_id:
            for idx, item in enumerate(self.all_results):
                if item.get("source") == "Hugging Face" and item.get("id") == model_id:
                    self.all_results[idx] = enriched_model
                    break
            self.refresh_table()
        self.update_status("Detailed metadata loaded.")
        self.open_model_detail_modal(enriched_model)

    @work(thread=True)
    def run_search_worker(
        self,
        query: str,
        query_key: str,
        search_id: int,
        providers: list[str] | None = None,
    ) -> None:
        if providers is None:
            providers = self._get_search_providers()

        specs = self.monitor.get_specs()
        ollama_results, ollama_errors = [], []
        hf_results, hf_errors = [], []
        extra_results, extra_errors = [], []
        if "ollama" in providers:
            self.call_from_thread(self.on_search_progress, search_id, "Checking Ollama runtime...")
            ollama_running = check_ollama_running()
            if ollama_running:
                self.call_from_thread(
                    self.on_search_progress,
                    search_id,
                    "Connecting to Ollama and fetching installed models...",
                )
                local_models = get_installed_ollama_models()
            else:
                self.call_from_thread(
                    self.on_search_progress,
                    search_id,
                    "Ollama not running; skipping installed model check...",
                )
                local_models = []

            self.call_from_thread(self.on_search_progress, search_id, "Fetching Ollama data...")
            ollama_page_size = config.settings.ollama_search_limit
            ollama_results, ollama_errors, _ollama_has_more = search_ollama_models(
                query,
                specs,
                local_models,
                page=self.current_page,
                page_size=ollama_page_size,
            )

        if "huggingface" in providers:
            self.call_from_thread(
                self.on_search_progress, search_id, "Fetching Hugging Face data..."
            )
            offset = self.current_page * self.page_size
            hf_token = config.settings.hf_token
            hf_results, hf_errors = search_hf_models(
                query,
                specs,
                self.hf_model_info_cache,
                limit=self.page_size,
                offset=offset,
                hf_token=hf_token,
            )

        if search_id != self.active_search_id:
            return

        for provider_cls in get_all_provider_classes():
            slug = provider_cls.slug
            if slug not in providers:
                continue
            if slug in ("ollama", "huggingface"):
                continue  # Already handled above

            self.call_from_thread(
                self.on_search_progress, search_id, f"Fetching {provider_cls.display_name} data..."
            )
            try:
                instance = provider_cls()
                if instance.detect():
                    p_results, p_errors = instance.search(query, specs, limit=self.page_size)
                    extra_results.extend(p_results)
                    extra_errors.extend(p_errors)
            except Exception:
                pass

        if search_id != self.active_search_id:
            return

        self.call_from_thread(self.on_search_progress, search_id, "Finalizing search results...")
        self.all_results = ollama_results + hf_results + extra_results
        self.dl.ensure_download_fields(self.all_results)
        self.last_search_error = " | ".join((ollama_errors + hf_errors + extra_errors)[:2])

        self.has_more_pages = has_more_pages_for_results(
            providers,
            hf_result_count=len(hf_results),
            ollama_result_count=len(ollama_results),
            page_size=self.page_size,
        )

        result_count = provider_result_count(
            providers,
            hf_result_count=len(hf_results),
            ollama_result_count=len(ollama_results),
        )
        self.call_from_thread(
            self.update_status,
            provider_search_status(
                providers,
                result_count=result_count,
                has_more_pages=self.has_more_pages,
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
        self.call_from_thread(self.on_search_completed, search_id)

    def on_search_completed(self, search_id: int) -> None:
        if search_id != self.active_search_id:
            return

        self._search_inflight_signature = None
        self._search_inflight_started_at = 0.0
        self._search_progress_visible = False

        table = self.query_one("#results-table", DataTable)
        table.loading = False
        self.refresh_table()

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

    def refresh_table(self):
        try:
            self._configure_results_table_columns()
            table = self.query_one("#results-table", DataTable)
        except Exception:
            return
        prev_cursor_row = table.cursor_row
        prev_scroll_x = table.scroll_x
        prev_scroll_y = table.scroll_y
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
