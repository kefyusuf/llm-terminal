import json
import re
import subprocess
import time
from urllib.error import HTTPError

import cache_db
import config
from download_status import (
    is_active_state,
    label_for_state,
    map_service_job_status,
    state_markup_from_state_and_label,
)
from download_manager import download_target_id
from hardware import HardwareMonitor, check_ollama_running
from providers.hf_provider import enrich_hf_model_details, search_hf_models
from providers.ollama_provider import get_installed_ollama_models, search_ollama_models
from service_client import (
    cancel_job,
    create_job,
    delete_job,
    ensure_service_running,
    get_active_download_debug,
    get_service_health,
    list_jobs,
)
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import ModalScreen
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


class ModelDetailModal(ModalScreen):
    """Modal overlay showing full metadata and the run/download command for a model."""

    CSS = """
    ModelDetailModal {
        align: center middle;
        background: $background;
    }
    #modal-container {
        width: 60;
        height: auto;
        background: #0f141f;
        border: round #4a5568;
        padding: 1 2;
    }
    #modal-header {
        background: #1e3a5f;
        padding: 0 2;
        margin: -1 -2 1 -2;
        height: 3;
    }
    #modal-title {
        text-style: bold;
        color: #e2e8f0;
        text-align: center;
        padding: 0;
    }
    #cmd-box {
        background: #0d1117;
        border: solid #2d3748;
        padding: 1;
        margin: 0 0 1 0;
        color: #4fd1c5;
    }
    #button-row {
        layout: horizontal;
        height: 3;
        align: center middle;
    }
    #button-row Button {
        margin: 0 1;
    }
    .fit-perfect { color: #48bb78; }
    .fit-partial { color: #ecc94b; }
    .fit-no-fit { color: #f56565; }
    """

    def __init__(self, data):
        super().__init__()
        self.data = data

    def compose(self) -> ComposeResult:
        if self.data["source"] == "Ollama":
            cmd_text = f"ollama run {self.data['name']}"
        else:
            repo_id = self.data.get("id", self.data["name"])
            cmd_text = f"huggingface-cli download {repo_id} --include '*.gguf'"

        size_source = self.data.get("size_source", "estimated")
        confidence_label = (
            "[#48bb78]Exact[/#48bb78]" if size_source == "exact" else "[#ecc94b]Estimated[/#ecc94b]"
        )

        fit_text = self.data.get("fit", "").lower()
        fit_class = (
            "fit-perfect"
            if "perfect" in fit_text or fit_text == "fit"
            else "fit-partial"
            if "partial" in fit_text or "slow" in fit_text
            else "fit-no-fit"
        )

        with Vertical(id="modal-container"):
            # Header
            yield Label(f"🤖 {self.data['name']}", id="modal-title")

            # Model Info
            yield Label("")
            yield Label("[bold #63b3ed]📊 Model Information[/bold #63b3ed]")
            yield Label(
                f"[#718096]Source:[/#718096] [#e2e8f0]{self.data['source']} / {self.data.get('publisher', '-')}[/#e2e8f0]"
            )
            yield Label(f"[#718096]Provider:[/#718096] [#e2e8f0]{self.data['provider']}[/#e2e8f0]")
            yield Label(f"[#718096]Use Case:[/#718096] [#e2e8f0]{self.data['use_case']}[/#e2e8f0]")
            yield Label(f"[#718096]Scale:[/#718096] [#e2e8f0]{self.data['params']}[/#e2e8f0]")

            # Technical Specs
            yield Label("")
            yield Label("[bold #63b3ed]⚙️ Technical Specifications[/bold #63b3ed]")
            yield Label(f"[#718096]Format:[/#718096] [#e2e8f0]{self.data['quant']}[/#e2e8f0]")
            yield Label(f"[#718096]Score:[/#718096] [#4299e1]{self.data['score']}[/#4299e1]")
            yield Label(
                f"[#718096]Size:[/#718096] [#e2e8f0]{self.data['size']} {confidence_label}[/#e2e8f0]"
            )
            yield Label(f"[#718096]Fit:[/#718096] [{fit_class}]{self.data['fit']}[/{fit_class}]")
            yield Label(f"[#718096]Mode:[/#718096] [#48bb78]{self.data['mode']}[/#48bb78]")

            # Command
            yield Label("")
            yield Label("[bold #63b3ed]🚀 Run Command[/bold #63b3ed]")
            yield Label(cmd_text, id="cmd-box")

            # Buttons
            with Horizontal(id="button-row"):
                current_download_state = self.data.get("download_state", "idle")
                if current_download_state in {"queued", "downloading"}:
                    yield Button("⏸ Cancel", variant="warning", id="cancel-download-btn")
                else:
                    yield Button("⬇ Download", id="download-btn")
                yield Button("📋 Copy", id="copy-btn")
                yield Button("✕ Close", id="close-btn")

    @on(Button.Pressed, "#close-btn")
    def close_modal(self):
        self.dismiss()

    @on(Button.Pressed, "#download-btn")
    def start_download(self):
        start_fn = getattr(self.app, "start_model_download", None)
        if callable(start_fn):
            start_fn(self.data.copy())
        self.dismiss()

    @on(Button.Pressed, "#cancel-download-btn")
    def cancel_download(self):
        cancel_fn = getattr(self.app, "cancel_model_download", None)
        if callable(cancel_fn):
            cancel_fn(self.data.copy())
        self.dismiss()

    @on(Button.Pressed, "#copy-btn")
    def copy_command(self):
        if self.data["source"] == "Ollama":
            cmd_text = f"ollama run {self.data['name']}"
        else:
            repo_id = self.data.get("id", self.data["name"])
            cmd_text = f"huggingface-cli download {repo_id} --include '*.gguf'"

        try:
            import pyperclip

            pyperclip.copy(cmd_text)
            if hasattr(self.app, "update_status"):
                self.app.update_status("Command copied to clipboard!")
        except Exception:
            if hasattr(self.app, "update_status"):
                self.app.update_status(f"Copy failed. Command: {cmd_text}")


class SystemInfoWidget(Static):
    """Header widget displaying live CPU, RAM, GPU VRAM, and Ollama status."""

    def update_info(self, specs, ollama_running):
        """Re-render the widget with fresh hardware *specs* and Ollama running state."""
        gpu_color = "#f0ef8a" if specs["has_gpu"] else "#ff7f8f"
        ollama_status = (
            "[bold #4fe08a]running[/bold #4fe08a]"
            if ollama_running
            else "[bold #ff7f8f]stopped[/bold #ff7f8f]"
        )
        text = (
            f"[#93a7d9]CPU:[/#93a7d9] {specs['cpu_name']} ({specs['cpu_cores']} cores)  "
            f"[#5f6f97]|[/#5f6f97] [#93a7d9]RAM:[/#93a7d9] [bold #7edfff]{specs['ram_free']:.1f} GB avail[/bold #7edfff] / {specs['ram_total']:.1f} GB  "
            f"[#5f6f97]|[/#5f6f97] [#93a7d9]GPU:[/#93a7d9] [{gpu_color}]{specs['gpu_name']} ({specs['vram_free']:.1f} GB)[/{gpu_color}]  "
            f"[#5f6f97]|[/#5f6f97] [#93a7d9]Ollama:[/#93a7d9] {ollama_status}"
        )
        self.update(text)


class DownloadJobModal(ModalScreen):
    """Modal overlay showing download job history details with cancel / delete actions."""

    CSS = """
    DownloadJobModal {
        align: center middle;
        background: $background;
    }
    #job-modal {
        width: 50;
        height: auto;
        background: #0f141f;
        border: round #4a5568;
        padding: 1 2;
    }
    #job-header {
        background: #2d4a6f;
        padding: 0 2;
        margin: -1 -2 1 -2;
        height: 3;
    }
    #job-title {
        text-style: bold;
        color: #e2e8f0;
        text-align: center;
    }
    .job-label {
        color: #718096;
    }
    .job-value {
        color: #e2e8f0;
    }
    .job-status-downloading { color: #4299e1; }
    .job-status-queued { color: #ed8936; }
    .job-status-completed { color: #48bb78; }
    .job-status-failed { color: #f56565; }
    .job-status-cancelled { color: #ecc94b; }
    #job-button-row {
        layout: horizontal;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    #job-button-row Button {
        margin: 0 1;
    }
    #job-cancel-btn {
        background: #dd8448;
        color: white;
    }
    #job-cancel-delete-btn {
        background: #9b2c2c;
        color: white;
    }
    #job-delete-btn {
        background: #c05621;
        color: white;
    }
    #job-delete-all-btn {
        background: #9b2c2c;
        color: white;
    }
    #job-close-btn {
        background: #718096;
        color: white;
    }
    """

    def __init__(self, entry):
        super().__init__()
        self.entry = entry

    def compose(self) -> ComposeResult:
        state = (self.entry.get("state") or self.entry.get("status") or "idle").lower()
        progress = self.entry.get("detail") or self.entry.get("progress") or "-"

        # Determine status color class
        status_class = {
            "downloading": "job-status-downloading",
            "queued": "job-status-queued",
            "completed": "job-status-completed",
            "failed": "job-status-failed",
            "cancelled": "job-status-cancelled",
        }.get(state, "")

        # Check if this is an active download
        is_active = state in {"queued", "downloading", "running"}

        with Vertical(id="job-modal"):
            # Header
            yield Label(f"⬇ {self.entry.get('name', '-')}", id="job-title")

            # Info
            yield Label(
                f"[#718096]Source:[/#718096] [#e2e8f0]{self.entry.get('source', '-')}[/#e2e8f0]"
            )
            yield Label(
                f"[#718096]Publisher:[/#718096] [#e2e8f0]{self.entry.get('publisher', '-')}[/#e2e8f0]"
            )
            yield Label(f"[#718096]Status:[/#718096] [{status_class}]{state}[/{status_class}]")
            yield Label(f"[#718096]Progress:[/#718096] [#e2e8f0]{progress}[/#e2e8f0]")

            # Buttons
            with Horizontal(id="job-button-row"):
                if is_active:
                    yield Button("Cancel", id="job-cancel-btn")
                    yield Button("Cancel & Delete", id="job-cancel-delete-btn")
                else:
                    yield Button("Delete", id="job-delete-btn")
                    yield Button("Delete All", id="job-delete-all-btn")
                yield Button("Close", id="job-close-btn")

    @on(Button.Pressed, "#job-cancel-btn")
    def cancel(self):
        cancel_fn = getattr(self.app, "cancel_model_download", None)
        if callable(cancel_fn):
            cancel_fn(
                {
                    "source": self.entry.get("source", "-"),
                    "name": self.entry.get("name", "-"),
                    "id": self.entry.get("target_id", "").split(":", maxsplit=1)[1]
                    if ":" in str(self.entry.get("target_id", ""))
                    else self.entry.get("target_id", ""),
                }
            )
        self.dismiss()

    @on(Button.Pressed, "#job-cancel-delete-btn")
    def cancel_and_delete(self):
        """Cancel active download AND delete partial data"""
        model_data = {
            "source": self.entry.get("source", "-"),
            "name": self.entry.get("name", "-"),
            "id": self.entry.get("target_id", "").split(":", maxsplit=1)[1]
            if ":" in str(self.entry.get("target_id", ""))
            else self.entry.get("target_id", ""),
        }

        # Cancel the download first
        cancel_fn = getattr(self.app, "cancel_model_download", None)
        if callable(cancel_fn):
            cancel_fn(model_data)

        # Then delete the data
        delete_fn = getattr(self.app, "delete_download_entry", None)
        if callable(delete_fn):
            delete_fn(str(self.entry.get("target_id", "")), delete_data=True)

        self.dismiss()

    @on(Button.Pressed, "#job-close-btn")
    def close(self):
        self.dismiss()

    @on(Button.Pressed, "#job-delete-btn")
    def delete(self):
        delete_fn = getattr(self.app, "delete_download_entry", None)
        if callable(delete_fn):
            delete_fn(str(self.entry.get("target_id", "")), delete_data=False)
        self.dismiss()

    @on(Button.Pressed, "#job-delete-all-btn")
    def delete_all(self):
        """Delete entry AND downloaded/partial data"""
        delete_fn = getattr(self.app, "delete_download_entry", None)
        if callable(delete_fn):
            delete_fn(str(self.entry.get("target_id", "")), delete_data=True)
        self.dismiss()


class AIModelViewer(App):
    """Main TUI application for discovering, comparing, and downloading local LLM models.

    Connects to a background :mod:`download_service` over HTTP, polls hardware
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
    #search-panel { width: 42%; min-width: 38; margin-right: 1; height: 3; }
    #provider-panel { width: 1fr; height: 3; }
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
    #use-case-filter { height: 3; }
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
        ("h", "toggle_hidden_gems", "Hidden Gems"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
    ]

    RESULTS_COLUMN_LABELS = {
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

    def __init__(self):
        super().__init__()
        self.monitor = HardwareMonitor()
        self.all_results = []
        self.current_filter = "Ollama"
        self.use_case_filter = "all"
        self.hidden_gems_only = False
        self.ollama_running = False
        self.last_search_error = ""
        self.search_counter = 0
        self.active_search_id = 0
        self.hf_model_info_cache = {}
        self.search_cache = {}
        self.search_cache_ttl_seconds = config.settings.search_cache_ttl_seconds
        self.search_cache_max_entries = config.settings.search_cache_max_entries
        self.search_cache_ram_threshold_gb = config.settings.search_cache_ram_threshold_gb
        self.search_cache_vram_threshold_gb = config.settings.search_cache_vram_threshold_gb
        self.system_metrics_timer = None
        self.ollama_status_timer = None
        self.download_status_timer = None
        self.latest_specs = None
        self.active_downloads = set()
        self.download_spinner_index = 0
        self.download_spinner_frames = ["-", "\\", "|", "/"]
        self.download_registry = {}
        self.download_history_limit = config.settings.download_history_limit
        self.download_history_refresh_interval = config.settings.download_history_refresh_interval
        self.last_download_history_refresh_at = 0.0
        self.download_history_refresh_pending = False
        self.results_column_keys = []
        self.results_column_widths = {}
        self.current_page = 0
        self.page_size = config.settings.hf_search_limit
        self.max_pages = config.settings.hf_search_max_pages
        self.total_results = 0
        self.has_more_pages = True

    def compose(self) -> ComposeResult:
        yield SystemInfoWidget(id="header")
        with Horizontal(id="search-filters-row"):
            with Vertical(id="search-panel"):
                yield Input(
                    placeholder="Press / to search Ollama models (e.g., qwen, llama)",
                    id="search-input",
                )
            with Vertical(id="provider-panel"):
                with RadioSet(id="filter-set"):
                    yield RadioButton("Ollama", value=True, id="filter-ollama")
                    yield RadioButton("Hugging Face", id="filter-hf")
        with RadioSet(id="use-case-filter"):
            yield RadioButton("Any Use", value=True, id="uc-all")
            yield RadioButton("Chat", id="uc-chat")
            yield RadioButton("Coding", id="uc-coding")
            yield RadioButton("Vision", id="uc-vision")
            yield RadioButton("Reason", id="uc-reasoning")
            yield RadioButton("Math", id="uc-math")
            yield RadioButton("Embed", id="uc-embedding")
            yield RadioButton("General", id="uc-general")
        yield Checkbox("Hidden gems only (Hugging Face)", id="gem-toggle")
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

    def on_mount(self) -> None:
        """Initialise the UI, start the download service, and set up polling timers."""
        self.title = "AI Model Explorer"
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
        service_ok = ensure_service_running()
        if not service_ok:
            self.update_status(
                "Download service is unavailable or outdated. Restart the app/service."
            )
        else:
            self.sync_download_jobs_from_service(force=True)

        cache_db.init_db()
        cache_db.cleanup_old_entries()

        cached_specs = cache_db.get_hardware_snapshot()
        if cached_specs is not None:
            self.latest_specs = cached_specs
            self.query_one(SystemInfoWidget).update_info(cached_specs, check_ollama_running())

        self.refresh_download_history_table()
        self.last_download_history_refresh_at = time.monotonic()
        self._update_results_meta(0)
        self.update_status(
            "Ready. Search defaults to Ollama. Select 'Hugging Face' filter for HF models."
        )
        self.update_system_info()
        if not self.ollama_running:
            self.update_status(
                "Ready. Ollama is not running; local install/runtime features disabled. Search HF for more models."
            )
        self.system_metrics_timer = self.set_interval(
            config.settings.hardware_poll_interval, self.update_system_info
        )
        self.ollama_status_timer = self.set_interval(
            config.settings.ollama_status_poll_interval, self.poll_ollama_status
        )
        self.download_status_timer = self.set_interval(
            config.settings.ui_download_poll_interval, self.refresh_download_progress
        )

    def on_resize(self, event: events.Resize) -> None:
        _ = event
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

    def action_cycle_provider(self) -> None:
        cycle = ["Ollama", "Hugging Face"]
        current = self.current_filter if self.current_filter in cycle else "Ollama"
        next_filter = cycle[(cycle.index(current) + 1) % len(cycle)]
        self.current_filter = next_filter

        current_query = self.query_one("#search-input", Input).value.strip()
        if current_query:
            self.start_search(current_query)
            provider_name = "Hugging Face" if next_filter == "Hugging Face" else "Ollama"
            self.update_status(f"Provider: {next_filter}. Searching {provider_name}...")
        else:
            self.refresh_table()
            self.update_status(f"Provider filter: {next_filter}")

    def action_toggle_hidden_gems(self) -> None:
        checkbox = self.query_one("#gem-toggle", Checkbox)
        checkbox.value = not checkbox.value
        self.hidden_gems_only = bool(checkbox.value)
        self.refresh_table()
        self.update_status(
            "Hidden gems filter enabled."
            if self.hidden_gems_only
            else "Hidden gems filter disabled."
        )

    def _configure_results_table_columns(self, force: bool = False, refresh_rows: bool = False):
        table = self.query_one("#results-table", DataTable)
        available_width = max(table.size.width, self.size.width, 80)

        if available_width >= 140:
            next_keys = [
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
                "download",
            ]
        elif available_width >= 120:
            next_keys = [
                "inst",
                "source",
                "publisher",
                "name",
                "params",
                "score",
                "quant",
                "fit",
                "download",
            ]
        elif available_width >= 100:
            next_keys = [
                "inst",
                "source",
                "name",
                "params",
                "score",
                "fit",
                "download",
            ]
        else:
            next_keys = ["inst", "source", "name", "score", "download"]

        if not force and next_keys == self.results_column_keys:
            return

        min_widths = {
            "inst": 8,
            "source": 10,
            "publisher": 8,
            "name": 18,
            "params": 6,
            "use_case": 5,
            "score": 8,
            "quant": 6,
            "mode": 8,
            "fit": 7,
            "download": 4,
        }
        separator_budget = len(next_keys) + 2
        target_content_width = max(40, available_width - separator_budget)
        base_widths = {key: min_widths[key] for key in next_keys}
        min_total = sum(base_widths.values())
        extra = max(0, target_content_width - min_total)

        expandable = ["name", "publisher", "use_case", "download"]
        while extra > 0:
            expanded = False
            for key in expandable:
                if key in base_widths and extra > 0:
                    base_widths[key] += 1
                    extra -= 1
                    expanded = True
            if not expanded:
                break

        table.clear(columns=True)
        for key in next_keys:
            width = base_widths[key]
            table.add_column(
                self._format_header_label(self.RESULTS_COLUMN_LABELS[key], width),
                width=width,
                key=key,
            )

        self.results_column_keys = next_keys
        self.results_column_widths = {key: base_widths[key] for key in next_keys}

        if refresh_rows:
            self.refresh_table()

    def _truncate_cell(self, value, max_len):
        text = str(value or "-")
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."

    def _truncate_plain_cell(self, value, max_len):
        text = re.sub(r"\[[^\]]+\]", "", str(value or "-")).strip()
        if not text:
            text = "-"
        return self._truncate_cell(text, max_len)

    def _format_header_label(self, label, width):
        if width <= 0:
            return str(label)
        text = self._truncate_plain_cell(label, width)
        return text.ljust(width)

    def _align_plain_cell(self, value, width, align="left"):
        text = str(value or "-")
        if width <= 0:
            return text
        if len(text) > width:
            text = self._truncate_cell(text, width)
        if align == "center":
            return text.center(width)
        if align == "right":
            return text.rjust(width)
        return text.ljust(width)

    def _blank_result_row(self):
        return {
            "inst": "-",
            "source": "-",
            "publisher": "-",
            "name": "-",
            "params": "-",
            "use_case": "-",
            "score": "-",
            "quant": "-",
            "mode": "-",
            "fit": "-",
            "download": "-",
        }

    def _fit_cell_markup(self, fit_text):
        plain = self._truncate_plain_cell(fit_text, 20)
        width = max(5, self.results_column_widths.get("fit", 7) - 1)
        plain = self._align_plain_cell(plain, width, "left")
        lowered = plain.lower()
        if "perfect" in lowered or lowered == "fit":
            return f"[bold #4fe08a]{plain}[/bold #4fe08a]"
        if "partial" in lowered or "slow" in lowered:
            return f"[bold #f2c46d]{plain}[/bold #f2c46d]"
        if "no fit" in lowered or "tight" in lowered:
            return f"[bold #ff7f8f]{plain}[/bold #ff7f8f]"
        return plain

    def _mode_cell_markup(self, mode_text):
        plain = self._truncate_plain_cell(mode_text, 20)
        width = max(5, self.results_column_widths.get("mode", 8) - 1)
        plain = self._align_plain_cell(plain, width, "left")
        lowered = plain.lower()
        if "gpu+cpu" in lowered:
            return f"[bold #f2c46d]{plain}[/bold #f2c46d]"
        if "gpu" in lowered:
            return f"[bold #4fe08a]{plain}[/bold #4fe08a]"
        if "cpu" in lowered:
            return f"[bold #f2c46d]{plain}[/bold #f2c46d]"
        return f"[#ff7f8f]{plain}[/#ff7f8f]" if plain != "-" else plain

    def _installed_cell_markup(self, installed_text):
        plain = self._truncate_plain_cell(installed_text, 8)
        if "✔" in plain or "installed" in plain.lower():
            marker = "●"
        else:
            marker = "·"
        width = max(3, self.results_column_widths.get("inst", 7) - 2)
        return (
            f"[bold #4fe08a]{self._align_plain_cell(marker, width, 'left')}[/bold #4fe08a]"
            if marker == "●"
            else f"[#6b789e]{self._align_plain_cell(marker, width, 'left')}[/#6b789e]"
        )

    def _source_cell_markup(self, source_text):
        plain = self._truncate_plain_cell(source_text, 20)
        width = max(10, self.results_column_widths.get("source", 13) - 1)
        plain = self._align_plain_cell(plain, width, "left")
        lowered = plain.lower()
        if "ollama" in lowered:
            return f"[bold #7edfff]{plain}[/bold #7edfff]"
        if "hugging" in lowered:
            return f"[bold #b6a3ff]{plain}[/bold #b6a3ff]"
        return f"[#9bb1e0]{plain}[/#9bb1e0]"

    def _score_cell_markup(self, score_text):
        plain = self._truncate_plain_cell(score_text, 24)
        match = re.search(r"(\d+(?:\.\d+)?)([KkMm]?)", plain)
        if not match:
            return "[#6b789e]-[/#6b789e]"

        value = float(match.group(1))
        suffix = match.group(2).upper()
        if suffix == "K":
            numeric = value * 1000.0
        elif suffix == "M":
            numeric = value * 1000000.0
        else:
            numeric = value

        if numeric >= 1000000:
            color = "#4fe08a"
        elif numeric >= 100000:
            color = "#7edfff"
        else:
            color = "#f2c46d"
        width = max(5, self.results_column_widths.get("score", 8) - 1)
        shown = self._align_plain_cell(f"{value:g}{suffix}", width, "left")
        return f"[bold {color}]{shown}[/bold {color}]"

    def _use_case_cell_markup(self, use_case_text):
        plain = self._truncate_plain_cell(use_case_text, 20)
        width = max(4, self.results_column_widths.get("use_case", 6) - 1)
        plain = self._align_plain_cell(plain, width, "left")
        lowered = plain.lower()
        if "coding" in lowered:
            return f"[#86a7ff]{plain}[/#86a7ff]"
        if "chat" in lowered:
            return f"[#8e97c7]{plain}[/#8e97c7]"
        if "reason" in lowered:
            return f"[#b89df3]{plain}[/#b89df3]"
        if "vision" in lowered:
            return f"[#9fc6ff]{plain}[/#9fc6ff]"
        return f"[#8ea3cf]{plain}[/#8ea3cf]"

    def _download_cell_markup(self, download_text, width=None):
        if width is None:
            width = max(3, self.results_column_widths.get("download", 4) - 1)
        plain = self._truncate_plain_cell(download_text, 24)
        aligned = self._align_plain_cell(plain, width, "left")
        lowered = plain.lower()
        if "downloading" in lowered or "queued" in lowered:
            return f"[bold #f2c46d]{aligned}[/bold #f2c46d]"
        if "done" in lowered or "installed" in lowered:
            return f"[bold #4fe08a]{aligned}[/bold #4fe08a]"
        if "failed" in lowered or "error" in lowered:
            return f"[bold #ff7f8f]{aligned}[/bold #ff7f8f]"
        return f"[#9bb1e0]{aligned}[/#9bb1e0]"

    def _row_cells_for_current_layout(self, row_data):
        return [row_data.get(key, "-") for key in self.results_column_keys]

    def _update_results_meta(self, shown_count):
        total = len(self.all_results)
        self.query_one("#results-meta", Static).update(
            f"Models ({shown_count} shown / {total} total)"
        )

    def update_status(self, text):
        """Update the status bar at the bottom of the screen with *text*."""
        self.query_one("#status-bar", Static).update(text)
        specs = self.monitor.get_specs()
        self.latest_specs = specs
        cache_db.set_hardware_snapshot(specs)
        self.poll_ollama_status(refresh_only=True)

    def update_system_info(self):
        """Update the system info header widget with current hardware specs."""
        specs = self.monitor.get_specs()
        self.latest_specs = specs
        cache_db.set_hardware_snapshot(specs)
        running_now = check_ollama_running()
        self.query_one(SystemInfoWidget).update_info(specs, running_now)

    def _find_model_by_target_id(self, target_id):
        """Find a model in all_results by its target_id."""
        for item in self.all_results:
            if download_target_id(item) == target_id:
                return item
        return None

    def poll_ollama_status(self, refresh_only=False):
        running_now = check_ollama_running()
        state_changed = running_now != self.ollama_running
        self.ollama_running = running_now

        specs = self.latest_specs if self.latest_specs is not None else self.monitor.get_specs()
        self.query_one(SystemInfoWidget).update_info(specs, running_now)

        if refresh_only or not state_changed:
            return
        if running_now:
            self.update_status("Ollama started. Local runtime features enabled.")
        else:
            self.update_status("Ollama stopped. Local runtime features disabled.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        self.start_search(query)

    def start_search(self, query: str) -> None:
        if not query:
            return
        providers = self._get_search_providers()
        provider_prefix = "hf" if providers == ["huggingface"] else "ollama"

        # Ollama doesn't support pagination, only HF does
        if providers == ["huggingface"]:
            query_key = f"{provider_prefix}:{query.lower()}:page{self.current_page}"
        else:
            query_key = f"{provider_prefix}:{query.lower()}"
            self.current_page = 0  # Reset page for Ollama

        current_specs = self.monitor.get_specs()
        self.last_search_error = ""
        self.search_counter += 1
        self.active_search_id = self.search_counter
        table = self.query_one("#results-table", DataTable)
        table.clear()
        table.loading = False
        self._update_results_meta(0)

        provider_name = "Hugging Face" if providers == ["huggingface"] else "Ollama"
        self.on_search_progress(self.active_search_id, f"Searching {provider_name}: {query}")

        cached = self._get_cached_search(query_key, current_specs)
        if cached:
            self.all_results = [item.copy() for item in cached["results"]]
            self._ensure_download_fields()
            self.last_search_error = cached["error"]
            if "has_more_pages" in cached:
                self.has_more_pages = cached["has_more_pages"]
            self.on_search_completed(self.active_search_id)
            cache_msg = (
                f" (cached page {self.current_page + 1})"
                if providers == ["huggingface"]
                else " (cached)"
            )
            self.update_status(f"Loaded{cache_msg}")
            return

        self.run_search_worker(query, query_key, self.active_search_id, providers)

    def on_search_progress(self, search_id: int, message: str) -> None:
        if search_id != self.active_search_id:
            return

        self.update_status(message)
        table = self.query_one("#results-table", DataTable)
        self._configure_results_table_columns()
        table.clear()
        row_data = self._blank_result_row()
        row_data["inst"] = "[cyan]...[/cyan]"
        row_data["source"] = "System"
        name_width = max(16, self.results_column_widths.get("name", 24))
        short_message = self._truncate_cell(f"Status: {message}", max(10, name_width - 1))
        row_data["name"] = short_message
        row_data["download"] = self._download_cell_markup("Working")
        table.add_row(*self._row_cells_for_current_layout(row_data), key="search-progress")
        self._update_results_meta(0)
        table.move_cursor(row=0, animate=False, scroll=False)
        table.scroll_to(x=0, y=0, animate=False, immediate=True)

    def _get_search_providers(self) -> list[str]:
        """Return list of providers to search based on current_filter."""
        if self.current_filter == "Hugging Face":
            return ["huggingface"]
        else:
            return ["ollama"]

    def _is_cache_compatible(self, current_specs, cached_specs):
        if not cached_specs:
            return True
        if current_specs.get("has_gpu") != cached_specs.get("has_gpu"):
            return False

        ram_delta = abs(current_specs.get("ram_free", 0.0) - cached_specs.get("ram_free", 0.0))
        if ram_delta > self.search_cache_ram_threshold_gb:
            return False

        if current_specs.get("has_gpu"):
            vram_delta = abs(
                current_specs.get("vram_free", 0.0) - cached_specs.get("vram_free", 0.0)
            )
            if vram_delta > self.search_cache_vram_threshold_gb:
                return False

        return True

    def _get_cached_search(self, query_key, current_specs):
        entry = self.search_cache.get(query_key)
        if not entry:
            return None

        age = time.monotonic() - entry["timestamp"]
        if age > self.search_cache_ttl_seconds:
            self.search_cache.pop(query_key, None)
            return None
        if not self._is_cache_compatible(current_specs, entry.get("specs")):
            self.search_cache.pop(query_key, None)
            return None
        return entry

    def _store_cached_search(self, query_key, results, error):
        specs = self.monitor.get_specs()
        self.search_cache[query_key] = {
            "timestamp": time.monotonic(),
            "results": [item.copy() for item in results],
            "error": error,
            "has_more_pages": self.has_more_pages,
            "specs": {
                "has_gpu": specs.get("has_gpu", False),
                "ram_free": specs.get("ram_free", 0.0),
                "vram_free": specs.get("vram_free", 0.0),
            },
        }
        if len(self.search_cache) > self.search_cache_max_entries:
            oldest_key = min(
                self.search_cache,
                key=lambda key: self.search_cache[key]["timestamp"],
            )
            self.search_cache.pop(oldest_key, None)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "use-case-filter":
            pid = event.pressed.id
            self.use_case_filter = pid.replace("uc-", "") if pid else "all"
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
        if page_num < 0:
            return
        if page_num >= self.max_pages:
            self.update_status(f"Reached page limit ({self.max_pages}).")
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
        is_hf = providers == ["huggingface"]

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
            entry = self.download_registry.get(target_id)

            # If no entry found, still try to show modal with basic info
            if not entry:
                # Try to create entry from row data
                entry = {
                    "target_id": target_id,
                    "source": target_id.split(":")[0] if ":" in target_id else "unknown",
                    "name": target_id.split(":")[1] if ":" in target_id else target_id,
                    "state": "idle",
                    "detail": "External download",
                }

            # Check if action column was clicked (column index 5)
            cursor_column = data_table.cursor_column
            if cursor_column == 5:  # Action column
                state = entry.get("state", "idle")

                # Check if this is external/unkown
                is_external = state == "idle" and not any(
                    x in str(entry.get("detail", "")).lower()
                    for x in ["download", "pull", "queue", "completed", "failed", "cancel"]
                )

                if is_external:
                    # External download - can't manage through our system
                    self.update_status("External download - cannot manage through this app")
                    return

                if state in {"downloading", "queued", "running"}:
                    # Cancel download
                    self.cancel_model_download(
                        {
                            "source": entry.get("source", "-"),
                            "name": entry.get("name", "-"),
                            "id": target_id.split(":", maxsplit=1)[1]
                            if ":" in target_id
                            else target_id,
                        }
                    )
                else:
                    # Delete entry (just history, keeps data for resume)
                    self.delete_download_entry(target_id, delete_data=False)
            else:
                # Show detail modal
                self.push_screen(DownloadJobModal(entry))
            return

        if data_table is not None and data_table.id != "results-table":
            return
        row_key = str(event.row_key.value)
        selected_model = next(
            (
                item
                for item in self.all_results
                if f"{item['source']}:{item.get('id', item['name'])}" == row_key
            ),
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
        target_id = download_target_id(model)
        try:
            response = cancel_job(target_id)
            _ = response.get("job")
            self.update_status(f"Cancel requested: {model.get('name', target_id)}")
            self.sync_download_jobs_from_service(force=True)
            self.refresh_download_history_table()
            self.refresh_table()  # Also refresh main results table
        except HTTPError as exc:
            detail = "Failed to cancel download through service."
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                error_text = payload.get("error")
                if error_text:
                    detail = f"Cancel failed: {error_text}"
            except Exception:
                pass
            self.update_status(detail)
        except Exception:
            self.update_status("Failed to cancel download through service.")

    def delete_download_entry(self, target_id, delete_data=False):
        """Delete download entry from history.

        Args:
            target_id: The download target ID
            delete_data: If True, also delete downloaded/partial data (e.g., ollama rm)
        """
        entry = self.download_registry.get(target_id)
        source = entry.get("source", "").lower() if entry else ""
        model_name = entry.get("name", "") if entry else ""
        state = entry.get("state", "idle") if entry else "idle"

        # If deleting data for active download, cancel it first
        if delete_data and state in {"downloading", "queued", "running"}:
            try:
                # Cancel the job through the service
                cancel_job(target_id)
                time.sleep(1)  # Wait a moment for cancellation
                self.update_status(f"Download canceled: {model_name}")
            except Exception as e:
                self.update_status(f"Could not cancel download: {e}")

        # If deleting data, remove the actual model/partial files
        if delete_data and source == "ollama" and model_name:
            try:
                # Try with :latest tag first, then without
                result = subprocess.run(
                    ["ollama", "rm", model_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    self.update_status(f"✅ Deleted model: {model_name}")
                else:
                    # Try with :latest suffix
                    result2 = subprocess.run(
                        ["ollama", "rm", f"{model_name}:latest"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result2.returncode == 0:
                        self.update_status(f"✅ Deleted model: {model_name}:latest")
                    else:
                        self.update_status(f"⚠️ Could not delete: {result2.stderr.strip()}")
            except subprocess.TimeoutExpired:
                self.update_status(f"⚠️ Timeout deleting model: {model_name}")
            except Exception as e:
                self.update_status(f"⚠️ Delete error: {str(e)}")

        try:
            delete_job(target_id)
        except HTTPError as exc:
            detail = "Failed to delete download entry."
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                error_text = payload.get("error")
                if error_text == "cannot delete active job":
                    detail = "Cannot delete active job. Cancel it first."
                elif error_text:
                    detail = f"Delete failed: {error_text}"
            except Exception:
                pass
            self.update_status(detail)
            return
        except Exception as exc:
            self.update_status(f"Failed to delete download entry: {exc}")
            return

        deleted_entry = self.download_registry.pop(target_id, None)
        if deleted_entry is not None:
            source_key = str(deleted_entry.get("source", "")).strip().lower()
            name_key = str(deleted_entry.get("name", "")).strip().lower()
        else:
            source_key = ""
            name_key = ""

        for item in self.all_results:
            item_target = download_target_id(item)
            if item_target == target_id:
                item["download_state"] = "idle"
                item["download_label"] = "Idle"
                item["download_detail"] = ""
                continue
            if (
                source_key
                and name_key
                and str(item.get("source", "")).strip().lower() == source_key
                and str(item.get("name", "")).strip().lower() == name_key
            ):
                item["download_state"] = "idle"
                item["download_label"] = "Idle"
                item["download_detail"] = ""

        self.refresh_table()
        self.refresh_download_history_table()
        self.update_status("Download entry deleted.")

    def start_model_download(self, model):
        if not ensure_service_running():
            self.update_status("Download service is unavailable.")
            return

        target_id = download_target_id(model)
        try:
            response = create_job(model)
            queued = bool(response.get("queued"))
            if queued:
                self.update_status(f"Download queued: {model.get('name', target_id)}")
            else:
                self.update_status(f"Download already active: {model.get('name', target_id)}")
            self.sync_download_jobs_from_service(force=True)
            self.refresh_download_history_table()
        except Exception as exc:
            self.update_status(f"Failed to queue download: {exc}")

    def _record_download_entry(self, target_id, model=None, state=None, label=None, detail=None):
        existing = self.download_registry.get(target_id, {})

        source = existing.get("source", "-")
        publisher = existing.get("publisher", "-")
        name = existing.get("name", target_id)

        if model is None:
            model = self._find_model_by_target_id(target_id)
        if model is not None:
            source = model.get("source", source)
            publisher = model.get("publisher", publisher)
            name = model.get("name", name)

        entry = {
            "target_id": target_id,
            "source": source,
            "publisher": publisher,
            "name": name,
            "created_at": existing.get("created_at", time.time()),
            "state": state if state is not None else existing.get("state", "idle"),
            "label": label if label is not None else existing.get("label", "Idle"),
            "detail": detail if detail is not None else existing.get("detail", ""),
            "updated_at": time.time(),
        }
        self.download_registry[target_id] = entry

        if len(self.download_registry) > self.download_history_limit:
            sorted_keys = sorted(
                self.download_registry,
                key=lambda key: self.download_registry[key].get("updated_at", 0),
            )
            for key in sorted_keys[: len(self.download_registry) - self.download_history_limit]:
                self.download_registry.pop(key, None)

    def sync_download_jobs_from_service(self, force=False):
        try:
            jobs = list_jobs(limit=self.download_history_limit)
        except Exception:
            return

        running_targets = set()
        for job in jobs:
            target_id = job.get("target_id")
            if not target_id:
                continue
            target_id = download_target_id(
                {
                    "source": job.get("source", "unknown"),
                    "id": target_id.split(":", maxsplit=1)[1] if ":" in target_id else target_id,
                }
            )
            mapped_status = map_service_job_status(
                job.get("status", "idle"),
                cancel_requested=bool(job.get("cancel_requested")),
                detail=job.get("detail", ""),
            )
            label = label_for_state(mapped_status)

            detail = job.get("progress") or job.get("detail") or ""
            self._record_download_entry(
                target_id,
                model={
                    "source": job.get("source", "-"),
                    "publisher": job.get("publisher", "-"),
                    "name": job.get("name", target_id),
                },
                state=mapped_status,
                label=label,
                detail=detail,
            )
            self.download_registry[target_id]["created_at"] = job.get(
                "created_at",
                self.download_registry[target_id].get("created_at", time.time()),
            )
            self.download_registry[target_id]["updated_at"] = job.get(
                "updated_at",
                self.download_registry[target_id].get("updated_at", time.time()),
            )
            if is_active_state(mapped_status):
                running_targets.add(target_id)

        self.active_downloads = running_targets
        state_changed = self._ensure_download_fields()
        if force:
            self.refresh_table()
            self.refresh_download_history_table()
        elif state_changed:
            self.refresh_table()

    def refresh_download_history_table(self):
        table = self.query_one("#download-history-table", DataTable)
        table.clear()

        entries = sorted(
            self.download_registry.values(),
            key=lambda item: (
                item.get("created_at", 0),
                item.get("target_id", ""),
            ),
            reverse=True,
        )

        for entry in entries[: self.download_history_limit]:
            state = entry.get("state", "idle")
            target_id = entry.get("target_id", "-")

            # Check if this is an external download (not tracked by our system)
            is_external = state == "idle" and not any(
                x in str(entry.get("detail", "")).lower()
                for x in ["download", "pull", "queue", "completed", "failed", "cancel"]
            )

            # Create action button based on state
            if is_external:
                action_btn = "External"
            elif is_active_state(state):
                action_btn = "Cancel"
            else:
                action_btn = "Delete"

            table.add_row(
                entry.get("source", "-"),
                entry.get("publisher", "-"),
                entry.get("name", "-"),
                self._download_status_text_from_state(
                    entry.get("state", "idle"),
                    entry.get("label", "Idle"),
                ),
                entry.get("detail", ""),
                action_btn,
                key=target_id,
            )

    def request_download_history_refresh(self, force=False):
        now = time.monotonic()
        if force or (
            now - self.last_download_history_refresh_at >= self.download_history_refresh_interval
        ):
            self.refresh_download_history_table()
            self.last_download_history_refresh_at = now
            self.download_history_refresh_pending = False
            return
        self.download_history_refresh_pending = True

    def _download_status_text_from_state(self, state, label=None):
        return state_markup_from_state_and_label(
            state,
            label,
            unknown_is_external=True,
        )

    def _set_download_state(
        self,
        target_id,
        state,
        label,
        detail,
        model=None,
        refresh_results=True,
        refresh_history=True,
    ):
        self._record_download_entry(target_id, model=model, state=state, label=label, detail=detail)
        model = self._find_model_by_target_id(target_id)
        if model and refresh_results:
            model["download_state"] = state
            model["download_label"] = label
            model["download_detail"] = detail
            self.refresh_table()
        elif model:
            model["download_state"] = state
            model["download_label"] = label
            model["download_detail"] = detail
        if refresh_history:
            self.request_download_history_refresh()

    def _download_cell_text(self, model):
        state = model.get("download_state", "idle")
        return state_markup_from_state_and_label(state, model.get("download_label"))

    def _ensure_download_fields(self):
        changed = False
        for item in self.all_results:
            target_id = download_target_id(item)
            entry = self.download_registry.get(target_id)
            if entry is None:
                source_key = str(item.get("source", "")).strip().lower()
                name_key = str(item.get("name", "")).strip().lower()
                for value in self.download_registry.values():
                    if (
                        str(value.get("source", "")).strip().lower() == source_key
                        and str(value.get("name", "")).strip().lower() == name_key
                    ):
                        entry = value
                        break
            if entry:
                next_state = entry.get("state", "idle")
                next_label = entry.get("label", "Idle")
                if (
                    item.get("download_state") != next_state
                    or item.get("download_label") != next_label
                ):
                    changed = True
                item["download_state"] = next_state
                item["download_label"] = next_label
                item["download_detail"] = entry.get("detail", "")
            else:
                if item.get("download_state") not in {None, "idle"}:
                    changed = True
                item.setdefault("download_state", "idle")
                item.setdefault("download_label", "Idle")
                item.setdefault("download_detail", "")
        return changed

    def refresh_download_progress(self):
        self.sync_download_jobs_from_service(force=False)
        self.refresh_download_debug()
        # Always refresh history table to catch state changes
        self.refresh_download_history_table()
        if not self.active_downloads:
            if self.download_history_refresh_pending:
                self.request_download_history_refresh()
            return
        if any(
            is_active_state(self.download_registry.get(target_id, {}).get("state"))
            for target_id in self.active_downloads
        ):
            self.request_download_history_refresh()

    def refresh_download_debug(self):
        try:
            debug = get_active_download_debug()
            count = debug.get("count", 0)
            has_duplicates = bool(debug.get("has_duplicates", False))
            worker_alive = bool(debug.get("worker_alive", True))
            dup_text = "yes" if has_duplicates else "no"
            worker_text = "up" if worker_alive else "down"
            self.query_one("#downloads-debug", Static).update(
                f"Workers: {count} ({worker_text}) | Duplicates: {dup_text}"
            )
        except Exception:
            try:
                health = get_service_health()
                version = health.get("version", "unknown")
                self.query_one("#downloads-debug", Static).update(
                    f"Workers: legacy service ({version}) | Duplicates: unknown"
                )
            except Exception:
                self.query_one("#downloads-debug", Static).update(
                    "Workers: unavailable | Duplicates: unknown"
                )

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
        self.open_model_detail_modal(enriched_model)
        self.update_status("Detailed metadata loaded.")

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
        ollama_has_more = False

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
            ollama_results, ollama_errors, ollama_has_more = search_ollama_models(
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

        self.call_from_thread(self.on_search_progress, search_id, "Finalizing search results...")
        self.all_results = ollama_results + hf_results
        self._ensure_download_fields()
        self.last_search_error = " | ".join((ollama_errors + hf_errors)[:2])

        # Determine if there are more pages based on which provider was searched
        # Note: Ollama doesn't support page-based pagination, only HF does
        if "huggingface" in providers and len(hf_results) > 0:
            self.has_more_pages = len(hf_results) == self.page_size
        elif "ollama" in providers and len(ollama_results) > 0:
            # Ollama returns all results at once, no pagination possible
            self.has_more_pages = False
        else:
            self.has_more_pages = False

        provider_name = "HF" if "huggingface" in providers else "Ollama"
        result_count = len(hf_results) if "huggingface" in providers else len(ollama_results)
        if "ollama" in providers:
            self.call_from_thread(
                self.update_status,
                f"Ollama: {result_count} results (pagination not supported by Ollama.com)",
            )
        else:
            self.call_from_thread(
                self.update_status,
                f"HF: {result_count} results, has_more={self.has_more_pages}, page={self.current_page}",
            )

        self._store_cached_search(query_key, self.all_results, self.last_search_error)
        self.call_from_thread(self.on_search_completed, search_id)

    def on_search_completed(self, search_id: int) -> None:
        if search_id != self.active_search_id:
            return

        table = self.query_one("#results-table", DataTable)
        table.loading = False
        self.refresh_table()

        if table.row_count > 0:
            total = len(self.all_results)
            shown = table.row_count
            page_info = f" (Page {self.current_page + 1})" if self.current_page > 0 else ""
            self.update_status(f"{shown} results listed{page_info}.")
        elif self.last_search_error:
            self.update_status(self.last_search_error[:120])
        else:
            providers = self._get_search_providers()
            if providers == ["ollama"]:
                self.update_status("No Ollama results. Try 'Hugging Face' filter for more models.")
            else:
                self.update_status("No results found.")

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
        self._configure_results_table_columns()
        table = self.query_one("#results-table", DataTable)
        prev_cursor_row = table.cursor_row
        prev_scroll_x = table.scroll_x
        prev_scroll_y = table.scroll_y
        table.clear()
        added = set()

        filtered_results = []
        for result in self.all_results:
            if result["source"] != self.current_filter:
                continue
            if self.use_case_filter != "all" and result.get("use_case_key") != self.use_case_filter:
                continue
            if self.hidden_gems_only and not result.get("is_hidden_gem", False):
                continue
            filtered_results.append(result)

        if self.hidden_gems_only:
            filtered_results.sort(
                key=lambda item: (
                    item.get("is_hidden_gem", False),
                    item.get("gem_score", 0.0),
                    item.get("downloads", 0),
                ),
                reverse=True,
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
                max(5, self.results_column_widths.get("fit", 7) - 1),
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

            unique_key = f"{result['source']}:{result.get('id', result['name'])}"
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
