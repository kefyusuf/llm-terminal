import time
import subprocess
import re

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Grid, Vertical
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

from hardware import HardwareMonitor, check_ollama_running
from download_manager import build_download_command, download_target_id
from providers.hf_provider import enrich_hf_model_details, search_hf_models
from providers.ollama_provider import get_installed_ollama_models, search_ollama_models


class ModelDetailModal(ModalScreen):
    CSS = """
    ModelDetailModal {
        align: center middle;
        background: $background;
    }
    #modal-container {
        width: 60%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #modal-title {
        text-align: center;
        text-style: bold;
        background: $primary;
        color: white;
        padding: 1;
        margin-bottom: 1;
    }
    #info-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        height: auto;
    }
    .label-key {
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
    }
    #cmd-box {
        background: $panel;
        border: solid green;
        padding: 1;
        margin: 1 0;
        text-align: center;
        color: $accent-lighten-2;
    }
    #close-btn { width: 100%; margin-top: 1; }
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
            "[green]Exact[/green]"
            if size_source == "exact"
            else "[yellow]Estimated[/yellow]"
        )

        with Vertical(id="modal-container"):
            yield Label(f"{self.data['name']}", id="modal-title")

            with Grid(id="info-grid"):
                yield Label(
                    f"[bold]Source:[/bold] {self.data['source']} / {self.data.get('publisher', '-')}"
                )
                yield Label(f"[bold]Provider:[/bold] {self.data['provider']}")
                yield Label(f"[bold]Use Case:[/bold] {self.data['use_case']}")
                yield Label(f"[bold]Parameters:[/bold] {self.data['params']}")
                yield Label(f"[bold]Format:[/bold] {self.data['quant']}")
                yield Label(f"[bold]Score:[/bold] {self.data['score']}")
                yield Label(f"[bold]Estimated Size:[/bold] {self.data['size']}")
                yield Label(f"[bold]Size Confidence:[/bold] {confidence_label}")
                yield Label(f"[bold]Hardware Fit:[/bold] {self.data['fit']}")
                yield Label(f"[bold]Run Mode:[/bold] {self.data['mode']}")

            yield Label("Run / Download Command:", classes="label-key")
            yield Label(cmd_text, id="cmd-box")
            yield Button("Download Now", variant="primary", id="download-btn")
            yield Button("Cancel Download", variant="warning", id="cancel-download-btn")
            yield Button("Close", variant="error", id="close-btn")

    @on(Button.Pressed, "#close-btn")
    def close_modal(self):
        self.dismiss()

    @on(Button.Pressed, "#download-btn")
    def start_download(self):
        self.dismiss("download")

    @on(Button.Pressed, "#cancel-download-btn")
    def cancel_download(self):
        self.dismiss("cancel_download")


class SystemInfoWidget(Static):
    def update_info(self, specs, ollama_running):
        gpu_color = "green" if specs["has_gpu"] else "red"
        ollama_status = (
            "[bold green]✔ Running[/bold green]"
            if ollama_running
            else "[bold red]X Stopped[/bold red]"
        )
        text = (
            f"[bold]CPU:[/bold] {specs['cpu_name']} ({specs['cpu_cores']} cores) | "
            f"[bold]RAM:[/bold][cyan]{specs['ram_free']:.1f} GB Free[/cyan] / {specs['ram_total']:.1f} GB | "
            f"[bold]GPU:[/bold] [{gpu_color}]{specs['gpu_name']} ({specs['vram_free']:.1f} GB Free)[/{gpu_color}] | "
            f"[bold]Ollama:[/bold] {ollama_status}"
        )
        self.update(text)


class AIModelViewer(App):
    CSS = """
    Screen { layout: vertical; padding: 1; }
    SystemInfoWidget { height: 3; border: round cyan; content-align: center middle; margin-bottom: 1; }
    Input { width: 100%; margin-bottom: 1; }
    RadioSet { layout: horizontal; width: 100%; height: 3; border: none; align: center middle; margin-bottom: 1; }
    #use-case-filter { height: 3; }
    #gem-toggle { margin-bottom: 1; }
    DataTable { height: 1fr; border: round grey; }
    #status-bar { height: 1; color: $text-muted; margin-top: 1; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Previous"),
    ]

    def __init__(self):
        super().__init__()
        self.monitor = HardwareMonitor()
        self.all_results = []
        self.current_filter = "All"
        self.use_case_filter = "all"
        self.hidden_gems_only = False
        self.ollama_running = False
        self.last_search_error = ""
        self.search_counter = 0
        self.active_search_id = 0
        self.hf_model_info_cache = {}
        self.search_cache = {}
        self.search_cache_ttl_seconds = 90
        self.search_cache_max_entries = 20
        self.search_cache_ram_threshold_gb = 1.0
        self.search_cache_vram_threshold_gb = 1.0
        self.system_metrics_timer = None
        self.ollama_status_timer = None
        self.latest_specs = None
        self.active_downloads = set()
        self.download_processes = {}
        self.cancelled_downloads = set()

    def compose(self) -> ComposeResult:
        yield SystemInfoWidget(id="header")
        yield Input(
            placeholder="🔍 Search models (e.g., llama, qwen) and press Enter...",
            id="search-input",
        )
        with RadioSet(id="filter-set"):
            yield RadioButton("All", value=True, id="filter-all")
            yield RadioButton("Ollama", id="filter-ollama")
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
        yield Checkbox(
            "Hidden gems only (HF: high downloads, low likes)", id="gem-toggle"
        )
        yield DataTable(id="results-table", cursor_type="row")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "AI Model Explorer"
        table = self.query_one(DataTable)
        table.add_columns(
            "Installed",
            "Source",
            "Publisher",
            "Model Name",
            "Params",
            "Use Case",
            "Score",
            "Format",
            "Runtime",
            "Fit",
            "Size",
            "Download",
        )
        self.update_status("Ready. Enter a model query and press Enter.")
        self.update_system_info()
        if not self.ollama_running:
            self.update_status(
                "Ready. Ollama is not running; local install/runtime features are disabled."
            )
        self.system_metrics_timer = self.set_interval(3.0, self.update_system_info)
        self.ollama_status_timer = self.set_interval(0.5, self.poll_ollama_status)

    def update_status(self, text):
        self.query_one("#status-bar", Static).update(text)

    def update_system_info(self):
        specs = self.monitor.get_specs()
        self.latest_specs = specs
        self.poll_ollama_status(refresh_only=True)

    def poll_ollama_status(self, refresh_only=False):
        running_now = check_ollama_running()
        state_changed = running_now != self.ollama_running
        self.ollama_running = running_now

        specs = (
            self.latest_specs
            if self.latest_specs is not None
            else self.monitor.get_specs()
        )
        self.query_one(SystemInfoWidget).update_info(specs, running_now)

        if refresh_only or not state_changed:
            return
        if running_now:
            self.update_status("Ollama started. Local runtime features enabled.")
        else:
            self.update_status("Ollama stopped. Local runtime features disabled.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        query_key = query.lower()
        current_specs = self.monitor.get_specs()
        self.last_search_error = ""
        self.search_counter += 1
        self.active_search_id = self.search_counter
        table = self.query_one(DataTable)
        table.clear()
        table.loading = True
        self.update_status(f"Searching: {query}")

        cached = self._get_cached_search(query_key, current_specs)
        if cached:
            self.all_results = [item.copy() for item in cached["results"]]
            self._ensure_download_fields()
            self.last_search_error = cached["error"]
            self.on_search_completed(self.active_search_id)
            self.update_status(f"Loaded cached results: {query}")
            return

        self.run_search_worker(query, query_key, self.active_search_id)

    def _is_cache_compatible(self, current_specs, cached_specs):
        if not cached_specs:
            return True
        if current_specs.get("has_gpu") != cached_specs.get("has_gpu"):
            return False

        ram_delta = abs(
            current_specs.get("ram_free", 0.0) - cached_specs.get("ram_free", 0.0)
        )
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
        if pid == "filter-all":
            self.current_filter = "All"
        elif pid == "filter-ollama":
            self.current_filter = "Ollama"
        elif pid == "filter-hf":
            self.current_filter = "Hugging Face"
        self.refresh_table()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id != "gem-toggle":
            return
        self.hidden_gems_only = bool(event.value)
        self.refresh_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
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
        self.push_screen(
            ModelDetailModal(model),
            lambda result, selected=model: self.on_model_detail_action(
                result, selected
            ),
        )

    def on_model_detail_action(self, result, model):
        if result != "download":
            if result == "cancel_download":
                self.cancel_model_download(model)
            return
        self.start_model_download(model)

    def cancel_model_download(self, model):
        target_id = download_target_id(model)
        process = self.download_processes.get(target_id)
        if process is None:
            self.update_status("No active download to cancel for selected model.")
            return

        self.cancelled_downloads.add(target_id)
        try:
            process.terminate()
        except OSError:
            pass
        self._set_download_state(target_id, "cancelled", "Canceled", "")
        self.update_status(f"Cancel requested: {model.get('name', target_id)}")

    def start_model_download(self, model):
        target_id = download_target_id(model)
        if target_id in self.active_downloads:
            self.update_status("Download already in progress for selected model.")
            return

        self.active_downloads.add(target_id)
        self._set_download_state(target_id, "queued", "Queued", "")
        self.update_status(f"Downloading: {model.get('name', target_id)}")
        self.download_model_worker(model.copy(), target_id)

    @work(thread=True)
    def download_model_worker(self, model, target_id):
        try:
            command = build_download_command(model)
        except ValueError as exc:
            self.call_from_thread(
                self.on_download_finished, target_id, model, False, str(exc)
            )
            return

        self.call_from_thread(
            self.on_download_progress,
            target_id,
            "downloading",
            "Downloading",
            "",
        )

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.download_processes[target_id] = process
            started_at = time.monotonic()
            last_report = started_at

            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    progress = self._extract_download_progress(line)
                    now = time.monotonic()

                    if progress is not None:
                        self.call_from_thread(
                            self.on_download_progress,
                            target_id,
                            "downloading",
                            "Downloading",
                            f"{progress}%",
                        )
                        last_report = now
                        continue

                    if now - last_report >= 1.0:
                        elapsed = int(now - started_at)
                        self.call_from_thread(
                            self.on_download_progress,
                            target_id,
                            "downloading",
                            "Downloading",
                            f"{elapsed}s",
                        )
                        last_report = now

            completed_code = process.wait()
        except FileNotFoundError:
            self.download_processes.pop(target_id, None)
            self.call_from_thread(
                self.on_download_finished,
                target_id,
                model,
                False,
                f"Required command is not installed: {command[0]}",
            )
            return
        except OSError as exc:
            self.download_processes.pop(target_id, None)
            self.call_from_thread(
                self.on_download_finished, target_id, model, False, str(exc)
            )
            return

        if completed_code == 0:
            self.download_processes.pop(target_id, None)
            self.call_from_thread(self.on_download_finished, target_id, model, True, "")
            return

        self.download_processes.pop(target_id, None)
        self.call_from_thread(
            self.on_download_finished,
            target_id,
            model,
            False,
            "download command exited with non-zero status",
        )

    def _extract_download_progress(self, line):
        if not line:
            return None
        match = re.search(r"(\d{1,3})%", line)
        if not match:
            return None
        value = int(match.group(1))
        if 0 <= value <= 100:
            return value
        return None

    def _find_model_by_target_id(self, target_id):
        for item in self.all_results:
            if download_target_id(item) == target_id:
                return item
        return None

    def _set_download_state(self, target_id, state, label, detail):
        model = self._find_model_by_target_id(target_id)
        if not model:
            return
        model["download_state"] = state
        model["download_label"] = label
        model["download_detail"] = detail
        self.refresh_table()

    def _download_cell_text(self, model):
        state = model.get("download_state", "idle")
        label = model.get("download_label", "Idle")
        detail = model.get("download_detail", "")

        if state == "completed":
            return "[green]Completed[/green]"
        if state == "failed":
            return f"[red]Failed[/red] {detail}" if detail else "[red]Failed[/red]"
        if state == "cancelled":
            return "[yellow]Canceled[/yellow]"
        if state == "downloading":
            return f"[yellow]{label}[/yellow] {detail}".strip()
        if state == "queued":
            return "[cyan]Queued[/cyan]"
        return "[grey50]Idle[/grey50]"

    def _ensure_download_fields(self):
        for item in self.all_results:
            item.setdefault("download_state", "idle")
            item.setdefault("download_label", "Idle")
            item.setdefault("download_detail", "")

    def on_download_progress(self, target_id, state, label, detail):
        self._set_download_state(target_id, state, label, detail)

    def on_download_finished(self, target_id, model, success, message):
        self.active_downloads.discard(target_id)
        self.download_processes.pop(target_id, None)
        if target_id in self.cancelled_downloads:
            self.cancelled_downloads.discard(target_id)
            self._set_download_state(target_id, "cancelled", "Canceled", "")
            self.update_status(f"Download canceled: {model.get('name', target_id)}")
            return
        if success:
            self._set_download_state(target_id, "completed", "Completed", "")
            if model.get("source") == "Ollama":
                self._mark_ollama_model_installed(model.get("name", ""))
            self.update_status(f"Download complete: {model.get('name', target_id)}")
            return
        self._set_download_state(target_id, "failed", "Failed", message[:40])
        self.update_status(f"Download failed: {message}")

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
    def run_search_worker(self, query: str, query_key: str, search_id: int) -> None:
        specs = self.monitor.get_specs()
        local_models = get_installed_ollama_models() if check_ollama_running() else []

        ollama_results, ollama_errors = search_ollama_models(query, specs, local_models)
        hf_results, hf_errors = search_hf_models(
            query,
            specs,
            self.hf_model_info_cache,
        )

        if search_id != self.active_search_id:
            return

        self.all_results = ollama_results + hf_results
        self._ensure_download_fields()
        self.last_search_error = " | ".join((ollama_errors + hf_errors)[:2])
        self._store_cached_search(query_key, self.all_results, self.last_search_error)
        self.call_from_thread(self.on_search_completed, search_id)

    def on_search_completed(self, search_id: int) -> None:
        if search_id != self.active_search_id:
            return

        table = self.query_one(DataTable)
        table.loading = False
        self.refresh_table()

        if table.row_count > 0:
            self.update_status(f"{table.row_count} results listed.")
        elif self.last_search_error:
            self.update_status(self.last_search_error[:120])
        else:
            self.update_status("No results found.")

        if table.row_count == 0 and self.last_search_error:
            table.add_row(
                "[red]![/red]",
                "System",
                "-",
                "Search error",
                "-",
                "-",
                f"[red]{self.last_search_error[:40]}[/red]",
                "-",
                "-",
                "-",
                "-",
                "-",
            )
        table.focus()

    def refresh_table(self):
        table = self.query_one(DataTable)
        table.clear()
        added = set()

        filtered_results = []
        for result in self.all_results:
            if self.current_filter != "All" and result["source"] != self.current_filter:
                continue
            if (
                self.use_case_filter != "all"
                and result.get("use_case_key") != self.use_case_filter
            ):
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
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."

            unique_key = f"{result['source']}:{result.get('id', result['name'])}"
            if unique_key in added:
                continue
            added.add(unique_key)

            table.add_row(
                result["inst"],
                result["source"],
                result.get("publisher", "-"),
                display_name,
                result["params"],
                result["use_case"],
                result["score"],
                result["quant"],
                result["mode"],
                result["fit"],
                result["size"],
                self._download_cell_text(result),
                key=unique_key,
            )
