"""Modal screens for the AI Model Explorer TUI."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from core.logging_ import get_logger

logger = get_logger(__name__)


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

    def on_mount(self) -> None:
        set_pause = getattr(self.app, "_set_modal_poll_pause", None)
        if callable(set_pause):
            set_pause(True)

    def on_unmount(self) -> None:
        set_pause = getattr(self.app, "_set_modal_poll_pause", None)
        if callable(set_pause):
            set_pause(False)

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
            yield Label(f"🤖 {self.data['name']}", id="modal-title")

            yield Label("")
            yield Label("[bold #63b3ed]📊 Model Information[/bold #63b3ed]")
            yield Label(
                f"[#718096]Source:[/#718096] [#e2e8f0]{self.data['source']} / {self.data.get('publisher', '-')}[/#e2e8f0]"
            )
            yield Label(
                f"[#718096]Provider:[/#718096] [#e2e8f0]{self.data.get('provider', '-')}[/#e2e8f0]"
            )
            yield Label(
                f"[#718096]Use Case:[/#718096] [#e2e8f0]{self.data.get('use_case', '-')}[/#e2e8f0]"
            )
            yield Label(
                f"[#718096]Scale:[/#718096] [#e2e8f0]{self.data.get('params', '-')}[/#e2e8f0]"
            )

            yield Label("")
            yield Label("[bold #63b3ed]⚙️ Technical Specifications[/bold #63b3ed]")
            yield Label(
                f"[#718096]Format:[/#718096] [#e2e8f0]{self.data.get('quant', '-')}[/#e2e8f0]"
            )
            yield Label(
                f"[#718096]Score:[/#718096] [#4299e1]{self.data.get('score', '-')}[/#4299e1]"
            )
            yield Label(
                f"[#718096]Size:[/#718096] [#e2e8f0]{self.data.get('size', '-')} {confidence_label}[/#e2e8f0]"
            )
            yield Label(
                f"[#718096]Fit:[/#718096] [{fit_class}]{self.data.get('fit', '-')}[/{fit_class}]"
            )
            yield Label(f"[#718096]Mode:[/#718096] [#48bb78]{self.data.get('mode', '-')}[/#48bb78]")

            yield Label("")
            yield Label("[bold #63b3ed]🚀 Run Command[/bold #63b3ed]")
            yield Label(cmd_text, id="cmd-box")

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
            logger.warning("Copy to clipboard failed for command: {}", cmd_text)
            if hasattr(self.app, "update_status"):
                self.app.update_status(f"Copy failed. Command: {cmd_text}")


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

    def on_mount(self) -> None:
        set_pause = getattr(self.app, "_set_modal_poll_pause", None)
        if callable(set_pause):
            set_pause(True)

    def on_unmount(self) -> None:
        set_pause = getattr(self.app, "_set_modal_poll_pause", None)
        if callable(set_pause):
            set_pause(False)

    def compose(self) -> ComposeResult:
        state = (self.entry.get("state") or self.entry.get("status") or "idle").lower()
        progress = self.entry.get("detail") or self.entry.get("progress") or "-"

        status_class = {
            "downloading": "job-status-downloading",
            "queued": "job-status-queued",
            "completed": "job-status-completed",
            "failed": "job-status-failed",
            "cancelled": "job-status-cancelled",
            "canceling": "job-status-cancelled",
        }.get(state, "")
        status_markup = f"[{status_class}]{state}[/{status_class}]" if status_class else state

        is_active = state in {"queued", "downloading", "running", "canceling"}

        with Vertical(id="job-modal"):
            yield Label(f"⬇ {self.entry.get('name', '-')}", id="job-title")

            yield Label(
                f"[#718096]Source:[/#718096] [#e2e8f0]{self.entry.get('source', '-')}[/#e2e8f0]"
            )
            yield Label(
                f"[#718096]Publisher:[/#718096] [#e2e8f0]{self.entry.get('publisher', '-')}[/#e2e8f0]"
            )
            yield Label(f"[#718096]Status:[/#718096] {status_markup}")
            yield Label(f"[#718096]Progress:[/#718096] [#e2e8f0]{progress}[/#e2e8f0]")

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

        cancel_fn = getattr(self.app, "cancel_model_download", None)
        if callable(cancel_fn):
            cancel_fn(model_data)

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


class PlanModeModal(ModalScreen):
    """Modal showing hardware requirements analysis for a model."""

    CSS = """
    PlanModeModal {
        align: center middle;
        background: $background;
    }
    #plan-container {
        width: 70;
        height: auto;
        max-height: 35;
        background: #0f141f;
        border: round #4a5568;
        padding: 1 2;
    }
    #plan-header {
        background: #2d4a6f;
        padding: 0 2;
        margin: -1 -2 1 -2;
        height: 3;
    }
    #plan-title {
        text-style: bold;
        color: #e2e8f0;
        text-align: center;
    }
    #plan-scrollable {
        height: auto;
        max-height: 25;
        overflow-y: auto;
    }
    #plan-close-btn {
        background: #718096;
        color: white;
        margin-top: 1;
    }
    """

    def __init__(self, model_name: str, plans: list[dict]):
        super().__init__()
        self.model_name = model_name
        self.plans = plans

    def on_mount(self) -> None:
        set_pause = getattr(self.app, "_set_modal_poll_pause", None)
        if callable(set_pause):
            set_pause(True)

    def on_unmount(self) -> None:
        set_pause = getattr(self.app, "_set_modal_poll_pause", None)
        if callable(set_pause):
            set_pause(False)

    def compose(self) -> ComposeResult:
        with Vertical(id="plan-container"):
            yield Label(
                f"[bold #63b3ed]Hardware Plan: {self.model_name}[/bold #63b3ed]", id="plan-title"
            )

            with Vertical(id="plan-scrollable"):
                yield Label("")
                yield Label(
                    "[bold #a0aec0]Quant    Size      VRAM    Mode       Min GPU[/bold #a0aec0]"
                )
                yield Label("[#4a5568]" + "-" * 62 + "[/#4a5568]")

                for p in self.plans:
                    quant = p["quant"]
                    size = f"{p['size_gb']:.1f}GB"
                    vram = f"{p['vram_needed']:.1f}GB"
                    mode = p["mode"]
                    gpu_class = p["min_gpu_class"]

                    rank = p.get("quality_rank", 5)
                    if rank >= 8:
                        color = "#4fe08a"
                    elif rank >= 5:
                        color = "#7edfff"
                    elif rank >= 3:
                        color = "#f2c46d"
                    else:
                        color = "#ff7f8f"

                    line = f"[{color}]{quant:<8}[/{color}][#e2e8f0]{size:<10}{vram:<8}{mode:<11}{gpu_class}[/#e2e8f0]"
                    yield Label(line)

            yield Button("Close", id="plan-close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()


class ComparisonModal(ModalScreen):
    """Modal showing side-by-side comparison of up to 4 selected models."""

    CSS = """
    ComparisonModal {
        align: center middle;
        background: $background;
    }
    #comparison-container {
        width: 85;
        height: auto;
        max-height: 40;
        background: #0f141f;
        border: round #4a5568;
        padding: 1 2;
    }
    #comparison-header {
        background: #2d4a6f;
        padding: 0 2;
        margin: -1 -2 1 -2;
        height: 3;
    }
    #comparison-title {
        text-style: bold;
        color: #e2e8f0;
        text-align: center;
    }
    #comparison-scrollable {
        height: auto;
        max-height: 30;
        overflow-y: auto;
    }
    #comparison-close-btn {
        background: #718096;
        color: white;
        margin-top: 1;
    }
    """

    def __init__(self, models: list[dict]):
        super().__init__()
        self.models = models

    def on_mount(self) -> None:
        set_pause = getattr(self.app, "_set_modal_poll_pause", None)
        if callable(set_pause):
            set_pause(True)

    def on_unmount(self) -> None:
        set_pause = getattr(self.app, "_set_modal_poll_pause", None)
        if callable(set_pause):
            set_pause(False)

    def compose(self) -> ComposeResult:
        with Vertical(id="comparison-container"):
            yield Label(
                f"[bold #63b3ed]Model Comparison ({len(self.models)} models)[/bold #63b3ed]",
                id="comparison-title",
            )

            with Vertical(id="comparison-scrollable"):
                names = " | ".join(m.get("name", "-")[:18] for m in self.models)
                yield Label("")
                yield Label(f"[bold #e2e8f0]{names}[/bold #e2e8f0]")
                yield Label("[#4a5568]" + "-" * 78 + "[/#4a5568]")

                rows = [
                    ("Source", lambda m: m.get("source", "-")),
                    ("Params", lambda m: m.get("params", "-")),
                    ("Quant", lambda m: m.get("quant", "-")),
                    ("Size", lambda m: m.get("size", "-")),
                    ("Fit", lambda m: self._strip_markup(m.get("fit", "-"))),
                    ("Mode", lambda m: self._strip_markup(m.get("mode", "-"))),
                    ("Quality", lambda m: str(m.get("score_quality", "-"))),
                    ("Speed", lambda m: str(m.get("score_speed", "-"))),
                    ("Fit Score", lambda m: str(m.get("score_fit", "-"))),
                    ("Context", lambda m: str(m.get("score_context", "-"))),
                    ("Composite", lambda m: str(m.get("score_composite", "-"))),
                    (
                        "Est. tok/s",
                        lambda m: (
                            f"{m.get('estimated_tok_s', 0):.0f}"
                            if m.get("estimated_tok_s")
                            else "-"
                        ),
                    ),
                    (
                        "MoE",
                        lambda m: (
                            f"MoE {m.get('active_experts', '?')}/{m.get('total_experts', '?')}"
                            if m.get("is_moe")
                            else "Dense"
                        ),
                    ),
                ]

                for label, getter in rows:
                    vals = " | ".join(getter(m)[:18].ljust(18) for m in self.models)
                    yield Label(f"[#718096]{label:<12}[/#718096][#e2e8f0]{vals}[/#e2e8f0]")

            yield Button("Close", id="comparison-close-btn")

    @staticmethod
    def _strip_markup(text: str) -> str:
        import re

        return re.sub(r"\[[^\]]+\]", "", text).strip() or "-"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()
