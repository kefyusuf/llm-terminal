"""Compact live download presentation using only provider-reported metrics."""

from __future__ import annotations

import re

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, ProgressBar, Static

from core.utils import extract_download_progress
from downloads.download_history import cancel_model_payload
from downloads.download_status import is_active_state

_SIZE = r"\d+(?:\.\d+)?\s*[KMGT]?i?B"


class DownloadProgressPanel(Vertical):
    """Keep progress and cancellation visible independently of table columns."""

    DEFAULT_CSS = """
    DownloadProgressPanel { height: 4; display: none; }
    DownloadProgressPanel .progress-row { height: 3; }
    DownloadProgressPanel Static { width: 1fr; height: 3; overflow: hidden; }
    DownloadProgressPanel Button { width: 12; min-width: 10; height: 3; margin: 0; }
    DownloadProgressPanel ProgressBar { height: 1; width: 100%; }
    DownloadProgressPanel Bar { width: 1fr; }
    #active-download-progress { dock: bottom; }
    """

    def __init__(self, *, id: str, show_cancel: bool = True):
        super().__init__(id=id)
        self.entry: dict = {}
        self.show_cancel = show_cancel

    def compose(self) -> ComposeResult:
        with Horizontal(classes="progress-row"):
            yield Static("", markup=False)
            button = Button("Cancel", id="progress-cancel-btn", variant="warning")
            button.display = self.show_cancel
            yield button
        yield ProgressBar(total=None, show_eta=False, show_percentage=False)

    def show_entry(self, entry: dict | None, *, other_active: int = 0) -> None:
        if not list(self.query(Static)):
            return
        self.entry = entry or {}
        self.display = bool(entry)
        if not entry:
            return
        state = entry.get("state", "idle")
        detail = str(entry.get("detail") or "")
        active = is_active_state(state)
        percent = extract_download_progress(detail) if active else None
        label = entry.get("label") or state.title()
        if state == "downloading":
            label = f"Layer {percent}%" if percent is not None else "Preparing"
            if detail.startswith("verifying"):
                label = "Verifying"
            elif detail.startswith(("writing manifest", "removing unused", "success")):
                label = "Finalizing"
        if state == "canceling":
            label = "Canceling..."
        suffix = f" (+{other_active} active)" if other_active else ""
        name = str(entry.get("name") or "Download")
        # Reserve a full line for progress so a long model name cannot hide it.
        columns = max(12, self.size.width - (12 if self.show_cancel else 0))
        title_width = max(4, columns - len(str(label)) - len(suffix) - 3)
        title = name if len(name) <= title_width else name[:max(1, title_width - 3)] + "..."
        metrics = re.search(rf"({_SIZE})\s*/\s*({_SIZE})", detail, re.I) if active else None
        speed = re.search(rf"({_SIZE})/s", detail, re.I) if active else None
        eta = re.search(r"\s((?:\d+h)?(?:\d+m)?\d+s)\s*$", detail) if active else None
        transfer = f"{metrics[1]} / {metrics[2]}" if metrics else "Size: unavailable"
        transfer += f" | {speed[1]}/s" if speed else " | Speed: calculating"
        remaining = f"Layer ETA (last report): {eta[1]}" if eta else "Remaining: calculating"
        if label in {"Verifying", "Finalizing"}:
            transfer, remaining = detail, "Transfer finished; completion pending"
        if not active:
            transfer = detail or label
            remaining = ""
        # The provider's ETA is per layer, not a model-wide completion promise.
        text = f"{title}{suffix} | {label}\n{transfer}\n{remaining}"
        self.query_one(Static).update(text)
        bar = self.query_one(ProgressBar)
        bar.display = active
        bar.update(total=100 if percent is not None or not active else None,
                   progress=percent or 0)
        button = self.query_one(Button)
        button.disabled = not active or state == "canceling"
        button.label = "Canceling" if state == "canceling" else "Cancel"

    @on(Button.Pressed, "#progress-cancel-btn")
    def cancel(self, event: Button.Pressed) -> None:
        event.stop()
        if not self.entry or not is_active_state(self.entry.get("state")):
            return
        cancel = getattr(self.app, "cancel_model_download", None)
        if callable(cancel):
            cancel(cancel_model_payload(str(self.entry.get("target_id", "")), self.entry))
