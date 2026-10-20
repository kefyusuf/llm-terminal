"""Status messages, modal polling pauses, and hardware snapshot refresh."""

from __future__ import annotations

import threading
from contextlib import suppress

from textual.widgets import Static

from app.widgets import SystemInfoWidget
from core import cache_db
from core.hardware import HardwareMonitor, check_ollama_running
from core.logging_ import get_logger

logger = get_logger(__name__)

_UNKNOWN_SPECS = {
    "cpu_name": "Detecting hardware",
    "cpu_cores": 0,
    "ram_free": 0.0,
    "ram_total": 0.0,
    "vram_free": 0.0,
    "vram_total": 0.0,
    "gpu_name": "Detecting hardware",
    "has_gpu": False,
}


class StatusController:
    """Own the status bar, hardware snapshot, Ollama liveness, and poll pausing."""

    def __init__(self, app):
        self.app = app
        self.monitor = None
        self._hardware_init_lock = threading.Lock()
        self.latest_specs = None
        self.ollama_running = False
        self.modal_poll_pause_count = 0
        self._refresh_running = False

    def update(self, text) -> None:
        """Show *text* in the status bar once it is mounted."""
        try:
            self.app.query_one("#status-bar", Static).update(text)
        except Exception:
            logger.debug("Status bar not yet mounted, skipping update")

    def text(self):
        try:
            return self.app.query_one("#status-bar", Static).renderable
        except Exception:
            return None

    def set_modal_poll_pause(self, enabled: bool) -> None:
        if enabled:
            self.modal_poll_pause_count += 1
            return
        self.modal_poll_pause_count = max(0, self.modal_poll_pause_count - 1)
        if self.modal_poll_pause_count == 0:
            self.request_system_info_refresh(force=True)
            self.app.downloads.request_poll(force=True)

    def hardware_monitor(self):
        """Initialize once from workers; failed construction remains retryable."""
        with self._hardware_init_lock:
            if self.monitor is None:
                self.monitor = HardwareMonitor()
            return self.monitor

    def specs_for_search(self):
        if self.latest_specs is not None:
            return self.latest_specs
        cached = cache_db.get_hardware_snapshot()
        if cached is not None:
            self.latest_specs = cached
            return cached
        return dict(_UNKNOWN_SPECS)

    def show_cached_specs(self) -> None:
        cached_specs = cache_db.get_hardware_snapshot()
        widget = self.app.query_one(SystemInfoWidget)
        if cached_specs is not None:
            self.latest_specs = cached_specs
            widget.update_info(cached_specs, None)
        else:
            widget.update("Detecting hardware...")

    def request_system_info_refresh(self, force=False) -> None:
        if self._refresh_running or (self.modal_poll_pause_count > 0 and not force):
            return
        self._refresh_running = True
        self.app.run_worker(self._refresh_worker, thread=True, group="system-info")

    def _refresh_worker(self) -> None:
        specs = None
        running_now = self.ollama_running
        try:
            specs = self.hardware_monitor().get_specs()
            running_now = check_ollama_running()
            cache_db.set_hardware_snapshot(specs)
        except Exception as exc:
            logger.warning("System info refresh failed: {}", exc)
        self.app.call_from_thread(self.apply_system_info_refresh, specs, running_now)

    def apply_system_info_refresh(self, specs, running_now) -> None:
        self._refresh_running = False
        if specs is not None:
            self.latest_specs = specs
        if self.latest_specs is not None:
            with suppress(Exception):
                self.app.query_one(SystemInfoWidget).update_info(self.latest_specs, running_now)
        else:
            with suppress(Exception):
                self.app.query_one(SystemInfoWidget).update(
                    "Hardware detection unavailable; polling will retry."
                )

        state_changed = running_now != self.ollama_running
        self.ollama_running = running_now
        if state_changed:
            if running_now:
                self.update("Ollama started. Local runtime features enabled.")
            else:
                self.update("Ollama stopped. Local runtime features disabled.")
