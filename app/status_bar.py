"""Status messages, modal polling pauses, and hardware snapshot rendering."""

from contextlib import suppress

from textual.widgets import (
    Static,
)

from app.widgets import SystemInfoWidget
from core.logging_ import get_logger

logger = get_logger(__name__)


class StatusBarMixin:
    """Status messages, modal polling pauses, and hardware snapshot rendering.

    Methods use the viewer's shared state and Textual APIs, preserving subclass hooks.
    """

    def _set_modal_poll_pause(self, enabled: bool) -> None:
        if enabled:
            self._modal_poll_pause_count += 1
            return

        self._modal_poll_pause_count = max(0, self._modal_poll_pause_count - 1)
        if self._modal_poll_pause_count == 0:
            self.request_system_info_refresh(force=True)
            self.request_download_poll(force=True)

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

    def _apply_system_info_refresh(self, specs, running_now):
        self._system_info_refresh_running = False
        if specs is not None:
            self.latest_specs = specs
        specs_to_render = self.latest_specs
        if specs_to_render is not None:
            with suppress(Exception):
                self.query_one(SystemInfoWidget).update_info(specs_to_render, running_now)
        else:
            self.query_one(SystemInfoWidget).update(
                "Hardware detection unavailable; polling will retry."
            )

        state_changed = running_now != self.ollama_running
        self.ollama_running = running_now
        if state_changed:
            if running_now:
                self.update_status("Ollama started. Local runtime features enabled.")
            else:
                self.update_status("Ollama stopped. Local runtime features disabled.")

    def poll_ollama_status(self, refresh_only=False):
        self.request_system_info_refresh(force=not refresh_only)
