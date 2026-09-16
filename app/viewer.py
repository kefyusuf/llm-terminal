"""Runtime TUI viewer extensions for provider selection."""

from __future__ import annotations

from collections.abc import Sequence

from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Input, Select

from app.search_constants import USE_CASE_OPTIONS
from providers import get_provider_filter_labels
from tui_app import AIModelViewer as BaseAIModelViewer


_PROVIDER_COMPACT_TAGS = {
    "Ollama": "OL",
    "Hugging Face": "HF",
    "LM Studio": "LM",
    "Docker": "DK",
    "MLX": "MLX",
}


def cycle_provider_label(labels: Sequence[str], current: str) -> str:
    """Return the next provider label, recovering unknown state to the first option."""
    if not labels:
        return current
    if current not in labels:
        return labels[0]
    return labels[(labels.index(current) + 1) % len(labels)]


def provider_compact_tag(label: str) -> str:
    """Return a short stable tag for a provider label in compact mode."""
    return _PROVIDER_COMPACT_TAGS.get(label, label[:3].upper() or "-")


class AIModelViewer(BaseAIModelViewer):
    """Run the main viewer with synchronized provider and use-case selectors."""

    def __init__(self):
        """Snapshot available provider labels for both mouse and keyboard selection."""
        super().__init__()
        labels = tuple(get_provider_filter_labels())
        self.provider_filter_labels = labels or ("Ollama",)
        if self.current_filter not in self.provider_filter_labels:
            self.current_filter = self.provider_filter_labels[0]

    async def on_mount(self) -> None:
        """Mount compact selectors after the base UI initializes."""
        super().on_mount()

        provider_panel = self.query_one("#provider-panel", Vertical)
        await provider_panel.remove_children()
        provider_selector = Select(
            ((label, label) for label in self.provider_filter_labels),
            value=self.current_filter,
            allow_blank=False,
            id="provider-select",
        )
        provider_selector.styles.width = "100%"
        provider_selector.styles.height = 3
        await provider_panel.mount(provider_selector)

        use_case_panel = self.query_one("#use-case-panel", Vertical)
        use_case_selector = Select(
            ((label, key) for key, label in USE_CASE_OPTIONS),
            value=self.use_case_filter,
            allow_blank=False,
            id="use-case-select",
        )
        use_case_selector.styles.width = "100%"
        use_case_selector.styles.height = 3
        await use_case_panel.mount(use_case_selector)
        self._apply_runtime_filter_controls()

    def _apply_runtime_filter_controls(self) -> None:
        """Prefer compact runtime selectors while preserving the hidden base RadioSet state."""
        try:
            self.query_one("#use-case-filter").styles.display = "none"
        except NoMatches:
            pass

        try:
            selector = self.query_one("#use-case-select", Select)
            selector.styles.display = "block"
            selector.styles.width = "100%"
            selector.styles.height = 3
        except NoMatches:
            pass

    def _apply_ui_mode(self) -> None:
        """Apply base mode styling, then restore runtime selector presentation."""
        super()._apply_ui_mode()
        self._apply_runtime_filter_controls()

    def _apply_provider_filter(self, label: str, *, sync_widget: bool) -> None:
        """Apply one provider label and keep the mounted selector synchronized."""
        if label not in self.provider_filter_labels or label == self.current_filter:
            return

        self.current_filter = label
        if sync_widget:
            try:
                selector = self.query_one("#provider-select", Select)
                if selector.value != label:
                    selector.value = label
            except NoMatches:
                pass

        current_query = self.query_one("#search-input", Input).value.strip()
        if current_query:
            self.start_search(current_query)
            self.update_status(f"Provider switched to {label}. Searching...")
        else:
            self.refresh_table()
            self.update_status(f"Provider filter set to {label}.")

    def _set_use_case_filter(self, key: str) -> None:
        """Apply a use-case key and synchronize hidden base state plus runtime selector."""
        super()._set_use_case_filter(key)
        try:
            selector = self.query_one("#use-case-select", Select)
            if selector.value != key:
                selector.value = key
        except NoMatches:
            pass

    def action_cycle_provider(self) -> None:
        """Cycle through the exact provider labels displayed by the selector."""
        next_filter = cycle_provider_label(self.provider_filter_labels, self.current_filter)
        self._apply_provider_filter(next_filter, sync_widget=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Apply provider or use-case changes made through runtime selectors."""
        if event.select.id == "provider-select":
            self._apply_provider_filter(str(event.value), sync_widget=False)
            return

        if event.select.id != "use-case-select":
            return

        key = str(event.value)
        if key == self.use_case_filter:
            return

        self._set_use_case_filter(key)
        self.refresh_table()
        self.update_status(f"Use Case filter set to {self._use_case_label(key)}.")

    def _apply_download_poll_snapshot(self, jobs, debug, health):
        """Apply the base snapshot and surface deduplicated poll status transitions."""
        super()._apply_download_poll_snapshot(jobs, debug, health)
        status_message = self.dl.take_poll_status_message()
        if status_message:
            self.update_status(status_message)

    def _compact_chip_text(self, shown_count: int, total: int) -> str:
        """Render compact search state using a provider-specific short tag."""
        provider_short = provider_compact_tag(self.current_filter)
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
