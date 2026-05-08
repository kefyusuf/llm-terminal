"""Shared helpers for download state normalization and display."""

from __future__ import annotations

STATE_LABELS = {
    "idle": "Idle",
    "queued": "Queued",
    "downloading": "Downloading",
    "running": "Running",
    "canceling": "Canceling",
    "completed": "Completed",
    "failed": "Failed",
    "cancelled": "Canceled",
}

STATE_MARKUP = {
    "idle": "[grey50]Idle[/grey50]",
    "queued": "[cyan]Queued[/cyan]",
    "downloading": "[yellow]Downloading[/yellow]",
    "running": "[yellow]Running[/yellow]",
    "canceling": "[orange]Canceling[/orange]",
    "completed": "[green]Completed[/green]",
    "failed": "[red]Failed[/red]",
    "cancelled": "[yellow]Canceled[/yellow]",
}

ACTIVE_STATES = frozenset({"queued", "downloading", "running", "canceling"})


def normalize_state(value: str | None) -> str:
    """Return a normalized state key used across UI and service layers."""
    state = str(value or "idle").strip().lower()
    if state == "canceled":
        return "cancelled"
    return state


def label_for_state(state: str | None) -> str:
    """Return user-facing label text for a state key."""
    return STATE_LABELS.get(normalize_state(state), "Idle")


def is_active_state(state: str | None) -> bool:
    """Return ``True`` when a state represents an active in-progress job."""
    return normalize_state(state) in ACTIVE_STATES


def map_service_job_status(
    raw_status: str | None,
    *,
    cancel_requested: bool = False,
    detail: str | None = None,
) -> str:
    """Map service job status values to app-facing download states."""
    status = normalize_state(raw_status)
    detail_text = str(detail or "").strip().lower()

    if status == "running":
        if cancel_requested or detail_text.startswith("cancel"):
            return "canceling"
        return "downloading"
    if status in {"queued", "completed", "failed", "cancelled", "canceling"}:
        return status
    return "idle"


def state_markup_from_state_and_label(
    state: str | None,
    label: str | None = None,
    *,
    unknown_is_external: bool = False,
) -> str:
    """Return Rich markup text for a state/label pair."""
    normalized_state = normalize_state(state)
    if normalized_state in STATE_MARKUP:
        return STATE_MARKUP[normalized_state]

    normalized_label = str(label or "").strip().lower()
    if "completed" in normalized_label or "done" in normalized_label:
        return STATE_MARKUP["completed"]
    if "failed" in normalized_label or "error" in normalized_label:
        return STATE_MARKUP["failed"]
    if "cancel" in normalized_label:
        return STATE_MARKUP["cancelled"]
    if "download" in normalized_label or "pull" in normalized_label:
        return STATE_MARKUP["downloading"]
    if "queue" in normalized_label:
        return STATE_MARKUP["queued"]

    if unknown_is_external:
        return "[red]![/red] External"
    return STATE_MARKUP["idle"]
