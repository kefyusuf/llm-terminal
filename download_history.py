"""Helpers for download history row behavior and actions."""

from __future__ import annotations

from typing import Any, Mapping

from download_status import is_active_state, normalize_state


_MANAGED_DETAIL_TOKENS = (
    "download",
    "pull",
    "queue",
    "completed",
    "failed",
    "cancel",
)


def fallback_entry_from_target(target_id: str) -> dict[str, Any]:
    """Create a minimal history entry when a registry record is missing."""
    source = target_id.split(":", maxsplit=1)[0] if ":" in target_id else "unknown"
    name = target_id.split(":", maxsplit=1)[1] if ":" in target_id else target_id
    return {
        "target_id": target_id,
        "source": source,
        "name": name,
        "state": "idle",
        "detail": "External download",
    }


def is_external_entry(entry: Mapping[str, Any]) -> bool:
    """Return ``True`` when entry does not appear managed by this app."""
    state = normalize_state(str(entry.get("state", "idle")))
    if state != "idle":
        return False
    detail_text = str(entry.get("detail", "")).lower()
    return not any(token in detail_text for token in _MANAGED_DETAIL_TOKENS)


def action_label_for_entry(entry: Mapping[str, Any]) -> str:
    """Return history action label for the entry state."""
    if is_external_entry(entry):
        return "External"
    if is_active_state(str(entry.get("state", "idle"))):
        return "Cancel"
    return "Delete"


def cancel_model_payload(target_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Build model payload used by ``cancel_model_download``."""
    model_id = target_id.split(":", maxsplit=1)[1] if ":" in target_id else target_id
    return {
        "source": entry.get("source", "-"),
        "name": entry.get("name", "-"),
        "id": model_id,
    }
