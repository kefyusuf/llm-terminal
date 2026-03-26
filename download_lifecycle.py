"""Helpers for download lifecycle state and error handling."""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib.error import HTTPError

from download_status import is_active_state


def cancel_error_detail_from_http_error(exc: HTTPError) -> str:
    """Build a user-facing cancel error message from service HTTP payload."""
    detail = "Failed to cancel download through service."
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        error_text = payload.get("error")
        if error_text:
            detail = f"Cancel failed: {error_text}"
    except Exception:
        pass
    return detail


def delete_error_detail_from_http_error(exc: HTTPError) -> str:
    """Build a user-facing delete error message from service HTTP payload."""
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
    return detail


def should_cancel_before_delete(delete_data: bool, state: str) -> bool:
    """Return ``True`` when an active job should be canceled before delete."""
    return bool(delete_data and is_active_state(state))


def should_delete_ollama_data(delete_data: bool, source: str, model_name: str) -> bool:
    """Return ``True`` when local Ollama model files should be removed."""
    return bool(delete_data and source == "ollama" and model_name)


def entry_identity_keys(entry: dict[str, Any] | None) -> tuple[str, str]:
    """Return normalized source/name identity keys from a registry entry."""
    if entry is None:
        return "", ""
    source_key = str(entry.get("source", "")).strip().lower()
    name_key = str(entry.get("name", "")).strip().lower()
    return source_key, name_key


def reset_results_download_state(
    all_results: list[dict[str, Any]],
    *,
    target_id: str,
    source_key: str,
    name_key: str,
    target_id_for_item: Callable[[dict[str, Any]], str],
) -> int:
    """Reset matching result entries to idle download state.

    Returns number of rows updated.
    """
    updated = 0
    for item in all_results:
        item_target = target_id_for_item(item)
        if item_target == target_id:
            item["download_state"] = "idle"
            item["download_label"] = "Idle"
            item["download_detail"] = ""
            updated += 1
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
            updated += 1
    return updated


def upsert_download_registry_entry(
    registry: dict[str, dict[str, Any]],
    *,
    target_id: str,
    model: dict[str, Any] | None,
    state: str | None,
    label: str | None,
    detail: str | None,
    now: float,
) -> None:
    """Insert or update a download registry entry in-place."""
    existing = registry.get(target_id, {})
    source = existing.get("source", "-")
    publisher = existing.get("publisher", "-")
    name = existing.get("name", target_id)

    if model is not None:
        source = model.get("source", source)
        publisher = model.get("publisher", publisher)
        name = model.get("name", name)

    registry[target_id] = {
        "target_id": target_id,
        "source": source,
        "publisher": publisher,
        "name": name,
        "created_at": existing.get("created_at", now),
        "state": state if state is not None else existing.get("state", "idle"),
        "label": label if label is not None else existing.get("label", "Idle"),
        "detail": detail if detail is not None else existing.get("detail", ""),
        "updated_at": now,
    }


def trim_download_registry(registry: dict[str, dict[str, Any]], limit: int) -> None:
    """Trim oldest download registry entries to keep at most *limit* items."""
    if len(registry) <= limit:
        return
    sorted_keys = sorted(registry, key=lambda key: registry[key].get("updated_at", 0))
    for key in sorted_keys[: len(registry) - limit]:
        registry.pop(key, None)
