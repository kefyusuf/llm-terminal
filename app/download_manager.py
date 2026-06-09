"""Download lifecycle manager — owns registry, job sync, and poll state."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError

from downloads.download_history import (
    action_label_for_entry,
)
from downloads.download_lifecycle import (
    cancel_error_detail_from_http_error,
    delete_error_detail_from_http_error,
    entry_identity_keys,
    should_cancel_before_delete,
    should_delete_ollama_data,
    trim_download_registry,
    upsert_download_registry_entry,
)
from downloads.download_manager import download_target_id
from downloads.download_status import (
    is_active_state,
    label_for_state,
    map_service_job_status,
    state_markup_from_state_and_label,
)
from downloads.service_client import (
    cancel_job,
    create_job,
    delete_job,
    ensure_service_running,
    get_active_download_debug,
    get_service_health,
    list_jobs,
)


class DownloadManager:
    """Manages download registry, job lifecycle, and polling state.

    Owns all download-related state and provides methods for the UI layer
    to delegate operations to.
    """

    def __init__(
        self,
        *,
        update_status: Callable[[str], None],
        refresh_table: Callable[[], None],
        refresh_download_history_table: Callable[[], None],
        request_download_history_refresh: Callable[[bool], None],
        render_download_debug: Callable[[Any, Any], None],
        find_model_by_target_id: Callable[[str], dict | None],
        history_limit: int = 50,
        history_refresh_interval: float = 0.9,
        poll_request_timeout: float = 0.35,
    ):
        self._update_status = update_status
        self._refresh_table = refresh_table
        self._refresh_download_history_table = refresh_download_history_table
        self._request_download_history_refresh = request_download_history_refresh
        self._render_download_debug = render_download_debug
        self._find_model_by_target_id = find_model_by_target_id

        self.download_registry: dict[str, dict] = {}
        self.active_downloads: set[str] = set()
        self._download_poll_running = False
        self.last_download_history_refresh_at = 0.0
        self.download_history_refresh_pending = False
        self.download_history_limit = history_limit
        self.download_history_refresh_interval = history_refresh_interval
        self.download_poll_request_timeout = poll_request_timeout

    def _record_entry(self, target_id, model=None, state=None, label=None, detail=None):
        if model is None:
            model = self._find_model_by_target_id(target_id)
        now = time.time()
        upsert_download_registry_entry(
            self.download_registry,
            target_id=target_id,
            model=model,
            state=state,
            label=label,
            detail=detail,
            now=now,
        )
        trim_download_registry(self.download_registry, self.download_history_limit)

    def set_download_state(
        self,
        target_id,
        state,
        label,
        detail,
        model=None,
        refresh_results=True,
        refresh_history=True,
    ):
        self._record_entry(target_id, model=model, state=state, label=label, detail=detail)
        model = self._find_model_by_target_id(target_id)
        if model and refresh_results:
            model["download_state"] = state
            model["download_label"] = label
            model["download_detail"] = detail
            self._refresh_table()
        elif model:
            model["download_state"] = state
            model["download_label"] = label
            model["download_detail"] = detail
        if refresh_history:
            self._request_download_history_refresh(False)

    def download_cell_text(self, model):
        state = model.get("download_state", "idle")
        return state_markup_from_state_and_label(state, model.get("download_label"))

    def status_text_from_state(self, state, label=None):
        return state_markup_from_state_and_label(state, label, unknown_is_external=True)

    def sync_jobs(self, force=False, jobs=None):
        if jobs is None:
            try:
                jobs = list_jobs(
                    limit=self.download_history_limit,
                    timeout=self.download_poll_request_timeout,
                )
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
            self._record_entry(
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

    def ensure_download_fields(self, all_results: list[dict]) -> bool:
        changed = False
        for item in all_results:
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

    def refresh_history_table(self, registry: dict[str, dict]):
        entries = sorted(
            registry.values(),
            key=lambda item: (
                item.get("created_at", 0),
                item.get("target_id", ""),
            ),
            reverse=True,
        )
        rows = []
        for entry in entries[: self.download_history_limit]:
            target_id = entry.get("target_id", "-")
            action_btn = action_label_for_entry(entry)
            rows.append(
                {
                    "source": entry.get("source", "-"),
                    "publisher": entry.get("publisher", "-"),
                    "name": entry.get("name", "-"),
                    "state": entry.get("state", "idle"),
                    "label": entry.get("label", "Idle"),
                    "detail": entry.get("detail", ""),
                    "action": action_btn,
                    "key": target_id,
                }
            )
        return rows

    def request_history_refresh(self, force=False):
        now = time.monotonic()
        if force or (
            now - self.last_download_history_refresh_at >= self.download_history_refresh_interval
        ):
            self.last_download_history_refresh_at = now
            self.download_history_refresh_pending = False
            return True
        self.download_history_refresh_pending = True
        return False

    def can_poll(self, modal_poll_pause_count: int, force: bool) -> bool:
        if modal_poll_pause_count > 0 and not force:
            return False
        return force or not self._download_poll_running

    def set_poll_running(self, running: bool):
        self._download_poll_running = running

    def poll_jobs(self):
        jobs = None
        debug = None
        health = None
        try:
            jobs = list_jobs(
                limit=self.download_history_limit,
                timeout=self.download_poll_request_timeout,
            )
        except Exception:
            jobs = None
        try:
            debug = get_active_download_debug(timeout=self.download_poll_request_timeout)
        except Exception:
            try:
                health = get_service_health(timeout=self.download_poll_request_timeout)
            except Exception:
                health = None
        return jobs, debug, health

    def cancel_download(self, model):
        target_id = download_target_id(model)
        try:
            response = cancel_job(target_id)
            _ = response.get("job")
            return True, f"Cancel requested: {model.get('name', target_id)}"
        except HTTPError as exc:
            return False, cancel_error_detail_from_http_error(exc)
        except Exception:
            return False, "Failed to cancel download through service."

    def start_download(self, model):
        if not ensure_service_running():
            return False, "Download service is unavailable."
        target_id = download_target_id(model)
        try:
            response = create_job(model)
            queued = bool(response.get("queued"))
            if queued:
                return True, f"Download queued: {model.get('name', target_id)}"
            return True, f"Download already active: {model.get('name', target_id)}"
        except Exception as exc:
            return False, f"Failed to queue download: {exc}"

    def delete_entry(self, target_id, delete_data=False):
        entry = self.download_registry.get(target_id)
        source = entry.get("source", "").lower() if entry else ""
        model_name = entry.get("name", "") if entry else ""
        state = entry.get("state", "idle") if entry else "idle"
        messages = []

        if should_cancel_before_delete(delete_data, state):
            try:
                cancel_job(target_id)
                time.sleep(1)
                messages.append(f"Download canceled: {model_name}")
            except Exception as e:
                messages.append(f"Could not cancel download: {e}")

        if should_delete_ollama_data(delete_data, source, model_name):
            import subprocess
            try:
                result = subprocess.run(
                    ["ollama", "rm", model_name],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    messages.append(f"Deleted model data: {model_name}")
                else:
                    result2 = subprocess.run(
                        ["ollama", "rm", f"{model_name}:latest"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if result2.returncode == 0:
                        messages.append(f"Deleted model data: {model_name}:latest")
                    else:
                        messages.append(f"Could not delete model data: {result2.stderr.strip()}")
            except subprocess.TimeoutExpired:
                messages.append(f"Timeout deleting model data: {model_name}")
            except Exception as e:
                messages.append(f"Delete model data error: {e!s}")

        try:
            delete_job(target_id)
        except HTTPError as exc:
            return False, delete_error_detail_from_http_error(exc), None, None
        except Exception as exc:
            return False, f"Failed to delete download entry: {exc}", None, None

        deleted_entry = self.download_registry.pop(target_id, None)
        source_key, name_key = entry_identity_keys(deleted_entry)
        return True, "Download entry deleted.", (source_key, name_key), target_id
