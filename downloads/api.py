"""HTTP API for the download service.

Extracted from downloads/download_service.py during the architecture
review (candidate #5). Exposes a :class:`BaseHTTPRequestHandler`
subclass that dispatches the service's 5 GET + 5 POST endpoints
against the shared :class:`DownloadServiceState` singleton.

The handler is the highest-bug-density surface in the service
(was 0/5 endpoints directly tested before the split). The split
makes it possible to construct a :class:`Handler` against a fake
``state`` in tests and exercise individual routes without spinning
up a real subprocess service.
"""

from __future__ import annotations

import hmac
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


def _can_terminate_process(process):
    return hasattr(process, "terminate") and callable(getattr(process, "terminate", None))


def _has_duplicates(values):
    return len(values) != len(set(values))


def _make_handler(state, auth_token: str | None = None):
    """Build a Handler class bound to *state* and an optional bearer token.

    BaseHTTPRequestHandler is instantiated per request, so we close
    over *state* in a small factory rather than referencing the
    module-level singleton directly. This makes the handler
    testable: tests can pass a fake state with a fake store.
    """
    # Lazy import: SERVICE_VERSION lives in download_service.py; this
    # closure binds it at handler-construction time so the module can
    # be imported without triggering a circular import.
    from .download_service import SERVICE_VERSION

    _can_term = _can_terminate_process

    class Handler(BaseHTTPRequestHandler):
        """Serve download-service requests against the factory-bound state."""

        def _json_response(self, status_code, payload, extra_headers=None):
            """Write one JSON response with optional extra headers."""
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            """Read and decode the request JSON body, returning an empty dict if absent."""
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            if not raw:
                return {}
            return json.loads(raw)

        def _is_authorized(self) -> bool:
            """Return whether this request satisfies the configured bearer token."""
            if not auth_token:
                return True
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {auth_token}"
            return hmac.compare_digest(supplied, expected)

        def _require_auth(self) -> bool:
            """Reject unauthorized requests and return whether dispatch may continue."""
            if self._is_authorized():
                return True
            self._json_response(
                401,
                {"error": "unauthorized"},
                {"WWW-Authenticate": "Bearer"},
            )
            return False

        def do_GET(self):
            """Dispatch GET requests, leaving only the health endpoint public."""
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json_response(200, {"ok": True, "version": SERVICE_VERSION})
                return

            if not self._require_auth():
                return

            if parsed.path == "/debug/active":
                active_targets = state.snapshot_active_targets()
                self._json_response(
                    200,
                    {
                        "active_targets": active_targets,
                        "count": len(active_targets),
                        "has_duplicates": _has_duplicates(active_targets),
                        "worker_alive": bool(
                            state.worker_thread is not None and state.worker_thread.is_alive()
                        ),
                    },
                )
                return

            if parsed.path == "/jobs":
                qs = parse_qs(parsed.query)
                limit = int(qs.get("limit", ["50"])[0])
                jobs = state.store.list_jobs(limit=limit)
                self._json_response(200, {"jobs": jobs})
                return

            self._json_response(404, {"error": "not found"})

        def do_POST(self):
            """Dispatch authenticated mutation requests."""
            if not self._require_auth():
                return

            if self.path in {"/jobs", "/jobs/plan"}:
                try:
                    payload = self._read_json()
                    if not isinstance(payload, dict):
                        raise ValueError("expected a JSON object")
                    model = payload.get("model") or {}
                    if not isinstance(model, dict):
                        raise ValueError("expected a model object")
                    import config

                    if self.path == "/jobs/plan":
                        plan = state.store.plan_job(model, config.settings.hf_models_dir)
                        self._json_response(200, {"plan": plan})
                        return
                    job, created_or_queued = state.store.upsert_job(model, models_dir=config.settings.hf_models_dir)
                    self._json_response(
                        200,
                        {
                            "job": job,
                            "queued": bool(created_or_queued),
                        },
                    )
                except ValueError as exc:
                    self._json_response(400, {"error": str(exc)})
                return

            if self.path == "/jobs/cancel":
                payload = self._read_json()
                target_id = payload.get("target_id")
                if not target_id:
                    self._json_response(400, {"error": "target_id is required"})
                    return

                job = state.store.mark_cancel_requested(target_id)
                if not job:
                    self._json_response(404, {"error": "job not found"})
                    return

                process = state.get_process(target_id)
                if process is not None and _can_terminate_process(process):
                    try:
                        process.terminate()
                    except OSError:
                        pass
                elif process is not None:
                    process = None

                if job.get("status") in {"queued", "running"} and process is None:
                    job = state.store.update_job(
                        target_id, status="cancelled", detail="Canceled", progress=""
                    )
                elif job.get("status") == "running":
                    job = state.store.update_job(
                        target_id,
                        status="running",
                        detail="Cancel requested",
                    )

                self._json_response(200, {"job": job})
                return

            if self.path == "/jobs/delete":
                payload = self._read_json()
                target_id = payload.get("target_id")
                if not target_id:
                    self._json_response(400, {"error": "target_id is required"})
                    return

                deleted, reason = state.store.delete_job(target_id)
                if not deleted and reason == "not_found":
                    self._json_response(404, {"error": "job not found"})
                    return
                if not deleted and reason == "active":
                    self._json_response(409, {"error": "cannot delete active job"})
                    return

                self._json_response(200, {"ok": True})
                return

            if self.path == "/shutdown":
                self._json_response(200, {"ok": True})
                state.request_shutdown()
                return

            self._json_response(404, {"error": "not found"})

        def log_message(self, format, *args):
            """Suppress the default BaseHTTPRequestHandler access log."""
            return

    return Handler


# Production binding: the module-level singleton state. Tests
# construct their own via ``_make_handler(fake_state)``. STATE is
# imported lazily at module-load time (after download_service.py has
# finished initialising) to break the circular dependency.
def _build_default_handler():
    """Build the production handler from process-wide state and settings."""
    import config

    from .download_service import STATE

    return _make_handler(STATE, auth_token=config.settings.download_service_token)


Handler = _build_default_handler()

__all__ = ["Handler", "_make_handler"]
