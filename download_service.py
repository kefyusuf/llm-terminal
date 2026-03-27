import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from download_manager import (
    build_download_command,
    download_target_id,
    normalize_target_id,
)
from utils import extract_download_progress

DB_PATH = Path(__file__).resolve().with_name("downloads.db")
HOST = "127.0.0.1"
PORT = 8765
SERVICE_VERSION = "1.7"


class DownloadStore:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self.lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    publisher TEXT,
                    name TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    progress TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    return_code INTEGER
                )
                """
            )

    def normalize_target_ids(self):
        with self.lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, target_id, updated_at FROM jobs ORDER BY updated_at DESC, id DESC"
            ).fetchall()

            keep_by_target = {}
            delete_ids = []
            update_rows = []

            for row in rows:
                row_id = row["id"]
                normalized = normalize_target_id(row["target_id"])
                if normalized in keep_by_target:
                    delete_ids.append(row_id)
                    continue
                keep_by_target[normalized] = row_id
                if normalized != row["target_id"]:
                    update_rows.append((normalized, row_id))

            for normalized, row_id in update_rows:
                conn.execute("UPDATE jobs SET target_id = ? WHERE id = ?", (normalized, row_id))

            for row_id in delete_ids:
                conn.execute("DELETE FROM jobs WHERE id = ?", (row_id,))

    def recover_orphaned_running_jobs(self):
        now = time.time()
        with self.lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', detail = 'service restarted during download',
                    progress = '', cancel_requested = 0, updated_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )

    def migrate_legacy_hf_commands(self):
        with self.lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, source, command_json FROM jobs WHERE source = 'Hugging Face'"
            ).fetchall()
            for row in rows:
                row_id = row["id"]
                try:
                    command = json.loads(row["command_json"])
                except (TypeError, ValueError):
                    continue
                if not isinstance(command, list) or len(command) < 2:
                    continue
                if command[0] == "hf_api_download":
                    continue
                if "huggingface_hub.commands.huggingface_cli" not in command:
                    continue
                repo_id = None
                if "download" in command:
                    try:
                        idx = command.index("download")
                        repo_id = command[idx + 1]
                    except (ValueError, IndexError):
                        repo_id = None
                if not repo_id:
                    continue
                conn.execute(
                    "UPDATE jobs SET command_json = ? WHERE id = ?",
                    (json.dumps(["hf_api_download", repo_id]), row_id),
                )

    def _row_to_dict(self, row):
        if row is None:
            return None
        return {
            "id": row["id"],
            "target_id": row["target_id"],
            "source": row["source"],
            "publisher": row["publisher"] or "-",
            "name": row["name"],
            "status": row["status"],
            "detail": row["detail"] or "",
            "progress": row["progress"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "cancel_requested": bool(row["cancel_requested"]),
            "return_code": row["return_code"],
        }

    def list_jobs(self, limit=50):
        with self.lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_job_by_target(self, target_id):
        with self.lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE target_id = ?", (target_id,)).fetchone()
        return self._row_to_dict(row)

    def upsert_job(self, model):
        target_id = download_target_id(model)
        command = build_download_command(model)
        now = time.time()

        with self.lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE target_id = ?", (target_id,)
            ).fetchone()

            if existing is not None and existing["status"] in {"queued", "running"}:
                return self._row_to_dict(existing), False

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        target_id, source, publisher, name, command_json, status,
                        detail, progress, created_at, updated_at, cancel_requested, return_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
                    """,
                    (
                        target_id,
                        model.get("source", "-"),
                        model.get("publisher", "-"),
                        model.get("name", target_id),
                        json.dumps(command),
                        "queued",
                        "Queued",
                        "",
                        now,
                        now,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET source = ?, publisher = ?, name = ?, command_json = ?,
                        status = 'queued', detail = 'Queued', progress = '',
                        updated_at = ?, cancel_requested = 0, return_code = NULL
                    WHERE target_id = ?
                    """,
                    (
                        model.get("source", existing["source"]),
                        model.get("publisher", existing["publisher"]),
                        model.get("name", existing["name"]),
                        json.dumps(command),
                        now,
                        target_id,
                    ),
                )

            row = conn.execute("SELECT * FROM jobs WHERE target_id = ?", (target_id,)).fetchone()
        return self._row_to_dict(row), True

    def mark_cancel_requested(self, target_id):
        now = time.time()
        with self.lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE target_id = ?",
                (now, target_id),
            )
            row = conn.execute("SELECT * FROM jobs WHERE target_id = ?", (target_id,)).fetchone()
        return self._row_to_dict(row)

    def claim_next_queued(self):
        now = time.time()
        with self.lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY updated_at ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE jobs SET status = 'running', detail = 'Starting', updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
        return self._row_to_dict(updated)

    def update_job(self, target_id, status=None, detail=None, progress=None, return_code=None):
        now = time.time()
        fields = ["updated_at = ?"]
        values = [now]
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if detail is not None:
            fields.append("detail = ?")
            values.append(detail)
        if progress is not None:
            fields.append("progress = ?")
            values.append(progress)
        if return_code is not None:
            fields.append("return_code = ?")
            values.append(return_code)
        values.append(target_id)

        with self.lock, self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE target_id = ?",
                tuple(values),
            )
            row = conn.execute("SELECT * FROM jobs WHERE target_id = ?", (target_id,)).fetchone()
        return self._row_to_dict(row)

    def get_command(self, target_id):
        with self.lock, self._connect() as conn:
            row = conn.execute(
                "SELECT command_json FROM jobs WHERE target_id = ?", (target_id,)
            ).fetchone()
        if row is None:
            return []
        return json.loads(row[0])

    def delete_job(self, target_id):
        with self.lock, self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE target_id = ?", (target_id,)
            ).fetchone()
            if row is None:
                return False, "not_found"
            if row["status"] in {"queued", "running"}:
                return False, "active"
            conn.execute("DELETE FROM jobs WHERE target_id = ?", (target_id,))
        return True, "deleted"


class DownloadServiceState:
    def __init__(self):
        self.store = DownloadStore(DB_PATH)
        self.store.normalize_target_ids()
        self.store.migrate_legacy_hf_commands()
        self.store.recover_orphaned_running_jobs()
        self.running_processes = {}
        self.running_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.server: Any = None
        self.worker_thread: Any = None

    def get_process(self, target_id):
        with self.running_lock:
            return self.running_processes.get(target_id)

    def set_process(self, target_id, process):
        with self.running_lock:
            self.running_processes[target_id] = process

    def clear_process(self, target_id):
        with self.running_lock:
            self.running_processes.pop(target_id, None)

    def snapshot_active_targets(self):
        with self.running_lock:
            return list(self.running_processes.keys())

    def request_shutdown(self):
        self.stop_event.set()
        server = self.server
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()


STATE = DownloadServiceState()


def _service_popen_kwargs():
    import config

    kwargs = {}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # Add HF token to environment if available
    env = os.environ.copy()
    hf_token = getattr(config.settings, "hf_token", None)
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["AIMODEL_HF_TOKEN"] = hf_token
    kwargs["env"] = env
    return kwargs


def _can_terminate_process(process):
    return hasattr(process, "terminate") and callable(getattr(process, "terminate", None))


def _has_duplicates(values):
    return len(values) != len(set(values))


def _extract_progress(line):
    """Wrap :func:`utils.extract_download_progress` returning a ``"N%"`` string."""
    value = extract_download_progress(line)
    return f"{value}%" if value is not None else None


def _is_hf_api_command(command):
    """Return ``True`` when *command* is an internal HF API download payload."""
    return isinstance(command, list) and len(command) > 0 and command[0] == "hf_api_download"


def _repo_id_from_hf_command(command):
    """Extract repository id from an internal HF API command payload."""
    if not _is_hf_api_command(command):
        return ""
    return command[1] if len(command) > 1 else ""


def _cancel_requested(target_id):
    latest = STATE.store.get_job_by_target(target_id)
    return bool(latest and latest.get("cancel_requested"))


def _run_hf_api_download_job(target_id, command):
    repo_id = _repo_id_from_hf_command(command)
    if not repo_id:
        STATE.store.update_job(
            target_id,
            status="failed",
            detail="missing Hugging Face repository id",
            return_code=1,
        )
        return

    STATE.store.update_job(
        target_id,
        status="running",
        detail="Downloading",
        progress="",
    )
    hf_script = (
        "from huggingface_hub import snapshot_download; "
        "import sys; "
        "snapshot_download("
        "repo_id=sys.argv[1], "
        "allow_patterns=['*.gguf'], "
        "local_dir='models', "
        "local_dir_use_symlinks=False"
        ")"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", hf_script, repo_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_service_popen_kwargs(),
    )
    STATE.set_process(target_id, process)
    cancel_sent_at = None

    try:
        while True:
            if _cancel_requested(target_id):
                if cancel_sent_at is None:
                    cancel_sent_at = time.monotonic()
                    try:
                        process.terminate()
                    except OSError:
                        pass
                elif process.poll() is None and (time.monotonic() - cancel_sent_at) > 1.5:
                    try:
                        process.kill()
                    except OSError:
                        pass

            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(0.25)

        if _cancel_requested(target_id):
            STATE.store.update_job(
                target_id,
                status="cancelled",
                detail="Canceled",
                progress="",
                return_code=return_code,
            )
        elif return_code == 0:
            STATE.store.update_job(
                target_id,
                status="completed",
                detail="Completed",
                progress="",
                return_code=0,
            )
        else:
            failure_detail = "hugging face download failed"
            if process.stderr is not None:
                err_text = process.stderr.read().strip()
                if err_text:
                    failure_detail = err_text.splitlines()[-1][:180]
            STATE.store.update_job(
                target_id,
                status="failed",
                detail=failure_detail,
                progress="",
                return_code=return_code,
            )
    except Exception as exc:
        detail = str(exc).strip() or "hugging face download failed"
        STATE.store.update_job(
            target_id,
            status="failed",
            detail=detail[:180],
            progress="",
            return_code=1,
        )
    finally:
        STATE.clear_process(target_id)


def _run_stream_download_job(target_id, command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **_service_popen_kwargs(),
    )
    STATE.set_process(target_id, process)
    start = time.monotonic()
    last_update = start
    last_line = ""

    try:
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if line:
                    last_line = line

                if _cancel_requested(target_id):
                    try:
                        process.terminate()
                    except OSError:
                        pass

                progress = _extract_progress(line)
                now = time.monotonic()
                if progress is not None:
                    STATE.store.update_job(
                        target_id,
                        status="running",
                        detail="Downloading",
                        progress=progress,
                    )
                    last_update = now
                    continue

                if now - last_update >= 1.0:
                    elapsed = int(now - start)
                    STATE.store.update_job(
                        target_id,
                        status="running",
                        detail="Downloading",
                        progress=f"{elapsed}s",
                    )
                    last_update = now

        return_code = process.wait()
        if _cancel_requested(target_id):
            STATE.store.update_job(
                target_id,
                status="cancelled",
                detail="Canceled",
                progress="",
                return_code=return_code,
            )
        elif return_code == 0:
            STATE.store.update_job(
                target_id,
                status="completed",
                detail="Completed",
                progress="",
                return_code=0,
            )
        else:
            failure_detail = "download command exited with non-zero status"
            if last_line:
                failure_detail = last_line[:180]
            STATE.store.update_job(
                target_id,
                status="failed",
                detail=failure_detail,
                progress="",
                return_code=return_code,
            )
    finally:
        STATE.clear_process(target_id)


def worker_loop():
    while not STATE.stop_event.is_set():
        try:
            job = STATE.store.claim_next_queued()
        except Exception:
            time.sleep(0.5)
            continue

        if job is None:
            time.sleep(0.25)
            continue

        target_id = job["target_id"]
        latest = STATE.store.get_job_by_target(target_id)
        if latest and latest.get("cancel_requested"):
            STATE.store.update_job(target_id, status="cancelled", detail="Canceled", progress="")
            continue

        try:
            cmd = STATE.store.get_command(target_id)
            if not cmd:
                STATE.store.update_job(
                    target_id, status="failed", detail="missing command", return_code=1
                )
                continue

            if _is_hf_api_command(cmd):
                _run_hf_api_download_job(target_id, cmd)
                continue

            _run_stream_download_job(target_id, cmd)
        except FileNotFoundError:
            STATE.store.update_job(
                target_id,
                status="failed",
                detail="required command not found",
                return_code=127,
            )
        except OSError as exc:
            STATE.store.update_job(target_id, status="failed", detail=str(exc)[:180], return_code=1)
        finally:
            STATE.clear_process(target_id)


class Handler(BaseHTTPRequestHandler):
    def _json_response(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json_response(200, {"ok": True, "version": SERVICE_VERSION})
            return

        if parsed.path == "/debug/active":
            active_targets = STATE.snapshot_active_targets()
            self._json_response(
                200,
                {
                    "active_targets": active_targets,
                    "count": len(active_targets),
                    "has_duplicates": _has_duplicates(active_targets),
                    "worker_alive": bool(
                        STATE.worker_thread is not None and STATE.worker_thread.is_alive()
                    ),
                },
            )
            return

        if parsed.path == "/jobs":
            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", ["50"])[0])
            jobs = STATE.store.list_jobs(limit=limit)
            self._json_response(200, {"jobs": jobs})
            return

        self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/jobs":
            try:
                payload = self._read_json()
                model = payload.get("model") or {}
                job, created_or_queued = STATE.store.upsert_job(model)
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

            job = STATE.store.mark_cancel_requested(target_id)
            if not job:
                self._json_response(404, {"error": "job not found"})
                return

            process = STATE.get_process(target_id)
            if process is not None and _can_terminate_process(process):
                try:
                    process.terminate()
                except OSError:
                    pass
            elif process is not None:
                process = None

            if job.get("status") in {"queued", "running"} and process is None:
                job = STATE.store.update_job(
                    target_id, status="cancelled", detail="Canceled", progress=""
                )
            elif job.get("status") == "running":
                job = STATE.store.update_job(
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

            deleted, reason = STATE.store.delete_job(target_id)
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
            STATE.request_shutdown()
            return

        self._json_response(404, {"error": "not found"})

    def log_message(self, format, *args):
        return


def main():
    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()
    STATE.worker_thread = worker

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    STATE.server = server
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        STATE.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
