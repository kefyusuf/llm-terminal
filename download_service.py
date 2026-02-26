import json
import re
import sqlite3
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from download_manager import build_download_command, download_target_id


DB_PATH = Path(__file__).resolve().with_name("downloads.db")
HOST = "127.0.0.1"
PORT = 8765


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
            row = conn.execute(
                "SELECT * FROM jobs WHERE target_id = ?", (target_id,)
            ).fetchone()
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

            row = conn.execute(
                "SELECT * FROM jobs WHERE target_id = ?", (target_id,)
            ).fetchone()
        return self._row_to_dict(row), True

    def mark_cancel_requested(self, target_id):
        now = time.time()
        with self.lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE target_id = ?",
                (now, target_id),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE target_id = ?", (target_id,)
            ).fetchone()
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
            updated = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (row["id"],)
            ).fetchone()
        return self._row_to_dict(updated)

    def update_job(
        self, target_id, status=None, detail=None, progress=None, return_code=None
    ):
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
            row = conn.execute(
                "SELECT * FROM jobs WHERE target_id = ?", (target_id,)
            ).fetchone()
        return self._row_to_dict(row)

    def get_command(self, target_id):
        with self.lock, self._connect() as conn:
            row = conn.execute(
                "SELECT command_json FROM jobs WHERE target_id = ?", (target_id,)
            ).fetchone()
        if row is None:
            return []
        return json.loads(row[0])


class DownloadServiceState:
    def __init__(self):
        self.store = DownloadStore(DB_PATH)
        self.running_processes = {}
        self.running_lock = threading.Lock()
        self.stop_event = threading.Event()

    def get_process(self, target_id):
        with self.running_lock:
            return self.running_processes.get(target_id)

    def set_process(self, target_id, process):
        with self.running_lock:
            self.running_processes[target_id] = process

    def clear_process(self, target_id):
        with self.running_lock:
            self.running_processes.pop(target_id, None)


STATE = DownloadServiceState()


def _extract_progress(line):
    match = re.search(r"(\d{1,3})%", line)
    if not match:
        return None
    value = int(match.group(1))
    if 0 <= value <= 100:
        return f"{value}%"
    return None


def worker_loop():
    while not STATE.stop_event.is_set():
        job = STATE.store.claim_next_queued()
        if job is None:
            time.sleep(0.25)
            continue

        target_id = job["target_id"]
        latest = STATE.store.get_job_by_target(target_id)
        if latest and latest.get("cancel_requested"):
            STATE.store.update_job(
                target_id, status="cancelled", detail="Canceled", progress=""
            )
            continue

        try:
            cmd = STATE.store.get_command(target_id)
            if not cmd:
                STATE.store.update_job(
                    target_id, status="failed", detail="missing command", return_code=1
                )
                continue

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            STATE.set_process(target_id, process)
            start = time.monotonic()
            last_update = start

            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.strip()

                    latest = STATE.store.get_job_by_target(target_id)
                    if latest and latest.get("cancel_requested"):
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
            latest = STATE.store.get_job_by_target(target_id)
            if latest and latest.get("cancel_requested"):
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
                STATE.store.update_job(
                    target_id,
                    status="failed",
                    detail="download command exited with non-zero status",
                    progress="",
                    return_code=return_code,
                )
        except FileNotFoundError:
            STATE.store.update_job(
                target_id,
                status="failed",
                detail="required command not found",
                return_code=127,
            )
        except OSError as exc:
            STATE.store.update_job(
                target_id, status="failed", detail=str(exc)[:180], return_code=1
            )
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
            self._json_response(200, {"ok": True})
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
            if process is not None:
                try:
                    process.terminate()
                except OSError:
                    pass

            if job.get("status") == "queued":
                job = STATE.store.update_job(
                    target_id, status="cancelled", detail="Canceled", progress=""
                )

            self._json_response(200, {"job": job})
            return

        self._json_response(404, {"error": "not found"})

    def log_message(self, format, *args):
        return


def main():
    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        STATE.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
