"""Download job persistence — SQLite-backed CRUD + migrations.

Extracted from downloads/download_service.py during the architecture
review (candidate #5). Owns the ``jobs`` table schema, the
``DownloadStore`` repository, and three one-shot migrations:

- ``normalize_target_ids``: collapse duplicate target_ids that
  accumulated across versions.
- ``recover_orphaned_running_jobs``: mark running jobs as failed on
  service startup (they were abandoned by the previous process).
- ``migrate_legacy_hf_commands``: rewrite old
  ``huggingface_hub.commands.huggingface_cli ...`` commands to the
  new ``hf_api_download`` sentinel format.

The store has zero imports outside ``sqlite3``/``json``/``threading``/
``pathlib`` and the project-internal ``download_manager`` (for
``download_target_id`` / ``build_download_command``). It can be
instantiated with a fresh path in tests and exercised in isolation
without spinning up the service.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from core.artifacts import bound_artifact_metadata

from .download_manager import build_download_command, download_target_id
from .preflight import plan_hf_download


class DownloadStore:
    """SQLite-backed repository for download jobs.

    Threading: every public method acquires ``self.lock`` and opens a
    fresh connection (with ``check_same_thread=False``). This is a
    defensive belt-and-braces — the connection-level isolation is
    enough, the lock is just so we never observe a half-applied
    transaction from another thread.
    """

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self.lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
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

            columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "artifact_metadata_json" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN artifact_metadata_json TEXT")
            if "download_plan_json" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN download_plan_json TEXT")

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
        artifact = None
        try:
            command = json.loads(row["command_json"])
            if isinstance(command, list) and len(command) >= 3 and command[0] == "hf_api_download":
                revision = command[3] if len(command) > 3 else None
                artifact = {
                    "repository": command[1],
                    "filename": command[2],
                    "revision": revision,
                    "revision_status": "pinned" if revision else "unknown",
                }
        except (TypeError, ValueError):
            pass
        metadata = None
        if artifact and row["artifact_metadata_json"]:
            try:
                metadata = bound_artifact_metadata({
                    "id": artifact["repository"], "target_file": artifact["filename"],
                    "resolved_revision": artifact["revision"],
                    "artifact_metadata": json.loads(row["artifact_metadata_json"]),
                })
            except (TypeError, ValueError):
                metadata = None
        plan = None
        plan_status = "unplanned"
        if row["download_plan_json"]:
            plan_status = "invalid"
            try:
                candidate = json.loads(row["download_plan_json"])
                if (isinstance(candidate, dict) and candidate.get("schema_version") == 1 and artifact
                    and candidate.get("repository") == artifact["repository"]
                    and candidate.get("filename") == artifact["filename"]
                    and candidate.get("resolved_revision") == artifact["revision"]
                    and candidate.get("size_bytes") == (metadata or {}).get("size_bytes")
                    and candidate.get("sha256") == (metadata or {}).get("sha256")):
                    plan = candidate
                    plan_status = "available"
            except (ValueError, TypeError):
                plan = None
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
            "artifact": artifact,
            "artifact_metadata": metadata,
            "download_plan": plan,
            "download_plan_status": plan_status,
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

    def _reserved_bytes(self, conn, exclude_target=None):
        rows = conn.execute("SELECT * FROM jobs WHERE status IN ('queued', 'running') AND target_id != ?", (exclude_target or "",)).fetchall()
        return sum((self._row_to_dict(row).get("artifact_metadata") or {}).get("size_bytes") or 0 for row in rows)

    def plan_job(self, model, models_dir):
        with self.lock, self._connect() as conn:
            reserved = self._reserved_bytes(conn, download_target_id(model))
            return plan_hf_download(model, models_dir, reserved_bytes=reserved,
                                    allow_unknown_size=model.get("allow_unknown_size") is True)

    def reserved_download_bytes(self, exclude_target=None):
        with self.lock, self._connect() as conn:
            return self._reserved_bytes(conn, exclude_target)

    def get_execution_metadata(self, target_id):
        return self.get_job_by_target(target_id)

    def upsert_job(self, model, *, models_dir=None):
        target_id = download_target_id(model)
        command = build_download_command(model)
        metadata = bound_artifact_metadata(model) if model.get("source") == "Hugging Face" else None
        now = time.time()

        with self.lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM jobs WHERE target_id = ?", (target_id,)
            ).fetchone()

            if existing is not None and existing["status"] in {"queued", "running"}:
                return self._row_to_dict(existing), False

            plan = None
            if models_dir is not None and model.get("source") == "Hugging Face":
                plan = plan_hf_download(model, models_dir, reserved_bytes=self._reserved_bytes(conn, target_id),
                                        allow_unknown_size=model.get("allow_unknown_size") is True)
                if not plan["allowed"]:
                    raise ValueError("download preflight blocked: " + plan["status"])

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        target_id, source, publisher, name, command_json, status,
                        detail, progress, created_at, updated_at, cancel_requested, return_code, artifact_metadata_json, download_plan_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
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
                        json.dumps(metadata) if metadata is not None else None,
                        json.dumps(plan) if plan is not None else None,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET source = ?, publisher = ?, name = ?, command_json = ?,
                        status = 'queued', detail = 'Queued', progress = '',
                        updated_at = ?, cancel_requested = 0, return_code = NULL, artifact_metadata_json = ?, download_plan_json = ?
                    WHERE target_id = ?
                    """,
                    (
                        model.get("source", existing["source"]),
                        model.get("publisher", existing["publisher"]),
                        model.get("name", existing["name"]),
                        json.dumps(command),
                        now,
                        json.dumps(metadata) if metadata is not None else None,
                        json.dumps(plan) if plan is not None else None,
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


# Re-exported for tests/legacy callers; this used to live in
# download_service.py and the migration module imports it directly.
def normalize_target_id(target_id: str) -> str:
    from .download_manager import normalize_target_id as _normalize

    return _normalize(target_id)
