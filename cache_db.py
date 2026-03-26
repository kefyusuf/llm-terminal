import json
import sqlite3
import threading
import time
from pathlib import Path

import config

_cache_db_path: Path | None = None
_init_lock = threading.Lock()


def get_cache_db_path() -> Path:
    global _cache_db_path
    if _cache_db_path is None:
        _cache_db_path = config.settings.cache_db_path
    return _cache_db_path


def _connect():
    conn = sqlite3.connect(str(get_cache_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _init_lock:
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_cache (
                    source TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    cached_at REAL NOT NULL,
                    PRIMARY KEY (source, model_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hardware_snapshot (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    specs_json TEXT NOT NULL,
                    cached_at REAL NOT NULL
                )
                """
            )


def get_model_cache(source: str, model_id: str) -> dict | None:
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT metadata_json, cached_at FROM model_cache WHERE source = ? AND model_id = ?",
                (source, model_id),
            ).fetchone()

            if row is None:
                return None

            cached_at = row["cached_at"]
            if time.time() - cached_at > config.settings.cache_ttl_seconds:
                conn.execute(
                    "DELETE FROM model_cache WHERE source = ? AND model_id = ?",
                    (source, model_id),
                )
                return None

            return json.loads(row["metadata_json"])
    except (sqlite3.Error, json.JSONDecodeError, OSError):
        return None


def set_model_cache(source: str, model_id: str, metadata: dict) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_cache (source, model_id, metadata_json, cached_at)
                VALUES (?, ?, ?, ?)
                """,
                (source, model_id, json.dumps(metadata), time.time()),
            )
    except (sqlite3.Error, json.JSONDecodeError, OSError):
        pass


def cleanup_old_entries(
    max_per_source: int | None = None,
    ttl_seconds: int | None = None,
) -> None:
    if max_per_source is None:
        max_per_source = config.settings.cache_max_per_source
    if ttl_seconds is None:
        ttl_seconds = config.settings.cache_ttl_seconds
    try:
        with _connect() as conn:
            cutoff_time = time.time() - ttl_seconds
            conn.execute(
                "DELETE FROM model_cache WHERE cached_at < ?",
                (cutoff_time,),
            )

            if max_per_source <= 0:
                conn.execute("DELETE FROM model_cache")
                return

            conn.execute(
                """
                DELETE FROM model_cache WHERE rowid IN (
                    SELECT rowid FROM (
                        SELECT rowid, ROW_NUMBER() OVER (
                            PARTITION BY source ORDER BY cached_at DESC
                        ) AS rn
                        FROM model_cache
                    ) ranked
                    WHERE rn > ?
                )
                """,
                (max_per_source,),
            )
    except (sqlite3.Error, OSError):
        pass


def get_hardware_snapshot() -> dict | None:
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT specs_json, cached_at FROM hardware_snapshot WHERE id = 1"
            ).fetchone()

            if row is None:
                return None

            cached_at = row["cached_at"]
            if time.time() - cached_at > config.settings.cache_ttl_seconds:
                conn.execute("DELETE FROM hardware_snapshot WHERE id = 1")
                return None

            return json.loads(row["specs_json"])
    except (sqlite3.Error, json.JSONDecodeError, OSError):
        return None


def set_hardware_snapshot(specs: dict) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO hardware_snapshot (id, specs_json, cached_at)
                VALUES (1, ?, ?)
                """,
                (json.dumps(specs), time.time()),
            )
    except (sqlite3.Error, json.JSONDecodeError, OSError):
        pass
