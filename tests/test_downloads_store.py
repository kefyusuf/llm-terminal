"""Download history CRUD, migrations, and interrupted work recovery."""

from __future__ import annotations

import json
import sqlite3

import pytest

from downloads.store import DownloadStore


def _make_store(tmp_path):
    return DownloadStore(tmp_path / "jobs.db")


def _make_model(target_id: str = "test/repo", source: str = "Hugging Face", name: str = "test/repo"):
    model = {
        "source": source,
        "publisher": "test",
        "id": target_id,
        "name": name,
    }
    if source == "Hugging Face":
        model["target_file"] = "model.Q4_K_M.gguf"
    return model


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_store_creates_schema_on_init(tmp_path):
    store = _make_store(tmp_path)
    assert store is not None
    # Schema is reachable — list_jobs returns an empty list, not an error
    assert store.list_jobs() == []


def test_store_creates_parent_directory(tmp_path):
    nested = tmp_path / "deep" / "nested" / "jobs.db"
    store = DownloadStore(nested)
    assert nested.parent.exists()
    assert store.list_jobs() == []


# ---------------------------------------------------------------------------
# upsert_job
# ---------------------------------------------------------------------------


def test_upsert_creates_new_job(tmp_path):
    store = _make_store(tmp_path)
    model = _make_model()
    job, created = store.upsert_job(model)
    assert created is True
    assert job["target_id"] == "hugging face:test/repo"
    assert job["status"] == "queued"


def test_upsert_idempotent_for_active_job(tmp_path):
    store = _make_store(tmp_path)
    model = _make_model()
    store.upsert_job(model)
    # Second upsert of same target_id while queued should NOT recreate
    job, created = store.upsert_job(model)
    assert created is False
    assert job["status"] == "queued"


def test_upsert_replaces_completed_job(tmp_path):
    store = _make_store(tmp_path)
    model = _make_model()
    store.upsert_job(model)
    # Mark completed manually — the store normalises source:identifier,
    # so the actual target_id is "hugging face:test/repo".
    store.update_job("hugging face:test/repo", status="completed", return_code=0)
    # New upsert of same model should re-queue
    job, created = store.upsert_job(model)
    assert created is True
    assert job["status"] == "queued"


# ---------------------------------------------------------------------------
# update_job
# ---------------------------------------------------------------------------


def test_update_job_writes_status_detail_progress(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_job(_make_model())

    job = store.update_job(
        "hugging face:test/repo",
        status="running",
        detail="Downloading",
        progress="42%",
    )
    assert job["status"] == "running"
    assert job["detail"] == "Downloading"
    assert job["progress"] == "42%"


# ---------------------------------------------------------------------------
# mark_cancel_requested
# ---------------------------------------------------------------------------


def test_mark_cancel_requested_flips_flag(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_job(_make_model())

    job = store.mark_cancel_requested("hugging face:test/repo")
    assert job["cancel_requested"] is True


# ---------------------------------------------------------------------------
# delete_job
# ---------------------------------------------------------------------------


def test_delete_completed_job_succeeds(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_job(_make_model())
    store.update_job("hugging face:test/repo", status="completed", return_code=0)
    ok, reason = store.delete_job("hugging face:test/repo")
    assert ok is True
    assert reason == "deleted"
    assert store.get_job_by_target("hugging face:test/repo") is None


def test_delete_running_job_blocked(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_job(_make_model())
    store.update_job("hugging face:test/repo", status="running")
    ok, reason = store.delete_job("hugging face:test/repo")
    assert ok is False
    assert reason == "active"


def test_delete_missing_job_returns_not_found(tmp_path):
    store = _make_store(tmp_path)
    ok, reason = store.delete_job("nonexistent")
    assert ok is False
    assert reason == "not_found"


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def test_normalize_target_ids_collapses_duplicates(tmp_path):
    store = _make_store(tmp_path)
    # Insert two rows that should normalize to the same target_id
    store.upsert_job(
        {
            "source": "Hugging Face",
            "publisher": "x",
            "id": "owner/repo",
            "name": "owner/repo",
            "target_file": "model.Q4_K_M.gguf",
        }
    )
    # Force a different format that should normalize to the same canonical form
    with store.lock, store._connect() as conn:
        conn.execute(
            "INSERT INTO jobs (target_id, source, publisher, name, command_json, status, detail, progress, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "Hugging Face:owner/repo",
                "Hugging Face",
                "x",
                "owner/repo",
                "[]",
                "completed",
                "",
                "",
                1.0,
                2.0,
            ),
        )

    store.normalize_target_ids()

    # After normalization, both rows should map to the same canonical id
    jobs = store.list_jobs(limit=100)
    # One of the original ids should be canonical; the other deleted
    assert len(jobs) == 1
    assert jobs[0]["target_id"] in {"hugging face:owner/repo", "Hugging Face:owner/repo"}


@pytest.mark.parametrize("status", ["queued", "running", "canceling"])
def test_recover_interrupted_jobs_marks_resumable_and_preserves_command(tmp_path, status):
    store = _make_store(tmp_path)
    job, _ = store.upsert_job(_make_model())
    target = job["target_id"]
    command = store.get_command(target)
    store.update_job(target, status=status, progress="50%", return_code=7)
    with store._connect() as conn:
        conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE target_id = ?", (target,))

    reopened = DownloadStore(store.db_path)
    reopened.recover_interrupted_jobs()

    recovered = reopened.get_job_by_target(target)
    assert recovered["status"] == "interrupted"
    assert "resume" in recovered["detail"]
    assert recovered["progress"] == ""
    assert recovered["cancel_requested"] is False
    assert recovered["return_code"] is None
    assert reopened.get_command(target) == command
    assert reopened.upsert_job(_make_model())[1] is True


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "interrupted"])
def test_recovery_leaves_terminal_jobs_unchanged(tmp_path, status):
    store = _make_store(tmp_path)
    job, _ = store.upsert_job(_make_model())
    expected = store.update_job(job["target_id"], status=status, detail="Saved detail")
    store.recover_interrupted_jobs()
    assert store.get_job_by_target(job["target_id"]) == expected


def test_existing_database_gains_metadata_columns_without_losing_history(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, target_id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL, publisher TEXT, name TEXT NOT NULL,
            command_json TEXT NOT NULL, status TEXT NOT NULL, detail TEXT,
            progress TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
            cancel_requested INTEGER NOT NULL DEFAULT 0, return_code INTEGER
        )""")
        conn.execute("""INSERT INTO jobs (
            target_id, source, name, command_json, status, detail, created_at, updated_at
        ) VALUES ('ollama:legacy', 'Ollama', 'legacy', '["ollama", "pull", "legacy"]',
            'completed', 'Saved download', 1, 2)""")
    store = DownloadStore(path)
    store.recover_interrupted_jobs()
    job = store.get_job_by_target("ollama:legacy")
    assert job["detail"] == "Saved download"
    assert job["status"] == "completed"
    assert job["created_at"] == 1
    assert job["artifact_metadata"] is None
    assert job["download_plan_status"] == "unplanned"
    assert store.get_command("ollama:legacy") == ["ollama", "pull", "legacy"]


def test_canceling_job_remains_reserved_until_worker_stops(tmp_path):
    store = _make_store(tmp_path)
    job, _ = store.upsert_job(_make_model())
    store.update_job(job["target_id"], status="canceling")
    assert store.upsert_job(_make_model())[1] is False
    assert store.delete_job(job["target_id"]) == (False, "active")


def test_migrate_legacy_hf_commands_rewrites_old_format(tmp_path):
    store = _make_store(tmp_path)
    target_id = "hugging face:test/legacy"
    with store.lock, store._connect() as conn:
        conn.execute(
            "INSERT INTO jobs (target_id, source, publisher, name, command_json, status, detail, progress, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                target_id,
                "Hugging Face",
                "test",
                "legacy",
                json.dumps(["huggingface_hub.commands.huggingface_cli", "download", "test/legacy"]),
                "queued",
                "Queued",
                "",
                1.0,
                1.0,
            ),
        )

    store.migrate_legacy_hf_commands()

    cmd = store.get_command(target_id)
    assert cmd == ["hf_api_download", "test/legacy"]


def test_migrate_legacy_hf_commands_leaves_already_migrated_alone(tmp_path):
    store = _make_store(tmp_path)
    target_id = "hugging face:test/already"
    with store.lock, store._connect() as conn:
        conn.execute(
            "INSERT INTO jobs (target_id, source, publisher, name, command_json, status, detail, progress, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                target_id,
                "Hugging Face",
                "test",
                "already",
                json.dumps(["hf_api_download", "test/already"]),
                "queued",
                "Queued",
                "",
                1.0,
                1.0,
            ),
        )

    store.migrate_legacy_hf_commands()

    cmd = store.get_command(target_id)
    assert cmd == ["hf_api_download", "test/already"]


# ---------------------------------------------------------------------------
# list_jobs + get_command
# ---------------------------------------------------------------------------


def test_list_jobs_respects_limit(tmp_path):
    store = _make_store(tmp_path)
    for i in range(5):
        store.upsert_job(_make_model(f"owner/repo-{i}", name=f"repo-{i}"))
    jobs = store.list_jobs(limit=3)
    assert len(jobs) == 3


def test_list_jobs_orders_by_created_at_desc(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_job(_make_model("a/repo", name="a/repo"))
    store.upsert_job(_make_model("b/repo", name="b/repo"))
    store.upsert_job(_make_model("c/repo", name="c/repo"))

    jobs = store.list_jobs()
    # Newest first
    assert jobs[0]["target_id"] == "hugging face:c/repo"
    assert jobs[-1]["target_id"] == "hugging face:a/repo"


def test_get_command_returns_empty_for_missing_job(tmp_path):
    store = _make_store(tmp_path)
    assert store.get_command("nonexistent") == []


def test_get_job_by_target_returns_none_for_missing(tmp_path):
    store = _make_store(tmp_path)
    assert store.get_job_by_target("nonexistent") is None
