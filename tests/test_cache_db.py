import json
import time

import cache_db


def _configure_temp_cache_db(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(cache_db, "_cache_db_path", db_path)
    monkeypatch.setattr(cache_db.config.settings, "cache_db_path", db_path)
    cache_db.init_db()


def test_cleanup_old_entries_keeps_recent_rows_per_source(tmp_path, monkeypatch):
    _configure_temp_cache_db(tmp_path, monkeypatch)
    now = time.time()

    with cache_db._connect() as conn:
        for source in ("huggingface", "ollama"):
            for i in range(5):
                conn.execute(
                    """
                    INSERT INTO model_cache (source, model_id, metadata_json, cached_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source, f"{source}-{i}", json.dumps({"i": i}), now - i),
                )

    cache_db.cleanup_old_entries(max_per_source=2, ttl_seconds=10_000)

    with cache_db._connect() as conn:
        rows = conn.execute(
            "SELECT source, COUNT(*) AS count FROM model_cache GROUP BY source"
        ).fetchall()

    assert len(rows) == 2
    assert all(row["count"] == 2 for row in rows)


def test_cleanup_old_entries_with_zero_limit_clears_cache(tmp_path, monkeypatch):
    _configure_temp_cache_db(tmp_path, monkeypatch)
    cache_db.set_model_cache("huggingface", "repo/a", {"ok": True})
    cache_db.set_model_cache("ollama", "repo/b", {"ok": True})

    cache_db.cleanup_old_entries(max_per_source=0, ttl_seconds=10_000)

    with cache_db._connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM model_cache").fetchone()[0]

    assert total == 0
