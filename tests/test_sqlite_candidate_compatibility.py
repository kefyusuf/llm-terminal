"""Backup snapshots and installed-candidate receipts fail closed."""
import sqlite3

import pytest


def test_backup_includes_committed_wal_and_passes_integrity(tmp_path):
    from scripts.verify_sqlite_compatibility import create_backup

    source = tmp_path / "source.db"
    with sqlite3.connect(source) as live:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("CREATE TABLE jobs(status TEXT, payload TEXT)")
        live.execute("INSERT INTO jobs VALUES ('completed', 'committed WAL data')")
        live.commit()
        result = create_backup(source, tmp_path / "backup.db")
        assert result["job_count"] == 1
        assert result["integrity"] == "ok"
        with sqlite3.connect(tmp_path / "backup.db") as restored:
            assert restored.execute("SELECT payload FROM jobs").fetchone()[0] == "committed WAL data"


def test_active_jobs_cannot_be_qualified_as_terminal_backup(tmp_path):
    from scripts.verify_sqlite_compatibility import create_backup

    source = tmp_path / "source.db"
    with sqlite3.connect(source) as live:
        live.execute("CREATE TABLE jobs(status TEXT)")
        live.execute("INSERT INTO jobs VALUES ('running')")
    with pytest.raises(ValueError, match="active"):
        create_backup(source, tmp_path / "backup.db")
    assert not (tmp_path / "backup.db").exists()


def test_backup_never_overwrites_an_existing_destination(tmp_path):
    from scripts.verify_sqlite_compatibility import create_backup

    destination = tmp_path / "backup.db"
    destination.write_bytes(b"preserved")
    with pytest.raises(ValueError, match="fresh"):
        create_backup(tmp_path / "missing.db", destination)
    assert destination.read_bytes() == b"preserved"


def test_equal_versions_do_not_hide_installed_wheel_mismatch():
    from scripts.verify_sqlite_compatibility import validate_candidate_receipt

    manifest = {"version": "1.0.1", "artifacts": [{"kind": "wheel", "sha256": "a" * 64}]}
    with pytest.raises(ValueError, match="wheel"):
        validate_candidate_receipt({"version": "1.0.1", "wheel_sha256": "b" * 64}, manifest)


def test_contract_change_is_not_reported_as_compatible():
    from scripts.verify_sqlite_compatibility import compare_contracts

    previous = {"integrity": "ok", "job_count": 1, "record_sha256": "a" * 64}
    current = {**previous, "record_sha256": "b" * 64}
    with pytest.raises(ValueError, match="contract"):
        compare_contracts(previous, current)


@pytest.mark.parametrize("change", [{"record_sha256": None}, {"record_sha256": "invalid"}, {"job_count": True}])
def test_matching_malformed_contracts_do_not_pass(change):
    from scripts.verify_sqlite_compatibility import compare_contracts

    contract = {"integrity": "ok", "job_count": 1, "record_sha256": "a" * 64, **change}
    with pytest.raises(ValueError, match="contract"):
        compare_contracts(contract, contract)
