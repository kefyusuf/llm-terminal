"""Backup snapshots and installed-candidate receipts fail closed."""
import json
import os
import sqlite3
import venv
from types import SimpleNamespace

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


@pytest.mark.skipif(os.name == "nt", reason="POSIX venv interpreter symlinks")
def test_rehearsal_preserves_the_selected_venv_interpreter(tmp_path, monkeypatch):
    from scripts import verify_sqlite_compatibility as rehearsal

    environment = tmp_path / "candidate-env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    executable = environment / "bin/python"
    assert executable.is_symlink()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = {"source_sha": "a" * 40, "version": "1.0.1",
                "artifacts": [{"kind": "wheel", "sha256": "b" * 64}]}
    (candidate / "candidate-manifest.json").write_text(json.dumps(manifest))
    source = tmp_path / "jobs.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE jobs(status TEXT)")
        conn.execute("INSERT INTO jobs VALUES ('completed')")
    monkeypatch.setattr(rehearsal, "verify_manifest", lambda *_: None)
    # Exercise the real isolated interpreter launch while replacing the package
    # contract probe with an environment-identity check, not receipt validation.
    monkeypatch.setattr(rehearsal, "PROBE", f"""
import json, sys
assert sys.prefix == {str(environment)!r}, (sys.prefix, sys.executable)
print(json.dumps({{'version':'1.0.1','wheel_sha256':{'b' * 64!r},
 'integrity':'ok','job_count':1,'record_sha256':{'c' * 64!r}}}))
""")
    args = SimpleNamespace(database=source, work_dir=tmp_path / "rehearsal",
        previous_dist=candidate, current_dist=candidate,
        previous_python=executable, current_python=executable,
        previous_source="a" * 40, current_source="a" * 40)
    assert rehearsal.rehearse(args)["status"] == "passed"
