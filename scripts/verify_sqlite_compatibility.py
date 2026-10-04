"""Rehearse terminal-job backup compatibility in two installed candidates; no workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.package_candidate import verify_manifest
from scripts.verify_hf_recovery import prepare_workspace


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def create_backup(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists():
        raise ValueError("backup destination must be fresh")
    wal = Path(str(source) + "-wal")
    total_bytes = source.stat().st_size + (wal.stat().st_size if wal.exists() else 0)
    if total_bytes > 256 * 1024 * 1024:
        raise ValueError("database exceeds the 256 MiB rehearsal limit")
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as live:
        live.execute("BEGIN")
        count = live.execute("SELECT count(*) FROM jobs").fetchone()[0]
        active = live.execute("SELECT count(*) FROM jobs WHERE status IN ('queued', 'running', 'canceling')").fetchone()[0]
        if active:
            raise ValueError("active jobs must be stopped before compatibility qualification")
        if count > 1000:
            raise ValueError("job count exceeds the 1000-record rehearsal limit")
        with destination.open("xb"):
            pass
        deadline = time.monotonic() + 30

        def progress(_status, _remaining, _total):
            if time.monotonic() > deadline:
                raise TimeoutError("SQLite backup deadline exceeded")

        with sqlite3.connect(destination) as backup:
            live.backup(backup, pages=256, progress=progress, sleep=0.05)
            integrity = backup.execute("PRAGMA integrity_check").fetchone()[0]
        live.rollback()
    if integrity != "ok":
        raise ValueError("backup integrity check failed")
    return {"integrity": integrity, "job_count": count, "backup_sha256": digest(destination)}


def validate_candidate_receipt(probe, manifest):
    wheel = next(item for item in manifest["artifacts"] if item["kind"] == "wheel")
    if probe.get("wheel_sha256") != wheel["sha256"] or probe.get("version") != manifest["version"]:
        raise ValueError("installed wheel receipt does not match the qualified candidate")


def compare_contracts(previous, current):
    fields = ("integrity", "job_count", "record_sha256")
    for contract in (previous, current):
        count = contract.get("job_count")
        record_digest = contract.get("record_sha256")
        if (type(count) is not int or not 0 <= count <= 1000 or not isinstance(record_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", record_digest)):
            raise ValueError("restored job contract is malformed")
    if previous.get("integrity") != "ok" or current.get("integrity") != "ok" or any(
        previous.get(field) != current.get(field) for field in fields
    ):
        raise ValueError("restored job contract differs between installed candidates")


PROBE = """
import hashlib, importlib.metadata, json, sqlite3, sys
from pathlib import Path
import downloads.store
distribution = importlib.metadata.distribution('ai-model-explorer')
origin = Path(downloads.store.__file__).resolve()
recorded = {Path(distribution.locate_file(item)).resolve() for item in distribution.files or []}
if not origin.is_relative_to(Path(sys.prefix).resolve()) or origin not in recorded:
    raise ValueError('store must belong to the installed distribution')
receipt = json.loads(distribution.read_text('direct_url.json') or '{}')
store = downloads.store.DownloadStore(sys.argv[1])
jobs = store.list_jobs(limit=1001)
if len(jobs) > 1000 or any(job['status'] in {'queued', 'running', 'canceling'} for job in jobs):
    raise ValueError('only bounded terminal jobs may be inspected')
records = []
for job in sorted(jobs, key=lambda item: item['target_id']):
    record = {key: job.get(key) for key in ('target_id', 'source', 'status', 'artifact',
              'artifact_metadata', 'download_plan', 'download_plan_status')}
    record['command'] = store.get_command(job['target_id'])
    records.append(record)
with sqlite3.connect(sys.argv[1]) as connection:
    integrity = connection.execute('PRAGMA integrity_check').fetchone()[0]
print(json.dumps({'version': distribution.version,
      'wheel_sha256': receipt.get('archive_info', {}).get('hashes', {}).get('sha256'),
      'integrity': integrity, 'job_count': len(jobs), 'record_sha256': hashlib.sha256(
          json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()}))
"""


def rehearse(args):
    manifests = {}
    for label in ("previous", "current"):
        directory = getattr(args, label + "_dist")
        with (directory / "candidate-manifest.json").open("rb") as handle:
            raw = handle.read(65537)
        if len(raw) > 65536:
            raise ValueError("candidate manifest exceeds the input limit")
        manifest = json.loads(raw)
        verify_manifest(directory, manifest, getattr(args, label + "_source"))
        manifests[label] = manifest
    root = prepare_workspace(args.work_dir)
    backup = root / "backup.db"
    snapshot = create_backup(args.database, backup)
    probes = {}
    for label in ("previous", "current"):
        copied = root / (label + ".db")
        shutil.copyfile(backup, copied)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(AIMODEL_CACHE_DB_PATH=str(root / (label + "-cache.db")),
                   AIMODEL_DOWNLOAD_DB_PATH=str(copied), AIMODEL_HF_MODELS_DIR=str(root / (label + "-models")))
        executable = Path(getattr(args, label + "_python")).resolve()
        result = subprocess.run([str(executable), "-I", "-c", PROBE, str(copied)],
                                cwd=root, env=env, capture_output=True, text=True, timeout=30, check=True)
        probes[label] = json.loads(result.stdout)
        validate_candidate_receipt(probes[label], manifests[label])
        if probes[label]["job_count"] != snapshot["job_count"]:
            raise ValueError("restored record count differs from backup")
    compare_contracts(probes["previous"], probes["current"])
    if digest(backup) != snapshot["backup_sha256"]:
        raise ValueError("immutable backup changed during rehearsal")
    report = {"schema_version": 1, "status": "passed", "python": sys.version.split()[0], "snapshot": snapshot,
              "candidates": {label: {"source_sha": manifests[label]["source_sha"], **probes[label]}
                             for label in probes}, "workers_started": False,
              "model_files_touched": False, "scope": "terminal-job store contract compatibility only"}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    for label in ("previous", "current"):
        parser.add_argument("--" + label + "-dist", type=Path, required=True)
        parser.add_argument("--" + label + "-python", type=Path, required=True)
        parser.add_argument("--" + label + "-source", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(rehearse(args), indent=2))
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        raise SystemExit("SQLite compatibility rehearsal failed: " + type(exc).__name__) from exc


if __name__ == "__main__":
    main()
