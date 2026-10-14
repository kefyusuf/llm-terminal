"""Observe saturated download queues through the real service HTTP boundary."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from downloads.store import DownloadStore

ROOT = Path(__file__).resolve().parents[1]


def _api(port, path, body=None):
    request = Request(f"http://127.0.0.1:{port}{path}",
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={"Authorization": "Bearer queue-test-only"})
    with build_opener(ProxyHandler({})).open(request, timeout=2) as response:
        return json.load(response)


def _wait(predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError("Queue test deadline exceeded")


def _start(tmp_path):
    store = DownloadStore(tmp_path / "jobs.db")
    script = (
        "from pathlib import Path\nimport sys,time\n"
        "Path(sys.argv[1]).touch()\n"
        "deadline=time.monotonic()+45\n"
        "while not Path(sys.argv[2]).exists():\n"
        "    if time.monotonic()>deadline: raise SystemExit(1)\n"
        "    time.sleep(.05)\n"
    )
    targets = []
    for index in range(3):
        job, _ = store.upsert_job({"source": "Ollama", "name": f"queue-test-{index}"})
        targets.append(job["target_id"])
        # Seed a harmless external subprocess instead of invoking a model provider.
        command = [sys.executable, "-c", script,
                   str(tmp_path / f"started-{index}"), str(tmp_path / f"release-{index}")]
        with store._connect() as conn:
            conn.execute("UPDATE jobs SET command_json = ? WHERE target_id = ?",
                         (json.dumps(command), job["target_id"]))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith("AIMODEL_")}
    env.update(AIMODEL_DOWNLOAD_DB_PATH=str(tmp_path / "jobs.db"),
               AIMODEL_CACHE_DB_PATH=str(tmp_path / "cache.db"),
               AIMODEL_HF_MODELS_DIR=str(tmp_path / "models"),
               AIMODEL_DOWNLOAD_SERVICE_HOST="127.0.0.1",
               AIMODEL_DOWNLOAD_SERVICE_PORT=str(port),
               AIMODEL_DOWNLOAD_SERVICE_TOKEN="queue-test-only",
               AIMODEL_DOWNLOAD_MAX_WORKERS="2")
    process = subprocess.Popen([sys.executable, "-m", "downloads.download_service"],
                               cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return process, port, targets


def _finish(process, port, tmp_path):
    for index in range(3):
        (tmp_path / f"release-{index}").touch()
    try:
        if process.poll() is None:
            _api(port, "/shutdown", {})
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def _jobs(port):
    return {job["target_id"]: job for job in _api(port, "/jobs")["jobs"]}


def test_waiting_job_can_be_cancelled_and_deleted_while_workers_are_busy(tmp_path):
    process, port, targets = _start(tmp_path)
    try:
        _wait(lambda: all((tmp_path / f"started-{index}").exists() for index in (0, 1)))
        assert _jobs(port)[targets[2]]["status"] == "queued"
        cancelled = _api(port, "/jobs/cancel", {"target_id": targets[2]})["job"]
        assert cancelled["status"] == "cancelled"
        assert all(_jobs(port)[target]["status"] == "running" for target in targets[:2])
        _api(port, "/jobs/delete", {"target_id": targets[2]})
        assert targets[2] not in _jobs(port)
        for index in (0, 1):
            (tmp_path / f"release-{index}").touch()
        _wait(lambda: all(_jobs(port)[target]["status"] == "completed" for target in targets[:2]))
        assert not (tmp_path / "started-2").exists()
    finally:
        _finish(process, port, tmp_path)


def test_waiting_job_starts_when_one_worker_finishes(tmp_path):
    process, port, targets = _start(tmp_path)
    try:
        _wait(lambda: all((tmp_path / f"started-{index}").exists() for index in (0, 1)))
        assert _jobs(port)[targets[2]]["status"] == "queued"
        (tmp_path / "release-0").touch()
        _wait(lambda: (tmp_path / "started-2").exists())
        assert _jobs(port)[targets[1]]["status"] == "running"
        assert _jobs(port)[targets[2]]["status"] == "running"
    finally:
        _finish(process, port, tmp_path)


def test_shutdown_preserves_unclaimed_job_when_workers_are_busy(tmp_path):
    process, port, targets = _start(tmp_path)
    try:
        _wait(lambda: all((tmp_path / f"started-{index}").exists() for index in (0, 1)))
        assert _jobs(port)[targets[2]]["status"] == "queued"
        _api(port, "/shutdown", {})
        process.communicate(timeout=10)
        assert process.returncode == 0
        store = DownloadStore(tmp_path / "jobs.db")
        assert store.get_job_by_target(targets[2])["status"] == "queued"
        assert all(store.get_job_by_target(target)["status"] == "cancelled" for target in targets[:2])
        assert not (tmp_path / "started-2").exists()
    finally:
        _finish(process, port, tmp_path)
