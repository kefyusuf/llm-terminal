"""Service startup failures must not take ownership of existing downloads."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from downloads.store import DownloadStore

ROOT = Path(__file__).resolve().parents[1]


def _environment(tmp_path, db, port):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("AIMODEL_")}
    env.update(AIMODEL_DOWNLOAD_DB_PATH=str(db),
               AIMODEL_CACHE_DB_PATH=str(tmp_path / "cache.db"),
               AIMODEL_HF_MODELS_DIR=str(tmp_path / "models"),
               AIMODEL_DOWNLOAD_SERVICE_HOST="127.0.0.1",
               AIMODEL_DOWNLOAD_SERVICE_PORT=str(port),
               AIMODEL_DOWNLOAD_SERVICE_TOKEN="startup-test-only")
    return env


def _launch(env):
    return subprocess.Popen([sys.executable, "-m", "downloads.download_service"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _stop(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _api(port, path, body=None):
    request = Request(f"http://127.0.0.1:{port}{path}",
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={"Authorization": "Bearer startup-test-only"})
    with build_opener(ProxyHandler({})).open(request, timeout=1) as response:
        return json.load(response)


def _ready(process, port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        assert process.poll() is None, "Service exited before becoming ready"
        try:
            if _api(port, "/health").get("ok"):
                return
        except OSError:
            time.sleep(.05)
    raise AssertionError("Service startup deadline exceeded")


def test_occupied_port_startup_leaves_running_job_unchanged(tmp_path):
    db = tmp_path / "jobs.db"
    store = DownloadStore(db)
    job, _ = store.upsert_job({"source": "Ollama", "name": "startup-test"})
    expected = store.update_job(job["target_id"], status="running",
                                detail="Downloading", progress="37%")
    with socket.socket() as occupied:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        process = _launch(_environment(tmp_path, db, occupied.getsockname()[1]))
        try:
            process.communicate(timeout=10)
            assert process.returncode != 0
            assert store.get_job_by_target(job["target_id"]) == expected
        finally:
            _stop(process)
    port = _port()
    retry = _launch(_environment(tmp_path, db, port))
    try:
        _ready(retry, port)
        recovered = _api(port, "/jobs")["jobs"][0]
        assert recovered["status"] == "failed"
        assert recovered["detail"] == "service restarted during download"
    finally:
        if retry.poll() is None:
            _api(port, "/shutdown", {})
        _stop(retry)


@pytest.mark.parametrize("smoke", [False, True])
def test_second_service_on_same_database_is_rejected_even_on_another_port(tmp_path, smoke):
    db = tmp_path / "jobs.db"
    store = DownloadStore(db)
    job, _ = store.upsert_job({"source": "Ollama", "name": "startup-test"})
    store.update_job(job["target_id"], status="completed")
    port = _port()
    owner = _launch(_environment(tmp_path, db, port))
    second = None
    try:
        _ready(owner, port)
        expected = store.update_job(job["target_id"], status="running",
                                    detail="Downloading", progress="37%")
        env = _environment(tmp_path, db, _port())
        if smoke:
            env["AIMODEL_SMOKE"] = "1"
        second = _launch(env)
        second.communicate(timeout=5)
        assert second.returncode != 0
        jobs = _api(port, "/jobs")["jobs"]
        assert jobs == [expected]
        assert owner.poll() is None
    finally:
        if second is not None:
            _stop(second)
        if owner.poll() is None:
            _api(port, "/shutdown", {})
        _stop(owner)


def test_owner_process_death_releases_database_for_recovery(tmp_path):
    db = tmp_path / "jobs.db"
    store = DownloadStore(db)
    job, _ = store.upsert_job({"source": "Ollama", "name": "startup-test"})
    store.update_job(job["target_id"], status="completed")
    port = _port()
    owner = _launch(_environment(tmp_path, db, port))
    replacement = None
    try:
        _ready(owner, port)
        store.update_job(job["target_id"], status="running", detail="Downloading")
        owner.kill()
        owner.communicate(timeout=10)
        port = _port()
        replacement = _launch(_environment(tmp_path, db, port))
        _ready(replacement, port)
        recovered = _api(port, "/jobs")["jobs"][0]
        assert recovered["status"] == "failed"
        assert recovered["detail"] == "service restarted during download"
    finally:
        _stop(owner)
        if replacement is not None:
            if replacement.poll() is None:
                _api(port, "/shutdown", {})
            _stop(replacement)


def test_independent_databases_can_run_concurrently(tmp_path):
    first_port = _port()
    first = _launch(_environment(tmp_path, tmp_path / "first.db", first_port))
    second = None
    try:
        _ready(first, first_port)
        second_port = _port()
        second = _launch(_environment(tmp_path, tmp_path / "second.db", second_port))
        _ready(second, second_port)
        assert _api(first_port, "/jobs")["jobs"] == []
        assert _api(second_port, "/jobs")["jobs"] == []
    finally:
        if second is not None:
            if second.poll() is None:
                _api(second_port, "/shutdown", {})
            _stop(second)
        if first.poll() is None:
            _api(first_port, "/shutdown", {})
        _stop(first)


def test_smoke_thread_start_failure_stops_worker_before_releasing_ownership(monkeypatch, tmp_path):
    import config
    from downloads import download_service

    monkeypatch.setattr(config.settings, "download_db_path", tmp_path / "jobs.db")
    state = download_service.DownloadServiceState()
    monkeypatch.setattr(download_service, "STATE", state)
    start = threading.Thread.start

    def fail_server_start(thread):
        if thread.name == "download-service-smoke-server":
            raise RuntimeError("injected thread startup failure")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_server_start)
    try:
        with pytest.raises(RuntimeError, match="injected thread startup failure"):
            download_service.run_smoke_check()
        assert not state.worker_thread.is_alive()
    finally:
        state.stop_event.set()
        if state.worker_thread is not None:
            state.worker_thread.join(timeout=10)


def test_development_smoke_is_isolated_from_a_running_download_service(tmp_path):
    db = tmp_path / "jobs.db"
    store = DownloadStore(db)
    job, _ = store.upsert_job({"source": "Ollama", "name": "startup-test"})
    store.update_job(job["target_id"], status="completed")
    port = _port()
    env = _environment(tmp_path, db, port)
    owner = _launch(env)
    try:
        _ready(owner, port)
        expected = store.update_job(job["target_id"], status="running",
                                    detail="Downloading", progress="37%")
        check = subprocess.run([sys.executable, str(ROOT / "scripts/dev.py"), "smoke"],
                               cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=60,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert check.returncode == 0, check.stdout + check.stderr
        assert _api(port, "/jobs")["jobs"] == [expected]
    finally:
        if owner.poll() is None:
            _api(port, "/shutdown", {})
        _stop(owner)
