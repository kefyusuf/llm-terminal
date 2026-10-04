"""Owned selected-file deletion and active-service shutdown contracts."""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.artifacts import hf_artifact_metadata
from downloads.store import DownloadStore


def queued(store, root):
    info = SimpleNamespace(sha="a" * 40, card_data={},
                           siblings=[SimpleNamespace(rfilename="model.gguf", size=10)])
    model = {"source": "Hugging Face", "id": "owner/repo", "target_file": "model.gguf",
             "resolved_revision": "a" * 40,
             "artifact_metadata": hf_artifact_metadata("owner/repo", "model.gguf", info)}
    return store.upsert_job(model, models_dir=root)[0]


def test_removal_deletes_only_selected_managed_file(tmp_path):
    store = DownloadStore(tmp_path / "jobs.db")
    root = tmp_path / "models"
    job = queued(store, root)
    selected = Path(job["download_plan"]["destination"])
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"0123456789")
    other = selected.parent / "other.gguf"
    other.write_bytes(b"other model")
    shared = root / "shared.gguf"
    shared.write_bytes(b"shared model")
    store.update_job(job["target_id"], status="completed")
    assert store.delete_job(job["target_id"], delete_data=True, models_dir=root) == (True, "deleted")
    assert not selected.exists()
    assert other.read_bytes() == b"other model"
    assert shared.read_bytes() == b"shared model"


@pytest.mark.parametrize("case", ["active", "legacy", "changed_root", "tampered", "hardlink"])
def test_removal_preserves_data_without_safe_terminal_ownership(tmp_path, case):
    import json
    import os

    store = DownloadStore(tmp_path / "jobs.db")
    root = tmp_path / "models"
    job = queued(store, root)
    selected = Path(job["download_plan"]["destination"])
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"0123456789")
    if case != "active":
        store.update_job(job["target_id"], status="completed")
    with store._connect() as conn:
        if case == "legacy":
            conn.execute("UPDATE jobs SET download_plan_json = NULL")
        elif case == "tampered":
            plan = dict(job["download_plan"], destination=str(tmp_path / "unowned.gguf"))
            conn.execute("UPDATE jobs SET download_plan_json = ?", (json.dumps(plan),))
    if case == "hardlink":
        os.link(selected, tmp_path / "shared-link.gguf")
    ok, _ = store.delete_job(job["target_id"], delete_data=True,
                             models_dir=tmp_path / "changed" if case == "changed_root" else root)
    assert not ok
    assert selected.read_bytes() == b"0123456789"
    assert store.get_job_by_target(job["target_id"]) is not None


def test_service_shutdown_cancels_active_children_and_preserves_queued_jobs(tmp_path):
    from downloads.download_service import DownloadServiceState

    state = DownloadServiceState.__new__(DownloadServiceState)
    state.store = DownloadStore(tmp_path / "jobs.db")
    active = queued(state.store, tmp_path / "models")
    queued_job, _ = state.store.upsert_job({"source": "Ollama", "name": "test:small"})
    state.running_processes = {active["target_id"]: object()}
    state.running_lock = threading.Lock()
    state.stop_event = threading.Event()
    state.server = None
    state.request_shutdown()
    assert state.stop_event.is_set()
    assert state.store.get_job_by_target(active["target_id"])["cancel_requested"]
    assert not state.store.get_job_by_target(queued_job["target_id"])["cancel_requested"]


def test_shutdown_claim_race_does_not_launch_new_child(tmp_path, monkeypatch):
    import config
    from downloads import runner

    monkeypatch.setattr(config.settings, "hf_models_dir", tmp_path / "models")
    store = DownloadStore(tmp_path / "jobs.db")
    job = queued(store, tmp_path / "models")
    stopped = threading.Event()
    stopped.set()
    state = SimpleNamespace(store=store, stop_event=stopped,
                            set_process=lambda *a: None, clear_process=lambda *a: None)
    launched = []
    process = SimpleNamespace(stderr=None, poll=lambda: 0)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: launched.append(a) or process)
    runner.run_hf_download(state, job["target_id"], store.get_command(job["target_id"]))
    assert not launched
    assert store.get_job_by_target(job["target_id"])["status"] == "cancelled"


def test_legacy_service_cannot_silently_ignore_data_removal(monkeypatch):
    from downloads import service_client

    sent = []
    monkeypatch.setattr(service_client, "ensure_service_running", lambda: False)
    monkeypatch.setattr(service_client, "_request", lambda *a, **kw: sent.append(a))
    with pytest.raises(RuntimeError, match="compatible"):
        service_client.delete_job("hugging face:owner/repo", delete_data=True)
    assert not sent


@pytest.mark.parametrize("worker", ["hf", "streamed"])
def test_claimed_cancel_before_process_registration_remains_active(tmp_path, monkeypatch, worker):
    import json
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    import config
    from downloads.api import _make_handler
    from downloads import runner

    root = tmp_path / "models"
    monkeypatch.setattr(config.settings, "hf_models_dir", root)
    store = DownloadStore(tmp_path / "jobs.db")
    job = queued(store, root)
    command = store.get_command(job["target_id"])
    assert store.claim_next_queued()["status"] == "running"
    selected = Path(job["download_plan"]["destination"])
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"0123456789")
    state = SimpleNamespace(store=store, get_process=lambda *_: None,
                            set_process=lambda *_: None, clear_process=lambda *_: None,
                            stop_event=threading.Event())
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state, auth_token="test-only"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path, data):
        request = Request(f"http://127.0.0.1:{server.server_port}{path}",
                          data=json.dumps(data).encode(),
                          headers={"Authorization": "Bearer test-only"})
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    launched = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: launched.append(a))
    try:
        cancelled = post("/jobs/cancel", {"target_id": job["target_id"]})["job"]
        assert cancelled["status"] == "running" and cancelled["cancel_requested"]
        with pytest.raises(HTTPError) as error:
            post("/jobs/delete", {"target_id": job["target_id"], "delete_data": True})
        assert error.value.code == 409
        assert selected.read_bytes() == b"0123456789"
        if worker == "hf":
            runner.run_hf_download(state, job["target_id"], command)
        else:
            runner.run_streamed_command(state, job["target_id"], ["ollama", "pull", "fixture"])
        assert not launched
        assert store.get_job_by_target(job["target_id"])["status"] == "cancelled"
        assert post("/jobs/delete", {"target_id": job["target_id"], "delete_data": True}) == {"ok": True}
        assert not selected.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_queued_cancel_is_atomic_with_worker_claim(tmp_path):
    store = DownloadStore(tmp_path / "jobs.db")
    job = queued(store, tmp_path / "models")
    cancelled = store.mark_cancel_requested(job["target_id"])
    assert cancelled["status"] == "cancelled"
    assert store.claim_next_queued() is None


def test_missing_job_is_a_worker_cancellation(tmp_path):
    from downloads.runner import _cancel_requested

    assert _cancel_requested(SimpleNamespace(store=DownloadStore(tmp_path / "jobs.db")), "deleted")


def test_authenticated_http_removal_uses_server_owned_root(tmp_path, monkeypatch):
    import json
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen

    import config
    from downloads.api import _make_handler

    root = tmp_path / "models"
    monkeypatch.setattr(config.settings, "hf_models_dir", root)
    store = DownloadStore(tmp_path / "jobs.db")
    job = queued(store, root)
    selected = Path(job["download_plan"]["destination"])
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"0123456789")
    store.update_job(job["target_id"], status="completed")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(SimpleNamespace(store=store),
                                                              auth_token="test-only"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(f"http://127.0.0.1:{server.server_port}/jobs/delete",
                          data=json.dumps({"target_id": job["target_id"], "delete_data": True,
                                           "models_dir": str(tmp_path / "ignored")}).encode(),
                          headers={"Authorization": "Bearer test-only"})
        with urlopen(request, timeout=3) as response:
            assert response.status == 200
        assert not selected.exists()
        assert store.get_job_by_target(job["target_id"]) is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_real_module_service_shutdown_exits_owned_process(tmp_path):
    import os
    import socket
    import subprocess
    import sys
    import time
    from urllib.request import ProxyHandler, Request, build_opener

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.pop("AIMODEL_SMOKE", None)
    env.update(AIMODEL_DOWNLOAD_SERVICE_PORT=str(port), AIMODEL_DOWNLOAD_SERVICE_TOKEN="test-only",
               AIMODEL_DOWNLOAD_DB_PATH=str(tmp_path / "jobs.db"),
               AIMODEL_CACHE_DB_PATH=str(tmp_path / "cache.db"),
               AIMODEL_HF_MODELS_DIR=str(tmp_path / "models"))
    process = subprocess.Popen([sys.executable, "-m", "downloads.download_service"], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    opener = build_opener(ProxyHandler({}))
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with opener.open(base + "/health", timeout=0.5):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise AssertionError("owned service did not start") from None
                time.sleep(0.05)
        request = Request(base + "/shutdown", data=b"{}",
                          headers={"Authorization": "Bearer test-only"})
        with opener.open(request, timeout=2) as response:
            assert response.status == 200
        assert process.wait(timeout=4) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
