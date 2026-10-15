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

import psutil
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


@pytest.mark.parametrize("existing_version,job_status,slow_cleanup", [
    (None, "completed", False), ("1.7", "completed", False),
    ("1.7", "queued", False), ("1.7", "running", False),
    ("1.7", "completed", True),
])
def test_concurrent_clients_start_and_reuse_one_service_without_changing_history(
    tmp_path, existing_version, job_status, slow_cleanup,
):
    db = tmp_path / "jobs.db"
    store = DownloadStore(db)
    job, _ = store.upsert_job({"source": "Ollama", "name": "startup-history"})
    expected = store.update_job(job["target_id"], status="completed", detail="Done")
    port = _port()
    env = _environment(tmp_path, db, port)
    script = (
        "import json,os,sys,time\nfrom pathlib import Path\n"
        "from downloads.service_client import ensure_service_running,list_jobs\n"
        "root=Path(sys.argv[1]); index=sys.argv[2]\n"
        "def wait(name):\n"
        "    deadline=time.monotonic()+30\n"
        "    while True:\n"
        "        if (root/'exit').exists(): raise SystemExit(0)\n"
        "        if (root/name).exists(): return\n"
        "        if time.monotonic()>deadline: raise RuntimeError('client gate timeout')\n"
        "        time.sleep(.02)\n"
        "from downloads import service_client\n"
        "original_request=service_client._request\n"
        "def retire_request(method,path,*args,**kwargs):\n"
        "    result=original_request(method,path,*args,**kwargs)\n"
        "    if path=='/shutdown-if-idle': (root/'legacy-stop-returned').touch()\n"
        "    return result\n"
        "service_client._request=retire_request\n"
        "original_popen=service_client.subprocess.Popen\n"
        "def launch(*args,**kwargs):\n"
        "    (root/(index+'.spawn')).touch()\n"
        "    return original_popen(*args,**kwargs)\n"
        "service_client.subprocess.Popen=launch\n"
        "ready=root/(index+'.ready')\n"
        "pending=ready.with_suffix('.tmp')\n"
        "pending.write_text(str(os.getpid()))\n"
        "pending.replace(ready)\n"
        "for phase in ('start','reuse'):\n"
        "    wait(phase)\n"
        "    ok=ensure_service_running()\n"
        "    jobs=list_jobs() if ok else []\n"
        "    result=root/(index+'.'+phase+'.json')\n"
        "    pending=result.with_suffix('.tmp')\n"
        "    pending.write_text(json.dumps({'ok':ok,'jobs':jobs}))\n"
        "    pending.replace(result)\n"
        "wait('exit')\n"
    )
    clients = []
    children = {}
    client_workers = {}
    previous = None
    previous_children = []

    def descendants():
        for index in range(len(clients)):
            marker = tmp_path / f"{index}.ready"
            pid = marker.read_text().strip() if marker.exists() else ""
            if not pid:
                continue
            try:
                worker = psutil.Process(int(pid))
                client_workers[int(pid)] = worker
                owned = worker.children(recursive=True)
            except psutil.NoSuchProcess:
                continue
            for child in owned:
                children[child.pid] = child

    def wait_for(predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for client in clients:
                if client.poll() is not None:
                    output, _ = client.communicate(timeout=5)
                    raise AssertionError("Startup client exited early: " + output.decode(errors="replace"))
            descendants()
            if predicate():
                return
            time.sleep(.05)
        raise AssertionError("Concurrent startup deadline exceeded")

    def living_children():
        # Only inspect descendants of the clients created by this test.
        alive = set()
        for pid, child in children.items():
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    alive.add(pid)
            except psutil.NoSuchProcess:
                pass
        return alive

    def service_roots():
        # Count service process trees, allowing Windows venv forwarding parents.
        alive = living_children()
        roots = set()
        for pid in alive:
            try:
                if children[pid].ppid() not in alive:
                    roots.add(pid)
            except psutil.NoSuchProcess:
                pass
        return roots

    try:
        if existing_version:
            bootstrap = (
                "from downloads import download_service\n"
                f"download_service.SERVICE_VERSION={existing_version!r}\n"
                # Advertise legacy job states without invoking a provider.
                "download_service.worker_loop=download_service.STATE.stop_event.wait\n"
                "from downloads.api import Handler\n"
                "import threading,time\nfrom pathlib import Path\n"
                f"retirements=Path({str(tmp_path / 'legacy-retirements')!r})\n"
                "original_post=Handler.do_POST\npost_lock=threading.Lock()\n"
                "def post(handler):\n"
                "    if handler.path in {'/shutdown','/shutdown-if-idle'}:\n"
                "        with post_lock,retirements.open('a') as handle: handle.write('shutdown\\n')\n"
                # Widen a real request overlap; delegate shutdown unchanged.
                "        time.sleep(.2)\n"
                "    original_post(handler)\n"
                "Handler.do_POST=post\n"
            )
            if slow_cleanup:
                bootstrap += (
                    f"root=Path({str(tmp_path)!r})\n"
                    "def worker():\n"
                    "    download_service.STATE.stop_event.wait()\n"
                    "    (root/'legacy-cleanup-paused').touch()\n"
                    "    deadline=time.monotonic()+30\n"
                    "    while not (root/'legacy-cleanup-release').exists():\n"
                    "        if time.monotonic()>deadline: raise RuntimeError('cleanup gate timeout')\n"
                    "        time.sleep(.02)\n"
                    "download_service.worker_loop=worker\n"
                )
            bootstrap += "download_service.main()\n"
            previous = subprocess.Popen(
                [sys.executable, "-c", bootstrap], cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _ready(previous, port)
            previous_children = psutil.Process(previous.pid).children(recursive=True)
            expected = store.update_job(job["target_id"], status=job_status, detail="Preserved")
        for index in range(4):
            clients.append(subprocess.Popen(
                [sys.executable, "-c", script, str(tmp_path), str(index)],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        wait_for(lambda: all((tmp_path / f"{index}.ready").exists() for index in range(4)))
        (tmp_path / "start").touch()
        if slow_cleanup:
            wait_for(lambda: (tmp_path / "legacy-cleanup-paused").exists())
            wait_for(lambda: (tmp_path / "legacy-stop-returned").exists())
            # Observe the interval after HTTP stop returned while old worker
            # ownership is still held; no replacement launch is permitted.
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                assert previous.poll() is None
                assert not list(tmp_path.glob("*.spawn")), "Replacement started before legacy ownership release"
                time.sleep(.02)
            (tmp_path / "legacy-cleanup-release").touch()
        wait_for(lambda: all((tmp_path / f"{index}.start.json").exists() for index in range(4)))
        allowed = job_status == "completed"
        for index in range(4):
            assert json.loads((tmp_path / f"{index}.start.json").read_text()) == {
                "ok": allowed, "jobs": [expected] if allowed else []}
        if not allowed:
            assert previous.poll() is None
            assert _api(port, "/health")["version"] == "1.7"
            assert _api(port, "/jobs")["jobs"] == [expected]
            assert not (tmp_path / "legacy-retirements").exists()
            assert not living_children()
            return
        wait_for(lambda: len(service_roots()) == 1)
        owner = service_roots()
        assert _api(port, "/debug/active")["worker_alive"]
        if previous is not None:
            previous.communicate(timeout=10)
            assert previous.returncode == 0
            assert (tmp_path / "legacy-retirements").read_text().splitlines() == ["shutdown"]

        (tmp_path / "reuse").touch()
        wait_for(lambda: all((tmp_path / f"{index}.reuse.json").exists() for index in range(4)))
        for index in range(4):
            assert json.loads((tmp_path / f"{index}.reuse.json").read_text()) == {
                "ok": True, "jobs": [expected]}
        assert service_roots() == owner
        _api(port, "/shutdown", {})
        wait_for(lambda: not living_children())
        (tmp_path / "exit").touch()
        for client in clients:
            output, _ = client.communicate(timeout=10)
            assert client.returncode == 0, output.decode(errors="replace")
    finally:
        (tmp_path / "legacy-cleanup-release").touch()
        # Capture owned descendants before their parents can exit.
        for client in clients:
            if client.poll() is None:
                descendants()
                break
        try:
            if living_children() or (previous is not None and previous.poll() is None):
                _api(port, "/shutdown", {})
        except OSError:
            pass
        (tmp_path / "exit").touch()
        try:
            for client in clients:
                _stop(client)
        finally:
            for child in [*children.values(), *client_workers.values(), *previous_children]:
                if child.is_running():
                    try:
                        child.kill()
                        child.wait(timeout=5)
                    except psutil.NoSuchProcess:
                        pass
            if previous is not None:
                _stop(previous)
