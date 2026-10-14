"""Concurrent HTTP submissions retain one active job and external invocation."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]


def _api(port, path, body=None):
    request = Request(f"http://127.0.0.1:{port}{path}",
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={"Authorization": "Bearer duplicate-test-only"})
    with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
        return json.load(response)


def _wait(ready):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(.05)
    raise AssertionError("Duplicate submission deadline exceeded")


def _start(tmp_path, *, pause_cleanup=False):
    # Substitute only the external provider command with a real, bounded child.
    # Store, HTTP handler, claiming, dispatcher and streamed runner remain real.
    child = (
        "from pathlib import Path\nimport os,sys,time\n"
        "with Path(sys.argv[1]).open('a') as handle:\n"
        "    handle.write(str(os.getpid())+'\\n')\n"
        "print('1%',flush=True)\n"
        "deadline=time.monotonic()+60\n"
        "while not Path(sys.argv[2]).exists():\n"
        "    if time.monotonic()>deadline: raise SystemExit(1)\n"
        "    time.sleep(.05)\n"
    )
    bootstrap = (
        "from pathlib import Path\nimport sys\nfrom downloads import runner\n"
        "original=runner.subprocess.Popen\n"
        f"root=Path({str(tmp_path)!r})\nchild={child!r}\n"
        "def start(command,**kwargs):\n"
        "    assert command[:2]==['ollama','pull']\n"
        "    name=command[2].lower()\n"
        "    assert name in {'duplicate-blocker-0','duplicate-blocker-1','duplicate-target'}\n"
        "    return original([sys.executable,'-c',child,str(root/(name+'.started')),"
        "str(root/(name+'.release'))],**kwargs)\n"
        "runner.subprocess.Popen=start\n"
    )
    if pause_cleanup:
        # Hold the first real cleanup after its terminal status becomes visible.
        bootstrap += (
            "import time\nfrom downloads.download_service import DownloadServiceState\n"
            "original_clear=DownloadServiceState.clear_process\npaused=False\n"
            "def clear(state,target):\n"
            "    global paused\n"
            "    if target=='ollama:duplicate-target' and not paused:\n"
            "        paused=True\n        (root/'cleanup-paused').touch()\n"
            "        deadline=time.monotonic()+60\n"
            "        while not (root/'cleanup-release').exists():\n"
            "            if time.monotonic()>deadline: raise RuntimeError('cleanup gate timeout')\n"
            "            time.sleep(.05)\n"
            "    original_clear(state,target)\n"
            "DownloadServiceState.clear_process=clear\n"
        )
    bootstrap += "from downloads.download_service import main\nmain()\n"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith("AIMODEL_")}
    env.update(AIMODEL_DOWNLOAD_DB_PATH=str(tmp_path / "jobs.db"),
               AIMODEL_CACHE_DB_PATH=str(tmp_path / "cache.db"),
               AIMODEL_HF_MODELS_DIR=str(tmp_path / "models"),
               AIMODEL_DOWNLOAD_SERVICE_HOST="127.0.0.1",
               AIMODEL_DOWNLOAD_SERVICE_PORT=str(port),
               AIMODEL_DOWNLOAD_SERVICE_TOKEN="duplicate-test-only",
               AIMODEL_DOWNLOAD_MAX_WORKERS="2")
    process = subprocess.Popen([sys.executable, "-c", bootstrap], cwd=ROOT, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return process, port


def _burst(port):
    barrier = threading.Barrier(8)

    def submit(index):
        barrier.wait(timeout=5)
        name = "DUPLICATE-TARGET" if index % 2 else "duplicate-target"
        return _api(port, "/jobs", {"model": {"source": "Ollama", "name": name}})

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(submit, index) for index in range(8)]
        return [future.result(timeout=10) for future in futures]


def _target(port):
    rows = [row for row in _api(port, "/jobs")["jobs"]
            if row["target_id"] == "ollama:duplicate-target"]
    assert len(rows) == 1
    return rows[0]


def test_concurrent_duplicates_keep_one_queued_and_running_download(tmp_path):
    process, port = _start(tmp_path)
    try:
        def ready():
            assert process.poll() is None, "Owned service exited before readiness"
            try:
                return _api(port, "/health")["ok"]
            except OSError:
                return False

        _wait(ready)
        for index in range(2):
            _api(port, "/jobs", {"model": {"source": "Ollama", "name": f"duplicate-blocker-{index}"}})
        _wait(lambda: all((tmp_path / f"duplicate-blocker-{index}.started").exists()
                          for index in range(2)))
        queued = _burst(port)
        assert sum(result["queued"] for result in queued) == 1
        assert {result["job"]["status"] for result in queued} == {"queued"}
        job_id = queued[0]["job"]["id"]
        assert {result["job"]["id"] for result in queued} == {job_id}
        assert _target(port)["status"] == "queued"
        assert not (tmp_path / "duplicate-target.started").exists()

        (tmp_path / "duplicate-blocker-0.release").touch()
        _wait(lambda: (tmp_path / "duplicate-target.started").exists())
        # Leave a free worker slot: single invocation must come from deduplication,
        # not from a saturated executor hiding a second claim.
        (tmp_path / "duplicate-blocker-1.release").touch()
        _wait(lambda: _api(port, "/debug/active")["active_targets"] == ["ollama:duplicate-target"])
        running = _burst(port)
        assert not any(result["queued"] for result in running)
        assert {result["job"]["status"] for result in running} == {"running"}
        assert {result["job"]["id"] for result in running} == {job_id}
        active = _api(port, "/debug/active")
        assert active["active_targets"] == ["ollama:duplicate-target"]
        assert not active["has_duplicates"] and active["worker_alive"]

        (tmp_path / "duplicate-target.release").touch()
        _wait(lambda: _target(port)["status"] == "completed")
        assert _target(port)["id"] == job_id
        assert len((tmp_path / "duplicate-target.started").read_text().splitlines()) == 1
        _api(port, "/shutdown", {})
        process.communicate(timeout=10)
        assert process.returncode == 0
    finally:
        for name in ("duplicate-blocker-0", "duplicate-blocker-1", "duplicate-target"):
            (tmp_path / f"{name}.release").touch()
        try:
            if process.poll() is None:
                _api(port, "/shutdown", {})
            process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


def test_cancel_retry_waits_for_old_cleanup_without_blocking_other_targets(tmp_path):
    process, port = _start(tmp_path, pause_cleanup=True)
    try:
        def ready():
            assert process.poll() is None, "Owned service exited before readiness"
            try:
                return _api(port, "/health")["ok"]
            except OSError:
                return False

        _wait(ready)
        model = {"source": "Ollama", "name": "duplicate-target"}
        created = _api(port, "/jobs", {"model": model})["job"]
        marker = tmp_path / "duplicate-target.started"
        _wait(marker.exists)
        _api(port, "/jobs/cancel", {"target_id": created["target_id"]})
        _wait(lambda: (tmp_path / "cleanup-paused").exists())
        assert _target(port)["status"] == "cancelled"
        retry = _api(port, "/jobs", {"model": model})
        assert retry["queued"] and retry["job"]["id"] == created["id"]
        _api(port, "/jobs", {"model": {"source": "Ollama", "name": "duplicate-blocker-0"}})
        probe = tmp_path / "duplicate-blocker-0.started"
        _wait(lambda: probe.exists() or len(marker.read_text().splitlines()) > 1)
        assert len(marker.read_text().splitlines()) == 1, "Retry overtook old worker cleanup"
        assert probe.exists(), "Other targets were blocked by the same-target retry"
        assert _target(port)["status"] == "queued"

        (tmp_path / "cleanup-release").touch()
        _wait(lambda: len(marker.read_text().splitlines()) == 2)
        _wait(lambda: "ollama:duplicate-target" in _api(port, "/debug/active")["active_targets"])
        assert _target(port)["status"] == "running"
        assert _target(port)["cancel_requested"] == 0
        (tmp_path / "duplicate-target.release").touch()
        _wait(lambda: _target(port)["status"] == "completed")
        assert _target(port)["id"] == created["id"]
        assert len(marker.read_text().splitlines()) == 2
    finally:
        (tmp_path / "cleanup-release").touch()
        for name in ("duplicate-blocker-0", "duplicate-blocker-1", "duplicate-target"):
            (tmp_path / f"{name}.release").touch()
        try:
            if process.poll() is None:
                _api(port, "/shutdown", {})
            process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
