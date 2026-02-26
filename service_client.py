import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psutil


SERVICE_HOST = "127.0.0.1"
SERVICE_PORT = 8765
SERVICE_BASE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"
MIN_SERVICE_VERSION = "1.5"


def _request(method, path, payload=None, timeout=2.0):
    url = f"{SERVICE_BASE_URL}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = Request(url=url, data=data, method=method, headers=headers)
    with urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)


def is_service_running():
    try:
        data = _request("GET", "/health", timeout=1.0)
        return bool(data.get("ok"))
    except (URLError, HTTPError, TimeoutError, ValueError):
        return False


def get_service_health():
    return _request("GET", "/health", timeout=1.0)


def is_service_compatible(health):
    version = str(health.get("version", "0"))
    return version >= MIN_SERVICE_VERSION


def _start_service_process():
    script_path = Path(__file__).resolve().with_name("download_service.py")
    if sys.platform.startswith("win"):
        detached = getattr(subprocess, "DETACHED_PROCESS", 0)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=detached | new_group,
        )
    else:
        subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def _wait_for_service(deadline_seconds=6.0):
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        try:
            health = get_service_health()
            if health.get("ok") and is_service_compatible(health):
                return True
        except (URLError, HTTPError, TimeoutError, ValueError):
            pass
        time.sleep(0.2)
    return False


def stop_service():
    stopped_any = False

    try:
        _request("POST", "/shutdown", payload={}, timeout=1.0)
        stopped_any = True
    except (URLError, HTTPError, TimeoutError, ValueError):
        pass

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            joined = " ".join(cmdline).lower()
            if "download_service.py" in joined:
                proc.kill()
                stopped_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if not stopped_any:
        return False

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not is_service_running():
            return True
        time.sleep(0.1)
    return not is_service_running()


def ensure_service_running():
    if is_service_running():
        try:
            health = get_service_health()
            if is_service_compatible(health):
                return True
            stop_service()
        except (URLError, HTTPError, TimeoutError, ValueError):
            stop_service()

    _start_service_process()
    return _wait_for_service(deadline_seconds=6.0)


def list_jobs(limit=50):
    data = _request("GET", f"/jobs?limit={int(limit)}", timeout=2.0)
    return data.get("jobs", [])


def get_active_download_debug():
    return _request("GET", "/debug/active", timeout=2.0)


def create_job(model):
    return _request("POST", "/jobs", payload={"model": model}, timeout=3.0)


def cancel_job(target_id):
    return _request(
        "POST", "/jobs/cancel", payload={"target_id": target_id}, timeout=2.0
    )


def delete_job(target_id, _retry=True):
    try:
        return _request(
            "POST", "/jobs/delete", payload={"target_id": target_id}, timeout=2.0
        )
    except HTTPError as exc:
        if exc.code == 404 and _retry and ensure_service_running():
            return delete_job(target_id, _retry=False)
        raise
