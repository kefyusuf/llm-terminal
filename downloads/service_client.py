import ipaddress
import json
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import config

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in lightweight envs
    psutil = None

MIN_SERVICE_VERSION = "2.0"
_NO_PROXY_OPENER = build_opener(ProxyHandler({}))
_owned_service_process = None


def _is_loopback_host(host: str) -> bool:
    """Return whether *host* is an explicit loopback address or localhost."""
    normalized = str(host).strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _url_host(host: str) -> str:
    """Return a URL-safe host, bracketing IPv6 literals when required."""
    normalized = str(host).strip()
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return normalized
    if address.version == 6:
        return f"[{normalized}]"
    return normalized


def service_base_url():
    """Return the configured loopback-only download-service base URL."""
    host = config.settings.download_service_host
    port = config.settings.download_service_port
    if not _is_loopback_host(host):
        raise RuntimeError(
            "Non-loopback download-service clients are disabled until authenticated TLS transport is supported"
        )
    return f"http://{_url_host(host)}:{port}"


def _parse_version(version_str):
    """Parse a dotted version string into a tuple of ints for correct ordering.

    Avoids the lexicographic trap where ``"1.10" < "1.6"``.
    Returns ``(0,)`` if the string cannot be parsed.
    """
    try:
        return tuple(int(x) for x in str(version_str).split("."))
    except (ValueError, AttributeError):
        return (0,)


def _request(method, path, payload=None, timeout=2.0):
    """Send an HTTP request directly to the loopback download service.

    Adds the configured bearer token only to the initial request. Non-loopback
    plaintext targets are rejected before request construction, environment proxy
    settings are bypassed, and redirects cannot forward the bearer token.
    """
    url = f"{service_base_url()}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = Request(url=url, data=data, method=method, headers=headers)
    token = config.settings.download_service_token
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    with _NO_PROXY_OPENER.open(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)


def is_service_running():
    """Return ``True`` if the download service is reachable and healthy."""
    try:
        data = _request("GET", "/health", timeout=1.0)
        return bool(data.get("ok"))
    except (URLError, HTTPError, TimeoutError, ValueError, RuntimeError):
        return False


def get_service_health(timeout=1.0):
    """Return the ``/health`` response dict from the download service."""
    return _request("GET", "/health", timeout=timeout)


def is_service_compatible(health):
    """Return True if the running service version meets the minimum requirement.

    Uses tuple comparison so ``1.10`` sorts correctly after ``1.9``.
    """
    version = str(health.get("version", "0"))
    return _parse_version(version) >= _parse_version(MIN_SERVICE_VERSION)


def _start_service_process():
    """Launch the download service module as a detached background process."""
    global _owned_service_process
    if sys.platform.startswith("win"):
        pythonw = sys.executable.replace("python.exe", "pythonw.exe")
        _owned_service_process = subprocess.Popen(
            [pythonw, "-m", "downloads.download_service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        _owned_service_process = subprocess.Popen(
            [sys.executable, "-m", "downloads.download_service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def _wait_for_service(deadline_seconds=6.0):
    """Poll until the service is healthy and compatible, or the deadline expires.

    Returns ``True`` on success, ``False`` on timeout.
    """
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        try:
            health = get_service_health()
            if health.get("ok") and is_service_compatible(health):
                return True
        except (URLError, HTTPError, TimeoutError, ValueError, RuntimeError):
            pass
        time.sleep(0.2)
    return False


def stop_service():
    """Stop the running download service and wait for it to exit.

    First tries a graceful ``/shutdown`` request. Forced termination is limited
    to the process handle launched by this client; other processes are never scanned.

    Returns ``True`` if the service stopped within 3 seconds.
    """
    global _owned_service_process
    stopped_any = False

    try:
        _request("POST", "/shutdown", payload={}, timeout=1.0)
        stopped_any = True
    except (URLError, HTTPError, TimeoutError, ValueError, RuntimeError):
        pass

    if not stopped_any and _owned_service_process is not None:
        try:
            if _owned_service_process.poll() is None:
                _owned_service_process.kill()
                stopped_any = True
        except OSError:
            stopped_any = False

    if not stopped_any:
        return False

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not is_service_running():
            _owned_service_process = None
            return True
        time.sleep(0.1)
    return not is_service_running()


def ensure_service_running():
    """Ensure a compatible download service is running, (re)starting it if necessary.

    Returns ``True`` when the service is ready to accept requests.
    """
    if is_service_running():
        try:
            health = get_service_health()
            if is_service_compatible(health):
                return True
            jobs = list_jobs(limit=1000, timeout=1.0)
            if len(jobs) >= 1000 or any(job.get("status") in {"queued", "running"} for job in jobs):
                return False
            if not stop_service():
                return False
        except (URLError, HTTPError, TimeoutError, ValueError, RuntimeError):
            return False

    _start_service_process()
    return _wait_for_service(deadline_seconds=6.0)


def list_jobs(limit=50, timeout=2.0):
    """Return a list of download job dicts from the service (most recent first)."""
    data = _request("GET", f"/jobs?limit={int(limit)}", timeout=timeout)
    return data.get("jobs", [])


def get_active_download_debug(timeout=2.0):
    """Return the ``/debug/active`` diagnostic dict from the service."""
    return _request("GET", "/debug/active", timeout=timeout)


def create_job(model):
    """Create (or upsert) a download job for *model* in the service."""
    return _request("POST", "/jobs", payload={"model": model}, timeout=3.0)


def preview_job(model):
    """Read the server destination/disk plan without queueing."""
    return _request("POST", "/jobs/plan", payload={"model": model}, timeout=3.0)["plan"]


def cancel_job(target_id):
    """Request cancellation of the running or queued job identified by *target_id*."""
    return _request("POST", "/jobs/cancel", payload={"target_id": target_id}, timeout=2.0)


def delete_job(target_id, _retry=True):
    """Delete the job record for *target_id* from the service.

    Automatically restarts an incompatible service and retries once if the
    initial request returns 404.
    """
    try:
        return _request("POST", "/jobs/delete", payload={"target_id": target_id}, timeout=2.0)
    except HTTPError as exc:
        if exc.code == 404 and _retry and ensure_service_running():
            return delete_job(target_id, _retry=False)
        raise
