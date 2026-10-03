"""User-invoked diagnostics with bounded network waits and allowlisted output."""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import shutil
import ssl
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from config import Settings


def _result(status: str, code: str, action: str = "", **details: Any) -> dict[str, Any]:
    return {"status": status, "code": code, "action": action, **details}


def probe_endpoint(url: str, timeout: float, required: bool) -> dict[str, Any]:
    """Probe without tokens, redirects, retries, or exporting server error text."""
    severity = "error" if required else "warning"
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or (required and parsed.scheme != "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            return _result(
                "error",
                "invalid_endpoint",
                "Use an HTTP(S) base URL without credentials or query parameters.",
            )
        _ = parsed.port
    except ValueError:
        return _result("error", "invalid_endpoint", "Check the configured endpoint URL and port.")
    try:
        with requests.Session() as session:
            # A truthy auth callable prevents Requests from loading .netrc credentials.
            session.auth = lambda request: request
            with session.get(
                url, timeout=timeout, verify=True, allow_redirects=False, stream=True
            ) as response:
                if response.status_code != 200:
                    return _result(
                        severity,
                        "http_status",
                        "Check server availability and access requirements.",
                        http_status=response.status_code,
                    )
                details: dict[str, Any] = {}
                if parsed.path.endswith("/api/version"):
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=1024):
                        body.extend(chunk)
                        if len(body) > 4096:
                            break
                    if len(body) <= 4096:
                        try:
                            payload = json.loads(body)
                            version = payload.get("version") if isinstance(payload, dict) else None
                            if isinstance(version, str) and re.fullmatch(
                                r"\d{1,4}(?:\.\d{1,4}){1,3}", version
                            ):
                                details["version"] = version
                        except (ValueError, UnicodeError):
                            pass
                return _result("ok", "reachable", **details)
    except requests.exceptions.SSLError:
        return _result(
            "error",
            "tls",
            "Check the trusted CA chain and REQUESTS_CA_BUNDLE. Keep TLS verification enabled.",
        )
    except requests.exceptions.Timeout:
        return _result(
            severity, "timeout", "Check server reachability or rerun with a larger --timeout."
        )
    except (requests.exceptions.RequestException, OSError, ValueError):
        return _result(
            severity,
            "connection",
            "Start the selected runtime or check endpoint, proxy, and network configuration.",
        )


def _path_check(path: Path, directory: bool) -> dict[str, Any]:
    try:
        exists = path.exists()
        if exists and (path.is_dir() != directory):
            return _result(
                "error",
                "path_type",
                "Configure a directory for models and files for databases.",
                exists=True,
            )
        parent = path if exists and directory else path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        free = shutil.disk_usage(parent).free
        return _result(
            "warning" if free == 0 else "ok",
            "disk_full" if free == 0 else "path_available",
            "Free disk space before downloading models." if free == 0 else "",
            exists=exists,
            free_bytes=free,
        )
    except (OSError, ValueError):
        return _result("error", "path_access", "Check access to the configured storage location.")


def _ca_check() -> dict[str, Any]:
    bundle = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("CURL_CA_BUNDLE")
    if not bundle:
        return _result("ok", "default_ca", configured=False)
    try:
        path = Path(bundle)
        if path.is_dir():
            ssl.create_default_context(capath=str(path))
        else:
            ssl.create_default_context(cafile=str(path))
        return _result("ok", "custom_ca", configured=True)
    except (OSError, ValueError):
        return _result(
            "error",
            "invalid_ca",
            "Set REQUESTS_CA_BUNDLE to a readable trusted PEM bundle or hashed CA directory.",
            configured=True,
        )


def _mlx_check() -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        return _result("skipped", "unsupported_platform")
    try:
        installed = version("mlx")
        details = (
            {"version": installed} if re.fullmatch(r"\d{1,4}(?:\.\d{1,4}){1,3}", installed) else {}
        )
        return _result("ok", "installed", **details)
    except PackageNotFoundError:
        return _result(
            "warning", "not_installed", "Install the optional MLX runtime if you intend to use it."
        )


def collect_diagnostics(
    settings: Settings, *, offline: bool = False, timeout: float = 3.0
) -> dict[str, Any]:
    """Collect local facts and concurrent probes under one network deadline.

    Daemon threads allow the caller to return even when OS DNS resolution blocks.
    The budget covers network collection, not imports or local filesystem calls.
    """
    if not 0 < timeout <= 30:
        raise ValueError("timeout must be greater than zero and at most 30 seconds")
    checks = [
        {"name": "python", **_result("ok", "version", version=platform.python_version())},
        {
            "name": "hf_token",
            **_result(
                "ok",
                "configured" if settings.hf_token else "not_configured",
                configured=bool(settings.hf_token),
            ),
        },
        {"name": "tls_ca", **_ca_check()},
        {"name": "mlx", **_mlx_check()},
    ]
    for name, path, directory in (
        ("models", settings.hf_models_dir, True),
        ("cache", settings.cache_db_path, False),
        ("download_history", settings.download_db_path, False),
    ):
        checks.append({"name": name, **_path_check(path, directory)})
    endpoints = {
        "ollama": (settings.ollama_api_base.rstrip("/") + "/api/version", False),
        "lmstudio": (
            os.getenv("LMSTUDIO_HOST", "http://localhost:1234").rstrip("/") + "/v1/models",
            False,
        ),
        "docker": (
            os.getenv("DOCKER_MODEL_RUNNER_HOST", "http://localhost:12434").rstrip("/") + "/models",
            False,
        ),
        "huggingface": (
            os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/"),
            True,
        ),
    }
    from downloads.service_client import service_base_url

    try:
        endpoints["download_service"] = (service_base_url() + "/health", False)
    except (RuntimeError, ValueError):
        checks.append(
            {
                "name": "download_service",
                **_result(
                    "error",
                    "invalid_endpoint",
                    "Configure a loopback download-service host and valid port.",
                ),
            }
        )
    completed: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()

    def run(name: str, url: str, required: bool) -> None:
        try:
            result = probe_endpoint(url, timeout, required)
        except Exception:
            result = _result(
                "error", "probe_failed", "Rerun diagnostics and check runtime configuration."
            )
        completed.put((name, result))

    results: dict[str, dict[str, Any]] = {}
    deadline = time.monotonic() + timeout
    if not offline:
        for name, (url, required) in endpoints.items():
            threading.Thread(target=run, args=(name, url, required), daemon=True).start()
        while len(results) < len(endpoints):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                name, result = completed.get(timeout=remaining)
                results[name] = result
            except queue.Empty:
                break
    for name, (_, required) in endpoints.items():
        result = (
            _result("skipped", "offline")
            if offline
            else results.get(
                name,
                _result(
                    "error" if required else "warning",
                    "deadline",
                    "Check reachability or rerun with a larger --timeout.",
                ),
            )
        )
        checks.append({"name": name, **result})
    status = (
        "error"
        if any(item["status"] == "error" for item in checks)
        else "warning"
        if any(item["status"] == "warning" for item in checks)
        else "ok"
    )
    return {
        "schema_version": 1,
        "status": status,
        "offline": offline,
        "network_budget_seconds": timeout,
        "checks": checks,
    }
