"""Cancellable provider transfers run inside the owning application process."""

from __future__ import annotations

import json
import re
import shutil
import time
from contextlib import contextmanager

from filelock import Timeout
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import WeakFileLock, build_hf_headers

import config
from core.http_client import create_session
from downloads.download_manager import prepare_hf_destination
from downloads.hf_paths import hf_download_paths
from downloads.preflight import (
    DISK_SAFETY_MARGIN,
    check_managed_directory,
    plan_hf_download,
    verify_hf_artifact,
)


class DownloadCancelled(Exception):
    """The owner requested cancellation of an active transfer."""


def _check_cancel(cancel_event):
    if cancel_event.is_set():
        raise DownloadCancelled("Download canceled")


@contextmanager
def _download_lock(path, cancel_event):
    while True:
        _check_cancel(cancel_event)
        lock = WeakFileLock(path, timeout=0.1)
        try:
            lock.__enter__()
        except Timeout:
            continue
        break
    try:
        _check_cancel(cancel_event)
        yield
    finally:
        lock.__exit__(None, None, None)


def _transfer_detail(label, completed, total, start, initial=0):
    elapsed = max(time.monotonic() - start, 0.001)
    rate = max(completed - initial, 0) / elapsed
    detail = f"{label} | {completed / 1024**2:.1f} MiB"
    progress = ""
    if total:
        detail += f" / {total / 1024**2:.1f} MiB"
        progress = f"{min(100, int(completed * 100 / total))}%"
    if rate:
        detail += f" | {rate / 1024**2:.1f} MiB/s"
        if total:
            detail += f" | ETA {max(0, int((total - completed) / rate))}s"
    return detail, progress


def _chunks(response, cancel_event):
    """Return available transport bytes, checking cancellation between reads."""
    while True:
        _check_cancel(cancel_event)
        chunk = response.raw.read1(64 * 1024, decode_content=True)
        _check_cancel(cancel_event)
        if not chunk:
            return
        yield chunk


def _records(response, cancel_event):
    pending = b""
    for chunk in _chunks(response, cancel_event):
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if len(line) > 65536:
                raise ValueError("Ollama progress record exceeds size limit")
            if line.strip():
                yield line
        if len(pending) > 65536:
            raise ValueError("Ollama progress record exceeds size limit")
    if pending.strip():
        yield pending


def _download_hf(job, command, cancel_event, report):
    if len(command) not in {3, 4}:
        raise ValueError("invalid Hugging Face download command")
    _check_cancel(cancel_event)
    if job.get("download_plan_status") == "invalid":
        raise ValueError("persisted download plan is invalid")
    metadata = job.get("artifact_metadata") or {}
    persisted = job.get("download_plan") or {}
    model = {
        "source": "Hugging Face",
        "id": command[1],
        "target_file": command[2],
        "resolved_revision": command[3] if len(command) > 3 else None,
        "artifact_metadata": metadata or None,
    }
    current = plan_hf_download(
        model,
        job.get("models_dir", config.settings.hf_models_dir),
        reserved_bytes=job.get("reserved_bytes", 0),
        allow_unknown_size=persisted.get("size_bytes") is None and persisted.get("allowed") is True,
    )
    if not current["allowed"] or (
        persisted and current["model_directory"] != persisted.get("model_directory")
    ):
        raise ValueError("download plan changed or is blocked")
    directory = prepare_hf_destination(current["model_directory"], command[2])
    token = config.settings.hf_token
    url = hf_hub_url(command[1], command[2], revision=model["resolved_revision"])
    remote = get_hf_file_metadata(url, token=token, timeout=10)
    _check_cancel(cancel_event)
    if not re.fullmatch(r"[a-fA-F0-9]{40}", remote.commit_hash or ""):
        raise ValueError("Hugging Face did not resolve an immutable revision")
    if (
        model["resolved_revision"]
        and remote.commit_hash.lower() != model["resolved_revision"].lower()
    ):
        raise ValueError("Hugging Face revision does not match download plan")
    if not re.fullmatch(r"(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})", remote.etag or ""):
        raise ValueError("Hugging Face did not provide a safe artifact identity")
    size = metadata.get("size_bytes")
    if size is not None and remote.size != size:
        raise ValueError("Hugging Face artifact byte count changed")
    size = remote.size
    if type(size) is not int or size < 0:
        raise ValueError("Hugging Face did not provide an artifact byte count")
    if shutil.disk_usage(directory).free < size + job.get("reserved_bytes", 0) + DISK_SAFETY_MARGIN:
        raise ValueError("insufficient_disk")
    sha256 = metadata.get("sha256") or (remote.etag if len(remote.etag) == 64 else None)
    # Keep the Hub SDK's etag-scoped partial paths, allowing old SDK downloads
    # and interrupted in-process transfers to resume the same exact artifact.
    paths = hf_download_paths(directory, command[2])
    with _download_lock(paths.lock_path, cancel_event):
        _download_hf_file(
            paths, directory, command, remote, size, sha256, token, cancel_event, report
        )


def _download_hf_file(paths, directory, command, remote, size, sha256, token, cancel_event, report):
    partial = paths.incomplete_path(remote.etag)
    check_managed_directory(directory)
    identity_matches = False
    if paths.file_path.exists() and paths.metadata_path.exists():
        try:
            commit, etag, observed_at = paths.metadata_path.read_text().splitlines()
            identity_matches = (
                commit == remote.commit_hash
                and etag == remote.etag
                and paths.file_path.stat().st_mtime - 1 <= float(observed_at)
            )
        except (ValueError, OSError):
            pass
    if paths.file_path.exists() and (sha256 or identity_matches):
        try:
            verify_hf_artifact(
                directory,
                command[2],
                size,
                sha256,
                cancel_check=lambda: _check_cancel(cancel_event),
            )
        except ValueError:
            pass
        else:
            report("Already downloaded", "100%")
            return
    initial = partial.stat().st_size if partial.exists() else 0
    if size is not None and initial > size:
        raise ValueError("partial download exceeds artifact byte count")
    headers = build_hf_headers(token=token)
    headers["Accept-Encoding"] = "identity"
    if initial:
        headers["Range"] = f"bytes={initial}-"
    url = hf_hub_url(command[1], command[2], revision=remote.commit_hash)
    start = time.monotonic()
    completed = initial
    if initial != size:
        with (
            create_session(retry=False) as session,
            session.get(url, headers=headers, stream=True, timeout=(5, 10)) as response,
        ):
            response.raise_for_status()
            if response.status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if (
                    not match
                    or int(match[1]) != initial
                    or int(match[3]) != size
                    or int(match[2]) != size - 1
                ):
                    raise ValueError("invalid Hugging Face resume range")
            elif response.status_code == 200:
                # A server may ignore Range. Restart rather than append and
                # silently corrupt a previously interrupted transfer.
                completed = initial = 0
            else:
                raise ValueError("unexpected Hugging Face download response")
            with partial.open("ab" if initial else "wb") as handle:
                for chunk in _chunks(response, cancel_event):
                    _check_cancel(cancel_event)
                    if chunk:
                        handle.write(chunk)
                        completed += len(chunk)
                        if size is not None and completed > size:
                            raise ValueError("download exceeds artifact byte count")
                        report(*_transfer_detail("Downloading", completed, size, start, initial))
    _check_cancel(cancel_event)
    if size is not None and completed != size:
        raise ValueError("downloaded artifact byte count does not match metadata")
    check_managed_directory(directory)
    if not partial.exists():
        partial.touch()
    partial.replace(paths.file_path)
    verify_hf_artifact(
        directory, command[2], size, sha256, cancel_check=lambda: _check_cancel(cancel_event)
    )
    _check_cancel(cancel_event)
    # The SDK helper acquires this same file lock again; write its three-line
    # format directly while already holding the cancellable transfer lock.
    paths.metadata_path.write_text(f"{remote.commit_hash}\n{remote.etag}\n{time.time()}\n")


def _download_ollama(command, cancel_event, report):
    if len(command) != 3 or command[1] != "pull" or not command[2]:
        raise ValueError("invalid Ollama download command")
    url = config.settings.ollama_api_base.rstrip("/") + "/api/pull"
    layers = {}
    _check_cancel(cancel_event)
    with (
        create_session(retry=False) as session,
        session.post(
            url, json={"name": command[2], "stream": True}, stream=True, timeout=(5, 10)
        ) as response,
    ):
        response.raise_for_status()
        for line in _records(response, cancel_event):
            _check_cancel(cancel_event)
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("invalid Ollama progress response")
            if record.get("error"):
                raise RuntimeError(str(record["error"]))
            status = str(record.get("status") or "Downloading")[:180]
            completed, total = record.get("completed"), record.get("total")
            if type(completed) is int and type(total) is int and total > 0:
                key = str(record.get("digest") or status)
                start, initial = layers.setdefault(key, (time.monotonic(), completed))
                report(*_transfer_detail(status, completed, total, start, initial))
            else:
                report(status, "")
            if status == "success":
                _check_cancel(cancel_event)
                return
    _check_cancel(cancel_event)
    raise RuntimeError("Ollama pull ended before success")


def download(job, command, cancel_event, report):
    """Transfer one exact provider target, or raise without declaring completion."""
    if not isinstance(command, list) or not command:
        raise ValueError("missing download command")
    try:
        if command[0] == "hf_api_download":
            _download_hf(job, command, cancel_event, report)
        elif command[0] == "ollama":
            _download_ollama(command, cancel_event, report)
        else:
            raise ValueError("unsupported download command")
    except Exception:
        _check_cancel(cancel_event)
        raise
