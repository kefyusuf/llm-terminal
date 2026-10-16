"""Subprocess runners for download jobs.

Extracted from downloads/download_service.py during the architecture
review (candidate #5). Two runners, both consume a ``DownloadStore``
+ the shared ``STATE`` singleton (imported from ``download_service``):

- :func:`run_hf_download`: ``hf_api_download`` sentinel command →
  one exact Hugging Face file via subprocess.
- :func:`run_streamed_command`: every other command (e.g. ``ollama
  pull``) → stdout-streamed subprocess.

Both write status to ``STATE.store.update_job()`` and respect
``STATE.store.mark_cancel_requested()`` by escalating from
``terminate()`` to ``kill()`` after 1.5 seconds.
"""

from __future__ import annotations

import codecs
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any

from loguru import logger

from core.utils import extract_download_progress
from downloads.download_manager import prepare_hf_destination
from downloads.preflight import plan_hf_download
from downloads.progress import ollama_transfer_detail


def _service_popen_kwargs() -> dict[str, Any]:
    import config

    kwargs: dict[str, Any] = {}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env = os.environ.copy()
    hf_token = getattr(config.settings, "hf_token", None)
    if hf_token:
        env["HF_TOKEN"] = hf_token
    kwargs["env"] = env
    return kwargs


def _can_terminate_process(process) -> bool:
    return hasattr(process, "terminate") and callable(getattr(process, "terminate", None))


def _extract_progress(line: str) -> str | None:
    """Wrap :func:`utils.extract_download_progress` returning a ``"N%"`` string."""
    value = extract_download_progress(line)
    return f"{value}%" if value is not None else None


def is_hf_api_command(command) -> bool:
    """Return True when *command* is an internal HF API download payload."""
    return isinstance(command, list) and len(command) > 0 and command[0] == "hf_api_download"


def _repo_id_from_hf_command(command) -> str:
    """Extract repository id from an internal HF API command payload."""
    if not is_hf_api_command(command):
        return ""
    return command[1] if len(command) > 1 else ""


def _target_file_from_hf_command(command) -> str:
    """Extract the exact repository filename from an internal HF API command payload."""
    if not is_hf_api_command(command):
        return ""
    return command[2] if len(command) > 2 else ""


def _hf_download_script() -> str:
    """Return the subprocess script used to download one exact HF file."""
    return (
        "import sys; "
        "from downloads.parent_lifetime import start_parent_watchdog; "
        "start_parent_watchdog() if len(sys.argv) > 7 and sys.argv[7] == 'parent-stdin' else None; "
        "from huggingface_hub import hf_hub_download; "
        "import sys; "
        "hf_hub_download("
        "repo_id=sys.argv[1], "
        "filename=sys.argv[2], "
        "local_dir=sys.argv[3], "
        "revision=(sys.argv[4] or None) if len(sys.argv) > 4 else None"
        "); "
        "from downloads.preflight import verify_hf_artifact; "
        "verify_hf_artifact(sys.argv[3], sys.argv[2], "
        "int(sys.argv[5]) if sys.argv[5] else None, sys.argv[6] or None) "
        "if len(sys.argv) > 6 else None"
    )


def _cancel_requested(state, target_id: str) -> bool:
    stop_event = getattr(state, "stop_event", None)
    if stop_event is not None and stop_event.is_set():
        return True
    latest = state.store.get_job_by_target(target_id)
    return latest is None or bool(latest.get("cancel_requested"))


def _read_stderr_tail(process) -> str:
    """Drain a child pipe while retaining at most 4096 characters."""
    tail = ""
    if process.stderr is None:
        return tail
    try:
        while chunk := process.stderr.read(4096):
            tail = (tail + chunk)[-4096:]
    except (OSError, ValueError):
        pass
    return tail


def _finalize_terminal(state, target_id: str, return_code, cancelled: bool, *, last_line: str = "") -> None:
    """Write the terminal state of a job (completed / failed / cancelled)."""
    if cancelled:
        state.store.update_job(
            target_id,
            status="cancelled",
            detail="Canceled",
            progress="",
            return_code=return_code,
        )
    elif return_code == 0:
        state.store.update_job(
            target_id,
            status="completed",
            detail="Completed",
            progress="",
            return_code=0,
        )
    else:
        failure_detail = "download command exited with non-zero status"
        if last_line:
            failure_detail = last_line[:180]
        state.store.update_job(
            target_id,
            status="failed",
            detail=failure_detail,
            progress="",
            return_code=return_code,
        )


def run_hf_download(state, target_id: str, command) -> None:
    """Download the exact Hugging Face target file for *target_id*.

    The command payload contains the repository id and selected filename.
    A subprocess calls :func:`huggingface_hub.hf_hub_download` so the
    existing terminate/kill cancellation behavior remains intact.
    """
    import config

    repo_id = _repo_id_from_hf_command(command)
    target_file = _target_file_from_hf_command(command)
    if not repo_id:
        state.store.update_job(
            target_id,
            status="failed",
            detail="missing Hugging Face repository id",
            return_code=1,
        )
        return
    if not target_file:
        state.store.update_job(
            target_id,
            status="failed",
            detail="missing Hugging Face target file",
            return_code=1,
        )
        return

    try:
        get_metadata = getattr(state.store, "get_execution_metadata", None)
        execution = get_metadata(target_id) if callable(get_metadata) else {}
        execution = execution if isinstance(execution, dict) else {}
        if execution.get("download_plan_status") == "invalid":
            raise ValueError("persisted download plan is invalid")
        metadata = execution.get("artifact_metadata") or {}
        persisted_plan = execution.get("download_plan")
        directory = config.settings.hf_models_dir
        if persisted_plan:
            selected = {"source": "Hugging Face", "id": repo_id, "target_file": target_file,
                        "resolved_revision": command[3] if len(command) > 3 else None,
                        "artifact_metadata": metadata or None}
            current = plan_hf_download(selected, directory,
                reserved_bytes=state.store.reserved_download_bytes(target_id),
                allow_unknown_size=persisted_plan.get("size_bytes") is None and persisted_plan.get("allowed") is True)
            if not current["allowed"] or current["model_directory"] != persisted_plan.get("model_directory"):
                raise ValueError("download plan changed or is blocked")
            directory = current["model_directory"]
        models_dir = prepare_hf_destination(directory, target_file)
    except (OSError, RuntimeError, ValueError):
        state.store.update_job(
            target_id,
            status="failed",
            detail="Hugging Face target file or destination is unsafe or unavailable",
            progress="",
            return_code=1,
        )
        return

    if _cancel_requested(state, target_id):
        _finalize_terminal(state, target_id, None, cancelled=True)
        return

    state.store.update_job(
        target_id,
        status="running",
        detail="Downloading",
        progress="",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _hf_download_script(),
            repo_id,
            target_file,
            str(models_dir),
            command[3] if len(command) > 3 else "",
            str(metadata["size_bytes"]) if metadata.get("size_bytes") is not None else "",
            metadata.get("sha256") or "",
            "parent-stdin",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_service_popen_kwargs(),
    )
    state.set_process(target_id, process)
    cancel_sent_at = None
    stderr_tail: list[str] = []
    stderr_reader = threading.Thread(
        target=lambda: stderr_tail.append(_read_stderr_tail(process)), daemon=True
    )
    stderr_reader.start()

    try:
        while True:
            if _cancel_requested(state, target_id):
                if cancel_sent_at is None:
                    cancel_sent_at = time.monotonic()
                    try:
                        process.terminate()
                    except OSError:
                        pass
                elif process.poll() is None and (time.monotonic() - cancel_sent_at) > 1.5:
                    try:
                        process.kill()
                    except OSError:
                        pass

            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(0.25)

        cancelled = _cancel_requested(state, target_id)
        stderr_reader.join(timeout=2)
        if cancelled:
            _finalize_terminal(state, target_id, return_code, cancelled=True)
        elif return_code == 0:
            _finalize_terminal(state, target_id, 0, cancelled=False)
        else:
            failure_detail = "hugging face download failed"
            if stderr_tail:
                err_text = stderr_tail[0].strip()
                if err_text:
                    failure_detail = err_text.splitlines()[-1][:180]
            state.store.update_job(
                target_id,
                status="failed",
                detail=failure_detail,
                progress="",
                return_code=return_code,
            )
    except Exception as exc:
        logger.warning("HF download failed for {}: {}", target_id, exc)
        detail = str(exc).strip() or "hugging face download failed"
        state.store.update_job(
            target_id,
            status="failed",
            detail=detail[:180],
            progress="",
            return_code=1,
        )
    finally:
        parent_pipe = getattr(process, "stdin", None)
        if parent_pipe is not None:
            try:
                parent_pipe.close()
            except (OSError, ValueError):
                pass
        stderr_reader.join(timeout=0.5)
        if not stderr_reader.is_alive() and process.stderr is not None:
            process.stderr.close()
        state.clear_process(target_id)


def _read_stream_chunk(stream, timeout: float = 0.1) -> bytes | None:
    """Read ready pipe bytes; None means idle and empty bytes mean EOF.

    This is the pipe's only reader. Do not mix buffered reads with os.read:
    buffered text reads can wait for another line or inherited writer's EOF.
    """
    descriptor = stream.fileno()
    if sys.platform.startswith("win"):
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        peek = kernel.PeekNamedPipe
        peek.argtypes = (
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
        )
        peek.restype = wintypes.BOOL
        available = wintypes.DWORD()
        handle = msvcrt.get_osfhandle(descriptor)
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error in {109, 232, 233}:  # Broken, closing, or disconnected pipe.
                return b""
            raise ctypes.WinError(error)
        if not available.value:
            time.sleep(timeout)
            return None
        return os.read(descriptor, min(4096, available.value))

    import select

    readable, _, _ = select.select([descriptor], [], [], timeout)
    return os.read(descriptor, 4096) if readable else None


def run_streamed_command(state, target_id: str, command) -> None:
    """Run a generic subprocess that streams progress on stdout.

    Used for ``ollama pull`` and any non-HF command. Progress is
    parsed from stdout lines via :func:`extract_download_progress`;
    Readiness polling keeps cancellation independent of line endings and EOF.
    An inherited writer cannot keep the job alive after its owned CLI exits.
    """
    if _cancel_requested(state, target_id):
        _finalize_terminal(state, target_id, None, cancelled=True)
        return
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        **_service_popen_kwargs(),
    )
    state.set_process(target_id, process)
    start = time.monotonic()
    last_update = start
    last_line = ""
    cancel_sent_at = None
    kill_sent = False
    exit_drain_deadline = None
    exit_drain_bytes = 0
    eof = process.stdout is None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    ollama_command = target_id.startswith("ollama:") or (
        isinstance(command, list) and bool(command)
        and os.path.basename(str(command[0])).lower() in {"ollama", "ollama.exe"}
    )
    transfer_snapshot: tuple[str, str] | None = None

    def record_line(line, *, partial=False):
        nonlocal last_line, last_update, transfer_snapshot
        line = line.strip()
        if line:
            last_line = line
        # ANSI redraws may have no line delimiter: use the latest complete
        # percentage, including one split across successive byte reads.
        percentages = re.findall(r"(?<!\d)(?:100|\d{1,2})%", line)
        progress = percentages[-1] if percentages else None
        now = time.monotonic()
        detail = ollama_transfer_detail(line) if ollama_command else None
        if detail is not None:
            snapshot = (detail, progress or "")
            if snapshot != transfer_snapshot:
                state.store.update_job(
                    target_id, status="running", detail=detail, progress=progress or "",
                )
                transfer_snapshot = snapshot
                last_update = now
        elif transfer_snapshot is not None:
            # A generic heartbeat must not discard current provider telemetry.
            return
        elif progress is not None:
            state.store.update_job(
                target_id, status="running", detail="Downloading", progress=progress,
            )
            last_update = now
        elif not partial and now - last_update >= 1.0:
            state.store.update_job(
                target_id, status="running", detail=f"Downloading ({int(now - start)}s)",
            )
            last_update = now

    def consume(text):
        nonlocal pending
        for part in (pending + text).splitlines(keepends=True):
            if part.endswith(("\r", "\n")):
                record_line(part)
                pending = ""
            else:
                # Bound a child that never emits a line ending.
                pending = part[-4096:]
        if pending:
            record_line(pending, partial=True)

    try:
        while True:
            now = time.monotonic()
            return_code = process.poll()
            if return_code is None and _cancel_requested(state, target_id):
                if cancel_sent_at is None:
                    cancel_sent_at = now
                    try:
                        process.terminate()
                    except OSError:
                        pass
                elif not kill_sent and now - cancel_sent_at > 1.5:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    kill_sent = True
            if return_code is not None:
                if exit_drain_deadline is None:
                    exit_drain_deadline = now + 0.2
                # Bound both silent inherited writers and continuously noisy ones.
                if eof or now >= exit_drain_deadline or exit_drain_bytes >= 65536:
                    break
            if eof:
                time.sleep(0.1)
                continue
            chunk = _read_stream_chunk(process.stdout)
            if chunk == b"":
                eof = True
            elif chunk:
                if exit_drain_deadline is not None:
                    exit_drain_bytes += len(chunk)
                consume(decoder.decode(chunk))

        consume(decoder.decode(b"", final=True))
        if pending:
            record_line(pending)
        cancelled = _cancel_requested(state, target_id)
        _finalize_terminal(state, target_id, return_code, cancelled, last_line=last_line)
    finally:
        try:
            # A pipe/decoder/store failure must not abandon our still-running CLI.
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.5)
        finally:
            try:
                if process.stdout is not None:
                    process.stdout.close()
            finally:
                state.clear_process(target_id)


def process_job(state, target_id: str) -> None:
    """Process a single download job (called in thread pool).

    Dispatches to the HF runner or the streamed runner based on the
    command payload stored in the job row.
    """
    try:
        cmd = state.store.get_command(target_id)
        if not cmd:
            state.store.update_job(
                target_id, status="failed", detail="missing command", return_code=1
            )
            return

        if is_hf_api_command(cmd):
            run_hf_download(state, target_id, cmd)
            return

        run_streamed_command(state, target_id, cmd)
    except FileNotFoundError:
        state.store.update_job(
            target_id,
            status="failed",
            detail="required command not found",
            return_code=127,
        )
    except OSError as exc:
        state.store.update_job(target_id, status="failed", detail=str(exc)[:180], return_code=1)
    finally:
        state.clear_process(target_id)
