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

import os
import subprocess
import sys
import threading
import time
from typing import Any

from loguru import logger

from core.utils import extract_download_progress


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
        "from huggingface_hub import hf_hub_download; "
        "import sys; "
        "hf_hub_download("
        "repo_id=sys.argv[1], "
        "filename=sys.argv[2], "
        "local_dir=sys.argv[3]"
        ")"
    )


def _cancel_requested(state, target_id: str) -> bool:
    latest = state.store.get_job_by_target(target_id)
    return bool(latest and latest.get("cancel_requested"))


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

    models_dir = config.settings.hf_models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    state.store.update_job(
        target_id,
        status="running",
        detail="Downloading",
        progress="",
    )
    process = subprocess.Popen(
        [sys.executable, "-c", _hf_download_script(), repo_id, target_file, str(models_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_service_popen_kwargs(),
    )
    state.set_process(target_id, process)
    cancel_sent_at = None

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
        if cancelled:
            _finalize_terminal(state, target_id, return_code, cancelled=True)
        elif return_code == 0:
            _finalize_terminal(state, target_id, 0, cancelled=False)
        else:
            failure_detail = "hugging face download failed"
            if process.stderr is not None:
                err_text = process.stderr.read().strip()
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
        state.clear_process(target_id)


def _watch_streamed_cancellation(
    state,
    target_id: str,
    process,
    stop_event: threading.Event,
    ready_event: threading.Event,
    started_at: float,
) -> None:
    """Cancel a streamed subprocess even while its stdout reader is blocked."""
    cancel_sent_at = None
    kill_sent = False
    first_check = True

    try:
        while not stop_event.is_set():
            if _cancel_requested(state, target_id):
                if cancel_sent_at is None:
                    cancel_sent_at = started_at if first_check else time.monotonic()
                    if _can_terminate_process(process):
                        try:
                            process.terminate()
                        except OSError:
                            pass
                elif not kill_sent and (time.monotonic() - cancel_sent_at) > 1.5:
                    poll_fn = getattr(process, "poll", None)
                    running = True
                    if callable(poll_fn):
                        try:
                            running = poll_fn() is None
                        except OSError:
                            running = True
                    if running:
                        kill_fn = getattr(process, "kill", None)
                        if callable(kill_fn):
                            try:
                                kill_fn()
                            except OSError:
                                pass
                    kill_sent = True

            first_check = False
            ready_event.set()

            poll_fn = getattr(process, "poll", None)
            if callable(poll_fn):
                try:
                    if poll_fn() is not None:
                        return
                except OSError:
                    return

            stop_event.wait(0.1)
    finally:
        ready_event.set()


def run_streamed_command(state, target_id: str, command) -> None:
    """Run a generic subprocess that streams progress on stdout.

    Used for ``ollama pull`` and any non-HF command. Progress is
    parsed from stdout lines via :func:`extract_download_progress`;
    cancellation is monitored independently so a silent stdout pipe
    cannot prevent terminate/kill escalation.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **_service_popen_kwargs(),
    )
    state.set_process(target_id, process)
    start = time.monotonic()
    last_update = start
    last_line = ""
    cancel_stop = threading.Event()
    cancel_ready = threading.Event()
    cancel_watcher = threading.Thread(
        target=_watch_streamed_cancellation,
        args=(state, target_id, process, cancel_stop, cancel_ready, start),
        daemon=True,
        name=f"download-cancel-{target_id}",
    )
    cancel_watcher.start()
    cancel_ready.wait()

    try:
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if line:
                    last_line = line

                progress = _extract_progress(line)
                now = time.monotonic()
                if progress is not None:
                    state.store.update_job(
                        target_id,
                        status="running",
                        detail="Downloading",
                        progress=progress,
                    )
                    last_update = now
                    continue

                if now - last_update >= 1.0:
                    elapsed = int(now - start)
                    state.store.update_job(
                        target_id,
                        status="running",
                        detail="Downloading",
                        progress=f"{elapsed}s",
                    )
                    last_update = now

        return_code = process.wait()
        cancelled = _cancel_requested(state, target_id)
        _finalize_terminal(state, target_id, return_code, cancelled, last_line=last_line)
    finally:
        cancel_stop.set()
        cancel_watcher.join(timeout=0.5)
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
