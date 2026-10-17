"""Application-owned downloads with bounded workers and durable history."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from downloads.backends import DownloadCancelled, download
from downloads.download_manager import download_target_id
from downloads.store import DownloadStore


class DownloadManager:
    """Serialize queue mutations while transfers run in a bounded thread pool.

    Backends receive a job, persisted provider arguments, a cancellation event,
    and a progress callback. No worker accesses TUI state.
    """

    def __init__(self, db_path=None, *, max_workers=None, models_dir=None, backend=None):
        self.store = DownloadStore(db_path or config.settings.download_db_path)
        self.models_dir = Path(models_dir or config.settings.hf_models_dir)
        self.store.normalize_target_ids()
        self.store.migrate_legacy_hf_commands()
        self.store.recover_interrupted_jobs()
        self._lock = threading.RLock()
        self._closed = False
        self._jobs = {}
        self._running = set()
        self._progress = {}
        self._backend = backend
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or max(1, config.settings.download_max_workers),
            thread_name_prefix="download",
        )

    def create_job(self, model):
        with self._lock:
            if self._closed:
                raise RuntimeError("Download manager is shut down")
            target = download_target_id(model)
            if target in self._jobs:
                return {"job": self.store.get_job_by_target(target), "queued": False}
            job, queued = self.store.upsert_job(model, models_dir=self.models_dir)
            if queued:
                target = job["target_id"]
                cancel = threading.Event()
                self._jobs[target] = cancel
                self._executor.submit(self._run, target, cancel)
            return {"job": job, "queued": queued}

    def list_jobs(self, limit=50):
        with self._lock:
            jobs = self.store.list_jobs(limit=limit)
            for job in jobs:
                job.update(self._progress.get(job["target_id"], {}))
            return jobs

    def preview_job(self, model):
        with self._lock:
            return self.store.plan_job(model, self.models_dir)

    def cancel_job(self, target_id):
        with self._lock:
            event = self._jobs.get(target_id)
            if event is not None:
                event.set()
                self.store.mark_cancel_requested(target_id)
                if target_id not in self._running:
                    self._jobs.pop(target_id, None)
            job = self.store.get_job_by_target(target_id)
            if job is None:
                raise ValueError("Download entry not found")
            return {"job": job}

    def delete_job(self, target_id, *, delete_data=False):
        with self._lock:
            if target_id in self._jobs:
                raise ValueError("Download is active; cancel it before deleting")
            deleted, reason = self.store.delete_job(
                target_id,
                delete_data=delete_data,
                models_dir=self.models_dir,
            )
            if not deleted:
                raise ValueError(f"Cannot delete download: {reason}")
            return {"ok": True}

    def get_active_download_debug(self):
        with self._lock:
            targets = sorted(self._running)
            return {
                "active_targets": targets,
                "count": len(targets),
                "has_duplicates": False,
                "worker_alive": not self._closed,
            }

    def _run(self, target, cancel):
        with self._lock:
            if self._jobs.get(target) is not cancel:
                return
            if self._closed or cancel.is_set():
                self._finish(target, "interrupted" if self._closed else "cancelled")
                return
            self._running.add(target)
            job = self.store.update_job(target, status="running", detail="Starting")
            job["models_dir"] = self.models_dir
            job["reserved_bytes"] = self.store.reserved_download_bytes(exclude_target=target)

        last_persisted = 0.0

        def report(detail, progress=""):
            nonlocal last_persisted
            with self._lock:
                if not cancel.is_set():
                    self._progress[target] = {"detail": detail, "progress": progress}
                    now = time.monotonic()
                    if now - last_persisted >= 0.25:
                        self.store.update_job(target, detail=detail, progress=progress)
                        last_persisted = now

        status, detail = "completed", "Download complete"
        try:
            (self._backend or download)(job, self.store.get_command(target), cancel, report)
        except DownloadCancelled:
            status, detail = "cancelled", "Canceled"
        except Exception as exc:
            status, detail = "failed", str(exc)[:500]
        with self._lock:
            if self._closed:
                status, detail = "interrupted", "App exited; download can be resumed"
            elif cancel.is_set():
                status, detail = "cancelled", "Canceled"
            self._finish(target, status, detail)

    def _finish(self, target, status, detail=None):
        self.store.update_job(
            target,
            status=status,
            detail=detail or status.capitalize(),
            progress="",
            return_code=0 if status == "completed" else 1,
        )
        self._running.discard(target)
        self._progress.pop(target, None)
        self._jobs.pop(target, None)

    def shutdown(self):
        """Stop admission, cancel transfers, and wait until all workers exit."""
        with self._lock:
            self._closed = True
            for event in self._jobs.values():
                event.set()
        self._executor.shutdown(wait=True)


def run_smoke_check():
    """Exercise the local queue lifecycle without contacting a provider."""
    from tempfile import TemporaryDirectory

    started = threading.Event()

    def backend(job, command, cancel, report):
        report("Downloading", "50%")
        started.set()
        if not cancel.wait(5):
            raise RuntimeError("smoke cancellation timed out")

    with TemporaryDirectory(prefix="download-smoke-") as directory:
        manager = DownloadManager(Path(directory) / "jobs.db", backend=backend)
        try:
            manager.create_job({"source": "Ollama", "name": "offline-smoke"})
            if not started.wait(5):
                raise RuntimeError("smoke worker did not start")
            assert manager.list_jobs()[0]["progress"] == "50%"
        finally:
            manager.shutdown()
        assert manager.list_jobs()[0]["status"] == "interrupted"
    print("download-manager-smoke-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_check())
