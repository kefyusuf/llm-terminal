"""Opt-in real HF SDK cancellation/retry trial in a fresh owned workspace.

Run from the checkout with the project environment and a pinned public sample.
Does not install runtimes, perform inference or remove acquired model files.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path


def validate_sample(metadata, *, max_bytes):
    size = metadata.get("size_bytes")
    if type(size) is not int or not 0 < size <= max_bytes:
        raise ValueError("sample size must be known, positive and within the byte limit")
    revision = metadata.get("resolved_revision") or ""
    digest = metadata.get("sha256") or ""
    if len(revision) != 40 or len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in (revision + digest).lower()
    ):
        raise ValueError("sample identity requires a full commit and SHA-256")


def prepare_workspace(path):
    root = Path(path).absolute()
    if root.resolve() != root or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise ValueError("trial workspace must be empty and contain no links")
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_trial(args):
    # Set isolation before imports that create default cache/service state.
    root = prepare_workspace(args.work_dir)
    os.environ.update({
        "AIMODEL_CACHE_DB_PATH": str(root / "cache.db"),
        "AIMODEL_DOWNLOAD_DB_PATH": str(root / "jobs.db"),
        "AIMODEL_HF_MODELS_DIR": str(root / "models"),
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    })
    import huggingface_hub
    from huggingface_hub import HfApi

    import config
    from core.artifacts import hf_artifact_metadata
    from downloads.download_manager import build_download_command
    from downloads.preflight import verify_hf_artifact
    from downloads.runner import run_hf_download
    from downloads.store import DownloadStore

    config.settings.hf_models_dir = root / "models"
    config.settings.hf_token = None
    model = {"source": "Hugging Face", "id": args.repository,
             "target_file": args.filename, "resolved_revision": args.revision}
    build_download_command(model)
    info = HfApi(token=False).model_info(args.repository, revision=args.revision,
                                      files_metadata=True, timeout=10)
    metadata = hf_artifact_metadata(args.repository, args.filename, info)
    validate_sample(metadata, max_bytes=args.max_bytes)
    if metadata["resolved_revision"] != args.revision:
        raise ValueError("upstream sample revision does not match the request")
    model["artifact_metadata"] = metadata

    class State:
        def __init__(self):
            self.store = DownloadStore(root / "jobs.db")
            self.process = None
            self.started = threading.Event()

        def set_process(self, target, process):
            self.process = process
            self.started.set()

        def clear_process(self, target):
            pass  # Retain the owned handle for bounded cleanup.

    state = State()
    job, _ = state.store.upsert_job(model, models_dir=root / "models")
    target = job["target_id"]
    duplicate, created = state.store.upsert_job(model, models_dir=root / "models")
    if created or duplicate["id"] != job["id"]:
        raise RuntimeError("active duplicate replaced the original job")

    def download(cancel_mode=None):
        state.started.clear()
        errors = []

        def worker():
            try:
                run_hf_download(state, target, state.store.get_command(target))
            except Exception as exc:
                errors.append(type(exc).__name__)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        partial_bytes = 0
        deadline = time.monotonic() + args.timeout
        try:
            if not state.started.wait(min(10, args.timeout)):
                raise RuntimeError("owned download child did not start")
            if cancel_mode == "early":
                state.store.mark_cancel_requested(target)
            elif cancel_mode == "partial":
                directory = Path(job["download_plan"]["model_directory"])
                while thread.is_alive() and time.monotonic() < deadline:
                    partial_bytes = sum(path.stat().st_size for path in directory.rglob("*.incomplete")
                                        if path.is_file())
                    if partial_bytes > 0:
                        state.store.mark_cancel_requested(target)
                        break
                    time.sleep(0.005)
            thread.join(max(0, deadline - time.monotonic()))
            if thread.is_alive() or errors:
                raise RuntimeError("owned download failed or exceeded the deadline")
            return state.store.get_job_by_target(target), partial_bytes
        finally:
            if thread.is_alive():
                state.store.mark_cancel_requested(target)
                thread.join(3)
                if state.process is not None and state.process.poll() is None:
                    state.process.kill()
                    thread.join(3)

    started = time.monotonic()
    cancelled, _ = download("early")
    if cancelled["status"] != "cancelled":
        raise RuntimeError("early cancellation did not produce a cancelled job")
    state.store = DownloadStore(root / "jobs.db")
    preserved = state.store.get_job_by_target(target)["download_plan"] == job["download_plan"]
    state.store.upsert_job(model, models_dir=root / "models")
    partial, partial_bytes = download("partial")
    partial_observed = partial["status"] == "cancelled" and partial_bytes > 0
    state.store = DownloadStore(root / "jobs.db")
    state.store.upsert_job(model, models_dir=root / "models")
    completed, _ = download()
    report = {"schema_version": 1, "platform": platform.system(),
              "ci_source_sha": os.environ.get("GITHUB_SHA"),
              "python_version": platform.python_version(),
              "huggingface_hub_version": huggingface_hub.__version__, "repository": args.repository,
              "filename": args.filename, "revision": args.revision, "size_bytes": metadata["size_bytes"],
              "sha256": metadata["sha256"], "license_status": metadata["license"]["status"],
              "duplicate_preserved": True, "early_cancel_status": cancelled["status"],
              "reopened_plan_preserved": preserved, "partial_cancel_observed": partial_observed,
              "partial_bytes_observed": partial_bytes, "completion_status": completed["status"],
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "verified_bytes_and_sha256": False,
              "limits": ["No inference or usage permission claim", "No forced service restart",
                         "Partial resume belongs to the HF SDK; network byte reuse not measured",
                         "Byte limit caps selected file size, not retries or total wire bytes"]}
    if completed["status"] == "completed":
        verify_hf_artifact(job["download_plan"]["model_directory"], args.filename,
                           metadata["size_bytes"], metadata["sha256"])
        report["verified_bytes_and_sha256"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if preserved and partial_observed and report["verified_bytes_and_sha256"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    parser.add_argument("filename")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.max_bytes <= 0 or not 0 < args.timeout <= 300:
        parser.error("positive byte limit and a timeout of at most 300 seconds required")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return run_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
