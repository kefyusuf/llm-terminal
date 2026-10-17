"""Owned selected-file deletion and active-service shutdown contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.artifacts import hf_artifact_metadata
from downloads.store import DownloadStore


def queued(store, root):
    info = SimpleNamespace(sha="a" * 40, card_data={},
                           siblings=[SimpleNamespace(rfilename="model.gguf", size=10)])
    model = {"source": "Hugging Face", "id": "owner/repo", "target_file": "model.gguf",
             "resolved_revision": "a" * 40,
             "artifact_metadata": hf_artifact_metadata("owner/repo", "model.gguf", info)}
    return store.upsert_job(model, models_dir=root)[0]


def test_removal_deletes_only_selected_managed_file(tmp_path):
    store = DownloadStore(tmp_path / "jobs.db")
    root = tmp_path / "models"
    job = queued(store, root)
    selected = Path(job["download_plan"]["destination"])
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"0123456789")
    other = selected.parent / "other.gguf"
    other.write_bytes(b"other model")
    shared = root / "shared.gguf"
    shared.write_bytes(b"shared model")
    store.update_job(job["target_id"], status="completed")
    assert store.delete_job(job["target_id"], delete_data=True, models_dir=root) == (True, "deleted")
    assert not selected.exists()
    assert other.read_bytes() == b"other model"
    assert shared.read_bytes() == b"shared model"


@pytest.mark.parametrize("case", ["active", "legacy", "changed_root", "tampered", "hardlink"])
def test_removal_preserves_data_without_safe_terminal_ownership(tmp_path, case):
    import json
    import os

    store = DownloadStore(tmp_path / "jobs.db")
    root = tmp_path / "models"
    job = queued(store, root)
    selected = Path(job["download_plan"]["destination"])
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"0123456789")
    if case != "active":
        store.update_job(job["target_id"], status="completed")
    with store._connect() as conn:
        if case == "legacy":
            conn.execute("UPDATE jobs SET download_plan_json = NULL")
        elif case == "tampered":
            plan = dict(job["download_plan"], destination=str(tmp_path / "unowned.gguf"))
            conn.execute("UPDATE jobs SET download_plan_json = ?", (json.dumps(plan),))
    if case == "hardlink":
        os.link(selected, tmp_path / "shared-link.gguf")
    ok, _ = store.delete_job(job["target_id"], delete_data=True,
                             models_dir=tmp_path / "changed" if case == "changed_root" else root)
    assert not ok
    assert selected.read_bytes() == b"0123456789"
    assert store.get_job_by_target(job["target_id"]) is not None
