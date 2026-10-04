"""Resolved HF revisions survive enrichment, persistence and execution."""

import sys
from types import SimpleNamespace

import pytest

from downloads.download_manager import build_download_command
from downloads.runner import _hf_download_script
from downloads.store import DownloadStore
from providers import hf_provider

REVISION = "a" * 40


def model():
    return {
        "source": "Hugging Face",
        "id": "owner/repo",
        "name": "repo",
        "target_file": "model.gguf",
        "resolved_revision": REVISION,
    }


def test_command_pins_resolved_commit():
    assert build_download_command(model()) == [
        "hf_api_download",
        "owner/repo",
        "model.gguf",
        REVISION,
    ]


@pytest.mark.parametrize("revision", ["main", "v1.0", "abc", 123])
def test_command_rejects_non_commit_revision(revision):
    selected = model()
    selected["resolved_revision"] = revision
    with pytest.raises(ValueError, match="revision"):
        build_download_command(selected)


@pytest.mark.parametrize("revision", [REVISION, ""])
def test_download_script_passes_revision_to_sdk(monkeypatch, revision):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **kw: calls.append(kw)),
    )
    monkeypatch.setattr(sys, "argv", ["-c", "owner/repo", "model.gguf", "destination", revision])
    exec(_hf_download_script(), {})
    assert calls == [
        {
            "repo_id": "owner/repo",
            "filename": "model.gguf",
            "local_dir": "destination",
            "revision": revision or None,
        }
    ]


def test_queued_revision_survives_restart_and_duplicate_request(tmp_path):
    path = tmp_path / "jobs.db"
    store = DownloadStore(path)
    queued, _ = store.upsert_job(model())
    changed = model()
    changed["resolved_revision"] = "b" * 40
    duplicate, created = store.upsert_job(changed)
    assert not created
    assert (
        duplicate["artifact"]
        == queued["artifact"]
        == {
            "repository": "owner/repo",
            "filename": "model.gguf",
            "revision": REVISION,
            "revision_status": "pinned",
        }
    )
    reopened = DownloadStore(path)
    assert reopened.get_command(queued["target_id"])[3] == REVISION
    assert reopened.claim_next_queued()["artifact"] == queued["artifact"]


def test_legacy_job_exposes_unknown_revision(tmp_path):
    selected = model()
    selected.pop("resolved_revision")
    job, _ = DownloadStore(tmp_path / "jobs.db").upsert_job(selected)
    assert job["artifact"]["revision"] is None
    assert job["artifact"]["revision_status"] == "unknown"


@pytest.mark.parametrize("cached", [False, True])
def test_enrichment_carries_revision_and_caches_it(monkeypatch, cached):
    metadata = {"target_file": "model.gguf", "size_gb": None, "resolved_revision": REVISION}
    monkeypatch.setattr(
        hf_provider.cache_db, "get_model_cache", lambda *_: metadata if cached else None
    )
    writes = []
    monkeypatch.setattr(
        hf_provider.cache_db, "set_model_cache", lambda *args: writes.append(args[2])
    )
    info = SimpleNamespace(
        sha=REVISION, siblings=[SimpleNamespace(rfilename="model.gguf", size=None)]
    )
    monkeypatch.setattr(
        hf_provider, "HfApi", lambda **_: SimpleNamespace(model_info=lambda *a, **kw: info)
    )
    selected = model()
    selected.pop("resolved_revision")
    result = hf_provider.enrich_hf_model_details(selected, None, {})
    assert result["resolved_revision"] == REVISION
    if not cached:
        assert writes[0]["resolved_revision"] == REVISION


def test_cached_other_variant_does_not_replace_selected_file(monkeypatch):
    monkeypatch.setattr(
        hf_provider.cache_db,
        "get_model_cache",
        lambda *_: {"target_file": "other.gguf", "size_gb": None, "resolved_revision": "b" * 40},
    )
    monkeypatch.setattr(hf_provider.cache_db, "set_model_cache", lambda *_: None)
    info = SimpleNamespace(
        sha=REVISION, siblings=[SimpleNamespace(rfilename="model.gguf", size=None)]
    )
    monkeypatch.setattr(
        hf_provider, "HfApi", lambda **_: SimpleNamespace(model_info=lambda *a, **kw: info)
    )
    selected = model()
    selected.pop("resolved_revision")
    result = hf_provider.enrich_hf_model_details(selected, None, {})
    assert result["target_file"] == "model.gguf"
    assert result["resolved_revision"] == REVISION


def test_legacy_cached_metadata_cannot_clear_selected_revision(monkeypatch):
    monkeypatch.setattr(
        hf_provider.cache_db,
        "get_model_cache",
        lambda *_: {"target_file": "model.gguf", "size_gb": None},
    )
    info = SimpleNamespace(sha=REVISION, siblings=[SimpleNamespace(rfilename="model.gguf", size=None)])
    monkeypatch.setattr(
        hf_provider, "HfApi", lambda **_: SimpleNamespace(model_info=lambda *a, **kw: info)
    )
    monkeypatch.setattr(hf_provider.cache_db, "set_model_cache", lambda *_: None)
    selected = model()
    assert hf_provider.enrich_hf_model_details(selected, None, {})["resolved_revision"] == REVISION


def test_terminal_job_can_be_requeued_at_new_revision(tmp_path):
    store = DownloadStore(tmp_path / "jobs.db")
    first, _ = store.upsert_job(model())
    store.update_job(first["target_id"], status="completed")
    changed = model()
    changed["resolved_revision"] = "b" * 40
    second, created = store.upsert_job(changed)
    assert created
    assert second["artifact"]["revision"] == "b" * 40
    assert store.get_command(second["target_id"])[3] == "b" * 40
