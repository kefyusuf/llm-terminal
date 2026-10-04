"""Declared artifact facts stay separate from estimates and permissions."""

from types import SimpleNamespace

import pytest

from downloads.store import DownloadStore
from providers import hf_provider

REVISION = "a" * 40


def test_hf_details_cache_and_queue_keep_declared_facts(tmp_path, monkeypatch):
    metadata = {}
    monkeypatch.setattr(hf_provider.cache_db, "get_model_cache", lambda *_: metadata.get("cached"))
    monkeypatch.setattr(
        hf_provider.cache_db, "set_model_cache", lambda *a: metadata.update(cached=a[2])
    )
    info = SimpleNamespace(
        sha=REVISION,
        card_data={"license": "apache-2.0"},
        siblings=[
            SimpleNamespace(
                rfilename="model.gguf", size=1234, lfs=SimpleNamespace(sha256="b" * 64)
            ),
            SimpleNamespace(rfilename="LICENSE"),
            SimpleNamespace(rfilename="README.md"),
        ],
    )
    monkeypatch.setattr(
        hf_provider, "HfApi", lambda **_: SimpleNamespace(model_info=lambda *a, **kw: info)
    )
    monkeypatch.setattr(hf_provider, "calculate_fit", lambda *a: ("Fit", "CPU", None))
    model = {"source": "Hugging Face", "id": "owner/repo", "target_file": "model.gguf"}
    hf_provider.enrich_hf_model_details(model, {}, {})
    artifact = model["artifact_metadata"]
    assert artifact["size_bytes"] == 1234
    assert artifact["sha256"] == "b" * 64
    assert artifact["license"]["id"] == "apache-2.0"
    assert artifact["license"]["status"] == "declared"
    assert (
        artifact["license"]["url"] == f"https://huggingface.co/owner/repo/blob/{REVISION}/LICENSE"
    )
    assert artifact["metadata_observed_at"] > 0
    cached = {"source": "Hugging Face", "id": "owner/repo", "target_file": "model.gguf"}
    hf_provider.enrich_hf_model_details(cached, {}, {})
    assert cached["artifact_metadata"] == artifact
    store = DownloadStore(tmp_path / "jobs.db")
    job, _ = store.upsert_job(model)
    assert (
        DownloadStore(tmp_path / "jobs.db").get_job_by_target(job["target_id"])["artifact_metadata"]
        == artifact
    )


def test_missing_metadata_is_explicit_not_application_mit():
    from core.artifacts import hf_artifact_metadata

    artifact = hf_artifact_metadata("owner/repo", "model.gguf", None)
    assert artifact["size_bytes"] is None
    assert artifact["resolved_revision"] is None
    assert artifact["sha256"] is None
    assert artifact["license"] == {"id": None, "name": None, "url": None, "status": "unknown"}
    assert artifact["metadata_status"] == "unavailable"


@pytest.mark.parametrize(
    "link,expected",
    [
        ("https://example.org/license", "https://example.org/license"),
        ("javascript:alert(1)", None),
        ("http://example.org/license", None),
        ("LICENSE.md", "https://huggingface.co/owner/repo/blob/" + REVISION + "/LICENSE.md"),
    ],
)
def test_custom_license_links_remain_declared_and_safe(link, expected):
    from core.artifacts import hf_artifact_metadata

    info = SimpleNamespace(
        sha=REVISION,
        card_data={"license": "other", "license_name": "custom", "license_link": link},
        siblings=[SimpleNamespace(rfilename="LICENSE.md")],
    )
    artifact = hf_artifact_metadata("owner/repo", "model.gguf", info)
    assert artifact["license"]["url"] == expected
    assert artifact["license"]["name"] == "custom"


def test_queue_rejects_metadata_for_another_selected_file(tmp_path):
    store = DownloadStore(tmp_path / "jobs.db")
    with pytest.raises(ValueError, match="artifact metadata"):
        store.upsert_job(
            {
                "source": "Hugging Face",
                "id": "owner/repo",
                "target_file": "model.gguf",
                "artifact_metadata": {
                    "schema_version": 1,
                    "repository": "owner/repo",
                    "filename": "other.gguf",
                },
            }
        )
    assert store.list_jobs() == []


def test_corrupt_metadata_does_not_break_job_listing(tmp_path):
    store = DownloadStore(tmp_path / "jobs.db")
    job, _ = store.upsert_job({"source": "Ollama", "name": "example"})
    with store._connect() as conn:
        conn.execute("UPDATE jobs SET artifact_metadata_json = ?", ("invalid JSON",))
    assert store.get_job_by_target(job["target_id"])["artifact_metadata"] is None


def test_upgrade_preserves_legacy_job_and_unknown_metadata(tmp_path):
    path = tmp_path / "jobs.db"
    store = DownloadStore(path)
    job, _ = store.upsert_job(
        {"source": "Hugging Face", "id": "owner/repo", "target_file": "model.gguf"}
    )
    with store._connect() as conn:
        conn.execute("ALTER TABLE jobs DROP COLUMN artifact_metadata_json")
    upgraded = DownloadStore(path)
    assert upgraded.get_job_by_target(job["target_id"])["artifact_metadata"] is None
    assert upgraded.get_command(job["target_id"]) == ["hf_api_download", "owner/repo", "model.gguf"]


def test_detail_modal_displays_declared_license_without_rich_markup():
    import asyncio

    from textual.app import App
    from textual.widgets import Label

    from app.modals import ModelDetailModal

    async def scenario():
        app = App()
        model = {
            "source": "Hugging Face",
            "name": "model",
            "id": "owner/repo",
            "target_file": "model.gguf",
            "artifact_metadata": {
                "license": {"id": "[red]custom", "status": "declared"},
                "size_bytes": 1234,
                "source_url": "https://huggingface.co/owner/repo",
            },
        }
        async with app.run_test(size=(80, 40)) as pilot:
            app.push_screen(ModelDetailModal(model))
            await pilot.pause()
            info = app.screen.query_one("#artifact-info", Label)
            assert "1234" in str(info.renderable)
            assert "[red]custom" in str(info.renderable)
            assert "declared" in str(info.renderable)
            assert "permission" in str(info.renderable)

    asyncio.run(scenario())
