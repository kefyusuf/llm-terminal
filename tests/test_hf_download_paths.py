"""Portable HF filename validation happens before a job is persisted."""

import pytest

from downloads.store import DownloadStore


@pytest.mark.parametrize(
    "filename",
    [
        "../outside.gguf",
        "nested/../../outside.gguf",
        "/absolute.gguf",
        "C:/outside.gguf",
        "C:relative.gguf",
        "\\\\server\\share\\model.gguf",
        "nested\\..\\outside.gguf",
        "nested/./model.gguf",
        "nested//model.gguf",
        "model.gguf:stream",
        "model.gguf\x00",
        "nested/",
        ".",
        "..",
        "CON.gguf",
        "nested/NUL",
        "nested/com1.gguf",
        "nested/LPT9.txt",
        "model.gguf.",
        "model.gguf ",
        "nested /model.gguf",
        "model?.gguf",
        42,
    ],
)
def test_unsafe_filename_is_rejected_before_queue_write(tmp_path, filename):
    store = DownloadStore(tmp_path / "jobs.db")
    with pytest.raises(ValueError, match="Hugging Face target file"):
        store.upsert_job({"source": "Hugging Face", "id": "owner/repo", "target_file": filename})
    assert store.list_jobs() == []


@pytest.mark.parametrize(
    "filename", ["model.gguf", "nested/model-Q4_K_M.gguf", "nested/model name.gguf"]
)
def test_safe_relative_filename_is_preserved(tmp_path, filename):
    store = DownloadStore(tmp_path / "jobs.db")
    job, created = store.upsert_job(
        {"source": "Hugging Face", "id": "owner/repo", "target_file": filename}
    )
    assert created
    assert job["artifact"]["filename"] == filename
