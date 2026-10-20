"""Provider transfers preserve progress, cancellation and verified completion."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from downloads import backends


class Response:
    def __init__(self, records=(), chunks=(), status=200, headers=None):
        self.records = records
        self.chunks = chunks
        self.status_code = status
        self.headers = headers or {}
        self.closed = False
        self.raw = self
        self.stream = iter(chunks or (json.dumps(record).encode() + b"\n" for record in records))

    def read1(self, *_, **__):
        return next(self.stream, b"")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ValueError("HTTP failure")

    def iter_lines(self, **_):
        yield from (json.dumps(record).encode() for record in self.records)

    def iter_content(self, **_):
        yield from self.chunks


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    get = post


def test_ollama_stream_reports_layer_progress_and_requires_success(monkeypatch):
    response = Response(
        records=[
            {"status": "pulling abc", "digest": "abc", "total": 100, "completed": 50},
            {"status": "success"},
        ]
    )
    session = Session(response)
    monkeypatch.setattr(backends, "create_session", lambda **_: session)
    snapshots = []
    backends.download(
        {},
        ["ollama", "pull", "tiny:latest"],
        threading.Event(),
        lambda detail, progress="": snapshots.append((detail, progress)),
    )
    assert any(progress == "50%" and "pulling abc" in detail for detail, progress in snapshots)
    assert session.calls[0][1]["json"] == {"name": "tiny:latest", "stream": True}
    assert session.closed and response.closed


@pytest.mark.parametrize("records", [[], [{"error": "not found"}]])
def test_ollama_incomplete_or_error_stream_fails(monkeypatch, records):
    monkeypatch.setattr(backends, "create_session", lambda **_: Session(Response(records=records)))
    with pytest.raises(RuntimeError):
        backends.download({}, ["ollama", "pull", "tiny"], threading.Event(), lambda *_: None)


def test_ollama_cancel_closes_stream(monkeypatch):
    cancel = threading.Event()
    response = Response(records=[{"status": "pulling abc"}, {"status": "success"}])
    session = Session(response)
    monkeypatch.setattr(backends, "create_session", lambda **_: session)
    with pytest.raises(backends.DownloadCancelled):
        backends.download({}, ["ollama", "pull", "tiny"], cancel, lambda *_: cancel.set())
    assert session.closed and response.closed


def hf_setup(tmp_path, monkeypatch, payload=b"model"):
    from downloads.preflight import plan_hf_download

    model = {
        "source": "Hugging Face",
        "id": "owner/repo",
        "target_file": "model.gguf",
        "resolved_revision": "a" * 40,
    }
    monkeypatch.setattr(backends.config.settings, "hf_models_dir", str(tmp_path))
    plan = plan_hf_download(model, tmp_path, allow_unknown_size=True)
    metadata = SimpleNamespace(commit_hash="a" * 40, etag="b" * 40, size=len(payload))
    monkeypatch.setattr(backends, "get_hf_file_metadata", lambda *_, **__: metadata)
    return {"download_plan": plan}, ["hf_api_download", "owner/repo", "model.gguf", "a" * 40]


def test_hf_download_verifies_artifact_and_reports_progress(tmp_path, monkeypatch):
    job, command = hf_setup(tmp_path, monkeypatch)
    session = Session(Response(chunks=[b"mo", b"del"]))
    monkeypatch.setattr(backends, "create_session", lambda **_: session)
    snapshots = []
    backends.download(job, command, threading.Event(), lambda *args: snapshots.append(args))
    assert Path(job["download_plan"]["destination"]).read_bytes() == b"model"
    assert snapshots[-1][1] == "100%"
    assert any("MiB/s" in detail and "ETA" in detail for detail, _ in snapshots)
    assert session.closed


def test_hf_cancel_retains_sdk_partial_and_resumes(tmp_path, monkeypatch):
    job, command = hf_setup(tmp_path, monkeypatch)
    cancel = threading.Event()
    first = Session(Response(chunks=[b"mo", b"del"]))
    monkeypatch.setattr(backends, "create_session", lambda **_: first)
    with pytest.raises(backends.DownloadCancelled):
        backends.download(job, command, cancel, lambda *_: cancel.set())
    destination = Path(job["download_plan"]["destination"])
    assert not destination.exists()
    partial = next(destination.parent.rglob("*.incomplete"))
    assert partial.read_bytes() == b"mo"
    resumed = Session(
        Response(chunks=[b"del"], status=206, headers={"Content-Range": "bytes 2-4/5"})
    )
    monkeypatch.setattr(backends, "create_session", lambda **_: resumed)
    backends.download(job, command, threading.Event(), lambda *_: None)
    assert resumed.calls[0][1]["headers"]["Range"] == "bytes=2-"
    assert destination.read_bytes() == b"model"
    assert not partial.exists()


def test_hf_ignored_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    job, command = hf_setup(tmp_path, monkeypatch)
    paths = backends.hf_download_paths(
        Path(job["download_plan"]["model_directory"]), "model.gguf"
    )
    paths.incomplete_path("b" * 40).write_bytes(b"mo")
    monkeypatch.setattr(
        backends, "create_session", lambda **_: Session(Response(chunks=[b"model"]))
    )
    backends.download(job, command, threading.Event(), lambda *_: None)
    assert paths.file_path.read_bytes() == b"model"


@pytest.mark.parametrize("content_range", ["bytes 0-2/5", "bytes 2-4/6", "malformed"])
def test_hf_invalid_resume_range_does_not_touch_partial(tmp_path, monkeypatch, content_range):
    job, command = hf_setup(tmp_path, monkeypatch)
    paths = backends.hf_download_paths(
        Path(job["download_plan"]["model_directory"]), "model.gguf"
    )
    partial = paths.incomplete_path("b" * 40)
    partial.write_bytes(b"mo")
    monkeypatch.setattr(
        backends,
        "create_session",
        lambda **_: Session(
            Response(chunks=[b"del"], status=206, headers={"Content-Range": content_range})
        ),
    )
    with pytest.raises(ValueError, match="resume range"):
        backends.download(job, command, threading.Event(), lambda *_: None)
    assert partial.read_bytes() == b"mo"
    assert not paths.file_path.exists()


def test_hf_truncated_response_cannot_complete(tmp_path, monkeypatch):
    job, command = hf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(backends, "create_session", lambda **_: Session(Response(chunks=[b"mo"])))
    with pytest.raises(ValueError, match="byte count"):
        backends.download(job, command, threading.Event(), lambda *_: None)
    assert not Path(job["download_plan"]["destination"]).exists()


def test_hf_mismatched_digest_cannot_complete(tmp_path, monkeypatch):
    import hashlib

    job, command = hf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        backends,
        "get_hf_file_metadata",
        lambda *_, **__: SimpleNamespace(
            commit_hash="a" * 40, etag=hashlib.sha256(b"other").hexdigest(), size=5
        ),
    )
    monkeypatch.setattr(
        backends, "create_session", lambda **_: Session(Response(chunks=[b"model"]))
    )
    with pytest.raises(ValueError, match="SHA-256"):
        backends.download(job, command, threading.Event(), lambda *_: None)


def test_hf_same_size_existing_file_needs_matching_identity(tmp_path, monkeypatch):
    job, command = hf_setup(tmp_path, monkeypatch)
    paths = backends.hf_download_paths(
        Path(job["download_plan"]["model_directory"]), "model.gguf"
    )
    paths.file_path.write_bytes(b"stale")
    # Three-line commit/etag/timestamp format shared with huggingface_hub.
    paths.metadata_path.write_text("\n".join(["c" * 40, "d" * 40, str(time.time())]) + "\n")
    session = Session(Response(chunks=[b"model"]))
    monkeypatch.setattr(backends, "create_session", lambda **_: session)
    backends.download(job, command, threading.Event(), lambda *_: None)
    assert session.calls
    assert paths.file_path.read_bytes() == b"model"


def test_ollama_cancel_during_unterminated_progress_record(monkeypatch):
    cancel = threading.Event()

    def chunks():
        yield b'{"status": "pulling'
        cancel.set()
        yield b" abc"
        pytest.fail("cancellation must stop reading before a newline")

    response = Response(chunks=chunks())
    monkeypatch.setattr(backends, "create_session", lambda **_: Session(response))
    with pytest.raises(backends.DownloadCancelled):
        backends.download({}, ["ollama", "pull", "tiny"], cancel, lambda *_: None)
    assert response.closed


def test_ollama_progress_record_is_bounded(monkeypatch):
    response = Response(chunks=[b"x" * 65537])
    monkeypatch.setattr(backends, "create_session", lambda **_: Session(response))
    with pytest.raises(ValueError, match="size limit"):
        backends.download({}, ["ollama", "pull", "tiny"], threading.Event(), lambda *_: None)


def test_hf_sdk_lock_wait_can_be_cancelled(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from huggingface_hub.utils import WeakFileLock

    job, command = hf_setup(tmp_path, monkeypatch)
    paths = backends.hf_download_paths(
        Path(job["download_plan"]["model_directory"]), "model.gguf"
    )
    attempted = threading.Event()
    cancel = threading.Event()
    errors = []

    @contextmanager
    def observe_lock(*args, **kwargs):
        attempted.set()
        with WeakFileLock(*args, **kwargs):
            yield

    def run():
        try:
            backends.download(job, command, cancel, lambda *_: None)
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(backends, "WeakFileLock", observe_lock)
    with WeakFileLock(paths.lock_path):
        worker = threading.Thread(target=run)
        worker.start()
        assert attempted.wait(2)
        cancel.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], backends.DownloadCancelled)
    assert not paths.file_path.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_hf_hash_verification_checks_cancel_between_chunks(tmp_path, monkeypatch, existing):
    from downloads import preflight

    payload = b"x" * (2 * 1024 * 1024)
    job, command = hf_setup(tmp_path, monkeypatch, payload)
    sha256 = preflight.hashlib.sha256
    expected = sha256(payload).hexdigest()
    monkeypatch.setattr(
        backends,
        "get_hf_file_metadata",
        lambda *_, **__: SimpleNamespace(commit_hash="a" * 40, etag=expected, size=len(payload)),
    )
    if existing:
        paths = backends.hf_download_paths(
            Path(job["download_plan"]["model_directory"]), "model.gguf"
        )
        paths.file_path.write_bytes(payload)
    monkeypatch.setattr(backends, "create_session", lambda **_: Session(Response(chunks=[payload])))
    cancel = threading.Event()
    hashed_chunks = []

    class CancellingHash:
        def __init__(self, *args, **kwargs):
            self.digest = sha256(*args, **kwargs)

        def update(self, chunk):
            hashed_chunks.append(len(chunk))
            self.digest.update(chunk)
            cancel.set()

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(preflight.hashlib, "sha256", CancellingHash)
    with pytest.raises(backends.DownloadCancelled):
        backends.download(job, command, cancel, lambda *_: None)
    assert hashed_chunks == [1024 * 1024]


@pytest.mark.parametrize("depth", [1, 220])
def test_hf_download_paths_match_sdk_layout_for_resume(tmp_path, depth):
    """Partials from earlier SDK-based downloads must resolve to the same paths."""
    from huggingface_hub._local_folder import get_local_download_paths

    local_dir = tmp_path / ("d" * depth)
    ours = backends.hf_download_paths(local_dir, "sub/model.gguf")
    sdk = get_local_download_paths(local_dir, "sub/model.gguf")
    assert (ours.file_path, ours.lock_path, ours.metadata_path) == (
        sdk.file_path,
        sdk.lock_path,
        sdk.metadata_path,
    )
    assert ours.incomplete_path("e" * 40) == sdk.incomplete_path("e" * 40)
