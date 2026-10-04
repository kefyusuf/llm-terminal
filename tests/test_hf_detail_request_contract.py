"""Detail fetches respect credential, deadline and selected revision boundaries."""
from types import SimpleNamespace

from requests.exceptions import RequestException

import config
from app.viewer import AIModelViewer
from providers import hf_provider

REVISION = "a" * 40


def selected():
    return {"source": "Hugging Face", "id": "owner/repo", "target_file": "model.gguf",
            "resolved_revision": REVISION}


def prepare(monkeypatch, response, cached=None):
    calls = []
    monkeypatch.setattr(hf_provider.cache_db, "get_model_cache", lambda *_: cached)
    monkeypatch.setattr(hf_provider.cache_db, "set_model_cache", lambda *_: None)

    def api(**options):
        calls.append(options)

        def model_info(repo, **kwargs):
            calls.append({"repository": repo, **kwargs})
            if isinstance(response, Exception):
                raise response
            return response

        return SimpleNamespace(model_info=model_info)

    monkeypatch.setattr(hf_provider, "HfApi", api)
    return calls


def info(revision=REVISION, filename="model.gguf"):
    return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename=filename, size=None)])


def test_detail_request_uses_configured_token_timeout_and_selected_commit(monkeypatch):
    calls = prepare(monkeypatch, info())
    monkeypatch.setattr(config.settings, "hf_token", "fixture-token")
    result = hf_provider.enrich_hf_model_details(selected(), {}, {})
    assert calls == [{"token": "fixture-token"}, {"repository": "owner/repo",
                    "files_metadata": True, "timeout": 10, "revision": REVISION}]
    assert result["metadata_fetch_status"] == "available"


def test_cached_other_commit_cannot_replace_selected_commit(monkeypatch):
    calls = prepare(monkeypatch, info(), {"target_file": "model.gguf", "resolved_revision": "b" * 40})
    result = hf_provider.enrich_hf_model_details(selected(), {}, {})
    assert len(calls) == 2
    assert result["resolved_revision"] == REVISION


def test_metadata_failure_is_explicit_and_does_not_expose_exception_text(monkeypatch):
    prepare(monkeypatch, RequestException("fixture-token private diagnostic"))
    result = hf_provider.enrich_hf_model_details(selected(), {}, {})
    assert result["metadata_fetch_status"] == "failed"
    assert result["metadata_fetch_error"] == "Hugging Face metadata request failed."
    assert "fixture-token" not in repr(result)
    assert result["resolved_revision"] == REVISION


def test_response_from_other_commit_is_rejected_without_cache_write(monkeypatch):
    prepare(monkeypatch, info("b" * 40))
    writes = []
    monkeypatch.setattr(hf_provider.cache_db, "set_model_cache", lambda *args: writes.append(args))
    result = hf_provider.enrich_hf_model_details(selected(), {}, {})
    assert result["resolved_revision"] == REVISION
    assert result["metadata_fetch_status"] == "failed"
    assert writes == []


def test_missing_selected_file_is_explicit_and_preserves_requested_revision(monkeypatch):
    prepare(monkeypatch, info(filename="other.gguf"))
    result = hf_provider.enrich_hf_model_details(selected(), {}, {})
    assert result["metadata_fetch_status"] == "unavailable"
    assert result["resolved_revision"] == REVISION
    assert result["target_file"] == "model.gguf"


def test_detail_ready_does_not_claim_failed_metadata_loaded():
    messages = []
    viewer = SimpleNamespace(all_results=[], refresh_table=lambda: None,
                             update_status=messages.append, open_model_detail_modal=lambda _: None)
    AIModelViewer.on_hf_detail_ready(viewer, {**selected(), "metadata_fetch_status": "failed"})
    assert messages == ["Detailed metadata unavailable. Showing retained facts and estimates."]
