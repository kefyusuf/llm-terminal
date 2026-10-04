"""Read-only plans, durable reservations and verified completion contracts."""

import hashlib
from types import SimpleNamespace

import pytest

from core.artifacts import hf_artifact_metadata
from downloads.store import DownloadStore


def selection(repository="owner/repo", size=100):
    info = SimpleNamespace(
        sha="a" * 40,
        card_data={"license": "mit"},
        siblings=[SimpleNamespace(rfilename="model.gguf", size=size)],
    )
    metadata = hf_artifact_metadata(repository, "model.gguf", info)
    return {
        "source": "Hugging Face",
        "id": repository,
        "target_file": "model.gguf",
        "resolved_revision": "a" * 40,
        "artifact_metadata": metadata,
    }


def test_plan_does_not_create_destination_and_isolates_repositories(tmp_path, monkeypatch):
    from downloads import preflight

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**9))
    root = tmp_path / "models"
    first = preflight.plan_hf_download(selection(), root)
    second = preflight.plan_hf_download(selection("other/repo"), root)
    assert first["allowed"]
    assert first["destination"] != second["destination"]
    assert first["size_bytes"] == 100
    assert first["required_bytes"] == 100 + preflight.DISK_SAFETY_MARGIN
    assert not root.exists()


def test_unknown_size_requires_explicit_acknowledgement(tmp_path, monkeypatch):
    from downloads import preflight

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**9))
    assert not preflight.plan_hf_download(selection(size=None), tmp_path)["allowed"]
    accepted = preflight.plan_hf_download(selection(size=None), tmp_path, allow_unknown_size=True)
    assert accepted["allowed"]
    assert accepted["status"] == "unknown_size"
    assert accepted["required_bytes"] is None


def test_disk_reservations_are_transactional_and_survive_restart(tmp_path, monkeypatch):
    from downloads import preflight

    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(free=preflight.DISK_SAFETY_MARGIN + 150),
    )
    path = tmp_path / "jobs.db"
    store = DownloadStore(path)
    first, _ = store.upsert_job(selection(), models_dir=tmp_path / "models")
    reopened = DownloadStore(path)
    with pytest.raises(ValueError, match="insufficient_disk"):
        reopened.upsert_job(selection("other/repo"), models_dir=tmp_path / "models")
    assert reopened.get_job_by_target(first["target_id"])["download_plan"] == first["download_plan"]
    store.update_job(first["target_id"], status="failed")
    assert reopened.upsert_job(selection("other/repo"), models_dir=tmp_path / "models")[1]


@pytest.mark.parametrize("kind", ["missing", "wrong_size", "wrong_digest", "valid"])
def test_completion_requires_real_matching_artifact(tmp_path, kind):
    from downloads.preflight import verify_hf_artifact

    if kind != "missing":
        (tmp_path / "model.gguf").write_bytes(b"content")
    size = 5 if kind == "wrong_size" else 7
    digest = "b" * 64 if kind == "wrong_digest" else hashlib.sha256(b"content").hexdigest()
    if kind == "valid":
        verify_hf_artifact(tmp_path, "model.gguf", size, digest)
    else:
        with pytest.raises(ValueError):
            verify_hf_artifact(tmp_path, "model.gguf", size, digest)


def test_linked_auxiliary_cache_blocks_plan(tmp_path, monkeypatch):
    from pathlib import Path

    from downloads import preflight

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**9))
    plan = preflight.plan_hf_download(selection(), tmp_path / "models")
    job_root = Path(plan["model_directory"])
    job_root.mkdir(parents=True)
    target = tmp_path / "external-cache"
    target.mkdir()
    # Hard links are available without Windows symlink privileges.
    target_file = target / "shared"
    target_file.write_bytes(b"shared")
    (job_root / ".cache").mkdir()
    (job_root / ".cache" / "shared").hardlink_to(target_file)
    rejected = preflight.plan_hf_download(selection(), tmp_path / "models")
    assert not rejected["allowed"]
    assert rejected["status"] == "unsafe_destination"


def test_hf_child_cannot_report_success_without_a_real_file(tmp_path, monkeypatch):
    import sys

    from downloads.runner import _hf_download_script

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lambda **kw: "missing")
    )
    monkeypatch.setattr(
        sys, "argv", ["-c", "owner/repo", "model.gguf", str(tmp_path), "a" * 40, "100", ""]
    )
    with pytest.raises(ValueError, match="missing"):
        exec(_hf_download_script(), {})


def test_service_previews_and_queues_server_owned_plan(tmp_path, monkeypatch):
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen

    import config
    from downloads import preflight
    from downloads.api import _make_handler

    monkeypatch.setattr(config.settings, "hf_models_dir", tmp_path / "models")
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**9))
    state = SimpleNamespace(store=DownloadStore(tmp_path / "jobs.db"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state, "test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:

        def request(path):
            req = Request(
                f"http://127.0.0.1:{server.server_port}{path}",
                data=json.dumps({"model": selection()}).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer test-token"},
                method="POST",
            )
            with urlopen(req, timeout=3) as response:
                return json.load(response)

        plan = request("/jobs/plan")["plan"]
        assert plan["allowed"]
        assert state.store.list_jobs() == []
        job = request("/jobs")["job"]
        assert job["download_plan"]["destination"] == plan["destination"]
        assert not (tmp_path / "models").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_cli_offline_plan_has_versioned_json_and_no_queue(tmp_path, monkeypatch):
    import json

    from click.testing import CliRunner

    import config
    from cli import cli

    monkeypatch.setattr(config.settings, "hf_models_dir", tmp_path / "models")
    result = CliRunner().invoke(
        cli, ["download-plan", "owner/repo", "model.gguf", "--offline", "--json"]
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["schema_version"] == 1
    assert plan["status"] == "unknown_size"
    assert not plan["allowed"]
    assert not (tmp_path / "models").exists()


def test_copied_hf_command_is_exact_and_pinned():
    from downloads.preflight import hf_download_command

    command = hf_download_command(selection())
    assert "*.gguf" not in command
    assert "model.gguf" in command
    assert "a" * 40 in command
    assert "--revision" in command


def test_preflight_requires_new_service_protocol():
    from downloads.service_client import is_service_compatible

    assert not is_service_compatible({"version": "1.8"})
    assert not is_service_compatible({"version": "2.0"})
    assert is_service_compatible({"version": "2.1"})


def test_unusable_configured_root_blocks_plan(tmp_path, monkeypatch):
    from downloads import preflight

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**9))
    root = tmp_path / "not-directory"
    root.write_text("data", encoding="utf-8")
    assert preflight.plan_hf_download(selection(), root)["status"] == "unsafe_destination"


def test_concurrent_services_cannot_overreserve_disk(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from downloads import preflight

    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(free=preflight.DISK_SAFETY_MARGIN + 150),
    )
    path = tmp_path / "jobs.db"
    stores = [DownloadStore(path), DownloadStore(path)]
    barrier = threading.Barrier(2)

    def enqueue(index):
        barrier.wait(timeout=5)
        try:
            stores[index].upsert_job(
                selection(f"owner/repo{index}"), models_dir=tmp_path / "models"
            )
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(enqueue, range(2))) == [False, True]


def test_powershell_command_uses_executable_syntax_and_quotes_apostrophes():
    from downloads.preflight import hf_download_command

    model = {"source": "Hugging Face", "id": "owner/repo", "target_file": "model's.gguf"}
    command = hf_download_command(model, powershell=True)
    assert command.startswith("& '")
    assert "huggingface_hub.commands.huggingface_cli" in command
    assert "'model''s.gguf'" in command


def test_unknown_size_ui_click_is_explicit_acknowledgement():
    import asyncio

    from textual.app import App
    from textual.widgets import Button, Label

    from app.modals import ModelDetailModal

    async def scenario():
        app = App()
        captured = []
        app.start_model_download = captured.append
        model = selection(size=None)
        model["name"] = "model"
        model["download_plan"] = {
            "status": "unknown_size",
            "allowed": False,
            "destination": "example",
        }
        async with app.run_test(size=(80, 50)) as pilot:
            app.push_screen(ModelDetailModal(model))
            await pilot.pause()
            assert "example" in str(app.screen.query_one("#download-plan-info", Label).renderable)
            button = app.screen.query_one("#download-btn", Button)
            assert not button.disabled
            button.scroll_visible()
            await pilot.pause()
            await pilot.click("#download-btn")
            assert captured[0]["allow_unknown_size"] is True

    asyncio.run(scenario())


def test_upgrade_does_not_interrupt_active_legacy_jobs(monkeypatch):
    from downloads import service_client

    actions = []
    monkeypatch.setattr(service_client, "is_service_running", lambda: True)
    monkeypatch.setattr(
        service_client, "get_service_health", lambda: {"ok": True, "version": "1.8"}
    )
    monkeypatch.setattr(service_client, "list_jobs", lambda **kw: [{"status": "running"}])
    monkeypatch.setattr(service_client, "stop_service", lambda: actions.append("stop") or True)
    monkeypatch.setattr(service_client, "_start_service_process", lambda: actions.append("start"))
    monkeypatch.setattr(service_client, "_wait_for_service", lambda **kw: True)
    assert service_client.ensure_service_running() is False
    assert actions == []


def test_stop_service_never_kills_unrelated_matching_processes(monkeypatch):
    from urllib.error import URLError

    from downloads import service_client

    def unavailable(*args, **kw):
        raise URLError("unavailable")

    monkeypatch.setattr(service_client, "_request", unavailable)
    monkeypatch.setattr(service_client, "_owned_service_process", None, raising=False)
    monkeypatch.setattr(
        service_client,
        "psutil",
        SimpleNamespace(process_iter=lambda *a: pytest.fail("must not enumerate other processes")),
    )
    assert service_client.stop_service() is False


@pytest.mark.parametrize("repository", [42, "a/b/c", "-flags", "../repo", "model name"])
def test_invalid_repository_is_rejected_before_queueing(tmp_path, repository):
    store = DownloadStore(tmp_path / "jobs.db")
    with pytest.raises(ValueError, match="repository"):
        store.upsert_job(
            {
                "source": "Hugging Face",
                "id": repository,
                "target_file": "model.gguf",
                "allow_unknown_size": True,
            },
            models_dir=tmp_path / "models",
        )
    assert store.list_jobs() == []


def test_offline_plan_preserves_requested_immutable_revision(tmp_path, monkeypatch):
    import json

    from click.testing import CliRunner

    import config
    from cli import cli

    monkeypatch.setattr(config.settings, "hf_models_dir", tmp_path / "models")
    result = CliRunner().invoke(
        cli,
        [
            "download-plan",
            "owner/repo",
            "model.gguf",
            "--offline",
            "--revision",
            "a" * 40,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["resolved_revision"] == "a" * 40
    assert plan["artifact_metadata"]["revision_origin"] == "requested"
