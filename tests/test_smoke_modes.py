import asyncio

import api_server
import downloads.download_service as download_service
import main
from app import AIModelViewer


def test_main_smoke_mode_exits_cleanly(monkeypatch):
    calls = []

    monkeypatch.setenv("AIMODEL_SMOKE", "1")
    monkeypatch.setattr(main, "run_smoke_check", lambda: calls.append("smoke") or 0)

    assert main.main() == 0

    assert calls == ["smoke"]


def test_api_server_smoke_mode_exits_cleanly(monkeypatch):
    calls = []

    monkeypatch.setenv("AIMODEL_SMOKE", "1")
    monkeypatch.setattr(api_server, "run_smoke_check", lambda: calls.append("smoke") or 0)

    assert api_server.main([]) == 0
    assert calls == ["smoke"]


def test_download_service_smoke_mode_exits_cleanly(monkeypatch):
    calls = []

    monkeypatch.setenv("AIMODEL_SMOKE", "1")
    monkeypatch.setattr(download_service, "run_smoke_check", lambda: calls.append("smoke") or 0)

    assert download_service.main() == 0
    assert calls == ["smoke"]


def test_textual_app_smoke_mode_exits_in_headless_test(monkeypatch):
    monkeypatch.setenv("AIMODEL_SMOKE", "1")

    async def _run() -> None:
        app = AIModelViewer()
        async with app.run_test() as pilot:
            await pilot.pause()
        assert app.return_code == 0

    asyncio.run(_run())


def test_ai_model_viewer_smoke_mode_schedules_timer_fallback(monkeypatch):
    viewer = AIModelViewer()
    scheduled = []

    class _DummyTable:
        def __init__(self):
            self.zebra_stripes = False
            self.columns = []

        def add_columns(self, *columns):
            self.columns.extend(columns)

    results_table = _DummyTable()
    download_table = _DummyTable()

    monkeypatch.setattr(viewer, "_apply_ui_mode", lambda: None)
    monkeypatch.setattr(viewer, "_configure_results_table_columns", lambda force=False: None)
    monkeypatch.setattr(viewer, "refresh_download_history_table", lambda: None)
    monkeypatch.setattr(viewer, "_update_results_meta", lambda _count: None)
    monkeypatch.setattr(viewer, "_smoke_mode_enabled", lambda: True)
    monkeypatch.setattr(viewer, "call_after_refresh", lambda callback: scheduled.append(("refresh", callback)))
    monkeypatch.setattr(viewer, "set_timer", lambda delay, callback: scheduled.append(("timer", delay, callback)))

    def _query_one(selector, _widget_type=None):
        if selector == "#results-table":
            return results_table
        if selector == "#download-history-table":
            return download_table
        raise AssertionError(selector)

    monkeypatch.setattr(viewer, "query_one", _query_one)

    viewer.on_mount()

    assert results_table.zebra_stripes is True
    assert download_table.columns == ["Source", "Publisher", "Model", "Status", "Detail", "Action"]
    assert scheduled[0] == ("refresh", viewer._finish_smoke_mode)
    assert scheduled[1] == ("timer", 1, viewer._finish_smoke_mode)
