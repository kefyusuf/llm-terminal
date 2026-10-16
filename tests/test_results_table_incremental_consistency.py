from __future__ import annotations

import asyncio
import time

import pytest
from textual.widgets import DataTable, Static

from app.viewer import AIModelViewer


class _LayoutOnlyViewer(AIModelViewer):
    def on_mount(self, event) -> None:
        """Skip service/timer side effects; exercise the shipped runtime results table only."""
        event.prevent_default()
        self._configure_results_table_columns(force=True)

    def on_resize(self, _event) -> None:
        """Avoid scheduling resize work in the headless table-only harness."""
        pass

    def _apply_resize_reflow(self, _generation: int) -> None:
        """Neutralize callbacks already queued before the resize handler override takes effect."""
        pass


@pytest.mark.parametrize("width, compact", [(80, True), (100, True), (120, True), (160, True), (100, False), (140, False)])
def test_download_column_remains_visible_without_horizontal_overflow(monkeypatch, width, compact):
    monkeypatch.setattr("app.viewer.get_provider_filter_labels", lambda: ("Ollama",))

    async def run():
        app = _LayoutOnlyViewer()
        async with app.run_test(size=(width, 40)) as pilot:
            app.compact_mode = compact
            app.current_filter = "Ollama"
            app.all_results = [
                _result(model_id=f"model-{index}", name=f"model-{index}", publisher="Test", quality=10)
                for index in range(50)
            ]
            app.refresh_table()
            await pilot.pause()
            app.refresh_table()
            await pilot.pause()

            table = app.query_one("#results-table", DataTable)
            assert "download" in app.results_column_keys
            assert table.row_count == 50
            assert table.virtual_size.width <= table.scrollable_content_region.width

    asyncio.run(run())


@pytest.mark.parametrize("terminal_state", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("independent_status", [False, True])
def test_terminal_download_snapshot_refreshes_history_and_owned_status(terminal_state, independent_status):
    async def run():
        app = _LayoutOnlyViewer()
        async with app.run_test(size=(100, 40)) as pilot:
            history = app.query_one("#download-history-table", DataTable)
            history.add_columns("Source", "Publisher", "Model", "Status", "Detail", "Action")
            job = {
                "target_id": "Ollama:fixture", "source": "Ollama",
                "name": "fixture", "status": "running", "detail": "Downloading",
            }
            app._apply_download_poll_snapshot([job], None, None)
            await pilot.pause()
            status = app.query_one("#status-bar", Static)
            assert status.renderable == "1 download(s) active"
            if independent_status:
                app.update_status("Search completed")
            app.dl.download_history_refresh_pending = False
            app.dl.last_download_history_refresh_at = time.time()

            app._apply_download_poll_snapshot([{**job, "status": terminal_state, "detail": "Finished"}], None, None)
            await pilot.pause()

            assert not app.dl.active_downloads
            assert app.dl.download_registry["ollama:fixture"]["state"] == terminal_state
            history = app.query_one("#download-history-table", DataTable)
            assert "Downloading" not in str(history.get_row_at(0)[3])
            assert str(history.get_row_at(0)[4]) == "Finished"
            expected = "Search completed" if independent_status else "No active downloads"
            assert status.renderable == expected

    asyncio.run(run())


def test_terminal_resize_preserves_selected_model_and_scroll():
    class ResizableViewer(_LayoutOnlyViewer):
        on_resize = AIModelViewer.on_resize
        _apply_resize_reflow = AIModelViewer._apply_resize_reflow

    async def run():
        app = ResizableViewer()
        async with app.run_test(size=(160, 40)) as pilot:
            app.current_filter = "Ollama"
            app.sort_mode = "name"
            app.all_results = [
                _result(model_id=f"model-{index:03d}", name=f"model-{index:03d}", publisher="Test", quality=10)
                for index in range(100)
            ]
            app.refresh_table()
            await pilot.pause(.3)
            table = app.query_one("#results-table", DataTable)
            table.move_cursor(row=35, animate=False, scroll=False)
            table.scroll_to(y=30, animate=False, immediate=True)
            await pilot.pause()
            before = table.cursor_row, table.scroll_y

            await pilot.resize_terminal(100, 40)
            await pilot.pause(.3)

            assert app._get_selected_model()["id"] == "model-035"
            assert (table.cursor_row, table.scroll_y) == before
            assert "download" in app.results_column_keys
            assert table.virtual_size.width <= table.scrollable_content_region.width

    asyncio.run(run())


def _result(
    *,
    model_id: str,
    name: str,
    publisher: str,
    quality: int,
) -> dict:
    return {
        "id": model_id,
        "source": "Ollama",
        "provider": "Ollama",
        "publisher": publisher,
        "name": name,
        "params": "7B",
        "use_case": "Coding",
        "use_case_key": "coding",
        "score": str(quality),
        "score_quality": quality,
        "score_composite": quality,
        "score_speed": quality,
        "estimated_tok_s": float(quality),
        "quant": "Q4_K_M",
        "mode": "GPU",
        "fit": "Perfect",
        "size": "4.0 GB",
        "downloads": 0,
        "likes": 0,
        "gem_score": 0.0,
        "is_hidden_gem": False,
        "inst": "-",
        "download_state": "idle",
    }


def test_same_key_set_reorders_rows_when_sort_mode_changes(monkeypatch):
    monkeypatch.setattr("app.viewer.get_provider_filter_labels", lambda: ("Ollama",))

    async def _run() -> None:
        app = _LayoutOnlyViewer()
        async with app.run_test(size=(160, 40)) as pilot:
            app.current_filter = "Ollama"
            app.use_case_filter = "all"
            app.fit_filter = "all"
            app.hidden_gems_only = False
            app.all_results = [
                _result(model_id="zeta-id", name="zeta", publisher="Zed", quality=99),
                _result(model_id="alpha-id", name="alpha", publisher="Alpha", quality=10),
            ]

            app.sort_mode = "name"
            app.refresh_table()
            await pilot.pause()

            table = app.query_one("#results-table", DataTable)
            assert table.get_row_index("Ollama:alpha-id") < table.get_row_index("Ollama:zeta-id")

            app.sort_mode = "quality"
            app.refresh_table()
            await pilot.pause()

            assert table.get_row_index("Ollama:zeta-id") < table.get_row_index("Ollama:alpha-id")

    asyncio.run(_run())


@pytest.mark.parametrize("width", [90, 160])
def test_download_refresh_only_writes_changed_cells_and_preserves_view(monkeypatch, width):
    async def _run():
        app = _LayoutOnlyViewer()
        async with app.run_test(size=(width, 30)) as pilot:
            app.current_filter = "Ollama"
            app.use_case_filter = app.fit_filter = "all"
            app.hidden_gems_only = False
            app.sort_mode = "name"
            app.all_results = [
                _result(model_id=f"model-{index:03d}", name=f"model-{index:03d}",
                        publisher="Fixture", quality=50) for index in range(100)
            ]
            app.refresh_table()
            await pilot.pause()
            table = app.query_one("#results-table", DataTable)
            table.move_cursor(row=50, animate=False, scroll=False)
            table.scroll_to(y=30, animate=False, immediate=True)
            await pilot.pause()
            before = (table.cursor_row, table.scroll_x, table.scroll_y)
            calls = []
            original = table.update_cell

            def record(*args, **kwargs):
                calls.append(args)
                return original(*args, **kwargs)

            monkeypatch.setattr(table, "update_cell", record)
            app.refresh_table()
            assert calls == []
            app.all_results[0]["download_state"] = "queued"
            app.refresh_table()
            await pilot.pause()
            if "download" in app.results_column_keys:
                assert len(calls) == 1
                assert calls[0][0] == "Ollama:model-000"
                assert calls[0][1] == "download"
            else:
                assert calls == []
            assert (table.cursor_row, table.scroll_x, table.scroll_y) == before
            assert app._get_selected_model()["id"] == "model-050"

    asyncio.run(_run())


def test_same_key_metadata_change_updates_non_download_cells(monkeypatch):
    monkeypatch.setattr("app.viewer.get_provider_filter_labels", lambda: ("Ollama",))

    async def _run() -> None:
        app = _LayoutOnlyViewer()
        async with app.run_test(size=(160, 40)) as pilot:
            app.current_filter = "Ollama"
            app.use_case_filter = "all"
            app.fit_filter = "all"
            app.hidden_gems_only = False
            app.sort_mode = "name"
            app.all_results = [
                _result(model_id="alpha-id", name="alpha", publisher="Alpha", quality=50)
            ]

            app.refresh_table()
            await pilot.pause()

            table = app.query_one("#results-table", DataTable)
            before = str(table.get_cell("Ollama:alpha-id", "score"))
            assert "50" in before

            app.all_results[0]["score"] = "99"
            app.all_results[0]["score_quality"] = 99
            app.all_results[0]["score_composite"] = 99
            app.refresh_table()
            await pilot.pause()

            after = str(table.get_cell("Ollama:alpha-id", "score"))
            assert "99" in after
            assert before != after

    asyncio.run(_run())
