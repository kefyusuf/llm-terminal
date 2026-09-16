from __future__ import annotations

import asyncio

from textual.widgets import DataTable

from tui_app import AIModelViewer


class _LayoutOnlyViewer(AIModelViewer):
    def on_mount(self) -> None:
        """Skip service/timer side effects; exercise the real results table only."""
        pass


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


def test_same_key_set_reorders_rows_when_sort_mode_changes():
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


def test_same_key_metadata_change_updates_non_download_cells():
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
