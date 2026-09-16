from __future__ import annotations

import asyncio

from app.viewer import AIModelViewer


class _LayoutOnlyViewer(AIModelViewer):
    def _smoke_mode_enabled(self) -> bool:
        return True

    def _finish_smoke_mode(self) -> None:
        pass


def test_runtime_filter_controls_stay_inside_80_column_viewport(monkeypatch):
    monkeypatch.setattr(
        "app.viewer.get_provider_filter_labels",
        lambda: ("Ollama", "Hugging Face"),
    )

    async def _run() -> None:
        app = _LayoutOnlyViewer()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()

            viewport_width = app.size.width
            row = app.query_one("#search-filters-row")
            search_input = app.query_one("#search-input")
            provider_select = app.query_one("#provider-select")
            use_case_select = app.query_one("#use-case-select")

            for widget in (row, search_input, provider_select, use_case_select):
                assert widget.region.x >= 0
                assert widget.region.x + widget.region.width <= viewport_width

            assert str(use_case_select.value) == "all"

            results_table = app.query_one("#results-table")
            results_table.focus()
            await pilot.pause()
            await pilot.press("u")
            await pilot.pause()

            assert app.use_case_filter == "chat"
            assert str(use_case_select.value) == "chat"

    asyncio.run(_run())
