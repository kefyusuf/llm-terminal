from __future__ import annotations

import asyncio

from tui_app import AIModelViewer


class _LayoutOnlyViewer(AIModelViewer):
    def on_mount(self) -> None:
        """Skip services/timers; this test exercises only the real composed layout."""
        pass


def test_filter_controls_stay_inside_80_column_viewport():
    async def _run() -> None:
        app = _LayoutOnlyViewer()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()

            viewport_width = app.size.width
            row = app.query_one("#search-filters-row")
            search_input = app.query_one("#search-input")
            provider_set = app.query_one("#filter-set")
            use_case_set = app.query_one("#use-case-filter")
            last_use_case = app.query_one("#uc-general")

            for widget in (row, search_input, provider_set, use_case_set, last_use_case):
                assert widget.region.x >= 0
                assert widget.region.x + widget.region.width <= viewport_width

    asyncio.run(_run())
