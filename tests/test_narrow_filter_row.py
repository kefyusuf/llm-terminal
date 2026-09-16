from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RadioButton, RadioSet


class _FilterRowApp(App):
    CSS = """
    Screen { padding: 1; }
    #search-filters-row { height: 3; }
    #search-panel { width: 34%; min-width: 28; margin-right: 1; height: 3; }
    #provider-panel { width: 22; margin-right: 1; height: 3; }
    #use-case-panel { width: 1fr; height: 3; }
    Input { width: 100%; height: 3; }
    RadioSet {
        layout: horizontal;
        width: 100%;
        height: 3;
        border: round;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="search-filters-row"):
            with Vertical(id="search-panel"):
                yield Input(id="search-input")
            with Vertical(id="provider-panel"), RadioSet(id="filter-set"):
                yield RadioButton("Ollama", value=True, id="filter-ollama")
                yield RadioButton("Hugging Face", id="filter-hf")
            with Vertical(id="use-case-panel"), RadioSet(id="use-case-filter"):
                yield RadioButton("Any Use", value=True, id="uc-all")
                yield RadioButton("Chat", id="uc-chat")
                yield RadioButton("Coding", id="uc-coding")
                yield RadioButton("Vision", id="uc-vision")
                yield RadioButton("Reason", id="uc-reasoning")
                yield RadioButton("Math", id="uc-math")
                yield RadioButton("Embed", id="uc-embedding")
                yield RadioButton("General", id="uc-general")


def test_filter_controls_stay_inside_80_column_viewport():
    async def _run() -> None:
        app = _FilterRowApp()
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
