"""Visible estimate explanations in the real comparison modal."""

import asyncio

from textual.app import App
from textual.widgets import Label

from app.modals import ComparisonModal


def test_comparison_discloses_estimates_and_legacy_unknown_basis():
    asyncio.run(comparison_scenario())


async def comparison_scenario():
    app = App()
    async with app.run_test() as pilot:
        app.push_screen(
            ComparisonModal(
                [
                    {
                        "name": "current",
                        "score_provenance": {"speed": {"bandwidth_source": "backend_default"}},
                    },
                    {"name": "legacy"},
                ]
            )
        )
        await pilot.pause()
        text = "\n".join(str(label.renderable) for label in app.screen.query(Label))
        assert "Heuristic estimates" in text
        assert "not measured context capacity" in text
        assert "backend_default" in text
        assert "Unknown" in text
        notice = app.screen.query_one("#comparison-estimates", Label)
        assert notice.region.y >= 0
        assert notice.region.bottom <= app.screen.size.height
