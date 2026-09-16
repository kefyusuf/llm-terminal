from __future__ import annotations

import asyncio

from textual.app import App

from app.responsive_modals import ModelDetailModal


class _HostApp(App):
    def _set_modal_poll_pause(self, _enabled: bool) -> None:
        pass


def test_model_detail_keeps_actions_visible_at_80x24():
    """The detail modal must keep its action row inside a standard 80x24 viewport."""
    data = {
        "source": "Ollama",
        "name": "qwen2.5-coder:32b-instruct-q4_K_M",
        "publisher": "Qwen",
        "provider": "Ollama",
        "use_case": "coding",
        "params": "32B",
        "quant": "Q4_K_M",
        "score": "92",
        "size": "19.8 GB",
        "size_source": "exact",
        "fit": "Partial",
        "mode": "CPU+GPU",
        "download_state": "idle",
    }

    async def _run() -> None:
        app = _HostApp()
        modal = ModelDetailModal(data)

        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(modal)
            await pilot.pause()

            container = modal.query_one("#modal-container")
            button_row = modal.query_one("#button-row")
            close_button = modal.query_one("#close-btn")

            viewport_height = app.size.height
            assert container.region.y >= 0
            assert container.region.y + container.region.height <= viewport_height
            assert button_row.region.y >= 0
            assert button_row.region.y + button_row.region.height <= viewport_height
            assert close_button.region.y >= 0
            assert close_button.region.y + close_button.region.height <= viewport_height

    asyncio.run(_run())
