from __future__ import annotations

import asyncio

import pytest
from textual.app import App
from textual.widgets import Button, Label

from app.responsive_modals import ModelDetailModal


class _HostApp(App):
    def _set_modal_poll_pause(self, _enabled: bool) -> None:
        pass


@pytest.mark.parametrize("viewport", [(80, 24), (120, 30)])
def test_blocked_download_explains_insufficient_disk_without_scrolling(viewport):
    data = {
        "source": "Hugging Face", "id": "fixture/repo", "name": "large-model",
        "target_file": "large.gguf", "size": "340.6 GB", "fit": "No Fit",
        "download_plan": {"allowed": False, "status": "insufficient_disk",
                          "free_bytes": 152788611072, "required_bytes": 365780795392},
    }

    async def run():
        app = _HostApp()
        async with app.run_test(size=viewport) as pilot:
            modal = ModelDetailModal(data)
            app.push_screen(modal)
            await pilot.pause()
            assert modal.query_one("#download-btn", Button).disabled
            notice = modal.query_one("#download-block-reason", Label)
            assert "Insufficient disk space" in str(notice.renderable)
            assert "340.7 GiB" in str(notice.renderable)
            assert "142.3 GiB" in str(notice.renderable)
            container = modal.query_one("#modal-container")
            for scroll_to_end in [False, True]:
                if scroll_to_end:
                    container.scroll_end(animate=False)
                    await pilot.pause()
                assert notice.region.y >= container.region.y
                assert notice.region.bottom <= modal.query_one("#button-row").region.y
                assert notice.region.bottom <= app.screen.size.height
                x, y = notice.region.x, notice.region.y
                assert app.screen.get_widget_at(x, y)[0] is notice

    asyncio.run(run())


@pytest.mark.parametrize("status, message", [
    ("service_unavailable", "Download plan unavailable"),
    ("disk_unavailable", "Cannot check free disk space"),
    ("unsafe_destination", "Download destination is unsafe or unavailable"),
])
def test_blocked_download_has_actionable_plan_feedback(status, message):
    async def run():
        app = _HostApp()
        async with app.run_test(size=(80, 24)) as pilot:
            modal = ModelDetailModal({
                "source": "Hugging Face", "id": "fixture/repo", "name": "model",
                "target_file": "model.gguf", "download_plan": {"allowed": False, "status": status},
            })
            app.push_screen(modal)
            await pilot.pause()
            assert modal.query_one("#download-btn", Button).disabled
            notice = modal.query_one("#download-block-reason", Label)
            assert message in str(notice.renderable)
            assert notice.region.bottom <= modal.query_one("#button-row").region.y

    asyncio.run(run())


def test_hardware_no_fit_does_not_block_an_allowed_download_plan():
    async def run():
        app = _HostApp()
        async with app.run_test(size=(80, 24)) as pilot:
            modal = ModelDetailModal({
                "source": "Hugging Face", "id": "fixture/repo", "name": "model",
                "target_file": "model.gguf", "fit": "No Fit",
                "download_plan": {"allowed": True, "status": "ready"},
            })
            app.push_screen(modal)
            await pilot.pause()
            assert not modal.query_one("#download-btn", Button).disabled
            assert not modal.query("#download-block-reason")

    asyncio.run(run())


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
