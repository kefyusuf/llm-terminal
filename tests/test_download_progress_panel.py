"""Progress presentation keeps unknown and finalizing states honest."""

import asyncio

from textual.app import App
from textual.widgets import Button, ProgressBar, Static

from app.download_progress import DownloadProgressPanel


def test_unknown_metrics_then_verification_then_terminal_clear_transfer_values():
    async def run():
        app = App()
        async with app.run_test(size=(80, 24)) as pilot:
            panel = DownloadProgressPanel(id='fixture')
            await app.mount(panel)
            entry = {'target_id': 'ollama:fixture', 'source': 'Ollama', 'name': 'fixture',
                     'state': 'downloading', 'detail': 'pulling manifest'}
            panel.show_entry(entry)
            await pilot.pause()
            summary = panel.query_one(Static)
            assert 'Preparing' in str(summary.renderable)
            assert 'calculating' in str(summary.renderable)
            assert panel.query_one(ProgressBar).total is None

            panel.show_entry({**entry, 'detail': 'pulling abcd 100% 1 GB/1 GB 10 MB/s 0s'})
            panel.show_entry({**entry, 'detail': 'verifying sha256 digest'})
            assert 'Verifying' in str(summary.renderable)
            assert '100%' not in str(summary.renderable)
            assert '10 MB/s' not in str(summary.renderable)
            assert panel.query_one(ProgressBar).total is None
            assert not panel.query_one(Button).disabled

            panel.show_entry({**entry, 'state': 'completed', 'label': 'Done', 'detail': 'Completed'})
            assert 'Completed' in str(summary.renderable)
            assert 'calculating' not in str(summary.renderable)
            assert panel.query_one(Button).disabled
            assert not panel.query_one(ProgressBar).display

    asyncio.run(run())
