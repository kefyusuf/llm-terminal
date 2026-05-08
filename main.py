"""Application entry point for AI Model Explorer."""

import asyncio
import os

from app import AIModelViewer


def smoke_mode_enabled() -> bool:
    return os.getenv("AIMODEL_SMOKE") == "1"


def run_smoke_check() -> int:
    async def _run() -> int:
        app = AIModelViewer()
        async with app.run_test() as pilot:
            await pilot.pause(1.5)
        return 0 if app.return_code in (None, 0) else app.return_code

    return asyncio.run(_run())


def main() -> int:
    """Run the main Textual application."""
    if smoke_mode_enabled():
        return run_smoke_check()
    AIModelViewer().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
