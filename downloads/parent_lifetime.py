"""Bind an owned HF child to its parent's private stdin pipe lifetime."""
from __future__ import annotations

import os
import sys
import threading


def start_parent_watchdog() -> None:
    """Exit the child on pipe EOF/error; the parent keeps its writer open."""
    descriptor = sys.stdin.fileno()

    def watch() -> None:
        try:
            # Avoid a buffered stdin lock held across interpreter shutdown.
            while os.read(descriptor, 1):
                pass
        except (OSError, ValueError):
            pass
        # SDK work can be blocked in native/network I/O; do not wait for it.
        os._exit(75)

    threading.Thread(target=watch, name="hf-parent-lifetime", daemon=True).start()
