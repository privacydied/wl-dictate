"""Non-blocking desktop notifications.

The old code called ``subprocess.run(["notify-send", ...], timeout=3)`` on the
Qt main thread, stalling the UI for up to 3 seconds per toast. This helper
fires and forgets.
"""

from __future__ import annotations

import subprocess

_TITLE = "Dictation Tool"


def notify(message: str, *, title: str = _TITLE, timeout_ms: int = 3000) -> None:
    try:
        subprocess.Popen(
            ["notify-send", "-t", str(timeout_ms), title, message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        pass  # notifications are best-effort, never fatal
