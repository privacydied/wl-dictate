"""Toggle client: pokes the tray app over its Unix socket."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

from .config import legacy_socket_paths, socket_path
from .notify import notify


def _candidate_sockets() -> list[Path]:
    return [socket_path(), *legacy_socket_paths()]


def main(verb: str = "toggle") -> int:
    """Send a toggle verb ("toggle" | "toggle-contextual") to the tray."""
    last_error: Exception | None = None
    for path in _candidate_sockets():
        if not os.path.exists(path):
            continue
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(1.0)
            sock.connect(str(path))
            sock.sendall(verb.encode())
            return 0
        except OSError as e:
            last_error = e
        finally:
            sock.close()
    if last_error is not None:
        notify(f"Dictation toggle failed: {last_error}", title="Dictation Error")
    else:
        notify("Tray app is not running", title="Dictation")
    return 1


if __name__ == "__main__":
    sys.exit(main())
