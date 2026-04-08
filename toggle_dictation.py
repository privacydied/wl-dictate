#!/usr/bin/env python3
"""
Send a toggle command to the dictation tray app via Unix socket.
Use this to bind Ctrl+Alt+F (or any key) in your compositor settings.

Usage:
    python toggle_dictation.py
    # or make it executable and point your keybinding to the full path
"""

import os
import socket
import sys

def _find_socket():
    """Find .dictation.sock — check binary dir first, then script dir."""
    binary_dir = os.path.dirname(os.path.abspath(sys.executable))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for d in (binary_dir, script_dir):
        path = os.path.join(d, ".dictation.sock")
        if os.path.exists(path):
            return path
    # Default: next to binary
    return os.path.join(binary_dir, ".dictation.sock")

SOCKET_PATH = _find_socket()


def main():
    if not os.path.exists(SOCKET_PATH):
        # Tray app not running — notify the user via system notification
        try:
            import subprocess

            subprocess.run(
                ["notify-send", "-t", "3000", "Dictation", "Tray app is not running"],
                timeout=3,
            )
        except Exception:
            pass
        sys.exit(1)

    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect(SOCKET_PATH)
        sock.sendall(b"toggle")
    except (ConnectionRefusedError, OSError) as e:
        try:
            import subprocess

            subprocess.run(
                ["notify-send", "-t", "3000", "Dictation Error", str(e)],
                timeout=3,
            )
        except Exception:
            pass
        sys.exit(1)
    finally:
        if sock:
            sock.close()


if __name__ == "__main__":
    main()
