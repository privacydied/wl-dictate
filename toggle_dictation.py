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

# Resolve the socket path relative to this script file, not CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOCKET_PATH = os.path.join(SCRIPT_DIR, ".dictation.sock")


def main():
    if not os.path.exists(SOCKET_PATH):
        # Tray app not running — notify the user via system notification
        try:
            import subprocess

            subprocess.run(
                ["notify-send", "Dictation", "Tray app is not running"],
                timeout=5,
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

            subprocess.run(["notify-send", "Dictation Error", str(e)], timeout=5)
        except Exception:
            pass
        sys.exit(1)
    finally:
        if sock:
            sock.close()


if __name__ == "__main__":
    main()
