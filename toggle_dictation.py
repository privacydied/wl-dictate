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

SOCKET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dictation.sock")


def main():
    if not os.path.exists(SOCKET_PATH):
        print("Dictation tray app is not running (socket not found).", file=sys.stderr)
        sys.exit(1)

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect(SOCKET_PATH)
        sock.sendall(b"toggle")
        sock.close()
    except (ConnectionRefusedError, OSError) as e:
        print(f"Failed to connect to dictation tray: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
