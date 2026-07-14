"""Unified entry point for wl-dictate.

Usage (source):
    python wl_dictate.py              # tray app
    python wl_dictate.py --worker     # dictation worker (spawned by the tray)
    python wl_dictate.py --toggle     # toggle dictation via Unix socket
    python wl_dictate.py --devices    # list input devices

Usage (compiled binary):
    wl-dictate [--worker|--toggle|--devices]
"""

from __future__ import annotations

import os
import sys

# System library dirs that must never be allowed to shadow the Python wheels'
# bundled libraries (RUNPATH loses to LD_LIBRARY_PATH). A stray
# "/usr/lib" entry there makes the PyQt5 wheel load the system Qt's
# libQt5WaylandClient — a private-ABI mismatch that kills the tray at startup.
_LD_BLOCKLIST = {"", ".", "/usr/lib", "/usr/lib64", "/lib", "/lib64"}


def _sanitize_ld_library_path() -> None:
    """Strip default/system dirs from LD_LIBRARY_PATH, re-execing if needed.

    The dynamic loader caches LD_LIBRARY_PATH at process start, so mutating
    os.environ is not enough — a clean re-exec is required.
    """
    raw = os.environ.get("LD_LIBRARY_PATH")
    if raw is None or os.environ.get("_WL_DICTATE_REEXEC") == "1":
        return
    parts = raw.split(":")
    cleaned = [p for p in parts if os.path.normpath(p) not in _LD_BLOCKLIST]
    if cleaned == parts:
        return
    env = os.environ.copy()
    env["_WL_DICTATE_REEXEC"] = "1"
    if cleaned:
        env["LD_LIBRARY_PATH"] = ":".join(cleaned)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    compiled = "__compiled__" in globals() or getattr(sys, "frozen", False)
    argv = list(sys.argv) if compiled else [sys.executable, *sys.argv]
    try:
        os.execve(sys.executable, argv, env)
    except OSError:
        pass  # fall through and hope for the best


def main() -> None:
    _sanitize_ld_library_path()
    args = sys.argv[1:]

    if args and args[0] == "--worker":
        from wldictate.worker import run

        sys.exit(run())

    elif args and args[0] == "--toggle":
        from wldictate.toggle import main as toggle_main

        sys.exit(toggle_main())

    elif args and args[0] == "--devices":
        from wldictate.audio import list_input_devices

        print("Available input devices:")
        for idx, name, sr in list_input_devices():
            sr_str = str(sr) if sr > 0 else "??"
            print(f"  [{idx}] {name} -- {sr_str} Hz")

    else:
        from wldictate.tray import DictationTrayApp

        DictationTrayApp().run()


if __name__ == "__main__":
    main()
