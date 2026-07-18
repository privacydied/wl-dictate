"""Hyprland focused-window tracking via the event socket (no subprocesses).

The Electron gate and the contextual-transform context capture both need
the focused window's class/title. Shelling out to ``hyprctl activewindow``
per rewrite (subprocess spawn, up to a 1 s timeout) on the typing hot path
is exactly the kind of per-keystroke cost this pipeline avoids elsewhere.

``FocusTracker`` subscribes ONCE to Hyprland's event socket (socket2) and
keeps the active window cached; reads are a lock-guarded tuple copy. The
initial state (and a refresh after every reconnect) comes from the request
socket (socket1, the same IPC ``hyprctl`` uses) — still no subprocess.

Non-Hyprland sessions get ``None`` from :func:`get_focus_tracker`; callers
fall back to their previous behavior (a TTL-cached ``hyprctl`` call, see
``emitter.focused_window``).
"""

from __future__ import annotations

import json
import os
import socket
import threading


def instance_dir(env: dict[str, str]) -> str | None:
    """Directory holding Hyprland's IPC sockets, or None when not Hyprland."""
    sig = env.get("HYPRLAND_INSTANCE_SIGNATURE", "")
    if not sig:
        return None
    runtime = env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    for base in (os.path.join(runtime, "hypr"), "/tmp/hypr"):
        d = os.path.join(base, sig)
        if os.path.isdir(d):
            return d
    return None


def parse_activewindow_event(line: str) -> tuple[str, str] | None:
    """``activewindow>>CLASS,TITLE`` -> (class, title); None for other events.

    The class never contains a comma; the title may — split once. An empty
    payload (``activewindow>>,``) means nothing is focused.
    """
    if not line.startswith("activewindow>>"):
        return None
    payload = line[len("activewindow>>") :]
    cls, _, title = payload.partition(",")
    return cls, title


class FocusTracker:
    """Cached (class, title) of the focused window, fed by socket2 events."""

    def __init__(self, socket2_path: str, socket1_path: str) -> None:
        self._socket2_path = socket2_path
        self._socket1_path = socket1_path
        self._lock = threading.Lock()
        self._focused: tuple[str, str] = ("", "")
        self._connected = False
        self._stop = False
        self._sock: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="hypr-events"
        )
        self._thread.start()

    @property
    def ok(self) -> bool:
        """True while the event subscription is live (cache is trustworthy)."""
        return self._connected

    def focused(self) -> tuple[str, str]:
        with self._lock:
            return self._focused

    def close(self) -> None:
        self._stop = True
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    # ── Internals ────────────────────────────────────────────────────────

    def _set(self, cls: str, title: str) -> None:
        with self._lock:
            self._focused = (cls, title)

    def _refresh(self) -> None:
        """One-shot ``j/activewindow`` request over socket1 (hyprctl's IPC)."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(self._socket1_path)
                s.sendall(b"j/activewindow")
                chunks = []
                while True:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
            data = json.loads(b"".join(chunks) or b"{}")
            cls = str(data.get("class") or data.get("initialClass") or "")
            title = str(data.get("title") or "")
            self._set(cls, title)
        except Exception:
            pass  # events will correct the cache on the next focus change

    def _loop(self) -> None:
        backoff = 0.5
        while not self._stop:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(self._socket2_path)
                self._sock = sock
            except OSError:
                self._sock = None
                if self._stop:
                    return
                self._stop_wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            self._connected = True
            backoff = 0.5
            self._refresh()  # events only fire on *changes*: seed the cache
            try:
                buf = b""
                while not self._stop:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for raw in lines:
                        parsed = parse_activewindow_event(
                            raw.decode("utf-8", "replace")
                        )
                        if parsed is not None:
                            self._set(*parsed)
            except OSError:
                pass
            finally:
                self._connected = False
                try:
                    sock.close()
                except OSError:
                    pass
                self._sock = None

    def _stop_wait(self, seconds: float) -> None:
        # Plain sleep chunked so close() is honored promptly.
        import time

        deadline = time.monotonic() + seconds
        while not self._stop and time.monotonic() < deadline:
            time.sleep(0.05)


# ── Process-wide singleton ───────────────────────────────────────────────────

_lock = threading.Lock()
_tracker: FocusTracker | None = None
_tracker_failed = False


def get_focus_tracker(env: dict[str, str]) -> FocusTracker | None:
    """Shared tracker for this process; None when not running under Hyprland."""
    global _tracker, _tracker_failed
    with _lock:
        if _tracker is not None:
            return _tracker
        if _tracker_failed:
            return None
        d = instance_dir(env)
        if d is None:
            _tracker_failed = True
            return None
        _tracker = FocusTracker(
            os.path.join(d, ".socket2.sock"), os.path.join(d, ".socket.sock")
        )
        return _tracker
