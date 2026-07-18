"""Hyprland focus tracker: event parsing, socket feed, fallback wiring."""

import json
import os
import socket
import threading
import time

import wldictate.hypr as hypr
from wldictate.hypr import FocusTracker, parse_activewindow_event


def test_parse_activewindow_event():
    assert parse_activewindow_event("activewindow>>kitty,~ — zsh") == (
        "kitty",
        "~ — zsh",
    )
    # Title may contain commas; class never does.
    assert parse_activewindow_event("activewindow>>vesktop,#dev, general") == (
        "vesktop",
        "#dev, general",
    )
    assert parse_activewindow_event("activewindow>>,") == ("", "")
    assert parse_activewindow_event("workspace>>3") is None


class FakeHyprland:
    """Serves .socket.sock (one-shot JSON requests) + .socket2.sock (events)."""

    def __init__(self, tmpdir, active=None):
        self.active = active or {"class": "kitty", "title": "boot"}
        self.s1_path = os.path.join(tmpdir, ".socket.sock")
        self.s2_path = os.path.join(tmpdir, ".socket2.sock")
        self._s1 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._s1.bind(self.s1_path)
        self._s1.listen(4)
        self._s2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._s2.bind(self.s2_path)
        self._s2.listen(4)
        self.event_conn = None
        self._event_ready = threading.Event()
        threading.Thread(target=self._serve_s1, daemon=True).start()
        threading.Thread(target=self._serve_s2, daemon=True).start()

    def _serve_s1(self):
        while True:
            try:
                conn, _ = self._s1.accept()
            except OSError:
                return
            conn.recv(4096)
            conn.sendall(json.dumps(self.active).encode())
            conn.close()

    def _serve_s2(self):
        while True:
            try:
                conn, _ = self._s2.accept()
            except OSError:
                return
            self.event_conn = conn
            self._event_ready.set()

    def send_event(self, line: str):
        assert self._event_ready.wait(timeout=5.0)
        self.event_conn.sendall(line.encode() + b"\n")

    def close(self):
        for s in (self._s1, self._s2):
            try:
                s.close()
            except OSError:
                pass


def _wait(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_tracker_seeds_from_socket1_and_follows_events(tmp_path):
    server = FakeHyprland(str(tmp_path), active={"class": "kitty", "title": "t0"})
    tracker = FocusTracker(server.s2_path, server.s1_path)
    try:
        assert _wait(lambda: tracker.focused() == ("kitty", "t0"))
        assert tracker.ok
        server.send_event("activewindow>>vesktop,#dev — Discord")
        assert _wait(lambda: tracker.focused() == ("vesktop", "#dev — Discord"))
        server.send_event("workspace>>2")  # unrelated events ignored
        server.send_event("activewindow>>,")  # focus lost
        assert _wait(lambda: tracker.focused() == ("", ""))
    finally:
        tracker.close()
        server.close()


def test_tracker_marks_stale_on_disconnect(tmp_path):
    server = FakeHyprland(str(tmp_path))
    tracker = FocusTracker(server.s2_path, server.s1_path)
    try:
        assert server._event_ready.wait(timeout=5.0)
        assert _wait(lambda: tracker.ok)
        # Stop the listener first so the tracker cannot instantly reconnect,
        # then drop the live event connection.
        server.close()
        os.unlink(server.s2_path)
        server.event_conn.close()
        assert _wait(lambda: not tracker.ok)
    finally:
        tracker.close()
        server.close()


def test_get_focus_tracker_none_outside_hyprland(monkeypatch):
    monkeypatch.setattr(hypr, "_tracker", None)
    monkeypatch.setattr(hypr, "_tracker_failed", False)
    assert hypr.get_focus_tracker({"XDG_RUNTIME_DIR": "/nonexistent"}) is None
    assert hypr._tracker_failed  # cached: no re-probing per call


def test_focused_window_prefers_tracker(monkeypatch):
    import wldictate.emitter as em

    class T:
        ok = True

        def focused(self):
            return ("vesktop", "chat")

    monkeypatch.setattr(hypr, "get_focus_tracker", lambda env: T())

    def no_spawn(*a, **k):
        raise AssertionError("subprocess fallback must not run")

    monkeypatch.setattr(em, "_focused_window_subprocess", no_spawn)
    assert em.focused_window({}) == ("vesktop", "chat")


def test_focused_window_fallback_is_ttl_cached(monkeypatch):
    import wldictate.emitter as em

    monkeypatch.setattr(hypr, "get_focus_tracker", lambda env: None)
    monkeypatch.setattr(em, "_fallback_at", 0.0)
    calls = []

    def fake_sub(env):
        calls.append(1)
        return ("kitty", "x")

    monkeypatch.setattr(em, "_focused_window_subprocess", fake_sub)
    assert em.focused_window({}) == ("kitty", "x")
    assert em.focused_window({}) == ("kitty", "x")
    assert len(calls) == 1  # second call served from the TTL cache
