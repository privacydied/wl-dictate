"""WtypeEmitter command construction — no real wtype/compositor needed."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from wldictate.emitter import (
    CorrectingEmitter,
    NullEmitter,
    StdoutEmitter,
    WtypeEmitter,
    make_emitter,
)


class _Capture:
    """Stand-in for subprocess.run that records the call and reports success."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_wtype_passes_delays_and_uses_stdin(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    WtypeEmitter(delay_ms=6, press_delay_ms=40).emit("hello world.")
    cmd, kwargs = cap.calls[0]
    # -s (pre-type settle) before -d (inter-key), text on stdin.
    assert cmd == ["wtype", "-s", "40", "-d", "6", "-"]
    # Text goes on stdin, not argv — so a leading "-" can't be read as a flag.
    assert kwargs["input"] == "hello world."


def test_wtype_zero_delays_omit_flags(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    WtypeEmitter(delay_ms=0, press_delay_ms=0).emit("hi")
    cmd, _ = cap.calls[0]
    assert cmd == ["wtype", "-"]


def test_wtype_press_delay_only(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    WtypeEmitter(delay_ms=0, press_delay_ms=40).emit("hi")
    cmd, _ = cap.calls[0]
    assert cmd == ["wtype", "-s", "40", "-"]


def test_wtype_leading_dash_text_is_safe(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    ok = WtypeEmitter(delay_ms=6).emit("-- dashes")
    assert ok is True
    _, kwargs = cap.calls[0]
    assert kwargs["input"] == "-- dashes"


def test_wtype_empty_text_is_noop(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    assert WtypeEmitter().emit("") is True
    assert cap.calls == []  # never shells out for empty text


def test_wtype_nonzero_exit_reports_failure(monkeypatch):
    def fail(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fail)
    assert WtypeEmitter().emit("x") is False


def test_make_emitter_threads_delays(monkeypatch):
    monkeypatch.delenv("WL_DICTATE_EMIT", raising=False)
    e = make_emitter(
        "commit", wtype_timeout_s=5.0, wtype_delay_ms=12, wtype_press_delay_ms=50
    )
    assert isinstance(e, WtypeEmitter)
    assert e._delay_ms == 12
    assert e._press_delay_ms == 50


def test_text_passes_through_verbatim_no_zwsp(monkeypatch):
    # The old Electron ZWSP "space gate" is gone: dropped spaces were really
    # caused by synthetic keys at made-up scancodes (fixed in vkbd). Text
    # must reach the device exactly as given — no invisible characters.
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    e = WtypeEmitter(delay_ms=0)
    monkeypatch.setattr(e, "_focused_window_class", lambda: "vesktop")
    e.emit(" Testing.")
    e.emit(" ")
    assert cap.calls[0][1]["input"] == " Testing."
    assert cap.calls[1][1]["input"] == " "


def test_make_emitter_env_override(monkeypatch):
    monkeypatch.setenv("WL_DICTATE_EMIT", "null")
    assert isinstance(make_emitter("commit"), NullEmitter)
    monkeypatch.setenv("WL_DICTATE_EMIT", "stdout")
    assert isinstance(make_emitter("commit"), StdoutEmitter)


def test_make_emitter_correcting_wraps_device(monkeypatch):
    monkeypatch.delenv("WL_DICTATE_EMIT", raising=False)
    e = make_emitter("correcting", wtype_delay_ms=12)
    assert isinstance(e, CorrectingEmitter)
    assert isinstance(e._device, WtypeEmitter)
    assert e._device._delay_ms == 12
    # Env override selects the *device*; the mode still wraps it.
    monkeypatch.setenv("WL_DICTATE_EMIT", "stdout")
    e = make_emitter("correcting")
    assert isinstance(e, CorrectingEmitter)
    assert isinstance(e._device, StdoutEmitter)


# ── rewrite (backspace + retype in one wtype invocation) ─────────────────────


def test_rewrite_argv_with_delay(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    assert WtypeEmitter(delay_ms=6).rewrite(3, "abc") == "abc"
    cmd, kwargs = cap.calls[0]
    # Keys are paced by interleaved -s (wtype's -d paces only text typing);
    # stdin text ("-") comes last so the retype lands after the backspaces.
    assert cmd == [
        "wtype", "-d", "6",
        "-s", "6", "-k", "BackSpace",
        "-s", "6", "-k", "BackSpace",
        "-s", "6", "-k", "BackSpace",
        "-",
    ]  # fmt: skip
    assert kwargs["input"] == "abc"


def test_rewrite_argv_without_delay(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    WtypeEmitter(delay_ms=0).rewrite(2, "x")
    cmd, _ = cap.calls[0]
    assert cmd == ["wtype", "-k", "BackSpace", "-k", "BackSpace", "-"]


def test_rewrite_pure_deletion_omits_stdin_placeholder(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    assert WtypeEmitter(delay_ms=0).rewrite(2, "") == ""
    cmd, _ = cap.calls[0]
    assert cmd == ["wtype", "-k", "BackSpace", "-k", "BackSpace"]


def test_rewrite_noop_never_shells_out(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    assert WtypeEmitter().rewrite(0, "") == ""
    assert cap.calls == []


def test_rewrite_passes_text_verbatim_with_and_without_backspaces(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    e = WtypeEmitter(delay_ms=0)
    monkeypatch.setattr(e, "_focused_window_class", lambda: "vesktop")
    assert e.rewrite(0, " x") == " x"
    assert e.rewrite(2, " y") == " y"
    _, kwargs = cap.calls[1]
    assert kwargs["input"] == " y"


def test_rewrite_failure_returns_none(monkeypatch):
    def fail(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fail)
    assert WtypeEmitter().rewrite(1, "x") is None


def test_emit_refactor_keeps_argv_and_reports_failure(monkeypatch):
    # emit() now routes through rewrite(0, text); argv must be unchanged.
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    assert WtypeEmitter(delay_ms=6, press_delay_ms=40).emit("hi") is True
    cmd, _ = cap.calls[0]
    assert cmd == ["wtype", "-s", "40", "-d", "6", "-"]


# ── paste fast-path (rewrite_bulk) ───────────────────────────────────────────

LONG = "x" * 200  # above _PASTE_MIN_CHARS


def test_bulk_paste_in_electron_uses_clipboard_and_ctrl_v(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    e = WtypeEmitter(delay_ms=6)
    monkeypatch.setattr(e, "_focused_window_class", lambda: "vesktop")
    assert e.rewrite_bulk(2, LONG) == LONG
    cmds = [c for c, _ in cap.calls]
    # wl-paste (save), wl-copy (set), wtype (backspaces + Ctrl+V)
    assert cmds[0][0] == "wl-paste"
    assert cmds[1][0] == "wl-copy"
    wtype_cmd, wtype_kwargs = cap.calls[2]
    assert wtype_cmd[0] == "wtype"
    assert wtype_cmd.count("BackSpace") == 2
    assert wtype_cmd[-6:] == ["-M", "ctrl", "-k", "v", "-m", "ctrl"]
    assert "-" not in wtype_cmd  # no keystroked text: it goes via clipboard
    _, copy_kwargs = cap.calls[1]
    assert copy_kwargs["input"] == LONG


def test_bulk_falls_back_to_typing_outside_electron(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    e = WtypeEmitter(delay_ms=0)
    monkeypatch.setattr(e, "_focused_window_class", lambda: "kitty")
    assert e.rewrite_bulk(1, LONG) == LONG
    cmds = [c for c, _ in cap.calls]
    assert all(c[0] == "wtype" for c in cmds)  # plain rewrite path


def test_bulk_short_text_always_types(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(subprocess, "run", cap)
    e = WtypeEmitter(delay_ms=0)
    monkeypatch.setattr(e, "_focused_window_class", lambda: "vesktop")
    e.rewrite_bulk(0, "short")
    assert all(c[0] == "wtype" for c, _ in cap.calls)
