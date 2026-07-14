"""WtypeEmitter command construction — no real wtype/compositor needed."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from wldictate.emitter import NullEmitter, StdoutEmitter, WtypeEmitter, make_emitter


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


def test_make_emitter_env_override(monkeypatch):
    monkeypatch.setenv("WL_DICTATE_EMIT", "null")
    assert isinstance(make_emitter("commit"), NullEmitter)
    monkeypatch.setenv("WL_DICTATE_EMIT", "stdout")
    assert isinstance(make_emitter("commit"), StdoutEmitter)
