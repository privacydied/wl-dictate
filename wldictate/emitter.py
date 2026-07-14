"""Text emitters — deliver committed text to the focused window.

``WtypeEmitter`` is the production emitter (append-only typing via wtype).
The abstraction exists so alternative sinks (stdout for debugging, a future
backspace-correcting emitter) can be swapped in without touching the
streaming engine.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from abc import ABC, abstractmethod


class Emitter(ABC):
    @abstractmethod
    def emit(self, text: str) -> bool:
        """Deliver text; returns True on success."""

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class NullEmitter(Emitter):
    """Discards text (used when typing is disabled, e.g. benchmarks)."""

    def emit(self, text: str) -> bool:
        return True


class StdoutEmitter(Emitter):
    """Prints text to stderr instead of typing (debugging/verification).

    stderr, not stdout: worker stdout is the JSON IPC channel.
    """

    def emit(self, text: str) -> bool:
        print(f"[emit] {text!r}", file=sys.stderr, flush=True)
        return True


def _guess_wayland_env(env: dict[str, str]) -> None:
    """Best-effort fill of WAYLAND_DISPLAY/XDG_RUNTIME_DIR from runtime dirs."""
    if env.get("WAYLAND_DISPLAY") and env.get("XDG_RUNTIME_DIR"):
        return
    candidates = []
    runtime = env.get("XDG_RUNTIME_DIR")
    if runtime:
        candidates.append(runtime)
    try:
        candidates.append(f"/run/user/{os.getuid()}")
    except Exception:
        pass
    for candidate in candidates:
        try:
            sockets = glob.glob(os.path.join(candidate, "wayland-*"))
        except OSError:
            continue
        sockets = [s for s in sockets if not s.endswith(".lock")]
        if sockets:
            env["XDG_RUNTIME_DIR"] = candidate
            env["WAYLAND_DISPLAY"] = os.path.basename(sockets[0])
            return


class WtypeEmitter(Emitter):
    """Types text into the focused window with wtype (append-only)."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        self._timeout = timeout_s
        self._env = os.environ.copy()
        _guess_wayland_env(self._env)

    def emit(self, text: str) -> bool:
        if not text:
            return True
        try:
            result = subprocess.run(
                ["wtype", text],
                env=self._env,
                timeout=self._timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            print("wtype timed out", file=sys.stderr)
            return False
        except FileNotFoundError:
            print("wtype is not installed", file=sys.stderr)
            return False
        except OSError as e:
            print(f"wtype failed to launch: {e}", file=sys.stderr)
            return False
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or f"exit code {result.returncode}"
            print(f"wtype failed: {detail}", file=sys.stderr)
            return False
        return True


def make_emitter(mode: str, *, wtype_timeout_s: float = 10.0) -> Emitter:
    """Factory honoring the WL_DICTATE_EMIT env override (wtype|stdout|null)."""
    override = os.environ.get("WL_DICTATE_EMIT", "").strip().lower()
    choice = override or mode
    if choice in ("null", "none"):
        return NullEmitter()
    if choice == "stdout":
        return StdoutEmitter()
    return WtypeEmitter(timeout_s=wtype_timeout_s)
