"""Text emitters — deliver committed text to the focused window.

``WtypeEmitter`` is the production device emitter (typing via wtype).
``CorrectingEmitter`` wraps a device emitter and rewrites already-typed
tentative text in place (backspace + retype) for the live self-correcting
``typing.mode = "correcting"``. Alternative sinks (stdout for debugging,
null for benchmarks) swap in without touching the streaming engine.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import threading
from abc import ABC, abstractmethod

#: Zero-width space — invisible gate-opener for Electron's leading-space drop.
ZWSP = "​"

#: Safety cap on backspaces per sync: a pathological diff rewrites at most the
#: last N physical characters instead of machine-gunning the whole utterance.
_MAX_BACKSPACES = 500

#: Bulk rewrites at/above this many typed characters go through the clipboard
#: paste fast-path (where the focused app supports Ctrl+V paste): ~6ms/char
#: keystroking turns a 200-char LLM replacement into >1s of visible churn;
#: paste is one keystroke.
_PASTE_MIN_CHARS = 120

#: How long after Ctrl+V before the previous clipboard is restored. Paste
#: consumers read the clipboard when handling the paste event; restoring too
#: early races them.
_CLIPBOARD_RESTORE_DELAY_S = 0.4


class Emitter(ABC):
    @abstractmethod
    def emit(self, text: str) -> bool:
        """Deliver text; returns True on success."""

    def rewrite(self, backspaces: int, text: str) -> str | None:
        """Delete ``backspaces`` characters, then type ``text``, atomically.

        Returns the *physically typed* string (which may differ from ``text``,
        e.g. an Electron ZWSP prefix), or None on failure. Default: devices
        that cannot delete just emit the text (best effort).
        """
        return text if self.emit(text) else None

    def rewrite_bulk(self, backspaces: int, text: str) -> str | None:
        """Like ``rewrite`` but for large one-shot replacements; devices may
        use a faster delivery (clipboard paste). Default: plain rewrite."""
        return self.rewrite(backspaces, text)

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class NullEmitter(Emitter):
    """Discards text (used when typing is disabled, e.g. benchmarks)."""

    def emit(self, text: str) -> bool:
        return True

    def rewrite(self, backspaces: int, text: str) -> str | None:
        return text


class StdoutEmitter(Emitter):
    """Prints text to stderr instead of typing (debugging/verification).

    stderr, not stdout: worker stdout is the JSON IPC channel.
    """

    def emit(self, text: str) -> bool:
        print(f"[emit] {text!r}", file=sys.stderr, flush=True)
        return True

    def rewrite(self, backspaces: int, text: str) -> str | None:
        print(f"[rewrite] -{backspaces} +{text!r}", file=sys.stderr, flush=True)
        return text


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


def focused_window(env: dict[str, str]) -> tuple[str, str]:
    """(class, title) of the focused window; ("", "") if unknown/not Hyprland."""
    try:
        result = subprocess.run(
            ["hyprctl", "-j", "activewindow"],
            env=env,
            timeout=1.0,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return "", ""
        import json

        data = json.loads(result.stdout)
        cls = str(data.get("class") or data.get("initialClass") or "")
        title = str(data.get("title") or "")
        return cls, title
    except Exception:
        return "", ""


class WtypeEmitter(Emitter):
    """Types text into the focused window with wtype."""

    #: Invisible gate-opener for Electron's leading-space drop (see rewrite()).
    _ZWSP = ZWSP

    def __init__(
        self,
        timeout_s: float = 10.0,
        delay_ms: int = 6,
        press_delay_ms: int = 0,
        electron_workaround: bool = True,
        electron_classes: tuple[str, ...] | list[str] = (
            "vesktop",
            "discord",
            "webcord",
            "legcord",
            "chromium",
            "chrome",
            "electron",
            "slack",
            "element",
            "signal",
        ),
    ) -> None:
        self._timeout = timeout_s
        # Per-keystroke delay (`wtype -d`), paces bursts for slow consumers.
        self._delay_ms = max(0, int(delay_ms))
        # Settle delay before the first keystroke of each call (`wtype -s`).
        self._press_delay_ms = max(0, int(press_delay_ms))
        self._electron_workaround = electron_workaround
        self._electron_classes = tuple(c.lower() for c in electron_classes)
        self._env = os.environ.copy()
        _guess_wayland_env(self._env)

    def _focused_window_class(self) -> str:
        """Window class of the focused window ("" if unknown / not Hyprland)."""
        return focused_window(self._env)[0]

    def _needs_electron_gate(self, text: str) -> bool:
        """True when the leading space of ``text`` would be eaten by the
        focused window.

        Chromium/Electron apps drop SPACE keys at the start of every fresh
        wtype connection — regardless of -s/-d delays — fusing dictated words
        ("TestingTesting"). An invisible zero-width space typed first "opens
        the gate" so the real space lands. Only applied when the focused
        window class matches a known Electron app, so terminals and editors
        never receive ZWSP characters.
        """
        if not self._electron_workaround or not text.startswith(" "):
            return False
        focused = self._focused_window_class().lower()
        if not focused:
            return False
        return any(cls in focused for cls in self._electron_classes)

    def emit(self, text: str) -> bool:
        if not text:
            return True
        return self.rewrite(0, text) is not None

    def rewrite(self, backspaces: int, text: str) -> str | None:
        """Delete ``backspaces`` chars then type ``text`` in ONE wtype call.

        wtype processes argv sequentially, so the BackSpace keys land before
        the stdin text (the trailing ``-``). ``-d`` paces only *text* typing;
        keys are paced by interleaving ``-s`` before each BackSpace (Electron
        drops keys that arrive too fast). Returns the physically typed string
        (ZWSP prefix included when the Electron gate fires) or None on error.
        """
        backspaces = max(0, backspaces)
        if backspaces == 0 and not text:
            return ""
        # Electron gate only on pure appends: with backspaces > 0 the
        # BackSpace keys themselves open Electron's fresh-connection gate, so
        # the retyped leading space lands without ZWSP accumulation.
        if backspaces == 0 and self._needs_electron_gate(text):
            text = self._ZWSP + text
        # Pass text on stdin via wtype's "-" placeholder rather than as an
        # argv word: text that begins with "-" (e.g. a spoken dash) would
        # otherwise be misparsed as a flag ("Missing argument to -foo").
        cmd = ["wtype"]
        if self._press_delay_ms > 0:
            cmd += ["-s", str(self._press_delay_ms)]
        if self._delay_ms > 0:
            cmd += ["-d", str(self._delay_ms)]
        for _ in range(backspaces):
            if self._delay_ms > 0:
                cmd += ["-s", str(self._delay_ms)]
            cmd += ["-k", "BackSpace"]
        if text:
            cmd.append("-")
        try:
            result = subprocess.run(
                cmd,
                input=text,
                env=self._env,
                timeout=self._timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            print("wtype timed out", file=sys.stderr)
            return None
        except FileNotFoundError:
            print("wtype is not installed", file=sys.stderr)
            return None
        except OSError as e:
            print(f"wtype failed to launch: {e}", file=sys.stderr)
            return None
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or f"exit code {result.returncode}"
            print(f"wtype failed: {detail}", file=sys.stderr)
            return None
        return text

    def rewrite_bulk(self, backspaces: int, text: str) -> str | None:
        """Large replacement: clipboard + Ctrl+V where the app supports it.

        Only used for Electron/Chromium apps (slowest to keystroke into AND
        reliably Ctrl+V-pasteable). Terminals/editors keep plain keystroking:
        Ctrl+V is not paste there. The previous clipboard is restored shortly
        after the paste lands. Falls back to plain rewrite on any failure.
        """
        if len(text) < _PASTE_MIN_CHARS or not self._electron_workaround:
            return self.rewrite(backspaces, text)
        focused = self._focused_window_class().lower()
        if not focused or not any(cls in focused for cls in self._electron_classes):
            return self.rewrite(backspaces, text)

        previous = self._read_clipboard()
        if not self._write_clipboard(text):
            return self.rewrite(backspaces, text)
        cmd = ["wtype"]
        if self._press_delay_ms > 0:
            cmd += ["-s", str(self._press_delay_ms)]
        for _ in range(max(0, backspaces)):
            if self._delay_ms > 0:
                cmd += ["-s", str(self._delay_ms)]
            cmd += ["-k", "BackSpace"]
        cmd += ["-s", str(max(10, self._delay_ms)), "-M", "ctrl", "-k", "v", "-m", "ctrl"]
        try:
            result = subprocess.run(
                cmd, env=self._env, timeout=self._timeout, capture_output=True, text=True
            )
            ok = result.returncode == 0
        except Exception as e:
            print(f"wtype paste failed: {e}", file=sys.stderr)
            ok = False
        if previous is not None:
            timer = threading.Timer(
                _CLIPBOARD_RESTORE_DELAY_S, self._write_clipboard, args=(previous,)
            )
            timer.daemon = True
            timer.start()
        if not ok:
            return None  # backspaces may have landed: caller treats as failure
        return text

    def _read_clipboard(self) -> str | None:
        try:
            result = subprocess.run(
                ["wl-paste", "--no-newline"],
                env=self._env,
                timeout=1.0,
                capture_output=True,
                text=True,
                errors="replace",
            )
            return result.stdout if result.returncode == 0 else None
        except Exception:
            return None

    def _write_clipboard(self, text: str) -> bool:
        try:
            result = subprocess.run(
                ["wl-copy"], input=text, env=self._env, timeout=1.0, capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False


class CorrectingEmitter(Emitter):
    """Rewrites tentative text in place (live self-correcting mode).

    Tracks the *physical* characters typed for the current utterance
    (including invisible ZWSPs the Electron gate injects) and syncs the
    screen to a desired string via a minimal common-prefix diff: backspace
    the divergent suffix, retype the new one. Never backspaces past what the
    current utterance typed, so text finalized before ``begin_utterance()``
    is immutable.
    """

    def __init__(self, device: Emitter) -> None:
        self._device = device
        self._screen = ""  # physical chars typed this utterance (may hold ZWSP)
        self._frozen = False  # device failure: screen state unknown

    def begin_utterance(self) -> None:
        """Reset the typing baseline; everything before it is immutable."""
        self._screen = ""
        self._frozen = False

    def _logical(self) -> str:
        return self._screen.replace(ZWSP, "")

    def sync(
        self,
        desired: str,
        *,
        max_backspaces: int | None = None,
        bulk: bool = False,
    ) -> bool:
        """Make the screen show ``desired`` (for this utterance's region).

        ``max_backspaces`` overrides the default safety cap — the contextual
        transform passes a large budget so a full-utterance replacement is
        never truncated (live decodes keep the tight default: a pathological
        mid-stream diff is recoverable by the next decode). ``bulk`` marks a
        one-shot replacement eligible for the device's paste fast-path.
        """
        if self._frozen:
            return False
        cap = _MAX_BACKSPACES if max_backspaces is None else max_backspaces
        logical = self._logical()
        # Longest common prefix (logical view — ZWSPs are invisible).
        p = 0
        for a, b in zip(logical, desired):
            if a != b:
                break
            p += 1
        # Map logical prefix length -> physical index. A ZWSP gating a kept
        # char stays with the kept prefix; a ZWSP *at* the boundary gated a
        # now-deleted char, so it is deleted too (no stranded invisibles).
        phys = 0
        remaining = p
        while phys < len(self._screen) and remaining > 0:
            if self._screen[phys] != ZWSP:
                remaining -= 1
            phys += 1
        backspaces = len(self._screen) - phys
        suffix = desired[p:]
        if backspaces > cap:
            # Pathological rewrite: keep the (possibly wrong) older prefix and
            # rewrite only the last ``cap`` physical chars.
            phys = len(self._screen) - cap
            backspaces = cap
            kept_logical = len(self._screen[:phys].replace(ZWSP, ""))
            suffix = desired[kept_logical:]
        if backspaces == 0 and not suffix:
            return True
        if bulk:
            typed = self._device.rewrite_bulk(backspaces, suffix)
        else:
            typed = self._device.rewrite(backspaces, suffix)
        if typed is None:
            self._frozen = True  # some keys may have landed: state unknown
            return False
        self._screen = self._screen[:phys] + typed
        return True

    def emit(self, text: str) -> bool:
        """Append-only compatibility path (tracked so later syncs can fix it)."""
        if not text:
            return True
        return self.sync(self._logical() + text)

    def close(self) -> None:
        self._device.close()


def make_emitter(
    mode: str,
    *,
    wtype_timeout_s: float = 10.0,
    wtype_delay_ms: int = 6,
    wtype_press_delay_ms: int = 0,
    electron_workaround: bool = True,
    electron_classes: tuple[str, ...] | list[str] | None = None,
) -> Emitter:
    """Factory honoring the WL_DICTATE_EMIT env override (wtype|stdout|null).

    The env override selects the *device* (useful for debugging: with
    ``WL_DICTATE_EMIT=stdout`` correcting mode prints its rewrite ops); the
    ``mode`` stays orthogonal — ``"correcting"`` wraps the device in a
    :class:`CorrectingEmitter`, anything else returns the bare device.
    """
    override = os.environ.get("WL_DICTATE_EMIT", "").strip().lower()
    device_choice = override or ("wtype" if mode in ("commit", "correcting") else mode)
    device: Emitter
    if device_choice in ("null", "none"):
        device = NullEmitter()
    elif device_choice == "stdout":
        device = StdoutEmitter()
    else:
        kwargs: dict = dict(
            timeout_s=wtype_timeout_s,
            delay_ms=wtype_delay_ms,
            press_delay_ms=wtype_press_delay_ms,
            electron_workaround=electron_workaround,
        )
        if electron_classes is not None:
            kwargs["electron_classes"] = electron_classes
        device = WtypeEmitter(**kwargs)
    if mode == "correcting":
        return CorrectingEmitter(device)
    return device
