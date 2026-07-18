"""Persistent Wayland virtual keyboard (zwp_virtual_keyboard_v1).

Every wtype invocation pays fork+exec, a fresh Wayland connect, and the
virtual-keyboard handshake — dozens of times per utterance in correcting
mode. Worse, Chromium/Electron drops leading SPACE keys at the start of
every *fresh* connection, which is the entire reason the ZWSP workaround in
the emitter exists.

This module speaks the Wayland wire protocol directly (no dependencies) and
keeps ONE virtual keyboard alive for the worker's lifetime — the same
first-principles move as ``persistent_capture`` for the microphone:
negotiate once, never again. Typing a rewrite becomes a handful of small
socket writes instead of a process spawn.

Keymap strategy (same as wtype): a generated xkb keymap maps one keycode
per distinct keysym; codepoints are written in xkbcommon's ``U+XXXX``
keysym form, named keys ("Return", "BackSpace", …) by name. Unlike wtype,
the keymap grows incrementally — new characters trigger a keymap re-upload
on the live connection; a full rebuild evicts stale entries if it ever
fills up.

Everything here is best-effort: any failure raises, and the emitter falls
back to the wtype subprocess path (see ``WtypeEmitter``).
"""

from __future__ import annotations

import array
import glob
import os
import socket
import struct
import sys
import threading
import time

_WL_DISPLAY = 1  # fixed singleton object id
_KEY_STATE_RELEASED = 0
_KEY_STATE_PRESSED = 1
_KEYMAP_FORMAT_XKB_V1 = 1
#: Real-modifier bitmask for Control in the fixed core order
#: (Shift=1, Lock=2, Control=4, Mod1..Mod5).
MOD_CONTROL = 4

#: Keymap capacity before a full rebuild evicts unused entries. Dictation
#: uses well under 150 distinct characters; this is just a safety bound.
_MAX_ENTRIES = 512


class VkbdUnavailable(RuntimeError):
    """The virtual keyboard cannot be set up (no compositor support, ...)."""


class VkbdError(RuntimeError):
    """A runtime failure on an established connection."""


def _pad(data: bytes) -> bytes:
    return data + b"\0" * (-len(data) % 4)


def _marshal_string(s: str) -> bytes:
    raw = s.encode("utf-8") + b"\0"
    return struct.pack("<I", len(raw)) + _pad(raw)


def _socket_path(env: dict[str, str]) -> str:
    display = env.get("WAYLAND_DISPLAY", "")
    runtime = env.get("XDG_RUNTIME_DIR", "")
    if not display:
        # Same best-effort discovery as the emitter's env guesser.
        candidates = [runtime] if runtime else []
        try:
            candidates.append(f"/run/user/{os.getuid()}")
        except Exception:
            pass
        for candidate in candidates:
            socks = [
                s
                for s in glob.glob(os.path.join(candidate, "wayland-*"))
                if not s.endswith(".lock")
            ]
            if socks:
                return socks[0]
        raise VkbdUnavailable("no Wayland display found")
    if os.path.isabs(display):
        return display
    if not runtime:
        raise VkbdUnavailable("XDG_RUNTIME_DIR not set")
    return os.path.join(runtime, display)


class WaylandVirtualKeyboard:
    """One long-lived zwp_virtual_keyboard_v1 over a raw Wayland socket.

    Not thread-safe by itself; in the worker all calls arrive on the single
    render thread (a lock still guards against misuse).
    """

    def __init__(self, env: dict[str, str] | None = None, timeout_s: float = 3.0) -> None:
        self._lock = threading.Lock()
        self._timeout = timeout_s
        self._out = bytearray()
        self._in_buf = b""
        self._next_id = 2
        self._callbacks: set[int] = set()  # ids of in-flight wl_callbacks
        self._done_callbacks: set[int] = set()
        self._globals: dict[str, tuple[int, int]] = {}  # iface -> (name, version)
        #: keysym spec -> evdev keycode (index in _entries + 1)
        self._sym_code: dict[str, int] = {}
        self._entries: list[str] = []
        #: total key events delivered on this connection (the Electron
        #: fresh-connection gate is open once this is > 0).
        self.keys_sent = 0

        path = _socket_path(env or os.environ.copy())
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.settimeout(timeout_s)
            self._sock.connect(path)
        except OSError as e:
            self._sock.close()
            raise VkbdUnavailable(f"cannot connect to {path}: {e}") from e

        try:
            self._registry = self._new_id()
            self._request(_WL_DISPLAY, 1, struct.pack("<I", self._registry))
            self._roundtrip()  # collect globals
            if "zwp_virtual_keyboard_manager_v1" not in self._globals:
                raise VkbdUnavailable(
                    "compositor lacks zwp_virtual_keyboard_manager_v1"
                )
            if "wl_seat" not in self._globals:
                raise VkbdUnavailable("no wl_seat advertised")
            self._seat = self._bind("wl_seat", 4)
            manager = self._bind("zwp_virtual_keyboard_manager_v1", 1)
            self._vk = self._new_id()
            self._request(manager, 0, struct.pack("<II", self._seat, self._vk))
            # Pre-seed printable ASCII so ordinary dictation never needs a
            # keymap re-upload mid-utterance.
            seed = [self._sym_for_char(chr(c)) for c in range(0x20, 0x7F)]
            seed += ["Return", "Tab", "BackSpace", "Escape"]
            self._add_syms(seed)
            self._upload_keymap()
            self._roundtrip()  # surface protocol errors before first use
        except Exception:
            self._sock.close()
            raise

    # ── Wire plumbing ────────────────────────────────────────────────────

    def _new_id(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid

    def _request(self, obj: int, opcode: int, args: bytes = b"") -> None:
        size = 8 + len(args)
        self._out += struct.pack("<II", obj, (size << 16) | opcode) + args

    def _flush(self, fds: list[int] | None = None) -> None:
        if not self._out and not fds:
            return
        data = bytes(self._out)
        self._out.clear()
        try:
            if fds:
                anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
                sent = self._sock.sendmsg([data], anc)
                data = data[sent:]
            while data:
                data = data[self._sock.send(data) :]
        except OSError as e:
            raise VkbdError(f"wayland write failed: {e}") from e

    def _bind(self, interface: str, want_version: int) -> int:
        name, advertised = self._globals[interface]
        oid = self._new_id()
        args = (
            struct.pack("<I", name)
            + _marshal_string(interface)
            + struct.pack("<II", min(want_version, advertised), oid)
        )
        self._request(self._registry, 0, args)
        return oid

    def _roundtrip(self) -> None:
        cb = self._new_id()
        self._callbacks.add(cb)
        self._request(_WL_DISPLAY, 0, struct.pack("<I", cb))
        self._flush()
        deadline = time.monotonic() + self._timeout
        while cb not in self._done_callbacks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VkbdError("wayland roundtrip timed out")
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout as e:
                raise VkbdError("wayland roundtrip timed out") from e
            except OSError as e:
                raise VkbdError(f"wayland read failed: {e}") from e
            if not chunk:
                raise VkbdError("wayland connection closed")
            self._in_buf += chunk
            self._dispatch()
        self._done_callbacks.discard(cb)
        self._callbacks.discard(cb)

    def drain(self) -> None:
        """Non-blocking read of queued events (surfaces protocol errors)."""
        while True:
            try:
                self._sock.setblocking(False)
                chunk = self._sock.recv(65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError as e:
                raise VkbdError(f"wayland read failed: {e}") from e
            finally:
                self._sock.settimeout(self._timeout)
            if not chunk:
                raise VkbdError("wayland connection closed")
            self._in_buf += chunk
            self._dispatch()

    def _dispatch(self) -> None:
        buf = self._in_buf
        pos = 0
        while len(buf) - pos >= 8:
            obj, sizeop = struct.unpack_from("<II", buf, pos)
            size = sizeop >> 16
            opcode = sizeop & 0xFFFF
            if size < 8 or len(buf) - pos < size:
                break
            payload = buf[pos + 8 : pos + size]
            pos += size
            self._handle_event(obj, opcode, payload)
        self._in_buf = buf[pos:]

    def _handle_event(self, obj: int, opcode: int, payload: bytes) -> None:
        if obj == _WL_DISPLAY:
            if opcode == 0:  # error(object_id, code, message)
                _oid, code = struct.unpack_from("<II", payload, 0)
                (strlen,) = struct.unpack_from("<I", payload, 8)
                msg = payload[12 : 12 + max(0, strlen - 1)].decode(
                    "utf-8", "replace"
                )
                raise VkbdError(f"wayland protocol error {code}: {msg}")
            return  # delete_id: ignored (ids are allocated monotonically)
        if obj == self._registry and opcode == 0:  # global(name, iface, ver)
            (name,) = struct.unpack_from("<I", payload, 0)
            (strlen,) = struct.unpack_from("<I", payload, 4)
            iface = payload[8 : 8 + max(0, strlen - 1)].decode("utf-8", "replace")
            padded = (strlen + 3) & ~3  # string bytes incl. NUL, 4-aligned
            (version,) = struct.unpack_from("<I", payload, 8 + padded)
            self._globals[iface] = (name, version)
            return
        # wl_callback.done; everything else (wl_seat capabilities/name,
        # registry global_remove, ...) is ignored.
        if obj in self._callbacks and opcode == 0:
            self._done_callbacks.add(obj)

    # ── Keymap ───────────────────────────────────────────────────────────

    @staticmethod
    def _sym_for_char(ch: str) -> str:
        """Hex keysym literal for one character.

        The xkb keymap grammar does NOT accept the ``U+XXXX`` unicode form
        (that's only valid for ``xkb_keysym_from_name``); numeric ``0x…``
        literals compile everywhere. Latin-1 printables map to themselves,
        everything else to the standard unicode keysym 0x0100_0000 | cp —
        the same values ``xkb_utf32_to_keysym`` produces for wtype.
        """
        if ch == "\n":
            return "Return"
        if ch == "\t":
            return "Tab"
        cp = ord(ch)
        keysym = cp if 0x20 <= cp <= 0x7E else 0x01000000 | cp
        return f"0x{keysym:08x}"

    def _add_syms(self, syms: list[str]) -> bool:
        added = False
        for sym in syms:
            if sym in self._sym_code:
                continue
            if len(self._entries) >= _MAX_ENTRIES:
                # Full: rebuild with only the syms needed right now.
                keep = [s for s in syms if s in self._sym_code] + [
                    s for s in syms if s not in self._sym_code
                ]
                self._entries = list(dict.fromkeys(keep))
                self._sym_code = {
                    s: i + 1 for i, s in enumerate(self._entries)
                }
                return True
            self._entries.append(sym)
            self._sym_code[sym] = len(self._entries)
            added = True
        return added

    def keymap_text(self) -> str:
        lines = [
            "xkb_keymap {",
            'xkb_keycodes "(unnamed)" {',
            "minimum = 8;",
            f"maximum = {len(self._entries) + 9};",
        ]
        for i in range(len(self._entries)):
            lines.append(f"<K{i + 1}> = {i + 9};")
        lines += [
            "};",
            'xkb_types "(unnamed)" { include "complete" };',
            'xkb_compatibility "(unnamed)" { include "complete" };',
            'xkb_symbols "(unnamed)" {',
        ]
        for i, sym in enumerate(self._entries):
            lines.append(f"key <K{i + 1}> {{[{sym}]}};")
        lines += ["};", "};", ""]
        return "\n".join(lines)

    def _upload_keymap(self) -> None:
        text = self.keymap_text().encode("utf-8")
        fd = os.memfd_create("wl-dictate-keymap")
        try:
            os.write(fd, text)
            self._request(
                self._vk, 0, struct.pack("<II", _KEYMAP_FORMAT_XKB_V1, len(text))
            )
            self._flush(fds=[fd])
        finally:
            os.close(fd)

    def _ensure(self, syms: list[str]) -> None:
        if self._add_syms(syms):
            self._upload_keymap()

    # ── Typing ───────────────────────────────────────────────────────────

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000) & 0xFFFFFFFF

    def _key_event(self, code: int, state: int) -> None:
        self._request(self._vk, 1, struct.pack("<III", self._now_ms(), code, state))

    def _tap(self, code: int, delay_ms: int) -> None:
        self._key_event(code, _KEY_STATE_PRESSED)
        self._key_event(code, _KEY_STATE_RELEASED)
        self.keys_sent += 1
        if delay_ms > 0:
            self._flush()
            time.sleep(delay_ms / 1000.0)

    def type_backspaces(self, count: int, delay_ms: int = 0) -> None:
        if count <= 0:
            return
        with self._lock:
            self._ensure(["BackSpace"])
            code = self._sym_code["BackSpace"]
            for _ in range(count):
                self._tap(code, delay_ms)
            self._flush()
            self.drain()

    def type_text(self, text: str, delay_ms: int = 0) -> None:
        if not text:
            return
        with self._lock:
            syms = [self._sym_for_char(ch) for ch in text]
            self._ensure(syms)
            for sym in syms:
                self._tap(self._sym_code[sym], delay_ms)
            self._flush()
            self.drain()

    def press_named(self, keysym: str, delay_ms: int = 0) -> None:
        """Tap one named xkb keysym (e.g. "Return", "Tab", "Escape")."""
        with self._lock:
            self._ensure([keysym])
            self._tap(self._sym_code[keysym], delay_ms)
            self._flush()
            self.drain()

    def ctrl_tap(self, ch: str) -> None:
        """Tap Control+<ch> (e.g. the paste chord Ctrl+V)."""
        with self._lock:
            sym = self._sym_for_char(ch)
            self._ensure([sym])
            code = self._sym_code[sym]
            self._request(self._vk, 2, struct.pack("<IIII", MOD_CONTROL, 0, 0, 0))
            self._key_event(code, _KEY_STATE_PRESSED)
            self._key_event(code, _KEY_STATE_RELEASED)
            self._request(self._vk, 2, struct.pack("<IIII", 0, 0, 0, 0))
            self.keys_sent += 1
            self._flush()
            self.drain()

    def close(self) -> None:
        try:
            self._request(self._vk, 3)  # destroy
            self._flush()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass


# ── Process-wide connection cache ────────────────────────────────────────────

_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], WaylandVirtualKeyboard | None] = {}


def _cache_key(env: dict[str, str]) -> tuple[str, str]:
    return (env.get("XDG_RUNTIME_DIR", ""), env.get("WAYLAND_DISPLAY", ""))


def get_virtual_keyboard(env: dict[str, str]) -> WaylandVirtualKeyboard | None:
    """Shared per-display connection; None (cached) when unavailable."""
    key = _cache_key(env)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
        try:
            vk = WaylandVirtualKeyboard(env)
        except Exception as e:
            print(
                f"virtual keyboard unavailable ({e}); falling back to wtype",
                file=sys.stderr,
            )
            vk = None
        _cache[key] = vk
        return vk


def invalidate(env: dict[str, str]) -> None:
    """Drop the cached connection (runtime failure): next call reconnects."""
    key = _cache_key(env)
    with _cache_lock:
        vk = _cache.pop(key, None)
    if vk is not None:
        vk.close()
