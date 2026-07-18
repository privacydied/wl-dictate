"""Persistent Wayland virtual keyboard (zwp_virtual_keyboard_v1).

Every wtype invocation pays fork+exec, a fresh Wayland connect, and the
virtual-keyboard handshake — dozens of times per utterance in correcting
mode. This module speaks the Wayland wire protocol directly (no
dependencies) and keeps ONE virtual keyboard alive for the worker's
lifetime — the same first-principles move as ``persistent_capture`` for
the microphone: negotiate once, never again. Typing a rewrite becomes a
handful of small socket writes instead of a process spawn.

Keymap strategy (unlike wtype, which invents keycodes): every key that
exists on a physical US keyboard sits at its REAL evdev scancode, with
shifted characters as a second shift level on their base key — because
Chromium derives DOM ``event.code`` from the scancode (not the keymap) and
apps break on synthetic input at made-up codes (Discord dropped every
space), while scancode-matching consumers (compositor media binds, logind)
can fire on them (a capital 'I' at KEY_MUTE muted the audio). Exotic
unicode allocates LRU slots from the few scancodes the kernel leaves
undefined, re-uploading the keymap on demand.

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
#: Real-modifier bitmasks in the fixed core order
#: (Shift=1, Lock=2, Control=4, Mod1..Mod5).
MOD_SHIFT = 1
MOD_CONTROL = 4

#: Real evdev scancodes (wire codes) for keysyms that exist on a physical US
#: keyboard, plus the named keys. Chromium derives DOM ``event.code`` from
#: the evdev scancode — NOT from the uploaded keymap — and apps that check
#: it break on synthetic input at made-up scancodes: Discord's editor
#: silently ignores a space whose scancode isn't KEY_SPACE, so EVERY
#: synthetic space vanished (ZWSP gate or not, vkbd and wtype alike).
#: Placing keys at their physical scancodes makes synthetic input
#: indistinguishable from a real keyboard.
_EVDEV_CODES: dict[str, int] = {"Escape": 1, "BackSpace": 14, "Tab": 15, "Return": 28}
for _chars, _base in (
    ("1234567890-=", 2),  # KEY_1 .. KEY_EQUAL
    ("qwertyuiop[]", 16),  # KEY_Q .. KEY_RIGHTBRACE
    ("asdfghjkl;'`", 30),  # KEY_A .. KEY_GRAVE
    ("\\zxcvbnm,./", 43),  # KEY_BACKSLASH .. KEY_SLASH
):
    for _i, _ch in enumerate(_chars):
        _EVDEV_CODES[f"0x{ord(_ch):08x}"] = _base + _i
_EVDEV_CODES["0x00000020"] = 57  # KEY_SPACE
del _chars, _base, _i, _ch

#: Shifted ASCII -> its unshifted US base key. These are typed as
#: Shift + base scancode (exactly like a real keyboard: correct scancode,
#: correct DOM code, correct shiftKey) via the two-level keymap below —
#: NEVER via made-up scancodes. The first scancode fix allocated shifted
#: chars sequentially from 90, which put capital 'I' on KEY_MUTE (113),
#: 'J'/'K' on volume, and 'L' on KEY_POWER — dictating "I" muted the
#: user's audio, because parts of the stack act on SCANCODES regardless of
#: the uploaded keymap.
_SHIFT_BASE: dict[str, str] = {}
for _lo in "abcdefghijklmnopqrstuvwxyz":
    _SHIFT_BASE[_lo.upper()] = _lo
for _b, _s in zip("1234567890-=[];'`\\,./", "!@#$%^&*()_+{}:\"~|<>?"):
    _SHIFT_BASE[_s] = _b
del _lo, _b, _s
_SHIFT_OF = {b: s for s, b in _SHIFT_BASE.items()}

#: The static keymap block: every physical key, two-level where a shifted
#: partner exists ([a, A], [1, exclam]).  (sym, wire code, shifted sym|None)
_STATIC_KEYS: list[tuple[str, int, str | None]] = []
for _sym, _code in sorted(_EVDEV_CODES.items(), key=lambda kv: kv[1]):
    _shifted = None
    try:
        _cp = int(_sym, 16)
    except ValueError:
        _cp = -1
    if 0x20 <= _cp <= 0x7E and chr(_cp) in _SHIFT_OF:
        _shifted = f"0x{ord(_SHIFT_OF[chr(_cp)]):08x}"
    _STATIC_KEYS.append((_sym, _code, _shifted))
del _sym, _code, _shifted, _cp

#: Wire codes for exotic keysyms (unicode with no physical key: é, emoji,
#: CJK from transforms). ONLY scancodes that are undefined in
#: linux/input-event-codes.h (84, 195-199) plus the inert KEY_F13..F24
#: block — any *real* key's scancode risks scancode-matching consumers
#: (compositor media binds, logind's power-key handling). LRU-evicted when
#: full; each new sym re-uploads the keymap.
_DYN_CODES: tuple[int, ...] = (84, 195, 196, 197, 198, 199) + tuple(range(183, 195))


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
        #: EXOTIC keysym spec -> wire code from _DYN_CODES, insertion-ordered
        #: for LRU eviction. Physical keys live in the static _EVDEV_CODES /
        #: _STATIC_KEYS tables and never appear here.
        self._sym_code: dict[str, int] = {}
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
            # The static block covers ALL of printable ASCII (unshifted keys
            # + shift levels), so ordinary dictation never re-uploads the
            # keymap mid-utterance; only exotic unicode does.
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

    def _exotic_code(self, sym: str) -> int:
        """Wire code for an exotic keysym, allocating (and re-uploading the
        keymap) on first use. LRU: reuse refreshes recency; when all
        ``_DYN_CODES`` slots are taken the least-recently-used sym is
        evicted. Physical keys never pass through here."""
        code = self._sym_code.get(sym)
        if code is not None:
            self._sym_code[sym] = self._sym_code.pop(sym)  # refresh recency
            return code
        used = set(self._sym_code.values())
        free = next((c for c in _DYN_CODES if c not in used), None)
        if free is None:
            oldest = next(iter(self._sym_code))
            free = self._sym_code.pop(oldest)
        self._sym_code[sym] = free
        self._upload_keymap()
        return free

    def keymap_text(self) -> str:
        dyn = sorted(self._sym_code.items(), key=lambda kv: kv[1])
        lines = [
            "xkb_keymap {",
            'xkb_keycodes "(unnamed)" {',
            "minimum = 8;",
            "maximum = 255;",
        ]
        for _sym, code, _sh in _STATIC_KEYS:
            lines.append(f"<K{code}> = {code + 8};")
        for _sym, code in dyn:
            lines.append(f"<K{code}> = {code + 8};")
        lines += [
            "};",
            'xkb_types "(unnamed)" { include "complete" };',
            'xkb_compatibility "(unnamed)" { include "complete" };',
            'xkb_symbols "(unnamed)" {',
        ]
        for sym, code, shifted in _STATIC_KEYS:
            if shifted:
                lines.append(f"key <K{code}> {{[{sym}, {shifted}]}};")
            else:
                lines.append(f"key <K{code}> {{[{sym}]}};")
        for sym, code in dyn:
            lines.append(f"key <K{code}> {{[{sym}]}};")
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

    # ── Typing ───────────────────────────────────────────────────────────

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000) & 0xFFFFFFFF

    def _key_event(self, code: int, state: int) -> None:
        self._request(self._vk, 1, struct.pack("<III", self._now_ms(), code, state))

    def _tap(self, code: int, delay_ms: int, *, mods: int = 0) -> None:
        if mods:
            self._request(self._vk, 2, struct.pack("<IIII", mods, 0, 0, 0))
        self._key_event(code, _KEY_STATE_PRESSED)
        self._key_event(code, _KEY_STATE_RELEASED)
        if mods:
            self._request(self._vk, 2, struct.pack("<IIII", 0, 0, 0, 0))
        self.keys_sent += 1
        if delay_ms > 0:
            self._flush()
            time.sleep(delay_ms / 1000.0)

    def _char_plan(self, ch: str) -> tuple[int, int]:
        """(wire code, modifier mask) delivering one character.

        Shifted ASCII is Shift + its real base key (like a physical
        keyboard); unshifted ASCII and named whitespace hit their own real
        scancodes; exotic unicode uses an LRU slot from ``_DYN_CODES``.
        """
        base = _SHIFT_BASE.get(ch)
        if base is not None:
            return _EVDEV_CODES[self._sym_for_char(base)], MOD_SHIFT
        sym = self._sym_for_char(ch)
        code = _EVDEV_CODES.get(sym)
        if code is not None:
            return code, 0
        return self._exotic_code(sym), 0

    def type_backspaces(self, count: int, delay_ms: int = 0) -> None:
        if count <= 0:
            return
        with self._lock:
            code = _EVDEV_CODES["BackSpace"]
            for _ in range(count):
                self._tap(code, delay_ms)
            self._flush()
            self.drain()

    def type_text(self, text: str, delay_ms: int = 0) -> None:
        if not text:
            return
        with self._lock:
            for ch in text:
                code, mods = self._char_plan(ch)
                self._tap(code, delay_ms, mods=mods)
            self._flush()
            self.drain()

    def press_named(self, keysym: str, delay_ms: int = 0) -> None:
        """Tap one named xkb keysym (e.g. "Return", "Tab", "Escape")."""
        with self._lock:
            code = _EVDEV_CODES.get(keysym)
            if code is None:
                code = self._exotic_code(keysym)
            self._tap(code, delay_ms)
            self._flush()
            self.drain()

    def ctrl_tap(self, ch: str) -> None:
        """Tap Control+<ch> (e.g. the paste chord Ctrl+V) at its real
        scancode, so apps that match the chord by DOM code accept it."""
        with self._lock:
            code, _ = self._char_plan(ch)
            self._tap(code, 0, mods=MOD_CONTROL)
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
