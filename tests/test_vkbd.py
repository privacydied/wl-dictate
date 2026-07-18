"""Virtual-keyboard device: keymap generation, wire format, emitter fallback."""

import struct

import pytest

import wldictate.vkbd as vkbd_mod
from wldictate.emitter import ZWSP, WtypeEmitter
from wldictate.vkbd import WaylandVirtualKeyboard, _marshal_string, _pad


# ── Pure helpers (no socket needed) ─────────────────────────────────────────


def test_marshal_string_pads_to_four_bytes():
    data = _marshal_string("hi")
    # length includes the NUL: 3, then "hi\0" padded to 4 bytes.
    assert data == struct.pack("<I", 3) + b"hi\0\0"
    assert len(data) % 4 == 0


def test_pad_alignment():
    assert _pad(b"abc") == b"abc\0"
    assert _pad(b"abcd") == b"abcd"


def bare_vkbd():
    """A WaylandVirtualKeyboard with no socket: exercises keymap logic only."""
    vk = WaylandVirtualKeyboard.__new__(WaylandVirtualKeyboard)
    vk._sym_code = {}
    vk._entries = []
    return vk


def test_sym_for_char_uses_hex_keysyms_and_named_keys():
    # ASCII printables map to themselves; others to 0x0100_0000 | codepoint.
    assert WaylandVirtualKeyboard._sym_for_char("A") == "0x00000041"
    assert WaylandVirtualKeyboard._sym_for_char(" ") == "0x00000020"
    assert WaylandVirtualKeyboard._sym_for_char("é") == "0x010000e9"
    assert WaylandVirtualKeyboard._sym_for_char("😀") == "0x0101f600"
    assert WaylandVirtualKeyboard._sym_for_char("\n") == "Return"
    assert WaylandVirtualKeyboard._sym_for_char("\t") == "Tab"


def test_keymap_text_matches_wtype_layout():
    vk = bare_vkbd()
    vk._add_syms(["0x00000041", "Return"])
    text = vk.keymap_text()
    assert "minimum = 8;" in text
    assert "maximum = 11;" in text  # 2 entries -> len + 9
    assert "<K1> = 9;" in text
    assert "<K2> = 10;" in text
    assert "key <K1> {[0x00000041]};" in text
    assert "key <K2> {[Return]};" in text
    assert 'xkb_types "(unnamed)" { include "complete" };' in text
    assert 'xkb_compatibility "(unnamed)" { include "complete" };' in text


def test_add_syms_assigns_stable_evdev_codes():
    vk = bare_vkbd()
    assert vk._add_syms(["0x00000061", "0x00000062"]) is True
    assert vk._sym_code == {"0x00000061": 1, "0x00000062": 2}
    assert vk._add_syms(["0x00000061"]) is False  # already mapped
    assert vk._add_syms(["0x00000063"]) is True
    assert vk._sym_code["0x00000063"] == 3


def test_add_syms_evicts_on_overflow(monkeypatch):
    monkeypatch.setattr(vkbd_mod, "_MAX_ENTRIES", 4)
    vk = bare_vkbd()
    vk._add_syms(["a", "b", "c", "d"])
    assert vk._add_syms(["d", "e"]) is True  # overflow: rebuild
    assert set(vk._sym_code) == {"d", "e"}
    assert len(vk._entries) == 2


def test_registry_global_event_parsing():
    vk = bare_vkbd()
    vk._registry = 2
    vk._callbacks = set()
    vk._done_callbacks = set()
    vk._globals = {}
    iface = b"wl_seat\0"
    payload = (
        struct.pack("<I", 7)  # name
        + struct.pack("<I", len(iface))
        + _pad(iface)
        + struct.pack("<I", 9)  # version
    )
    vk._handle_event(2, 0, payload)
    assert vk._globals == {"wl_seat": (7, 9)}


def test_display_error_event_raises():
    vk = bare_vkbd()
    vk._registry = 2
    vk._callbacks = set()
    msg = b"bad object\0"
    payload = struct.pack("<II", 3, 1) + struct.pack("<I", len(msg)) + _pad(msg)
    with pytest.raises(vkbd_mod.VkbdError, match="bad object"):
        vk._handle_event(1, 0, payload)


# ── Emitter integration (stubbed device) ────────────────────────────────────


class FakeVkbd:
    def __init__(self):
        self.calls = []
        self.keys_sent = 0
        self.fail_on = None  # "backspaces" | "text" | "text-after-send"

    def type_backspaces(self, count, delay_ms=0):
        if self.fail_on == "backspaces":
            raise vkbd_mod.VkbdError("boom")
        self.calls.append(("backspaces", count))
        self.keys_sent += count

    def type_text(self, text, delay_ms=0):
        if self.fail_on == "text":
            raise vkbd_mod.VkbdError("boom")
        if self.fail_on == "text-after-send":
            self.keys_sent += 1
            raise vkbd_mod.VkbdError("boom")
        self.calls.append(("text", text))
        self.keys_sent += len(text)

    def press_named(self, keysym, delay_ms=0):
        self.calls.append(("press", keysym))
        self.keys_sent += 1

    def ctrl_tap(self, ch):
        self.calls.append(("ctrl", ch))
        self.keys_sent += 1

    def close(self):
        pass


@pytest.fixture
def fake_vkbd(monkeypatch):
    fake = FakeVkbd()
    monkeypatch.setattr(vkbd_mod, "get_virtual_keyboard", lambda env: fake)
    monkeypatch.setattr(vkbd_mod, "invalidate", lambda env: None)
    return fake


class NoRun:
    """subprocess.run stub that fails the test if the fallback path fires."""

    def __call__(self, *a, **k):
        raise AssertionError("subprocess fallback should not run")


def test_rewrite_routes_through_vkbd(fake_vkbd, monkeypatch):
    import wldictate.emitter as em

    monkeypatch.setattr(em.subprocess, "run", NoRun())
    e = WtypeEmitter(backend="auto", delay_ms=0)
    assert e.rewrite(2, "hello") == "hello"
    assert fake_vkbd.calls == [("backspaces", 2), ("text", "hello")]


def test_rewrite_falls_back_to_wtype_when_nothing_sent(fake_vkbd, monkeypatch):
    import wldictate.emitter as em

    fake_vkbd.fail_on = "text"
    ran = {}

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        ran["cmd"] = cmd
        return Result()

    monkeypatch.setattr(em.subprocess, "run", fake_run)
    e = WtypeEmitter(backend="auto", delay_ms=0, electron_workaround=False)
    assert e.rewrite(0, "hello") == "hello"
    assert ran["cmd"][0] == "wtype"  # subprocess fallback fired


def test_rewrite_partial_delivery_returns_none(fake_vkbd, monkeypatch):
    import wldictate.emitter as em

    fake_vkbd.fail_on = "text-after-send"
    monkeypatch.setattr(em.subprocess, "run", NoRun())
    e = WtypeEmitter(backend="auto", delay_ms=0, electron_workaround=False)
    # Keys may have landed: screen unknown; must NOT retype via wtype.
    assert e.rewrite(0, "hello") is None


def test_electron_gate_fires_per_emission(fake_vkbd, monkeypatch):
    # Chromium re-arms the leading-space drop per input burst, so EVERY pure
    # append with a leading space into an Electron app gets the ZWSP —
    # identical to the wtype-subprocess behavior.
    e = WtypeEmitter(backend="auto", delay_ms=0)
    monkeypatch.setattr(e, "_focused_window_class", lambda: "vesktop")
    assert e.rewrite(0, " one") == ZWSP + " one"
    assert e.rewrite(0, " two") == ZWSP + " two"
    # Rewrites with backspaces open the gate themselves: no ZWSP.
    assert e.rewrite(2, " three") == " three"
    typed = [c for c in fake_vkbd.calls if c[0] == "text"]
    assert typed == [
        ("text", ZWSP + " one"),
        ("text", ZWSP + " two"),
        ("text", " three"),
    ]


def test_press_key_routes_through_vkbd(fake_vkbd, monkeypatch):
    import wldictate.emitter as em

    monkeypatch.setattr(em.subprocess, "run", NoRun())
    e = WtypeEmitter(backend="auto")
    assert e.press_key("Return") is True
    assert ("press", "Return") in fake_vkbd.calls


def test_wtype_backend_never_touches_vkbd(monkeypatch):
    import wldictate.emitter as em

    def explode(env):
        raise AssertionError("vkbd must not be used with backend='wtype'")

    monkeypatch.setattr(vkbd_mod, "get_virtual_keyboard", explode)

    class Result:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(em.subprocess, "run", lambda *a, **k: Result())
    e = WtypeEmitter(backend="wtype", electron_workaround=False)
    assert e.rewrite(0, "x") == "x"


def test_bulk_paste_uses_vkbd_chord(fake_vkbd, monkeypatch):
    import wldictate.emitter as em

    e = WtypeEmitter(backend="auto", delay_ms=0)
    monkeypatch.setattr(e, "_focused_window_class", lambda: "vesktop")
    monkeypatch.setattr(e, "_read_clipboard", lambda: "old")
    written = []
    monkeypatch.setattr(e, "_write_clipboard", lambda t: written.append(t) or True)
    monkeypatch.setattr(em.subprocess, "run", NoRun())
    text = "x" * 200
    assert e.rewrite_bulk(3, text) == text
    assert written[0] == text
    assert ("backspaces", 3) in fake_vkbd.calls
    assert ("ctrl", "v") in fake_vkbd.calls
