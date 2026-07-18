"""Virtual-keyboard device: keymap generation, wire format, emitter fallback."""

import struct

import pytest

import wldictate.vkbd as vkbd_mod
from wldictate.emitter import WtypeEmitter
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
    vk.uploads = 0

    def fake_upload():
        vk.uploads += 1

    vk._upload_keymap = fake_upload
    return vk


def test_sym_for_char_uses_hex_keysyms_and_named_keys():
    # ASCII printables map to themselves; others to 0x0100_0000 | codepoint.
    assert WaylandVirtualKeyboard._sym_for_char("A") == "0x00000041"
    assert WaylandVirtualKeyboard._sym_for_char(" ") == "0x00000020"
    assert WaylandVirtualKeyboard._sym_for_char("é") == "0x010000e9"
    assert WaylandVirtualKeyboard._sym_for_char("😀") == "0x0101f600"
    assert WaylandVirtualKeyboard._sym_for_char("\n") == "Return"
    assert WaylandVirtualKeyboard._sym_for_char("\t") == "Tab"


def test_static_keymap_uses_real_scancodes_with_shift_levels():
    text = bare_vkbd().keymap_text()
    assert "minimum = 8;" in text
    assert "maximum = 255;" in text
    # Physical keys at their REAL evdev scancodes (xkb = evdev + 8):
    # Chromium derives event.code from the scancode, not the keymap.
    assert "<K57> = 65;" in text  # KEY_SPACE — the Discord space fix
    assert "key <K57> {[0x00000020]};" in text
    assert "key <K28> {[Return]};" in text  # KEY_ENTER
    assert "key <K14> {[BackSpace]};" in text  # KEY_BACKSPACE
    # Two-level keys: shifted chars ride their base key like a real keyboard
    # ('a'/'A' on KEY_A=30, '1'/'!' on KEY_1=2) — NEVER made-up scancodes,
    # which landed on media keys (capital 'I' -> KEY_MUTE muted the audio).
    assert "key <K30> {[0x00000061, 0x00000041]};" in text
    assert "key <K2> {[0x00000031, 0x00000021]};" in text
    assert 'xkb_types "(unnamed)" { include "complete" };' in text
    assert 'xkb_compatibility "(unnamed)" { include "complete" };' in text


def test_char_plan_shift_combos_and_real_codes():
    vk = bare_vkbd()
    assert vk._char_plan("a") == (30, 0)  # KEY_A
    assert vk._char_plan("A") == (30, vkbd_mod.MOD_SHIFT)  # Shift+KEY_A
    assert vk._char_plan(" ") == (57, 0)  # KEY_SPACE
    assert vk._char_plan("I") == (23, vkbd_mod.MOD_SHIFT)  # Shift+KEY_I, NOT KEY_MUTE
    assert vk._char_plan("!") == (2, vkbd_mod.MOD_SHIFT)  # Shift+KEY_1
    assert vk._char_plan("\n") == (28, 0)  # Return
    assert vk.uploads == 0  # ASCII never re-uploads the keymap


def test_exotic_chars_use_safe_slots_with_lru_eviction():
    vk = bare_vkbd()
    code, mods = vk._char_plan("é")
    assert mods == 0
    assert code in vkbd_mod._DYN_CODES
    assert vk.uploads == 1
    assert vk._char_plan("é") == (code, 0)  # cached: no re-upload
    assert vk.uploads == 1
    # Fill every slot, then one more: the LRU sym is evicted, code reused.
    for i in range(len(vkbd_mod._DYN_CODES) - 1):
        vk._char_plan(chr(0x4E00 + i))
    assert vk._char_plan("œ")[0] == code  # 'é' was oldest -> evicted
    assert "0x010000e9" not in vk._sym_code


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


def test_emissions_pass_through_verbatim(fake_vkbd, monkeypatch):
    # No ZWSP gate: spaces land because every key is at its real evdev
    # scancode. Emissions reach the device exactly as given.
    e = WtypeEmitter(backend="auto", delay_ms=0)
    monkeypatch.setattr(e, "_focused_window_class", lambda: "vesktop")
    assert e.rewrite(0, " one") == " one"
    assert e.rewrite(2, " two") == " two"
    typed = [c for c in fake_vkbd.calls if c[0] == "text"]
    assert typed == [("text", " one"), ("text", " two")]


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
