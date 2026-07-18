"""CorrectingEmitter diffing/state — no wtype, driven by a fake device."""

from __future__ import annotations

from wldictate.emitter import CorrectingEmitter, Emitter
import wldictate.emitter as emitter_mod


class FakeDevice(Emitter):
    """Records (backspaces, text) ops; optionally rewrites the typed text
    or fails."""

    def __init__(self):
        self.ops: list[tuple[int, str]] = []
        self.transform = lambda n, text: text  # physical string actually typed
        self.fail = False

    def emit(self, text: str) -> bool:
        return self.rewrite(0, text) is not None

    def rewrite(self, backspaces: int, text: str) -> str | None:
        if self.fail:
            return None
        self.ops.append((backspaces, text))
        return self.transform(backspaces, text)


def make() -> tuple[CorrectingEmitter, FakeDevice]:
    dev = FakeDevice()
    return CorrectingEmitter(dev), dev


def test_growth_is_pure_append():
    ce, dev = make()
    assert ce.sync("hello")
    assert ce.sync("hello world")
    assert dev.ops == [(0, "hello"), (0, " world")]


def test_suffix_fix_backspaces_divergence():
    ce, dev = make()
    ce.sync("hello word")
    ce.sync("hello world")
    assert dev.ops[-1] == (1, "ld")


def test_shrink_deletes_excess():
    ce, dev = make()
    ce.sync("hello world")
    ce.sync("hello")
    assert dev.ops[-1] == (6, "")


def test_sync_to_same_text_is_noop():
    ce, dev = make()
    ce.sync("hello")
    ce.sync("hello")
    assert dev.ops == [(0, "hello")]


def test_clear_screen():
    ce, dev = make()
    ce.sync("oops")
    ce.sync("")
    assert dev.ops[-1] == (4, "")


def test_begin_utterance_resets_baseline():
    ce, dev = make()
    ce.sync("first utterance. ")
    ce.begin_utterance()
    ce.sync("second")
    # New utterance never backspaces into finalized text.
    assert dev.ops[-1] == (0, "second")
    ce.sync("sec")
    assert dev.ops[-1] == (3, "")  # shrink stops at the baseline


def test_device_failure_freezes_until_next_utterance():
    ce, dev = make()
    ce.sync("hello")
    dev.fail = True
    assert ce.sync("hello world") is False
    dev.fail = False
    assert ce.sync("hello there") is False  # frozen: screen state unknown
    assert dev.ops == [(0, "hello")]
    ce.begin_utterance()
    assert ce.sync("fresh") is True
    assert dev.ops[-1] == (0, "fresh")


def test_backspace_cap_clamps_rewrite(monkeypatch):
    monkeypatch.setattr(emitter_mod, "_MAX_BACKSPACES", 5)
    ce, dev = make()
    ce.sync("abcdefghij")
    ce.sync("XYZ")  # full rewrite would need 10 backspaces
    n, text = dev.ops[-1]
    assert n == 5  # clamped: keeps the (possibly wrong) older prefix
    assert text == ""  # desired is shorter than the kept prefix


def test_emit_appends_and_is_tracked():
    ce, dev = make()
    assert ce.emit("hello") is True
    assert ce.emit(" world") is True
    assert dev.ops == [(0, "hello"), (0, " world")]
    ce.sync("hello word")
    assert dev.ops[-1] == (2, "d")


def test_max_backspaces_override_allows_full_replacement():
    ce, dev = make()
    ce.sync("a" * 700)
    # Default cap would clamp at 500; the transform path passes a big budget.
    assert ce.sync("b" * 3, max_backspaces=4000)
    assert dev.ops[-1] == (700, "bbb")


def test_begin_utterance_carry_accumulates_previous():
    ce, dev = make()
    ce.sync("chunk one. ")
    ce.begin_utterance(carry=True)
    ce.sync("chunk two. ")
    ce.begin_utterance(carry=True)
    ce.sync("chunk three.")
    assert ce.previous_len == len("chunk one. chunk two. ")
    # merge_previous folds the WHOLE chain for a combined rewrite.
    assert ce.merge_previous()
    assert ce.logical == "chunk one. chunk two. chunk three."
    assert ce.sync("All rewritten.")
    assert dev.ops[-1][0] == len("chunk one. chunk two. chunk three.")
