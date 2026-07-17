"""TransformCoordinator against the real CorrectingEmitter — the contextual
session's replace-in-place, cancel, failure, and drain semantics."""

from __future__ import annotations

import threading

import wldictate.transform as transform_mod
from wldictate.emitter import CorrectingEmitter, Emitter
from wldictate.textproc import TextFormatter
from wldictate.transform import TransformCoordinator, TransformError


class ScreenDevice(Emitter):
    """Models the actual display: applies ops to a screen string."""

    def __init__(self):
        self.ops = []
        self.screen = ""

    def emit(self, text):
        return self.rewrite(0, text) is not None

    def rewrite(self, backspaces, text):
        self.ops.append((backspaces, text))
        keep = len(self.screen) - backspaces
        self.screen = self.screen[:keep] + text
        return text


class FakeTransformer:
    """Controllable stand-in: blocks until released, then replies/raises."""

    def __init__(self, reply="TRANSFORMED"):
        self.reply = reply
        self.release = threading.Event()
        self.release.set()  # default: complete immediately

    def transform(self, transcript, context=None):
        assert self.release.wait(timeout=5.0), "test forgot to release"
        self.last_context = context
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def prefetch_context(self):
        return "FAKE-CONTEXT"

    def prewarm(self, context):
        self.prewarmed = context


class Harness:
    def __init__(self, reply="TRANSFORMED", notify_enabled=True):
        self.device = ScreenDevice()
        self.emitter = CorrectingEmitter(self.device)
        self.formatter = TextFormatter()
        self.transformer = FakeTransformer(reply)
        self.errors = []
        self.notifications = []
        self.coordinator = TransformCoordinator(
            self.transformer,
            self.emitter,
            self.formatter,
            timeout_s=5.0,
            notify_enabled=notify_enabled,
            on_error=self.errors.append,
        )

    def speak_utterance(self, raw: str) -> str:
        """Simulate a correcting-mode utterance reaching finalize."""
        self.emitter.begin_utterance()
        self.formatter.on_utterance_start()
        text = self.formatter.format_delta(raw)
        trailer = self.formatter.end_utterance()
        assert self.emitter.sync(text + trailer)
        return text + trailer

    def poll_until_applied(self, timeout=5.0):
        deadline = threading.Event()
        for _ in range(int(timeout * 100)):
            self.coordinator.poll()
            if self.coordinator._pending is None:
                return
            deadline.wait(0.01)


def _wait_done(h: Harness):
    future = h.coordinator._pending[0]
    future.exception(timeout=5.0)  # settles either way


def test_replace_in_place_with_trailer():
    h = Harness(reply="Polished, contextual text!")
    final = h.speak_utterance(" polished text.")
    assert h.device.screen == "Polished text. "
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    # Transformed body + regenerated trailer replaced the utterance in place.
    assert h.device.screen == "Polished, contextual text! "
    assert not h.errors
    h.coordinator.shutdown()


def test_separator_space_preserved():
    h = Harness(reply="Second thought.")
    h.speak_utterance(" first part.")
    h.speak_utterance(" second part.")
    assert h.device.screen == "First part. Second part. "
    # finalize's text for utterance 2 has no leading space (the trailer of
    # utterance 1 already spaced it) — but when it DOES, it must survive:
    h2 = Harness(reply="Replacement.")
    h2.formatter.format_delta(" intro")  # tail now "intro" (no trailing space)
    final2 = h2.speak_utterance(" more words.")
    assert final2.startswith(" ")  # separator space
    h2.emitter.begin_utterance = lambda: None  # keep baseline for assertion
    h2.coordinator.submit(final2)
    _wait_done(h2)
    h2.coordinator.poll()
    assert h2.device.screen.endswith(" Replacement. ")
    assert "  " not in h2.device.screen
    h.coordinator.shutdown()
    h2.coordinator.shutdown()


def test_cancel_on_new_utterance_discards_late_result():
    h = Harness(reply="LATE RESULT")
    h.transformer.release.clear()  # hold the LLM
    final = h.speak_utterance(" first utterance.")
    h.coordinator.submit(final)
    # User starts speaking again before the transform returns:
    h.coordinator.cancel_pending()
    second = h.speak_utterance(" second utterance.")
    h.transformer.release.set()  # LLM finishes late
    h.coordinator.poll()
    import time

    time.sleep(0.1)
    h.coordinator.poll()
    # The late result was discarded: raw text of both utterances intact.
    assert h.device.screen == "First utterance. Second utterance. "
    assert "LATE" not in h.device.screen
    assert second  # second utterance finalized normally
    h.coordinator.shutdown()


def test_failure_keeps_whisper_text_and_notifies_once(monkeypatch):
    toasts = []
    monkeypatch.setattr(transform_mod, "notify", lambda *a, **k: toasts.append(a))
    h = Harness(reply=TransformError("api down"))
    final = h.speak_utterance(" keep me.")
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    assert h.device.screen == "Keep me. "  # untouched
    assert h.errors  # surfaced to the event log

    # Second failure: error event again, but no second toast.
    final = h.speak_utterance(" again.")
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    failure_toasts = [t for t in toasts if "failed" in t[0]]
    assert len(failure_toasts) == 1
    h.coordinator.shutdown()


def test_reseed_drives_next_utterance_spacing():
    h = Harness(reply="Done now")  # replacement WITHOUT sentence punctuation
    final = h.speak_utterance(" done now.")
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    assert h.device.screen == "Done now"  # no trailer: no sentence end
    # Next utterance must space off the REPLACED text, not the original.
    nxt = h.speak_utterance(" and more.")
    assert h.device.screen == "Done now and more. "
    assert nxt.startswith(" ")
    h.coordinator.shutdown()


def test_noop_transform_types_nothing():
    h = Harness(reply="Same text.")
    final = h.speak_utterance(" same text.")
    ops_before = len(h.device.ops)
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    assert len(h.device.ops) == ops_before  # zero keystrokes
    h.coordinator.shutdown()


def test_drain_applies_in_budget_result():
    h = Harness(reply="Drained result.")
    h.transformer.release.clear()
    final = h.speak_utterance(" quick words.")
    h.coordinator.submit(final)
    # Session stops (toggle-off); result arrives within the budget.
    threading.Timer(0.05, h.transformer.release.set).start()
    h.coordinator.drain(timeout_s=2.0)
    assert h.device.screen == "Drained result. "
    h.coordinator.shutdown()


def test_drain_times_out_gracefully(monkeypatch):
    monkeypatch.setattr(transform_mod, "notify", lambda *a, **k: None)
    h = Harness(reply="never applied")
    h.transformer.release.clear()  # LLM hangs
    final = h.speak_utterance(" hang case.")
    h.coordinator.submit(final)
    h.coordinator.drain(timeout_s=0.1)
    assert h.device.screen == "Hang case. "  # raw text stays
    assert h.errors
    h.transformer.release.set()
    h.coordinator.shutdown()


def test_long_replacement_exceeds_default_cap():
    long_reply = "word " * 150  # ~750 chars > default 500 backspace cap
    h = Harness(reply=long_reply.strip() + ".")
    original = " " + "x" * 600 + "."
    final = h.speak_utterance(original)
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    # The transform path uses the raised cap: full replacement, no stale
    # prefix. (Capitalization of the replacement is the LLM's job — the
    # formatter only reseeds spacing state.)
    assert h.device.screen.startswith("word word")
    assert "x" not in h.device.screen
    h.coordinator.shutdown()


def test_prefetch_context_reused_by_transform():
    h = Harness(reply="Out.")
    h.coordinator.prefetch()
    final = h.speak_utterance(" in text.")
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    assert h.transformer.last_context == "FAKE-CONTEXT"  # no re-capture
    assert h.transformer.prewarmed == "FAKE-CONTEXT"  # LLM prewarmed
    h.coordinator.shutdown()


def test_stale_prefetch_not_reused_after_cancel():
    h = Harness(reply="Out.")
    h.coordinator.prefetch()
    h.coordinator.cancel_pending()  # new utterance: old context is stale
    final = h.speak_utterance(" fresh words.")
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    assert h.transformer.last_context is None  # falls back to fresh capture
    h.coordinator.shutdown()
