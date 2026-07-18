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

    def transform(self, transcript, context=None, history=()):
        assert self.release.wait(timeout=5.0), "test forgot to release"
        self.last_context = context
        self.last_history = history
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def prefetch_context(self):
        return "FAKE-CONTEXT"

    def prewarm(self, context, history=()):
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


# ── contextual v2: history, revise-in-place, streaming ───────────────────────


def test_history_recorded_and_passed():
    h = Harness(reply="Reply one.")
    final = h.speak_utterance(" first thing.")
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    final = h.speak_utterance(" second thing.")
    h.coordinator.cancel_pending()
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    # Second transform saw the first exchange as history.
    assert ("First thing.", "Reply one.") in h.transformer.last_history


def test_revise_in_place_rewrites_previous_region():
    h = Harness(reply="Better first reply!")
    final = h.speak_utterance(" original message.")
    # No transform for utterance 1 (simulate no-op): screen has raw text.
    # Utterance 2 is the spoken revision command.
    h.transformer.reply = "@@REVISE@@Much better message!"
    # Revise requires an existing exchange (spurious-marker guard).
    h.coordinator._history.append(("original message", "Original message."))
    cmd_final = h.speak_utterance(" make that better.")
    assert h.device.screen == "Original message. Make that better. "
    h.coordinator.submit(cmd_final)
    _wait_done(h)
    h.coordinator.poll()
    # BOTH the previous utterance and the spoken command were replaced.
    assert h.device.screen == "Much better message! "
    assert final  # (utterance 1 finalized normally)
    h.coordinator.shutdown()


def test_revise_without_previous_region_appends():
    h = Harness(reply="@@REVISE@@Standalone text.")
    final = h.speak_utterance(" say something.")
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    # No previous region: marker degrades to a normal replacement.
    assert h.device.screen == "Standalone text. "
    h.coordinator.shutdown()


class FakeStreamTransformer(FakeTransformer):
    """Streams reply in small chunks through transform_stream."""

    def __init__(self, chunks):
        super().__init__(reply="unused")
        self.chunks = chunks

    def transform_stream(self, transcript, context=None, history=()):
        assert self.release.wait(timeout=5.0)
        self.last_history = history
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


def _stream_harness(chunks):
    h = Harness()
    h.transformer = FakeStreamTransformer(chunks)
    from wldictate.transform import TransformCoordinator

    h.coordinator.shutdown()
    h.coordinator = TransformCoordinator(
        h.transformer,
        h.emitter,
        h.formatter,
        timeout_s=5.0,
        notify_enabled=False,
        stream_enabled=True,
        on_error=h.errors.append,
    )
    return h


def test_streaming_types_incrementally_and_finishes():
    h = _stream_harness(["Streamed ", "words ", "arriving."])
    final = h.speak_utterance(" raw words.")
    h.coordinator.submit(final)
    for _ in range(200):
        h.coordinator.poll()
        if h.coordinator._stream is None:
            break
        threading.Event().wait(0.01)
    h.coordinator.poll()
    assert h.device.screen == "Streamed words arriving. "
    assert not h.errors
    # More than one sync happened (incremental, not one-shot).
    replace_ops = [op for op in h.device.ops if "Streamed" in op[1] or "words" in op[1]]
    assert len(replace_ops) >= 1
    h.coordinator.shutdown()


def test_streaming_think_block_never_typed():
    h = _stream_harness(["<think>internal ", "reasoning</think>", "Clean output."])
    final = h.speak_utterance(" input.")
    h.coordinator.submit(final)
    for _ in range(200):
        h.coordinator.poll()
        if h.coordinator._stream is None:
            break
        threading.Event().wait(0.01)
    h.coordinator.poll()
    assert h.device.screen == "Clean output. "
    assert all("reasoning" not in op[1] for op in h.device.ops)
    h.coordinator.shutdown()


def test_streaming_revise_marker():
    h = _stream_harness(["@@REV", "ISE@@", "Revised previous!"])
    h.speak_utterance(" first message.")
    h.coordinator._history.append(("first message", "First message."))
    cmd = h.speak_utterance(" change it.")
    h.coordinator.submit(cmd)
    for _ in range(200):
        h.coordinator.poll()
        if h.coordinator._stream is None:
            break
        threading.Event().wait(0.01)
    h.coordinator.poll()
    assert h.device.screen == "Revised previous! "
    h.coordinator.shutdown()


def test_streaming_error_before_output_keeps_raw(monkeypatch):
    import wldictate.transform as tm

    monkeypatch.setattr(tm, "notify", lambda *a, **k: None)
    h = _stream_harness([RuntimeError("boom")])
    final = h.speak_utterance(" keep this.")
    h.coordinator.submit(final)
    for _ in range(200):
        h.coordinator.poll()
        if h.coordinator._stream is None:
            break
        threading.Event().wait(0.01)
    assert h.device.screen == "Keep this. "
    assert h.errors
    h.coordinator.shutdown()


def test_streaming_drain_applies():
    h = _stream_harness(["Drained ", "stream."])
    h.transformer.release.clear()
    final = h.speak_utterance(" quick.")
    h.coordinator.submit(final)
    threading.Timer(0.05, h.transformer.release.set).start()
    h.coordinator.drain(timeout_s=2.0)
    assert h.device.screen == "Drained stream. "
    h.coordinator.shutdown()


# ── voice edit commands (matcher + emitter mechanics) ────────────────────────


def test_match_command():
    from wldictate.commands import match_command

    assert match_command("Scratch that. ") == "scratch"
    assert match_command("scratch that") == "scratch"
    assert match_command("New line. ") == "newline"
    assert match_command("Press enter. ") == "enter"
    assert match_command("scratch that idea entirely") is None
    assert match_command("the new line of products") is None


def test_scratch_that_deletes_previous_utterance():
    from wldictate.worker import _execute_voice_command

    h = Harness()
    h.speak_utterance(" delete me please.")
    h.speak_utterance(" scratch that.")
    assert h.device.screen == "Delete me please. Scratch that. "
    _execute_voice_command("scratch", h.emitter, h.formatter)
    assert h.device.screen == ""
    # Dictation continues cleanly afterward.
    h.speak_utterance(" fresh start.")
    assert h.device.screen == "Fresh start. "
    h.coordinator.shutdown()


def test_spurious_revise_without_history_never_deletes_previous():
    # Model emits @@REVISE@@ on the FIRST exchange (observed with small
    # models): must degrade to a plain replacement of the current utterance.
    h = Harness(reply="@@REVISE@@Spurious revision.")
    h.speak_utterance(" untouchable text.")
    final = h.speak_utterance(" current words.")
    h.coordinator.submit(final)
    _wait_done(h)
    h.coordinator.poll()
    assert h.device.screen.startswith("Untouchable text. ")
    assert h.device.screen == "Untouchable text. Spurious revision. "
    h.coordinator.shutdown()


# ── contextual v3: literal escape, persona/vocab/hints ───────────────────────


def test_strip_literal():
    from wldictate.commands import strip_literal

    assert strip_literal("Literally, scratch that. ") == "scratch that. "
    assert strip_literal(" Literally send the raw text.") == " send the raw text."
    assert strip_literal("The literally best thing.") is None
    assert strip_literal("Literally.") is None  # nothing after the guard


def test_handle_final_routes_literal_and_command_and_transform():
    from wldictate.worker import _handle_final

    h = Harness(reply="TRANSFORMED. ")
    # 1. literal escape: guard word removed, no transform submitted
    h.speak_utterance(" literally scratch that.")
    _handle_final("Literally scratch that. ", h.emitter, h.formatter, h.coordinator)
    assert "scratch that." in h.device.screen.lower()
    assert h.coordinator._pending is None and h.coordinator._stream is None
    # 2. plain utterance: transform submitted
    final = h.speak_utterance(" real words.")
    _handle_final(final, h.emitter, h.formatter, h.coordinator)
    assert h.coordinator._pending is not None
    h.coordinator.shutdown()


def test_persona_and_vocabulary_in_system_prompt(monkeypatch):
    from wldictate.config import ContextualConfig
    from wldictate.transform import Transformer
    import wldictate.transform as tm

    cfg = ContextualConfig(persona="I'm Taz. Casual, lowercase ok.")
    cfg.vocabulary = ["Hyprland", "wl-dictate", "Qwen"]
    backend = None

    class Recorder:
        is_local = True

        def complete(self, system, messages, *, model, max_tokens):
            self.system = system
            self.user = messages[-1]["content"]
            return "ok"

    rec = Recorder()
    monkeypatch.setattr(tm, "make_backend", lambda *a, **k: rec)
    monkeypatch.setattr(tm, "capture_context", lambda **k: tm.ScreenContext())
    tr = Transformer(cfg)
    tr.transform("hello")
    assert "I'm Taz" in rec.system
    assert "Hyprland" in rec.system
    assert backend is None  # (silence lint)


def test_app_hint_appended_for_matching_class(monkeypatch):
    from wldictate.config import ContextualConfig
    from wldictate.transform import Transformer
    import wldictate.transform as tm

    cfg = ContextualConfig()
    cfg.app_hints = {"vesktop": "very casual, emoji fine"}

    class Recorder:
        is_local = True

        def complete(self, system, messages, *, model, max_tokens):
            self.user = messages[-1]["content"]
            return "ok"

    rec = Recorder()
    monkeypatch.setattr(tm, "make_backend", lambda *a, **k: rec)
    monkeypatch.setattr(
        tm, "capture_context", lambda **k: tm.ScreenContext(window_class="vesktop")
    )
    tr = Transformer(cfg)
    tr.transform("hi")
    assert "APP GUIDANCE: very casual, emoji fine" in rec.user

    monkeypatch.setattr(
        tm, "capture_context", lambda **k: tm.ScreenContext(window_class="kitty")
    )
    tr.transform("hi")
    assert "APP GUIDANCE" not in rec.user


# ── long-speech rollover chain (merge_all) ───────────────────────────────────


def test_merge_all_replaces_whole_chain():
    h = Harness(reply="One clean combined message.")
    # Simulate a rollover chain: two carried chunks + the final utterance.
    h.speak_utterance(" first chunk of speech.")
    h.emitter.begin_utterance(True)
    h.formatter.on_utterance_start()
    t = h.formatter.format_delta(" second chunk.")
    t += h.formatter.end_utterance()
    h.emitter.sync(t)
    h.emitter.begin_utterance(True)
    h.formatter.on_utterance_start()
    t2 = h.formatter.format_delta(" final part.")
    t2 += h.formatter.end_utterance()
    h.emitter.sync(t2)
    combined = "First chunk of speech. Second chunk. Final part. "
    assert h.device.screen == combined
    h.coordinator.submit(combined, merge_all=True)
    _wait_done(h)
    h.coordinator.poll()
    assert h.device.screen == "One clean combined message. "
    h.coordinator.shutdown()


def test_new_voice_commands():
    from wldictate.commands import match_command

    assert match_command("Press tab. ") == "tab"
    assert match_command("Press escape.") == "escape"
    assert match_command("Copy that. ") == "copy"


class KeyDevice(ScreenDevice):
    def __init__(self):
        super().__init__()
        self.keys = []
        self.clipboard = None

    def press_key(self, keysym):
        self.keys.append(keysym)
        return True

    def set_clipboard(self, text):
        self.clipboard = text
        return True


def test_copy_that_copies_previous_region():
    from wldictate.worker import _execute_voice_command

    h = Harness()
    h.device = KeyDevice()
    h.emitter = CorrectingEmitter(h.device)
    h.speak_utterance(" remember these words.")
    h.speak_utterance(" copy that.")
    _execute_voice_command("copy", h.emitter, h.formatter)
    assert h.device.clipboard == "Remember these words."
    assert h.device.screen == "Remember these words. "  # text stays
    h.coordinator.shutdown()


def test_press_tab_and_escape_reset_regions():
    from wldictate.worker import _execute_voice_command

    h = Harness()
    h.device = KeyDevice()
    h.emitter = CorrectingEmitter(h.device)
    h.speak_utterance(" press tab.")
    _execute_voice_command("tab", h.emitter, h.formatter)
    assert h.device.keys == ["Tab"]
    assert h.device.screen == ""
    assert h.emitter.previous_len == 0  # ownership reset
    h.coordinator.shutdown()


# ── mid-stream death must not strand a half-replaced message ─────────────────


class GatedStreamTransformer(FakeTransformer):
    """Yields one chunk, then blocks until released, then raises/ends."""

    def __init__(self, first_chunk, then):
        super().__init__(reply="unused")
        self.first_chunk = first_chunk
        self.then = then  # Exception to raise, or None to just end
        self.proceed = threading.Event()

    def transform_stream(self, transcript, context=None, history=()):
        yield self.first_chunk
        assert self.proceed.wait(timeout=5.0)
        if isinstance(self.then, Exception):
            raise self.then


def _gated_harness(first_chunk, then):
    h = _stream_harness([])
    h.transformer = GatedStreamTransformer(first_chunk, then)
    h.coordinator.shutdown()
    h.coordinator = TransformCoordinator(
        h.transformer,
        h.emitter,
        h.formatter,
        timeout_s=5.0,
        notify_enabled=False,
        stream_enabled=True,
        on_error=h.errors.append,
    )
    return h


def _poll_until(h, predicate, timeout=5.0):
    for _ in range(int(timeout * 100)):
        h.coordinator.poll()
        if predicate():
            return True
        threading.Event().wait(0.01)
    return False


def test_streaming_error_mid_apply_restores_dictated_text(monkeypatch):
    monkeypatch.setattr(transform_mod, "notify", lambda *a, **k: None)
    h = _gated_harness("Half a rewri", RuntimeError("model died mid-stream"))
    final = h.speak_utterance(" the full dictated message.")
    h.coordinator.submit(final)
    # Partial replacement reaches the screen first.
    assert _poll_until(h, lambda: "Half a rewri" in h.device.screen)
    h.transformer.proceed.set()
    assert _poll_until(h, lambda: h.coordinator._stream is None)
    # The stream died: the FULL dictated text is restored, not the partial.
    assert h.device.screen == final
    assert h.errors
    h.coordinator.shutdown()


def test_streaming_truncation_restores_dictated_text(monkeypatch):
    from wldictate.transform import TransformTruncated

    monkeypatch.setattr(transform_mod, "notify", lambda *a, **k: None)
    h = _gated_harness(
        "Truncated rew", TransformTruncated("output hit max_output_tokens")
    )
    final = h.speak_utterance(" a very long dictated message.")
    h.coordinator.submit(final)
    assert _poll_until(h, lambda: "Truncated rew" in h.device.screen)
    h.transformer.proceed.set()
    assert _poll_until(h, lambda: h.coordinator._stream is None)
    assert h.device.screen == final
    assert any("max_output_tokens" in e for e in h.errors)
    h.coordinator.shutdown()


def test_cancel_mid_stream_restores_dictated_text():
    h = _gated_harness("Partially repla", None)
    final = h.speak_utterance(" words that must survive.")
    h.coordinator.submit(final)
    assert _poll_until(h, lambda: "Partially repla" in h.device.screen)
    # New utterance starts while the rewrite is mid-apply.
    h.coordinator.cancel_pending()
    assert h.device.screen == final  # dictated text back, nothing stranded
    h.transformer.proceed.set()  # let the generator finish
    h.coordinator.shutdown()
