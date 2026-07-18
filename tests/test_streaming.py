import time

import numpy as np

from wldictate.emitter import CorrectingEmitter, Emitter
from wldictate.streaming import SAMPLE_RATE, StreamingSession
from wldictate.textproc import TextFormatter
from wldictate.transcriber import FakeTranscriber, Word


class CaptureEmitter(Emitter):
    def __init__(self):
        self.out = []

    def emit(self, text):
        self.out.append(text)
        return True


def W(text, start, end, prob=0.9):
    return Word(text, start, end, prob)


class Harness:
    """Drives a StreamingSession with a fake clock and scripted decodes."""

    def __init__(self, script, **session_kw):
        self.fake = FakeTranscriber(script)
        self.emitter = CaptureEmitter()
        self.commits = []
        self.t = 0.0
        kw = dict(infer_interval_s=0.5, min_new_audio_s=0.1, min_speech_s=0.1)
        kw.update(session_kw)
        self.session = StreamingSession(
            self.fake,
            # Capitalization off: these tests assert on the lowercase fixture
            # words to exercise agreement/commit/trim logic, not casing.
            TextFormatter(capitalize_sentences=False),
            self.emitter,
            on_commit=self.commits.append,
            clock=lambda: self.t,
            **kw,
        )

    def speak(self, seconds, step=0.1):
        """Feed audio in `step`-second chunks, ticking (and settling) each step."""
        n = int(seconds / step)
        chunk = np.zeros(int(step * SAMPLE_RATE), dtype=np.float32)
        for _ in range(n):
            self.session.feed([chunk])
            self.t += step
            self.session.tick()
            self.settle()
            self.session.tick()

    def settle(self):
        deadline = time.monotonic() + 2.0
        while self.session._inflight is not None and time.monotonic() < deadline:
            if self.session._inflight[0].done():
                break
            time.sleep(0.001)

    def text(self):
        return "".join(self.emitter.out)


def test_local_agreement_commits_stable_prefix():
    script = [
        [W(" a", 0, 0.2), W(" b", 0.2, 0.4)],
        [W(" a", 0, 0.2), W(" b", 0.2, 0.4), W(" c", 0.4, 0.6)],
        [W(" a", 0, 0.2), W(" b", 0.2, 0.4), W(" c", 0.4, 0.6), W(" d", 0.6, 0.8)],
    ]
    h = Harness(script)
    h.session.start_utterance()
    h.speak(2.0)
    h.session.finalize()
    assert h.text() == "a b c d"
    # Nothing re-emitted: commits are disjoint.
    assert h.commits == ["a b", " c", " d"]


def test_unstable_words_not_committed_until_agreement():
    script = [
        [W(" the", 0, 0.2), W(" quit", 0.2, 0.5)],
        [W(" the", 0, 0.2), W(" quick", 0.2, 0.5)],   # 'quit' -> 'quick': no agree yet
        [W(" the", 0, 0.2), W(" quick", 0.2, 0.5)],   # agree now
    ]
    h = Harness(script)
    h.session.start_utterance()
    h.speak(2.0)
    # after 3 interim decodes: "the quick" committed, never "quit"
    assert "quit " not in h.text()
    assert h.text().startswith("the")
    h.session.finalize()
    assert h.text() == "the quick"


def test_finalize_commits_tail_without_agreement():
    script = [
        [W(" one", 0, 0.3)],
        [W(" one", 0, 0.3), W(" two", 0.3, 0.6)],
    ]
    h = Harness(script)
    h.session.start_utterance()
    h.speak(1.0)
    h.session.finalize()  # final decode repeats last script entry
    assert h.text() == "one two"


def test_short_utterance_skipped():
    h = Harness([[W(" noise", 0, 0.1)]], min_speech_s=0.5)
    h.session.start_utterance()
    h.session.feed([np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)])
    h.session.finalize()
    assert h.text() == ""
    # No decode should have run at all.
    assert h.fake.calls == []


def test_backpressure_skips_ticks_while_decode_inflight():
    class SlowFake(FakeTranscriber):
        def transcribe(self, audio, *, final=False, prompt=None):
            time.sleep(0.05)
            return super().transcribe(audio, final=final, prompt=prompt)

    h = Harness([])
    h.session._transcriber = slow = SlowFake([[W(" x", 0, 0.2)]])
    h.session.start_utterance()
    chunk = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
    # Tick rapidly with fake time advancing: only one decode may be in flight.
    for _ in range(20):
        h.session.feed([chunk])
        h.t += 0.1
        h.session.tick()
    inflight_calls = len(slow.calls)
    assert inflight_calls <= 2  # first decode plus at most one after it finished
    h.settle()
    h.session.finalize()


def test_buffer_trim_bounds_decode_cost_and_keeps_text():
    """Long utterance: buffer is trimmed, text stays ordered and gapless.

    Uses a timeline-aware fake: like a real decoder, it transcribes whatever
    audio window it is given. The engine's buffer is always a suffix of the
    stream, so the window start is (total fed) - (window length). One "word"
    w<j> is spoken every 0.5s of stream time.
    """
    h = Harness([], max_buffer_s=6.0)

    class TimelineFake(FakeTranscriber):
        def transcribe(self, audio, *, final=False, prompt=None):
            self.calls.append({"samples": len(audio), "final": final, "prompt": prompt})
            start = (h.fed_samples - len(audio)) / SAMPLE_RATE
            end = h.fed_samples / SAMPLE_RATE
            words = []
            j = 0
            while j * 0.5 + 0.4 <= end:
                if j * 0.5 >= start:
                    words.append(W(f" w{j}", j * 0.5 - start, j * 0.5 + 0.4 - start))
                j += 1
            return words

    h.fed_samples = 0
    orig_feed = h.session.feed

    def counting_feed(frames):
        h.fed_samples += sum(len(f) for f in frames)
        orig_feed(frames)

    h.session.feed = counting_feed
    h.session._transcriber = TimelineFake([])

    h.session.start_utterance()
    h.speak(15.0)
    # Buffer must have been trimmed below the cap (plus slack for the audio
    # accumulated since the last decode).
    assert len(h.session._buffer) <= (6.0 + 1.5) * SAMPLE_RATE
    h.session.finalize()
    # Committed text is the full word sequence, in order, no dupes, no gaps.
    words = h.text().split()
    assert len(words) >= 25
    assert words == [f"w{i}" for i in range(len(words))]


def test_stale_decode_from_previous_utterance_dropped():
    script = [
        [W(" old", 0, 0.3)],
        [W(" old", 0, 0.3)],
        [W(" new", 0, 0.3)],
        [W(" new", 0, 0.3)],
    ]
    h = Harness(script)
    h.session.start_utterance()
    h.speak(0.6)
    h.session.finalize()
    first = h.text()
    h.session.start_utterance()
    h.speak(0.6)
    h.session.finalize()
    assert first.strip() == "old"
    assert h.text() == "old new"


def test_decode_error_reported_not_fatal():
    class BoomFake(FakeTranscriber):
        def transcribe(self, audio, *, final=False, prompt=None):
            raise RuntimeError("boom")

    errors = []
    h = Harness([])
    h.session._transcriber = BoomFake([])
    h.session._on_error = errors.append
    h.session.start_utterance()
    h.speak(1.0)
    h.session.finalize()
    assert errors  # surfaced
    assert h.text() == ""  # nothing typed


def test_prompt_carries_committed_context():
    script = [
        [W(" alpha", 0, 0.3), W(" beta", 0.3, 0.6)],
        [W(" alpha", 0, 0.3), W(" beta", 0.3, 0.6), W(" gamma", 0.6, 0.9)],
        [W(" alpha", 0, 0.3), W(" beta", 0.3, 0.6), W(" gamma", 0.6, 0.9)],
    ]
    h = Harness(script)
    h.session.start_utterance()
    h.speak(2.0)
    h.session.finalize()
    prompts = [c["prompt"] for c in h.fake.calls]
    assert prompts[0] is None
    assert any(p and "alpha" in p for p in prompts[1:])


# ── correcting mode (live rewrite) ───────────────────────────────────────────


class ScreenDevice(Emitter):
    """Fake device that models the actual display: applies backspaces/text to
    a screen string and records every (backspaces, text) op."""

    def __init__(self):
        self.ops = []
        self.screen = ""
        self.fail = False

    def emit(self, text):
        return self.rewrite(0, text) is not None

    def rewrite(self, backspaces, text):
        if self.fail:
            return None
        self.ops.append((backspaces, text))
        keep = len(self.screen) - backspaces
        self.screen = self.screen[:keep] + text
        return text


class CorrectingHarness(Harness):
    """Harness variant running the engine in correcting mode against a
    real CorrectingEmitter over a ScreenDevice."""

    def __init__(self, script, **session_kw):
        session_kw.setdefault("correcting", True)
        super().__init__(script, **session_kw)
        self.device = ScreenDevice()
        self.session._emitter = CorrectingEmitter(self.device)

    def text(self):
        return self.device.screen


def test_correcting_tentative_tail_visible_before_agreement():
    script = [
        [W(" the", 0, 0.2), W(" quit", 0.2, 0.5)],
        [W(" the", 0, 0.2), W(" quick", 0.2, 0.5)],
    ]
    h = CorrectingHarness(script)
    h.session.start_utterance()
    h.speak(2.0)
    # The unstable word was on screen immediately (append-only mode never
    # shows it — see test_unstable_words_not_committed_until_agreement).
    assert h.device.ops[0] == (0, "the quit")
    # …then visibly fixed in place via backspaces.
    assert (1, "ck") in h.device.ops
    h.session.finalize()
    assert h.text() == "the quick"
    assert h.commits == ["the quick"]  # exactly one commit, at finalize


def test_correcting_empty_redecode_clears_screen():
    script = [
        [W(" oops", 0, 0.3)],
        [],
    ]
    h = CorrectingHarness(script)
    h.session.start_utterance()
    h.speak(1.5)
    h.session.finalize()
    assert h.text() == ""
    assert (4, "") in h.device.ops  # the retraction was typed, then erased
    assert h.commits == []


def test_correcting_finalize_appends_trailer_and_locks_utterance():
    script = [
        [W(" done", 0, 0.4)],
        [W(" done.", 0, 0.4)],
    ]
    h = CorrectingHarness(script)
    h.session.start_utterance()
    h.speak(1.0)
    h.session.finalize()  # final decode repeats " done."
    assert h.text() == "done. "  # sentence trailer included in the final sync
    ops_before = len(h.device.ops)
    h.session._transcriber = FakeTranscriber([[W(" more", 0, 0.4)]])
    h.session.start_utterance()
    h.speak(1.0)
    h.session.finalize()
    assert h.text() == "done. more"
    # The second utterance never backspaced into the finalized first one.
    for n, _ in h.device.ops[ops_before:]:
        assert n <= len("more")


def test_correcting_trim_keeps_full_text_on_screen():
    """Correcting analog of the buffer-trim test: trimmed words survive on
    screen via _trimmed_raw."""
    h = CorrectingHarness([], max_buffer_s=6.0)

    class TimelineFake(FakeTranscriber):
        def transcribe(self, audio, *, final=False, prompt=None):
            self.calls.append({"samples": len(audio), "final": final, "prompt": prompt})
            start = (h.fed_samples - len(audio)) / SAMPLE_RATE
            end = h.fed_samples / SAMPLE_RATE
            words = []
            j = 0
            while j * 0.5 + 0.4 <= end:
                if j * 0.5 >= start:
                    words.append(W(f" w{j}", j * 0.5 - start, j * 0.5 + 0.4 - start))
                j += 1
            return words

    h.fed_samples = 0
    orig_feed = h.session.feed

    def counting_feed(frames):
        h.fed_samples += sum(len(f) for f in frames)
        orig_feed(frames)

    h.session.feed = counting_feed
    h.session._transcriber = TimelineFake([])

    h.session.start_utterance()
    h.speak(15.0)
    assert len(h.session._buffer) <= (6.0 + 1.5) * SAMPLE_RATE  # trim happened
    h.session.finalize()
    words = h.text().split()
    assert len(words) >= 25
    assert words == [f"w{i}" for i in range(len(words))]


def test_correcting_sync_failure_freezes_utterance_then_recovers():
    script = [
        [W(" hello", 0, 0.3)],
        [W(" hello", 0, 0.3), W(" world", 0.3, 0.6)],
        [W(" again", 0, 0.3)],
    ]
    errors = []
    h = CorrectingHarness(script)
    h.session._on_error = errors.append
    h.session.start_utterance()
    h.speak(1.0)  # first decode lands
    h.device.fail = True
    h.speak(1.0)  # second decode: sync fails
    assert errors
    h.device.fail = False
    ops_after_fail = len(h.device.ops)
    h.speak(1.0)
    h.session.finalize()  # skips the final sync: screen state unknown
    assert len(h.device.ops) == ops_after_fail
    # Next utterance recovers (fresh baseline).
    h.session.start_utterance()
    h.speak(1.0)
    h.session.finalize()
    assert h.text().endswith("again")


def test_correcting_final_decode_failure_keeps_tentative_text():
    class FinalBoom(FakeTranscriber):
        def transcribe(self, audio, *, final=False, prompt=None):
            if final:
                raise RuntimeError("boom")
            return super().transcribe(audio, final=final, prompt=prompt)

    errors = []
    h = CorrectingHarness([])
    h.session._transcriber = FinalBoom([[W(" one", 0, 0.3)]])
    h.session._on_error = errors.append
    h.session.start_utterance()
    h.speak(1.0)
    h.session.finalize()
    assert errors
    assert h.text() == "one"  # tentative text never deleted
    # Formatter state advanced to match the screen: next utterance spaces off.
    h.session._transcriber = FakeTranscriber([[W(" two", 0, 0.3)]])
    h.session.start_utterance()
    h.speak(1.0)
    h.session.finalize()
    assert h.text() == "one two"


def test_correcting_finalize_returns_final_text():
    script = [[W(" hello", 0, 0.3)], [W(" hello.", 0, 0.3)]]
    h = CorrectingHarness(script)
    h.session.start_utterance()
    h.speak(1.0)
    final = h.session.finalize()
    assert final == "hello. "  # text + trailer, exactly what's on screen
    assert h.text() == "hello. "


def test_commit_mode_finalize_returns_none():
    h = Harness([[W(" hello", 0, 0.3)]])
    h.session.start_utterance()
    h.speak(1.0)
    assert h.session.finalize() is None


def test_correcting_finalize_returns_none_after_sync_failure():
    # Three distinct hypotheses so a decode with NEW text (a real device
    # write, not a no-op diff) still lands after the device starts failing.
    h = CorrectingHarness(
        [
            [W(" one", 0, 0.3)],
            [W(" one two", 0, 0.6)],
            [W(" one two three", 0, 0.9)],
        ]
    )
    h.session.start_utterance()
    h.speak(1.0)
    h.device.fail = True
    h.speak(1.0)  # a decode syncs -> fails -> _render_ok False
    assert h.session.finalize() is None  # screen state unknown: no transform


# ── speculative finalize + adaptive cadence ──────────────────────────────────


def test_speculative_finalize_reuses_decode():
    script = [[W(" hello", 0, 0.3)], [W(" hello there.", 0, 0.6)]]
    h = CorrectingHarness(script)
    h.session.start_utterance()
    h.speak(1.0)  # one interim decode
    h.session.speculate_final()
    assert h.session._speculative is not None
    h.session._speculative[0].result(timeout=5.0)  # let the decode land
    calls_before = len(h.fake.calls)
    final = h.session.finalize()
    # finalize used the speculative decode: no new transcribe call.
    assert len(h.fake.calls) == calls_before
    assert h.fake.calls[-1]["final"] is True
    assert final == "hello there. "


def test_cancelled_speculation_falls_back_to_fresh_decode():
    script = [[W(" one", 0, 0.3)]]
    h = CorrectingHarness(script)
    h.session.start_utterance()
    h.speak(1.0)
    h.session.speculate_final()
    spec_future = h.session._speculative[0]
    h.session.cancel_speculation()  # speech resumed
    spec_future.result(timeout=5.0)  # discarded decode finishes in background
    calls_before = len(h.fake.calls)
    h.session.finalize()
    assert len(h.fake.calls) == calls_before + 1  # fresh final decode ran


def test_tick_suppressed_while_speculating():
    class Slow(FakeTranscriber):
        def transcribe(self, audio, *, final=False, prompt=None):
            time.sleep(0.05)
            return super().transcribe(audio, final=final, prompt=prompt)

    h = Harness([])
    h.session._transcriber = Slow([[W(" x", 0, 0.2)]])
    h.session.start_utterance()
    h.speak(1.0)
    h.settle()
    h.session.speculate_final()
    calls = len(h.session._transcriber.calls)
    h.speak(1.0)  # interim ticks would normally fire here
    assert len(h.session._transcriber.calls) == calls  # suppressed
    h.session.finalize()


def test_adaptive_interval_floors_at_min():
    # FakeTranscriber decodes instantly -> effective interval hits the floor.
    script = [[W(f" w{i}", 0, 0.2)] for i in range(20)]
    fast = Harness(script, min_infer_interval_s=0.2)
    fast.session.start_utterance()
    fast.speak(2.0)
    fast_calls = len(fast.fake.calls)
    fixed = Harness(script)  # no floor: fixed 0.5s cadence
    fixed.session.start_utterance()
    fixed.speak(2.0)
    assert fast_calls > len(fixed.fake.calls)  # tighter cadence, more decodes
    fast.session.finalize()
    fixed.session.finalize()


# ── Tentative-tail confidence gating ─────────────────────────────────────────


def test_low_confidence_tail_held_until_confirmed():
    script = [
        [W(" the", 0, 0.2), W(" quix", 0.2, 0.5, prob=0.1)],
        [W(" the", 0, 0.2), W(" quix", 0.2, 0.5, prob=0.1)],  # survives a decode
    ]
    h = CorrectingHarness(script)
    h.session.start_utterance()
    h.speak(2.0)
    # First render held the unconfirmed low-confidence word back...
    assert h.device.ops[0] == (0, "the")
    # ...then rendered it once a second decode agreed.
    assert h.text().endswith("quix")
    h.session.finalize()
    assert h.text() == "the quix"


def test_final_decode_renders_low_confidence_words():
    script = [[W(" um", 0, 0.3, prob=0.05)]]
    h = CorrectingHarness(script)
    h.session.start_utterance()
    # Short enough that only ONE interim decode runs: the unconfirmed
    # low-confidence word stays gated...
    h.speak(0.4)
    assert h.text() == ""  # gated during the live render
    h.session.finalize()
    assert h.text() == "um"  # ...but the final decode renders everything


def test_confidence_gate_disabled_renders_everything():
    script = [[W(" um", 0, 0.3, prob=0.05)]]
    h = CorrectingHarness(script, tail_confidence_min=0.0)
    h.session.start_utterance()
    h.speak(1.0)
    assert h.text() == "um"


def test_confidence_gate_only_trims_the_tail():
    # A low-prob word FOLLOWED by a high-prob word is not hidden (only a
    # contiguous unconfirmed tail may be trimmed).
    script = [[W(" foo", 0, 0.2, prob=0.1), W(" bar", 0.2, 0.4, prob=0.9)]]
    h = CorrectingHarness(script)
    h.session.start_utterance()
    h.speak(1.0)
    assert h.text() == "foo bar"
    h.session.finalize()


# ── First-paint fast path ────────────────────────────────────────────────────


def test_first_decode_fires_at_min_speech_without_interval_wait():
    h = Harness([[W(" hi", 0, 0.2)]], min_speech_s=0.1, min_new_audio_s=0.3)
    h.session.start_utterance()
    # Feed just min_speech worth of audio with NO clock advance: the interval
    # and min_new gates would both normally block this decode.
    h.session.feed([np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)])
    h.session.tick()
    assert h.session._inflight is not None  # first decode submitted instantly
    h.settle()
    h.session.finalize()
    assert h.text() == "hi"


def test_first_decode_still_waits_for_min_speech():
    h = Harness([[W(" hi", 0, 0.2)]], min_speech_s=0.5)
    h.session.start_utterance()
    h.session.feed([np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)])
    h.session.tick()
    assert h.session._inflight is None  # below min_speech: no decode yet


def test_subsequent_decodes_keep_normal_pacing():
    h = Harness(
        [[W(" a", 0, 0.2)], [W(" a", 0, 0.2), W(" b", 0.2, 0.4)]],
        min_speech_s=0.1,
        min_new_audio_s=0.3,
    )
    h.session.start_utterance()
    h.session.feed([np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)])
    h.session.tick()  # fast-path first decode
    h.settle()
    h.session.tick()  # drain result
    calls = len(h.fake.calls)
    # Tiny new audio + no time advance: the second decode must NOT fire.
    h.session.feed([np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.float32)])
    h.session.tick()
    assert len(h.fake.calls) == calls
    h.session.finalize()


def test_fast_path_resets_per_utterance():
    h = Harness(
        [[W(" one", 0, 0.2)], [W(" one", 0, 0.2)], [W(" two", 0, 0.2)]],
        min_speech_s=0.1,
        min_new_audio_s=0.3,
    )
    h.session.start_utterance()
    h.session.feed([np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)])
    h.session.tick()
    h.settle()
    h.session.finalize()
    h.session.start_utterance()
    h.session.feed([np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)])
    h.session.tick()
    assert h.session._inflight is not None  # fast path again for utterance 2
    h.settle()
    h.session.finalize()
