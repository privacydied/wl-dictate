"""Realtime streaming transcription engine (LocalAgreement-2).

While an utterance is in progress, the engine re-decodes the utterance buffer
every ``infer_interval_s`` and *commits* the longest prefix of words on which
two consecutive hypotheses agree.

Two typing modes share the engine:

- **commit** (append-only): committed text is emitted immediately and never
  revised — words appear roughly a second behind speech.
- **correcting** (live rewrite): after every decode the *full* current
  hypothesis (stable prefix + tentative tail) is rendered and synced to the
  screen through a ``CorrectingEmitter`` — raw words appear instantly and are
  visibly fixed in place as the hypothesis refines; the final decode replaces
  the tail once more and the utterance becomes immutable.

Decodes run on a single background thread (CTranslate2 models must not be
called concurrently); if a decode is still in flight when the next tick
arrives, the tick is skipped — natural backpressure with no queue growth.

All engine logic is synchronous and injectable (transcriber, emitter, clock),
so it is fully unit-testable without a GPU, model, or microphone.
"""

from __future__ import annotations

import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import numpy as np

from .transcriber import Transcriber, Word
from .textproc import TextFormatter
from .emitter import Emitter

SAMPLE_RATE = 16000

_SENTENCE_END = (".", "!", "?")
_PROMPT_TAIL_CHARS = 200
#: Budget for one final decode. The decode window is bounded by
#: streaming.max_buffer_s (≤ 30 s of audio); a machine that can't decode
#: that in 10 s can't run this pipeline in real time anyway — waiting
#: longer would only wedge finalize(). Public because the worker derives
#: its session-thread join timeout from it: finalize() can wait for at most
#: TWO of these back-to-back (a timed-out speculative decode followed by a
#: fresh one), so worst-case finalize ≈ 2 × FINAL_DECODE_TIMEOUT_S.
FINAL_DECODE_TIMEOUT_S = 10.0


def _normalize(text: str) -> str:
    """Normalization for word agreement: compare by content, not rendering."""
    text = unicodedata.normalize("NFKC", text).strip().casefold()
    return text.strip(".,!?;:\"'()[]{}…-–—")


class StreamingSession:
    """One engine instance per audio session; handles many utterances."""

    def __init__(
        self,
        transcriber: Transcriber,
        formatter: TextFormatter,
        emitter: Emitter,
        *,
        infer_interval_s: float = 0.5,
        min_infer_interval_s: float | None = None,
        min_new_audio_s: float = 0.3,
        max_buffer_s: float = 12.0,
        min_speech_s: float = 0.3,
        streaming_enabled: bool = True,
        correcting: bool = False,
        tail_confidence_min: float = 0.3,
        on_commit: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transcriber = transcriber
        self._formatter = formatter
        self._emitter = emitter
        self._interval = infer_interval_s
        # Adaptive cadence: with a floor set, the effective interval tracks
        # 1.5x the measured decode time (fast model + short buffer -> tighter
        # ticks; long buffer -> natural backoff). None = fixed interval.
        self._min_interval = min_infer_interval_s
        self._last_decode_duration = 0.0
        self._min_new = int(min_new_audio_s * SAMPLE_RATE)
        self._max_buffer = int(max_buffer_s * SAMPLE_RATE)
        self._min_speech = int(min_speech_s * SAMPLE_RATE)
        self._streaming_enabled = streaming_enabled
        self._correcting = correcting
        # Live-render confidence gate: trailing words below this probability
        # are held back until they survive a second decode. The words that
        # visibly flicker are exactly the low-probability ones; holding them
        # one decode round costs no real latency (they would have been
        # corrected anyway) and keeps wrong words off the screen. 0 disables.
        self._tail_confidence_min = tail_confidence_min
        self._on_commit = on_commit
        self._on_error = on_error
        self._clock = clock

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode")
        self._inflight: tuple[Future, int] | None = None
        # Speculative final decode started at the gate's "maybe ended" signal;
        # (future, utterance_id). Valid only while the silence that triggered
        # it continues (the gate cancels it if speech resumes).
        self._speculative: tuple[Future, int] | None = None

        self._utterance_id = 0
        self._in_utterance = False
        self._reset_utterance_state()

    # ── Utterance lifecycle ──────────────────────────────────────────────

    def _reset_utterance_state(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        # O(1) feed: frames accumulate here and are concatenated lazily once
        # per decode instead of on every 32ms frame.
        self._chunks: list[np.ndarray] = []
        self._buffer_len = 0
        self._committed: list[Word] = []
        self._prev_norm: list[str] = []
        self._committed_text = ""  # raw committed text (prompt context)
        self._decoded_len = 0
        self._first_decode_done = False
        self._last_decode_t = self._clock()
        self._speculative = None
        # Correcting-mode state:
        self._trimmed_raw = ""  # raw text of committed words trimmed from buffer
        self._last_raw = ""  # raw of the last rendered hypothesis
        self._render_ok = True  # False after a sync failure: screen unknown

    def start_utterance(self, carry: bool = False) -> None:
        """Begin an utterance; ``carry`` marks a seamless long-speech rollover
        (the emitter keeps accumulating the previous region)."""
        self._utterance_id += 1
        self._in_utterance = True
        self._reset_utterance_state()
        self._formatter.on_utterance_start()
        if self._correcting:
            begin = getattr(self._emitter, "begin_utterance", None)
            if begin is not None:
                begin(carry)

    def feed(self, frames: list[np.ndarray]) -> None:
        if not self._in_utterance or not frames:
            return
        self._chunks.extend(frames)
        self._buffer_len += sum(len(f) for f in frames)

    def _audio(self) -> np.ndarray:
        """Materialize the audio buffer (lazy concat of pending frames)."""
        if self._chunks:
            self._buffer = np.concatenate([self._buffer, *self._chunks])
            self._chunks = []
        return self._buffer

    # ── Periodic work (call frequently from the session loop) ───────────

    def _effective_interval(self) -> float:
        if self._min_interval is None:
            return self._interval
        # Track the model's real speed: 1.5x decode time, floored and capped.
        return min(self._interval, max(self._min_interval, self._last_decode_duration * 1.5))

    def tick(self) -> None:
        self._drain_inflight(block=False)
        if not (self._in_utterance and self._streaming_enabled):
            return
        if self._inflight is not None:
            return  # decode in flight: skip (backpressure)
        if self._speculative is not None:
            return  # final decode already speculating on this audio
        if self._buffer_len < self._min_speech:
            return
        now = self._clock()
        # First-paint fast path: perceived responsiveness is set by the FIRST
        # hypothesis, so the first decode of an utterance fires the moment
        # min_speech is buffered — skipping the cadence interval and the
        # min_new_audio accumulation gate (together ~0.55 s of dead wait).
        # Note the pre-roll ring means min_speech is often already buffered
        # at utterance start. Subsequent decodes keep the normal pacing.
        if self._first_decode_done:
            if now - self._last_decode_t < self._effective_interval():
                return
            if self._buffer_len - self._decoded_len < self._min_new:
                return
        audio = self._audio().copy()
        self._decoded_len = len(audio)
        self._last_decode_t = now
        self._first_decode_done = True
        self._inflight = (
            self._submit_timed(audio, final=False),
            self._utterance_id,
        )

    def _submit_timed(self, audio: np.ndarray, *, final: bool) -> Future:
        """Submit a decode, recording its wall-clock duration for adaptivity."""
        prompt = self._prompt_tail()

        def run() -> list[Word]:
            t0 = time.monotonic()
            try:
                return self._transcriber.transcribe(audio, final=final, prompt=prompt)
            finally:
                self._last_decode_duration = time.monotonic() - t0

        return self._pool.submit(run)

    # ── Speculative finalize ─────────────────────────────────────────────

    def speculate_final(self) -> None:
        """Start the final decode early (gate says the utterance probably
        ended). If the silence holds, ``finalize`` uses this result and the
        pause-to-final-text latency collapses to ~zero; if speech resumes the
        gate cancels it and only some GPU time was wasted."""
        if not self._in_utterance or self._speculative is not None:
            return
        if self._buffer_len < self._min_speech and not self._committed:
            return
        self._drain_inflight(block=False)
        self._first_decode_done = True  # a decode is running: no fast path
        self._speculative = (
            self._submit_timed(self._audio().copy(), final=True),
            self._utterance_id,
        )

    def cancel_speculation(self) -> None:
        """Speech resumed: the speculative final no longer covers the audio."""
        self._speculative = None

    # ── Finalization ─────────────────────────────────────────────────────

    def finalize(self) -> str | None:
        """End the current utterance: final decode, commit everything.

        In correcting mode, returns the utterance's final emitted text
        (including trailer) when it was successfully synced to the screen —
        the input for a contextual transform. Returns None in commit mode and
        on every path where the screen state is unknown or nothing was typed.
        """
        if not self._in_utterance:
            return None
        self._in_utterance = False
        self._drain_inflight(block=True)

        if self._buffer_len < self._min_speech and not self._committed:
            self._speculative = None
            self._reset_utterance_state()
            return None

        # Prefer the speculative final decode: it was started at the "maybe
        # ended" silence point and covers all speech (the tail added since is
        # silence by construction — speech would have cancelled it).
        spec = self._speculative
        self._speculative = None
        words: list[Word] | None = None
        if spec is not None and spec[1] == self._utterance_id:
            try:
                words = spec[0].result(timeout=FINAL_DECODE_TIMEOUT_S)
            except Exception:
                words = None  # fall through to a fresh final decode
        try:
            if words is None:
                future = self._submit_timed(self._audio(), final=True)
                words = future.result(timeout=FINAL_DECODE_TIMEOUT_S)
        except Exception as e:
            self._error(f"final decode failed: {e}")
            if self._correcting and self._last_raw:
                # Tentative text is on screen; never delete the user's words.
                # Advance formatter state to match what's stranded there so
                # the next utterance's spacing is computed correctly.
                self._formatter.format_delta(self._last_raw)
                self._formatter.end_utterance()
            self._reset_utterance_state()
            return None

        final_text: str | None = None
        if self._correcting:
            final_text = self._finalize_correcting(words)
        else:
            tail = words[len(self._committed) :]
            if tail:
                self._commit_words(tail)
            trailer = self._formatter.end_utterance()
            if trailer and not self._emitter.emit(trailer):
                self._error("emitter failed to type utterance trailer")
        self._reset_utterance_state()
        return final_text

    def _finalize_correcting(self, words: list[Word]) -> str | None:
        """Replace the tentative tail with the final decode + trailer.

        Returns the synced final text, or None when the screen wasn't (or
        couldn't be) updated — callers must not build on unknown screen state.
        """
        raw = self._trimmed_raw + "".join(w.text for w in words)
        # The one state-mutating format of this utterance (peek never mutates).
        text = self._formatter.format_delta(raw) if raw.strip() else ""
        synced = False
        trailer = self._formatter.end_utterance()
        if self._render_ok:
            synced = self._emitter.sync(text + trailer)
            if not synced:
                self._error("emitter failed to sync final text")
        if text and self._on_commit is not None:
            self._on_commit(text + trailer)
        return (text + trailer) if (text and synced) else None

    def stop(self) -> None:
        """Session teardown; finalizes any in-flight utterance first."""
        self.finalize()
        self._pool.shutdown(wait=True, cancel_futures=True)

    # ── Internals ────────────────────────────────────────────────────────

    def _drain_inflight(self, *, block: bool) -> None:
        if self._inflight is None:
            return
        future, utt_id = self._inflight
        if not block and not future.done():
            return
        self._inflight = None
        try:
            words = future.result(timeout=FINAL_DECODE_TIMEOUT_S)
        except Exception as e:
            self._error(f"decode failed: {e}")
            return
        if utt_id != self._utterance_id or not self._in_utterance:
            return  # stale result from a previous utterance
        self._apply_hypothesis(words)

    def _apply_hypothesis(self, words: list[Word]) -> None:
        norm_new = [_normalize(w.text) for w in words]
        # Longest common prefix with the previous hypothesis (LocalAgreement-2).
        agree = 0
        for a, b in zip(self._prev_norm, norm_new):
            if a != b or not a:
                break
            agree += 1
        if agree > len(self._committed):
            fresh = words[len(self._committed) : agree]
            if self._correcting:
                # Bookkeeping only (prompt context); rendering happens below.
                self._committed_text += "".join(w.text for w in fresh)
            else:
                self._commit_words(fresh)
            self._committed = list(words[:agree])
        self._prev_norm = norm_new
        if self._correcting:
            self._render(self._gate_tail(words, agree))
        self._maybe_trim()

    def _gate_tail(self, words: list[Word], agree: int) -> list[Word]:
        """Trim trailing low-confidence words that no second decode has
        confirmed yet (index >= the agreed prefix). Only a contiguous tail
        is trimmed — words can't be hidden mid-hypothesis. The final decode
        renders everything, so nothing is ever lost."""
        if self._tail_confidence_min <= 0.0:
            return words
        stable = max(agree, len(self._committed))
        end = len(words)
        while end > stable and words[end - 1].prob < self._tail_confidence_min:
            end -= 1
        return words if end == len(words) else words[:end]

    def _render(self, words: list[Word]) -> None:
        """Correcting mode: sync the screen to the full current hypothesis."""
        if not self._render_ok:
            return
        raw = self._trimmed_raw + "".join(w.text for w in words)
        self._last_raw = raw
        desired = self._formatter.peek(raw) if raw.strip() else ""
        publish = getattr(self._emitter, "publish", None)
        if publish is not None:
            # Latest-wins async render: never blocks the session loop, and a
            # hypothesis superseded before it was typed is skipped entirely.
            # Failures freeze the emitter (reported by the render thread);
            # the finalize-time sync — a barrier — still observes them.
            publish(desired)
            return
        if not self._emitter.sync(desired):
            self._render_ok = False
            self._error("emitter failed to sync tentative text")

    def _commit_words(self, words: list[Word]) -> None:
        raw = "".join(w.text for w in words)
        self._committed_text += raw
        text = self._formatter.format_delta(raw)
        if not text:
            return
        if not self._emitter.emit(text):
            self._error("emitter failed to type committed text")
        if self._on_commit is not None:
            self._on_commit(text)

    def _maybe_trim(self) -> None:
        """Bound decode cost: cut committed audio out of the buffer."""
        if self._buffer_len <= self._max_buffer or not self._committed:
            return
        # Prefer the last committed sentence end; fall back to the last
        # committed word. Trim at the word's *end* timestamp.
        trim_word = self._committed[-1]
        for w in reversed(self._committed):
            if w.text.rstrip().endswith(_SENTENCE_END):
                trim_word = w
                break
        cut = int(trim_word.end * SAMPLE_RATE)
        if cut <= 0:
            return
        buffer = self._audio()  # materialize before slicing
        cut = min(cut, len(buffer))
        self._buffer = buffer[cut:]
        self._buffer_len = len(self._buffer)
        self._decoded_len = max(0, self._decoded_len - cut)
        offset = cut / SAMPLE_RATE

        def keep(w: Word) -> bool:
            return w.end > offset

        # Correcting mode renders trimmed words from their raw text, so the
        # full utterance stays reconstructible after the audio is cut.
        self._trimmed_raw += "".join(w.text for w in self._committed if not keep(w))
        kept_committed = [w.rebased(offset) for w in self._committed if keep(w)]
        # prev_norm must stay index-aligned with what the *next* decode of the
        # trimmed buffer will produce: committed words inside the cut vanish.
        dropped = len(self._committed) - len(kept_committed)
        self._committed = kept_committed
        self._prev_norm = self._prev_norm[dropped:]

    def _prompt_tail(self) -> str | None:
        tail = self._committed_text.strip()
        return tail[-_PROMPT_TAIL_CHARS:] if tail else None

    def _error(self, msg: str) -> None:
        if self._on_error is not None:
            self._on_error(msg)
