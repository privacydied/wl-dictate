"""Realtime streaming transcription engine (LocalAgreement-2).

While an utterance is in progress, the engine re-decodes the utterance buffer
every ``infer_interval_s`` and *commits* the longest prefix of words on which
two consecutive hypotheses agree. Committed text is emitted immediately
(append-only — it is never revised), so words appear roughly a second behind
speech instead of after the whole utterance.

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
_FINAL_DECODE_TIMEOUT_S = 30.0


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
        min_new_audio_s: float = 0.3,
        max_buffer_s: float = 12.0,
        min_speech_s: float = 0.3,
        streaming_enabled: bool = True,
        on_commit: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transcriber = transcriber
        self._formatter = formatter
        self._emitter = emitter
        self._interval = infer_interval_s
        self._min_new = int(min_new_audio_s * SAMPLE_RATE)
        self._max_buffer = int(max_buffer_s * SAMPLE_RATE)
        self._min_speech = int(min_speech_s * SAMPLE_RATE)
        self._streaming_enabled = streaming_enabled
        self._on_commit = on_commit
        self._on_error = on_error
        self._clock = clock

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode")
        self._inflight: tuple[Future, int] | None = None

        self._utterance_id = 0
        self._in_utterance = False
        self._reset_utterance_state()

    # ── Utterance lifecycle ──────────────────────────────────────────────

    def _reset_utterance_state(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._committed: list[Word] = []
        self._prev_norm: list[str] = []
        self._committed_text = ""  # raw committed text (prompt context)
        self._decoded_len = 0
        self._last_decode_t = self._clock()

    def start_utterance(self) -> None:
        self._utterance_id += 1
        self._in_utterance = True
        self._reset_utterance_state()
        self._formatter.on_utterance_start()

    def feed(self, frames: list[np.ndarray]) -> None:
        if not self._in_utterance or not frames:
            return
        self._buffer = np.concatenate([self._buffer, *frames])

    # ── Periodic work (call frequently from the session loop) ───────────

    def tick(self) -> None:
        self._drain_inflight(block=False)
        if not (self._in_utterance and self._streaming_enabled):
            return
        if self._inflight is not None:
            return  # decode in flight: skip (backpressure)
        now = self._clock()
        if now - self._last_decode_t < self._interval:
            return
        if len(self._buffer) - self._decoded_len < self._min_new:
            return
        if len(self._buffer) < self._min_speech:
            return
        audio = self._buffer.copy()
        self._decoded_len = len(audio)
        self._last_decode_t = now
        self._inflight = (
            self._pool.submit(
                self._transcriber.transcribe,
                audio,
                final=False,
                prompt=self._prompt_tail(),
            ),
            self._utterance_id,
        )

    # ── Finalization ─────────────────────────────────────────────────────

    def finalize(self) -> None:
        """End the current utterance: final decode, commit everything."""
        if not self._in_utterance:
            return
        self._in_utterance = False
        self._drain_inflight(block=True)

        if len(self._buffer) < self._min_speech and not self._committed:
            self._reset_utterance_state()
            return

        try:
            future = self._pool.submit(
                self._transcriber.transcribe,
                self._buffer,
                final=True,
                prompt=self._prompt_tail(),
            )
            words = future.result(timeout=_FINAL_DECODE_TIMEOUT_S)
        except Exception as e:
            self._error(f"final decode failed: {e}")
            self._reset_utterance_state()
            return

        tail = words[len(self._committed) :]
        if tail:
            self._commit_words(tail)
        self._reset_utterance_state()

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
            words = future.result(timeout=_FINAL_DECODE_TIMEOUT_S)
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
            self._commit_words(fresh)
            self._committed = list(words[:agree])
        self._prev_norm = norm_new
        self._maybe_trim()

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
        if len(self._buffer) <= self._max_buffer or not self._committed:
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
        cut = min(cut, len(self._buffer))
        self._buffer = self._buffer[cut:]
        self._decoded_len = max(0, self._decoded_len - cut)
        offset = cut / SAMPLE_RATE

        def keep(w: Word) -> bool:
            return w.end > offset

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
