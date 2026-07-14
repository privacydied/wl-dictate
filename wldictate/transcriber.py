"""Transcriber backends.

``Transcriber`` is the minimal interface the streaming engine needs; the
default backend is faster-whisper (CTranslate2). The abstraction keeps the
engine swappable (e.g. Parakeet/Moonshine later) and makes the streaming
logic unit-testable via ``FakeTranscriber``.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000

# Hallucination guard: segments that are almost certainly silence artifacts.
_NO_SPEECH_PROB_MAX = 0.85
_AVG_LOGPROB_MIN = -1.2


@dataclass(frozen=True)
class Word:
    """One decoded word. ``text`` keeps whisper's leading-space convention."""

    text: str
    start: float
    end: float
    prob: float = 1.0

    def rebased(self, offset_s: float) -> "Word":
        return Word(self.text, self.start - offset_s, self.end - offset_s, self.prob)


class Transcriber(ABC):
    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, *, final: bool = False, prompt: str | None = None
    ) -> list[Word]:
        """Decode 16 kHz float32 mono audio into timestamped words."""


class FasterWhisperTranscriber(Transcriber):
    def __init__(
        self,
        model_name: str = "distil-small.en",
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device_pref = device
        self._compute_pref = compute_type
        self._model = None
        self.device = "unloaded"
        self.compute_type = "unloaded"

    # ── Lifecycle ────────────────────────────────────────────────────────

    @staticmethod
    def _cuda_available() -> bool:
        try:
            from ctranslate2 import get_cuda_device_count

            return get_cuda_device_count() > 0
        except Exception:
            return False

    def _resolve_target(self) -> tuple[str, str]:
        device = self._device_pref
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        elif device == "cuda" and not self._cuda_available():
            print("WARNING: CUDA not available, using CPU", file=sys.stderr)
            device = "cpu"
        compute = self._compute_pref
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return device, compute

    def load(self) -> None:
        from faster_whisper import WhisperModel

        device, compute = self._resolve_target()
        try:
            self._model = WhisperModel(
                self._model_name, device=device, compute_type=compute
            )
        except Exception as e:
            if device == "cpu":
                raise
            print(
                f"WARNING: failed to load on {device}/{compute}, retrying on CPU: {e}",
                file=sys.stderr,
            )
            device, compute = "cpu", "int8"
            self._model = WhisperModel(
                self._model_name, device=device, compute_type=compute
            )
        self.device = device
        self.compute_type = compute

    def warmup(self) -> float:
        """One throwaway decode so the first real utterance is not slow.

        Returns the warmup decode duration in seconds.
        """
        if self._model is None:
            raise RuntimeError("model not loaded")
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32)
        t0 = time.monotonic()
        self.transcribe(audio, final=True)
        return time.monotonic() - t0

    # ── Decoding ─────────────────────────────────────────────────────────

    def transcribe(
        self, audio: np.ndarray, *, final: bool = False, prompt: str | None = None
    ) -> list[Word]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=2 if final else 1,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            vad_filter=False,  # the pipeline gates speech with its own VAD
        )
        words: list[Word] = []
        for seg in segments:
            if (
                seg.no_speech_prob > _NO_SPEECH_PROB_MAX
                and seg.avg_logprob < _AVG_LOGPROB_MIN
            ):
                continue  # near-certain silence hallucination
            for w in seg.words or []:
                words.append(Word(w.word, float(w.start), float(w.end), float(w.probability)))
        return words


class FakeTranscriber(Transcriber):
    """Scripted transcriber for tests and benchmarks.

    Each call pops the next scripted hypothesis (a list of Words). When the
    script is exhausted, the last hypothesis is repeated.
    """

    def __init__(self, script: list[list[Word]]) -> None:
        self._script = list(script)
        self._last: list[Word] = []
        self.calls: list[dict] = []

    def transcribe(
        self, audio: np.ndarray, *, final: bool = False, prompt: str | None = None
    ) -> list[Word]:
        self.calls.append(
            {"samples": len(audio), "final": final, "prompt": prompt}
        )
        if self._script:
            self._last = self._script.pop(0)
        return list(self._last)
