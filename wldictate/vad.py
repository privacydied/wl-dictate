"""Streaming voice activity detection.

``StreamingSileroVAD`` runs the Silero VAD ONNX model bundled with
faster-whisper one 512-sample (32 ms @ 16 kHz) frame at a time, holding the
LSTM state and 64-sample audio context across calls — i.e. true streaming use
rather than faster-whisper's batch helper.

``EnergyVAD`` is a dependency-free fallback (port of the old EMA-RMS logic).

``VadGate`` is the utterance state machine: onset debounce, pre-roll, offset
hold, and forced finalization for very long utterances.
"""

from __future__ import annotations

import glob
import os
import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # Silero v5/v6 frame size at 16 kHz
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE  # 0.032
_CONTEXT_SAMPLES = 64


class VadUnavailableError(RuntimeError):
    pass


class StreamingSileroVAD:
    """Frame-by-frame Silero VAD with persistent recurrent state."""

    def __init__(self) -> None:
        try:
            import onnxruntime
            from faster_whisper.utils import get_assets_path
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise VadUnavailableError(f"silero VAD unavailable: {e}") from e

        model_path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        if not os.path.exists(model_path):
            candidates = sorted(
                glob.glob(os.path.join(get_assets_path(), "silero*.onnx"))
            )
            if not candidates:
                raise VadUnavailableError("no bundled silero VAD model found")
            model_path = candidates[-1]

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample float32 frame."""
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"expected ({FRAME_SAMPLES},) frame, got {frame.shape}")
        inp = np.concatenate([self._context, frame]).reshape(1, -1).astype(np.float32)
        out, self._h, self._c = self._session.run(
            None, {"input": inp, "h": self._h, "c": self._c}
        )
        self._context = frame[-_CONTEXT_SAMPLES:].copy()
        return float(out.reshape(-1)[0])


class EnergyVAD:
    """EMA-RMS energy gate returning pseudo-probabilities (0.0 / 1.0).

    Thresholds are the old int16-scale values converted to float32 scale.
    """

    _ALPHA = 0.3
    _SPEECH_THRESHOLD = 200.0 / 32767.0
    _SILENCE_RATIO = 0.25
    _SILENCE_FLOOR = 100.0 / 32767.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._ema = 0.0
        self._peak = 0.0
        self._in_speech = False

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        self._ema = self._ALPHA * rms + (1.0 - self._ALPHA) * self._ema
        if not self._in_speech:
            # Onset from *instantaneous* RMS: a decaying EMA after speech must
            # not re-trigger (the VadGate's onset debounce handles noise spikes).
            if rms >= self._SPEECH_THRESHOLD:
                self._in_speech = True
                self._peak = self._ema
                return 1.0
            return 0.0
        self._peak = max(self._peak, self._ema)
        effective_floor = max(self._SILENCE_FLOOR, self._peak * self._SILENCE_RATIO)
        if self._ema < effective_floor and rms < self._SPEECH_THRESHOLD:
            self._in_speech = False
            self._peak = 0.0
            return 0.0
        return 1.0


def make_vad(backend: str = "auto"):
    """Build the configured VAD backend, falling back to energy on failure."""
    if backend == "energy":
        return EnergyVAD()
    try:
        return StreamingSileroVAD()
    except VadUnavailableError as e:
        if backend == "silero":
            raise
        print(f"WARNING: {e}; falling back to energy VAD", file=sys.stderr)
        return EnergyVAD()


@dataclass
class GateResult:
    """Outcome of feeding one frame to the gate."""

    speech_frames: list[np.ndarray] = field(default_factory=list)
    utterance_started: bool = False
    utterance_ended: bool = False
    forced: bool = False
    # Speculative finalize: silence has lasted ``speculative_silence_ms`` —
    # the utterance will *probably* end. The engine can start the final decode
    # now so the result is ready when (if) the gate closes.
    utterance_maybe_ended: bool = False
    # Speech resumed after ``utterance_maybe_ended``: discard the speculation.
    speculation_cancelled: bool = False
    # Seamless rollover: the utterance hit max_utterance_s while speech was
    # STILL ACTIVE. It was force-ended (utterance_ended+forced) and a new one
    # begins immediately — no onset re-detection, no dropped audio, and the
    # caller should treat the two as one logical message (carry).
    utterance_restarted: bool = False


class VadGate:
    """Utterance segmentation on top of per-frame speech probabilities.

    - IDLE: keeps a pre-roll ring of recent frames. ``onset_frames``
      consecutive frames above ``onset`` start an utterance; the pre-roll is
      prepended so word onsets are not clipped.
    - SPEECH: every frame belongs to the utterance. Sustained probability
      below ``offset`` for ``min_silence_ms`` ends it. Utterances longer than
      ``max_utterance_s`` are force-ended (streaming commits keep latency low,
      this is just a safety bound).
    """

    def __init__(
        self,
        vad,
        *,
        onset: float = 0.5,
        offset: float = 0.35,
        onset_frames: int = 2,
        min_silence_ms: int = 500,
        pre_roll_ms: int = 320,
        max_utterance_s: float = 28.0,
        speculative_silence_ms: int = 200,
    ) -> None:
        self._vad = vad
        self._onset = onset
        self._offset = offset
        self._onset_frames = max(1, onset_frames)
        self._silence_frames_needed = max(1, round(min_silence_ms / 1000 / FRAME_SECONDS))
        # Speculative finalize point (0 disables). Clamped below the real
        # silence threshold so "maybe ended" always precedes "ended".
        if speculative_silence_ms and speculative_silence_ms < min_silence_ms:
            self._spec_frames_needed = max(
                1, round(speculative_silence_ms / 1000 / FRAME_SECONDS)
            )
        else:
            self._spec_frames_needed = 0
        self._spec_fired = False
        self._pre_roll_frames = max(0, round(pre_roll_ms / 1000 / FRAME_SECONDS))
        self._max_utterance_frames = max(1, round(max_utterance_s / FRAME_SECONDS))

        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, self._pre_roll_frames))
        self.reset()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._in_speech = False
        self._onset_run = 0
        self._silence_run = 0
        self._utterance_frames = 0
        self._spec_fired = False
        self._pre_roll.clear()
        if hasattr(self._vad, "reset"):
            self._vad.reset()

    def process(self, frame: np.ndarray) -> GateResult:
        result = GateResult()
        p = self._vad.prob(frame)

        if not self._in_speech:
            if p >= self._onset:
                self._onset_run += 1
            else:
                self._onset_run = 0
            if self._onset_run >= self._onset_frames:
                # Utterance begins: pre-roll (which already contains the onset
                # run frames) + current frame.
                self._in_speech = True
                self._silence_run = 0
                result.utterance_started = True
                result.speech_frames = list(self._pre_roll) + [frame]
                self._utterance_frames = len(result.speech_frames)
                self._pre_roll.clear()
                self._onset_run = 0
            else:
                if self._pre_roll_frames:
                    self._pre_roll.append(frame)
            return result

        # In speech: frame always belongs to the utterance.
        result.speech_frames = [frame]
        self._utterance_frames += 1

        if p < self._offset:
            self._silence_run += 1
        else:
            self._silence_run = 0
            if self._spec_fired:
                # Speech resumed after the speculative point.
                self._spec_fired = False
                result.speculation_cancelled = True

        if (
            self._spec_frames_needed
            and not self._spec_fired
            and self._silence_run >= self._spec_frames_needed
        ):
            self._spec_fired = True
            result.utterance_maybe_ended = True

        if self._silence_run >= self._silence_frames_needed:
            self._end_utterance(result, forced=False)
        elif self._utterance_frames >= self._max_utterance_frames:
            # Length cap hit while the user is STILL TALKING: end this
            # utterance (bounds decode/typing state) but roll straight into a
            # new one — no onset re-detection gap, no chopped message.
            self._end_utterance(result, forced=True)
            self._in_speech = True
            self._silence_run = 0
            self._utterance_frames = 0
            result.utterance_restarted = True
        return result

    def flush(self) -> GateResult:
        """Force-end any in-flight utterance (session stop)."""
        result = GateResult()
        if self._in_speech:
            self._end_utterance(result, forced=True)
        return result

    def _end_utterance(self, result: GateResult, *, forced: bool) -> None:
        self._in_speech = False
        self._onset_run = 0
        self._silence_run = 0
        self._utterance_frames = 0
        self._spec_fired = False
        result.utterance_ended = True
        result.forced = forced
