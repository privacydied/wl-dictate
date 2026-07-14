import numpy as np
import pytest

from wldictate.vad import EnergyVAD, FRAME_SAMPLES, VadGate

SILENCE = np.zeros(FRAME_SAMPLES, dtype=np.float32)
LOUD = np.full(FRAME_SAMPLES, 0.1, dtype=np.float32)


class ScriptedVAD:
    """Deterministic probability source for gate tests."""

    def __init__(self, probs):
        self._probs = list(probs)

    def prob(self, frame):
        return self._probs.pop(0) if self._probs else 0.0

    def reset(self):
        pass


def make_gate(probs, **kw):
    defaults = dict(
        onset=0.5, offset=0.35, onset_frames=2, min_silence_ms=128, pre_roll_ms=96
    )
    defaults.update(kw)
    return VadGate(ScriptedVAD(probs), **defaults)


def run_gate(gate, n):
    results = [gate.process(SILENCE) for _ in range(n)]
    return results


def test_onset_needs_consecutive_frames():
    # One high frame then low: no utterance.
    gate = make_gate([0.9, 0.1, 0.9, 0.1, 0.0, 0.0])
    results = run_gate(gate, 6)
    assert not any(r.utterance_started for r in results)


def test_onset_includes_preroll():
    gate = make_gate([0.0, 0.0, 0.0, 0.9, 0.9], pre_roll_ms=96)  # 3 frames pre-roll
    results = run_gate(gate, 5)
    started = [r for r in results if r.utterance_started]
    assert len(started) == 1
    # pre-roll (3 frames, includes first onset frame) + current frame
    assert len(started[0].speech_frames) == 4


def test_offset_ends_after_sustained_silence():
    # onset (2 frames), speech, then sustained low probability
    probs = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    gate = make_gate(probs, min_silence_ms=128)  # 4 frames of silence needed
    results = run_gate(gate, 8)
    assert sum(r.utterance_started for r in results) == 1
    ended = [r for r in results if r.utterance_ended]
    assert len(ended) == 1
    assert not ended[0].forced


def test_brief_dip_does_not_end_utterance():
    probs = [0.9, 0.9] + [0.1, 0.9] * 6  # dips never sustained
    gate = make_gate(probs, min_silence_ms=128)
    results = run_gate(gate, len(probs))
    assert sum(r.utterance_started for r in results) == 1
    assert not any(r.utterance_ended for r in results)


def test_forced_finalize_on_max_utterance():
    n = 40
    probs = [0.9] * n
    frame_s = FRAME_SAMPLES / 16000
    gate = make_gate(probs, max_utterance_s=frame_s * 20)
    results = run_gate(gate, n)
    ended = [r for r in results if r.utterance_ended]
    assert len(ended) >= 1
    assert ended[0].forced


def test_flush_ends_in_flight_utterance():
    gate = make_gate([0.9, 0.9, 0.9])
    run_gate(gate, 3)
    assert gate.in_speech
    result = gate.flush()
    assert result.utterance_ended and result.forced
    assert not gate.in_speech
    # flushing when idle is a no-op
    assert not gate.flush().utterance_ended


def test_energy_vad_no_retrigger_on_decay():
    vad = EnergyVAD()
    probs = []
    for frame in [LOUD] * 10 + [SILENCE] * 20:
        probs.append(vad.prob(frame))
    tail = probs[10:]
    # Once it reports silence it must stay silent (no 0 -> 1 flapping).
    first_zero = tail.index(0.0)
    assert all(p == 0.0 for p in tail[first_zero:])


def test_energy_vad_detects_speech():
    vad = EnergyVAD()
    assert vad.prob(SILENCE) == 0.0
    assert vad.prob(LOUD) == 1.0
