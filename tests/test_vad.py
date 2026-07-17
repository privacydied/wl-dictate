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


def _speech_frame():
    return np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)


def _silence_frame():
    return np.zeros(FRAME_SAMPLES, dtype=np.float32)


class _ScriptVAD:
    """prob() pops scripted probabilities."""

    def __init__(self, probs):
        self.probs = list(probs)

    def prob(self, frame):
        return self.probs.pop(0) if self.probs else 0.0

    def reset(self):
        pass


def test_speculative_maybe_ended_then_ended():
    # spec at ~2 frames (64ms), end at ~4 frames (128ms)
    gate = VadGate(
        _ScriptVAD([1, 1, 1, 0, 0, 0, 0]),
        onset_frames=2,
        min_silence_ms=128,
        speculative_silence_ms=64,
        pre_roll_ms=0,
    )
    events = []
    for _ in range(7):
        r = gate.process(_silence_frame())  # probs come from the script
        events.append(
            (r.utterance_started, r.utterance_maybe_ended, r.utterance_ended)
        )
    maybe_idx = [i for i, e in enumerate(events) if e[1]]
    ended_idx = [i for i, e in enumerate(events) if e[2]]
    assert len(maybe_idx) == 1 and len(ended_idx) == 1
    assert maybe_idx[0] < ended_idx[0]  # speculation strictly precedes the end


def test_speculation_cancelled_when_speech_resumes():
    gate = VadGate(
        _ScriptVAD([1, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
        onset_frames=2,
        min_silence_ms=128,
        speculative_silence_ms=64,
        pre_roll_ms=0,
    )
    maybes = cancels = ends = 0
    for _ in range(10):
        r = gate.process(_silence_frame())
        maybes += r.utterance_maybe_ended
        cancels += r.speculation_cancelled
        ends += r.utterance_ended
    assert maybes == 2  # once per silence stretch
    assert cancels == 1  # speech resumed after the first speculation
    assert ends == 1


def test_speculation_disabled_when_zero_or_too_large():
    for spec_ms in (0, 128, 500):
        gate = VadGate(
            _ScriptVAD([1, 1, 0, 0, 0, 0]),
            onset_frames=2,
            min_silence_ms=128,
            speculative_silence_ms=spec_ms,
            pre_roll_ms=0,
        )
        maybes = 0
        for _ in range(6):
            maybes += gate.process(_silence_frame()).utterance_maybe_ended
        assert maybes == 0, f"spec_ms={spec_ms} should disable speculation"


def test_long_speech_rolls_over_seamlessly():
    # Cap at ~4 frames of continuous speech: forced end + immediate restart.
    gate = VadGate(
        _ScriptVAD([1] * 12),
        onset_frames=2,
        min_silence_ms=128,
        speculative_silence_ms=0,
        pre_roll_ms=0,
        max_utterance_s=4 * FRAME_SAMPLES / 16000,
    )
    started = ended = restarted = 0
    for _ in range(12):
        r = gate.process(_speech_frame())
        started += r.utterance_started
        ended += r.utterance_ended
        restarted += r.utterance_restarted
        if r.utterance_ended:
            assert r.forced and r.utterance_restarted
        assert not (r.utterance_started and r.utterance_ended)
    assert started == 1  # only the initial onset
    assert ended == restarted >= 2  # every cap hit rolled over, no gap
    assert gate.in_speech  # still listening at the end
