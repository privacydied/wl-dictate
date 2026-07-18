"""WER + latency eval for the transcription side of the pipeline.

The transform side has utils/eval_transform.py; this is its STT sibling —
without it, VAD/beam/model knobs are tuned blind.

Data layout: a directory of paired files ``NAME.wav`` + ``NAME.txt``
(the reference transcript). Mono 16 kHz WAV is ideal; other rates and
stereo are converted. Record clips with e.g.:

    pw-record --rate 16000 --channels 1 clip.wav

Usage:
    .venv/bin/python utils/eval_stt.py <data_dir> [--model NAME] [--streaming]

Reports, per clip and aggregate:
- WER of the FINAL decode (what lands on screen at finalize)
- decode latency and real-time factor
- with --streaming: WER of the streaming engine's committed output
  (LocalAgreement path, exactly as the worker runs it)
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wldictate.emitter import NullEmitter  # noqa: E402
from wldictate.streaming import SAMPLE_RATE, StreamingSession, _normalize  # noqa: E402
from wldictate.textproc import TextFormatter  # noqa: E402
from wldictate.transcriber import FasterWhisperTranscriber  # noqa: E402


def load_wav(path: Path) -> np.ndarray:
    """16 kHz mono float32 from any PCM WAV (stereo folded, rate resampled)."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n_out = int(len(data) * SAMPLE_RATE / rate)
        data = np.interp(
            np.linspace(0, len(data) - 1, n_out), np.arange(len(data)), data
        ).astype(np.float32)
    return data


def norm_words(text: str) -> list[str]:
    return [n for n in (_normalize(w) for w in text.split()) if n]


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance (substitution/insert/delete = 1)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + (r != h),  # substitution / match
            )
        prev = curr
    return prev[-1]


def run_streaming(transcriber, audio: np.ndarray) -> str:
    """Feed the clip through the real streaming engine (commit mode) and
    return everything it commits — the LocalAgreement path end to end."""
    out: list[str] = []
    session = StreamingSession(
        transcriber,
        TextFormatter(),
        NullEmitter(),
        min_infer_interval_s=0.2,
        on_commit=out.append,
    )
    session.start_utterance()
    chunk_len = int(0.32 * SAMPLE_RATE)
    for start in range(0, len(audio), chunk_len):
        session.feed([audio[start : start + chunk_len]])
        session.tick()
        time.sleep(0.01)  # let in-flight decodes land like the live loop does
    session.finalize()
    session.stop()
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--model", default=None, help="whisper model (default: config)")
    ap.add_argument(
        "--streaming",
        action="store_true",
        help="also run the full streaming engine per clip",
    )
    args = ap.parse_args()

    pairs = sorted(
        (wav, wav.with_suffix(".txt"))
        for wav in args.data_dir.glob("*.wav")
        if wav.with_suffix(".txt").exists()
    )
    if not pairs:
        print(f"no NAME.wav + NAME.txt pairs in {args.data_dir}", file=sys.stderr)
        return 1

    model = args.model
    if model is None:
        from wldictate.config import Config
        from wldictate.hardware import resolve_whisper_model

        model, _ = resolve_whisper_model(Config.load().model)
    print(f"model: {model}  clips: {len(pairs)}")

    tr = FasterWhisperTranscriber(model_name=model)
    tr.load()
    tr.warmup()

    tot_edits = tot_ref = 0
    stream_edits = stream_ref = 0
    tot_audio_s = tot_decode_s = 0.0
    for wav, txt in pairs:
        audio = load_wav(wav)
        ref = norm_words(txt.read_text())
        t0 = time.perf_counter()
        words = tr.transcribe(audio, final=True)
        dt = time.perf_counter() - t0
        hyp = norm_words("".join(w.text for w in words))
        edits = edit_distance(ref, hyp)
        tot_edits += edits
        tot_ref += len(ref)
        tot_audio_s += len(audio) / SAMPLE_RATE
        tot_decode_s += dt
        line = (
            f"  {wav.stem:24} WER {edits / max(1, len(ref)):6.1%}"
            f"  decode {dt:5.2f}s  rtf {dt / (len(audio) / SAMPLE_RATE):.2f}"
        )
        if args.streaming:
            s_hyp = norm_words(run_streaming(tr, audio))
            s_edits = edit_distance(ref, s_hyp)
            stream_edits += s_edits
            stream_ref += len(ref)
            line += f"  stream-WER {s_edits / max(1, len(ref)):6.1%}"
        print(line)

    print(
        f"\nfinal-decode WER: {tot_edits / max(1, tot_ref):.1%}"
        f"  ({tot_edits} edits / {tot_ref} ref words)"
    )
    print(f"mean rtf: {tot_decode_s / max(1e-9, tot_audio_s):.2f}")
    if args.streaming:
        print(f"streaming WER:    {stream_edits / max(1, stream_ref):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
