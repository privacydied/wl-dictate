"""Voice dictation tool – faster-whisper + VAD + wtype typing."""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import warnings
from queue import Empty, Queue

import numpy as np
import sounddevice as sd

from faster_whisper import WhisperModel
from threading import Event, Thread

warnings.filterwarnings("ignore", message=r"resource_tracker:.*semaphore")

# ── Config ──────────────────────────────────────────────────────────────────

FASTER_WHISPER_MODEL = "tiny.en"
FASTER_WHISPER_DEVICE = "cuda"
FASTER_WHISPER_COMPUTE_TYPE = "float16"

# Transcription quality / performance
TRANSCRIBE_BEAM_SIZE = 2            # tiny.en gains nothing from beam>2
TRANSCRIBE_VAD_FILTER = True        # built-in Silero VAD trims silence from audio

BLOCK_DURATION = 0.5                # seconds per VAD evaluation block
SILENCE_BLOCKS = 2                  # consecutive quiet blocks to flush
MAX_SPEECH_BLOCKS = 30              # forced flush after ~15s
SAMPLE_RATE = 16000

VAD_EMA_ALPHA = 0.3                 # EMA smoothing factor
VAD_SPEECH_THRESHOLD = 200          # above this = speech begins
VAD_SILENCE_THRESHOLD = 100         # floor for relative silence detection
VAD_MIN_SPEECH_S = 0.3              # minimum speech duration to transcribe
VAD_SILENCE_RATIO = 0.25            # silence when EMA drops below 25% of peak
SILENCE_DEBLOCK_BLOCKS = 2          # hysteresis after flushing

WTYPE_TIMEOUT = 10
DEBUG_MODE = True

# Pre-computed constants (avoid re-evaluation every block)
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_DURATION)
EMA_INVERSE = 1.0 - VAD_EMA_ALPHA
WHISPER_TIMEOUT = 30

# Pre-compiled regexes (hot path in transcribe_and_type)
_RE_PARENS = re.compile(r"\([^)]*\)\s*")
_RE_BRACKETS = re.compile(r"\[[^\]]*\]\s*")
_RE_DOTS = re.compile(r"(?:\s*\.\s*){3,}")
_RE_WHITESPACE = re.compile(r"\s{2,}")

# Process-wide state
WHISPER_MODEL: WhisperModel | None = None
_WTYPE_ENV: dict[str, str] = {}
_WTYPE_CMD: list[str] = []

# ── Bootstrap ───────────────────────────────────────────────────────────────

def _probe_cuda_available() -> bool:
    """Check if CUDA devices are detected by CTranslate2."""
    try:
        from ctranslate2 import get_cuda_device_count  # pyright: ignore[reportMissingImports]
        return get_cuda_device_count() > 0
    except Exception:
        return False


def bootstrap() -> None:
    """Load the faster-whisper model with CUDA fallback to CPU."""
    global WHISPER_MODEL, _WTYPE_ENV, _WTYPE_CMD

    device = FASTER_WHISPER_DEVICE
    compute_type = FASTER_WHISPER_COMPUTE_TYPE

    if device == "cuda" and not _probe_cuda_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
        compute_type = "int8"

    print(f"Loading faster-whisper '{FASTER_WHISPER_MODEL}' ({device}/{compute_type})...")
    try:
        WHISPER_MODEL = WhisperModel(
            FASTER_WHISPER_MODEL, device=device, compute_type=compute_type,
        )
        print("Model loaded successfully")
    except Exception as e:
        if device == "cpu":
            raise
        print(f"WARNING: Failed to load with {device}, falling back to CPU: {e}")
        WHISPER_MODEL = WhisperModel(
            FASTER_WHISPER_MODEL, device="cpu", compute_type="int8",
        )
        print("Model loaded successfully (CPU fallback)")

    # Cache wtype subprocess environment once at startup
    _WTYPE_ENV = os.environ.copy()
    _WTYPE_ENV.setdefault("WAYLAND_DISPLAY", os.environ.get("WAYLAND_DISPLAY", ""))
    _WTYPE_ENV.setdefault("XDG_RUNTIME_DIR", os.environ.get("XDG_RUNTIME_DIR", ""))
    _WTYPE_CMD = ["wtype", " "]


# ── Utilities ───────────────────────────────────────────────────────────────

def resolve_device(device_arg: str | None) -> int | None:
    """Resolve device argument to an integer device index."""
    if device_arg is None:
        return None
    try:
        return int(device_arg)
    except ValueError:
        pass
    try:
        devices = sd.query_devices()
    except OSError as e:
        raise ValueError(f"Cannot query audio devices: {e}") from e
    for idx, dev in enumerate(devices):
        if device_arg == dev["name"]:
            return idx
    for idx, dev in enumerate(devices):
        if device_arg in dev["name"]:
            return idx
    raise ValueError(f"No matching device for: {device_arg}")


def _guess_wayland_display() -> bool:
    """Try to infer WAYLAND_DISPLAY from runtime dirs."""
    candidates = ["/run/user/1000"]
    try:
        candidates.append(f"/run/user/{os.getuid()}")
    except Exception:
        pass
    for candidate in candidates:
        try:
            sockets = glob.glob(os.path.join(candidate, "wayland-*"))
        except OSError:
            continue
        if sockets:
            display = os.path.basename(sockets[0])
            # Atomic assignment: build all values first, then set at once
            os.environ["XDG_RUNTIME_DIR"] = candidate
            os.environ["WAYLAND_DISPLAY"] = display
            _WTYPE_ENV["XDG_RUNTIME_DIR"] = candidate
            _WTYPE_ENV["WAYLAND_DISPLAY"] = display
            print(f"Guessed WAYLAND_DISPLAY={os.environ['WAYLAND_DISPLAY']}")
            return True
    return False


# ── Transcription ───────────────────────────────────────────────────────────

def transcribe_and_type(audio: np.ndarray) -> None:
    """Transcribe audio and type the result via wtype."""
    if WHISPER_MODEL is None:
        print("WARNING: Whisper model not loaded, skipping transcription")
        return
    if "WAYLAND_DISPLAY" not in os.environ and "XDG_RUNTIME_DIR" not in os.environ:
        _guess_wayland_display()

    audio_f32 = audio.astype(np.float32) / 32767.0
    try:
        segments, _ = WHISPER_MODEL.transcribe(
            audio_f32,
            beam_size=TRANSCRIBE_BEAM_SIZE,
            vad_filter=TRANSCRIBE_VAD_FILTER,
            language="en",
            condition_on_previous_text=False,   # avoid hallucination loops
            initial_prompt="Transcribe verbatim.",  # prime the decoder
        )
        text = " ".join(seg.text for seg in segments).strip()
    except RuntimeError as e:
        if "libcublas" in str(e) or "CUDA" in str(e):
            print(f"WARNING: CUDA error during transcription: {e}")
            return
        raise
    except Exception:
        print("Transcription error:")
        sys.excepthook(*sys.exc_info())
        return
    if not text:
        print("Nothing recognized.")
        return

    # Clean annotations and artifacts
    text = _RE_PARENS.sub("", text)
    text = _RE_BRACKETS.sub("", text)
    text = _RE_DOTS.sub("...", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    if not text:
        print("Nothing recognized after cleanup.")
        return
    if DEBUG_MODE:
        print(f"Cleaned transcription: {repr(text)}")

    # Type via wtype
    try:
        cmd = _WTYPE_CMD + [text]
        subprocess.run(cmd, check=True, timeout=WTYPE_TIMEOUT, env=_WTYPE_ENV)
        print(f"Typed: {text}")
    except subprocess.TimeoutExpired:
        print("wtype timed out")
    except subprocess.CalledProcessError as e:
        print(f"wtype failed: {e.stderr.strip() if e.stderr else e}")
    except FileNotFoundError:
        print("wtype is not installed")


# ── Main loop ───────────────────────────────────────────────────────────────

_TRANSCRIBE_QUEUE: Queue[tuple[list[np.ndarray] | None, str]] = Queue()
_TRANSCRIBE_DONE = Event()
_sample_count = 0  # running counter, avoids sum(len(c)) per block


def _transcribe_worker_loop() -> None:
    while True:
        try:
            audio_chunks, label = _TRANSCRIBE_QUEUE.get(timeout=0.1)
        except Empty:
            continue
        if audio_chunks is None:
            _TRANSCRIBE_DONE.set()
            return
        # Fast path for single chunk (common case — avoids np.concatenate overhead)
        if len(audio_chunks) == 1:
            full_audio = audio_chunks[0]
        else:
            full_audio = np.concatenate(audio_chunks)
        if DEBUG_MODE:
            print(f"  [{label}] transcribing {len(full_audio) / SAMPLE_RATE:.1f}s")
        transcribe_and_type(full_audio)
        _TRANSCRIBE_QUEUE.task_done()


def _enqueue_transcribe(chunks: list[np.ndarray], label: str = "") -> None:
    _TRANSCRIBE_QUEUE.put((chunks, label))


def main(device_arg: str | None = None) -> None:
    global _sample_count

    # Start transcription worker
    worker = Thread(target=_transcribe_worker_loop, daemon=True)
    worker.start()

    audio_queue: Queue[np.ndarray] = Queue()

    def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status and DEBUG_MODE:
            print(f"  [stream] {status}")
        audio_queue.put(indata.copy())

    device_idx = resolve_device(device_arg) if device_arg else None

    with sd.InputStream(
        samplerate=None, channels=1, dtype="float32",
        device=device_idx, callback=_audio_callback,
        blocksize=BLOCK_SAMPLES,
    ):
        if device_idx is not None:
            input_sr = int(sd.query_devices(device_idx, "input")["default_samplerate"])
            if input_sr <= 0:
                print(f"WARNING: Device {device_idx} reports invalid samplerate {input_sr}, using default {SAMPLE_RATE}")
                input_sr = SAMPLE_RATE
        else:
            input_sr = SAMPLE_RATE

        silent_blocks = 0
        speech_chunks: list[np.ndarray] = []
        ema_rms = 0.0
        in_speech = False
        silence_debounce = 0
        energy_floor = float("inf")
        speech_peak_rms = 0.0

        print("Listening... (Ctrl+C to stop)")

        try:
            while True:
                raw = audio_queue.get()
                block = raw.flatten()

                # Resample if device samplerate differs from target
                if input_sr != SAMPLE_RATE and len(block) > 0:
                    target_len = int(len(block) * SAMPLE_RATE / input_sr)
                    if target_len > 0:
                        block = np.interp(
                            np.linspace(0, len(block), target_len, endpoint=False),
                            np.arange(len(block)), block,
                        )

                # Scale to int16 range for VAD thresholds
                block = np.clip(block, -1, 1) * 32767
                block = block.astype(np.int16)

                # RMS energy
                rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
                _sample_count += len(block)

                # EMA smoothing & floor tracking
                ema_rms = VAD_EMA_ALPHA * rms + EMA_INVERSE * ema_rms
                if not in_speech and ema_rms < energy_floor:
                    energy_floor = ema_rms

                if in_speech:
                    if ema_rms > speech_peak_rms:
                        speech_peak_rms = ema_rms
                    effective_silence = max(
                        VAD_SILENCE_THRESHOLD, speech_peak_rms * VAD_SILENCE_RATIO
                    )
                    if ema_rms < effective_silence:
                        silent_blocks += 1
                        if DEBUG_MODE:
                            print(
                                f"  silence {silent_blocks}/{SILENCE_BLOCKS}"
                                f" (EMA {ema_rms:.0f}, peak {speech_peak_rms:.0f})"
                            )
                        if silent_blocks >= SILENCE_BLOCKS:
                            if _sample_count >= VAD_MIN_SPEECH_S * SAMPLE_RATE:
                                _enqueue_transcribe(list(speech_chunks), "silence")
                            speech_chunks.clear()
                            _sample_count = 0
                            silent_blocks = 0
                            speech_peak_rms = 0.0
                            silence_debounce = SILENCE_DEBLOCK_BLOCKS
                            in_speech = False
                    else:
                        silent_blocks = 0
                        speech_chunks.append(block)
                        if DEBUG_MODE:
                            print(
                                f"  speech (EMA {ema_rms:.0f}, peak {speech_peak_rms:.0f})"
                                f" +{len(block) / SAMPLE_RATE:.1f}s"
                            )
                else:
                    if ema_rms >= VAD_SPEECH_THRESHOLD:
                        if silence_debounce > 0:
                            silence_debounce -= 1
                        else:
                            in_speech = True
                            speech_chunks.append(block)
                            silent_blocks = 0
                            if DEBUG_MODE:
                                print(f"  onset at EMA {ema_rms:.0f}")

                # Runaway guard: force flush after MAX_SPEECH_BLOCKS
                if _sample_count >= BLOCK_SAMPLES * MAX_SPEECH_BLOCKS:
                    secs_before_reset = _sample_count / SAMPLE_RATE
                    _enqueue_transcribe(list(speech_chunks), "forced")
                    speech_chunks.clear()
                    _sample_count = 0
                    silent_blocks = 0
                    speech_peak_rms = 0.0
                    energy_floor = float("inf")
                    in_speech = False
                    if DEBUG_MODE:
                        print(f"  forced flush ({secs_before_reset:.1f}s)")

        except KeyboardInterrupt:
            print("\nStopping...")
            if speech_chunks:
                secs = _sample_count / SAMPLE_RATE
                print(f"Final flush: {secs:.1f}s")
                _enqueue_transcribe(list(speech_chunks), "final")

    _TRANSCRIBE_QUEUE.put((None, ""))
    _TRANSCRIBE_DONE.wait(timeout=60)


if __name__ == "__main__":
    bootstrap()
    if len(sys.argv) == 1:
        print("Available input devices:")
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    print(f"  [{i}] {dev['name']} -- {dev['default_samplerate']} Hz")
        except OSError as e:
            print(f"Cannot query devices: {e}")
    main(device_arg=sys.argv[1] if len(sys.argv) > 1 else None)
