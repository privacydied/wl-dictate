import glob
import os
import re
import subprocess
import sys
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from faster_whisper import WhisperModel

# Config
FASTER_WHISPER_MODEL = "tiny.en"
FASTER_WHISPER_DEVICE = "cpu"
FASTER_WHISPER_COMPUTE_TYPE = "int8"
BLOCK_DURATION = 0.5
SILENCE_BLOCKS = 2
MAX_SPEECH_BLOCKS = 30
TRANSCRIBE_CHUNK_SIZE = 5
SAMPLE_RATE = 16000
CHANNELS = 1
VAD_RMS_THRESHOLD = 500         # kept for reference
VAD_EMA_ALPHA = 0.3             # EMA smoothing factor
VAD_SPEECH_THRESHOLD = 600      # above this = speech begins
VAD_SILENCE_THRESHOLD = 350     # below this = speech ends
VAD_MIN_SPEECH_S = 0.3          # minimum speech duration to transcribe
SILENCE_DEBLOCK_BLOCKS = 2      # extra silence after transcription before re-arming
WHISPER_TIMEOUT = 30
WTYPE_TIMEOUT = 10
DEBUG_MODE = True

WHISPER_MODEL = None


def bootstrap():
    """Initialize the faster-whisper model."""
    global WHISPER_MODEL
    print(f"Loading faster-whisper '{FASTER_WHISPER_MODEL}' ({FASTER_WHISPER_DEVICE}/{FASTER_WHISPER_COMPUTE_TYPE})...")
    WHISPER_MODEL = WhisperModel(
        FASTER_WHISPER_MODEL, device=FASTER_WHISPER_DEVICE, compute_type=FASTER_WHISPER_COMPUTE_TYPE,
    )
    print("Model loaded successfully")


def resolve_device(device_arg):
    """Resolve device argument to device index."""
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


def _guess_wayland_display():
    """Try to infer WAYLAND_DISPLAY from runtime dirs."""
    candidates = ["/run/user/1000"]
    try:
        import pwd
        candidates.append(f"/run/user/{os.getuid()}")
    except Exception:
        pass
    for candidate in candidates:
        try:
            sockets = glob.glob(os.path.join(candidate, "wayland-*"))
        except OSError:
            continue
        if sockets:
            os.environ["XDG_RUNTIME_DIR"] = candidate
            os.environ["WAYLAND_DISPLAY"] = os.path.basename(sockets[0])
            print(f"Guessed WAYLAND_DISPLAY={os.environ['WAYLAND_DISPLAY']}")
            return True
    return False


def record_chunk(duration, device_id=None):
    """Record audio chunk from specified device, resample to 16kHz mono."""
    device_idx = None
    if device_id is not None:
        try:
            device_idx = resolve_device(device_id)
        except ValueError as e:
            print(f"Error: {e}")
            return None
    try:
        devices = sd.query_devices()
    except OSError as e:
        print(f"Cannot query audio devices: {e}")
        return None
    input_devices = [i for i, dev in enumerate(devices) if dev["max_input_channels"] > 0]
    if device_idx is not None and device_idx not in input_devices:
        print(f"Invalid input device index: {device_idx}. Available: {input_devices}")
        return None
    try:
        if device_idx is None:
            device_info = sd.query_devices(sd.default.device[0], "input")
        else:
            device_info = sd.query_devices(device_idx, "input")
    except OSError as e:
        print(f"Cannot query device info: {e}")
        return None
    input_sr = int(device_info["default_samplerate"])
    device_name = device_info.get("name", "unknown")
    print(f"Recording from '{device_name}' at {input_sr} Hz, 1 channel(s)")
    audio = sd.rec(int(duration * input_sr), samplerate=input_sr,
                   channels=1, dtype='float32', device=device_idx)
    sd.wait()
    if audio.ndim > 1:
        audio = audio.squeeze()
    if audio is None or len(audio) == 0 or np.isnan(audio).any():
        print("Warning: Invalid or empty audio buffer")
        return None
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if input_sr != SAMPLE_RATE:
        dur = len(audio) / input_sr
        target = int(SAMPLE_RATE * dur)
        audio = np.interp(np.linspace(0, len(audio), target, endpoint=False),
                          np.arange(len(audio)), audio)
    audio *= 32767
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    if DEBUG_MODE:
        print(f"Audio RMS: {rms:.2f}")
    return audio


def transcribe_and_type(audio):
    """Transcribe audio via faster-whisper and type the result with wtype."""
    import traceback as _traceback
    if audio is None or WHISPER_MODEL is None:
        return
    if "WAYLAND_DISPLAY" not in os.environ and "XDG_RUNTIME_DIR" not in os.environ:
        _guess_wayland_display()
    # Convert int16 audio to float32 normalized [-1.0, 1.0]
    audio_normalized = audio.astype(np.float32) / 32767.0
    try:
        segments, info = WHISPER_MODEL.transcribe(audio_normalized, beam_size=5, language="en")
        text = " ".join(seg.text for seg in segments).strip()
    except Exception:
        print("Transcription error:")
        _traceback.print_exc()
        return
    if not text:
        print("Nothing recognized.")
        return
    # Strip non-speech annotations and collapse whitespace
    text = re.sub(r"\([^)]*\)\s*", "", text)
    text = re.sub(r"\[[^\]]*\]\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if not text:
        print("Nothing recognized after cleanup.")
        return
    if DEBUG_MODE:
        print(f"Cleaned transcription: {repr(text)}")
    # Type via wtype
    try:
        wtype_env = os.environ.copy()
        if "DISPLAY" not in wtype_env:
            wtype_env["DISPLAY"] = ":1"
        wtype_env.setdefault("WAYLAND_DISPLAY", os.environ.get("WAYLAND_DISPLAY", ""))
        wtype_env.setdefault("XDG_RUNTIME_DIR", os.environ.get("XDG_RUNTIME_DIR", ""))
        subprocess.run(["wtype", " " + text], check=True, timeout=WTYPE_TIMEOUT, env=wtype_env)
        print(f"Typed: {text}")
    except subprocess.TimeoutExpired:
        print("wtype timed out -- X11/Wayland may be unavailable or no focused window")
    except subprocess.CalledProcessError as e:
        print(f"wtype failed: {e.stderr.strip() if e.stderr else e}")
    except FileNotFoundError:
        print("wtype is not installed")


if __name__ == "__main__":
    bootstrap()
    if len(sys.argv) == 1:
        print("Available input devices:")
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    print(f"  [{i}] {dev['name']} -- {dev['default_samplerate']} Hz")
        except OSError as e:
            print(f"Cannot query devices: {e}")
    device_arg = sys.argv[1] if len(sys.argv) > 1 else None
    device_idx = None
    if device_arg is not None:
        try:
            device_idx = resolve_device(device_arg)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    import queue
    import threading

    _transcribe_queue = queue.Queue()
    _transcribe_worker_done = threading.Event()

    def _transcribe_worker():
        while True:
            try:
                audio_chunks, label = _transcribe_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio_chunks is None:
                _transcribe_worker_done.set()
                return
            full_audio = np.concatenate(audio_chunks)
            total_samples = len(full_audio)
            chunk_samples = int(TRANSCRIBE_CHUNK_SIZE * SAMPLE_RATE)
            total_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)
            for i in range(0, max(1, total_samples), chunk_samples):
                seg = full_audio[i:i + chunk_samples]
                if DEBUG_MODE:
                    seg_s = len(seg) / SAMPLE_RATE
                    print(f"  [{label}] segment {i // chunk_samples + 1}/{total_chunks} ({seg_s:.1f}s)")
                transcribe_and_type(seg)
            _transcribe_queue.task_done()

    _worker_t = threading.Thread(target=_transcribe_worker, daemon=True)
    _worker_t.start()

    def _enqueue_transcribe(chunks, label=""):
        _transcribe_queue.put((chunks, label))

    silent_blocks = 0
    speech_chunks = []
    ema_rms = 0.0
    in_speech = False
    silence_debounce = 0
    energy_floor = float('inf')
    print("🎙️ Listening... (Ctrl+C to stop)")
    audio_queue = queue.Queue()

    def _audio_callback(indata, frames, time_info, status):
        if status and DEBUG_MODE:
            print(f"  [stream] {status}")
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=None, channels=1, dtype='float32',
        device=device_idx, callback=_audio_callback,
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
    ):
        if device_idx is not None:
            input_sr = int(sd.query_devices(device_idx, 'input')['default_samplerate'])
        else:
            input_sr = SAMPLE_RATE
        try:
            while True:
                raw = audio_queue.get()
                block = raw.flatten()
                if input_sr != SAMPLE_RATE and len(block) > 0:
                    target_len = int(len(block) * SAMPLE_RATE / input_sr)
                    if target_len > 0:
                        block = np.interp(
                            np.linspace(0, len(block), target_len, endpoint=False),
                            np.arange(len(block)), block,
                        )
                block = (np.clip(block, -1, 1) * 32767).astype(np.int16)
                rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
                # EMA energy smoothing & energy floor tracking
                ema_rms = VAD_EMA_ALPHA * rms + (1 - VAD_EMA_ALPHA) * ema_rms
                if not in_speech and ema_rms < energy_floor:
                    energy_floor = ema_rms
                if in_speech:
                    if ema_rms < VAD_SILENCE_THRESHOLD:
                        silent_blocks += 1
                        if silent_blocks >= SILENCE_BLOCKS:
                            total_speech_s = sum(len(c) for c in speech_chunks) / SAMPLE_RATE
                            if total_speech_s >= VAD_MIN_SPEECH_S:
                                to_transcribe = list(speech_chunks)
                                speech_chunks.clear()
                                silent_blocks = 0
                                silence_debounce = SILENCE_DEBLOCK_BLOCKS
                                in_speech = False
                                _enqueue_transcribe(to_transcribe, "silence")
                            else:
                                speech_chunks.clear()
                                silent_blocks = 0
                                in_speech = False
                        if DEBUG_MODE:
                            print(f"  🔇 silence {silent_blocks}/{SILENCE_BLOCKS}")
                    else:
                        silent_blocks = 0
                        speech_chunks.append(block)
                        if DEBUG_MODE:
                            print(f"  🗣 speech (EMA {ema_rms:.0f}) +{len(block) / SAMPLE_RATE:.1f}s")
                else:
                    if ema_rms >= VAD_SPEECH_THRESHOLD:
                        if silence_debounce > 0:
                            silence_debounce -= 1
                        else:
                            in_speech = True
                            speech_chunks.append(block)
                            silent_blocks = 0
                            if DEBUG_MODE:
                                print(f"  🗣 {ema_rms:.0f} onset")
                # Runaway guard (MAX_SPEECH_BLOCKS = 30 blocks = ~15s)
                total_speech_s = sum(len(c) for c in speech_chunks) / SAMPLE_RATE
                if total_speech_s >= BLOCK_DURATION * MAX_SPEECH_BLOCKS:
                    to_transcribe = list(speech_chunks)
                    speech_chunks.clear()
                    in_speech = False
                    _enqueue_transcribe(to_transcribe, "forced")
                    if DEBUG_MODE:
                        print(f"  ⚠️ forced flush ({total_speech_s:.1f}s)")
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            if speech_chunks:
                print(f"Final flush: {sum(len(c) for c in speech_chunks) / SAMPLE_RATE:.1f}s")
                _enqueue_transcribe(list(speech_chunks), "final")

    _transcribe_queue.put((None, None))
    _transcribe_worker_done.wait(timeout=60)
