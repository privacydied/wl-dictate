import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Config
BLOCK_DURATION = 0.3       # seconds per recording block
SILENCE_BLOCKS = 2         # blocks of silence to trigger transcribe (~0.6s pause)
MAX_SPEECH_BLOCKS = 7      # force transcribe after ~2.1s of speech even without silence
SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_BINARY = "whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL_NAME = "tiny.en"  # tiny.en = near-instant on CPU
DESIRED_MODEL_PATH = f"whisper.cpp/models/ggml-{WHISPER_MODEL_NAME}.bin"
VAD_RMS_THRESHOLD = 500     # below this = silent block
THREADS = 16                # Ryzen 3700X has 8C/16T
WHISPER_TIMEOUT = 30
WTYPE_TIMEOUT = 10
DEBUG_MODE = True
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_LIB_DIR = os.path.join(SCRIPT_DIR, "whisper.cpp", "build", "src")
WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"

def _ensure_whisper_cli():
    """Build whisper-cli from the whisper.cpp submodule if the binary is missing."""
    if os.path.isfile(WHISPER_BINARY):
        return
    print("whisper-cli not found -- building from source...")
    source_dir = os.path.join(SCRIPT_DIR, "whisper.cpp")
    if not os.path.isdir(source_dir):
        print("whisper.cpp/ directory not found. Run: git submodule update --init --recursive")
        sys.exit(1)
    subprocess.check_call(
        ["make", "-j" + str(max(2, os.cpu_count() or 2))],
        cwd=source_dir,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if not os.path.isfile(WHISPER_BINARY):
        print("Build succeeded but whisper-cli still not found")
        sys.exit(1)
    print("whisper-cli built successfully")


def _ensure_whisper_model():
    """Download the configured whisper model if it's missing."""
    if os.path.isfile(DESIRED_MODEL_PATH):
        return
    url = WHISPER_MODEL_URL.format(name=WHISPER_MODEL_NAME)
    print(f"Whisper model not found -- downloading {url} ...")
    model_dir = os.path.dirname(DESIRED_MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)

    download_script = os.path.join(model_dir, "download-ggml-model.sh")
    if os.path.isfile(download_script):
        subprocess.check_call(
            [download_script, WHISPER_MODEL_NAME],
            cwd=model_dir,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    elif shutil.which("curl"):
        dest = os.path.join(model_dir, os.path.basename(DESIRED_MODEL_PATH))
        subprocess.check_call(
            ["curl", "-#L", "-o", dest, url],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    elif shutil.which("wget"):
        subprocess.check_call(
            ["wget", "-O", os.path.join(model_dir, os.path.basename(DESIRED_MODEL_PATH)), url],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    else:
        print(f"No curl or wget available. Download manually from {url}")
        sys.exit(1)

    if not os.path.isfile(DESIRED_MODEL_PATH):
        print("Model download failed -- file not found after download")
        sys.exit(1)
    print("Whisper model downloaded successfully")


def bootstrap():
    """Ensure whisper-cli is built and the model is present."""
    _ensure_whisper_cli()
    _ensure_whisper_model()


def resolve_device(device_arg):
    """Resolve device argument to device index."""
    if device_arg is None:
        return None  # use default

    # Try as integer ID
    try:
        return int(device_arg)
    except ValueError:
        pass

    # Try to match exact name
    try:
        devices = sd.query_devices()
    except OSError as e:
        raise ValueError(f"Cannot query audio devices: {e}") from e

    for idx, dev in enumerate(devices):
        if device_arg == dev["name"]:
            return idx

    # Try partial match
    for idx, dev in enumerate(devices):
        if device_arg in dev["name"]:
            return idx

    raise ValueError(f"No matching device for: {device_arg}")


def _guess_wayland_display():
    """Try to infer WAYLAND_DISPLAY from runtime dirs [W5]."""
    candidates = ["/run/user/1000"]
    try:
        import pwd
        uid = os.getuid()
        candidates.append(f"/run/user/{uid}")
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

    # Validate device index
    try:
        devices = sd.query_devices()
    except OSError as e:
        print(f"Cannot query audio devices: {e}")  # [W3]
        return None

    input_devices = [i for i, dev in enumerate(devices) if dev["max_input_channels"] > 0]

    if device_idx is not None and device_idx not in input_devices:
        print(f"Invalid input device index: {device_idx}. Available input devices: {input_devices}")
        return None

    # Get device info for the selected device (or default)
    try:
        if device_idx is None:
            device_info = sd.query_devices(sd.default.device[0], "input")
        else:
            device_info = sd.query_devices(device_idx, "input")
    except OSError as e:
        print(f"Cannot query device info: {e}")  # [W3]
        return None

    input_sr = int(device_info["default_samplerate"])
    input_channels = 1

    device_name = device_info.get("name", "unknown")
    print(f"Recording from '{device_name}' at {input_sr} Hz, {input_channels} channel(s)")

    # Record at device's native settings
    audio = sd.rec(int(duration * input_sr),
                   samplerate=input_sr,
                   channels=input_channels,
                   dtype='float32',
                   device=device_idx)  # None means default
    sd.wait()

    # Handle multi-dimensional audio
    if audio.ndim > 1:
        audio = audio.squeeze()

    # Validate audio
    if audio is None or len(audio) == 0 or np.isnan(audio).any():
        print("Warning: Invalid or empty audio buffer")
        return None

    # Downmix to mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample to 16kHz
    if input_sr != SAMPLE_RATE:
        duration_original = len(audio) / input_sr
        target_length = int(SAMPLE_RATE * duration_original)
        audio = np.interp(
            np.linspace(0, len(audio), target_length, endpoint=False),
            np.arange(len(audio)),
            audio,
        )

    # Convert to int16
    audio *= 32767
    audio = np.clip(audio, -32768, 32767).astype(np.int16)

    # Check audio level
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    if DEBUG_MODE:
        print(f"Audio RMS: {rms:.2f}")

    return audio


def transcribe_and_type(audio):
    """Transcribe audio via whisper-cli and type the result with wtype."""
    import traceback as _traceback

    if audio is None:
        return

    binary_path = os.path.join(SCRIPT_DIR, WHISPER_BINARY)
    model_path = os.path.join(SCRIPT_DIR, DESIRED_MODEL_PATH)

    if not os.path.isfile(binary_path):
        print(f"whisper-cli not found at {binary_path}")
        return
    if not os.path.isfile(model_path):
        print(f"Whisper model not found at {model_path}")
        return

    # Ensure Wayland env is available
    if "WAYLAND_DISPLAY" not in os.environ and "XDG_RUNTIME_DIR" not in os.environ:
        _guess_wayland_display()

    wav_fd, wav_path = None, None
    try:
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        wavfile.write(wav_path, SAMPLE_RATE, audio)
    except Exception as e:
        print(f"Failed to write WAV: {e}")
        return

    # Transcribe using whisper-cli
    run_env = os.environ.copy()
    run_env["LD_LIBRARY_PATH"] = WHISPER_LIB_DIR
    try:
        result = subprocess.run(
            [binary_path, "-m", model_path, "-f", wav_path, "-t", str(THREADS)],
            capture_output=True,
            text=True,
            check=True,
            timeout=WHISPER_TIMEOUT,
            env=run_env,
        )

        output = result.stdout
        if DEBUG_MODE:
            print(f"Raw whisper output: {output}")

        transcription = output.strip()
        if transcription:
            # Remove SRT timestamps and index lines
            cleaned = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", transcription)
            cleaned = re.sub(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\s*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"^\d+\s*$", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"\s*\[\d{2}:\d{2}:\d{2}\.\d{3}.*", "", cleaned)
            cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip()).strip()

            # Strip non-speech annotations: (heavy breathing), [Music], etc.
            cleaned = re.sub(r"\([^)]*\)\s*", "", cleaned)
            cleaned = re.sub(r"\[[^\]]*\]\s*", "", cleaned)

            # Collapse extra whitespace
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

            if not cleaned:
                print("Nothing recognized after cleanup.")
                return

            if DEBUG_MODE:
                print(f"Cleaned transcription: {repr(cleaned)}")

            # Type via wtype, prepending a space to separate from prior text
            try:
                wtype_env = os.environ.copy()
                if "DISPLAY" not in wtype_env:
                    wtype_env["DISPLAY"] = ":1"
                wtype_env.setdefault("WAYLAND_DISPLAY", os.environ.get("WAYLAND_DISPLAY", ""))
                wtype_env.setdefault("XDG_RUNTIME_DIR", os.environ.get("XDG_RUNTIME_DIR", ""))

                subprocess.run(
                    ["wtype", " " + cleaned],
                    check=True,
                    timeout=WTYPE_TIMEOUT,
                    env=wtype_env,
                )
                print(f"Typed: {cleaned}")
            except subprocess.TimeoutExpired:
                print("wtype timed out -- X11/Wayland may be unavailable or no focused window")
            except subprocess.CalledProcessError as e:
                print(f"wtype failed: {e.stderr.strip() if e.stderr else e}")
            except FileNotFoundError:
                print("wtype is not installed")
        else:
            print("Nothing recognized.")
    except subprocess.TimeoutExpired:
        print(f"whisper-cli timed out after {WHISPER_TIMEOUT}s")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else str(e)
        print(f"Transcription failed (exit {e.returncode}): {stderr}")
    except FileNotFoundError:
        print(f"whisper-cli not found at {binary_path}")
    except Exception:
        print("Unexpected transcription error:")
        _traceback.print_exc()
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    bootstrap()

    # List devices if no arguments
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

    # VAD-based streaming loop
    # - Accumulate 0.3s blocks
    # - Transcribe after 0.6s silence (~2 blocks)
    # - Also force-transcribe every ~2.1s of speech so text appears during long speech
    try:
        silent_blocks = 0
        speech_blocks = 0
        buffer = []

        while True:
            block = record_chunk(BLOCK_DURATION, device_id=device_idx)
            if block is None:
                continue

            rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))

            if rms >= VAD_RMS_THRESHOLD:
                # Speech — accumulate
                silent_blocks = 0
                speech_blocks += 1
                buffer.append(block)
                if DEBUG_MODE:
                    secs = speech_blocks * BLOCK_DURATION
                    print(f"  🗣 speech (RMS {rms:.0f}) — {secs:.1f}s accumulated")

                # Force transcribe if we've accumulated enough speech
                if speech_blocks >= MAX_SPEECH_BLOCKS:
                    full_audio = np.concatenate(buffer)
                    buffer.clear()
                    speech_blocks = 0
                    if DEBUG_MODE:
                        print(f"⚡ Force transcribe {len(full_audio) / SAMPLE_RATE:.1f}s")
                    transcribe_and_type(full_audio)
            else:
                # Silence — increment counter
                silent_blocks += 1
                if DEBUG_MODE:
                    print(f"  🔇 silence {silent_blocks}/{SILENCE_BLOCKS}")

                if len(buffer) > 0 and silent_blocks >= SILENCE_BLOCKS:
                    # Enough silence — transcribe the accumulated buffer
                    full_audio = np.concatenate(buffer)
                    buffer.clear()
                    silent_blocks = 0
                    speech_blocks = 0
                    if DEBUG_MODE:
                        total_s = len(full_audio) / SAMPLE_RATE
                        print(f"→ Transcribing {total_s:.1f}s of audio")
                    transcribe_and_type(full_audio)

    except KeyboardInterrupt:
        # Flush remaining buffer on exit
        if buffer:
            full_audio = np.concatenate(buffer)
            if DEBUG_MODE:
                print(f"Final flush: {len(full_audio) / SAMPLE_RATE:.1f}s")
            transcribe_and_type(full_audio)
        pass
