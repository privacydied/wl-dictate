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
CHUNK_DURATION = 5  # seconds
SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_BINARY = "whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.en.bin"
WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_LIB_DIR = os.path.join(SCRIPT_DIR, "whisper.cpp", "build", "src")
DEBUG_MODE = True  # Set to False in production
THREADS = 4  # Number of threads for whisper-cli
WHISPER_TIMEOUT = 60  # max seconds for whisper-cli to transcribe one chunk
WTYPE_TIMEOUT = 10  # max seconds for wtype to type the text


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
    """Download the ggml-base.en model if it's missing."""
    if os.path.isfile(WHISPER_MODEL):
        return
    print(f"Whisper model not found -- downloading {WHISPER_MODEL_URL} ...")
    model_dir = os.path.dirname(WHISPER_MODEL)
    os.makedirs(model_dir, exist_ok=True)

    download_script = os.path.join(os.path.dirname(WHISPER_MODEL), "download-ggml-model.sh")
    if os.path.isfile(download_script):
        subprocess.check_call(
            [download_script, "base.en"],
            cwd=os.path.dirname(WHISPER_MODEL),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    elif shutil.which("curl"):
        dest = os.path.join(model_dir, os.path.basename(WHISPER_MODEL))
        subprocess.check_call(
            ["curl", "-#L", "-o", dest, WHISPER_MODEL_URL],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    elif shutil.which("wget"):
        subprocess.check_call(
            ["wget", "-O", os.path.join(model_dir, os.path.basename(WHISPER_MODEL)), WHISPER_MODEL_URL],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    else:
        print(f"No curl or wget available to download the model. Download manually from {WHISPER_MODEL_URL}")
        sys.exit(1)

    if not os.path.isfile(WHISPER_MODEL):
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
                   dtype="float32",
                   device=device_idx)
    sd.wait()

    # Handle multi-dimensional audio
    if audio.ndim > 1:
        audio = audio.squeeze()

    # Validate audio
    if audio is None or len(audio) == 0 or np.isnan(audio).any():
        print("Skipping invalid or empty audio buffer")
        return None

    # Downmix to mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample to 16kHz if needed
    if input_sr != SAMPLE_RATE:
        duration_original = len(audio) / input_sr
        target_length = int(SAMPLE_RATE * duration_original)
        if target_length == 0:  # [W1] guard against zero-length linspace
            print("Audio chunk too short to resample, skipping")
            return None
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
    print(f"Audio RMS: {rms:.2f}")
    if rms < 100:
        print("Warning: Audio level very low")

    return audio


def transcribe_and_type(audio):
    if audio is None:
        return

    # Resolve WHISPER_BINARY relative to SCRIPT_DIR if needed
    if not os.path.isabs(WHISPER_BINARY):
        binary_path = os.path.join(SCRIPT_DIR, WHISPER_BINARY)
    else:
        binary_path = WHISPER_BINARY

    # Resolve WHISPER_MODEL relative to SCRIPT_DIR
    if not os.path.isabs(WHISPER_MODEL):
        model_path = os.path.join(SCRIPT_DIR, WHISPER_MODEL)
    else:
        model_path = WHISPER_MODEL

    # Validate binaries before doing any work
    if not os.path.isfile(binary_path):
        print(f"whisper-cli not found at {binary_path}")
        return
    if shutil.which("wtype") is None and not os.path.isfile("/usr/bin/wtype"):
        print("wtype is not installed or not in PATH")
        return

    # Validate model
    if not os.path.isfile(model_path):
        print(f"Whisper model not found at {model_path}")
        return

    # Ensure wtype can reach the active display
    if "WAYLAND_DISPLAY" not in os.environ and "XDG_RUNTIME_DIR" not in os.environ:
        _guess_wayland_display()

    # Save audio to WAV file
    wav_fd = None
    wav_path = None
    try:
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        wavfile.write(wav_path, SAMPLE_RATE, audio)
    except Exception as e:
        print(f"Failed to write WAV: {e}")
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        return

    if DEBUG_MODE:
        print(f"Debug: WAV saved to {wav_path}")

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

        # Extract transcription from stdout
        output = result.stdout
        if DEBUG_MODE:
            print(f"Raw whisper output: {output}")

        # Parse transcription
        transcription = output.strip()
        if transcription:
            # Remove SRT-style bracketed timestamps
            cleaned = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", transcription)
            # Remove timestamps at line start
            cleaned = re.sub(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\s*", "", cleaned, flags=re.MULTILINE)
            # Remove standalone numeric index lines
            cleaned = re.sub(r"^\d+\s*$", "", cleaned, flags=re.MULTILINE)
            # Remove residual trailing bracketed timestamps
            cleaned = re.sub(r"\s*\[\d{2}:\d{2}:\d{2}\.\d{3}.*", "", cleaned)
            # Remove empty lines and join
            cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip()).strip()

            if not cleaned:
                print("Nothing recognized after cleanup.")
                return

            if DEBUG_MODE:
                print(f"Cleaned transcription: {repr(cleaned)}")

            # Type out the transcription via wtype
            try:
                wtype_env = os.environ.copy()
                if "DISPLAY" not in wtype_env:
                    wtype_env["DISPLAY"] = ":0"

                subprocess.run(
                    ["wtype", cleaned],
                    check=True,
                    timeout=WTYPE_TIMEOUT,
                    env=wtype_env,
                )
                print(f"Typed: {cleaned}")
            except subprocess.TimeoutExpired:
                print("wtype timed out -- X11 may be unavailable or no focused window")
            except subprocess.CalledProcessError as e:
                print(f"wtype failed: {e.stderr.strip() if e.stderr else e}")
            except FileNotFoundError:
                print("wtype is not installed")
        else:
            print("Nothing recognized.")
    except subprocess.TimeoutExpired:
        print(f"whisper-cli timed out after {WHISPER_TIMEOUT}s")
    except subprocess.CalledProcessError as e:
        print(f"Transcription failed: {e.stderr}")
    except FileNotFoundError:
        print(f"whisper-cli not found at {WHISPER_BINARY}")
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

    try:
        while True:
            audio = record_chunk(CHUNK_DURATION, device_id=device_idx)
            if audio is not None:
                transcribe_and_type(audio)
    except KeyboardInterrupt:
        pass
