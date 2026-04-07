import os
import sys
import time
import subprocess
import numpy as np
import sounddevice as sd
import tempfile
import shutil
from scipy.io import wavfile

# Config
CHUNK_DURATION = 5  # seconds
SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_BINARY = "whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.en.bin"
WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
DEBUG_MODE = True  # Set to False in production
THREADS = 4  # Number of threads for whisper-cli
WHISPER_TIMEOUT = 60  # max seconds for whisper-cli to transcribe one chunk
WTYPE_TIMEOUT = 10  # max seconds for wtype to type the text


def _ensure_whisper_cli():
    """Build whisper-cli from the whisper.cpp submodule if the binary is missing."""
    if os.path.isfile(WHISPER_BINARY):
        return
    print("⚠️ whisper-cli not found — building from source…")
    source_dir = os.path.dirname(WHISPER_BINARY).rsplit("/", 2)[0]  # whisper.cpp/
    if not os.path.isdir(source_dir):
        print(f"❌ whisper.cpp/ directory not found. Run: git submodule update --init --recursive")
        sys.exit(1)
    subprocess.check_call(
        ["make", "-j" + str(max(2, os.cpu_count() or 2))],
        cwd=source_dir,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if not os.path.isfile(WHISPER_BINARY):
        print("❌ Build succeeded but whisper-cli still not found")
        sys.exit(1)
    print("✔️ whisper-cli built successfully")


def _ensure_whisper_model():
    """Download the ggml-base.en model if it's missing."""
    if os.path.isfile(WHISPER_MODEL):
        return
    print(f"⚠️ Whisper model not found — downloading {WHISPER_MODEL_URL} …")
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
        print(f"❌ No curl or wget available to download the model. Download manually from {WHISPER_MODEL_URL}")
        sys.exit(1)

    if not os.path.isfile(WHISPER_MODEL):
        print("❌ Model download failed — file not found after download")
        sys.exit(1)
    print("✔️ Whisper model downloaded successfully")


def bootstrap():
    """Ensure whisper-cli is built and the model is present."""
    _ensure_whisper_cli()
    _ensure_whisper_model()
    # Update WHISPER_MODEL constant when called from tray_app.py
    # (the constants above are set at module level, so just verifying them is enough)

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
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if device_arg == dev["name"]:
            return idx
    
    # Try partial match
    for idx, dev in enumerate(devices):
        if device_arg in dev["name"]:
            return idx
            
    raise ValueError(f"No matching device for: {device_arg}")

def record_chunk(duration, device_id=None):
    """Record audio chunk from specified device, resample to 16kHz mono."""
    # Resolve device index if provided, else use default
    device_idx = None
    if device_id is not None:
        try:
            device_idx = resolve_device(device_id)
        except ValueError as e:
            print(f"❌ {e}")
            return None
    
    # Validate device index
    devices = sd.query_devices()
    input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
    
    if device_idx is not None and device_idx not in input_devices:
        print(f"❌ Invalid input device index: {device_idx}. Available input devices: {input_devices}")
        return None
    
    # Get device info for the selected device (or default)
    if device_idx is None:
        # Use the default device
        device_info = sd.query_devices(sd.default.device[0], 'input')
    else:
        device_info = sd.query_devices(device_idx, 'input')
    
    input_sr = int(device_info['default_samplerate'])
    # Force mono recording
    input_channels = 1
    
    print(f"🎙️ Recording from '{device_info['name']}' at {input_sr} Hz, {input_channels} channel(s)")
    
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
        print("⚠️ Skipping invalid or empty audio buffer")
        return None
    
    # Downmix to mono if needed (shouldn't be needed since we record mono, but keep for safety)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if needed
    if input_sr != SAMPLE_RATE:
        duration_original = len(audio) / input_sr
        target_length = int(SAMPLE_RATE * duration_original)
        audio = np.interp(
            np.linspace(0, len(audio), target_length, endpoint=False),
            np.arange(len(audio)),
            audio
        )
    
    # Convert to int16
    audio *= 32767
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    
    # Check audio level
    rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
    print(f"🎚️ Audio RMS: {rms:.2f}")
    if rms < 100:  # Threshold for silence
        print("⚠️ Warning: Audio level very low")
    
    return audio

def transcribe_and_type(audio):
    import re  # moved to function-level so it's available for the regex below

    if audio is None:
        return

    # Validate binaries before doing any work [REH][S3]
    if not os.path.isfile(WHISPER_BINARY):
        print(f"❌ whisper-cli not found at {WHISPER_BINARY}")
        return
    if _find_executable("wtype") is None and not os.path.isfile("/usr/bin/wtype"):
        print("❌ wtype is not installed or not in PATH")
        return

    # Save audio to WAV file
    wav_fd = None
    wav_path = None
    try:
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        wavfile.write(wav_path, SAMPLE_RATE, audio)
    except Exception as e:
        print(f"❌ Failed to write WAV: {e}")
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        return

    if DEBUG_MODE:
        print(f"Debug: WAV saved to {wav_path}")

    # Transcribe using whisper-cli
    whisper_error = None
    try:
        result = subprocess.run(
            [WHISPER_BINARY, "-m", WHISPER_MODEL, "-f", wav_path, "-t", str(THREADS)],
            capture_output=True,
            text=True,
            check=True,
            timeout=WHISPER_TIMEOUT,
        )

        # Extract transcription from stdout
        output = result.stdout
        if DEBUG_MODE:
            print(f"Raw whisper output: {output}")

        # Parse transcription
        transcription = output.strip()
        if transcription:
            # Clean up transcription
            # Remove SRT-style bracketed timestamps
            cleaned = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", transcription)

            # Remove timestamps at line start
            cleaned = re.sub(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\s*", "", cleaned, flags=re.MULTILINE)

            # Remove standalone numeric index lines (from SRT)
            cleaned = re.sub(r"^\d+\s*$", "", cleaned, flags=re.MULTILINE)

            # Remove residual trailing bracketed timestamps on same line
            cleaned = re.sub(r"\s*\[\d{2}:\d{2}:\d{2}\.\d{3}.*", "", cleaned)

            # Remove empty lines and join lines with space
            cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip()).strip()

            if not cleaned:
                print("⚠️ Nothing recognized after cleanup.")
                return

            if DEBUG_MODE:
                print(f"Cleaned transcription: {repr(cleaned)}")

            # Type out the transcription via wtype [REH][IV]
            try:
                subprocess.run(
                    ["wtype", cleaned],
                    check=True,
                    timeout=WTYPE_TIMEOUT,
                )
                print(f"✍️ Typed: {cleaned}")
            except subprocess.TimeoutExpired:
                print("❌ wtype timed out — X11 may be unavailable or no focused window")
            except subprocess.CalledProcessError as e:
                print(f"❌ wtype failed: {e.stderr.strip() if e.stderr else e}")
            except FileNotFoundError:
                print("❌ wtype is not installed")
        else:
            print("⚠️ Nothing recognized.")
    except subprocess.TimeoutExpired:
        print(f"❌ whisper-cli timed out after {WHISPER_TIMEOUT}s")
        whisper_error = True
    except subprocess.CalledProcessError as e:
        print(f"❌ Transcription failed: {e.stderr}")
        whisper_error = True
    except FileNotFoundError:
        print(f"❌ whisper-cli not found at {WHISPER_BINARY}")
        whisper_error = True
    finally:
        # Clean up temporary file
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

if __name__ == "__main__":
    bootstrap()
    # List devices if no arguments
    if len(sys.argv) == 1:
        print("Available input devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}] {dev['name']} — {dev['default_samplerate']} Hz")
    
    device_arg = sys.argv[1] if len(sys.argv) > 1 else None
    device_idx = None
    if device_arg is not None:
        try:
            device_idx = resolve_device(device_arg)
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
    
    try:
        while True:
            audio = record_chunk(CHUNK_DURATION, device_id=device_idx)
            if audio is not None:
                transcribe_and_type(audio)
    except KeyboardInterrupt:
        pass
