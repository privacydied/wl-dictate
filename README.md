# wl-dictate

Voice dictation for Wayland/X11 Linux — records your mic, transcribes with Whisper.cpp, and types the text into any focused window using `wtype`.

## How It Works

1. Press a hotkey (or use the system tray) to start dictation
2. The tool records audio in chunks
3. Each chunk is transcribed by `whisper-cli`
4. The result is typed into your currently focused app via `wtype`

## Project Files

| File | Purpose |
|---|---|
| `tray_app.py` | System tray app (tray icon, device selector, start/stop) |
| `whisper_dictate.py` | Recording + transcription + typing worker |
| `hotkey_listener.py` | Raw evdev keyboard listener for Ctrl+Alt+D hotkey |
| `config.json` | Saves selected input device |
| `whisper.cpp/` | Git submodule — source for whisper-cli binary |

## Prerequisites

### 1. System packages

```bash
# Debian/Ubuntu
sudo apt install -y wtype libportaudio2 make

# Arch
sudo pacman -S wtype portaudio base-devel

# Fedora
sudo dnf install wtype portaudio make
```

- **`wtype`** — types text into the focused window (Wayland + X11)
- **`libportaudio2` / `portaudio`** — audio backend for sounddevice
- **`make`** / **`base-devel`** — needed for building whisper.cpp

> **Note:** `whisper-cli` is built and the model is downloaded automatically on first run.
> If `curl` or `wget` is installed the model download is fully automatic; otherwise download
> `ggml-base.en.bin` manually into `whisper.cpp/models/`.

### 2. Python dependencies

```bash
pip install sounddevice scipy numpy PyQt5 evdev
```

## Running

### System Tray App (recommended)

Launches a tray icon with device selection and a dictation toggle:

```bash
python tray_app.py
```

Left-click the tray icon to toggle dictation on/off. Right-click for a context menu where you can:
- Pick a microphone from the **Input Device** list
- Reload detected devices
- Quit the app

### CLI Mode

Runs a continuous record-transcribe-type loop in the terminal:

```bash
# List available microphones
python whisper_dictate.py

# Run with a specific device (use the index or a partial name)
python whisper_dictate.py 0
python whisper_dictate.py "USB Microphone"
```

Stops with `Ctrl+C`.

### Hotkey Listener

Listens for **Ctrl+Alt+D** at the raw input-device level (works regardless of focus/DE):

```bash
sudo python hotkey_listener.py
```

> **Note:** Requires `sudo` (or appropriate udev rules) because it reads raw `/dev/input/event*` devices.

## Configuration

`config.json` stores your selected microphone index. The tray app reads/writes this automatically.

```json
{"input_device": 2}
```

Edit `whisper_dictate.py` constants to tune behaviour:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_DURATION` | `5` | Seconds of audio per transcription cycle |
| `SAMPLE_RATE` | `16000` | Whisper-compatible sample rate |
| `THREADS` | `4` | CPU threads for whisper-cli |
| `WHISPER_TIMEOUT` | `60` | Max seconds for whisper-cli per chunk |
| `WTYPE_TIMEOUT` | `10` | Max seconds for wtype to send keystrokes |
| `DEBUG_MODE` | `True` | Print raw output and WAV paths |

## Troubleshooting

**`whisper-cli not found`** — the app tries to build it automatically. If the build fails, ensure the `whisper.cpp/` submodule is present (`git submodule update --init --recursive`) and that `make` is installed.

**`wtype is not installed`** — install the `wtype` system package (see step 1).

**No audio / silence** — run `python whisper_dictate.py` with no args to list available microphones, then pass the correct device index.

**wtype does nothing** — make sure a text field is focused. On Wayland, `wtype` needs `XDG_CURRENT_DESKTOP=wlroots` or equivalent. Test with: `wtype "hello"`.

**Hotkey needs root** — raw evdev access requires read permission on `/dev/input/event*`. Either run with `sudo` or create a udev rule to grant your user access.
