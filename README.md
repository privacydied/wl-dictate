# wl-dictate

Realtime voice dictation for Linux Wayland/X11.

It runs as a PyQt5 tray app, keeps a faster-whisper worker warm in the background, listens to your microphone, and **types words into the focused window while you are still speaking** — text appears roughly a second behind your voice and is never rewritten.

![Hyprland tray example](example.jpg)

Example tray appearance on Hyprland. The tray icon/menu (the literal microphone, fourth from the right) is the main control surface for starting dictation, switching microphones, and quitting the app.

## How the realtime streaming works

Whisper is not a streaming model, so the worker fakes it properly:

1. Audio is captured at 16 kHz mono (with correct, stateful resampling when the hardware can't do 16 kHz natively).
2. A streaming **Silero VAD** (the ONNX model bundled with faster-whisper, run frame-by-frame with persistent state) segments speech, with ~320 ms of pre-roll so word onsets are never clipped.
3. While you speak, the current utterance buffer is re-decoded every ~0.5 s with word timestamps.
4. **LocalAgreement-2**: words that two consecutive decodes agree on are *committed* — cleaned up, spaced correctly, and typed immediately via `wtype`. Committed text is append-only.
5. When you pause, a final decode flushes the rest. Long utterances trim already-committed audio out of the buffer (the committed text is fed back as decoder context), so decode cost stays bounded no matter how long you talk.

Decodes run on a single background thread; if a decode is still in flight when the next tick arrives, the tick is skipped — natural backpressure, no queue growth.

## Project layout

- `wl_dictate.py` — unified entry point (tray / `--worker` / `--toggle` / `--devices`)
- `wldictate/` — the package:
  - `tray.py` — PyQt5 tray app, worker lifecycle + auto-restart, device menu, toggle socket
  - `worker.py` — controlled worker: JSON commands in, JSON events out
  - `audio.py` — 16 kHz capture, stateful resamplers, bounded queue
  - `vad.py` — streaming Silero VAD + energy fallback + utterance gate
  - `streaming.py` — LocalAgreement-2 streaming engine
  - `transcriber.py` — faster-whisper backend behind a swappable interface
  - `textproc.py` — incremental transcript cleanup and spacing
  - `emitter.py` — wtype typing (plus stdout/null emitters for testing)
  - `config.py` / `ipc.py` / `notify.py` / `toggle.py`
- `toggle_dictation.py` — compat shim for existing keybinds (`wl_dictate.py --toggle` is equivalent)
- `hotkey_listener.py` — legacy raw-evdev listener (off the main path)
- `tests/` — unit tests (no GPU/mic needed)
- `utils/benchmark_latency.py` — latency benchmark against the worker protocol

## Runtime architecture

Two long-lived processes:

1. **Tray app** (`wl_dictate.py`)
   - tray icon + device menu, config in `~/.config/wl-dictate/config.json`
   - toggle socket at `$XDG_RUNTIME_DIR/wl-dictate.sock` (same-user check via `SO_PEERCRED`)
   - prewarms the worker at startup, **auto-restarts it with backoff if it dies**
   - tees worker output to `~/.local/state/wl-dictate/worker.log`
   - repairs/installs a Hyprland `Ctrl+Alt+F` runtime bind when possible

2. **Worker** (`wl_dictate.py --worker`)
   - loads faster-whisper once (default `distil-small.en`, CUDA float16, CPU int8 fallback) and warms it up
   - JSON-lines protocol on stdin/stdout: `{"cmd": "start", "device": 3}` in, `{"ev": "commit", "text": "..."}` out
   - streams transcription as described above

## Configuration

`~/.config/wl-dictate/config.json` (created on first run; a legacy `config.json` next to the binary is migrated automatically):

```json
{
  "model": "distil-small.en",
  "device": "auto",
  "compute_type": "auto",
  "input_device": null,
  "streaming": { "enabled": true, "infer_interval_s": 0.5, "min_new_audio_s": 0.3, "max_buffer_s": 12.0 },
  "vad": { "backend": "auto", "onset": 0.5, "offset": 0.35, "onset_frames": 2, "min_silence_ms": 500, "pre_roll_ms": 320, "min_speech_s": 0.3, "max_utterance_s": 28.0 },
  "typing": { "mode": "commit", "wtype_timeout_s": 10.0, "sentence_trailing_space": true },
  "audio": { "persistent_capture": true }
}
```

Useful knobs:

- `model` — any faster-whisper model id (`tiny.en`, `base.en`, `small.en`, `distil-small.en`, …). Bigger models are still realtime on a decent GPU.
- `streaming.enabled: false` — revert to type-after-you-pause batch behavior.
- `vad.min_silence_ms` — how long a pause ends an utterance.
- `audio.persistent_capture` — keep the mic stream open across toggles (default true). Opening/closing a USB mic renegotiates isochronous bandwidth on its USB controller, which can audibly glitch *other* audio devices on the same controller; persistent capture negotiates once. While dictation is off, captured audio is discarded immediately and never transcribed. Set `false` to fully release the mic on toggle-off.
- Invalid values fall back to defaults with a warning in the log; unknown keys are reported, never fatal.

Environment overrides: `WL_DICTATE_EMIT=stdout|null` (debug/benchmark: print or discard instead of typing).

## Requirements

- **OS:** Linux with Wayland or X11. Python ≥ 3.13.
- **System:** `wtype` (types the text), `portaudio` (mic capture backend).
- **GPU (optional):** an NVIDIA GPU with CUDA + cuDNN. Without it, the worker
  transparently falls back to CPU (`int8`), which is still realtime for the
  `distil-small.en` default.

### Arch Linux (`paru`)

Some deps live in the official repos, the rest come from PyPI (via `uv`/`pip`)
or the AUR — they aren't packaged in `extra`/`core`.

```bash
# From the official repos:
paru -S wtype portaudio python-pyqt5 python-numpy python-scipy \
        python-huggingface-hub python-evdev

# GPU only (skip on a CPU-only box):
paru -S cuda cudnn
```

`sounddevice`, `faster-whisper`, `ctranslate2`, `tokenizers`, and `onnxruntime`
are **not** in the official repos. Install them into the project's virtualenv
with `uv sync` (recommended, see below), or from the AUR
(`python-onnxruntime` / `python-onnxruntime-cpu`, etc.) if you prefer a global
install — but `uv sync` is the tested path.

### Python packages

See `pyproject.toml`: `PyQt5`, `sounddevice`, `numpy`, `scipy`,
`faster-whisper`, `onnxruntime`.

```bash
uv sync              # creates .venv and installs everything pinned in uv.lock
```

## Run

```bash
uv run python wl_dictate.py             # tray app
uv run python wl_dictate.py --devices   # list microphones
uv run python wl_dictate.py --toggle    # toggle dictation (bind this to a key)
```

(Drop the `uv run` prefix if you activated `.venv` yourself, or if you're
running a bundled binary — then it's just `./dist/wl-dictate [--devices|--toggle]`.)

## Hyprland setup

```ini
# From source (via uv):
exec = uv run --project /path/to/wl-dictate python /path/to/wl-dictate/wl_dictate.py
bind = CTRL ALT, f, exec, uv run --project /path/to/wl-dictate python /path/to/wl-dictate/wl_dictate.py --toggle

# Or, if you built the binary and copied it to /usr/local/bin:
# exec = wl-dictate
# bind = CTRL ALT, f, exec, wl-dictate --toggle
```

The tray app also tries to repair/install the runtime `Ctrl+Alt+F` bind automatically when no conflicting bind exists. Existing binds pointing at `toggle_dictation.py` keep working.

## Tests

```bash
uv run pytest        # or: .venv/bin/python -m pytest
```

## Build a single binary

You don't actually need to freeze this into a binary — for a long-lived tray app
the simplest, fastest-to-iterate distribution is just `uv sync` + a wrapper that
runs `uv run python wl_dictate.py`. Bundling only helps if you want to ship a
single self-contained file to a machine without the toolchain. When you do:

**Recommended — PyInstaller** (fast build, fine startup for a persistent tray app):

```bash
paru -S python-pyinstaller     # or: uv pip install pyinstaller
./build-pyinstaller.sh         # -> dist/wl-dictate
```

PyInstaller bundles bytecode + shared libs, so the build finishes in ~a minute
instead of the many minutes Nuitka spends compiling this ML stack to C. For a
process that starts once and stays resident, its slightly slower cold start
doesn't matter.

**Alternative — Nuitka** (slow to build, produces a tighter/faster binary):

```bash
paru -S nuitka gcc             # nuitka is on the AUR
./build.sh                     # -> dist/wl-dictate
```

Only reach for Nuitka if binary size / startup latency genuinely matter to you —
the compile is *significantly* slower, especially the first time.

> **GPU binaries:** neither freezer bundles the CUDA runtime itself. On the
> target machine you still need `cuda`/`cudnn` installed (`paru -S cuda cudnn`)
> for GPU decode; otherwise the binary runs CPU-only.
