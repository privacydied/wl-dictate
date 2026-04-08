# wl-dictate

Low-latency voice dictation for Linux Wayland/X11.

It runs as a PyQt5 tray app, keeps a faster-whisper worker warm in the background, listens to your microphone with `sounddevice`, and types the cleaned transcript into the currently focused window with `wtype`.

![Hyprland tray example](example.jpg)

Example tray appearance on Hyprland. `tray_app.py` The tray icon/menu (the literal microphone, fourth from the right) is the main control surface for starting dictation, switching microphones, and quitting the app.

## What it does

- starts from a system tray icon
- prewarms the speech model once so toggling dictation is fast after launch
- records audio from the selected microphone
- detects speech with a lightweight VAD loop
- transcribes with `faster-whisper`
- cleans up noisy transcript formatting before typing
- types into the focused app with `wtype`
- exposes a local Unix socket so external keybinds can toggle dictation
- repairs/installs a Hyprland `Ctrl+Alt+F` runtime bind automatically when possible

## Project files

- `tray_app.py` — PyQt5 tray app, worker lifecycle, device menu, socket listener, Hyprland bind repair
- `whisper_dictate.py` — warm worker, audio capture, VAD, transcription, transcript cleanup, typing
- `toggle_dictation.py` — tiny CLI that sends `toggle` over the Unix socket
- `hotkey_listener.py` — legacy raw-evdev listener for `Ctrl+Alt+F` when running with elevated input permissions
- `mic-on.png` / `mic-off.png` — tray icons
- `example.jpg` — screenshot of the tray on Hyprland
- `utils/benchmark_latency.py` — benchmark helper for worker boot/start/stop timing
- `docs/performance-audit-20260408-v1.md` — current performance audit and optimization plan

## Runtime architecture

There are two long-lived processes in the normal tray workflow:

1. `tray_app.py`
   - creates the tray icon and context menu
   - loads/saves `config.json`
   - listens on `.dictation.sock`
   - prewarms the worker at startup
   - sends `start`, `stop`, and `quit` commands to the worker

2. `whisper_dictate.py --controlled`
   - loads `faster-whisper` once
   - waits for commands on stdin
   - starts/stops audio sessions without reloading the model
   - writes state lines like `Worker ready`, `Listening...`, and `Session stopped`

That warm-worker design is what removes the big per-toggle startup penalty.

## Requirements

System packages you need:

- `wtype`
- `portaudio` / `libportaudio2`
- Python 3

Python packages used by the code:

- `PyQt5`
- `sounddevice`
- `numpy`
- `faster-whisper`
- `evdev` (legacy raw hotkey path)
- `ctranslate2` via `faster-whisper`

## Run the tray app

Use the project venv if you have one:

```bash
/mnt/nasirjones/py/wl-dictate/.venv/bin/python3 /mnt/nasirjones/py/wl-dictate/tray_app.py
```

Or from the repo directory:

```bash
python tray_app.py
```

When the tray app starts it:

- opens the tray icon
- creates the local control socket
- prewarms the worker
- starts showing `mic-off.png`
- allows left-click toggle and right-click menu actions

## Hyprland setup

The intended Wayland path is a compositor bind that triggers `toggle_dictation.py`.

Hyprland example:

```ini
exec = python /home/pry/wl-dictate/tray_app.py
bind = CTRL ALT, f, exec, python /home/pry/wl-dictate/toggle_dictation.py
```

Notes:

- the tray app should be launched once per session
- the bind talks to the tray app over the Unix socket
- the tray app also tries to repair/install the runtime Hyprland bind automatically if there is no conflicting bind already using `Ctrl+Alt+F`

## Manual toggle command

If you just want to trigger dictation from a shell or another launcher:

```bash
python toggle_dictation.py
```

That command does not transcribe anything by itself. It only asks the running tray app to toggle state.

## Legacy raw hotkey listener

`hotkey_listener.py` still exists, but it reads raw `/dev/input/event*` devices. That means it is not the normal Hyprland path.

Run it only if you specifically want the raw-input mode and have the required device permissions:

```bash
sudo python hotkey_listener.py
```

## CLI worker mode

You can still run the transcription worker directly:

```bash
python whisper_dictate.py
python whisper_dictate.py 0
python whisper_dictate.py "USB Microphone"
```

That bypasses the tray workflow and runs a direct dictation session.

## Microphone selection

The tray menu shows all input devices returned by `sounddevice.query_devices()`.

Behavior:

- your selected device index is saved in `config.json`
- if the saved device disappears, the tray app falls back to the current default input device
- if no working input device exists, dictation will not start

Current `config.json` format:

```json
{"input_device": 2}
```

## Transcript cleanup behavior

Before typing, the worker normalizes the transcript.

Current cleanup includes:

- removing bracketed and parenthesized noise like `[noise]` or `(music)`
- trimming leading whitespace, including weird Unicode whitespace
- collapsing repeated whitespace to a single space
- normalizing ellipses
- inserting missing spaces after sentence punctuation in cases like:
  - `Testing.How are we doing today?One, two, three.`
  - becomes `Testing. How are we doing today? One, two, three.`
- preserving common numeric punctuation like decimals such as `3.14`

## Performance notes

Important current behavior from the codebase:

- the worker is prewarmed at tray startup
- dictation start now reuses the loaded model instead of spawning a fresh one each time
- VAD block duration is `0.2s`
- tray worker polling interval is `100ms`
- debug logging is off by default
- enable debug logs with:

```bash
WL_DICTATE_DEBUG=1 python tray_app.py
```

Latency benchmark helper:

```bash
python utils/benchmark_latency.py
```

## Wayland typing notes

Typing depends on `wtype`, which needs a valid Wayland session environment.

The worker caches:

- `WAYLAND_DISPLAY`
- `XDG_RUNTIME_DIR`

If they are missing, the worker tries to guess them from the runtime directory.

## Troubleshooting

### Tray app starts but toggle does nothing

Check:

- the tray app is actually running
- `.dictation.sock` exists in the project directory
- your Hyprland bind points to the current repo path, not an old directory

### `Ctrl+Alt+F` does nothing on Hyprland

Check your live binds:

```bash
hyprctl -j binds | grep toggle_dictation.py
```

If the bind points at an old path, update it and reload Hyprland.

### Dictation starts but nothing gets typed

Check:

- `wtype` is installed
- a text field is focused
- the worker has valid `WAYLAND_DISPLAY` and `XDG_RUNTIME_DIR`

### Dictation feels slow the first time after launching the tray

The worker is loading the model during prewarm. After that, repeated toggles should be much faster.

### Saved microphone broke after unplugging hardware

The tray app validates the saved device. If it is gone, it falls back to the default input device and updates `config.json`.

### CUDA fails

If CUDA is unavailable, the worker falls back to CPU/int8 mode.

## Development notes

Useful checks:

```bash
python -m py_compile tray_app.py whisper_dictate.py toggle_dictation.py hotkey_listener.py utils/benchmark_latency.py
python utils/benchmark_latency.py
```

## Summary

Use the tray app for normal use.
Use the Hyprland `Ctrl+Alt+F` bind to toggle it.
Keep the tray app running so the warm worker stays loaded and dictation starts fast.
