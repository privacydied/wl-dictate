# wl-dictate

Realtime voice dictation for Linux Wayland/X11.

It runs as a PyQt5 tray app, keeps a faster-whisper worker warm in the background, listens to your microphone, and **types words into the focused window while you are still speaking** — raw words appear almost instantly, then spelling, casing, and punctuation are **visibly fixed in place** as the recognizer refines its hypothesis (like the Claude mobile app's dictation).

![Hyprland tray example](example.jpg)

Example tray appearance on Hyprland. The tray icon/menu (the literal microphone, fourth from the right) is the main control surface for starting dictation, switching microphones, and quitting the app.

## How the realtime streaming works

Whisper is not a streaming model, so the worker fakes it properly:

1. Audio is captured at 16 kHz mono (with correct, stateful resampling when the hardware can't do 16 kHz natively).
2. A streaming **Silero VAD** (the ONNX model bundled with faster-whisper, run frame-by-frame with persistent state) segments speech, with ~320 ms of pre-roll so word onsets are never clipped.
3. While you speak, the current utterance buffer is re-decoded every ~0.5 s with word timestamps.
4. **Live correction** (`typing.mode: "correcting"`, the default): after every decode the *full* current hypothesis — stable prefix plus tentative tail — is rendered and diffed against what's on screen; the divergent suffix is backspaced and retyped in a single `wtype` invocation. Raw words land instantly and visibly self-correct. **LocalAgreement-2** (words two consecutive decodes agree on) still tracks the stable prefix for decoder context and buffer trimming. In `typing.mode: "commit"` only the agreed words are typed, append-only — text is never rewritten but trails speech by about a second.
5. When you pause, a final decode replaces the tentative tail one last time (in commit mode: flushes the rest); the utterance then becomes immutable — later utterances never backspace into it. Long utterances trim already-committed audio out of the buffer (the committed text is fed back as decoder context), so decode cost stays bounded no matter how long you talk.

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
  - `emitter.py` — wtype typing + the backspace-correcting emitter (plus stdout/null emitters for testing)
  - `transform.py` — contextual dictation: screen-context capture, LLM backends (OpenAI-compatible + Anthropic), transform coordinator
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
   - repairs/installs Hyprland runtime binds when possible: `Ctrl+Alt+F` (standard toggle), `Ctrl+Alt+D` (contextual toggle), and `Ctrl+Alt+Tab` (push-to-talk: hold to speak, release finalizes **immediately** — no silence-detection wait)

2. **Worker** (`wl_dictate.py --worker`)
   - loads faster-whisper once (default `small.en`, CUDA float16, CPU int8 fallback) and warms it up
   - JSON-lines protocol on stdin/stdout: `{"cmd": "start", "device": 3}` in, `{"ev": "commit", "text": "..."}` out
   - streams transcription as described above

## Contextual dictation (Ctrl+Alt+D)

A second dictation mode inspired by heyclicky-style contextual dictation. You speak; raw words appear instantly and self-correct exactly like standard dictation — then, when you pause, the whole utterance is **rewritten in place by an LLM that sees your screen**.

### What it can do

| you say | what happens |
|---|---|
| plain speech | cleaned up (grammar, punctuation, mishears) in the app's register — terse in a terminal, casual in Discord, prose in email. Nothing more: your wording and voice are preserved. |
| "reply to this saying I can't make it tonight" | the **reply** is typed, not the instruction — the model reads the conversation from a screenshot of the focused window (or your selection) |
| "give me the command to find what's using that port" (error in clipboard) | `lsof -i :8080` |
| "say good morning everyone in spanish" | `Buenos días a todos …` — and replies to a foreign-language thread match its language automatically |
| *(after a reply)* "actually make it more apologetic" | the previous output is **rewritten in place** — sculpt text by talking at it |
| "literally …anything…" | verbatim escape: the guard word is removed and the rest is typed exactly as dictated, never sent to the LLM |

**Context the model sees**: focused window class+title, primary selection, clipboard, the last 4 exchanges (so "that"/"it" resolve), and — with a vision model — a **screenshot of the focused window** (`contextual.screenshot`: `"local"` default = images never leave the machine, `"always"`, `"off"`).

**It streams**: the replacement is typed token-by-token (`contextual.stream`) — perceived latency is time-to-first-token (~0.3 s on the local model, prewarmed while you speak), with a final authoritative pass fixing anything the stream got wrong.

**You lose nothing, ever**: LLM down/slow/misconfigured → the dictated text is already on screen and stays. Speaking again before the transform lands cancels it. Toggling off right after speaking waits (bounded by `contextual.timeout_s`).

### Voice edit commands (LLM-free, work in standard mode too)

An utterance that is *exactly* the phrase: **"scratch that"** (or "delete that"/"undo that") deletes the previous utterance · **"new line"** inserts a line break · **"press enter"** (or "hit enter") hits Return · **"press tab"** / **"press escape"** press those keys · **"copy that"** puts the previous utterance on the clipboard.

### Long speech just works

There is no practical utterance length limit: internally the engine rolls into a fresh utterance at `vad.max_utterance_s` (default 120 s) with **no onset gap and no chopped words**, and in contextual mode the whole chain is transformed as **one message** at your real pause. Thinking pauses don't fragment your message either — contextual mode uses a longer pause threshold (`contextual.min_silence_ms`, default 800 ms) than standard mode's 500 ms.

### Make it sound like you

```json
"contextual": {
  "persona": "I'm Taz. Casual with friends, lowercase ok. Sign work emails 'T'.",
  "vocabulary": ["Hyprland", "wl-dictate", "Qwen", "OpenRouter"],
  "app_hints": { "vesktop": "very casual, emoji fine", "betterbird": "professional email tone" }
}
```

- `persona` — who's speaking; shapes tone and sign-offs in composed replies.
- `vocabulary` — names/jargon fed to **both** the LLM and Whisper's decoder prompt, so your terms stop being misheard in *both* dictation modes.
- `app_hints` — window-class substring → extra instruction for that app.

### Endpoint profiles

Switching is the single `contextual.profile` field:

| profile | backend | endpoint | model |
|---|---|---|---|
| `local` (default) | OpenAI-compatible | `http://127.0.0.1:8890/v1` (llama.cpp) | `qwen3.5-9b` (vision + MTP) |
| `local35` | OpenAI-compatible | `http://127.0.0.1:8888/v1` | `qwen36-35b-a3b` — smarter transforms when your 35B server is running |
| `openrouter` | OpenAI-compatible | `https://openrouter.ai/api/v1` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` |
| `anthropic` | Anthropic SDK | api.anthropic.com | `claude-haiku-4-5` |

**Local server**: `scripts/llama-contextual.sh` launches `llama-server`; the model and MTP settings live in **`~/.config/wl-dictate/llama.toml`** (falls back to `scripts/llama.toml`):

```toml
[server]
model = "unsloth/Qwen3.5-9B-MTP-GGUF:Q4_K_M"  # any llama.cpp -hf spec
alias = "qwen3.5-9b"
port = 8890
ctx_size = 16384

[mtp]
enabled = true    # requires an MTP-preserving GGUF; disable otherwise
draft_max = 11    # tune by "draft acceptance" in the server log
draft_min = 0
```

Environment variables (`PORT`, `CTX_SIZE`, `MTP_DRAFT_MAX`, …) override the TOML. MTP speculative decoding is the big generation-speed multiplier; the log prints per-request `draft acceptance` — consistently high (>0.5) → raise `draft_max`, consistently low (<0.2) → lower it.

**Right-sizing the model to the machine**: the default `local` profile serves a 9B model — great on a desktop GPU, too big for a laptop. `wl-dictate --check-models` reports detected VRAM/RAM and which tiers and profiles fit:

```
$ wl-dictate --check-models
Hardware:
  GPU : none detected
  RAM : 15772 MB
Local model tiers:
  9b    Qwen3.5-9B    needs 8000 MB VRAM / 12000 MB RAM  -> OK (CPU, slow)
  4b    Qwen3.5-4B    needs 4200 MB VRAM / 6000 MB RAM   -> OK (CPU, slow)
  ...
```

The worker acts on this automatically: `contextual.auto_select` (default **true**) keeps your configured profile if this machine can run it, otherwise falls back to the largest local model that fits, then to a cloud profile — so the same config works on both a desktop and a laptop. Set `auto_select: false` (or `WL_DICTATE_NO_AUTOSELECT=1`) to always honour `profile` as-is. Set `model = "auto"` in `llama.toml` to have the launch script pick the largest local GGUF that fits (`wl-dictate --check-models --pick-model`). Tune the VRAM/RAM floors in `wldictate/hardware.py` (`LOCAL_MODEL_TIERS`).

**API keys** (cloud profiles): the key file is the systemd-friendly path —

```sh
echo sk-or-... > ~/.config/wl-dictate/openrouter.key && chmod 600 ~/.config/wl-dictate/openrouter.key
echo sk-ant-... > ~/.config/wl-dictate/anthropic.key && chmod 600 ~/.config/wl-dictate/anthropic.key
```

or set the env var (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) via a systemd drop-in (`systemctl --user edit wl-dictate.service` → `[Service]` `Environment=OPENAI_API_KEY=...`). The file wins if both exist.

**Privacy**: on the `openrouter`/`anthropic` profiles your transcript, selection, and clipboard are sent to that provider (screenshots only if you set `screenshot: "always"`) — the default profile is fully local.

## Configuration

`~/.config/wl-dictate/config.json` (created on first run; a legacy `config.json` next to the binary is migrated automatically):

```json
{
  "model": "small.en",
  "device": "auto",
  "compute_type": "auto",
  "input_device": null,
  "streaming": { "enabled": true, "infer_interval_s": 0.5, "min_infer_interval_s": 0.25, "min_new_audio_s": 0.3, "max_buffer_s": 8.0 },
  "vad": { "backend": "auto", "onset": 0.5, "offset": 0.35, "onset_frames": 2, "min_silence_ms": 500, "speculative_silence_ms": 200, "pre_roll_ms": 320, "min_speech_s": 0.3, "max_utterance_s": 28.0 },
  "typing": { "mode": "correcting", "wtype_timeout_s": 10.0, "wtype_delay_ms": 6, "sentence_trailing_space": true, "capitalize_sentences": true, "electron_workaround": true },
  "audio": { "persistent_capture": true }
}
```

Useful knobs:

- `model` — any faster-whisper model id (`tiny.en`, `base.en`, `small.en`, `distil-small.en`, …). Bigger models are still realtime on a decent GPU. Set `model: "auto"` to have the worker pick per-machine from detected hardware (large-v3 on a big GPU → … → `base.en` on a CPU laptop), so one config is portable across a desktop and a laptop — see `wl-dictate --check-models`.
- `streaming.enabled: false` — revert to type-after-you-pause batch behavior.
- `typing.mode` — `"correcting"` (default): words appear instantly and are
  fixed in place via backspace + retype as the hypothesis refines. Corrections
  are keystroke injection, so don't type or move the cursor mid-utterance —
  finished utterances are never touched again. `"commit"`: the old append-only
  behavior — nothing is ever rewritten, text trails speech by ~1 s.
- `typing.capitalize_sentences` — capitalize the first letter of each sentence
  (utterance start and after `.`/`!`/`?`). Whisper often lowercases the word
  after a pause; default true. Set `false` to keep the model's raw casing.
- `typing.wtype_delay_ms` — per-keystroke delay for `wtype` (default 6).
  Set 0 for maximum speed if nothing drops characters.
- `typing.electron_workaround` — when the focused window class matches
  `typing.electron_app_classes` (Vesktop, Discord, Slack, …), large one-shot
  replacements (contextual transforms) are delivered as a clipboard paste
  (Ctrl+V) instead of keystroked — Electron apps are the slowest to
  keystroke into and reliably paste-capable; terminals/editors always
  keystroke. Default true. (The old zero-width-space "space gate" is gone:
  Electron's dropped spaces were actually caused by synthetic keys at
  made-up scancodes, fixed for good by typing every key at its real evdev
  scancode.)
- `contextual.*` — contextual dictation (see the section above): `profile` selects the LLM endpoint, `timeout_s` bounds each transform, `context_max_chars` caps selection/clipboard context, `notify` controls toasts.
- `vad.min_silence_ms` — how long a pause ends an utterance.
- `vad.speculative_silence_ms` — **speculative finalize**: after this much silence the final decode starts early, so when the pause reaches `min_silence_ms` the result is already computed — final text lands the instant the utterance ends. Resuming speech discards the speculation (some GPU time wasted, nothing else). 0 disables.
- `streaming.min_infer_interval_s` — adaptive decode cadence floor: the live re-decode interval tracks 1.5× the measured decode time between this floor and `infer_interval_s`, so a fast GPU + short utterance self-corrects at up to 4 Hz.
- Long contextual replacements (≥120 chars) in Electron apps are delivered via clipboard paste (Ctrl+V, previous clipboard restored) instead of keystrokes — a 200-char rewrite lands in one keystroke instead of ~1.2 s of typing.
- `ui.osd` — on-screen "🎤 Dictating" status pill (default true); `ui.sound_cues` — start/stop sounds (default false); `ui.idle_stop_s` — auto-stop dictation after N seconds without speech, 0 = never (mic privacy).
- `audio.persistent_capture` — keep the mic stream open across toggles (default true). Opening/closing a USB mic renegotiates isochronous bandwidth on its USB controller, which can audibly glitch *other* audio devices on the same controller; persistent capture negotiates once. While dictation is off, captured audio is discarded immediately and never transcribed. Set `false` to fully release the mic on toggle-off.
- Invalid values fall back to defaults with a warning in the log; unknown keys are reported, never fatal.

Environment overrides: `WL_DICTATE_EMIT=stdout|null` (debug/benchmark: print or discard instead of typing).

## Requirements

- **OS:** Linux with Wayland or X11. Python ≥ 3.13.
- **System:** `wtype` (types the text), `portaudio` (mic capture backend).
- **GPU (optional):** an NVIDIA GPU with CUDA + cuDNN. Without it, the worker
  transparently falls back to CPU (`int8`), which is still realtime for the
  `small.en` default.

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

## Run as a service (recommended)

`exec-once` in a compositor config only runs at session start — it won't relaunch
the app after you kill it or reload the compositor. Run it as a **systemd user
service** instead: it auto-restarts on crash *or* manual kill, and survives
config reloads.

A ready-made unit lives at [`systemd/wl-dictate.service`](systemd/wl-dictate.service).
Assuming you cloned to `~/wl-dictate` (edit the `ExecStart` paths otherwise):

```bash
mkdir -p ~/.config/systemd/user
cp systemd/wl-dictate.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now wl-dictate.service

# The tray needs the compositor's env (WAYLAND_DISPLAY etc.). If the service
# can't find the display, import it once from your session:
#   systemctl --user import-environment WAYLAND_DISPLAY XDG_RUNTIME_DIR
```

Then in your Hyprland config, replace the `exec-once` python line with an
idempotent start (safe to re-run on every reload):

```ini
exec = systemctl --user start wl-dictate.service
bind = CTRL ALT, f, exec, /path/to/wl-dictate/.venv/bin/python /path/to/wl-dictate/wl_dictate.py --toggle
```

Manage it with `systemctl --user {status,restart,stop} wl-dictate.service` and
tail logs with `journalctl --user -u wl-dictate -f`.

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
