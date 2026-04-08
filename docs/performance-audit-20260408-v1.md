# Dictation performance audit and optimization plan

Date: 2026-04-08
Project: wl-dictate

## Executive summary

The biggest latency source was worker startup, specifically loading faster-whisper each time dictation was toggled. That has now been fixed by keeping a controlled worker warm in the background.

Measured after the warm-worker change:
- Worker bootstrap to "Worker ready": ~1.57s
- Start command to "Listening...": ~0.06s

That means the startup penalty is now mostly paid once when the tray app launches, not every time dictation is enabled.

The remaining user-visible latency now comes from smaller sources that add up:
- VAD block duration
- silence detection window
- Qt polling interval for readiness/worker events
- debug/logging overhead
- subprocess and notification overhead
- resampling cost for non-16k input devices
- transcription inference time itself
- typing latency via wtype

## Codebase audit by path

### tray_app.py
Responsibilities:
- system tray UI
- device discovery and persistence
- Unix socket listener for toggle_dictation.py
- Hyprland bind repair/install path
- controlled worker lifecycle
- readiness/stopped notifications

Performance-sensitive areas:
- worker process creation
- startup prewarm timing
- worker event polling interval
- notify-send subprocess calls
- log writes from worker relay

### whisper_dictate.py
Responsibilities:
- faster-whisper bootstrap
- audio capture via sounddevice InputStream
- optional resampling to 16 kHz
- VAD state machine
- background transcription queue
- typing output with wtype
- controlled worker mode for warm reuse

Performance-sensitive areas:
- model load and CUDA probe
- audio callback and queue behavior
- VAD block size / flush thresholds
- resampling frequency and cost
- transcription settings
- stdout/debug printing

### toggle_dictation.py
Responsibilities:
- lightweight socket client to toggle the tray app

Performance-sensitive areas:
- negligible in steady state
- tiny startup/subprocess overhead only

### hotkey_listener.py
Responsibilities:
- legacy raw-input hotkey listener

Performance-sensitive areas:
- not on the main Wayland path anymore
- low priority for optimization

## Bottleneck map

### 1. Repeated Whisper model bootstrap
Status: fixed
Impact: very high
Root cause:
- each dictation toggle spawned a fresh worker which reloaded faster-whisper

Fix implemented:
- persistent controlled worker subprocess
- tray app prewarms worker at startup
- start/stop commands now reuse the loaded model

### 2. VAD block duration and silence window
Status: partially fixed
Impact: high for perceived responsiveness
Root cause:
- larger blocks delay onset/silence decisions

Fix implemented:
- BLOCK_DURATION reduced from 0.5s to 0.2s
- MAX_SPEECH_BLOCKS adjusted to preserve the ~15s forced flush cap

Expected effect:
- faster first speech pickup
- faster end-of-utterance detection

### 3. Debug logging in hot path
Status: fixed
Impact: medium
Root cause:
- DEBUG_MODE was always true, causing extra printing and log flushes in the audio/VAD path

Fix implemented:
- debug mode is now opt-in via WL_DICTATE_DEBUG=1

Expected effect:
- less stdout traffic
- less log flush overhead
- less scheduler noise during transcription

### 4. Tray worker polling interval
Status: fixed
Impact: small-to-medium
Root cause:
- worker state transitions were only noticed every 500ms

Fix implemented:
- reduced worker timer interval from 500ms to 100ms

Expected effect:
- readiness and failure state propagate faster to the tray UI

## Full optimization plan

Below is the exhaustive performance plan, ordered roughly by ROI.

### A. Startup / warm path
1. Keep controlled worker prewarmed for the whole tray lifetime
   - implemented
2. Optionally defer Hyprland bind repair until after tray UI is visible
   - tiny win
3. Cache successful device resolution for the current session
   - small win
4. Measure notify-send overhead and consider suppressing some startup toasts
   - small win

### B. Audio capture and VAD
1. Tune block duration lower while keeping stable VAD behavior
   - implemented 0.5s -> 0.2s
2. Revisit SILENCE_BLOCKS after real-use testing
   - current 2 blocks means about 0.4s with 0.2s blocks
   - possible future win: dynamic silence window based on utterance length
3. Add explicit warm-up/discard of the first few audio callbacks if hardware wake-up noise causes false onset
   - situational win
4. Reduce Python work inside the main loop where possible
   - possible future refactor: split state transitions into helper functions and prebind locals
5. Investigate using 16k capture directly on supported devices to skip resampling
   - medium win on some microphones
6. Avoid np.interp if input device already supports 16000 Hz
   - partly already true via conditional path

### C. Transcription
1. Keep tiny.en warm in memory
   - implemented
2. Re-evaluate model choice versus latency budget
   - tiny.en is already the fastest practical default
3. Benchmark compute_type and device combinations on this machine
   - float16/cuda vs int8/cpu fallback
4. Add explicit one-time no-op inference warmup after bootstrap
   - possible medium first-inference win if the first real transcription is still slower than later ones
5. Consider batching cleanup work less aggressively if transcript is empty/noisy
   - tiny win

### D. Typing/output
1. Measure wtype startup cost per utterance
   - may justify a persistent typer helper only if significant
2. If wtype dominates, consider sending larger final strings less frequently rather than many small ones
   - depends on user preference and VAD tuning

### E. Tray/process management
1. Replace periodic polling with signal/FD-driven readiness handling where feasible
   - small architectural win
2. Avoid reopening log file unnecessarily
   - already improved by keeping worker alive
3. Consider line-buffered or batched log writes if logging becomes measurable again
   - only if debug enabled heavily

### F. Code structure / maintainability for performance work
1. Break up long functions in tray_app.py and whisper_dictate.py
   - easier benchmarking and micro-optimization
2. Add a dedicated benchmark script under utils/
   - measure bootstrap, listen-ready, silence flush, and first transcript latency
3. Add regression tests for warm-worker protocol and stop/start lifecycle
   - protects performance improvements from accidental rollback

## Implemented in this pass
- persistent warm worker architecture
- controlled start/stop/quit protocol
- startup prewarm from tray app
- reduced VAD block duration from 0.5s to 0.2s
- preserved forced flush cap by increasing MAX_SPEECH_BLOCKS from 30 to 75
- debug logging disabled by default unless WL_DICTATE_DEBUG=1
- tray worker poll interval reduced from 500ms to 100ms

## Recommended next implementation pass
1. Benchmark first real transcription after warm worker versus later transcriptions
2. If first inference is still slower, add explicit post-bootstrap warmup inference
3. Add utils/benchmark_latency.py to measure:
   - tray launch to worker ready
   - start command to listening
   - speech end to flush enqueue
   - enqueue to typed output
4. Profile resampling on each microphone and prefer true 16k devices when available
5. Tune silence window against actual speech samples from the user

## Risk notes
- Lower VAD block duration improves responsiveness but may increase callback churn and sensitivity to noise.
- A persistent worker improves speed but makes lifecycle/state handling more important.
- Disabling debug logs by default reduces observability unless WL_DICTATE_DEBUG=1 is set.

## Current status
The largest recurring latency problem has been eliminated. The app should now feel much faster on repeated toggles, and the remaining work is mostly micro-optimization and VAD tuning rather than architecture-level delay.
