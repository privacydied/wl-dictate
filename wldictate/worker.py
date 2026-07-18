"""Controlled dictation worker.

Long-lived subprocess owned by the tray app. Loads the model once, then
starts/stops audio sessions on JSON commands from stdin and reports JSON
events on stdout (see ``wldictate.ipc``). All human-oriented text goes to
stderr or ``log`` events — stdout is exclusively the protocol channel.
"""

from __future__ import annotations

import sys
import threading
import time
from queue import Empty, Queue

from . import ipc
from .audio import AudioCapture
from .commands import match_command, strip_literal
from .config import Config
from .emitter import CorrectingEmitter, make_emitter
from .render import RenderProxy
from .streaming import StreamingSession
from .textproc import TextFormatter
from .transcriber import FasterWhisperTranscriber
from .transform import Transformer, TransformCoordinator, TransformUnavailable
from .vad import VadGate, make_vad

_print_lock = threading.Lock()


def _emit(ev: str, *, text: str | None = None, msg: str | None = None) -> None:
    with _print_lock:
        print(ipc.format_event(ev, text=text, msg=msg), flush=True)


def _log(msg: str) -> None:
    _emit("log", msg=msg)


def _handle_final(final, emitter, formatter, coordinator, merge_all=False) -> None:
    """Route a finalized utterance: voice command, verbatim escape, or
    contextual transform (in that priority order)."""
    if not final:
        return
    correcting = isinstance(getattr(emitter, "wrapped", emitter), CorrectingEmitter)
    if not merge_all:  # a rollover chain is one long message, never a command
        command = match_command(final)
        if command and correcting:
            _execute_voice_command(command, emitter, formatter)
            return
        literal = strip_literal(final) if correcting else None
        if literal is not None:
            # "literally ..." — type verbatim: remove the guard word from the
            # screen and never send this utterance to the LLM.
            emitter.sync(literal)
            formatter.reseed(literal)
            return
    if coordinator is not None:
        coordinator.submit(final, merge_all=merge_all)


def _execute_voice_command(action: str, emitter: CorrectingEmitter, formatter) -> None:
    """Run one LLM-free edit command. The spoken command itself is on screen
    (live-typed) as the current utterance region and is always removed."""
    if action == "scratch":
        emitter.merge_previous()  # reach back over the previous utterance too
        emitter.sync("")
        formatter.reseed("")
    elif action == "newline":
        emitter.sync("\n")  # command text replaced by a line break
        formatter.reseed("")
    elif action == "enter":
        emitter.sync("")  # remove the command text first
        emitter.press_key("Return")
        # Return typically submits (chat send) — the text is no longer ours.
        emitter.reset_regions()
        formatter.reseed("")
    elif action == "tab":
        emitter.sync("")
        emitter.press_key("Tab")
        # Tab may move focus — screen ownership is unknown afterwards.
        emitter.reset_regions()
        formatter.reseed("")
    elif action == "escape":
        emitter.sync("")
        emitter.press_key("Escape")
        emitter.reset_regions()
        formatter.reseed("")
    elif action == "copy":
        # Copy the previous utterance to the clipboard; remove the spoken
        # command; the text itself stays on screen.
        emitter.sync("")
        text = emitter.previous_logical.strip()
        if text:
            emitter.set_clipboard(text)
        formatter.reseed(emitter.previous_logical)


class _CaptureManager:
    """Owns the AudioCapture; in persistent mode the mic stream stays open
    across dictation toggles.

    Rationale: opening/closing a USB microphone renegotiates isochronous
    bandwidth on its USB controller, audibly glitching other audio devices on
    the same controller (e.g. a USB output interface). Persistent capture
    pays that cost once. Idle audio is discarded by the bounded queue and
    flushed on session start.
    """

    def __init__(self, persistent: bool) -> None:
        self._persistent = persistent
        self._capture: AudioCapture | None = None

    def acquire(self, device: int | None, device_name: str | None = None) -> AudioCapture:
        if self._capture is not None and self._capture.active:
            # Identity is the device *name*: Pulse/PipeWire indices drift as
            # streams appear/disappear, and reopening on a mere index change
            # is exactly the USB renegotiation glitch this manager prevents.
            current = self._capture.device_name
            same = (
                (device_name is not None and current == device_name)
                or (device_name is None and self._capture.device == device)
            )
            if same:
                self._capture.flush()
                return self._capture
            self._close()  # genuinely different device: reopen
        else:
            self._close()
        capture = AudioCapture(device)
        capture.start()
        self._capture = capture
        return capture

    def release(self) -> None:
        if not self._persistent:
            self._close()

    def invalidate(self) -> None:
        """Session hit an audio error: force a clean reopen next time."""
        self._close()

    def _close(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None

    def shutdown(self) -> None:
        self._close()


def _run_session(
    cfg: Config,
    transcriber: FasterWhisperTranscriber,
    captures: _CaptureManager,
    device: int | None,
    device_name: str | None,
    stop_event: threading.Event,
    mode: str = "standard",
    transformer: "Transformer | None" = None,
) -> None:
    # Contextual mode replaces utterances in place, which REQUIRES the
    # correcting emitter — force it regardless of typing.mode.
    contextual = mode == "contextual" and transformer is not None
    typing_mode = "correcting" if contextual else cfg.typing.mode
    # RenderProxy moves all typing I/O onto a dedicated render thread with
    # latest-wins coalescing, so the session loop (VAD + decode dispatch)
    # never blocks behind keystroke delivery.
    emitter = RenderProxy(
        make_emitter(
            typing_mode,
            wtype_timeout_s=cfg.typing.wtype_timeout_s,
            wtype_delay_ms=cfg.typing.wtype_delay_ms,
            wtype_press_delay_ms=cfg.typing.wtype_press_delay_ms,
            electron_workaround=cfg.typing.electron_workaround,
            electron_classes=cfg.typing.electron_app_classes,
            backend=cfg.typing.backend,
        ),
        on_error=lambda msg: _emit("error", msg=msg),
    )
    # Fresh formatter per session: the cursor may have moved anywhere between
    # toggles, so spacing context must never leak across sessions (a stale
    # "needs a space" flag typed leading spaces into new text fields).
    formatter = TextFormatter(
        sentence_trailing_space=cfg.typing.sentence_trailing_space,
        capitalize_sentences=cfg.typing.capitalize_sentences,
    )
    # Contextual mode uses a longer pause threshold so thinking pauses don't
    # fragment one message into separate transforms.
    min_silence_ms = cfg.vad.min_silence_ms
    if contextual and cfg.contextual.min_silence_ms > 0:
        min_silence_ms = max(min_silence_ms, cfg.contextual.min_silence_ms)
    gate = VadGate(
        make_vad(cfg.vad.backend),
        onset=cfg.vad.onset,
        offset=cfg.vad.offset,
        onset_frames=cfg.vad.onset_frames,
        min_silence_ms=min_silence_ms,
        pre_roll_ms=cfg.vad.pre_roll_ms,
        max_utterance_s=cfg.vad.max_utterance_s,
        speculative_silence_ms=cfg.vad.speculative_silence_ms,
    )
    session = StreamingSession(
        transcriber,
        formatter,
        emitter,
        infer_interval_s=cfg.streaming.infer_interval_s,
        min_infer_interval_s=cfg.streaming.min_infer_interval_s,
        min_new_audio_s=cfg.streaming.min_new_audio_s,
        max_buffer_s=cfg.streaming.max_buffer_s,
        min_speech_s=cfg.vad.min_speech_s,
        streaming_enabled=cfg.streaming.enabled,
        correcting=typing_mode == "correcting",
        on_commit=lambda text: _emit("commit", text=text),
        on_error=lambda msg: _emit("error", msg=msg),
    )
    coordinator: TransformCoordinator | None = None
    if contextual:
        coordinator = TransformCoordinator(
            transformer,
            emitter,  # RenderProxy over a CorrectingEmitter (barrier ops)
            formatter,
            timeout_s=cfg.contextual.timeout_s,
            notify_enabled=cfg.contextual.notify,
            stream_enabled=cfg.contextual.stream,
            on_error=lambda msg: _emit("error", msg=msg),
        )

    try:
        capture = captures.acquire(device, device_name)
        try:
            if capture.sample_rate_in != 16000:
                _log(f"capturing at {capture.sample_rate_in} Hz (resampling to 16 kHz)")
            _emit("listening")
            last_drop_warn = 0.0
            # Transcripts of forced-rollover chunks (long speech split across
            # engine utterances but ONE logical message for the transform).
            chunk_parts: list[str] = []
            while not stop_event.is_set():
                # 50ms cap: even with a silent queue the loop still drains
                # completed decodes / transforms promptly.
                for frame in capture.get_frames(timeout=0.05):
                    result = gate.process(frame)
                    if result.utterance_started:
                        if coordinator is not None:
                            # A late transform must never rewrite into the new
                            # utterance's region: cancel BEFORE the baseline
                            # reset in start_utterance().
                            coordinator.cancel_pending()
                        session.start_utterance()
                        if coordinator is not None:
                            # Capture screen context + prewarm the LLM while
                            # the user is speaking.
                            coordinator.prefetch()
                    if result.speech_frames:
                        session.feed(result.speech_frames)
                    if result.speculation_cancelled:
                        session.cancel_speculation()
                    if result.utterance_maybe_ended:
                        session.speculate_final()
                    if result.utterance_ended:
                        final = session.finalize()
                        if result.utterance_restarted:
                            # Long-speech rollover: keep speaking seamlessly;
                            # hold the transform until the real pause.
                            if final:
                                chunk_parts.append(final)
                            session.start_utterance(carry=True)
                        else:
                            merge_all = bool(chunk_parts)
                            if merge_all:
                                if final:
                                    final = "".join(chunk_parts) + final
                                else:
                                    final = None  # chain tail unknown: keep raw
                                chunk_parts.clear()
                            _handle_final(
                                final, emitter, formatter, coordinator, merge_all
                            )
                session.tick()
                if coordinator is not None:
                    coordinator.poll()
                dropped = capture.take_dropped()
                if dropped:
                    now = time.monotonic()
                    if now - last_drop_warn > 5.0:
                        _emit("error", msg=f"audio overrun: dropped {dropped} chunks")
                        last_drop_warn = now
            # Session stop: flush any in-flight utterance so trailing words
            # are not lost.
            flush = gate.flush()
            if flush.utterance_ended:
                final = session.finalize()
                merge_all = bool(chunk_parts)
                if merge_all:
                    final = ("".join(chunk_parts) + final) if final else None
                    chunk_parts.clear()
                _handle_final(final, emitter, formatter, coordinator, merge_all)
            if coordinator is not None:
                # Toggle-off right after speaking is the common case: give the
                # in-flight transform its full budget, then apply or give up.
                coordinator.drain()
            captures.release()
        except Exception:
            captures.invalidate()
            raise
    except Exception as e:
        _emit("error", msg=f"session failed: {e}")
    finally:
        if coordinator is not None:
            coordinator.shutdown()
        try:
            session.stop()
        except Exception:
            pass
        try:
            emitter.close()  # drain + stop the render thread
        except Exception:
            pass
        _emit("stopped")


def run() -> int:
    cfg = Config.load()
    for warning in cfg.warnings:
        _log(f"config: {warning}")

    # Pick a contextual profile this machine can actually run (a laptop can't
    # serve the 9B local model): downgrade to a smaller local model or a cloud
    # profile before the first transform, so it fails loudly here, not silently
    # on connect.
    from .hardware import autoselect_profile, resolve_whisper_model

    for note in autoselect_profile(cfg.contextual):
        _log(f"contextual: {note}")

    # Resolve model="auto" to a concrete Whisper model for THIS machine's GPU/
    # CPU (large-v3 on a big GPU, base.en on a CPU laptop) — so one config is
    # portable across machines.
    whisper_model, whisper_note = resolve_whisper_model(cfg.model)
    if whisper_note:
        _log(f"transcription: {whisper_note}")

    _log(f"loading model '{whisper_model}'...")
    transcriber = FasterWhisperTranscriber(
        model_name=whisper_model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        # Vocabulary biases Whisper decoding in BOTH dictation modes.
        vocabulary=" ".join(cfg.contextual.vocabulary),
    )
    t0 = time.monotonic()
    try:
        transcriber.load()
    except Exception as e:
        _emit("error", msg=f"model load failed: {e}")
        return 1
    warmup_s = transcriber.warmup()
    _log(
        f"model ready on {transcriber.device}/{transcriber.compute_type} "
        f"(load {time.monotonic() - t0 - warmup_s:.2f}s, warmup {warmup_s:.2f}s)"
    )
    _emit("ready")

    captures = _CaptureManager(persistent=cfg.audio.persistent_capture)
    if cfg.audio.persistent_capture:
        # Pre-open the mic now: the one-time USB isochronous bandwidth
        # negotiation happens at boot, not mid-playback on the first toggle.
        # Resolve by name — saved indices go stale as Pulse devices shift.
        device = cfg.input_device
        if cfg.input_device_name:
            try:
                from .audio import resolve_device

                device = resolve_device(cfg.input_device_name)
            except ValueError:
                pass
        try:
            captures.acquire(device, cfg.input_device_name)
            _log("persistent capture pre-opened")
        except Exception as e:
            _log(f"capture pre-open failed (will retry on start): {e}")

    commands: Queue[ipc.Command | None] = Queue()

    def _reader() -> None:
        try:
            for line in sys.stdin:
                cmd = ipc.parse_command(line)
                if cmd is not None:
                    commands.put(cmd)
                elif line.strip():
                    _log(f"ignoring malformed command: {line.strip()!r}")
        except Exception:
            pass
        finally:
            commands.put(None)  # EOF: parent died -> shut down

    threading.Thread(target=_reader, daemon=True, name="stdin-reader").start()

    stop_event = threading.Event()
    session_thread: threading.Thread | None = None

    def _session_running() -> bool:
        return session_thread is not None and session_thread.is_alive()

    def _stop_session() -> None:
        nonlocal session_thread
        if session_thread is None:
            return
        stop_event.set()
        session_thread.join(timeout=10.0)
        if session_thread.is_alive():
            _emit("error", msg="session thread did not stop in time")
        session_thread = None

    while True:
        try:
            command = commands.get(timeout=0.2)
        except Empty:
            continue
        if command is None or command.cmd == "quit":
            break
        if command.cmd == "stop":
            _stop_session()
        elif command.cmd == "start":
            if _session_running():
                _log("start ignored: session already active")
                continue
            transformer: Transformer | None = None
            if command.mode == "contextual":
                try:
                    transformer = Transformer(cfg.contextual)
                except TransformUnavailable as e:
                    _emit("error", msg=f"contextual dictation unavailable: {e}")
                    try:
                        from .notify import notify

                        notify(f"Contextual dictation unavailable: {e}")
                    except Exception:
                        pass
                    continue  # standard mode is unaffected; session not started
            # Re-resolve by name at start time: indices drift.
            device = command.device
            if command.device_name:
                try:
                    from .audio import resolve_device

                    device = resolve_device(command.device_name)
                except ValueError:
                    _log(
                        f"device {command.device_name!r} not found by name; "
                        f"using index {device}"
                    )
            stop_event.clear()
            session_thread = threading.Thread(
                target=_run_session,
                args=(
                    cfg,
                    transcriber,
                    captures,
                    device,
                    command.device_name,
                    stop_event,
                    command.mode,
                    transformer,
                ),
                daemon=True,
                name="audio-session",
            )
            session_thread.start()

    _stop_session()
    captures.shutdown()
    _log("worker exiting")
    return 0


if __name__ == "__main__":
    sys.exit(run())
