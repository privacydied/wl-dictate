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
from .config import Config
from .emitter import make_emitter
from .streaming import StreamingSession
from .textproc import TextFormatter
from .transcriber import FasterWhisperTranscriber
from .vad import VadGate, make_vad

_print_lock = threading.Lock()


def _emit(ev: str, *, text: str | None = None, msg: str | None = None) -> None:
    with _print_lock:
        print(ipc.format_event(ev, text=text, msg=msg), flush=True)


def _log(msg: str) -> None:
    _emit("log", msg=msg)


def _run_session(
    cfg: Config,
    transcriber: FasterWhisperTranscriber,
    formatter: TextFormatter,
    device: int | None,
    stop_event: threading.Event,
) -> None:
    emitter = make_emitter(cfg.typing.mode, wtype_timeout_s=cfg.typing.wtype_timeout_s)
    gate = VadGate(
        make_vad(cfg.vad.backend),
        onset=cfg.vad.onset,
        offset=cfg.vad.offset,
        onset_frames=cfg.vad.onset_frames,
        min_silence_ms=cfg.vad.min_silence_ms,
        pre_roll_ms=cfg.vad.pre_roll_ms,
        max_utterance_s=cfg.vad.max_utterance_s,
    )
    session = StreamingSession(
        transcriber,
        formatter,
        emitter,
        infer_interval_s=cfg.streaming.infer_interval_s,
        min_new_audio_s=cfg.streaming.min_new_audio_s,
        max_buffer_s=cfg.streaming.max_buffer_s,
        min_speech_s=cfg.vad.min_speech_s,
        streaming_enabled=cfg.streaming.enabled,
        on_commit=lambda text: _emit("commit", text=text),
        on_error=lambda msg: _emit("error", msg=msg),
    )

    try:
        with AudioCapture(device) as capture:
            if capture.sample_rate_in != 16000:
                _log(f"capturing at {capture.sample_rate_in} Hz (resampling to 16 kHz)")
            _emit("listening")
            last_drop_warn = 0.0
            while not stop_event.is_set():
                for frame in capture.get_frames(timeout=0.1):
                    result = gate.process(frame)
                    if result.utterance_started:
                        session.start_utterance()
                    if result.speech_frames:
                        session.feed(result.speech_frames)
                    if result.utterance_ended:
                        session.finalize()
                session.tick()
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
                session.finalize()
    except Exception as e:
        _emit("error", msg=f"session failed: {e}")
    finally:
        try:
            session.stop()
        except Exception:
            pass
        _emit("stopped")


def run() -> int:
    cfg = Config.load()
    for warning in cfg.warnings:
        _log(f"config: {warning}")

    _log(f"loading model '{cfg.model}'...")
    transcriber = FasterWhisperTranscriber(
        model_name=cfg.model, device=cfg.device, compute_type=cfg.compute_type
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

    formatter = TextFormatter()  # worker-lifetime: spacing survives toggles

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
            stop_event.clear()
            session_thread = threading.Thread(
                target=_run_session,
                args=(cfg, transcriber, formatter, command.device, stop_event),
                daemon=True,
                name="audio-session",
            )
            session_thread.start()

    _stop_session()
    _log("worker exiting")
    return 0


if __name__ == "__main__":
    sys.exit(run())
