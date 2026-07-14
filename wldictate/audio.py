"""Audio capture: 16 kHz mono float32 frames, whatever the hardware speaks.

Strategy (fixes the old code, which opened the stream at the device default
rate while sizing blocks for 16 kHz and resampling with stateless np.interp):

1. Ask PortAudio for 16 kHz directly — on modern PipeWire/Pulse setups this
   almost always works and no resampling is needed.
2. Otherwise open at an integer multiple (48/32 kHz) and decimate with a
   stateful FIR low-pass (scipy.signal.lfilter with carried filter state, so
   chunk boundaries are artifact-free).
3. Otherwise open at the device default rate and use phase-continuous linear
   interpolation as a last resort.

The PortAudio callback only copies chunks into a bounded queue (drop-oldest
on overflow); resampling and re-framing to 512-sample VAD frames happen on
the consumer side.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
_QUEUE_MAX_CHUNKS = 256  # bounded: stalls drop audio instead of eating RAM


# ── Device resolution ────────────────────────────────────────────────────────


def resolve_device(device_arg: str | int | None) -> int | None:
    """Resolve an index or (partial) device name to a device index."""
    if device_arg is None or isinstance(device_arg, int):
        return device_arg
    try:
        return int(device_arg)
    except ValueError:
        pass
    try:
        devices = sd.query_devices()
    except Exception as e:
        raise ValueError(f"cannot query audio devices: {e}") from e
    for idx, dev in enumerate(devices):
        if device_arg == dev["name"] and dev["max_input_channels"] > 0:
            return idx
    for idx, dev in enumerate(devices):
        if device_arg in dev["name"] and dev["max_input_channels"] > 0:
            return idx
    raise ValueError(f"no matching input device for: {device_arg!r}")


def list_input_devices() -> list[tuple[int, str, int]]:
    out = []
    try:
        devices = sd.query_devices()
    except Exception:
        return out
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            sr = dev.get("default_samplerate") or 0
            out.append((idx, dev["name"], int(sr)))
    return out


# ── Resamplers (stateful, chunk-boundary safe) ──────────────────────────────


class _Passthrough:
    def process(self, chunk: np.ndarray) -> np.ndarray:
        return chunk


class _FirDecimator:
    """Integer-factor decimation: stateful FIR low-pass + strided pick."""

    def __init__(self, factor: int) -> None:
        from scipy.signal import firwin

        self._factor = factor
        # Cut off at 90% of the output Nyquist, normalized to input Nyquist.
        self._taps = firwin(numtaps=127, cutoff=0.9 / factor).astype(np.float64)
        self._zi = np.zeros(len(self._taps) - 1)
        self._phase = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter

        if len(chunk) == 0:
            return chunk
        y, self._zi = lfilter(self._taps, 1.0, chunk.astype(np.float64), zi=self._zi)
        idx = np.arange(self._phase, len(y), self._factor)
        out = y[idx].astype(np.float32) if len(idx) else np.zeros(0, dtype=np.float32)
        self._phase = (
            (int(idx[-1]) + self._factor - len(y)) if len(idx) else self._phase - len(y)
        )
        return out


class _LinearResampler:
    """Phase-continuous linear interpolation for arbitrary ratios (fallback)."""

    def __init__(self, in_rate: float) -> None:
        self._ratio = in_rate / SAMPLE_RATE
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._buf, chunk]) if len(self._buf) else chunk
        if len(x) < 2:
            self._buf = x
            return np.zeros(0, dtype=np.float32)
        max_pos = len(x) - 1
        n_out = int((max_pos - self._pos) / self._ratio) + 1 if self._pos <= max_pos else 0
        if n_out <= 0:
            self._buf = x
            return np.zeros(0, dtype=np.float32)
        positions = self._pos + self._ratio * np.arange(n_out)
        out = np.interp(positions, np.arange(len(x)), x).astype(np.float32)
        next_pos = self._pos + self._ratio * n_out
        keep_from = min(int(next_pos), len(x) - 1)
        self._buf = x[keep_from:]
        self._pos = next_pos - keep_from
        return out


def _make_resampler(in_rate: int):
    if in_rate == SAMPLE_RATE:
        return _Passthrough()
    if in_rate % SAMPLE_RATE == 0:
        return _FirDecimator(in_rate // SAMPLE_RATE)
    return _LinearResampler(in_rate)


# ── Capture ──────────────────────────────────────────────────────────────────


class AudioCapture:
    """Bounded-queue microphone capture yielding 512-sample 16 kHz frames."""

    def __init__(self, device: int | None) -> None:
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAX_CHUNKS)
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._resampler = None
        self._pending = np.zeros(0, dtype=np.float32)
        self.sample_rate_in = SAMPLE_RATE

    # Rates to try, best first. The device default is appended at open time.
    _PREFERRED_RATES = (SAMPLE_RATE, 48000, 32000)

    def _candidate_rates(self) -> list[int]:
        rates = list(self._PREFERRED_RATES)
        try:
            info = sd.query_devices(self._device, "input")
            default = int(info.get("default_samplerate") or 0)
            if default > 0 and default not in rates:
                rates.append(default)
        except Exception:
            pass
        return rates

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        chunk = indata[:, 0].copy()
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            with self._drop_lock:
                self._dropped += 1
            try:
                self._queue.get_nowait()  # drop oldest
                self._queue.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                pass

    def start(self) -> None:
        last_error: Exception | None = None
        for rate in self._candidate_rates():
            try:
                stream = sd.InputStream(
                    samplerate=rate,
                    device=self._device,
                    channels=1,
                    dtype="float32",
                    callback=self._callback,
                )
                stream.start()
            except Exception as e:
                last_error = e
                continue
            self._stream = stream
            self.sample_rate_in = int(stream.samplerate)
            self._resampler = _make_resampler(self.sample_rate_in)
            return
        raise RuntimeError(f"could not open audio input: {last_error}")

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def active(self) -> bool:
        return self._stream is not None

    @property
    def device(self) -> int | None:
        return self._device

    def take_dropped(self) -> int:
        """Return and reset the dropped-chunk counter."""
        with self._drop_lock:
            n, self._dropped = self._dropped, 0
            return n

    def flush(self) -> None:
        """Discard buffered audio (stale frames from an idle persistent stream)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._pending = np.zeros(0, dtype=np.float32)
        self.take_dropped()

    def get_frames(self, timeout: float = 0.1) -> Iterator[np.ndarray]:
        """Wait up to ``timeout`` for audio; yield complete 512-sample frames."""
        try:
            chunk = self._queue.get(timeout=timeout)
        except queue.Empty:
            return
        chunks = [chunk]
        # Drain whatever else is already buffered (catch-up after decode stalls).
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        for c in chunks:
            resampled = self._resampler.process(c)
            if len(resampled) == 0:
                continue
            data = (
                np.concatenate([self._pending, resampled])
                if len(self._pending)
                else resampled
            )
            n_frames = len(data) // FRAME_SAMPLES
            for i in range(n_frames):
                yield data[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            self._pending = data[n_frames * FRAME_SAMPLES :]
