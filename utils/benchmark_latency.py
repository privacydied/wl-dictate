"""Benchmark worker latency over the JSON IPC protocol.

Measures:
- worker boot (spawn -> {"ev": "ready"})
- start command -> {"ev": "listening"}
- stop command -> {"ev": "stopped"}
- decode latency for 3s/6s/12s windows (direct model benchmark)

Run:  .venv/bin/python utils/benchmark_latency.py [device_index]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wldictate import ipc  # noqa: E402

ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "WL_DICTATE_EMIT": "null"}


def _wait_event(process: subprocess.Popen, name: str, timeout: float = 300.0) -> float:
    started = time.perf_counter()
    deadline = started + timeout
    while time.perf_counter() < deadline:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited before event {name!r}")
        event = ipc.parse_event(line)
        if event is not None and event.ev == name:
            return time.perf_counter() - started
    raise RuntimeError(f"timed out waiting for event {name!r}")


def _send(process: subprocess.Popen, cmd: str, device: int | None = None) -> None:
    process.stdin.write(ipc.format_command(cmd, device) + "\n")
    process.stdin.flush()


def bench_protocol(device: int | None) -> dict:
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "wl_dictate.py"), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=ENV,
        cwd=str(ROOT),
    )
    try:
        boot_s = _wait_event(process, "ready")
        _send(process, "start", device)
        listen_s = _wait_event(process, "listening")
        _send(process, "stop")
        stop_s = _wait_event(process, "stopped")
        _send(process, "quit")
        process.wait(timeout=10)
        return {
            "worker_boot_s": round(boot_s, 4),
            "start_to_listening_s": round(listen_s, 4),
            "stop_to_stopped_s": round(stop_s, 4),
            "exit_code": process.returncode,
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def bench_decode() -> dict:
    """Direct decode latency: what one streaming re-decode costs."""
    import numpy as np

    from wldictate.config import Config
    from wldictate.transcriber import FasterWhisperTranscriber

    cfg = Config.load()
    tr = FasterWhisperTranscriber(cfg.model, cfg.device, cfg.compute_type)
    tr.load()
    warmup_s = tr.warmup()
    rng = np.random.default_rng(0)
    out = {
        "model": cfg.model,
        "device": tr.device,
        "compute_type": tr.compute_type,
        "warmup_s": round(warmup_s, 4),
    }
    for seconds in (3, 6, 12):
        audio = (rng.standard_normal(16000 * seconds) * 0.01).astype(np.float32)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            tr.transcribe(audio, final=False)
            times.append(time.perf_counter() - t0)
        out[f"decode_{seconds}s_median_s"] = round(sorted(times)[len(times) // 2], 4)
    return out


def main() -> int:
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    results = {"protocol": bench_protocol(device), "decode": bench_decode()}
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
