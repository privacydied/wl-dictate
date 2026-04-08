from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "whisper_dictate.py"
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}


def main() -> int:
    process = subprocess.Popen(
        [sys.executable, str(WORKER), "--controlled"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=ENV,
        cwd=str(ROOT),
    )

    try:
        boot_started = time.perf_counter()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("worker exited before reporting readiness")
            if "Worker ready" in line:
                boot_elapsed = time.perf_counter() - boot_started
                break

        listen_started = time.perf_counter()
        process.stdin.write("start 12\n")
        process.stdin.flush()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("worker exited before entering listening state")
            if "Listening..." in line:
                listen_elapsed = time.perf_counter() - listen_started
                break

        stop_started = time.perf_counter()
        process.stdin.write("stop\n")
        process.stdin.flush()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("worker exited before stop confirmation")
            if "Session stopped" in line:
                stop_elapsed = time.perf_counter() - stop_started
                break

        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=5)

        print(
            json.dumps(
                {
                    "worker_boot_s": round(boot_elapsed, 4),
                    "start_to_listening_s": round(listen_elapsed, 4),
                    "stop_to_stopped_s": round(stop_elapsed, 4),
                    "exit_code": process.returncode,
                },
                indent=2,
            )
        )
        return 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
