"""PyQt5 system tray app — worker lifecycle, device menu, toggle socket.

Hardened vs. the old tray_app.py:
- JSON event protocol with the worker (no stdout string matching)
- automatic worker restart with exponential backoff on abnormal exit
- toggle socket in $XDG_RUNTIME_DIR with SO_PEERCRED same-user check
- non-blocking notifications (no more 3s Qt main-thread stalls)
- config in XDG config dir, log in XDG state dir
"""

from __future__ import annotations

import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from PyQt5.QtCore import QSocketNotifier, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QActionGroup, QMenu, QSystemTrayIcon

from . import ipc
from .config import Config, socket_path, state_dir
from .notify import notify

_EVENT_TIMER_MS = 100
_LOG_ROTATE_BYTES = 512 * 1024
_RESTART_BACKOFF_BASE_S = 1.0
_RESTART_BACKOFF_MAX_S = 30.0
_RESTART_STABLE_RESET_S = 60.0


class _WorkerRelay(threading.Thread):
    """Reads worker stdout: tees raw lines to the log, queues parsed events."""

    def __init__(self, app: "DictationTrayApp", process: subprocess.Popen) -> None:
        super().__init__(daemon=True, name="worker-relay")
        self._app = app
        self._process = process

    def run(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                self._app._write_worker_log(line)
                event = ipc.parse_event(line)
                if event is not None:
                    self._app._events.append(event)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass


class _WorkerMonitor(threading.Thread):
    """Waits for worker exit and queues a synthetic exit event."""

    def __init__(self, app: "DictationTrayApp", process: subprocess.Popen) -> None:
        super().__init__(daemon=True, name="worker-monitor")
        self._app = app
        self._process = process
        self.cancelled = threading.Event()

    def run(self) -> None:
        exit_code = self._process.wait()
        if not self.cancelled.is_set():
            self._app._events.append(("exit", exit_code))


class DictationTrayApp:
    def __init__(self) -> None:
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.tray_icon = QSystemTrayIcon()
        self._device_group: QActionGroup | None = None

        self.app.aboutToQuit.connect(self.cleanup)
        self._sig_flag = False
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_sig_flag", True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_sig_flag", True))
        self._sig_timer = QTimer()
        self._sig_timer.timeout.connect(self._check_signals)
        self._sig_timer.start(250)

        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._binary_dir = os.path.dirname(os.path.abspath(sys.executable))

        self.config = Config.load()
        for warning in self.config.warnings:
            print(f"config: {warning}", file=sys.stderr)

        self.worker_process: subprocess.Popen | None = None
        self._worker_monitor: _WorkerMonitor | None = None
        self._events: deque = deque()
        self._log_lock = threading.Lock()
        self._log_file = None
        self._worker_ready = False
        self.is_dictating = False
        self._listening_notified = False
        self._cleaned = False
        self._shutting_down = False
        self._worker_spawned_at = 0.0
        self._restart_attempts = 0
        self._restart_pending = False

        self._event_timer = QTimer()
        self._event_timer.timeout.connect(self._process_events)
        self._event_timer.start(_EVENT_TIMER_MS)

        self._socket: socket.socket | None = None
        self._notifier: QSocketNotifier | None = None
        self._socket_path = str(socket_path())
        self._start_socket_listener()
        self._ensure_wayland_hotkey_binding()

        self.set_icon(False)
        self.tray_icon.show()

        self.menu = QMenu()
        self.device_menu = self.menu.addMenu("Input Device")
        self.reload_devices()
        self.toggle_action = self.menu.addAction("Toggle Dictation")
        self.toggle_action.triggered.connect(self.toggle_dictation)
        self.reload_action = self.menu.addAction("Reload Devices")
        self.reload_action.triggered.connect(self.reload_devices)
        self.quit_action = self.menu.addAction("Quit")
        self.quit_action.triggered.connect(self.quit_app)
        self.tray_icon.setContextMenu(self.menu)
        self.tray_icon.activated.connect(self.on_icon_click)

        self._prewarm_worker()

    # ── Toggle socket ────────────────────────────────────────────────────

    def _start_socket_listener(self) -> None:
        try:
            sock_path = Path(self._socket_path)
            sock_path.parent.mkdir(parents=True, exist_ok=True)
            if sock_path.exists():
                sock_path.unlink()
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.bind(self._socket_path)
            os.chmod(self._socket_path, 0o600)
            self._socket.listen(5)
            self._socket.setblocking(False)
            self._notifier = QSocketNotifier(self._socket.fileno(), QSocketNotifier.Read)
            self._notifier.activated.connect(self._socket_ready)
        except Exception as e:
            print(f"Could not start socket listener: {e}", file=sys.stderr)

    def _socket_ready(self) -> None:
        if not self._socket:
            return
        try:
            conn, _ = self._socket.accept()
        except OSError:
            return
        try:
            if not self._peer_is_same_user(conn):
                return
            conn.settimeout(0.5)
            data = conn.recv(64).decode(errors="replace").strip()
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass
        if data == "toggle":
            self.toggle_dictation()

    @staticmethod
    def _peer_is_same_user(conn: socket.socket) -> bool:
        try:
            creds = conn.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            )
            _pid, uid, _gid = struct.unpack("3i", creds)
            return uid == os.getuid()
        except OSError:
            return False

    # ── Hyprland keybinding repair (unchanged behavior) ──────────────────

    def _run_hyprctl(self, *args):
        return subprocess.run(
            ["hyprctl", *args], capture_output=True, text=True, timeout=3, check=False
        )

    def _is_compiled(self) -> bool:
        return "__compiled__" in globals() or getattr(sys, "frozen", False)

    def _get_toggle_command(self) -> str:
        if self._is_compiled():
            return sys.executable + " --toggle"
        entry = Path(self.script_dir) / "wl_dictate.py"
        return f"{sys.executable} {entry} --toggle"

    def _ensure_wayland_hotkey_binding(self) -> None:
        if os.environ.get("XDG_CURRENT_DESKTOP") != "Hyprland":
            return
        toggle_command = self._get_toggle_command()
        try:
            import json as _json

            result = self._run_hyprctl("-j", "binds")
            if result.returncode != 0:
                return
            binds = _json.loads(result.stdout)
        except Exception:
            return

        conflicting_bind = None
        for bind in binds:
            if bind.get("key", "").lower() != "f":
                continue
            if bind.get("modmask") != 12:
                continue
            arg = bind.get("arg", "")
            # Any Ctrl+Alt+F bind that already toggles dictation is good enough,
            # regardless of the exact interpreter path ("python" vs "python3")
            # or entry point. Installing our own on top produces a DUPLICATE
            # bind, so every keypress fires twice (start immediately followed by
            # stop). Leave an existing toggle bind untouched.
            if "toggle_dictation" in arg or "--toggle" in arg:
                return
            conflicting_bind = bind

        if conflicting_bind is not None:
            return
        bind_result = self._run_hyprctl(
            "keyword", "bind", f"CTRL ALT, F, exec, {toggle_command}"
        )
        if bind_result.returncode != 0:
            return
        notify("Installed Hyprland Ctrl+Alt+F dictation toggle")

    # ── Worker lifecycle ─────────────────────────────────────────────────

    def _get_worker_command(self) -> list[str]:
        if self._is_compiled():
            return [sys.executable, "--worker"]
        entry = os.path.join(self.script_dir, "wl_dictate.py")
        return [sys.executable, entry, "--worker"]

    def _prewarm_worker(self) -> None:
        try:
            self._write_worker_log(
                f"\n--- worker prewarm at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            )
            self._ensure_worker_process()
        except Exception as e:
            self._write_worker_log(f"Prewarm failed: {e}\n")

    def _ensure_worker_process(self) -> None:
        if self.worker_process and self.worker_process.poll() is None:
            return
        env = os.environ.copy()
        for var in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XDG_CURRENT_DESKTOP"):
            if not os.environ.get(var):
                env.pop(var, None)
        env["PYTHONUNBUFFERED"] = "1"
        self._worker_ready = False
        self.worker_process = subprocess.Popen(
            self._get_worker_command(),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._worker_spawned_at = time.monotonic()
        _WorkerRelay(self, self.worker_process).start()
        self._worker_monitor = _WorkerMonitor(self, self.worker_process)
        self._worker_monitor.start()

    def _send_worker_command(
        self, cmd: str, device: int | None = None, device_name: str | None = None
    ) -> None:
        if not self.worker_process or self.worker_process.stdin is None:
            raise RuntimeError("dictation worker is not running")
        try:
            self.worker_process.stdin.write(
                ipc.format_command(cmd, device, device_name) + "\n"
            )
            self.worker_process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"dictation worker is not running ({e})") from e

    # ── Event pump ───────────────────────────────────────────────────────

    def _check_signals(self) -> None:
        if self._sig_flag:
            self._sig_flag = False
            self.cleanup()
            self.app.quit()

    def _process_events(self) -> None:
        while self._events:
            item = self._events.popleft()
            if isinstance(item, tuple) and item[0] == "exit":
                self._on_worker_exit(item[1])
                continue
            event: ipc.Event = item
            if event.ev == "ready":
                self._worker_ready = True
                self._restart_attempts = 0
            elif event.ev == "listening":
                if self.is_dictating and not self._listening_notified:
                    self._listening_notified = True
                    notify("Dictation ready — speak now")
            elif event.ev == "stopped":
                pass  # informational; is_dictating is tray-driven
            elif event.ev == "error":
                self._write_worker_log(f"worker error: {event.msg}\n")
            elif event.ev == "commit":
                pass  # already typed by the worker; logged via raw tee

    def _on_worker_exit(self, exit_code: int) -> None:
        self._worker_ready = False
        self.worker_process = None
        if self._shutting_down:
            return
        was_dictating = self.is_dictating
        if was_dictating:
            self.is_dictating = False
            self.set_icon(False)
        # Killed by our own terminate/quit (negative signals) during normal
        # stop is fine; anything else during operation triggers a restart.
        if time.monotonic() - self._worker_spawned_at > _RESTART_STABLE_RESET_S:
            self._restart_attempts = 0
        backoff = min(
            _RESTART_BACKOFF_MAX_S,
            _RESTART_BACKOFF_BASE_S * (2 ** self._restart_attempts),
        )
        self._restart_attempts += 1
        self._write_worker_log(
            f"worker exited (code {exit_code}); restarting in {backoff:.0f}s\n"
        )
        if was_dictating or exit_code not in (0, -15):
            notify(f"Dictation worker died (code {exit_code}); restarting…")
        if not self._restart_pending:
            self._restart_pending = True
            QTimer.singleShot(int(backoff * 1000), self._restart_worker)

    def _restart_worker(self) -> None:
        self._restart_pending = False
        if self._shutting_down:
            return
        try:
            self._ensure_worker_process()
        except Exception as e:
            self._write_worker_log(f"worker restart failed: {e}\n")

    # ── Devices ──────────────────────────────────────────────────────────

    def reload_devices(self) -> None:
        import sounddevice as sd

        self.device_menu.clear()
        try:
            devices = sd.query_devices()
        except Exception:
            action = self.device_menu.addAction("No audio devices available")
            action.setEnabled(False)
            return
        input_devices = [
            (idx, dev) for idx, dev in enumerate(devices) if dev["max_input_channels"] > 0
        ]
        if not input_devices:
            action = self.device_menu.addAction("No input devices found")
            action.setEnabled(False)
            return
        if self._device_group:
            self._device_group.deleteLater()
        self._device_group = QActionGroup(self.device_menu)
        for idx, dev in input_devices:
            sr = dev.get("default_samplerate")
            sr_str = f"{int(sr)}" if sr and sr > 0 else "??"
            action = self.device_menu.addAction(f"{idx}: {dev['name']} ({sr_str} Hz)")
            action.setCheckable(True)
            action.setChecked(idx == self.config.input_device)
            self._device_group.addAction(action)
            action.triggered.connect(lambda _, i=idx: self.set_input_device(i))
        self.device_menu.addSeparator()
        reload_action = self.device_menu.addAction("Reload Devices")
        reload_action.triggered.connect(self.reload_devices)

    def _get_input_device_info(self, device_idx):
        import sounddevice as sd

        if not isinstance(device_idx, int):
            return None
        try:
            return sd.query_devices(device_idx, "input")
        except Exception:
            return None

    def _resolve_start_device(self):
        import sounddevice as sd

        # The saved *name* is authoritative: Pulse/PipeWire device indices
        # shift as streams appear and disappear, so a bare index can silently
        # point at a different microphone (or nothing) between sessions.
        idx = self.config.input_device
        name = self.config.input_device_name
        device_info = self._get_input_device_info(idx)
        if name and (device_info is None or device_info.get("name") != name):
            from .audio import resolve_device

            try:
                idx = resolve_device(name)
                device_info = self._get_input_device_info(idx)
            except Exception:
                device_info = None
        if device_info is not None:
            if idx != self.config.input_device or not name:
                self.config.input_device = idx
                self.config.input_device_name = device_info.get("name")
                self._save_config()
            return idx, device_info, None

        try:
            default_input, _ = sd.default.device
        except Exception:
            default_input = None
        default_info = self._get_input_device_info(default_input)
        if default_info is None:
            self.config.input_device = None
            self.config.input_device_name = None
            self._save_config()
            return None, None, "No working input device is available."
        self.config.input_device = int(default_input)
        self.config.input_device_name = default_info.get("name")
        self._save_config()
        return (
            self.config.input_device,
            default_info,
            "Saved microphone disappeared; switched to the default input.",
        )

    def set_input_device(self, device_idx: int) -> None:
        device_info = self._get_input_device_info(device_idx)
        if device_info is None:
            notify(f"Input device {device_idx} is unavailable")
            return
        self.config.input_device = device_idx
        self.config.input_device_name = device_info.get("name")
        self._save_config()
        self.set_icon(self.is_dictating)
        notify(f"Input device set to: {device_info.get('name', device_idx)}")

    def _save_config(self) -> None:
        try:
            self.config.save()
        except OSError as e:
            print(f"could not save config: {e}", file=sys.stderr)

    # ── Dictation control ────────────────────────────────────────────────

    def on_icon_click(self, reason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            self.toggle_dictation()

    def toggle_dictation(self) -> None:
        if self.is_dictating:
            self.stop_dictation()
        else:
            self.start_dictation()

    def start_dictation(self) -> None:
        if self.is_dictating:
            return
        try:
            device_idx, device_info, notice = self._resolve_start_device()
            if device_info is None:
                notify(notice or "No working input device is available.")
                return
            self._write_worker_log(
                f"\n--- dictation started at {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(device {device_idx}: {device_info.get('name', 'unknown')}) ---\n"
            )
            was_ready = self._worker_ready
            self._ensure_worker_process()
            self._listening_notified = False
            self._send_worker_command(
                "start", device_idx, device_info.get("name") if device_info else None
            )
            self.is_dictating = True
            self.set_icon(True)
            if notice:
                notify(notice)
            elif not was_ready:
                notify("Warming up dictation…")
        except Exception as e:
            notify(f"Failed to start dictation: {e}")

    def stop_dictation(self, during_cleanup: bool = False) -> None:
        if not self.is_dictating:
            return
        self.is_dictating = False
        self._listening_notified = False
        if self.worker_process and self.worker_process.poll() is None:
            try:
                self._send_worker_command("stop")
            except Exception:
                pass
        self.set_icon(False)
        if not during_cleanup and not self._shutting_down:
            notify("Dictation stopped")

    # ── Icons / resources ────────────────────────────────────────────────

    def _resource_path(self, filename: str) -> str:
        candidates = [
            os.path.join(self.script_dir, filename),
            os.path.join(self._binary_dir, filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def set_icon(self, active: bool) -> None:
        self.tray_icon.setIcon(
            QIcon(self._resource_path("mic-on.png" if active else "mic-off.png"))
        )
        status = "ON" if active else "OFF"
        device = (
            f" ({self.config.input_device})"
            if self.config.input_device is not None
            else ""
        )
        self.tray_icon.setToolTip(f"Dictation: {status}{device}")

    # ── Logging ──────────────────────────────────────────────────────────

    def _open_log_file(self):
        log_dir = state_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "worker.log"
        try:
            if log_path.exists() and log_path.stat().st_size > _LOG_ROTATE_BYTES:
                for i in range(2, 0, -1):
                    src = log_path.with_suffix(f".log.{i}")
                    dst = log_path.with_suffix(f".log.{i + 1}")
                    if src.exists():
                        os.rename(src, dst)
                os.rename(log_path, log_path.with_suffix(".log.1"))
        except OSError:
            pass
        return open(log_path, "a")

    def _write_worker_log(self, text: str) -> None:
        if not text:
            return
        with self._log_lock:
            if self._log_file is None:
                try:
                    self._log_file = self._open_log_file()
                except OSError:
                    return
            try:
                self._log_file.write(text)
                self._log_file.flush()
            except OSError:
                pass

    # ── Shutdown ─────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self._shutting_down = True
        self._sig_timer.stop()
        self._event_timer.stop()
        if self._notifier:
            try:
                self._notifier.activated.disconnect()
            except Exception:
                pass
            self._notifier.setEnabled(False)
        if self._socket:
            self._socket.close()
        if os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass
        self.stop_dictation(during_cleanup=True)
        if self._worker_monitor:
            self._worker_monitor.cancelled.set()
        if self.worker_process and self.worker_process.poll() is None:
            try:
                self._send_worker_command("quit")
                self.worker_process.wait(timeout=3.0)
            except Exception:
                try:
                    self.worker_process.kill()
                    self.worker_process.wait(timeout=2.0)
                except Exception:
                    pass
        self.worker_process = None
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def quit_app(self) -> None:
        self.cleanup()
        self.app.quit()

    def run(self) -> None:
        sys.exit(self.app.exec_())
