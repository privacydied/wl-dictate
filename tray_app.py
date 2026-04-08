import os
import subprocess
import sys
import json
import atexit
import time
import socket
import threading
from pathlib import Path
import sounddevice as sd

WORKER_TIMER_MS = 100
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QActionGroup
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSocketNotifier, QTimer
class WorkerMonitor(threading.Thread):
    """Watches the dictation worker subprocess and signals on exit."""

    def __init__(self, tray_app, process):
        super().__init__(daemon=True)
        self.tray_app = tray_app
        self.process = process
        self._stop = threading.Event()

    def run(self):
        exit_code = self.process.wait()
        if self._stop.is_set():
            return
        # exit_code < 0 means killed by signal (e.g. -15 = SIGTERM from terminate())
        if exit_code != 0 and exit_code != -15:
            print(f"WARNING: Dictation worker exited abnormally (code {exit_code})")
        self.tray_app._worker_event.set()


class WorkerLogRelay(threading.Thread):
    """Tee worker stdout into the log file and detect readiness."""

    def __init__(self, tray_app, process):
        super().__init__(daemon=True)
        self.tray_app = tray_app
        self.process = process

    def run(self):
        stream = self.process.stdout
        if stream is None:
            return
        for line in stream:
            self.tray_app._write_worker_log(line)
            if "Worker ready" in line:
                self.tray_app._worker_boot_event.set()
            if "Listening..." in line:
                self.tray_app._worker_ready_event.set()
        try:
            stream.close()
        except Exception:
            pass


class DictationTrayApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.tray_icon = QSystemTrayIcon()
        self._device_group = None

        atexit.register(self.cleanup)
        self.app.aboutToQuit.connect(self.cleanup)

        import signal as _signal

        self._sig_flag = False
        _signal.signal(_signal.SIGINT, lambda *_: setattr(self, "_sig_flag", True))
        _signal.signal(_signal.SIGTERM, lambda *_: setattr(self, "_sig_flag", True))
        self._sig_timer = QTimer()
        self._sig_timer.timeout.connect(self._check_signals)
        self._sig_timer.start(250)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.script_dir, "config.json")
        self.load_config()

        self.dictation_process = None
        self._worker_monitor = None
        self._worker_log_relay = None
        self._worker_event = threading.Event()
        self._worker_boot_event = threading.Event()
        self._worker_ready_event = threading.Event()
        self._log_lock = threading.Lock()
        self._log_file = None
        self._worker_booted = False
        self._worker_ready = False
        self.is_dictating = False
        self._intentional_stop = False
        self._cleaned = False
        self._shutting_down = False

        self._worker_timer = QTimer()
        self._worker_timer.timeout.connect(self._check_worker)
        self._worker_timer.start(WORKER_TIMER_MS)

        self._socket_path = os.path.join(self.script_dir, ".dictation.sock")
        self._socket = None
        self._notifier = None
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

    def _start_socket_listener(self):
        try:
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.bind(self._socket_path)
            self._socket.listen(5)
            self._socket.setblocking(False)
            self._notifier = QSocketNotifier(
                self._socket.fileno(), QSocketNotifier.Read
            )
            self._notifier.activated.connect(self._socket_ready)
        except Exception as e:
            print(f"Could not start socket listener: {e}")

    def _socket_ready(self):
        if not self._socket:
            return
        try:
            conn, _ = self._socket.accept()
            try:
                data = conn.recv(64).decode().strip()
            finally:
                conn.close()
            if data == "toggle":
                self.toggle_dictation()
        except OSError:
            pass

    def _run_hyprctl(self, *args):
        return subprocess.run(
            ["hyprctl", *args],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )

    def _ensure_wayland_hotkey_binding(self):
        if os.environ.get("XDG_CURRENT_DESKTOP") != "Hyprland":
            return
        toggle_script = Path(self.script_dir) / "toggle_dictation.py"
        if not toggle_script.exists():
            return
        try:
            result = self._run_hyprctl("-j", "binds")
            if result.returncode != 0:
                return
            binds = json.loads(result.stdout)
        except Exception:
            return

        matching_bind = None
        conflicting_bind = None
        expected_suffix = str(toggle_script)
        for bind in binds:
            if bind.get("key", "").lower() != "f":
                continue
            if bind.get("modmask") != 12:
                continue
            arg = bind.get("arg", "")
            if "toggle_dictation.py" in arg:
                matching_bind = bind
                if expected_suffix in arg:
                    return
            else:
                conflicting_bind = bind

        if conflicting_bind is not None:
            return

        if matching_bind is not None:
            self._run_hyprctl("keyword", "unbind", "CTRL ALT, F")

        command = f"{sys.executable} {toggle_script}"
        bind_value = f"CTRL ALT, F, exec, {command}"
        bind_result = self._run_hyprctl("keyword", "bind", bind_value)
        if bind_result.returncode != 0:
            return
        message = "Installed Hyprland Ctrl+Alt+F dictation toggle"
        if matching_bind is not None:
            message = "Repaired Hyprland Ctrl+Alt+F dictation toggle"
        try:
            subprocess.run(
                ["notify-send", "-t", "3000", "Dictation Tool", message],
                timeout=3,
            )
        except Exception:
            pass

    def _check_signals(self):
        if self._sig_flag:
            self._sig_flag = False
            self.cleanup()
            self.app.quit()

    def _check_worker(self):
        if self._worker_boot_event.is_set():
            self._worker_boot_event.clear()
            self._worker_booted = True

        if self._worker_ready_event.is_set():
            self._worker_ready_event.clear()
            if self.is_dictating and not self._worker_ready:
                self._worker_ready = True
                try:
                    subprocess.run(
                        ["notify-send", "-t", "3000", "Dictation Tool", "Dictation ready"],
                        timeout=3,
                    )
                except Exception:
                    pass

        if self._worker_event.is_set():
            self._worker_event.clear()
            self._worker_booted = False
            self._worker_ready = False
            if self._intentional_stop:
                self._intentional_stop = False
                return
            if self._shutting_down:
                return  # don't spam notification during shutdown
            if self.is_dictating:
                self.is_dictating = False
                self.set_icon(False)
                try:
                    subprocess.run(
                        [
                            "notify-send",
                            "-t",
                            "3000",
                            "Dictation Tool",
                            "Dictation stopped unexpectedly",
                        ],
                        timeout=3,
                    )
                except Exception:
                    pass

    def on_icon_click(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.toggle_dictation()

    def load_config(self):
        self.input_device = None
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    raw = json.load(f).get("input_device")
                    if isinstance(raw, int):
                        self.input_device = raw
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                pass

    def save_config(self):
        with open(self.config_path, "w") as f:
            json.dump({"input_device": self.input_device}, f)

    def reload_devices(self):
        self.device_menu.clear()
        try:
            devices = sd.query_devices()
        except OSError:
            no_devices = self.device_menu.addAction("No audio devices available")
            no_devices.setEnabled(False)
            return
        input_devices = [
            (idx, dev)
            for idx, dev in enumerate(devices)
            if dev["max_input_channels"] > 0
        ]
        if not input_devices:
            no_devices = self.device_menu.addAction("No input devices found")
            no_devices.setEnabled(False)
            return
        if hasattr(self, "_device_group") and self._device_group:
            self._device_group.deleteLater()
        self._device_group = QActionGroup(self.device_menu)
        for idx, dev in input_devices:
            sr = dev.get("default_samplerate")
            sr_str = f"{int(sr)}" if sr and sr > 0 else "??"
            action = self.device_menu.addAction(f"{idx}: {dev['name']} ({sr_str} Hz)")
            action.setCheckable(True)
            action.setChecked(idx == self.input_device)
            self._device_group.addAction(action)
            action.triggered.connect(lambda _, i=idx: self.set_input_device(i))
        self.device_menu.addSeparator()
        reload_action = self.device_menu.addAction("Reload Devices")
        reload_action.triggered.connect(self.reload_devices)

    def _get_input_device_info(self, device_idx):
        if not isinstance(device_idx, int):
            return None
        try:
            return sd.query_devices(device_idx, "input")
        except Exception:
            return None

    def _resolve_start_device(self):
        device_info = self._get_input_device_info(self.input_device)
        if device_info is not None:
            return self.input_device, device_info, None

        try:
            default_input, _ = sd.default.device
        except Exception:
            default_input = None
        if default_input is None:
            self.input_device = None
            self.save_config()
            return None, None, "Saved microphone is unavailable and no default input device is configured."

        default_info = self._get_input_device_info(default_input)
        if default_info is None:
            self.input_device = None
            self.save_config()
            return None, None, "No working input device is available."

        self.input_device = int(default_input)
        self.save_config()
        return self.input_device, default_info, "Saved microphone disappeared; switched to the default input."

    def set_input_device(self, device_idx):
        if not isinstance(device_idx, int):
            return
        device_info = self._get_input_device_info(device_idx)
        if device_info is None:
            subprocess.run(
                [
                    "notify-send",
                    "-t",
                    "3000",
                    "Dictation Tool",
                    f"Input device {device_idx} is unavailable",
                ],
                timeout=3,
            )
            return
        self.input_device = device_idx
        self.save_config()
        self.set_icon(self.is_dictating)
        name = device_info.get("name", f"device {device_idx}")
        subprocess.run(
            [
                "notify-send",
                "-t",
                "3000",
                "Dictation Tool",
                f"Input device set to: {name}",
            ],
            timeout=3,
        )

    def _send_worker_command(self, command):
        if not self.dictation_process or self.dictation_process.stdin is None:
            raise RuntimeError("Dictation worker is not running")
        self.dictation_process.stdin.write(f"{command}\n")
        self.dictation_process.stdin.flush()

    def _prewarm_worker(self):
        try:
            if self._log_file is None:
                self._log_file = self._open_log_file()
            self._write_worker_log(
                f"\n--- worker prewarm at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            )
            self._ensure_worker_process()
        except Exception as e:
            self._write_worker_log(f"Prewarm failed: {e}\n")

    def _ensure_worker_process(self):
        if self.dictation_process and self.dictation_process.poll() is None:
            return
        worker_script = os.path.join(self.script_dir, "whisper_dictate.py")
        env = os.environ.copy()
        for var in (
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XDG_RUNTIME_DIR",
            "XDG_CURRENT_DESKTOP",
        ):
            if not os.environ.get(var):
                env.pop(var, None)
        env["PYTHONUNBUFFERED"] = "1"
        if self._log_file is None:
            self._log_file = self._open_log_file()
        self._worker_event.clear()
        self._worker_boot_event.clear()
        self._worker_ready_event.clear()
        self._worker_booted = False
        self._worker_ready = False
        self.dictation_process = subprocess.Popen(
            [sys.executable, worker_script, "--controlled"],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._worker_log_relay = WorkerLogRelay(self, self.dictation_process)
        self._worker_log_relay.start()
        self._worker_monitor = WorkerMonitor(self, self.dictation_process)
        self._worker_monitor.start()

    def set_icon(self, active):
        icon_path = os.path.join(
            self.script_dir, "mic-on.png" if active else "mic-off.png"
        )
        self.tray_icon.setIcon(QIcon(icon_path))
        status = "ON" if active else "OFF"
        device_info = f" ({self.input_device})" if self.input_device is not None else ""
        self.tray_icon.setToolTip(f"Dictation: {status}{device_info}")

    def toggle_dictation(self):
        self.stop_dictation() if self.is_dictating else self.start_dictation()

    def start_dictation(self):
        if self.is_dictating:
            return
        try:
            device_idx, device_info, notice = self._resolve_start_device()
            if device_idx is None and device_info is None:
                subprocess.run(
                    [
                        "notify-send",
                        "-t",
                        "5000",
                        "Dictation Tool Error",
                        notice or "No working input device is available.",
                    ],
                    timeout=3,
                )
                return

            if self._log_file is None:
                self._log_file = self._open_log_file()
            self._write_worker_log(
                f"\n--- dictation started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            )
            if notice:
                self._write_worker_log(f"{notice}\n")
            if device_info is not None:
                self._write_worker_log(
                    f"Using input device {device_idx}: {device_info.get('name', 'unknown')}\n"
                )

            was_booted = self._worker_booted
            self._ensure_worker_process()
            self._worker_ready_event.clear()
            self._worker_ready = False
            command = f"start {device_idx}" if device_idx is not None else "start"
            self._send_worker_command(command)
            self.is_dictating = True
            self.set_icon(True)
            message = notice or ("Starting dictation..." if was_booted else "Warming up dictation...")
            subprocess.run(
                ["notify-send", "-t", "3000", "Dictation Tool", message],
                timeout=3,
            )
        except Exception as e:
            subprocess.run(
                [
                    "notify-send",
                    "-t",
                    "5000",
                    "Dictation Tool Error",
                    f"Failed to start: {str(e)}",
                ],
                timeout=3,
            )

    def stop_dictation(self, during_cleanup=False):
        if not self.is_dictating:
            return
        self.is_dictating = False
        self._worker_ready = False
        self._worker_ready_event.clear()
        if self.dictation_process and self.dictation_process.poll() is None:
            try:
                self._send_worker_command("stop")
            except Exception:
                pass
        self.set_icon(False)
        if not during_cleanup and not self._shutting_down:
            try:
                subprocess.run(
                    [
                        "notify-send",
                        "-t",
                        "3000",
                        "Dictation Tool",
                        "Dictation stopped",
                    ],
                    timeout=3,
                )
            except Exception:
                pass

    def _open_log_file(self):
        log_path = os.path.join(self.script_dir, "dictation.log")
        # Rotate on open: if file > 512KB, shift .1 → .2, .2 → .3, current → .1
        try:
            if os.path.getsize(log_path) > 512 * 1024:
                for i in range(2, 0, -1):
                    src = f"{log_path}.{i}"
                    dst = f"{log_path}.{i + 1}"
                    if os.path.exists(src):
                        os.rename(src, dst)
                os.rename(log_path, f"{log_path}.1")
        except OSError:
            pass
        return open(log_path, "a")

    def _write_worker_log(self, text):
        if not text:
            return
        with self._log_lock:
            if not self._log_file:
                return
            self._log_file.write(text)
            self._log_file.flush()

    def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        self._shutting_down = True
        self._sig_timer.stop()
        self._worker_timer.stop()
        if self._notifier:
            try:
                self._notifier.activated.disconnect()
            except Exception:
                pass
            self._notifier.setEnabled(False)
        if self._socket:
            self._socket.close()
        if self._socket_path and os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass
        self.stop_dictation(during_cleanup=True)
        if self.dictation_process and self.dictation_process.poll() is None:
            self._intentional_stop = True
            try:
                self._send_worker_command("quit")
                self.dictation_process.wait(timeout=2.0)
            except Exception:
                try:
                    self.dictation_process.kill()
                    self.dictation_process.wait(timeout=2.0)
                except Exception:
                    pass
        self.dictation_process = None
        if self._worker_monitor:
            self._worker_monitor._stop.set()
            self._worker_monitor = None
        self._worker_log_relay = None
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def quit_app(self):
        self.cleanup()
        self.app.quit()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    app = DictationTrayApp()
    app.run()
