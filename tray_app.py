import logging
import os
import subprocess
import signal
import sys
import json
import atexit
import time
import socket
import threading
from logging.handlers import RotatingFileHandler
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QActionGroup, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSocketNotifier, QTimer
from whisper_dictate import bootstrap

log = logging.getLogger(__name__)


class WorkerMonitor(threading.Thread):
    """Watches the dictation worker subprocess so a crash syncs is_dictating [T3]."""

    def __init__(self, tray_app, process):
        super().__init__(daemon=True)
        self.tray_app = tray_app
        self.process = process
        self._stop = threading.Event()

    def run(self):
        exit_code = self.process.wait()
        if self._stop.is_set():
            return
        # Process died while we thought it was running
        if exit_code != 0:
            log.warning("Dictation worker exited abnormally (code %d)", exit_code)
        # Sync state from worker thread — use QMetaObject for thread safety
        if self.tray_app.is_dictating:
            self.tray_app.is_dictating = False
            self.tray_app.set_icon(False)  # safe: Qt icon ops are thread-safe enough for simple state
            try:
                subprocess.run(
                    ["notify-send", "Dictation Tool", "Dictation stopped unexpectedly"],
                    timeout=3,
                )
            except Exception:
                pass


class DictationTrayApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.tray_icon = QSystemTrayIcon()

        # Register cleanup
        atexit.register(self.cleanup)

        # Connect aboutToQuit
        self.app.aboutToQuit.connect(self.cleanup)

        # Set up signal handlers via QTimer so they run on the Qt main thread
        import signal as _signal
        self._sig_flag = False

        def _catch_signal(*_args):
            self._sig_flag = True

        _signal.signal(_signal.SIGINT, _catch_signal)
        _signal.signal(_signal.SIGTERM, _catch_signal)

        self._sig_timer = QTimer()
        self._sig_timer.timeout.connect(self._check_signal)
        self._sig_timer.start(250)  # check every 250ms

        # Get script directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Configuration
        self.config_path = os.path.join(self.script_dir, "config.json")
        self.load_config()

        self.set_icon(False)
        self.tray_icon.show()

        # Create context menu
        self.menu = QMenu()

        # Input device submenu
        self.device_menu = self.menu.addMenu("Input Device")
        self.reload_devices()

        self.toggle_action = self.menu.addAction("Toggle Dictation")
        self.toggle_action.triggered.connect(self.toggle_dictation)

        self.reload_action = self.menu.addAction("Reload Devices")
        self.reload_action.triggered.connect(self.reload_devices)

        self.quit_action = self.menu.addAction("Quit")
        self.quit_action.triggered.connect(self.quit_app)

        self.tray_icon.setContextMenu(self.menu)

        # Connect icon click
        self.tray_icon.activated.connect(self.on_icon_click)

        self.dictation_process = None
        self._worker_monitor = None
        self.is_dictating = False
        self._cleaned = False
        self._shutting_down = False  # [T4] distinguish quit from normal stop

        # Unix socket listener for hotkey toggle
        self._socket_path = os.path.join(self.script_dir, ".dictation.sock")
        self._socket = None
        self._notifier = None
        self._start_socket_listener()

    def _start_socket_listener(self):
        """Listen on a Unix socket so a toggle script can control dictation."""
        try:
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.bind(self._socket_path)
            self._socket.listen(5)
            self._socket.setblocking(False)
            self._notifier = QSocketNotifier(self._socket.fileno(), QSocketNotifier.Read)
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
                conn.close()  # [T1] always close the connection
            if data == "toggle":
                self.toggle_dictation()
        except OSError:
            pass

    def _check_signal(self):
        """Called by QTimer on the main thread — handles SIGINT/SIGTERM."""
        if self._sig_flag:
            self._sig_flag = False
            self.cleanup()
            self.app.quit()

    def on_icon_click(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.toggle_dictation()

    def load_config(self):
        self.input_device = None
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    self.input_device = config.get("input_device")
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                pass

    def save_config(self):
        config = {"input_device": self.input_device}
        with open(self.config_path, "w") as f:
            json.dump(config, f)

    def reload_devices(self):
        """Reload available input devices."""
        self.device_menu.clear()

        try:
            devices = sd.query_devices()
        except OSError:
            no_devices = self.device_menu.addAction("No audio devices available")
            no_devices.setEnabled(False)
            return

        input_devices = []
        for idx, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append((idx, device))

        if not input_devices:
            no_devices = self.device_menu.addAction("No input devices found")
            no_devices.setEnabled(False)
            return

        # Replace previous group to avoid orphaned QActionGroup objects [T5]
        if hasattr(self, "_device_group") and self._device_group:
            self._device_group.deleteLater()
        self.device_group = QActionGroup(self.device_menu)
        for idx, device in input_devices:
            device_name = device["name"]
            sample_rate = int(device["default_samplerate"])
            action = self.device_menu.addAction(f"{idx}: {device_name} ({sample_rate} Hz)")
            action.setCheckable(True)
            action.setChecked(idx == self.input_device)
            self.device_group.addAction(action)
            action.triggered.connect(lambda checked, idx=idx: self.set_input_device(idx))

        self.device_menu.addSeparator()
        reload_action = self.device_menu.addAction("Reload Devices")
        reload_action.triggered.connect(self.reload_devices)

    def set_input_device(self, device_idx):
        self.input_device = device_idx
        self.save_config()
        self.set_icon(self.is_dictating)

        try:
            device_info = sd.query_devices(device_idx, "input")
            device_name = device_info.get("name", f"device {device_idx}")  # [T7]
            subprocess.run(["notify-send", "Dictation Tool", f"Input device set to: {device_name}"])
        except OSError:
            subprocess.run(["notify-send", "Dictation Tool", f"Input device set to index {device_idx}"])

    def set_icon(self, active):
        icon_name = "mic-on.png" if active else "mic-off.png"
        icon_path = os.path.join(self.script_dir, icon_name)
        self.tray_icon.setIcon(QIcon(icon_path))

        status = "ON" if active else "OFF"
        device_info = f" ({self.input_device})" if self.input_device is not None else ""
        self.tray_icon.setToolTip(f"Dictation: {status}{device_info}")

    def toggle_dictation(self):
        if self.is_dictating:
            self.stop_dictation()
        else:
            self.start_dictation()

    def _open_log_file(self):
        """Open a rotating log file for worker output [T6]."""
        log_path = os.path.join(self.script_dir, "dictation.log")
        handler = RotatingFileHandler(log_path, maxBytes=512 * 1024, backupCount=3)
        handler.stream = handler._open()
        return handler.stream

    def start_dictation(self):
        if self.is_dictating:
            return
        try:
            worker_script = os.path.join(self.script_dir, "whisper_dictate.py")
            cmd = [sys.executable, worker_script]
            if self.input_device is not None:
                cmd.append(str(self.input_device))
            env = os.environ.copy()
            for var in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XDG_CURRENT_DESKTOP"):
                if not os.environ.get(var):
                    env.pop(var, None)

            log_f = self._open_log_file()
            log_f.write(f"\n--- dictation started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_f.flush()

            self.dictation_process = subprocess.Popen(
                cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT
            )
            self.is_dictating = True
            self.set_icon(True)
            subprocess.run(["notify-send", "Dictation Tool", "Dictation started"])

            # Start monitoring thread [T3]
            self._worker_monitor = WorkerMonitor(self, self.dictation_process)
            self._worker_monitor.start()
        except Exception as e:
            subprocess.run(["notify-send", "Dictation Tool Error", f"Failed to start: {str(e)}"])

    def stop_dictation(self, during_cleanup=False):
        if self.is_dictating:
            if self.dictation_process:
                self.dictation_process.terminate()
                try:
                    self.dictation_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.dictation_process.kill()
                    self.dictation_process.wait(timeout=2.0)
                self.dictation_process = None
            self.is_dictating = False
            self.set_icon(False)
            # [T2] avoid calling stop_dictation recursively from cleanup
            if not during_cleanup and not self._shutting_down:
                subprocess.run(["notify-send", "Dictation Tool", "Dictation stopped"])

    def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        self._shutting_down = True

        # Clean up socket
        if self._notifier:
            self._notifier.activated.disconnect()
            self._notifier.setEnabled(False)
        if self._socket:
            self._socket.close()
        sock_path = self._socket_path
        if sock_path and os.path.exists(sock_path):
            try:
                os.unlink(sock_path)
            except OSError:
                pass

        # Stop worker once — don't call stop_dictation which is no-op-safe [T2]
        if self._worker_monitor:
            self._worker_monitor._stop.set()
            self._worker_monitor = None
        if self.dictation_process:
            self.dictation_process.terminate()
            try:
                self.dictation_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.dictation_process.kill()
                self.dictation_process.wait(timeout=2.0)
            self.dictation_process = None
        self.is_dictating = False
        self.set_icon(False)

    def quit_app(self):
        self.cleanup()
        self.app.quit()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    bootstrap()
    app = DictationTrayApp()
    app.run()
