import logging
import os
import subprocess
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
        if exit_code != 0:
            log.warning("Dictation worker exited abnormally (code %d)", exit_code)
        self.tray_app._worker_event.set()


class DictationTrayApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.tray_icon = QSystemTrayIcon()

        atexit.register(self.cleanup)
        self.app.aboutToQuit.connect(self.cleanup)

        # SIGINT/SIGTERM via QTimer (avoids Qt issues with C signal handlers)
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
        self._worker_event = threading.Event()
        self._log_file = None
        self.is_dictating = False
        self._cleaned = False
        self._shutting_down = False

        # Worker crash sync: main-thread QTimer polls the threading.Event
        self._worker_timer = QTimer()
        self._worker_timer.timeout.connect(self._check_worker)
        self._worker_timer.start(500)

        # Unix socket for hotkey toggle (no root needed)
        self._socket_path = os.path.join(self.script_dir, ".dictation.sock")
        self._socket = None
        self._notifier = None
        self._start_socket_listener()

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

    def _start_socket_listener(self):
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
                conn.close()
            if data == "toggle":
                self.toggle_dictation()
        except OSError:
            pass

    def _check_signals(self):
        if self._sig_flag:
            self._sig_flag = False
            self.cleanup()
            self.app.quit()

    def _check_worker(self):
        if self._worker_event.is_set():
            self._worker_event.clear()
            if self.is_dictating:
                self.is_dictating = False
                self.set_icon(False)
                try:
                    subprocess.run(["notify-send", "Dictation Tool", "Dictation stopped unexpectedly"], timeout=3)
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
                    self.input_device = json.load(f).get("input_device")
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
            (idx, dev) for idx, dev in enumerate(devices)
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
            action = self.device_menu.addAction(
                f"{idx}: {dev['name']} ({int(dev['default_samplerate'])} Hz)"
            )
            action.setCheckable(True)
            action.setChecked(idx == self.input_device)
            self._device_group.addAction(action)
            action.triggered.connect(lambda _, i=idx: self.set_input_device(i))

        self.device_menu.addSeparator()
        reload_action = self.device_menu.addAction("Reload Devices")
        reload_action.triggered.connect(self.reload_devices)

    def set_input_device(self, device_idx):
        self.input_device = device_idx
        self.save_config()
        self.set_icon(self.is_dictating)
        try:
            name = sd.query_devices(device_idx, "input").get("name", f"device {device_idx}")
            subprocess.run(["notify-send", "Dictation Tool", f"Input device set to: {name}"])
        except OSError:
            subprocess.run(["notify-send", "Dictation Tool", f"Input device set to index {device_idx}"])

    def set_icon(self, active):
        icon_path = os.path.join(self.script_dir, "mic-on.png" if active else "mic-off.png")
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
            worker_script = os.path.join(self.script_dir, "whisper_dictate.py")
            cmd = [sys.executable, worker_script]
            if self.input_device is not None:
                cmd.append(str(self.input_device))

            env = os.environ.copy()
            for var in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XDG_CURRENT_DESKTOP"):
                if not os.environ.get(var):
                    env.pop(var, None)

            self._log_file = self._open_log_file()
            self._log_file.write(f"\n--- dictation started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            self._log_file.flush()

            self._worker_event.clear()
            self.dictation_process = subprocess.Popen(cmd, env=env, stdout=self._log_file, stderr=subprocess.STDOUT)
            self.is_dictating = True
            self.set_icon(True)
            subprocess.run(["notify-send", "Dictation Tool", "Dictation started"])

            self._worker_monitor = WorkerMonitor(self, self.dictation_process)
            self._worker_monitor.start()
        except Exception as e:
            subprocess.run(["notify-send", "Dictation Tool Error", f"Failed to start: {str(e)}"])

    def stop_dictation(self, during_cleanup=False):
        if not self.is_dictating:
            return
        self._worker_event.clear()

        if self.dictation_process:
            self.dictation_process.terminate()
            try:
                self.dictation_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.dictation_process.kill()
                self.dictation_process.wait(timeout=2.0)
            self.dictation_process = None

        if self._worker_monitor:
            self._worker_monitor._stop.set()
            self._worker_monitor = None

        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        self.set_icon(False)
        if not during_cleanup and not self._shutting_down:
            try:
                subprocess.run(["notify-send", "Dictation Tool", "Dictation stopped"], timeout=3)
            except Exception:
                pass

    def _open_log_file(self):
        log_path = os.path.join(self.script_dir, "dictation.log")
        handler = RotatingFileHandler(log_path, maxBytes=512 * 1024, backupCount=3)
        handler.stream = handler._open()
        return handler.stream

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

    def quit_app(self):
        self.cleanup()
        self.app.quit()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    bootstrap()
    app = DictationTrayApp()
    app.run()
