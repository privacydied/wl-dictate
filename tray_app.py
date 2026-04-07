import os
import sys
import subprocess
import signal
import json
import atexit
import time
import socket
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QActionGroup, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSocketNotifier
from whisper_dictate import bootstrap

class DictationTrayApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.tray_icon = QSystemTrayIcon()
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        # Connect aboutToQuit
        self.app.aboutToQuit.connect(self.cleanup)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
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
        self.is_dictating = False
        self._cleaned = False

        # Unix socket listener for hotkey toggle (no root needed)
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
            # QSocketNotifier fires on the main thread
            self._notifier = QSocketNotifier(self._socket.fileno(), QSocketNotifier.Read, self.app)
            self._notifier.activated.connect(self._socket_ready)
        except Exception as e:
            print(f"⚠️ Could not start socket listener: {e}")

    def _socket_ready(self):
        if not self._socket:
            return
        try:
            conn, _ = self._socket.accept()
            data = conn.recv(64).decode().strip()
            conn.close()
            if data == "toggle":
                self.toggle_dictation()
        except OSError:
            pass

    def signal_handler(self, signum, frame):
        self.cleanup()
        sys.exit(0)

    def on_icon_click(self, reason):
        if reason == QSystemTrayIcon.Trigger:  # Left click
            self.toggle_dictation()

    def load_config(self):
        self.input_device = None
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.input_device = config.get('input_device')
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                pass

    def save_config(self):
        config = {'input_device': self.input_device}
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

    def reload_devices(self):
        """Reload available input devices."""
        # Clear the current device menu
        self.device_menu.clear()
        
        # Query all devices and filter for input
        devices = sd.query_devices()
        input_devices = []
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((idx, device))
        
        # If no input devices found, show a disabled action
        if not input_devices:
            no_devices = self.device_menu.addAction("No input devices found")
            no_devices.setEnabled(False)
            return
        
        # Add actions for each input device
        self.device_group = QActionGroup(self.device_menu)
        for idx, device in input_devices:
            device_name = device['name']
            sample_rate = int(device['default_samplerate'])
            action = self.device_menu.addAction(f"{idx}: {device_name} ({sample_rate} Hz)")
            action.setCheckable(True)
            action.setChecked(idx == self.input_device)
            self.device_group.addAction(action)
            action.triggered.connect(lambda checked, idx=idx: self.set_input_device(idx))
        
        # Add reload action
        self.device_menu.addSeparator()
        reload_action = self.device_menu.addAction("Reload Devices")
        reload_action.triggered.connect(self.reload_devices)

    def set_input_device(self, device_idx):
        self.input_device = device_idx
        self.save_config()
        self.set_icon(self.is_dictating)
        
        # Notify user
        device_info = sd.query_devices(device_idx, 'input')
        device_name = device_info['name']
        subprocess.run(["notify-send", "Dictation Tool", f"Input device set to: {device_name}"])

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

    def start_dictation(self):
        if self.is_dictating:
            return
        try:
            worker_script = os.path.join(self.script_dir, "whisper_dictate.py")
            cmd = [sys.executable, worker_script]
            if self.input_device is not None:
                cmd.append(str(self.input_device))
            # Forward display variables so wtype can reach Wayland
            env = os.environ.copy()
            for var in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XDG_CURRENT_DESKTOP"):
                if not os.environ.get(var):
                    env.pop(var, None)
            # Log worker output for debugging
            log_path = os.path.join(self.script_dir, "dictation.log")
            with open(log_path, "a") as log_f:
                log_f.write(f"\n--- dictation started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                self.dictation_process = subprocess.Popen(
                    cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT
                )
            self.is_dictating = True
            self.set_icon(True)
            subprocess.run(["notify-send", "Dictation Tool", "Dictation started"])
        except Exception as e:
            subprocess.run(["notify-send", "Dictation Tool Error", f"Failed to start: {str(e)}"])

    def stop_dictation(self):
        if self.is_dictating:
            if self.dictation_process:
                self.dictation_process.terminate()
                try:
                    self.dictation_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.dictation_process.kill()
                self.dictation_process = None
            self.is_dictating = False
            self.set_icon(False)
            subprocess.run(["notify-send", "Dictation Tool", "Dictation stopped"])

    def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
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
        self.stop_dictation()
        if hasattr(self, 'dictation_process') and self.dictation_process:
            self.dictation_process.terminate()
            try:
                self.dictation_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.dictation_process.kill()
            self.dictation_process = None
        
    def quit_app(self):
        self.cleanup()
        self.app.quit()

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    bootstrap()
    app = DictationTrayApp()
    app.run()
