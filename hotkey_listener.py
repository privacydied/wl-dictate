import evdev
import select
import threading
import time
from queue import Queue


class HotkeyListener(threading.Thread):
    def __init__(self, callback):
        super().__init__(daemon=True)
        self.callback = callback
        self._running = threading.Event()
        self._running.set()
        self.keyboard_devices = []
        self.pressed_keys = set()
        # Accept both left and right modifiers for Ctrl+Alt+D [H2]
        self.hotkey_combination = {
            evdev.ecodes.KEY_LEFTCTRL,
            evdev.ecodes.KEY_RIGHTCTRL,
            evdev.ecodes.KEY_LEFTALT,
            evdev.ecodes.KEY_RIGHTALT,
            evdev.ecodes.KEY_D,
        }
        self._devices_lock = threading.Lock()
        self._open_devices = []  # track open handles for cleanup [H3]

    def try_detect_keyboards(self):
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        except OSError:
            return
        self.keyboard_devices = []
        self._open_devices = []  # reset on rescan
        for device in devices:
            caps = device.capabilities()
            if evdev.ecodes.EV_KEY in caps:
                keys = caps[evdev.ecodes.EV_KEY]
                if evdev.ecodes.KEY_A in keys:
                    self.keyboard_devices.append(device)
                    self._open_devices.append(device)
        print(f"Found {len(self.keyboard_devices)} keyboard devices")

    def _close_device_handles(self):
        """Close all open evdev handles to prevent FD leaks [H3]."""
        with self._devices_lock:
            for dev in self._open_devices:
                try:
                    dev.close()
                except Exception:
                    pass
            self._open_devices.clear()
            self.keyboard_devices.clear()

    def run(self):
        self.try_detect_keyboards()
        while self._running.is_set():
            try:
                if not self.keyboard_devices:
                    self.try_detect_keyboards()
                    self._running.wait(1.0)
                    continue

                with self._devices_lock:
                    devices_to_watch = list(self.keyboard_devices)
                    all_modifiers = {
                        evdev.ecodes.KEY_LEFTCTRL,
                        evdev.ecodes.KEY_RIGHTCTRL,
                        evdev.ecodes.KEY_LEFTALT,
                        evdev.ecodes.KEY_RIGHTALT,
                    }
                r, w, x = select.select(devices_to_watch, [], [], 0.1)
                for device in r:
                    try:
                        for event in device.read():
                            if event.type == evdev.ecodes.EV_KEY:
                                scancode = event.code
                                with self._devices_lock:
                                    if event.value == 1:  # Key down
                                        self.pressed_keys.add(scancode)
                                    elif event.value == 0:  # Key up
                                        self.pressed_keys.discard(scancode)
                                        # Check hotkey on release of the trigger key [H1]
                                        if self.pressed_keys <= all_modifiers and scancode == evdev.ecodes.KEY_D:
                                            self.callback()
                    except (OSError, IOError):
                        with self._devices_lock:
                            self.keyboard_devices = []
                            self._open_devices = []
            except (OSError, IOError):
                with self._devices_lock:
                    self.keyboard_devices = []
                    self._open_devices = []

    def stop(self):
        self._running.clear()
        self._close_device_handles()


if __name__ == "__main__":

    def test_callback():
        print("Hotkey triggered: Ctrl+Alt+D")

    listener = HotkeyListener(test_callback)
    listener.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop()
