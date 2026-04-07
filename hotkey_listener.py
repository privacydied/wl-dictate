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
        self.hotkey_combination = {
            evdev.ecodes.KEY_LEFTCTRL, evdev.ecodes.KEY_LEFTALT, evdev.ecodes.KEY_F
        }
        self._devices_lock = threading.Lock()

    def try_detect_keyboards(self):
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        self.keyboard_devices = []
        for device in devices:
            caps = device.capabilities()
            # Check if device has keys and specifically has KEY_A (as a proxy for keyboard)
            if evdev.ecodes.EV_KEY in caps:
                keys = caps[evdev.ecodes.EV_KEY]
                if evdev.ecodes.KEY_A in keys:
                    self.keyboard_devices.append(device)
        print(f"🔑 Found {len(self.keyboard_devices)} keyboard devices")

    def run(self):
        self.try_detect_keyboards()
        while self._running.is_set():
            try:
                # Reconnect to keyboards if needed
                if not self.keyboard_devices:
                    self.try_detect_keyboards()
                    self._running.wait(1.0)
                    continue

                # Select on keyboard devices
                with self._devices_lock:
                    devices_to_watch = list(self.keyboard_devices)
                r, w, x = select.select(devices_to_watch, [], [], 0.1)
                for device in r:
                    try:
                        for event in device.read():
                            if event.type == evdev.ecodes.EV_KEY:
                                key_event = evdev.categorize(event)
                                scancode = event.code
                                if event.value == 1:  # Key down
                                    self.pressed_keys.add(scancode)
                                elif event.value == 0:  # Key up
                                    self.pressed_keys.discard(scancode)
                                    # Check hotkey on key release
                                    if self.pressed_keys == self.hotkey_combination and scancode == evdev.ecodes.KEY_D:
                                        self.callback()
                    except (OSError, IOError):
                        # Device disconnected mid-read, reset
                        with self._devices_lock:
                            self.keyboard_devices = []
            except (OSError, IOError):
                # Device disconnected, reset
                with self._devices_lock:
                    self.keyboard_devices = []

    def stop(self):
        self._running.clear()

# Example usage
if __name__ == "__main__":
    def test_callback():
        print("🔑 Hotkey triggered: Ctrl+Alt+D")
    
    listener = HotkeyListener(test_callback)
    listener.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop()
