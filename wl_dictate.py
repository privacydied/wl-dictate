"""Unified entry point for wl-dictate.

Usage (source):
    python wl_dictate.py              # tray app
    python wl_dictate.py --worker     # whisper worker (called by tray)
    python wl_dictate.py --toggle     # toggle dictation
    python wl_dictate.py --devices    # list input devices

Usage (compiled binary):
    wl-dictate                        # tray app
    wl-dictate --worker               # whisper worker
    wl-dictate --toggle               # toggle dictation
    wl-dictate --devices              # list input devices
"""

import sys


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "--worker":
        # Whisper dictation worker — controlled mode
        # Remaining args: --worker [device_idx]
        import whisper_dictate

        whisper_dictate.bootstrap()
        from threading import Thread

        worker = Thread(target=whisper_dictate._transcribe_worker_loop, daemon=True)
        worker.start()
        whisper_dictate.controlled_main()

    elif args and args[0] == "--toggle":
        # Send toggle signal over Unix socket
        import toggle_dictation

        toggle_dictation.main()

    elif args and args[0] == "--devices":
        # List available input devices
        import sounddevice as sd

        print("Available input devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                sr = dev.get("default_samplerate")
                sr_str = f"{int(sr)}" if sr and sr > 0 else "??"
                print(f"  [{i}] {dev['name']} -- {sr_str} Hz")

    else:
        # Default: launch tray app
        from tray_app import DictationTrayApp

        app = DictationTrayApp()
        sys.exit(app.app.exec_())


if __name__ == "__main__":
    main()
