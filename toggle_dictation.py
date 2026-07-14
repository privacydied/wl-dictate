#!/usr/bin/env python3
"""Compat shim: toggle dictation via the tray app's Unix socket.

Kept because existing compositor keybindings point here.
Equivalent to: python wl_dictate.py --toggle
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wldictate.toggle import main

if __name__ == "__main__":
    sys.exit(main())
