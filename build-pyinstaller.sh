#!/usr/bin/env bash
# Build wl-dictate into a single binary using PyInstaller.
#
# Much faster to build than Nuitka (bundles bytecode + shared libs instead of
# compiling everything to C), and fast enough at startup for a long-lived tray
# app.
#
# IMPORTANT: this builds from the project's .venv (Python 3.13, versions pinned
# in uv.lock) — NOT the system Python. That matters: the system onnxruntime is a
# ~520 MB CUDA build, while the .venv one is a ~50 MB CPU build. Bundling the
# venv is smaller, faster to build, and reproducible.
set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
PY="$VENV/bin/python"

# --- Preflight -------------------------------------------------------------
if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found. Create the venv first:" >&2
    echo "         uv sync" >&2
    exit 1
fi

if ! "$PY" -c "import PyInstaller" 2>/dev/null; then
    echo "==> PyInstaller not in $VENV; installing it..."
    uv pip install --python "$PY" pyinstaller
fi

echo "==> Building from: $("$PY" --version) ($PY)"

# PyInstaller's Qt hook can silently fail to collect Qt's plugin tree (seen on
# Python 3.14: libs + translations bundled, plugins/ missing -> "Could not find
# the Qt platform plugin ... in ''" at startup). Bundle it explicitly.
QT_PLUGINS="$("$PY" -c "import os, PyQt5; print(os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins'))")"
if [[ ! -d "$QT_PLUGINS/platforms" ]]; then
    echo "ERROR: Qt platform plugins not found at $QT_PLUGINS" >&2
    exit 1
fi

echo "==> Cleaning previous build..."
rm -rf build/ dist/ wl-dictate.spec

echo "==> Building with PyInstaller..."
# Run PyInstaller as a module of the venv interpreter so it bundles the venv's
# site-packages (the pinned, CPU-only deps), not whatever is on PATH.
"$PY" -m PyInstaller \
    --onefile \
    --noconfirm \
    --name wl-dictate \
    --add-data "mic-on.png:." \
    --add-data "mic-off.png:." \
    --add-data "$QT_PLUGINS:PyQt5/Qt5/plugins" \
    --collect-all faster_whisper \
    --collect-all ctranslate2 \
    --collect-all onnxruntime \
    --collect-submodules tokenizers \
    --collect-submodules huggingface_hub \
    --collect-data sounddevice \
    --hidden-import scipy.signal \
    --exclude-module torch \
    --exclude-module cv2 \
    --exclude-module PIL \
    --exclude-module matplotlib \
    --exclude-module PyQt6 \
    --exclude-module pytest \
    --exclude-module tkinter \
    wl_dictate.py

echo ""
echo "==> Build complete!"
ls -lh dist/wl-dictate
echo ""
echo "Install with:"
echo "  sudo cp dist/wl-dictate /usr/local/bin/"
echo "  # or just run: ./dist/wl-dictate"
