#!/usr/bin/env bash
# Build wl-dictate into a single binary using Nuitka.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Cleaning previous build..."
rm -rf build/ wl-dictate.dist/ wl-dictate.build/ wl-dictate.onefile-build/ dist/

echo "==> Building with Nuitka (this will take a while)..."
python3 -m nuitka \
    --onefile \
    --output-filename=wl-dictate \
    --output-dir=dist \
    --include-data-file=mic-on.png=mic-on.png \
    --include-data-file=mic-off.png=mic-off.png \
    --include-data-file=config.json=config.json \
    --enable-plugin=pyqt5 \
    --include-module=faster_whisper \
    --include-module=ctranslate2 \
    --include-module=tokenizers \
    --include-module=huggingface_hub \
    --include-module=sounddevice \
    --include-module=numpy \
    --include-module=evdev \
    --include-package=faster_whisper \
    --include-package-data=faster_whisper \
    --include-package-data=ctranslate2 \
    --assume-yes-for-downloads \
    wl_dictate.py

echo ""
echo "==> Build complete!"
ls -lh dist/wl-dictate
echo ""
echo "Install with:"
echo "  sudo cp dist/wl-dictate /usr/local/bin/"
echo "  # or just run: ./dist/wl-dictate"
