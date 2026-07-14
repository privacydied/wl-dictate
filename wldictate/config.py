"""Configuration: XDG-located config.json with validation and legacy migration.

The config lives at ``$XDG_CONFIG_HOME/wl-dictate/config.json`` (default
``~/.config/wl-dictate/config.json``).  On first run, a legacy ``config.json``
sitting next to the binary or the source tree is migrated automatically.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

APP_NAME = "wl-dictate"


# ── Paths ────────────────────────────────────────────────────────────────────


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / APP_NAME


def config_path() -> Path:
    return config_dir() / "config.json"


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / APP_NAME


def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    if base and os.path.isdir(base):
        return Path(base)
    # Fallback for odd sessions: something user-private and writable.
    return Path(f"/tmp/{APP_NAME}-{os.getuid()}")


def socket_path() -> Path:
    return runtime_dir() / f"{APP_NAME}.sock"


def legacy_socket_paths() -> list[Path]:
    """Old socket locations (next to binary / script) kept for compat."""
    paths = []
    for d in (
        os.path.dirname(os.path.abspath(sys.executable)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ):
        paths.append(Path(d) / ".dictation.sock")
    return paths


def _legacy_config_paths() -> list[Path]:
    return [
        Path(os.path.dirname(os.path.abspath(sys.executable))) / "config.json",
        Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "config.json",
    ]


# ── Schema ───────────────────────────────────────────────────────────────────


@dataclass
class StreamingConfig:
    enabled: bool = True
    infer_interval_s: float = 0.5
    min_new_audio_s: float = 0.3
    max_buffer_s: float = 12.0


@dataclass
class VadConfig:
    backend: str = "auto"  # auto | silero | energy
    onset: float = 0.5
    offset: float = 0.35
    onset_frames: int = 2
    min_silence_ms: int = 500
    pre_roll_ms: int = 320
    min_speech_s: float = 0.3
    max_utterance_s: float = 28.0


@dataclass
class TypingConfig:
    mode: str = "commit"  # commit (append-only); future: correcting
    wtype_timeout_s: float = 10.0
    # Per-keystroke delay (ms) passed to `wtype -d`. Electron/Chromium apps
    # (Vesktop, Discord, VSCode, Slack) drop characters — usually spaces and
    # punctuation — when keystrokes arrive too fast; a small delay fixes it.
    # 0 = no delay (fastest; fine for native GTK/Qt fields).
    wtype_delay_ms: int = 6
    # Type a single trailing space after sentence-final punctuation when an
    # utterance ends, so manual typing can continue naturally.
    sentence_trailing_space: bool = True
    # Capitalize the first letter of each sentence (utterance start and after
    # . ! ?). Whisper often lowercases the word after a pause.
    capitalize_sentences: bool = True


@dataclass
class AudioConfig:
    # Keep the microphone stream open across dictation toggles. Opening and
    # closing a USB mic renegotiates isochronous bandwidth on its USB
    # controller, which audibly disrupts *other* audio devices on the same
    # controller. Persistent capture negotiates once and never again (frames
    # are discarded while dictation is off). Set false to fully release the
    # mic whenever dictation is off.
    persistent_capture: bool = True


@dataclass
class Config:
    model: str = "small.en"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "auto"  # auto -> float16 on cuda, int8 on cpu
    # Device indices shift as PulseAudio/PipeWire streams appear/disappear, so
    # the name is authoritative; the index is a hint from the last resolution.
    input_device: int | None = None
    input_device_name: str | None = None
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    typing: TypingConfig = field(default_factory=TypingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    # Populated by load(); not serialized.
    warnings: list[str] = field(default_factory=list, repr=False)

    # ── (De)serialization ────────────────────────────────────────────────

    _NESTED = {
        "streaming": StreamingConfig,
        "vad": VadConfig,
        "typing": TypingConfig,
        "audio": AudioConfig,
    }

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "input_device": self.input_device,
            "input_device_name": self.input_device_name,
        }
        for name in self._NESTED:
            out[name] = dict(vars(getattr(self, name)))
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls()
        if not isinstance(raw, dict):
            cfg.warnings.append("config root is not an object; using defaults")
            return cfg

        def assign(obj: Any, key: str, value: Any, path: str) -> None:
            spec = {f.name: f for f in fields(obj)}
            if key not in spec or key == "warnings":
                cfg.warnings.append(f"unknown config key ignored: {path}")
                return
            current = getattr(obj, key)
            # Optional fields
            if key == "input_device":
                if value is None or isinstance(value, int):
                    obj.input_device = value
                else:
                    cfg.warnings.append(f"invalid value for {path}; keeping default")
                return
            if key == "input_device_name":
                if value is None or isinstance(value, str):
                    obj.input_device_name = value
                else:
                    cfg.warnings.append(f"invalid value for {path}; keeping default")
                return
            if isinstance(current, bool):
                ok = isinstance(value, bool)
            elif isinstance(current, int):
                ok = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(current, float):
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
                value = float(value) if ok else value
            elif isinstance(current, str):
                ok = isinstance(value, str)
            else:
                ok = False
            if ok:
                setattr(obj, key, value)
            else:
                cfg.warnings.append(f"invalid value for {path}; keeping default")

        for key, value in raw.items():
            if key in cls._NESTED:
                if not isinstance(value, dict):
                    cfg.warnings.append(f"invalid section {key}; keeping defaults")
                    continue
                section = getattr(cfg, key)
                for sub_key, sub_value in value.items():
                    assign(section, sub_key, sub_value, f"{key}.{sub_key}")
            else:
                assign(cfg, key, value, key)

        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if self.device not in ("auto", "cuda", "cpu"):
            self.warnings.append(f"invalid device '{self.device}'; using 'auto'")
            self.device = "auto"
        if self.compute_type not in ("auto", "float16", "float32", "int8", "int8_float16"):
            self.warnings.append(
                f"invalid compute_type '{self.compute_type}'; using 'auto'"
            )
            self.compute_type = "auto"
        if self.typing.mode not in ("commit",):
            self.warnings.append(f"unsupported typing.mode '{self.typing.mode}'; using 'commit'")
            self.typing.mode = "commit"
        if self.vad.backend not in ("auto", "silero", "energy"):
            self.warnings.append(f"invalid vad.backend '{self.vad.backend}'; using 'auto'")
            self.vad.backend = "auto"
        s = self.streaming
        if not (0.1 <= s.infer_interval_s <= 5.0):
            self.warnings.append("streaming.infer_interval_s out of range [0.1, 5]; using 0.5")
            s.infer_interval_s = 0.5
        if not (2.0 <= s.max_buffer_s <= 30.0):
            self.warnings.append("streaming.max_buffer_s out of range [2, 30]; using 12")
            s.max_buffer_s = 12.0
        v = self.vad
        if not (0.0 < v.offset <= v.onset <= 1.0):
            self.warnings.append("vad onset/offset invalid; using defaults")
            v.onset, v.offset = 0.5, 0.35

    # ── Load/save ────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "Config":
        """Load config from XDG path, migrating a legacy file if present."""
        path = config_path()
        raw: dict[str, Any] | None = None
        migrated_from: Path | None = None

        if path.exists():
            raw = _read_json(path)
        else:
            for legacy in _legacy_config_paths():
                if legacy.exists():
                    raw = _read_json(legacy)
                    if raw is not None:
                        migrated_from = legacy
                    break

        if raw is None:
            cfg = cls()
        else:
            cfg = cls.from_dict(raw)

        if migrated_from is not None:
            cfg.warnings.append(f"migrated legacy config from {migrated_from}")
            try:
                cfg.save()
            except OSError as e:
                cfg.warnings.append(f"could not write migrated config: {e}")
        return cfg

    def save(self) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
        os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None
