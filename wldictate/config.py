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
    # Maximum re-decode cadence; the effective cadence adapts down to
    # min_infer_interval_s when the model decodes fast (1.5x measured decode
    # time), so short utterances on a fast GPU tick tighter than this.
    infer_interval_s: float = 0.5
    min_infer_interval_s: float = 0.25
    min_new_audio_s: float = 0.3
    # Committed audio is trimmed out of the buffer beyond this: smaller
    # windows decode faster (every tick re-decodes the whole window).
    max_buffer_s: float = 8.0


@dataclass
class VadConfig:
    backend: str = "auto"  # auto | silero | energy
    onset: float = 0.5
    offset: float = 0.35
    onset_frames: int = 2
    min_silence_ms: int = 500
    # Speculative finalize: start the final decode after this much silence
    # (while still counting toward min_silence_ms). If the pause holds, the
    # final text lands the instant the utterance ends; if you resume speaking,
    # the speculative decode is discarded. 0 disables.
    speculative_silence_ms: int = 200
    pre_roll_ms: int = 320
    min_speech_s: float = 0.3
    # Length cap per engine utterance. Hitting it no longer chops your
    # message: the gate rolls seamlessly into a new utterance (no onset gap)
    # and contextual mode transforms the whole chain as one message. Buffer
    # trimming bounds decode cost, so this is just a state-bounding safety.
    max_utterance_s: float = 120.0


@dataclass
class TypingConfig:
    mode: str = "correcting"  # correcting (live rewrite) | commit (append-only)
    # Keystroke device: "auto" prefers a persistent in-process virtual
    # keyboard (one Wayland connection for the worker lifetime — no
    # per-rewrite process spawn, no fresh-connection Electron space drop)
    # and falls back to the wtype subprocess when unavailable. "wtype"
    # forces the subprocess path; "vkbd" forces the persistent device.
    backend: str = "auto"  # auto | vkbd | wtype
    wtype_timeout_s: float = 10.0
    # Per-keystroke delay (ms) passed to `wtype -d`. Electron/Chromium apps
    # (Vesktop, Discord, VSCode, Slack) drop characters — usually spaces and
    # punctuation — when keystrokes arrive too fast; a small delay fixes it.
    # 0 = no delay (fastest; fine for native GTK/Qt fields).
    wtype_delay_ms: int = 6
    # Settle delay (ms) BEFORE the first keystroke of each wtype call, passed
    # as `wtype -s`. Empirically this does NOT fix Electron's leading-space
    # drop (see electron_workaround below) — kept as a knob for other slow
    # compositor paths. 0 = disabled.
    wtype_press_delay_ms: int = 0
    # Chromium/Electron apps drop leading SPACE keys on every fresh wtype
    # connection (regardless of delays), fusing words: "TestingTesting".
    # Workaround: when the focused window is one of electron_app_classes and
    # the text starts with a space, prefix an invisible zero-width space
    # (U+200B) — it "opens the gate" so the real space lands. Only applied to
    # matching apps so terminals/editors never receive ZWSP junk.
    electron_workaround: bool = True
    electron_app_classes: list[str] = field(
        default_factory=lambda: [
            "vesktop",
            "discord",
            "webcord",
            "legcord",
            "chromium",
            "chrome",
            "electron",
            "slack",
            "element",
            "signal",
        ]
    )
    # Type a single trailing space after sentence-final punctuation when an
    # utterance ends, so manual typing can continue naturally.
    sentence_trailing_space: bool = True
    # Capitalize the first letter of each sentence (utterance start and after
    # . ! ?). Whisper often lowercases the word after a pause.
    capitalize_sentences: bool = True


@dataclass
class ContextualProfile:
    """One LLM endpoint for contextual dictation.

    backend "openai" is any OpenAI-compatible server (local llama.cpp
    llama-server, OpenRouter, ...); backend "anthropic" uses the official
    Anthropic SDK. The API key is read from ``api_key_file`` (preferred —
    systemd-friendly; ``~`` is expanded) or the ``api_key_env`` environment
    variable. Both empty means no key (fine for a local server).
    """

    backend: str = "openai"  # openai (OpenAI-compatible) | anthropic
    base_url: str = ""  # unused for the anthropic backend
    model: str = ""
    api_key_env: str = ""
    api_key_file: str = ""


def _default_contextual_profiles() -> dict[str, ContextualProfile]:
    return {
        "local": ContextualProfile(
            backend="openai",
            base_url="http://127.0.0.1:8890/v1",
            model="qwen3.5-9b",  # matches scripts/llama-contextual.sh --alias
        ),
        "openrouter": ContextualProfile(
            backend="openai",
            base_url="https://openrouter.ai/api/v1",
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            api_key_env="OPENAI_API_KEY",
            api_key_file="~/.config/wl-dictate/openrouter.key",
        ),
        "local35": ContextualProfile(
            backend="openai",
            # The q35-fast llama-server (Qwen3.6-35B): much smarter transforms
            # when it's running; text-only unless its mmproj is loaded.
            base_url="http://127.0.0.1:8888/v1",
            model="qwen36-35b-a3b",
        ),
        "anthropic": ContextualProfile(
            backend="anthropic",
            # claude-haiku-4-5: latency-appropriate for per-utterance
            # dictation transforms; change to any Claude model id.
            model="claude-haiku-4-5",
            api_key_env="ANTHROPIC_API_KEY",
            api_key_file="~/.config/wl-dictate/anthropic.key",
        ),
    }


@dataclass
class ContextualConfig:
    """Contextual dictation (Ctrl+Alt+D): LLM transform of each utterance."""

    profile: str = "local"  # key into profiles — switching endpoints is one field
    # Auto-pick a runnable profile at worker startup: if the configured profile
    # is a local model this machine can't run (checked against detected VRAM/
    # RAM), fall back to the largest local model that fits, then to a cloud
    # profile. Set false to always honour ``profile`` as-is.
    auto_select: bool = True
    timeout_s: float = 10.0  # whole LLM budget; on timeout the Whisper text stays
    max_output_tokens: int = 1000
    context_max_chars: int = 4000  # per-source cap for selection/clipboard
    # Attach a screenshot of the focused window (vision models see the
    # conversation you're replying to). "local": only for local endpoints
    # (privacy default) | "always": also cloud profiles | "off".
    screenshot: str = "local"
    # Stream the replacement: tokens are typed as they generate (perceived
    # latency = time-to-first-token) instead of waiting for the full output.
    stream: bool = True
    # Pause length that ends an utterance in CONTEXTUAL mode. Longer than the
    # standard vad.min_silence_ms so thinking pauses don't fragment your
    # message into separate transforms. 0 = use vad.min_silence_ms.
    min_silence_ms: int = 800
    notify: bool = True  # "Transforming…" / error toasts via notify-send
    # Who is speaking: name, tone preferences, sign-offs — included in the
    # transform system prompt so replies sound like YOU.
    # e.g. "I'm Taz. Casual with friends, lowercase ok. Sign work emails 'T'."
    persona: str = ""
    # Names/jargon the speech recognizer and the transform should know
    # (project names, people, technical terms). Also biases Whisper decoding,
    # so these words stop being misheard in BOTH dictation modes.
    vocabulary: list[str] = field(default_factory=list)
    # Per-app guidance: substring of the window class -> extra instruction.
    # e.g. {"vesktop": "very casual, emoji fine", "betterbird": "professional email tone"}
    app_hints: dict[str, str] = field(default_factory=dict)
    profiles: dict[str, ContextualProfile] = field(
        default_factory=_default_contextual_profiles
    )

    # dict[str, dataclass] doesn't fit Config's generic scalar (de)serializer,
    # so this section owns its own round-trip (special-cased in Config).

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "profile": self.profile,
            "auto_select": self.auto_select,
            "timeout_s": self.timeout_s,
            "max_output_tokens": self.max_output_tokens,
            "context_max_chars": self.context_max_chars,
            "screenshot": self.screenshot,
            "stream": self.stream,
            "min_silence_ms": self.min_silence_ms,
            "notify": self.notify,
            "persona": self.persona,
            "vocabulary": list(self.vocabulary),
            "app_hints": dict(self.app_hints),
            "profiles": {name: dict(vars(p)) for name, p in self.profiles.items()},
        }
        return out

    def apply_dict(self, raw: dict[str, Any], warnings: list[str]) -> None:
        """Merge a user-provided section over the defaults, with type checks."""
        scalar_types = {
            "profile": str,
            "auto_select": bool,
            "timeout_s": (int, float),
            "max_output_tokens": int,
            "context_max_chars": int,
            "screenshot": str,
            "stream": bool,
            "min_silence_ms": int,
            "notify": bool,
            "persona": str,
        }
        for key, value in raw.items():
            if key == "vocabulary":
                if isinstance(value, list) and all(isinstance(v, str) for v in value):
                    self.vocabulary = value
                else:
                    warnings.append("invalid contextual.vocabulary; keeping default")
                continue
            if key == "app_hints":
                if isinstance(value, dict) and all(
                    isinstance(k, str) and isinstance(v, str) for k, v in value.items()
                ):
                    self.app_hints = value
                else:
                    warnings.append("invalid contextual.app_hints; keeping default")
                continue
            if key == "profiles":
                if not isinstance(value, dict):
                    warnings.append("invalid contextual.profiles; keeping defaults")
                    continue
                for name, profile_raw in value.items():
                    if not isinstance(name, str) or not isinstance(profile_raw, dict):
                        warnings.append(
                            f"invalid contextual profile entry {name!r}; ignored"
                        )
                        continue
                    profile = self.profiles.setdefault(name, ContextualProfile())
                    for p_key, p_value in profile_raw.items():
                        if p_key not in vars(profile):
                            warnings.append(
                                f"unknown config key ignored: contextual.profiles.{name}.{p_key}"
                            )
                        elif isinstance(p_value, str):
                            setattr(profile, p_key, p_value)
                        else:
                            warnings.append(
                                f"invalid value for contextual.profiles.{name}.{p_key};"
                                " keeping default"
                            )
                continue
            if key not in scalar_types:
                warnings.append(f"unknown config key ignored: contextual.{key}")
                continue
            expected = scalar_types[key]
            ok = isinstance(value, expected)
            if expected is not bool and isinstance(value, bool):
                ok = False  # bool is an int subclass; reject for numeric fields
            if ok:
                if key == "timeout_s":
                    value = float(value)
                setattr(self, key, value)
            else:
                warnings.append(f"invalid value for contextual.{key}; keeping default")

    def validate(self, warnings: list[str]) -> None:
        if self.profile not in self.profiles:
            warnings.append(
                f"unknown contextual.profile '{self.profile}'; using 'local'"
            )
            self.profile = "local"
            self.profiles.setdefault("local", _default_contextual_profiles()["local"])
        if not (0.5 <= self.timeout_s <= 60.0):
            warnings.append("contextual.timeout_s out of range [0.5, 60]; using 10")
            self.timeout_s = 10.0
        if self.screenshot not in ("off", "local", "always"):
            warnings.append(
                f"invalid contextual.screenshot '{self.screenshot}'; using 'local'"
            )
            self.screenshot = "local"
        for name, profile in self.profiles.items():
            if profile.backend not in ("openai", "anthropic"):
                warnings.append(
                    f"invalid backend '{profile.backend}' in contextual profile"
                    f" '{name}'; using 'openai'"
                )
                profile.backend = "openai"


@dataclass
class UiConfig:
    # On-screen status pill ("● Dictating…" / mode) — the tray icon alone is
    # easy to miss; this gives an unmissable are-we-listening indicator.
    osd: bool = True
    # Play freedesktop start/stop sounds on toggle (paplay).
    sound_cues: bool = False
    # Auto-stop dictation after this many seconds with no committed speech
    # (mic privacy / battery). 0 = never.
    idle_stop_s: float = 0.0


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
    ui: UiConfig = field(default_factory=UiConfig)
    # Special-cased in to_dict/from_dict (owns its own round-trip).
    contextual: ContextualConfig = field(default_factory=ContextualConfig)

    # Populated by load(); not serialized.
    warnings: list[str] = field(default_factory=list, repr=False)

    # ── (De)serialization ────────────────────────────────────────────────

    _NESTED = {
        "streaming": StreamingConfig,
        "vad": VadConfig,
        "typing": TypingConfig,
        "audio": AudioConfig,
        "ui": UiConfig,
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
        out["contextual"] = self.contextual.to_dict()
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
            elif isinstance(current, list):
                ok = isinstance(value, list) and all(
                    isinstance(item, str) for item in value
                )
            else:
                ok = False
            if ok:
                setattr(obj, key, value)
            else:
                cfg.warnings.append(f"invalid value for {path}; keeping default")

        for key, value in raw.items():
            if key == "contextual":
                if isinstance(value, dict):
                    cfg.contextual.apply_dict(value, cfg.warnings)
                else:
                    cfg.warnings.append("invalid section contextual; keeping defaults")
            elif key in cls._NESTED:
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
        if self.typing.mode not in ("commit", "correcting"):
            self.warnings.append(
                f"unsupported typing.mode '{self.typing.mode}'; using 'correcting'"
            )
            self.typing.mode = "correcting"
        if self.typing.backend not in ("auto", "vkbd", "wtype"):
            self.warnings.append(
                f"invalid typing.backend '{self.typing.backend}'; using 'auto'"
            )
            self.typing.backend = "auto"
        if self.vad.backend not in ("auto", "silero", "energy"):
            self.warnings.append(f"invalid vad.backend '{self.vad.backend}'; using 'auto'")
            self.vad.backend = "auto"
        s = self.streaming
        if not (0.1 <= s.infer_interval_s <= 5.0):
            self.warnings.append("streaming.infer_interval_s out of range [0.1, 5]; using 0.5")
            s.infer_interval_s = 0.5
        if not (0.1 <= s.min_infer_interval_s <= s.infer_interval_s):
            self.warnings.append(
                "streaming.min_infer_interval_s out of range; using infer_interval_s"
            )
            s.min_infer_interval_s = s.infer_interval_s
        if not (2.0 <= s.max_buffer_s <= 30.0):
            self.warnings.append("streaming.max_buffer_s out of range [2, 30]; using 8")
            s.max_buffer_s = 8.0
        if self.vad.speculative_silence_ms < 0:
            self.warnings.append("vad.speculative_silence_ms negative; using 200")
            self.vad.speculative_silence_ms = 200
        v = self.vad
        if not (0.0 < v.offset <= v.onset <= 1.0):
            self.warnings.append("vad onset/offset invalid; using defaults")
            v.onset, v.offset = 0.5, 0.35
        self.contextual.validate(self.warnings)

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
