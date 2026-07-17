import json

import pytest

from wldictate.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.model == "small.en"
    assert cfg.streaming.enabled is True
    assert cfg.vad.min_silence_ms == 500


def test_input_device_name_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = Config()
    cfg.input_device = 26
    cfg.input_device_name = "HD Pro Webcam C920 Analog Stereo"
    cfg.save()
    loaded = Config.load()
    assert loaded.input_device == 26
    assert loaded.input_device_name == "HD Pro Webcam C920 Analog Stereo"
    bad = Config.from_dict({"input_device_name": 42})
    assert bad.input_device_name is None and bad.warnings


def test_from_dict_valid():
    cfg = Config.from_dict(
        {
            "model": "tiny.en",
            "input_device": 4,
            "streaming": {"infer_interval_s": 1.0},
            "vad": {"min_silence_ms": 700},
        }
    )
    assert cfg.model == "tiny.en"
    assert cfg.input_device == 4
    assert cfg.streaming.infer_interval_s == 1.0
    assert cfg.vad.min_silence_ms == 700
    assert cfg.warnings == []


def test_from_dict_unknown_keys_warn():
    cfg = Config.from_dict({"nope": 1, "vad": {"bogus": True}})
    assert any("nope" in w for w in cfg.warnings)
    assert any("vad.bogus" in w for w in cfg.warnings)


def test_from_dict_bad_types_keep_defaults():
    cfg = Config.from_dict({"model": 5, "streaming": {"infer_interval_s": "fast"}})
    assert cfg.model == "small.en"
    assert cfg.streaming.infer_interval_s == 0.5
    assert len(cfg.warnings) == 2


def test_out_of_range_values_clamped():
    cfg = Config.from_dict({"streaming": {"infer_interval_s": 99.0}})
    assert cfg.streaming.infer_interval_s == 0.5
    assert cfg.warnings


def test_invalid_device_and_mode():
    cfg = Config.from_dict({"device": "tpu", "typing": {"mode": "chaos"}})
    assert cfg.device == "auto"
    assert cfg.typing.mode == "correcting"
    assert cfg.warnings


def test_typing_mode_defaults_to_correcting_and_accepts_commit():
    assert Config.from_dict({}).typing.mode == "correcting"
    cfg = Config.from_dict({"typing": {"mode": "commit"}})
    assert cfg.typing.mode == "commit"
    assert not cfg.warnings


def test_legacy_migration(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    legacy = tmp_path / "legacy" / "config.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"input_device": 26}))
    monkeypatch.setattr(
        "wldictate.config._legacy_config_paths", lambda: [legacy]
    )
    cfg = Config.load()
    assert cfg.input_device == 26
    assert any("migrated" in w for w in cfg.warnings)
    # Migrated file written to the XDG location.
    written = json.loads((tmp_path / "xdg" / "wl-dictate" / "config.json").read_text())
    assert written["input_device"] == 26
    assert written["model"] == "small.en"


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = Config()
    cfg.input_device = 7
    cfg.save()
    loaded = Config.load()
    assert loaded.input_device == 7
    assert loaded.warnings == []


def test_corrupt_config_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "wl-dictate" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    monkeypatch.setattr("wldictate.config._legacy_config_paths", lambda: [])
    cfg = Config.load()
    assert cfg.model == "small.en"


# ── contextual dictation config ──────────────────────────────────────────────


def test_contextual_defaults_round_trip():
    cfg = Config()
    d = json.loads(json.dumps(cfg.to_dict()))
    assert d["contextual"]["profile"] == "local"
    assert d["contextual"]["profiles"]["local"]["base_url"] == "http://127.0.0.1:8890/v1"
    cfg2 = Config.from_dict(d)
    assert cfg2.contextual.profile == "local"
    assert cfg2.contextual.profiles["openrouter"].model.startswith("nvidia/")
    assert cfg2.contextual.profiles["anthropic"].backend == "anthropic"
    assert not cfg2.warnings


def test_contextual_unknown_profile_falls_back():
    cfg = Config.from_dict({"contextual": {"profile": "chaos"}})
    assert cfg.contextual.profile == "local"
    assert cfg.warnings


def test_contextual_partial_profile_merge():
    cfg = Config.from_dict(
        {"contextual": {"profiles": {"local": {"model": "other-model"}}}}
    )
    assert cfg.contextual.profiles["local"].model == "other-model"
    # Untouched fields keep their defaults; other profiles survive the merge.
    assert cfg.contextual.profiles["local"].base_url == "http://127.0.0.1:8890/v1"
    assert "openrouter" in cfg.contextual.profiles
    assert not cfg.warnings


def test_contextual_new_custom_profile():
    cfg = Config.from_dict(
        {
            "contextual": {
                "profile": "work",
                "profiles": {
                    "work": {
                        "backend": "openai",
                        "base_url": "https://llm.example/v1",
                        "model": "m",
                    }
                },
            }
        }
    )
    assert cfg.contextual.profile == "work"
    assert cfg.contextual.profiles["work"].base_url == "https://llm.example/v1"
    assert not cfg.warnings


def test_contextual_invalid_values_fall_back():
    cfg = Config.from_dict(
        {
            "contextual": {
                "timeout_s": 999,
                "notify": "yes",
                "profiles": {"local": {"backend": "grpc", "model": 7}},
            }
        }
    )
    assert cfg.contextual.timeout_s == 10.0
    assert cfg.contextual.notify is True
    assert cfg.contextual.profiles["local"].backend == "openai"
    assert cfg.contextual.profiles["local"].model == "qwen3.5-9b"
    assert cfg.warnings
