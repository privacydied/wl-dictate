import json

import pytest

from wldictate.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.model == "distil-small.en"
    assert cfg.streaming.enabled is True
    assert cfg.vad.min_silence_ms == 500


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
    assert cfg.model == "distil-small.en"
    assert cfg.streaming.infer_interval_s == 0.5
    assert len(cfg.warnings) == 2


def test_out_of_range_values_clamped():
    cfg = Config.from_dict({"streaming": {"infer_interval_s": 99.0}})
    assert cfg.streaming.infer_interval_s == 0.5
    assert cfg.warnings


def test_invalid_device_and_mode():
    cfg = Config.from_dict({"device": "tpu", "typing": {"mode": "chaos"}})
    assert cfg.device == "auto"
    assert cfg.typing.mode == "commit"


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
    assert written["model"] == "distil-small.en"


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
    assert cfg.model == "distil-small.en"
