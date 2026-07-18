import wldictate.hardware as hw
from wldictate.config import ContextualConfig


def _res(vram_mb=0, ram_mb=8000, gpu_name=""):
    return hw.SystemResources(vram_mb=vram_mb, ram_mb=ram_mb, gpu_name=gpu_name)


# ── detection parsing ─────────────────────────────────────────────────────────


def test_detect_nvidia_picks_largest(monkeypatch):
    monkeypatch.setattr(
        hw, "_run", lambda *a, **k: "8192, NVIDIA GeForce RTX 3070\n24564, RTX 4090\n"
    )
    mb, name = hw._detect_nvidia()
    assert mb == 24564 and "4090" in name


def test_detect_nvidia_absent(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda *a, **k: None)
    assert hw._detect_nvidia() == (0, "")


def test_detect_ram(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16277372 kB\nMemFree: 100 kB\n")
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *a, **k: real_open(meminfo, *a, **k)
        if path == "/proc/meminfo"
        else real_open(path, *a, **k),
    )
    assert hw._detect_ram_mb() == 16277372 // 1024


# ── model tier selection ──────────────────────────────────────────────────────


def test_recommend_local_model_gpu_ladder():
    # ram_mb=0 isolates the VRAM ladder from the CPU-RAM fallback.
    assert hw.recommend_local_model(_res(vram_mb=24000, ram_mb=0)).key == "35b"
    assert hw.recommend_local_model(_res(vram_mb=8500, ram_mb=0)).key == "9b"
    assert hw.recommend_local_model(_res(vram_mb=5000, ram_mb=0)).key == "4b"
    assert hw.recommend_local_model(_res(vram_mb=2500, ram_mb=0)).key == "1.7b"


def test_recommend_local_model_cpu_fallback():
    # No GPU, but plenty of RAM -> largest tier whose RAM floor fits.
    assert hw.recommend_local_model(_res(vram_mb=0, ram_mb=13000)).key == "9b"
    assert hw.recommend_local_model(_res(vram_mb=0, ram_mb=6500)).key == "4b"


def test_recommend_local_model_none_when_tiny():
    assert hw.recommend_local_model(_res(vram_mb=0, ram_mb=2000)) is None


# ── Whisper model selection ───────────────────────────────────────────────────


def test_whisper_passthrough_when_not_auto():
    name, reason = hw.select_whisper_model("small.en", _res(vram_mb=24000), cuda=True)
    assert name == "small.en" and reason is None


def test_whisper_cpu_default():
    name, reason = hw.select_whisper_model("auto", _res(vram_mb=0), cuda=False)
    assert name == "base.en" and reason


def test_whisper_gpu_ladder():
    assert hw.select_whisper_model("auto", _res(vram_mb=24576), cuda=True)[0] == "large-v3"
    assert hw.select_whisper_model("auto", _res(vram_mb=4000), cuda=True)[0] == "medium.en"
    assert hw.select_whisper_model("auto", _res(vram_mb=2000), cuda=True)[0] == "small.en"


def test_whisper_gpu_present_but_vram_unknown():
    # CUDA works but nvidia-smi gave nothing -> safe GPU default, not CPU.
    assert hw.select_whisper_model("auto", _res(vram_mb=0), cuda=True)[0] == "small.en"


def test_whisper_cuda_gates_gpu_tiers():
    # Big VRAM reported but no usable CUDA runtime -> stay on CPU model.
    name, _ = hw.select_whisper_model("auto", _res(vram_mb=24000), cuda=False)
    assert name == "base.en"


# ── profile fit + auto-selection ──────────────────────────────────────────────


def test_cloud_profiles_always_fit():
    cfg = ContextualConfig()
    tiny = _res(vram_mb=0, ram_mb=1000)
    assert hw.profile_fits("anthropic", cfg.profiles["anthropic"], tiny)
    assert hw.profile_fits("openrouter", cfg.profiles["openrouter"], tiny)


def test_local_profile_needs_hardware():
    cfg = ContextualConfig()
    assert not hw.profile_fits("local", cfg.profiles["local"], _res(vram_mb=4000, ram_mb=4000))
    assert hw.profile_fits("local", cfg.profiles["local"], _res(vram_mb=12000))


def test_desktop_keeps_configured_local_profile():
    cfg = ContextualConfig()  # profile="local" (9B)
    notes = hw.autoselect_profile(cfg, res=_res(vram_mb=24000, gpu_name="RTX 4090"))
    assert notes == [] and cfg.profile == "local"


def test_laptop_downgrades_to_cloud():
    cfg = ContextualConfig()  # wants local 9B; laptop can't run any local model
    notes = hw.autoselect_profile(cfg, res=_res(vram_mb=0, ram_mb=8000))
    assert notes, "expected a downgrade note"
    assert hw.is_cloud_profile(cfg.profiles[cfg.profile])


def test_prefers_smaller_local_over_cloud():
    cfg = ContextualConfig()
    # Add a small local rung the machine CAN run; 9B configured but doesn't fit.
    cfg.profiles["local_small"] = type(cfg.profiles["local"])(
        backend="openai",
        base_url="http://127.0.0.1:8890/v1",
        model="qwen3-4b",
    )
    notes = hw.autoselect_profile(cfg, res=_res(vram_mb=5000))
    assert notes and cfg.profile == "local_small"


def test_autoselect_respects_disable_flag():
    cfg = ContextualConfig()
    cfg.auto_select = False
    notes = hw.autoselect_profile(cfg, res=_res(vram_mb=0, ram_mb=8000))
    assert notes == [] and cfg.profile == "local"


def test_autoselect_respects_env(monkeypatch):
    monkeypatch.setenv("WL_DICTATE_NO_AUTOSELECT", "1")
    cfg = ContextualConfig()
    notes = hw.autoselect_profile(cfg, res=_res(vram_mb=0, ram_mb=8000))
    assert notes == [] and cfg.profile == "local"


# ── config round-trip ─────────────────────────────────────────────────────────


def test_auto_select_config_roundtrip():
    cfg = ContextualConfig()
    assert cfg.auto_select is True
    warnings: list[str] = []
    cfg.apply_dict({"auto_select": False}, warnings)
    assert cfg.auto_select is False and not warnings
    assert cfg.to_dict()["auto_select"] is False


def test_auto_select_rejects_non_bool():
    cfg = ContextualConfig()
    warnings: list[str] = []
    cfg.apply_dict({"auto_select": "yes"}, warnings)
    assert cfg.auto_select is True and warnings
