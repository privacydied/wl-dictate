"""Hardware capability detection and contextual-model selection.

The contextual-dictation transform runs an LLM. The default ``local`` profile
serves Qwen3.5-9B via ``scripts/llama-contextual.sh`` — great on a desktop GPU,
but a laptop with 8 GB RAM and no discrete GPU can't run it and the transform
either fails to connect or thrashes.

This module probes the machine (GPU VRAM via ``nvidia-smi``/``rocm-smi``, then
system RAM via ``/proc/meminfo``) and answers two questions:

- **Which local GGUF should the server load?** ``recommend_local_model`` picks
  the largest tier that fits — used by the launch script (``--pick-model``).
- **Which contextual profile should the app use?** ``autoselect_profile``
  keeps the configured profile if the hardware can run it, otherwise falls back
  down the local tiers and finally to a cloud profile.

Detection is best-effort and never raises: a missing ``nvidia-smi`` just means
"no GPU detected" (VRAM 0), and selection falls back to CPU-RAM checks. Results
are cached for the process — probing shells out and the answer can't change
mid-run.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

# ── Detection ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SystemResources:
    """What the machine can offer an inference server."""

    vram_mb: int  # largest single-GPU VRAM in MB (0 = no GPU detected)
    ram_mb: int  # total system RAM in MB
    gpu_name: str  # e.g. "NVIDIA GeForce RTX 4090"; "" when no GPU

    @property
    def has_gpu(self) -> bool:
        return self.vram_mb > 0


def _run(cmd: list[str], timeout: float = 4.0) -> str | None:
    """Run a probe command, returning stdout or None on any failure."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _detect_nvidia() -> tuple[int, str]:
    """(VRAM MB, name) of the largest NVIDIA GPU, or (0, "")."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,name",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return 0, ""
    best_mb, best_name = 0, ""
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mb = int(float(parts[0]))
        except ValueError:
            continue
        if mb > best_mb:
            best_mb, best_name = mb, parts[1]
    return best_mb, best_name


def _detect_amd() -> tuple[int, str]:
    """(VRAM MB, name) of the largest AMD GPU via rocm-smi, or (0, "")."""
    # --showmeminfo reports bytes; --csv keeps parsing simple across versions.
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if not out:
        return 0, ""
    best_mb = 0
    for line in out.splitlines():
        # Rows look like: card0,<total bytes>,<used bytes>
        m = re.search(r"(\d{7,})", line)  # VRAM totals are >= ~1e7 bytes
        if not m:
            continue
        mb = int(m.group(1)) // (1024 * 1024)
        best_mb = max(best_mb, mb)
    if best_mb == 0:
        return 0, ""
    return best_mb, "AMD GPU"


def _detect_ram_mb() -> int:
    """Total system RAM in MB from /proc/meminfo (0 if unavailable)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024  # value is in kB
    except (OSError, ValueError, IndexError):
        pass
    return 0


def probe_resources() -> SystemResources:
    """Detect GPU VRAM + system RAM (uncached — prefer ``detect_resources``)."""
    vram_mb, gpu_name = _detect_nvidia()
    if vram_mb == 0:
        vram_mb, gpu_name = _detect_amd()
    return SystemResources(
        vram_mb=vram_mb,
        ram_mb=_detect_ram_mb(),
        gpu_name=gpu_name,
    )


_cache: SystemResources | None = None


def detect_resources(*, refresh: bool = False) -> SystemResources:
    """Process-cached hardware probe. Hardware can't change mid-run, and each
    probe shells out, so we detect once."""
    global _cache
    if _cache is None or refresh:
        _cache = probe_resources()
    return _cache


# ── Model tiers ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelTier:
    """A local GGUF the launcher can serve, with its resource floor.

    ``min_vram_mb`` runs the model fully on the GPU (weights + q8 KV cache at
    the profile's context). ``min_ram_mb`` is the (higher) floor to run it on
    CPU — much slower, but the only option on a laptop with no discrete GPU.
    """

    key: str  # short id, e.g. "9b"
    hf_spec: str  # llama.cpp -hf spec (repo:quant)
    label: str  # human name for reports
    min_vram_mb: int
    min_ram_mb: int


#: Local tiers, largest first. ``recommend_local_model`` walks this in order
#: and returns the first that fits, so ordering encodes preference. The VRAM
#: floors assume the Q4_K_M/Q6 quants these repos ship plus a q8 KV cache at
#: ~16k context; they are deliberately a little generous so selection errs
#: toward "actually runs" over "technically maybe fits".
LOCAL_MODEL_TIERS: tuple[ModelTier, ...] = (
    ModelTier(
        key="35b",
        hf_spec="unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
        label="Qwen3.6-35B-A3B",
        min_vram_mb=22000,
        min_ram_mb=40000,
    ),
    ModelTier(
        key="9b",
        hf_spec="unsloth/Qwen3.5-9B-MTP-GGUF:Q6_K",
        label="Qwen3.5-9B",
        min_vram_mb=8000,
        min_ram_mb=12000,
    ),
    ModelTier(
        key="4b",
        # MTP GGUF: ships speculative-decoding weights, so the launcher's
        # --spec-type draft-mtp flags apply (faster generation on the laptop
        # tier, where it matters most).
        hf_spec="unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M",
        label="Qwen3.5-4B",
        min_vram_mb=4200,
        min_ram_mb=6000,
    ),
    ModelTier(
        key="1.7b",
        hf_spec="unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
        label="Qwen3-1.7B",
        min_vram_mb=2200,
        min_ram_mb=3500,
    ),
)

#: Resource floors for the shipped *local* profiles, keyed by profile name, so
#: profile auto-selection matches the launcher's tier logic. Profiles not
#: listed here fall back to a size estimate parsed from the model id.
_PROFILE_TIER: dict[str, ModelTier] = {
    "local": LOCAL_MODEL_TIERS[1],  # 9b
    "local35": LOCAL_MODEL_TIERS[0],  # 35b
}

# ── Whisper (transcription) model selection ──────────────────────────────────

WHISPER_AUTO = "auto"

#: GPU tiers for the transcription model, largest first: (min VRAM MB, name).
#: The float16 weights are small next to an LLM (large-v3 ~3 GB), so these
#: floors are modest — they mostly separate "big GPU -> best accuracy" from
#: "small GPU -> lighter model". The last entry (threshold 0) is the default
#: for any working CUDA device whose VRAM we couldn't size.
WHISPER_GPU_TIERS: tuple[tuple[int, str], ...] = (
    (6000, "large-v3"),
    (3000, "medium.en"),
    (0, "small.en"),
)

#: No usable GPU -> transcribe on CPU. A light English model keeps dictation
#: low-latency (large-v3 on CPU is unusably slow for realtime).
WHISPER_CPU_MODEL = "base.en"


def cuda_available() -> bool:
    """Whether CTranslate2 (faster-whisper's backend) can actually use a GPU.

    Mirrors ``FasterWhisperTranscriber._cuda_available`` — VRAM alone isn't
    enough (the driver can be present without a usable CUDA runtime), so the
    Whisper picker gates the GPU tiers on this, not just ``res.vram_mb``.
    """
    try:
        from ctranslate2 import get_cuda_device_count

        return get_cuda_device_count() > 0
    except Exception:
        return False


def select_whisper_model(
    model: str, res: SystemResources, *, cuda: bool
) -> tuple[str, str | None]:
    """Resolve a possibly-``"auto"`` Whisper model to a concrete name.

    Returns ``(name, reason)``: ``reason`` is None when ``model`` is already a
    concrete name (passed through untouched), or an explanation when ``"auto"``
    is resolved from the detected hardware.
    """
    if model != WHISPER_AUTO:
        return model, None
    if not cuda:
        return (
            WHISPER_CPU_MODEL,
            f"model 'auto' -> '{WHISPER_CPU_MODEL}' (no GPU; CPU transcription)",
        )
    for min_vram, name in WHISPER_GPU_TIERS:
        if res.vram_mb >= min_vram:
            where = f"{res.gpu_name} {res.vram_mb} MB VRAM" if res.vram_mb else "GPU"
            return name, f"model 'auto' -> '{name}' ({where})"
    return WHISPER_CPU_MODEL, None  # unreachable (last tier is 0), keeps mypy happy


def resolve_whisper_model(model: str) -> tuple[str, str | None]:
    """``select_whisper_model`` with hardware + CUDA detected for you."""
    return select_whisper_model(model, detect_resources(), cuda=cuda_available())


_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def _estimate_local_floor(model: str) -> tuple[int, int]:
    """(min_vram_mb, min_ram_mb) guessed from a model id like 'qwen-7b'.

    Rough: ~0.9 GB VRAM per billion params (Q4-ish weights + KV), CPU RAM floor
    a bit higher. Only used for custom local profiles with no registered tier.
    """
    m = _PARAMS_RE.search(model)
    params = float(m.group(1)) if m else 8.0  # unknown -> assume mid-size
    vram = int(params * 900) + 1500
    ram = int(params * 1100) + 2500
    return vram, ram


def _is_local_url(base_url: str) -> bool:
    return base_url.startswith(("http://127.", "http://localhost", "http://[::1]"))


def is_cloud_profile(profile) -> bool:
    """A profile served off-machine — always 'runnable' regardless of hardware
    (it needs network + a key instead, validated elsewhere)."""
    if profile.backend == "anthropic":
        return True
    return not _is_local_url(profile.base_url)


def profile_floor(name: str, profile) -> tuple[int, int]:
    """(min_vram_mb, min_ram_mb) a local profile needs. (0, 0) for cloud."""
    if is_cloud_profile(profile):
        return 0, 0
    tier = _PROFILE_TIER.get(name)
    if tier is not None:
        return tier.min_vram_mb, tier.min_ram_mb
    return _estimate_local_floor(profile.model)


def profile_fits(name: str, profile, res: SystemResources) -> bool:
    """Can this machine run ``profile``? Cloud profiles always fit; a local
    profile fits if the GPU has the VRAM, or there's enough RAM for CPU."""
    if is_cloud_profile(profile):
        return True
    vram_floor, ram_floor = profile_floor(name, profile)
    if res.vram_mb >= vram_floor:
        return True
    return ram_floor > 0 and res.ram_mb >= ram_floor


def recommend_local_model(res: SystemResources) -> ModelTier | None:
    """Largest local tier the machine can serve (GPU or CPU), or None."""
    for tier in LOCAL_MODEL_TIERS:
        if res.vram_mb >= tier.min_vram_mb or res.ram_mb >= tier.min_ram_mb:
            return tier
    return None


# ── Profile auto-selection ────────────────────────────────────────────────────

_AUTOSELECT_ENV = "WL_DICTATE_NO_AUTOSELECT"


def _local_rank(name: str, profile) -> int:
    """Bigger local models sort first (more capable); cloud sorts last so a
    fitting local model is always preferred over cloud."""
    vram, _ = profile_floor(name, profile)
    return vram


def select_profile(
    cfg, res: SystemResources
) -> tuple[str, str | None]:
    """Choose a contextual profile for ``res``.

    Returns ``(profile_name, reason)`` where ``reason`` is None when the
    configured profile is kept, or a human-readable explanation when it is
    swapped. Preference: keep the configured profile if it fits; else the
    largest *local* profile that fits; else a cloud profile; else keep the
    configured one (nothing better available — let the normal error surface).
    """
    current = cfg.profiles.get(cfg.profile)
    if current is not None and profile_fits(cfg.profile, current, res):
        return cfg.profile, None

    local_fits = sorted(
        (
            (name, p)
            for name, p in cfg.profiles.items()
            if not is_cloud_profile(p) and profile_fits(name, p, res)
        ),
        key=lambda item: _local_rank(*item),
        reverse=True,
    )
    if local_fits:
        name = local_fits[0][0]
        return name, _swap_reason(cfg.profile, name, res, "largest local model that fits")

    for name, p in cfg.profiles.items():
        if is_cloud_profile(p):
            return name, _swap_reason(cfg.profile, name, res, "no local model fits")

    return cfg.profile, None


def _swap_reason(old: str, new: str, res: SystemResources, why: str) -> str:
    gpu = f"{res.gpu_name} {res.vram_mb} MB VRAM" if res.has_gpu else "no GPU"
    return (
        f"contextual profile '{old}' can't run on this machine "
        f"({gpu}, {res.ram_mb} MB RAM); using '{new}' ({why})"
    )


def autoselect_profile(cfg, *, res: SystemResources | None = None) -> list[str]:
    """Mutate ``cfg.profile`` to a runnable choice; return log messages.

    No-op (empty list) when auto-select is disabled via ``cfg.auto_select`` or
    the ``WL_DICTATE_NO_AUTOSELECT`` env var, or when the configured profile
    already fits.
    """
    if not getattr(cfg, "auto_select", True):
        return []
    if os.environ.get(_AUTOSELECT_ENV) == "1":
        return []
    if res is None:
        res = detect_resources()
    chosen, reason = select_profile(cfg, res)
    if reason is None:
        return []
    cfg.profile = chosen
    return [reason]


# ── Reporting / CLI ───────────────────────────────────────────────────────────


def _report(cfg, res: SystemResources) -> str:
    lines = ["Hardware:"]
    if res.has_gpu:
        lines.append(f"  GPU : {res.gpu_name} ({res.vram_mb} MB VRAM)")
    else:
        lines.append("  GPU : none detected")
    lines.append(f"  RAM : {res.ram_mb} MB")
    cuda = cuda_available()
    lines.append(f"  CUDA: {'available' if cuda else 'not available'}")
    lines.append("")
    wm, _ = select_whisper_model(WHISPER_AUTO, res, cuda=cuda)
    lines.append(f"Transcription (Whisper) model 'auto' -> {wm}")
    lines.append("")
    lines.append("Local LLM (contextual) tiers:")
    for tier in LOCAL_MODEL_TIERS:
        gpu_ok = res.vram_mb >= tier.min_vram_mb
        cpu_ok = res.ram_mb >= tier.min_ram_mb
        if gpu_ok:
            verdict = "OK (GPU)"
        elif cpu_ok:
            verdict = "OK (CPU, slow)"
        else:
            verdict = "NO"
        lines.append(
            f"  {tier.key:<5} {tier.label:<18} "
            f"needs {tier.min_vram_mb} MB VRAM / {tier.min_ram_mb} MB RAM  -> {verdict}"
        )
    if cfg is not None:
        lines.append("")
        lines.append("Contextual profiles:")
        for name, p in cfg.profiles.items():
            kind = "cloud" if is_cloud_profile(p) else "local"
            verdict = "OK" if profile_fits(name, p, res) else "NO"
            marker = " *" if name == cfg.profile else "  "
            lines.append(f" {marker}{name:<12} {kind:<6} {p.model:<28} -> {verdict}")
        chosen, reason = select_profile(cfg, res)
        lines.append("")
        if reason:
            lines.append(f"Auto-select -> '{chosen}': {reason}")
        else:
            lines.append(f"Auto-select -> '{chosen}' (configured profile fits)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """`python -m wldictate.hardware [--pick-model|--json|--check-models]`."""
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="wldictate.hardware")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pick-model",
        action="store_true",
        help="print the -hf spec of the largest local model that fits (for the "
        "launch script); prints nothing and exits 1 if none fits",
    )
    group.add_argument(
        "--json", action="store_true", help="emit detected resources as JSON"
    )
    args = parser.parse_args(argv)

    res = detect_resources()

    if args.pick_model:
        tier = recommend_local_model(res)
        if tier is None:
            return 1
        print(tier.hf_spec)
        return 0

    if args.json:
        print(
            json.dumps(
                {
                    "vram_mb": res.vram_mb,
                    "ram_mb": res.ram_mb,
                    "gpu_name": res.gpu_name,
                    "has_gpu": res.has_gpu,
                }
            )
        )
        return 0

    # Default: the human report (also what `wl-dictate --check-models` prints).
    try:
        from .config import Config

        cfg = Config.load().contextual
    except Exception:
        cfg = None
    print(_report(cfg, res))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
