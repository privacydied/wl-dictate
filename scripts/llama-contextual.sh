#!/usr/bin/env bash
# llama-server for wl-dictate contextual dictation (the default "local"
# profile). Model + MTP settings come from a TOML config:
#   1. ~/.config/wl-dictate/llama.toml   (user copy — wins)
#   2. scripts/llama.toml                (repo defaults)
# Environment variables (PORT, CTX_SIZE, MTP_DRAFT_MAX, ...) override both.
#
# The default Qwen3.5-9B ships MTP weights (fast speculative decoding) and a
# vision projector (screenshot context). Watch "draft acceptance" in the log
# to tune [mtp].draft_max.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOML="${LLAMA_TOML:-$HOME/.config/wl-dictate/llama.toml}"
[ -f "$TOML" ] || TOML="$SCRIPT_DIR/llama.toml"

# Read TOML -> shell assignments (tomllib is stdlib in python3.11+).
eval "$(python3 - "$TOML" <<'PY'
import sys, tomllib
try:
    with open(sys.argv[1], "rb") as f:
        cfg = tomllib.load(f)
except Exception:
    cfg = {}
server = cfg.get("server", {})
mtp = cfg.get("mtp", {})
def emit(name, value):
    print(f'TOML_{name}="{value}"')
emit("MODEL", server.get("model", "unsloth/Qwen3.5-9B-MTP-GGUF:Q4_K_M"))
emit("ALIAS", server.get("alias", "qwen3.5-9b"))
emit("PORT", server.get("port", 8890))
emit("CTX", server.get("ctx_size", 16384))
emit("MTP_ENABLED", 1 if mtp.get("enabled", True) else 0)
emit("MTP_MAX", mtp.get("draft_max", 11))
emit("MTP_MIN", mtp.get("draft_min", 0))
PY
)"

MODEL="${MODEL:-$TOML_MODEL}"
ALIAS="${ALIAS:-$TOML_ALIAS}"
PORT="${PORT:-$TOML_PORT}"
CTX_SIZE="${CTX_SIZE:-$TOML_CTX}"
MTP_ENABLED="${MTP_ENABLED:-$TOML_MTP_ENABLED}"
MTP_DRAFT_MAX="${MTP_DRAFT_MAX:-$TOML_MTP_MAX}"
MTP_DRAFT_MIN="${MTP_DRAFT_MIN:-$TOML_MTP_MIN}"

export HF_HOME="${HF_HOME:-/mnt/SSD2/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
# Batch-1 generation is faster with MMQ (no fp16 dequant); cuBLAS only wins
# on big prompt batches.
export GGML_CUDA_FORCE_MMQ=1

MTP_ARGS=()
if [ "$MTP_ENABLED" = "1" ]; then
  MTP_ARGS=(
    --spec-type draft-mtp
    --spec-draft-n-max "$MTP_DRAFT_MAX"
    --spec-draft-n-min "$MTP_DRAFT_MIN"
  )
fi

echo "llama-contextual: model=$MODEL alias=$ALIAS port=$PORT ctx=$CTX_SIZE mtp=$MTP_ENABLED (config: $TOML)" >&2

exec llama-server \
  -hf "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --alias "$ALIAS" \
  -ngl all \
  --ctx-size "$CTX_SIZE" \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --kv-offload \
  --jinja \
  --cache-prompt \
  --cache-reuse 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 1 --top-p 0.8 --top-k 20 --min-p 0.0 \
  --metrics \
  "${MTP_ARGS[@]}"
