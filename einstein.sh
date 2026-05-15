#!/usr/bin/env bash
# End-to-end Einstein figure-vertical demo driver (Unix / Git Bash / WSL).
#
# Composes the three existing Wave K-P shell scripts:
#   scripts/figure_collect_einstein.sh             (corpus + curated bundle)
#   scripts/figure_bake_einstein_persona_lora.sh   (PEFT persona LoRA)
#   scripts/figure_verify_einstein_persona.sh      (4-gate verification)
#
# Pure-Windows equivalent (no bash dependency): einstein.ps1
#
# Two demo modes:
#
#   smoke (default, ~5 min, no GPU)
#       DEMO_MODE=smoke ./einstein.sh
#     tiny-gpt2 + synthetic corpus path. Proves the L1/L2/L3/L4 chain
#     wires end-to-end. Not a real-LoRA demo — synthetic delta is zeroed
#     by Qwen LayerNorm (debt #40).
#
#   real (opt-in, ~30-45 min, GPU strongly recommended)
#       DEMO_MODE=real ./einstein.sh
#     Qwen/Qwen2.5-1.5B-Instruct + PEFT q/k/v/o on the Wave K curated
#     Einstein corpus (444 chunks). Produces the demo where the chat UI
#     vertical dropdown (einstein-raw / einstein-bundle / einstein-full)
#     shows distinct behaviour.
#     PEFT_DEVICE is auto-detected (cuda → mps → cpu) when not pinned.
#     Override explicitly with e.g. PEFT_DEVICE=cuda or PEFT_DEVICE=mps.
#
# Phase skipping (re-run a single phase without redoing earlier ones):
#   SKIP_COLLECT=1 ./einstein.sh         # bake + verify only
#   SKIP_COLLECT=1 SKIP_BAKE=1 ./einstein.sh   # verify only
#   SKIP_VERIFY=1 ./einstein.sh          # collect + bake only

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEMO_MODE="${DEMO_MODE:-smoke}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_BAKE="${SKIP_BAKE:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

case "$DEMO_MODE" in
    smoke|real) ;;
    *) echo "DEMO_MODE must be 'smoke' or 'real', got '$DEMO_MODE'." >&2; exit 1 ;;
esac

# Phase 1 — collect corpus + curated bundle.
if [ "$DEMO_MODE" = "real" ]; then
    export REQUIRE_VERIFY="${REQUIRE_VERIFY:-1}"
else
    export REQUIRE_VERIFY="${REQUIRE_VERIFY:-1}"
fi
export METADATA_FILE="${METADATA_FILE:-packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.curated_metadata.jsonl}"

if [ "$SKIP_COLLECT" != "1" ]; then
    bash scripts/figure_collect_einstein.sh
else
    echo "[phase 1] SKIPPED (SKIP_COLLECT=1)"
fi

# Auto-detect the best available torch device when caller did not pin one.
# Probe order: cuda → mps (Apple Silicon) → cpu. Falls back to cpu silently
# if torch is not importable so smoke mode (which never reaches torch) still
# works on a clean env.
_detect_peft_device() {
    python - <<'PY' 2>/dev/null || echo cpu
try:
    import torch
    if torch.cuda.is_available():
        print("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
except Exception:
    print("cpu")
PY
}

# Phase 2 — persona LoRA bake.
if [ "$DEMO_MODE" = "real" ]; then
    export QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
    export PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
    export PEFT_MAX_STEPS="${PEFT_MAX_STEPS:-200}"
    if [ -z "${PEFT_DEVICE:-}" ]; then
        PEFT_DEVICE="$(_detect_peft_device)"
        echo "[phase 2] PEFT_DEVICE auto-detected: ${PEFT_DEVICE}"
    fi
    export PEFT_DEVICE
else
    export QWEN_MODEL_ID="${QWEN_MODEL_ID:-sshleifer/tiny-gpt2}"
    export PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES:-c_attn}"
    export PEFT_MAX_STEPS="${PEFT_MAX_STEPS:-50}"
    export PEFT_DEVICE="${PEFT_DEVICE:-cpu}"
fi

if [ "$SKIP_BAKE" != "1" ]; then
    bash scripts/figure_bake_einstein_persona_lora.sh
else
    echo "[phase 2] SKIPPED (SKIP_BAKE=1)"
fi

# Phase 3 — 4-gate verification harness.
if [ "$DEMO_MODE" = "real" ]; then
    export RUNTIME_BACKEND="${RUNTIME_BACKEND:-transformers}"
else
    export RUNTIME_BACKEND="${RUNTIME_BACKEND:-synthetic}"
fi
export SKIP_BAKE="${SKIP_BAKE:-1}"  # phase 3 should not re-bake; phase 2 did it.

if [ "$SKIP_VERIFY" != "1" ]; then
    bash scripts/figure_verify_einstein_persona.sh
else
    echo "[phase 3] SKIPPED (SKIP_VERIFY=1)"
fi

echo ""
echo "pipeline complete"
