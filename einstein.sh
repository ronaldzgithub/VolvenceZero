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
# USAGE
#   ./einstein.sh [--mode smoke|real] [--device cpu|cuda|mps|auto] \
#                 [--require-verify 0|1|auto] [--run-id <id>] \
#                 [--skip-collect] [--skip-bake] [--skip-verify] [-h|--help]
#
# Two demo modes:
#
#   smoke (default, ~5 min, no GPU)
#       ./einstein.sh
#     tiny-gpt2 + synthetic corpus path. Proves the L1/L2/L3/L4 chain
#     wires end-to-end. Not a real-LoRA demo -- synthetic delta is zeroed
#     by Qwen LayerNorm (debt #40).
#
#   real (opt-in, ~30-45 min, GPU/MPS strongly recommended)
#       ./einstein.sh --mode real
#     Qwen/Qwen2.5-1.5B-Instruct + PEFT q/k/v/o on the Wave K curated
#     Einstein corpus. Produces the demo where the chat UI vertical
#     dropdown (einstein-raw / einstein-bundle / einstein-full) shows
#     distinct behaviour.
#
# Device selection:
#   --device cpu  / --device cuda  / --device mps  / --device auto
#   'auto' (default) -> smoke: cpu, real: probe torch (cuda -> mps -> cpu).
#   Apple Silicon Metal: ./einstein.sh --mode real --device mps
#
# Verification gate:
#   --require-verify 0  / --require-verify 1  / --require-verify auto
#   'auto' (default) -> smoke: 0, real: 1.
#   Smoke can never satisfy the gate offline (V1 metadata stubs land
#   NEEDS_REVIEW on 4 of 7 axes); real expects curator overrides.
#
# Phase skipping:
#   --skip-collect              (re-bake + re-verify only)
#   --skip-collect --skip-bake  (verify only)
#   --skip-verify               (collect + bake only)
#
# Less-used overrides keep their env-var fallback (consulted only when
# the matching CLI flag was NOT supplied). See:
#   SEEDS_FILE / METADATA_FILE / PROVENANCE_FILE / FIGURE_CONTEXT_FILE
#   CORPUS_ROOT / BUNDLE_ROOT / AUDIT_ROOT
#   MAX_PAGES / RATE_RPS / BURST
#   QWEN_MODEL_ID / PEFT_TARGET_MODULES / PEFT_RANK / PEFT_MAX_STEPS
#   RUNTIME_BACKEND / BUNDLE_ID / VERIFY_OUT / MAX_IN_CORPUS_QUESTIONS

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# --- argument parsing -------------------------------------------------------

MODE='smoke'
DEVICE='auto'
REQ_VERIFY='auto'
RUN_ID_ARG=''
SKIP_COLLECT=0
SKIP_BAKE=0
SKIP_VERIFY=0

usage() {
    sed -n '1,55p' "${BASH_SOURCE[0]}" | sed -n '/^# USAGE/,/^# Less-used/p'
}

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--mode)
            [ $# -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            MODE="$2"; shift 2
            ;;
        -d|--device)
            [ $# -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            DEVICE="$2"; shift 2
            ;;
        --require-verify)
            [ $# -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            REQ_VERIFY="$2"; shift 2
            ;;
        --run-id)
            [ $# -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            RUN_ID_ARG="$2"; shift 2
            ;;
        --skip-collect) SKIP_COLLECT=1; shift ;;
        --skip-bake)    SKIP_BAKE=1;    shift ;;
        --skip-verify)  SKIP_VERIFY=1;  shift ;;
        -h|--help)      usage; exit 0 ;;
        --)             shift; break ;;
        -*)
            echo "ERROR: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
        *)
            echo "ERROR: unexpected positional argument '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    smoke|real) ;;
    *) echo "ERROR: --mode must be 'smoke' or 'real', got '$MODE'." >&2; exit 2 ;;
esac
case "$DEVICE" in
    cpu|cuda|mps|auto) ;;
    *) echo "ERROR: --device must be 'cpu' / 'cuda' / 'mps' / 'auto', got '$DEVICE'." >&2; exit 2 ;;
esac
case "$REQ_VERIFY" in
    0|1|auto) ;;
    *) echo "ERROR: --require-verify must be '0' / '1' / 'auto', got '$REQ_VERIFY'." >&2; exit 2 ;;
esac

# --- mode-derived defaults --------------------------------------------------

# RUN_ID: --run-id wins; otherwise env var; otherwise hard-coded default.
if [ -n "$RUN_ID_ARG" ]; then
    export RUN_ID="$RUN_ID_ARG"
else
    export RUN_ID="${RUN_ID:-einstein-2026Q2}"
fi

# Probe torch for cuda > mps > cpu. Returns 'cpu' if torch is missing.
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

# Resolve --device 'auto': smoke -> cpu; real -> probe torch.
if [ "$DEVICE" = "auto" ]; then
    if [ "$MODE" = "real" ]; then
        DEVICE="$(_detect_peft_device)"
        echo "[device-detect] --device auto -> $DEVICE (probed via torch)"
    else
        DEVICE='cpu'
    fi
fi
export PEFT_DEVICE="$DEVICE"

# Resolve --require-verify 'auto': smoke -> 0; real -> 1.
# Rationale (smoke=0): offline V1 metadata stubs guarantee NEEDS_REVIEW
# on 4 of 7 axes; gating bake-bundle would always BLOCK. Smoke is a
# wiring check, not a curatorial verification.
# Rationale (real=1): real mode opts into the curator verification
# gate by default; the caller is expected to have arranged
# --metadata-mode=live + human review overrides (debt #26 closure
# path). Pass --require-verify 0 to opt out for now.
if [ "$REQ_VERIFY" = "auto" ]; then
    if [ "$MODE" = "real" ]; then
        REQ_VERIFY=1
    else
        REQ_VERIFY=0
    fi
fi
export REQUIRE_VERIFY="$REQ_VERIFY"

# Mode-derived defaults for Phase 2 / 3. Power users can still override
# any single axis via env var (consulted only when env var is set).
if [ "$MODE" = "real" ]; then
    export QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
    export PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
    export PEFT_MAX_STEPS="${PEFT_MAX_STEPS:-200}"
    export RUNTIME_BACKEND="${RUNTIME_BACKEND:-transformers}"
else
    export QWEN_MODEL_ID="${QWEN_MODEL_ID:-sshleifer/tiny-gpt2}"
    export PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES:-c_attn}"
    export PEFT_MAX_STEPS="${PEFT_MAX_STEPS:-50}"
    export RUNTIME_BACKEND="${RUNTIME_BACKEND:-synthetic}"
fi

# Path-like fallbacks (only set when env var was not already supplied).
export METADATA_FILE="${METADATA_FILE:-packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.curated_metadata.jsonl}"

echo "================================================================"
echo " Einstein figure-vertical pipeline (Unix driver)"
echo " mode          = $MODE"
echo " run_id        = $RUN_ID"
echo " peft_device   = $PEFT_DEVICE"
echo " qwen_model_id = $QWEN_MODEL_ID"
echo " runtime       = $RUNTIME_BACKEND"
echo " require_verify= $REQUIRE_VERIFY  (1=bake-bundle gates on verifier PASS)"
echo " skip          = collect=$SKIP_COLLECT bake=$SKIP_BAKE verify=$SKIP_VERIFY"
echo "================================================================"

# --- Phase 1 -- collect corpus + curated bundle ----------------------------

if [ "$SKIP_COLLECT" != "1" ]; then
    bash scripts/figure_collect_einstein.sh
else
    echo "[phase 1] SKIPPED (--skip-collect)"
fi

# --- Phase 2 -- persona LoRA bake ------------------------------------------

if [ "$SKIP_BAKE" != "1" ]; then
    bash scripts/figure_bake_einstein_persona_lora.sh
else
    echo "[phase 2] SKIPPED (--skip-bake)"
fi

# --- Phase 3 -- 4-gate verification harness --------------------------------
# Phase 3's underlying script can also bake; tell it not to, since
# Phase 2 already did (or was explicitly skipped). We override
# SKIP_BAKE for *Phase 3 only* because the upstream verify script
# reads it as "skip the inner persona-bake step", not as our
# top-level --skip-bake.
export SKIP_BAKE=1

if [ "$SKIP_VERIFY" != "1" ]; then
    bash scripts/figure_verify_einstein_persona.sh
else
    echo "[phase 3] SKIPPED (--skip-verify)"
fi

echo ""
echo "pipeline complete"
