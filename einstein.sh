#!/usr/bin/env bash
# End-to-end Einstein figure-vertical demo driver (Unix / Mac / Git Bash / WSL).
#
# Composes the three existing Wave K-P shell scripts:
#   scripts/figure_collect_einstein.sh             (corpus + curated bundle)
#   scripts/figure_bake_einstein_persona_lora.sh   (PEFT persona LoRA)
#   scripts/figure_verify_einstein_persona.sh      (4-gate verification)
#
# Pure-Windows equivalent (no bash dependency): einstein.ps1
# This script is the Mac / Linux / Git-Bash / WSL twin; the two
# enforce identical CLI / Mode / Device / RequireVerify semantics.
#
# USAGE
#   ./einstein.sh [--mode smoke|real] [--device cpu|cuda|mps|auto] \
#                 [--require-verify 0|1|auto] [--run-id <id>] \
#                 [--skip-collect] [--skip-bake] [--skip-verify] [-h|--help]
#
# Two demo modes:
#
#   real (default, ~30-60 min on a 24 GB GPU, GPU strongly recommended)
#       ./einstein.sh
#     Qwen/Qwen2.5-7B-Instruct + PEFT q/k/v/o on the Wave K curated
#     Einstein corpus. Produces the demo where the chat UI vertical
#     dropdown (einstein-raw / einstein-bundle / einstein-full) shows
#     distinct behaviour. Edit QWEN_MODEL_ID below to swap to
#     Qwen2.5-1.5B-Instruct on tighter-VRAM hosts.
#
#   smoke (opt-in, ~5 min, no GPU)
#       ./einstein.sh --mode smoke
#     tiny-gpt2 + synthetic corpus path. Proves the L1/L2/L3/L4 chain
#     wires end-to-end. Not a real-LoRA demo -- synthetic delta is zeroed
#     by Qwen LayerNorm (debt #40).
#
# Device selection:
#   --device cpu  / --device cuda  / --device mps  / --device auto
#   'auto' (default) -> smoke: cpu, real: probe torch (cuda -> mps -> cpu).
#   Apple Silicon Metal: ./einstein.sh --mode real --device mps
#
# Verification gate:
#   --require-verify 0  / --require-verify 1  / --require-verify auto
#   'auto' (default) -> 0 for both modes (offline pilot seeds still
#   NEEDS_REVIEW on 4 of 7 axes). Pass --require-verify 1 once live
#   curator overrides are staged.
#
# Phase skipping:
#   --skip-collect              (re-bake + re-verify only)
#   --skip-collect --skip-bake  (verify only)
#   --skip-verify               (collect + bake only)
#
# Env-var overrides (path-y / mode-independent knobs only). NOT writable
# back by the script; only read. The Mode-driven knobs listed at the
# very end are deliberately NOT env-overridable -- see the in-script
# reset-env block for the rationale.
#   SEEDS_FILE / METADATA_FILE / PROVENANCE_FILE / FIGURE_CONTEXT_FILE
#   CORPUS_ROOT / BUNDLE_ROOT / AUDIT_ROOT
#   MAX_PAGES / RATE_RPS / BURST
#   PEFT_RANK / MAX_IN_CORPUS_QUESTIONS
#   BUNDLE_ID / ROLLBACK_EVIDENCE
#
# Mode-driven (NOT env-overridable; use --mode / --device / --require-verify):
#   QWEN_MODEL_ID, PEFT_TARGET_MODULES, PEFT_MAX_STEPS, PEFT_DEVICE,
#   RUNTIME_BACKEND, VERIFY_OUT, REQUIRE_VERIFY

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

resolve_project_python() {
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    echo "${ROOT_DIR}/.venv/bin/python"
    return 0
  fi
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"
    return 0
  fi
  echo "python"
}

has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

initialize_hf_download_env() {
  local model_id="$1"
  local hf_home="$2"
  hf_probe() {
    local endpoint="$1"
    local temp_home="$2"
    if [[ -n "$endpoint" ]]; then
      export HF_ENDPOINT="$endpoint"
    else
      unset HF_ENDPOINT
    fi
    export HF_HOME="$temp_home"
    mkdir -p "$temp_home"
    HF_PROBE_MODEL="$model_id" "$PYTHON_BIN" -c "
import os, sys, warnings
warnings.filterwarnings('ignore')
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(os.environ['HF_PROBE_MODEL'], 'config.json')
except Exception:
    sys.exit(1)
" >/dev/null 2>&1
  }

  if [[ -n "${VOLVENCE_FORCE_HF_ENDPOINT:-}" ]]; then
    export HF_HOME="$hf_home"
    if [[ -n "${HF_ENDPOINT:-}" ]]; then
      echo "[einstein] hf_endpoint=${HF_ENDPOINT} (forced)"
    else
      echo "[einstein] hf_endpoint=<default huggingface.co> (forced)"
    fi
    return 0
  fi

  local temp_home configured_endpoint
  configured_endpoint="${HF_ENDPOINT:-}"
  if [[ -z "$configured_endpoint" ]]; then
    temp_home="$(mktemp -d 2>/dev/null || mktemp -d -t vz_hf_probe)"
    if hf_probe "" "$temp_home"; then
      rm -rf "$temp_home"
      export HF_HOME="$hf_home"
      echo "[einstein] hf_endpoint=<default huggingface.co>"
      return 0
    fi
    rm -rf "$temp_home"
    export HF_HOME="$hf_home"
    echo "Cannot reach huggingface.co to download '${model_id}'." >&2
    exit 1
  fi

  echo "[einstein] probing HF_ENDPOINT=${configured_endpoint} ..."
  temp_home="$(mktemp -d 2>/dev/null || mktemp -d -t vz_hf_probe)"
  if hf_probe "$configured_endpoint" "$temp_home"; then
    rm -rf "$temp_home"
    export HF_HOME="$hf_home"
    export HF_ENDPOINT="$configured_endpoint"
    echo "[einstein] hf_endpoint=${configured_endpoint}"
    return 0
  fi
  rm -rf "$temp_home"
  echo "[einstein] WARN: HF_ENDPOINT=${configured_endpoint} failed huggingface_hub probe; falling back to huggingface.co" >&2
  unset HF_ENDPOINT
  temp_home="$(mktemp -d 2>/dev/null || mktemp -d -t vz_hf_probe)"
  if hf_probe "" "$temp_home"; then
    rm -rf "$temp_home"
    export HF_HOME="$hf_home"
    echo "[einstein] hf_endpoint=<default huggingface.co> (mirror fallback)"
    return 0
  fi
  rm -rf "$temp_home"
  export HF_HOME="$hf_home"
  echo "Cannot download '${model_id}' from HF_ENDPOINT or huggingface.co." >&2
  exit 1
}

test_einstein_preflight() {
  local mode="$1"
  if ! "$PYTHON_BIN" -c "import torch, transformers, peft" >/dev/null 2>&1; then
    cat >&2 <<EOF
Missing torch/transformers/peft in '${PYTHON_BIN}'.
Install: VOLVENCE_EXTRAS=full ./install.sh
EOF
    exit 1
  fi
  if [[ "$mode" == "real" ]] && has_nvidia_gpu; then
    if ! "$PYTHON_BIN" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
      cat >&2 <<EOF
Real mode on a GPU machine requires CUDA-enabled torch.
Install: ${PYTHON_BIN} -m pip install torch --index-url https://download.pytorch.org/whl/cu126
EOF
      exit 1
    fi
  fi
}

PYTHON_BIN="$(resolve_project_python)"
export PYTHON="$PYTHON_BIN"

# --- defend against historical pollution -----------------------------------
# Mirror einstein.ps1's reset-env block. An earlier version of this script
# (and its PowerShell twin) stamped Mode-derived defaults onto the process
# environment, so a shell session that ever ran the old code carries the
# residue indefinitely. Reading $QWEN_MODEL_ID / $RUNTIME_BACKEND from env
# would then silently override the --mode parameter, and a stale $VERIFY_OUT
# would pin every run to the very first dir the shell ever saw. Refusing to
# read them is the only robust fix; we unset + warn so deliberate operators
# learn to use the CLI flags instead.
_MODE_DRIVEN_ENV_VARS=(QWEN_MODEL_ID PEFT_TARGET_MODULES PEFT_MAX_STEPS
                       PEFT_DEVICE RUNTIME_BACKEND VERIFY_OUT REQUIRE_VERIFY)
for _n in "${_MODE_DRIVEN_ENV_VARS[@]}"; do
    if [ -n "${!_n:-}" ]; then
        printf '[reset-env] ignoring stale $%s=%q (Mode-driven; use --mode / --device / --require-verify instead)\n' \
            "$_n" "${!_n}" >&2
        unset "$_n"
    fi
done
unset _MODE_DRIVEN_ENV_VARS _n

# --- argument parsing -------------------------------------------------------

MODE='real'
DEVICE='auto'
REQ_VERIFY='auto'
RUN_ID_ARG=''
SKIP_COLLECT=0
SKIP_BAKE=0
SKIP_VERIFY=0

usage() {
    sed -n '/^# USAGE/,/^# Mode-driven/p' "${BASH_SOURCE[0]}"
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
# RUN_ID is mode-independent so env override is legitimate.
if [ -n "$RUN_ID_ARG" ]; then
    export RUN_ID="$RUN_ID_ARG"
else
    export RUN_ID="${RUN_ID:-einstein-2026Q2}"
fi

# Probe torch for cuda > mps > cpu. Returns 'cpu' if torch is missing.
_detect_peft_device() {
    "$PYTHON_BIN" - <<'PY' 2>/dev/null || echo cpu
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

# Resolve --device 'auto': smoke -> cpu; real + NVIDIA -> cuda; else probe torch.
if [ "$DEVICE" = "auto" ]; then
    if [ "$MODE" = "real" ]; then
        if has_nvidia_gpu; then
            DEVICE='cuda'
            echo "[device-detect] --device auto -> cuda (NVIDIA GPU present)"
        else
            DEVICE="$(_detect_peft_device)"
            echo "[device-detect] --device auto -> $DEVICE (probed via torch)"
        fi
    else
        DEVICE='cpu'
    fi
fi
export PEFT_DEVICE="$DEVICE"

# Resolve --require-verify 'auto': default 0 for both modes.
# Offline pilot metadata still lands NEEDS_REVIEW on 4 of 7 axes;
# gating bake-bundle would BLOCK without live curator overrides.
# Pass --require-verify 1 once --metadata-mode=live + review land.
if [ "$REQ_VERIFY" = "auto" ]; then
    REQ_VERIFY=0
fi
export REQUIRE_VERIFY="$REQ_VERIFY"

# Mode-driven defaults for Phase 2 / 3. These are LITERALS, deliberately
# NOT env-overridable -- see the reset-env block above for the rationale.
# A power user who wants something different should run with --mode flipped
# and/or edit this block. The four knobs (QWEN_MODEL_ID / PEFT_TARGET_MODULES
# / PEFT_MAX_STEPS / RUNTIME_BACKEND) are the ones that cause the most
# damage when stale env residue silently overrides Mode -- so they get
# the strictest treatment.
if [ "$MODE" = "real" ]; then
    # Default: Qwen2.5-7B-Instruct (bf16 ~15 GB VRAM; fits 24 GB
    # GPU like RTX 4090 / RTX 4090D / A6000 / L40). For tighter
    # VRAM hosts swap this literal to 'Qwen/Qwen2.5-1.5B-Instruct'.
    export QWEN_MODEL_ID='Qwen/Qwen2.5-7B-Instruct'
    export PEFT_TARGET_MODULES='q_proj,k_proj,v_proj,o_proj'
    export PEFT_MAX_STEPS='1000'
    export RUNTIME_BACKEND='transformers'
else
    export QWEN_MODEL_ID='sshleifer/tiny-gpt2'
    export PEFT_TARGET_MODULES='c_attn'
    export PEFT_MAX_STEPS='50'
    export RUNTIME_BACKEND='synthetic'
fi

# Always generate a fresh per-invocation VERIFY_OUT subdir. This is the
# directory the downstream verify script reads via $VERIFY_OUT; with
# no env-overridability we cannot pin multiple invocations to the same
# dir and overwrite tracked artifacts.
_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
export VERIFY_OUT="${ROOT_DIR}/artifacts/figure_verify/${RUN_ID}-${_TIMESTAMP}"
mkdir -p "$VERIFY_OUT"

if [[ -z "${HF_HOME+x}" ]]; then
  export HF_HOME="${ROOT_DIR}/.local/hf-cache"
fi
if [[ -n "${HF_HOME}" ]]; then
  mkdir -p "${HF_HOME}"
fi
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"

test_einstein_preflight "$MODE"
initialize_hf_download_env "$QWEN_MODEL_ID" "$HF_HOME"

# Path-like fallbacks (env override legitimate; mode-independent).
export METADATA_FILE="${METADATA_FILE:-packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.curated_metadata.jsonl}"

echo "================================================================"
echo " Einstein figure-vertical pipeline (Unix driver)"
echo " mode          = $MODE"
echo " run_id        = $RUN_ID"
echo " python        = $PYTHON_BIN"
echo " peft_device   = $PEFT_DEVICE"
echo " qwen_model_id = $QWEN_MODEL_ID"
echo " runtime       = $RUNTIME_BACKEND"
echo " verify_out    = $VERIFY_OUT"
echo " require_verify= $REQUIRE_VERIFY  (1=bake-bundle gates on verifier PASS)"
echo " skip          = collect=$SKIP_COLLECT bake=$SKIP_BAKE verify=$SKIP_VERIFY"
echo "================================================================"

# --- helpers ----------------------------------------------------------------

# Run a child phase script. Treat exit 2 (OFFLINE gate BLOCK) as a known
# fatal signal -- the script handled the BLOCK and wrote its audit row, so
# we abort the pipeline cleanly (phases 2/3 cannot run without a bundle /
# LoRA) instead of letting `set -e` produce a bare bash trace. Anything
# else non-zero is a real failure.
_run_phase_or_block() {
    local label="$1"; shift
    local advice_block="$1"; shift
    local rc=0
    set +e
    "$@"
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        return 0
    fi
    if [ $rc -eq 2 ]; then
        printf '%s' "$advice_block" >&2
        exit 2
    fi
    echo "[$label] failed with exit $rc" >&2
    exit $rc
}

# --- Phase 1 -- collect corpus + curated bundle ----------------------------

if [ "$SKIP_COLLECT" != "1" ]; then
    _phase1_block_advice=$(cat <<EOF

=================================================================
 phase 1: OFFLINE gate BLOCKED bundle compilation (exit 2).

 The audit row is at
   ${ROOT_DIR}/data/figure_audit/<timestamp>_BAKE_BUNDLE_einstein_*.json
 and carries the verifier failures that triggered the BLOCK.

 Most common cause: --metadata-mode=offline ships V1 stubs that
 mark 4 of 7 verifier axes as NEEDS_REVIEW; --require-verify=1
 then always BLOCKs unless live metadata + reviewed overrides are
 staged. This is by design (R10 OFFLINE gate).

 Phases 2 and 3 cannot run without a bundle. Pick one of:
EOF
)
    if [ "$REQUIRE_VERIFY" = "1" ]; then
        _phase1_block_advice+=$'\n'"   ./einstein.sh --mode $MODE --require-verify 0"
        _phase1_block_advice+=$'\n'"       Skip the curator verification gate; bundle is baked"
        _phase1_block_advice+=$'\n'"       with verifier NEEDS_REVIEW notes attached. L3 evidence"
        _phase1_block_advice+=$'\n'"       + L4 refusal demo signal is still live."
        _phase1_block_advice+=$'\n'
    fi
    _phase1_block_advice+=$'\n'"   ./einstein.sh"
    _phase1_block_advice+=$'\n'"       Smoke mode (default); --require-verify=0 auto, tiny-gpt2,"
    _phase1_block_advice+=$'\n'"       fully offline, ~30s on CPU. Wiring check, not curation."
    _phase1_block_advice+=$'\n'"================================================================="
    _phase1_block_advice+=$'\n'

    _run_phase_or_block 'phase 1' "$_phase1_block_advice" \
        bash scripts/figure_collect_einstein.sh
else
    echo "[phase 1] SKIPPED (--skip-collect)"
fi

# --- Phase 2 -- persona LoRA bake ------------------------------------------

if [ "$SKIP_BAKE" != "1" ]; then
    _phase2_block_advice=$(cat <<EOF

=================================================================
 phase 2: OFFLINE gate BLOCKED persona LoRA artifact (exit 2).
 The audit row records the block_reason; the source bundle is
 unchanged.

 Phase 3 cannot verify a non-existent LoRA-bearing bundle.
 Re-run with --skip-bake to verify the base bundle's L4 refusal
 + L3 evidence signal against raw (BUNDLE_LORA condition will
 fall through to BUNDLE since no LoRA is registered):

   ./einstein.sh --mode $MODE --skip-collect --skip-bake
=================================================================

EOF
)
    _run_phase_or_block 'phase 2' "$_phase2_block_advice" \
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

# Phase 3 exit codes:
#   0 = all 4 gates PASS
#   2 = harness completed; one or more gates FAIL (verdict.json still written)
#   3 = setup error
# Exit 2 is informative, not fatal -- we let it surface as the script's
# overall exit code (matching einstein.ps1) so CI can distinguish "verdict
# rendered, some gates failed" from "real crash".
if [ "$SKIP_VERIFY" != "1" ]; then
    set +e
    bash scripts/figure_verify_einstein_persona.sh
    _phase3_rc=$?
    set -e
    if [ $_phase3_rc -ne 0 ] && [ $_phase3_rc -ne 2 ]; then
        echo "[phase 3] failed with exit $_phase3_rc" >&2
        exit $_phase3_rc
    fi
else
    echo "[phase 3] SKIPPED (--skip-verify)"
    _phase3_rc=0
fi

echo ""
echo "pipeline complete"
exit "${_phase3_rc:-0}"
