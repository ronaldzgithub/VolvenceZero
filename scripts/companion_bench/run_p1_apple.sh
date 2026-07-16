#!/usr/bin/env bash
# Apple-silicon / macOS SSOT orchestrator for CompanionBench P1 same-substrate
# directional ablation. Mirrors scripts/companion_bench/run_p1_windows.ps1.
#
# Pipeline:
#   1. load .local/llm.env + derive cross-family extractor env
#   2. default MPS substrate + ACTIVE torch learned backends
#   3. preflight_llm.py (weights / bootstrap / MPS / judge connectivity)
#   4. serve_same_substrate_ablation.sh (9-track topology)
#   5. run_same_substrate_ablation.py --phase p1
#   6. teardown serve.pids unless --keep-services
#
# Defaults (override via env or CLI flags):
#   VZ_SUBSTRATE_MODEL_ID      Qwen/Qwen2.5-1.5B-Instruct
#   VZ_SUBSTRATE_DEVICE        mps   (cpu allowed; cuda refused on Darwin)
#   VZ_TEMPORAL_SSL_BACKEND    active
#   VZ_TEMPORAL_RUNTIME_BACKEND active
#   VZ_INTERNAL_RL_BACKEND     active
#   VZ_CMS_TORCH_BACKEND       active
#   VZ_TORCH_BACKENDS          active  (legacy visibility; per-owner wins)
#   VZ_P1_VERTICAL_PROBE_TIMEOUT_S  180 (MPS cold-start route probe budget)
#   VZ_P1_SUT_MAX_TOKENS       96 on MPS, 256 on CPU (scoring response cap)
#   VZ_P1_MIN_FREE_DISK_GIB    15    (artifact + macOS swap safety floor)
#
# Usage:
#   bash scripts/companion_bench/run_p1_apple.sh
#   bash scripts/companion_bench/run_p1_apple.sh --dry-run
#   bash scripts/companion_bench/run_p1_apple.sh --resume --artifact-dir artifacts/companion-ablation/<tag>
#   bash scripts/companion_bench/run_p1_apple.sh --keep-services

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ARTIFACT_DIR=""
DRY_RUN=0
RESUME=0
KEEP_SERVICES=0

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      [[ $# -ge 2 ]] || { echo "error: --artifact-dir requires a path" >&2; exit 2; }
      ARTIFACT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --keep-services)
      KEEP_SERVICES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

die() {
  printf '[p1] ERROR: %s\n' "$*" >&2
  exit 2
}

log() {
  printf '[p1] %s\n' "$*"
}

require_darwin() {
  [[ "$(uname -s)" == "Darwin" ]] || die "run_p1_apple.sh targets macOS; on Windows use run_p1_windows.ps1"
}

require_free_disk() {
  local minimum_gib="${VZ_P1_MIN_FREE_DISK_GIB:-15}"
  if [[ ! "$minimum_gib" =~ ^[1-9][0-9]*$ ]]; then
    die "VZ_P1_MIN_FREE_DISK_GIB must be a positive integer, got '${minimum_gib}'"
  fi
  local available_kib
  available_kib="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 { print $4 }')"
  if [[ ! "$available_kib" =~ ^[0-9]+$ ]]; then
    die "could not determine free disk space for ${REPO_ROOT}"
  fi
  local required_kib=$((minimum_gib * 1024 * 1024))
  local available_gib=$((available_kib / 1024 / 1024))
  if (( available_kib < required_kib )); then
    die "only ${available_gib} GiB disk free; P1 requires at least ${minimum_gib} GiB for artifacts and macOS swap"
  fi
  log "disk preflight: ${available_gib} GiB free (minimum ${minimum_gib} GiB)"
}

load_llm_env() {
  local env_file="${REPO_ROOT}/.local/llm.env"
  [[ -f "$env_file" ]] || die ".local/llm.env not found — need OPENROUTER_API_KEY + ABLATION_* models"
  # setdefault: never override variables already in the environment
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%$'\r'}"
    [[ -z "$line" || "$line" == \#* || "$line" != *"="* ]] && continue
    local name="${line%%=*}" value="${line#*=}"
    name="$(printf '%s' "$name" | tr -d '[:space:]')"
    [[ -z "$name" ]] && continue
    if [[ -z "${!name:-}" ]]; then
      export "${name}=${value}"
    fi
  done < "$env_file"
}

# Mirror run_p1_windows.ps1 Set-P1CrossFamilyExtractorEnv
derive_extractor_env() {
  local or_base="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
  if [[ -z "${REFH_EXTRACTOR_MODEL:-}" ]]; then
    local refh_model="${ABLATION_REFH_EXTRACTOR_MODEL:-${ABLATION_PERTURN_MODEL:-}}"
    if [[ -n "$refh_model" ]]; then
      export REFH_EXTRACTOR_MODEL="$refh_model"
      export REFH_EXTRACTOR_BASE_URL="$or_base"
      export REFH_EXTRACTOR_KEY_ENV="OPENROUTER_API_KEY"
    fi
  fi
  if [[ -z "${CAMEL_COMPACTION_MODEL:-}" ]]; then
    local camel_model="${ABLATION_CAMEL_COMPACTION_MODEL:-${ABLATION_REFH_EXTRACTOR_MODEL:-${ABLATION_PERTURN_MODEL:-}}}"
    if [[ -n "$camel_model" ]]; then
      export CAMEL_COMPACTION_MODEL="$camel_model"
      export CAMEL_COMPACTION_BASE_URL="$or_base"
      export CAMEL_COMPACTION_KEY_ENV="OPENROUTER_API_KEY"
    fi
  fi
}

# Activate the main torch / learned backends. Existing explicit per-owner
# env wins (setdefault). VZ_TORCH_BACKENDS is also set for legacy visibility
# when unset; per-owner values still take precedence in resolve_final_rollout_config.
activate_torch_backends() {
  export VZ_TEMPORAL_SSL_BACKEND="${VZ_TEMPORAL_SSL_BACKEND:-active}"
  export VZ_TEMPORAL_RUNTIME_BACKEND="${VZ_TEMPORAL_RUNTIME_BACKEND:-active}"
  export VZ_INTERNAL_RL_BACKEND="${VZ_INTERNAL_RL_BACKEND:-active}"
  export VZ_CMS_TORCH_BACKEND="${VZ_CMS_TORCH_BACKEND:-active}"
  export VZ_TORCH_BACKENDS="${VZ_TORCH_BACKENDS:-active}"
  log "torch backends: ssl=${VZ_TEMPORAL_SSL_BACKEND} runtime=${VZ_TEMPORAL_RUNTIME_BACKEND} rl=${VZ_INTERNAL_RL_BACKEND} cms=${VZ_CMS_TORCH_BACKEND} (legacy VZ_TORCH_BACKENDS=${VZ_TORCH_BACKENDS})"
}

require_darwin
if [[ "$DRY_RUN" -eq 0 ]]; then
  require_free_disk
fi
load_llm_env
derive_extractor_env
activate_torch_backends

export VZ_SUBSTRATE_MODEL_ID="${VZ_SUBSTRATE_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
export VZ_SUBSTRATE_DEVICE="${VZ_SUBSTRATE_DEVICE:-mps}"
export LIFEFORM_LOCAL_API_KEY="${LIFEFORM_LOCAL_API_KEY:-local-ablation-key}"

if [[ "$VZ_SUBSTRATE_DEVICE" == cuda* ]]; then
  die "VZ_SUBSTRATE_DEVICE=cuda on macOS; use mps (default) or cpu"
fi
if [[ "$VZ_SUBSTRATE_DEVICE" == mps* ]]; then
  export VZ_P1_SUT_MAX_TOKENS="${VZ_P1_SUT_MAX_TOKENS:-96}"
  export VZ_P1_PER_SYSTEM_TIMEOUT_MIN="${VZ_P1_PER_SYSTEM_TIMEOUT_MIN:-480}"
else
  export VZ_P1_SUT_MAX_TOKENS="${VZ_P1_SUT_MAX_TOKENS:-256}"
  export VZ_P1_PER_SYSTEM_TIMEOUT_MIN="${VZ_P1_PER_SYSTEM_TIMEOUT_MIN:-240}"
fi
if [[ ! "$VZ_P1_SUT_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  die "VZ_P1_SUT_MAX_TOKENS must be a positive integer, got '${VZ_P1_SUT_MAX_TOKENS}'"
fi

if [[ -z "$ARTIFACT_DIR" ]]; then
  ARTIFACT_DIR="artifacts/companion-ablation/$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ "$ARTIFACT_DIR" != /* ]]; then
  ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_DIR}"
fi
export ARTIFACT_DIR
mkdir -p "$ARTIFACT_DIR"

OPENROUTER_BASE="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
USER_SIM_MODEL="${ABLATION_USER_SIM_MODEL:-openai/gpt-5-mini}"
PERTURN_MODEL="${ABLATION_PERTURN_MODEL:-$USER_SIM_MODEL}"
ARC_MODEL="${ABLATION_ARC_MODEL:-anthropic/claude-3.7-sonnet}"

RUNNER_ARGS=(
  scripts/companion_bench/run_same_substrate_ablation.py
  --phase p1
  --output-dir "$ARTIFACT_DIR"
  --vertical-probe-timeout-s "${VZ_P1_VERTICAL_PROBE_TIMEOUT_S:-180}"
  --per-system-timeout-min "${VZ_P1_PER_SYSTEM_TIMEOUT_MIN}"
  --sut-max-tokens "$VZ_P1_SUT_MAX_TOKENS"
  --user-sim-base-url "$OPENROUTER_BASE"
  --user-sim-model "$USER_SIM_MODEL"
  --user-sim-key-env OPENROUTER_API_KEY
  --perturn-base-url "$OPENROUTER_BASE"
  --perturn-model "$PERTURN_MODEL"
  --perturn-key-env OPENROUTER_API_KEY
  --arc-base-url "$OPENROUTER_BASE"
  --arc-model "$ARC_MODEL"
  --arc-key-env OPENROUTER_API_KEY
)
[[ "$RESUME" -eq 1 ]] && RUNNER_ARGS+=(--resume)

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "dry-run only; no service, MPS, or paid API call will start"
  RUNNER_ARGS+=(--dry-run)
  python "${RUNNER_ARGS[@]}"
  exit $?
fi

log "substrate=${VZ_SUBSTRATE_MODEL_ID} device=${VZ_SUBSTRATE_DEVICE}"
log "artifact-dir=${ARTIFACT_DIR}"
log "judges: user-sim=${USER_SIM_MODEL} perturn=${PERTURN_MODEL} arc=${ARC_MODEL}"

PREFLIGHT_ARGS=(
  scripts/companion_bench/preflight_llm.py
  --model-id "$VZ_SUBSTRATE_MODEL_ID"
  --substrate-device "$VZ_SUBSTRATE_DEVICE"
  --artifact-dir "$ARTIFACT_DIR"
)
if [[ -n "${VZ_SUBSTRATE_WEIGHTS_PATH:-}" ]]; then
  PREFLIGHT_ARGS+=(--weights-path "$VZ_SUBSTRATE_WEIGHTS_PATH")
fi

PID_FILE="${ARTIFACT_DIR}/serve.pids"

stop_stale_ablation_services() {
  if [[ -f "$PID_FILE" ]]; then
    bash scripts/companion_bench/stop_same_substrate_ablation.sh "$PID_FILE" || true
    log "stopped stale ablation services from ${PID_FILE}"
    sleep 1
  fi
}

clear_occupied_p1_ports() {
  local freed=""
  if ! freed="$(python - <<'PY'
import sys

sys.path.insert(0, "scripts/companion_bench")
from p1_readiness import P1ReadinessError, free_p1_ports, require_ports_free

killed = free_p1_ports()
if killed:
    print(",".join(str(pid) for pid in killed), end="")
try:
    require_ports_free()
except P1ReadinessError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(2) from exc
PY
)"; then
    die "could not free P1 ports; see error above"
  fi
  if [[ -n "$freed" ]]; then
    log "freed P1 ports (8000/8500/8501/8502/8600) by killing pid(s): ${freed}"
  fi
}

run_with_wake_lock() {
  if command -v caffeinate >/dev/null 2>&1; then
    log "sleep lock: caffeinate -dimsu (keep lid open; clamshell sleep is not blocked)"
    caffeinate -dimsu "$@"
  else
    "$@"
  fi
}

cleanup_services() {
  if [[ "$KEEP_SERVICES" -eq 1 ]]; then
    log "KEEP_SERVICES=1 — leaving endpoints up (PIDs in ${PID_FILE})"
    return 0
  fi
  if [[ -f "$PID_FILE" ]]; then
    bash scripts/companion_bench/stop_same_substrate_ablation.sh "$PID_FILE" || true
    log "stopped services from ${PID_FILE}"
  fi
}
trap cleanup_services EXIT

stop_stale_ablation_services
clear_occupied_p1_ports
python "${PREFLIGHT_ARGS[@]}"
log "booting same-substrate endpoints (ablation-bundle + ref-harness + memory-only + rag + camel)..."
run_with_wake_lock bash scripts/companion_bench/serve_same_substrate_ablation.sh

run_with_wake_lock python "${RUNNER_ARGS[@]}"
log "verdict: ${ARTIFACT_DIR}/verdict_p1.json"
log "progress: bash scripts/companion_bench/watch_p1_progress.sh ${ARTIFACT_DIR} --follow"
