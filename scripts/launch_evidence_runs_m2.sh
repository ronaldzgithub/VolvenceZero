#!/usr/bin/env bash
# Complete Apple-silicon (M1/M2/M3) launcher for the two evidence runs:
#
#   Run 1  learned-shadow soak       (500 synthetic turns, CPU, no keys needed)
#   Run 2  same-substrate ablation   (9-track Companion Bench P1, Qwen on MPS,
#                                     needs .local/llm.env with OPENROUTER key)
#
# Prefer the thin P1 mirror for a Windows-parity one-shot:
#   bash run_companion_bench_p1.sh
# This script remains the broader evidence suite (setup / soak / ablation / all).
#
# Usage (from the repo root, macOS; `bash` prefix avoids exec-bit issues after
# a Windows-side checkout):
#   bash scripts/launch_evidence_runs_m2.sh setup      # one-time: venv + wheels + torch(MPS)
#   bash scripts/launch_evidence_runs_m2.sh soak       # Run 1
#   bash scripts/launch_evidence_runs_m2.sh ablation   # Run 2
#   bash scripts/launch_evidence_runs_m2.sh all        # setup-if-needed + soak + ablation
#
# Environment knobs (all optional):
#   TURNS                    soak turn count            (default 500)
#   SOAK_OUTPUT_DIR          soak artifact dir          (default artifacts/learned_shadow_soak)
#   ARTIFACT_DIR             ablation artifact dir      (default artifacts/companion-ablation/<utc-tag>)
#   MAX_ATTEMPTS             ablation retry attempts    (default 3; retries pass --resume)
#   RESUME=1                 first ablation attempt already passes --resume
#   KEEP_SERVICES=1          leave ablation endpoints up after the run
#                            (ports 8000/8500/8501/8502/8600)
#   VZ_SUBSTRATE_MODEL_ID    default Qwen/Qwen2.5-1.5B-Instruct
#   VZ_SUBSTRATE_DEVICE      default mps (cpu as last resort; cuda refused on Mac)
#   VOLVENCE_VENV_DIR        default <repo>/.venv
#
# Prerequisites for `ablation` (fail-loud, checked by preflight):
#   * .local/llm.env copied from your main box (OPENROUTER_API_KEY +
#     ABLATION_USER_SIM_MODEL / ABLATION_PERTURN_MODEL / ABLATION_ARC_MODEL /
#     ABLATION_REFH_EXTRACTOR_MODEL). Never commit this file.
#   * network access to huggingface.co (first run downloads the Qwen substrate)
#     and openrouter.ai (judges / user-sim / extractors).
#
# What is honest about an M2 run:
#   * Run 1 is the synthetic lane by definition; verdicts are expected BLOCKED
#     on the real-trace gates. Directional evidence only.
#   * Run 2 on MPS uses byte-identical weights (weights_sha256 fingerprint gate
#     still enforced), so the causal claim structure is unchanged; only
#     throughput differs from a CUDA box. Expect hours, not minutes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VOLVENCE_VENV_DIR:-${REPO_ROOT}/.venv}"
TURNS="${TURNS:-500}"
SOAK_OUTPUT_DIR="${SOAK_OUTPUT_DIR:-artifacts/learned_shadow_soak}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
ABLATION_PORTS="8000 8500 8501 8502 8600"

log() { printf '[m2] %s\n' "$*"; }
die() { printf '[m2] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

require_macos_arm() {
  [[ "$(uname -s)" == "Darwin" ]] || die "this launcher targets macOS; on Linux/Windows use the existing launchers"
  [[ "$(uname -m)" == "arm64" ]] || die "expected Apple silicon (arm64), got $(uname -m)"
}

activate_venv() {
  [[ -x "${VENV_DIR}/bin/python" ]] || die "venv missing at ${VENV_DIR}; run: $0 setup"
  export PATH="${VENV_DIR}/bin:${PATH}"
  export VIRTUAL_ENV="${VENV_DIR}"
}

load_llm_env() {
  local env_file="${REPO_ROOT}/.local/llm.env"
  [[ -f "$env_file" ]] || die ".local/llm.env not found 鈥?copy it from your main machine (OPENROUTER_API_KEY + ABLATION_* models)"
  # setdefault semantics: never override variables already in the environment.
  while IFS= read -r line; do
    line="${line%%$'\r'}"
    [[ -z "$line" || "$line" == \#* || "$line" != *"="* ]] && continue
    local name="${line%%=*}" value="${line#*=}"
    name="$(echo "$name" | tr -d '[:space:]')"
    if [[ -z "${!name:-}" ]]; then
      export "${name}=${value}"
    fi
  done < "$env_file"
}

# Mirror run_p1_windows.ps1 Set-P1CrossFamilyExtractorEnv: derive the
# ref-harness / camel cross-family extractor config from the ABLATION_* vars.
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

ports_busy() {
  python - <<PY
import socket, sys
busy = []
for port in (${ABLATION_PORTS// /, }):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            busy.append(port)
print(" ".join(str(p) for p in busy))
sys.exit(1 if busy else 0)
PY
}

# ---------------------------------------------------------------------------
# setup 鈥?venv + all workspace wheels + torch with MPS
# ---------------------------------------------------------------------------

cmd_setup() {
  require_macos_arm

  local python_bin=""
  local candidate
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      python_bin="$candidate"
      break
    fi
  done
  [[ -n "$python_bin" ]] || die "Python >= 3.11 not found (brew install python@3.12)"
  log "using $($python_bin -c 'import sys; print(sys.executable, sys.version.split()[0])')"

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "creating venv: ${VENV_DIR}"
    "$python_bin" -m venv "$VENV_DIR"
  else
    log "reusing venv: ${VENV_DIR}"
  fi
  export PATH="${VENV_DIR}/bin:${PATH}"
  python -m pip install --upgrade pip setuptools wheel

  log "installing workspace wheels (VOLVENCE_EXTRAS=hf; macOS torch wheels ship MPS support)"
  PYTHON="${VENV_DIR}/bin/python" VOLVENCE_EXTRAS=hf bash ./install.sh

  # Benchmark-only wheel; not part of install.sh's core list but required for
  # the camel ablation track.
  log "installing companion-camel-baseline (ablation camel track)"
  python -m pip install -e packages/companion-camel-baseline

  log "verifying console scripts + MPS availability"
  local name
  for name in lifeform-serve companion-ref-harness companion-camel-baseline; do
    command -v "${VENV_DIR}/bin/${name}" >/dev/null 2>&1 || die "console script missing after install: ${name}"
  done
  python - <<'PY'
import torch
ok = torch.backends.mps.is_available()
print(f"torch {torch.__version__} mps_available={ok}")
if not ok:
    raise SystemExit(
        "torch.backends.mps.is_available() is false - expected true on Apple "
        "silicon; check the torch install (needs the official macOS arm64 wheel)."
    )
PY

  log "running brain kernel smoke turn"
  python - <<'PY'
from volvence_zero.brain import Brain, BrainConfig
session = Brain(BrainConfig()).create_session(session_id="m2-setup-smoke")
text = (session.run_turn("I need help making a careful decision.").response.text or "").strip()
if not text:
    raise SystemExit("smoke turn returned empty response")
print(text[:160])
PY

  log "setup complete. Next: $0 soak  /  $0 ablation"
}

# ---------------------------------------------------------------------------
# soak 鈥?Run 1 (synthetic lane, CPU)
# ---------------------------------------------------------------------------

cmd_soak() {
  require_macos_arm
  activate_venv
  log "Run 1: learned-shadow soak 鈥?${TURNS} turns -> ${SOAK_OUTPUT_DIR}"
  bash scripts/run_learned_shadow_soak.sh "$TURNS" "$SOAK_OUTPUT_DIR"
}

# ---------------------------------------------------------------------------
# ablation — Run 2 (9-track same-substrate Companion Bench P1 on MPS)
# Prefer `bash run_companion_bench_p1.sh` when you only need the thin P1 mirror.
# ---------------------------------------------------------------------------

cmd_ablation() {
  require_macos_arm
  activate_venv
  load_llm_env
  derive_extractor_env

  export VZ_SUBSTRATE_MODEL_ID="${VZ_SUBSTRATE_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
  export VZ_SUBSTRATE_DEVICE="${VZ_SUBSTRATE_DEVICE:-mps}"
  export LIFEFORM_LOCAL_API_KEY="${LIFEFORM_LOCAL_API_KEY:-local-ablation-key}"
  [[ "$VZ_SUBSTRATE_DEVICE" == "cuda"* ]] && die "VZ_SUBSTRATE_DEVICE=cuda on a Mac; use mps (default) or cpu"

  local busy
  if ! busy="$(ports_busy)"; then
    log "ablation ports already in use: ${busy}"
    log "an ablation stack (possibly an in-flight run) is up; stop it first:"
    log "  bash scripts/companion_bench/stop_same_substrate_ablation.sh <artifact-dir>/serve.pids"
    die "refusing to start: ports busy"
  fi

  local date_tag
  date_tag="$(date -u +%Y%m%dT%H%M%SZ)"
  export ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/companion-ablation/${date_tag}}"
  mkdir -p "$ARTIFACT_DIR"
  local transcript="${ARTIFACT_DIR}/launch_p1_m2_$(date +%Y%m%d_%H%M%S).log"

  local or_base="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
  local user_sim="${ABLATION_USER_SIM_MODEL:-openai/gpt-5-mini}"
  local perturn="${ABLATION_PERTURN_MODEL:-$user_sim}"
  local arc="${ABLATION_ARC_MODEL:-anthropic/claude-3.7-sonnet}"

  log "Run 2: same-substrate ablation P1"
  log "  substrate    = ${VZ_SUBSTRATE_MODEL_ID} on ${VZ_SUBSTRATE_DEVICE}"
  log "  judges       = user-sim=${user_sim} perturn=${perturn} arc=${arc} (all via OpenRouter)"
  log "  artifact_dir = ${ARTIFACT_DIR}"
  log "  transcript   = ${transcript}"

  # Everything below is teed into the transcript.
  {
    log "preflight (weights fingerprint + manifest + judge connectivity)..."
    python scripts/companion_bench/preflight_llm.py \
      --model-id "$VZ_SUBSTRATE_MODEL_ID" \
      --substrate-device "$VZ_SUBSTRATE_DEVICE" \
      --artifact-dir "$ARTIFACT_DIR" \
      ${VZ_SUBSTRATE_WEIGHTS_PATH:+--weights-path "$VZ_SUBSTRATE_WEIGHTS_PATH"}

    log "booting the 5 same-substrate endpoints..."
    bash scripts/companion_bench/serve_same_substrate_ablation.sh

    teardown() {
      if [[ "${KEEP_SERVICES:-0}" == "1" ]]; then
        log "KEEP_SERVICES=1 鈥?leaving endpoints up (PIDs in ${ARTIFACT_DIR}/serve.pids)"
        return 0
      fi
      if [[ -f "${ARTIFACT_DIR}/serve.pids" ]]; then
        bash scripts/companion_bench/stop_same_substrate_ablation.sh "${ARTIFACT_DIR}/serve.pids" || true
      fi
    }
    trap teardown EXIT

    local rc=1 attempt
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
      local resume_flag=()
      if [[ "${RESUME:-0}" == "1" || "$attempt" -gt 1 ]]; then
        resume_flag=(--resume)
      fi
      log "scoring attempt ${attempt}/${MAX_ATTEMPTS} (resume=${resume_flag[*]:-no})..."
      if python scripts/companion_bench/run_same_substrate_ablation.py \
          --phase p1 \
          --output-dir "$ARTIFACT_DIR" \
          --user-sim-base-url "$or_base" --user-sim-model "$user_sim" --user-sim-key-env OPENROUTER_API_KEY \
          --perturn-base-url "$or_base" --perturn-model "$perturn" --perturn-key-env OPENROUTER_API_KEY \
          --arc-base-url "$or_base" --arc-model "$arc" --arc-key-env OPENROUTER_API_KEY \
          ${resume_flag[@]+"${resume_flag[@]}"}; then
        rc=0
        break
      fi
      log "attempt ${attempt} failed"
      if [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
        log "retrying with --resume in 30s (finished track summaries are reused, not re-paid)..."
        sleep 30
      fi
    done

    if [[ "$rc" -ne 0 ]]; then
      log "P1 did not complete after ${MAX_ATTEMPTS} attempts."
      log "partial results kept in ${ARTIFACT_DIR}/scores; rerun with:"
      log "  ARTIFACT_DIR=\"${ARTIFACT_DIR}\" RESUME=1 $0 ablation"
      exit "$rc"
    fi

    log "verdict summary:"
    python - "${ARTIFACT_DIR}/verdict_p1.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    verdict = json.load(handle)
print("=" * 60)
print(f" state = {verdict['state']}  (P1 single seed => stability not claimable)")
print("-" * 60)
for claim in verdict["claims"]:
    print(f" {claim['claim_id']}: {claim['status']}")
    print(f"   {claim['detail']}")
print("-" * 60)
print(" tracks (final_mean):")
for track, score in verdict["tracks"].items():
    print(f"   {track:<16} {score}")
for rec in verdict.get("recommendations", []):
    print(f" recommendation: {rec}")
print("=" * 60)
PY
    log "verdict    : ${ARTIFACT_DIR}/verdict_p1.json"
  } 2>&1 | tee "$transcript"
}

# ---------------------------------------------------------------------------
# all 鈥?setup-if-needed + both runs
# ---------------------------------------------------------------------------

cmd_all() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    cmd_setup
  fi
  cmd_soak
  cmd_ablation
}

case "${1:-all}" in
  setup)    cmd_setup ;;
  soak)     cmd_soak ;;
  ablation) cmd_ablation ;;
  all)      cmd_all ;;
  -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown command '${1:-}' (expected setup | soak | ablation | all)" ;;
esac
