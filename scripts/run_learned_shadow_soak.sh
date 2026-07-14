#!/usr/bin/env bash
# W3 (intent-alignment remediation) 鈥?GPU-server launcher for the
# learned-shadow long soak.
#
# Thin wrapper around scripts/run_learned_shadow_soak.py that:
#   * forces unbuffered output (progress lines survive nohup/redirect),
#   * tees stdout to a timestamped log next to the artifact,
#   * prints the learned_active_gate verdict summary from the artifact
#     when the run completes.
#
# Usage:
#   ./scripts/run_learned_shadow_soak.sh [TURNS] [OUTPUT_DIR]
#
# Defaults:
#   TURNS      = 500
#   OUTPUT_DIR = artifacts/learned_shadow_soak
#
# Typical overnight invocation on the GPU server:
#   nohup ./scripts/run_learned_shadow_soak.sh 500 &
#
# Notes:
#   * This is the SYNTHETIC lane: verdicts are expected BLOCKED on the
#     real-trace gates (real_trace_turns=0 by definition). The artifact is
#     directional evidence only; ACTIVE promotion needs the real-trace lane.
#   * CPU-only is fine; a GPU merely speeds up the substrate forward pass.
#
# Exit codes: mirrors the python soak (0 = artifact + manifest written and
# all in-run verifications passed; non-zero = fail loudly, see the log).

set -euo pipefail

TURNS="${1:-500}"
OUTPUT_DIR="${2:-artifacts/learned_shadow_soak}"

mkdir -p "${OUTPUT_DIR}"
LOG_PATH="${OUTPUT_DIR}/soak_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo " learned-shadow soak"
echo " turns      = ${TURNS}"
echo " output_dir = ${OUTPUT_DIR}"
echo " log        = ${LOG_PATH}"
echo "============================================================"

python -u scripts/run_learned_shadow_soak.py \
  --turns "${TURNS}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

echo ""
echo "[soak] learned_active_gate verdict summary:"
python - "${OUTPUT_DIR}/learned_shadow_soak.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
gate = payload["learned_active_gate"]
print(f"  note: {gate['note']}")
print(f"  latency_slo_ok: {gate['latency_slo_ok']}")
for verdict in gate["verdicts"]:
    missing = ", ".join(verdict["missing_gates"]) or "-"
    print(
        f"  {verdict['component']}: eligible={verdict['eligible']} "
        f"missing=[{missing}]"
    )
PY
