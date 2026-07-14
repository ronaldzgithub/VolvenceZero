#!/usr/bin/env bash
# Build and evaluate learned-backend promotion evidence.
#
# Thin root wrapper around:
#   scripts/build_learned_promotion_evidence.py
#   scripts/evaluate_learned_backend_promotion.py
#
# Usage:
#   bash run_learned_promotion_evidence.sh --soak-artifact artifacts/.../learned_shadow_soak.json
#   bash run_learned_promotion_evidence.sh --soak-artifact artifacts/.../learned_shadow_soak.json --ablation-verdict artifacts/.../verdict_p1.json
#   OUTPUT_DIR=artifacts/my_promotion bash run_learned_promotion_evidence.sh --soak-artifact artifacts/.../learned_shadow_soak.json
#
# All arguments are forwarded to the build step. The resulting evidence artifact
# is then evaluated into ${OUTPUT_DIR}/promotion_report.json unless --output
# points elsewhere, in which case the report is written next to that file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/learned_backend_promotion}"
EVIDENCE_PATH="${OUTPUT_DIR}/promotion_evidence.json"
BUILD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || { echo "error: --output requires a path" >&2; exit 2; }
      EVIDENCE_PATH="$2"
      shift 2
      ;;
    -h|--help)
      "$PYTHON_BIN" scripts/build_learned_promotion_evidence.py --help
      exit 0
      ;;
    *)
      BUILD_ARGS+=("$1")
      shift
      ;;
  esac
done

REPORT_PATH="$(dirname "$EVIDENCE_PATH")/promotion_report.json"

"$PYTHON_BIN" scripts/build_learned_promotion_evidence.py \
  ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} \
  --output "$EVIDENCE_PATH"

"$PYTHON_BIN" scripts/evaluate_learned_backend_promotion.py \
  --artifact "$EVIDENCE_PATH" \
  --output "$REPORT_PATH"

echo "[learned-promotion] evidence: ${EVIDENCE_PATH}"
echo "[learned-promotion] report:   ${REPORT_PATH}"
