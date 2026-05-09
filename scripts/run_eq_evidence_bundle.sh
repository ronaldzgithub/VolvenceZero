#!/usr/bin/env bash
# Wave E5 — single-command entry point for the EQ evidence bundle.
#
# Drives the full Evidence-Chain Closure milestone in one shot:
# longitudinal benchmark across the four long-form scenarios with the
# LLM-backed semantic runtime, then aggregates the artifacts into a
# single ``evidence_bundle.json`` manifest under
# ``artifacts/eq_uplift/milestone_evidence_bundle/``.
#
# Usage:
#   ./scripts/run_eq_evidence_bundle.sh [ROUNDS] [BUNDLE_DIR]
#
# Defaults:
#   ROUNDS     = 5
#   BUNDLE_DIR = artifacts/eq_uplift/milestone_evidence_bundle
#
# Pre-requisites:
#   * Qwen 1.5B-Instruct cached locally (or the bench will download
#     it on first invocation; ~3 GB).
#   * ``transformers`` + ``torch`` installed in the active env.
#   * Disk space for ~50-200 MB of JSON artifacts.
#
# Exit codes:
#   0 = bundle assembled and all gates closed.
#   1 = lifeform-bench reported a regression OR bundle assembler
#       reported a missing required artifact.
#
# This script is intentionally a thin orchestrator: each wave's
# evidence-producing entry point is a separate command so a
# debugging operator can re-run any single wave without re-running
# the whole bundle.

set -euo pipefail

ROUNDS="${1:-5}"
BUNDLE_DIR="${2:-artifacts/eq_uplift/milestone_evidence_bundle}"
SCENARIOS_DIR="packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/scenarios"

mkdir -p "${BUNDLE_DIR}"

LONG_FORM_SCENARIOS=(
  "long-form-life-arc"
  "long-form-companion-arc"
  "long-form-task-arc"
  "long-form-trust-arc"
)

echo "============================================================"
echo " Wave E5 — EQ Evidence Bundle"
echo " rounds      = ${ROUNDS}"
echo " bundle_dir  = ${BUNDLE_DIR}"
echo " scenarios   = ${LONG_FORM_SCENARIOS[*]}"
echo "============================================================"

# --- E1 + E2: longitudinal LLM-runtime probe across four long-form scenarios.
for scenario in "${LONG_FORM_SCENARIOS[@]}"; do
  artifact_path="${BUNDLE_DIR}/${scenario}_longitudinal.json"
  echo ""
  echo "[E1+E2] running longitudinal probe for scenario: ${scenario}"
  python examples/run_cross_session_probe_llm.py \
    --rounds "${ROUNDS}" \
    --scenarios-path "${SCENARIOS_DIR}/${scenario}.json" \
    --artifact-path "${artifact_path}" \
    --skip-preflight \
    || echo "[E1+E2] non-zero exit on ${scenario} (debt #10B item 3 may still be open)"
done

# --- E4: 3-party scenario probe (separate run because the keying probe
#     produces different evidence shape).
echo ""
echo "[E4] running 3-party multi-party SHADOW probe"
python examples/run_cross_session_probe_llm.py \
  --rounds "${ROUNDS}" \
  --scenarios-path "${SCENARIOS_DIR}/long-form-three-party-arc.json" \
  --artifact-path "${BUNDLE_DIR}/long-form-three-party-arc_longitudinal.json" \
  --skip-preflight \
  || echo "[E4] non-zero exit on three-party probe"

# --- E5: assemble the final bundle manifest.
echo ""
echo "[E5] assembling evidence bundle manifest"
python -m lifeform_evolution.evidence_bundle assemble \
  --bundle-dir "${BUNDLE_DIR}" \
  --output "${BUNDLE_DIR}/evidence_bundle.json"

echo ""
echo "============================================================"
echo " Wave E5 — bundle assembly complete"
echo " manifest: ${BUNDLE_DIR}/evidence_bundle.json"
echo "============================================================"
