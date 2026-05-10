#!/usr/bin/env bash
# known-debts #23 — one-shot demo of the figure vertical bake / gate /
# rollback CLI on the synthetic Einstein corpus.
#
# Drives the full F1-F6 chain in one command:
#   1. bake-bundle      — compile profile + corpus into a base bundle.
#   2. bake-steering    — bake contrast steering, route through OFFLINE
#                         gate, persist new bundle.
#   3. bake-lora        — bake synthetic persona LoRA, route through
#                         OFFLINE gate, persist new bundle, register
#                         in default PersonaLoRAPool.
#   4. list             — enumerate every persisted bundle.
#
# Usage:
#   ./scripts/figure_demo_einstein.sh [BUNDLE_ROOT] [AUDIT_ROOT]
#
# Defaults:
#   BUNDLE_ROOT = data/figure_bundles
#   AUDIT_ROOT  = data/figure_audit
#
# Exit codes (mirrors the CLI):
#   0 = full chain succeeded.
#   1 = CLI argument error.
#   2 = OFFLINE gate BLOCKed a step (the audit row carries
#       block_reasons; this script does NOT swallow the block).
#   3 = I/O / schema error.
#
# Notes:
# * Bundle ids are extracted from each subcommand's JSON stdout via
#   ``python -c`` so the script works on Windows / Git Bash without
#   requiring ``jq``.
# * The CLI is invoked as ``python -m lifeform_domain_figure.cli`` so
#   the script does not depend on the ``figure-bake`` console script
#   being installed (works in any wheel-installed env).

set -euo pipefail

BUNDLE_ROOT="${1:-data/figure_bundles}"
AUDIT_ROOT="${2:-data/figure_audit}"

mkdir -p "${BUNDLE_ROOT}" "${AUDIT_ROOT}"

CLI=(python -m lifeform_domain_figure.cli
    --bundle-root "${BUNDLE_ROOT}"
    --audit-root "${AUDIT_ROOT}")

extract_bundle_id() {
    python -c "import json,sys; print(json.loads(sys.stdin.read())['bundle_id'])"
}

echo "============================================================"
echo " known-debts #23 — figure vertical bake demo (Einstein)"
echo " bundle_root = ${BUNDLE_ROOT}"
echo " audit_root  = ${AUDIT_ROOT}"
echo "============================================================"

echo ""
echo "[1/4] bake-bundle einstein (synthetic corpus)"
BAKE_BUNDLE_OUT="$("${CLI[@]}" bake-bundle --figure einstein --corpus-mode synthetic)"
echo "${BAKE_BUNDLE_OUT}"
BASE_BUNDLE_ID="$(printf '%s' "${BAKE_BUNDLE_OUT}" | extract_bundle_id)"
echo "    base bundle id: ${BASE_BUNDLE_ID}"

echo ""
echo "[2/4] bake-steering einstein (contrastive, OFFLINE gate)"
BAKE_STEER_OUT="$("${CLI[@]}" bake-steering \
    --figure einstein \
    --bundle "${BASE_BUNDLE_ID}" \
    --evaluation-snapshot default-clean \
    --rollback-evidence "prev_steering=absent;base=${BASE_BUNDLE_ID}")"
echo "${BAKE_STEER_OUT}"
STEER_BUNDLE_ID="$(printf '%s' "${BAKE_STEER_OUT}" | extract_bundle_id)"
echo "    steering bundle id: ${STEER_BUNDLE_ID}"

echo ""
echo "[3/4] bake-lora einstein (synthetic backend, OFFLINE gate)"
BAKE_LORA_OUT="$("${CLI[@]}" bake-lora \
    --figure einstein \
    --bundle "${STEER_BUNDLE_ID}" \
    --backend synthetic \
    --rank 8 \
    --evaluation-snapshot default-clean \
    --rollback-evidence "prev_lora=absent;base=${STEER_BUNDLE_ID}")"
echo "${BAKE_LORA_OUT}"
LORA_BUNDLE_ID="$(printf '%s' "${BAKE_LORA_OUT}" | extract_bundle_id)"
echo "    final bundle id: ${LORA_BUNDLE_ID}"

echo ""
echo "[4/4] list einstein"
"${CLI[@]}" list --figure einstein

echo ""
echo "============================================================"
echo " demo complete"
echo " base    : ${BASE_BUNDLE_ID}"
echo " steering: ${STEER_BUNDLE_ID}"
echo " lora    : ${LORA_BUNDLE_ID}"
echo "------------------------------------------------------------"
echo " audit dir: ${AUDIT_ROOT}"
echo "============================================================"
