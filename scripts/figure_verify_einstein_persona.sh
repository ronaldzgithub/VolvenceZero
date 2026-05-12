#!/usr/bin/env bash
# Wave P.1 — drive the Einstein persona verification harness end-to-end.
#
# Steps:
#   0. assert prerequisites (curated bundle id, cleaning store, metadata file)
#   1. (optional) bake a real PEFT persona LoRA on the Wave K curated bundle
#      via figure_bake_einstein_persona_lora.sh (skip with SKIP_BAKE=1)
#   2. drive python -m lifeform_domain_figure.verification.persona.cli
#      to generate questions, run the 3-condition ablation, score, and
#      emit verdict.json + transcript.md
#   3. print the verdict (exit code 0 iff all 4 gates pass; 2 if any
#      gate fails; 3 on setup error)
#
# Environment overrides:
#   RUN_ID           identifier for this verification run
#                    (default: einstein-persona-<unix-timestamp>)
#   BUNDLE_ID        curated bundle id (resolved to latest if unset)
#   BUNDLE_ROOT      persisted bundle root (default: data/figure_bundles)
#   AUDIT_ROOT       audit log root (default: data/figure_audit)
#   CLEANING_ROOT    L1 cleaning store (default: data/figure_corpus)
#   METADATA_FILE    curator metadata JSONL
#   VERIFY_OUT       output dir (default: artifacts/figure_verify/${RUN_ID})
#   QWEN_MODEL_ID    HF model id (default: sshleifer/tiny-gpt2; recommend
#                    Qwen/Qwen2.5-1.5B-Instruct for real run)
#   PEFT_TARGET_MODULES  comma-separated target modules (default: c_attn;
#                    use q_proj,k_proj,v_proj,o_proj for Qwen)
#   SKIP_BAKE        set to 1 to skip the bake step (assumes pool already
#                    populated)
#   RUNTIME_BACKEND  'transformers' (default) or 'synthetic' for harness
#                    smoke tests without HF download
#
# Exit codes mirror the verification CLI.

set -euo pipefail

RUN_ID="${RUN_ID:-einstein-persona-$(date +%s)}"
BUNDLE_ROOT="${BUNDLE_ROOT:-data/figure_bundles}"
AUDIT_ROOT="${AUDIT_ROOT:-data/figure_audit}"
CLEANING_ROOT="${CLEANING_ROOT:-data/figure_corpus}"
METADATA_FILE="${METADATA_FILE:-packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.curated_metadata.jsonl}"
VERIFY_OUT="${VERIFY_OUT:-artifacts/figure_verify/${RUN_ID}}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-sshleifer/tiny-gpt2}"
PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES:-c_attn}"
SKIP_BAKE="${SKIP_BAKE:-0}"
RUNTIME_BACKEND="${RUNTIME_BACKEND:-transformers}"

if [ ! -d "${CLEANING_ROOT}/raw" ]; then
    echo "ERROR: L1 cleaning store missing: ${CLEANING_ROOT}/raw" >&2
    echo "Run scripts/figure_collect_einstein.sh first." >&2
    exit 3
fi

if [ ! -f "${METADATA_FILE}" ]; then
    echo "ERROR: curator metadata file not found: ${METADATA_FILE}" >&2
    exit 3
fi

# Resolve BUNDLE_ID to the latest curated einstein bundle if unset.
if [ -z "${BUNDLE_ID:-}" ]; then
    BUNDLE_ID="$(python -c "
import json, sys
from pathlib import Path
root = Path('${BUNDLE_ROOT}/einstein')
if not root.exists():
    sys.exit('ERROR: no bundle root at ${BUNDLE_ROOT}/einstein/')
manifests = []
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    mf = d / 'manifest.json'
    if not mf.exists():
        continue
    payload = json.loads(mf.read_text())
    manifests.append((payload.get('created_at_iso', ''), payload.get('bundle_id', d.name)))
if not manifests:
    sys.exit('ERROR: no manifests under ${BUNDLE_ROOT}/einstein/')
manifests.sort(reverse=True)
print(manifests[0][1])
")"
fi

echo "============================================================"
echo " Wave P persona verification harness"
echo " run_id        = ${RUN_ID}"
echo " bundle_id     = ${BUNDLE_ID}"
echo " bundle_root   = ${BUNDLE_ROOT}"
echo " verify_out    = ${VERIFY_OUT}"
echo " qwen_model_id = ${QWEN_MODEL_ID}"
echo " runtime       = ${RUNTIME_BACKEND}"
echo " skip_bake     = ${SKIP_BAKE}"
echo "============================================================"

mkdir -p "${VERIFY_OUT}"

# Step 1 — bake real PEFT persona LoRA (idempotent: re-running is fine,
# the OFFLINE gate just creates a new audit row attached to the same
# pool record id derived from (figure_id, source_bundle_id)).
if [ "${SKIP_BAKE}" != "1" ]; then
    BUNDLE_ID="${BUNDLE_ID}" \
    BUNDLE_ROOT="${BUNDLE_ROOT}" \
    AUDIT_ROOT="${AUDIT_ROOT}" \
    CLEANING_ROOT="${CLEANING_ROOT}" \
    METADATA_FILE="${METADATA_FILE}" \
    QWEN_MODEL_ID="${QWEN_MODEL_ID}" \
    PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES}" \
        bash scripts/figure_bake_einstein_persona_lora.sh
else
    echo "SKIP_BAKE=1 — assuming persona LoRA already in default pool"
fi

# Step 2 — drive the verification CLI.
python -m lifeform_domain_figure.verification.persona.cli \
    --bundle-id "${BUNDLE_ID}" \
    --figure einstein \
    --bundle-root "${BUNDLE_ROOT}" \
    --output-dir "${VERIFY_OUT}" \
    --runtime "${RUNTIME_BACKEND}" \
    --qwen-model-id "${QWEN_MODEL_ID}" \
    --max-in-corpus-questions 20 \
    --conditions raw,bundle,bundle_lora \
    --questions-cache "${VERIFY_OUT}/questions.jsonl"

VERDICT_FILE="${VERIFY_OUT}/verdict.json"

if [ -f "${VERDICT_FILE}" ]; then
    echo ""
    echo "============================================================"
    echo " verdict.json:"
    cat "${VERDICT_FILE}"
    echo ""
    echo "transcript: ${VERIFY_OUT}/transcript.md"
    echo "============================================================"
fi
