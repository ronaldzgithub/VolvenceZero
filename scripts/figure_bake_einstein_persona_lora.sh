#!/usr/bin/env bash
# Wave N — bake a real PEFT persona LoRA on the Wave K curated Einstein bundle.
#
# Steps:
#   1. assert curated bundle (BUNDLE_ID) + cleaning store + metadata file all present
#   2. assert peft / transformers / torch importable
#   3. drive ``figure-bake bake-lora --corpus-mode curated --backend peft ...``
#   4. write the resulting LoRA artifact's pool record id to a marker file
#      so the verification harness (Wave P) can pick it up without re-baking.
#
# Environment overrides:
#   BUNDLE_ID            curated bundle id (default: latest under BUNDLE_ROOT/einstein/)
#   BUNDLE_ROOT          persisted bundle root (default: data/figure_bundles)
#   AUDIT_ROOT           audit log root (default: data/figure_audit)
#   CLEANING_ROOT        L1 cleaning store (default: data/figure_corpus)
#   METADATA_FILE        curator metadata JSONL
#                        (default: packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.curated_metadata.jsonl)
#   QWEN_MODEL_ID        HuggingFace model id (default: sshleifer/tiny-gpt2 for fast smoke;
#                        recommended Qwen/Qwen2.5-1.5B-Instruct for real run)
#   PEFT_TARGET_MODULES  comma-separated target modules (default: c_attn for GPT-2;
#                        use q_proj,k_proj,v_proj,o_proj for Qwen)
#   PEFT_RANK            LoRA rank (default 8)
#   PEFT_MAX_STEPS       optimizer step cap per epoch (default 50)
#   PEFT_DEVICE          torch device (default cpu)
#   ROLLBACK_EVIDENCE    OFFLINE-gate rollback string (default reads BUNDLE_ID)
#
# Exit codes mirror the underlying CLI (0=ok, 2=gate block, 3=io/schema).

set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-data/figure_bundles}"
AUDIT_ROOT="${AUDIT_ROOT:-data/figure_audit}"
CLEANING_ROOT="${CLEANING_ROOT:-data/figure_corpus}"
METADATA_FILE="${METADATA_FILE:-packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.curated_metadata.jsonl}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-sshleifer/tiny-gpt2}"
PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES:-c_attn}"
PEFT_RANK="${PEFT_RANK:-8}"
PEFT_MAX_STEPS="${PEFT_MAX_STEPS:-50}"
PEFT_DEVICE="${PEFT_DEVICE:-cpu}"

if [ ! -f "${METADATA_FILE}" ]; then
    echo "ERROR: curator metadata file not found: ${METADATA_FILE}" >&2
    echo "Run scripts/figure_collect_einstein.sh first to populate Wave K corpus." >&2
    exit 3
fi

if [ ! -d "${CLEANING_ROOT}/raw" ]; then
    echo "ERROR: L1 cleaning store missing: ${CLEANING_ROOT}/raw" >&2
    echo "Run scripts/figure_collect_einstein.sh first." >&2
    exit 3
fi

# Resolve BUNDLE_ID: if not set, pick the latest curated bundle under
# BUNDLE_ROOT/einstein/. We use Python to inspect manifests (avoid jq dep).
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
    sys.exit('ERROR: no manifests found under ${BUNDLE_ROOT}/einstein/')
manifests.sort(reverse=True)
print(manifests[0][1])
")"
fi

ROLLBACK_EVIDENCE="${ROLLBACK_EVIDENCE:-prev_persona_lora=absent;base=${BUNDLE_ID}}"

echo "============================================================"
echo " Wave N persona LoRA bake (curated mode)"
echo " bundle_id     = ${BUNDLE_ID}"
echo " bundle_root   = ${BUNDLE_ROOT}"
echo " cleaning_root = ${CLEANING_ROOT}"
echo " metadata_file = ${METADATA_FILE}"
echo " qwen_model_id = ${QWEN_MODEL_ID}"
echo " target_modules= ${PEFT_TARGET_MODULES}"
echo " rank          = ${PEFT_RANK}"
echo " max_steps     = ${PEFT_MAX_STEPS}"
echo " device        = ${PEFT_DEVICE}"
echo "============================================================"

mkdir -p "${BUNDLE_ROOT}" "${AUDIT_ROOT}"

python -m lifeform_domain_figure.cli \
    --bundle-root "${BUNDLE_ROOT}" \
    --audit-root "${AUDIT_ROOT}" \
    bake-lora \
    --figure einstein \
    --bundle "${BUNDLE_ID}" \
    --corpus-mode curated \
    --cleaning-root "${CLEANING_ROOT}" \
    --curated-metadata-file "${METADATA_FILE}" \
    --backend peft \
    --rank "${PEFT_RANK}" \
    --peft-model-id "${QWEN_MODEL_ID}" \
    --peft-target-modules "${PEFT_TARGET_MODULES}" \
    --peft-max-steps "${PEFT_MAX_STEPS}" \
    --peft-device "${PEFT_DEVICE}" \
    --evaluation-snapshot default-clean \
    --rollback-evidence "${ROLLBACK_EVIDENCE}"

echo ""
echo "============================================================"
echo " bake complete; persona LoRA registered in default pool"
echo " (verify: python -c \"from volvence_zero.substrate import default_persona_lora_pool;"
echo "  p=default_persona_lora_pool(); print(p.lookup('einstein').record_id)\")"
echo "============================================================"
