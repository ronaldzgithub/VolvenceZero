#!/usr/bin/env bash
# F-D: production rollback drill for the figure vertical (debt #50).
#
# Runs the full production-grade rollback flow on a real Qwen
# substrate + a curated figure bundle:
#
#   bake → adopt → activate → 10 turn → rollback → audit verification
#
# SHADOW: this script is a scaffold; the real bake/adopt/activate/
# rollback steps each require the corresponding ACTIVE artifact (P1
# packet #41 real PEFT bake / Wave E DLaaS adopt / Wave D LoRA
# hot-swap / debt #23 audit). Until ACTIVE, ./--dry-run prints what
# would run.
#
# Usage:
#   FIGURE_ID=einstein bash scripts/rollback_drill_figure.sh --dry-run
#   FIGURE_ID=einstein bash scripts/rollback_drill_figure.sh
#
# Outputs:
#   artifacts/rollback_drill/figure-<date>.json
#
# Refs:
#   docs/specs/rollback-drill-cadence.md
#   docs/moving forward/cross-cutting-foundation-packet.md §2.4

set -euo pipefail

FIGURE_ID="${FIGURE_ID:-einstein}"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

OUT_DIR="artifacts/rollback_drill"
mkdir -p "$OUT_DIR"
DATE=$(date +%Y-%m-%d)
OUT_FILE="$OUT_DIR/figure-${FIGURE_ID}-${DATE}.json"

if [ "$DRY_RUN" -eq 1 ]; then
  cat > "$OUT_FILE" <<EOF
{
  "scaffold_status": "SHADOW",
  "vertical": "figure",
  "figure_id": "${FIGURE_ID}",
  "drill_date": "${DATE}",
  "dry_run": true,
  "steps": [
    "bake bundle (figure-bake bake-bundle --figure ${FIGURE_ID})",
    "DLaaS adopt with bundle",
    "activate persona LoRA via PersonaLoRAPool",
    "generate 10 turn → record logits baseline",
    "trigger rollback (figure-bake rollback)",
    "generate 10 turn → assert logits L1 < 1e-6 vs baseline",
    "verify audit log append-only across rollback"
  ],
  "expected_active_outputs": {
    "logits_l1_distance": null,
    "rollback_audit_id": null,
    "audit_chain_byte_stable": null
  },
  "notes": "Full drill lands with F-D ACTIVE (Phase B W2-W3) + figure-evidence packet #41 real Qwen PEFT bake."
}
EOF
  echo "wrote SHADOW placeholder: $OUT_FILE"
  exit 0
fi

echo "F-D SHADOW: real rollback drill not yet wired." >&2
echo "Re-run with --dry-run for SHADOW placeholder." >&2
exit 2
