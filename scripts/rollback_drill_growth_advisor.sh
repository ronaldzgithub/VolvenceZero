#!/usr/bin/env bash
# F-D: production rollback drill for the growth-advisor vertical
# (debt #50).
#
# Drill flow:
#
#   compile reviewed profile → activate → run onboarding-arc playbook
#   3 rounds → trigger OFFLINE-gate re-compile → rollback to prior
#   profile version → run onboarding-arc playbook 1 round → audit
#   verification
#
# SHADOW scaffold; real wire-up depends on growth-advisor packet
# G-D MonthlyReportOwner + revision history + audit chain. Until
# ACTIVE, --dry-run emits the placeholder.
#
# Usage:
#   PROFILE_ID=cheng-laoshi bash scripts/rollback_drill_growth_advisor.sh --dry-run
#
# Refs:
#   docs/specs/rollback-drill-cadence.md
#   docs/moving forward/growth-advisor-pilot-packet.md §2.4

set -euo pipefail

PROFILE_ID="${PROFILE_ID:-cheng-laoshi}"
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
OUT_FILE="$OUT_DIR/growth-advisor-${PROFILE_ID}-${DATE}.json"

if [ "$DRY_RUN" -eq 1 ]; then
  cat > "$OUT_FILE" <<EOF
{
  "scaffold_status": "SHADOW",
  "vertical": "growth_advisor",
  "profile_id": "${PROFILE_ID}",
  "drill_date": "${DATE}",
  "dry_run": true,
  "steps": [
    "compile reviewed profile (lifeform_builder.compile)",
    "activate via DLaaS",
    "run onboarding-arc playbook × 3 rounds (record baseline boundary trigger rates)",
    "trigger OFFLINE-gate re-compile (e.g. profile reviewer update)",
    "rollback to prior profile version",
    "run onboarding-arc playbook × 1 round → assert per-boundary rate identical to baseline",
    "verify audit log append-only across rollback"
  ],
  "expected_active_outputs": {
    "boundary_trigger_rate_baseline": null,
    "boundary_trigger_rate_post_rollback": null,
    "audit_chain_byte_stable": null
  },
  "notes": "Full drill lands with F-D ACTIVE + growth-advisor packet G-D MonthlyReportOwner."
}
EOF
  echo "wrote SHADOW placeholder: $OUT_FILE"
  exit 0
fi

echo "F-D SHADOW: real rollback drill not yet wired." >&2
echo "Re-run with --dry-run for SHADOW placeholder." >&2
exit 2
