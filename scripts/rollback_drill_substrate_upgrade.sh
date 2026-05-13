#!/usr/bin/env bash
# F-D: substrate upgrade N → N+1 → rollback drill (debt #50 + #47).
#
# Validates that any substrate upgrade can be cleanly rolled back
# to the prior version, with all bundles / profiles / runtime
# fingerprints staying consistent.
#
# Drill flow:
#
#   record N substrate fingerprint
#   → upgrade to N+1
#   → record N+1 substrate fingerprint
#   → load existing figure bundle (compatible_substrates from N)
#   → expect fail-loud (or graceful degrade per upgrade-protocol §4)
#   → rollback substrate to N
#   → load same bundle → expect success
#   → byte-identical audit chain check
#
# SHADOW scaffold; depends on F-C SubstrateFingerprint ACTIVE +
# F-D rollback drill ACTIVE.
#
# Usage:
#   bash scripts/rollback_drill_substrate_upgrade.sh --dry-run
#
# Refs:
#   docs/specs/substrate-upgrade-protocol.md
#   docs/specs/rollback-drill-cadence.md

set -euo pipefail

OLD_MODEL_ID="${OLD_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
NEW_MODEL_ID="${NEW_MODEL_ID:-Qwen/Qwen3-1.5B-Instruct}"
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
OUT_FILE="$OUT_DIR/substrate-upgrade-${DATE}.json"

if [ "$DRY_RUN" -eq 1 ]; then
  cat > "$OUT_FILE" <<EOF
{
  "scaffold_status": "SHADOW",
  "drill_kind": "substrate_upgrade",
  "old_model_id": "${OLD_MODEL_ID}",
  "new_model_id": "${NEW_MODEL_ID}",
  "drill_date": "${DATE}",
  "dry_run": true,
  "steps": [
    "record N fingerprint (runtime.fingerprint())",
    "upgrade substrate weights to N+1",
    "record N+1 fingerprint",
    "attempt to load figure bundle baked against N (expect fail-loud per substrate-upgrade-protocol §4)",
    "rollback substrate weights to N",
    "load same bundle → expect success",
    "byte-identical audit chain check"
  ],
  "expected_active_outputs": {
    "n_fingerprint": null,
    "n_plus_1_fingerprint": null,
    "incompatibility_detected": null,
    "rollback_load_success": null
  },
  "notes": "Depends on F-C SubstrateFingerprint ACTIVE + F-D rollback drill ACTIVE (Phase B W2-W3)."
}
EOF
  echo "wrote SHADOW placeholder: $OUT_FILE"
  exit 0
fi

echo "F-D SHADOW: real substrate upgrade drill not yet wired." >&2
echo "Re-run with --dry-run for SHADOW placeholder." >&2
exit 2
