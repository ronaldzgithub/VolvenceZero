#!/usr/bin/env bash
# Learned-shadow P0 wiring smoke launcher.
#
# Thin root wrapper around:
#   scripts/run_learned_shadow_evidence_smoke.py
#
# Usage:
#   bash run_learned_shadow_smoke.sh
#   bash run_learned_shadow_smoke.sh --turns 8 --output-dir artifacts/learned_shadow_evidence_smoke
#
# All arguments are forwarded to the Python smoke runner.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" scripts/run_learned_shadow_evidence_smoke.py "$@"
