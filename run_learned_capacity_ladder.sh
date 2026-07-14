#!/usr/bin/env bash
# Capacity->gain ladder launcher.
#
# Thin root wrapper around:
#   scripts/run_capacity_ladder.py
#
# Usage:
#   bash run_learned_capacity_ladder.sh
#   bash run_learned_capacity_ladder.sh --n-z 16,64,256 --turns 500
#   bash run_learned_capacity_ladder.sh --execute --n-z 16 --turns 50
#
# All arguments are forwarded to the Python ladder runner.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" scripts/run_capacity_ladder.py "$@"
