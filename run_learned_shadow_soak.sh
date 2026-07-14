#!/usr/bin/env bash
# Learned-shadow soak launcher.
#
# Thin root wrapper around:
#   scripts/run_learned_shadow_soak.py
#
# Usage:
#   bash run_learned_shadow_soak.sh --turns 500 --substrate-mode hf --substrate-device mps
#   bash run_learned_shadow_soak.sh --turns 500 --substrate-mode hf --substrate-device cuda
#   bash run_learned_shadow_soak.sh --turns 50 --substrate-mode synthetic
#
# All arguments are forwarded to the Python soak runner. Use the active evidence
# orchestrator for resume markers; this script runs one continuous soak artifact.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" -u scripts/run_learned_shadow_soak.py "$@"
