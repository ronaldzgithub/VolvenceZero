#!/usr/bin/env bash
# Resume-safe learned-backend ACTIVE evidence launcher.
#
# Thin root wrapper around:
#   scripts/run_learned_active_evidence.py
#
# Usage:
#   bash run_learned_active_evidence.sh --resume
#   bash run_learned_active_evidence.sh --dry-run
#   bash run_learned_active_evidence.sh --substrate-mode hf --substrate-device mps --substrate-allow-download
#   bash run_learned_active_evidence.sh --ablation-verdict artifacts/.../verdict_p1.json
#
# All arguments are forwarded to the Python orchestrator.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" scripts/run_learned_active_evidence.py "$@"
