#!/usr/bin/env bash
# Semantic-grounding evidence pipeline launcher.
#
# Thin root wrapper around:
#   scripts/run_semantic_grounding_evidence.py
#
# Usage:
#   bash run_semantic_grounding_evidence.sh                       # unit + smoke
#   bash run_semantic_grounding_evidence.sh --lane hf --substrate-device mps
#   bash run_semantic_grounding_evidence.sh --lane all
#
# All arguments are forwarded to the Python orchestrator.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" scripts/run_semantic_grounding_evidence.py "$@"
