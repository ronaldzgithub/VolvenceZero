#!/usr/bin/env bash
# Affordance score learner Stage 0 probe launcher (G3).
#
# Thin root wrapper around:
#   scripts/probe_affordance_score_learner.py
#
# Usage:
#   bash run_affordance_learner_probe.sh
#
# Drives the real lifeform path (registry -> module -> invoker ->
# outcome listener) with deterministic in-process tools so the SHADOW
# affordance score learner accumulates settles. Machinery evidence only;
# promotion still gates on >=50 real-usage settles.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" -u scripts/probe_affordance_score_learner.py "$@"
