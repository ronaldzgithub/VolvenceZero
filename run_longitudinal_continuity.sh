#!/usr/bin/env bash
# Cross-session learned-state continuity lane (currentstatus 第三优先).
#
# Runs the longitudinal owner-hydration suites against real Brain
# constructors with per-user scoped persistence:
#   - tests/longitudinal/test_cross_session_owner_hydration.py
#   - tests/longitudinal/test_cross_session_learned_state_continuity.py
#     (20 sessions, social/regime/PE-heads/dual-track-gate/credit-heads
#      accumulation + cross-user isolation)
#
# Usage:
#   bash run_longitudinal_continuity.sh
#
# Extra pytest arguments are forwarded (e.g. -k, -x).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" -m pytest \
  tests/longitudinal/test_cross_session_owner_hydration.py \
  tests/longitudinal/test_cross_session_learned_state_continuity.py \
  -q "$@"
