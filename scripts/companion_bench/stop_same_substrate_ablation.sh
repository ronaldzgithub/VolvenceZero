#!/usr/bin/env bash
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.
#
# Tear down the same-substrate ablation endpoints started by
# serve_same_substrate_ablation.sh.
#
# Usage: bash scripts/companion_bench/stop_same_substrate_ablation.sh <serve.pids>

set -euo pipefail

PID_FILE="${1:?usage: stop_same_substrate_ablation.sh <serve.pids>}"
if [[ ! -f "$PID_FILE" ]]; then
  echo "error: PID file not found: $PID_FILE" >&2
  exit 2
fi

while read -r pid; do
  [[ -z "$pid" ]] && continue
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop] killing $pid"
    kill "$pid" 2>/dev/null || true
  fi
done < "$PID_FILE"

echo "[stop] done"
