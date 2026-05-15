#!/usr/bin/env bash
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.
#
# Build the static leaderboard site for local preview. The actual
# deploy is handled by .github/workflows/companion-bench-publish.yml;
# this script only refreshes the data file from the latest reference
# run and starts a local HTTP server.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REF_DIR="${COMPANIONBENCH_REFERENCE_DIR:-artifacts/companion-bench/reference}"
SITE_DATA="site/leaderboard/data/aggregate_results.json"

if [[ -f "$REF_DIR/aggregate_results.json" ]]; then
  echo "[build_leaderboard_site] copying $REF_DIR/aggregate_results.json → $SITE_DATA"
  mkdir -p "$(dirname "$SITE_DATA")"
  cp "$REF_DIR/aggregate_results.json" "$SITE_DATA"
else
  echo "[build_leaderboard_site] no real aggregate at $REF_DIR; using demo data" >&2
  python scripts/companion_bench/generate_demo_aggregate.py --output "$SITE_DATA"
fi

PORT="${COMPANIONBENCH_PREVIEW_PORT:-8089}"
echo "[build_leaderboard_site] preview at http://localhost:$PORT"
exec python -m http.server "$PORT" --directory site/leaderboard
