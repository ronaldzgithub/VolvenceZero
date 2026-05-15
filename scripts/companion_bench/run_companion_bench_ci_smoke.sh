#!/usr/bin/env bash
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.
#
# CI smoke entrypoint for Companion Bench. Public-only deterministic-fake run;
# no API keys required. Used by .github/workflows/companion-bench-ci-smoke.yml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[companionbench-ci-smoke] running unit tests..."
python -m pytest packages/companion-bench/tests -q

echo "[companionbench-ci-smoke] running no-internal-imports contract test..."
python -m pytest tests/contracts/test_companion_bench_no_internal_imports.py -q

echo "[companionbench-ci-smoke] verifying public scenario hash table..."
python scripts/companion_bench/emit_scenario_hashes.py --output /tmp/companionbench-regen-hashes.txt
diff <(grep -v '^#' docs/external/companion-bench-public-scenario-hashes.txt | grep -v '^$') \
     <(grep -v '^#' /tmp/companionbench-regen-hashes.txt | grep -v '^$') \
  && echo "[companionbench-ci-smoke] hash table OK" \
  || { echo "[companionbench-ci-smoke] hash table drifted; regenerate via emit_scenario_hashes.py" >&2; exit 1; }

echo "[companionbench-ci-smoke] running F2 family smoke..."
python -m companion_bench.cli smoke \
  --family F2 \
  --artifact-dir artifacts/companion-bench/ci-smoke \
  --summary artifacts/companion-bench/ci-smoke/summary.json

echo "[companionbench-ci-smoke] OK"
