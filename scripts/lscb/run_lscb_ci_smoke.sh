#!/usr/bin/env bash
# Copyright 2026 LSCB Contributors
# Licensed under the Apache License, Version 2.0.
#
# CI smoke entrypoint for LSCB. Public-only deterministic-fake run;
# no API keys required. Used by .github/workflows/lscb-ci-smoke.yml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[lscb-ci-smoke] running unit tests..."
python -m pytest packages/lscb-bench/tests -q

echo "[lscb-ci-smoke] running no-internal-imports contract test..."
python -m pytest tests/contracts/test_lscb_bench_no_internal_imports.py -q

echo "[lscb-ci-smoke] verifying public scenario hash table..."
python scripts/lscb/emit_scenario_hashes.py --output /tmp/lscb-regen-hashes.txt
diff <(grep -v '^#' docs/external/lscb-public-scenario-hashes.txt | grep -v '^$') \
     <(grep -v '^#' /tmp/lscb-regen-hashes.txt | grep -v '^$') \
  && echo "[lscb-ci-smoke] hash table OK" \
  || { echo "[lscb-ci-smoke] hash table drifted; regenerate via emit_scenario_hashes.py" >&2; exit 1; }

echo "[lscb-ci-smoke] running F2 family smoke..."
python -m lscb_bench.cli smoke \
  --family F2 \
  --artifact-dir artifacts/lscb/ci-smoke \
  --summary artifacts/lscb/ci-smoke/summary.json

echo "[lscb-ci-smoke] OK"
