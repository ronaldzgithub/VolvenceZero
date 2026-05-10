#!/usr/bin/env bash
# Copyright 2026 LSCB Contributors
# Licensed under the Apache License, Version 2.0.
#
# Release-tier full paper-suite for LSCB. Pulls held-out submodule,
# runs all 10 reference systems × 3 paraphrase seeds × (24 public +
# 96 held-out) scenarios. Cost: $5-15k per run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d external/lscb-heldout/scenarios ]]; then
  echo "[lscb-paper-suite-full] held-out submodule missing; init it first" >&2
  exit 1
fi

heldout_count=$(ls external/lscb-heldout/scenarios/*.yaml | wc -l)
if [[ $heldout_count -ne 96 ]]; then
  echo "[lscb-paper-suite-full] expected 96 held-out scenarios, got $heldout_count" >&2
  exit 1
fi

OUT_DIR="${LSCB_OUTPUT_DIR:-artifacts/lscb/reference-full-${LSCB_RELEASE_TAG:-untagged}}"
mkdir -p "$OUT_DIR"

# All 10 reference systems by default (no env-var filtering at this
# tier; missing keys cause individual system runs to fail loudly).
python scripts/lscb/score_reference_systems.py \
  --output-dir "$OUT_DIR" \
  --user-sim-model anthropic/claude-3.7-sonnet \
  --user-sim-key-env ANTHROPIC_API_KEY \
  --perturn-model anthropic/claude-3.7-sonnet \
  --perturn-key-env ANTHROPIC_API_KEY \
  --arc-model openai/gpt-5 \
  --arc-key-env OPENAI_API_KEY \
  --paraphrase-seeds 0,1,2 \
  --include-heldout \
  --require-heldout

echo "[lscb-paper-suite-full] aggregate → $OUT_DIR/aggregate_results.json"
