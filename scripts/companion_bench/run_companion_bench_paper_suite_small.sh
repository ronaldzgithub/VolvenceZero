#!/usr/bin/env bash
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.
#
# Nightly small paper-suite for Companion Bench. Public-only (24 scenarios), 1
# paraphrase seed, against ~5 reference systems. Cost: ~$200-400/night.
#
# Required env vars (any subset of these enables the matching system):
#   OPENAI_API_KEY
#   ANTHROPIC_API_KEY
#   GOOGLE_API_KEY
#   DEEPSEEK_API_KEY
#   TOGETHER_API_KEY
#   MISTRAL_API_KEY
#
# Per-turn judge defaults to anthropic/claude-3.7-sonnet; arc judge
# defaults to openai/gpt-5 (different family per RFC §6.3).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${LSCB_OUTPUT_DIR:-artifacts/companion-bench/reference}"
mkdir -p "$OUT_DIR"

systems=()
for env in OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY DEEPSEEK_API_KEY TOGETHER_API_KEY MISTRAL_API_KEY; do
  if [[ -n "${!env:-}" ]]; then
    case "$env" in
      OPENAI_API_KEY) systems+=("openai/gpt-5") ;;
      ANTHROPIC_API_KEY) systems+=("anthropic/claude-opus-4.6") ;;
      GOOGLE_API_KEY) systems+=("google/gemini-3-pro") ;;
      DEEPSEEK_API_KEY) systems+=("deepseek/deepseek-v3") ;;
      TOGETHER_API_KEY) systems+=("meta/llama-3-70b" "qwen/qwen2.5-72b-instruct") ;;
      MISTRAL_API_KEY) systems+=("mistral/mistral-large") ;;
    esac
  fi
done

if [[ ${#systems[@]} -eq 0 ]]; then
  echo "[lscb-paper-suite-small] no API keys set; aborting" >&2
  exit 1
fi

systems_csv=$(IFS=, ; echo "${systems[*]}")

echo "[lscb-paper-suite-small] scoring ${#systems[@]} systems: $systems_csv"
python scripts/companion_bench/score_reference_systems.py \
  --output-dir "$OUT_DIR" \
  --user-sim-model anthropic/claude-3.7-sonnet \
  --user-sim-key-env ANTHROPIC_API_KEY \
  --perturn-model anthropic/claude-3.7-sonnet \
  --perturn-key-env ANTHROPIC_API_KEY \
  --arc-model openai/gpt-5 \
  --arc-key-env OPENAI_API_KEY \
  --paraphrase-seeds 0 \
  --systems "$systems_csv"

echo "[lscb-paper-suite-small] aggregate → $OUT_DIR/aggregate_results.json"
