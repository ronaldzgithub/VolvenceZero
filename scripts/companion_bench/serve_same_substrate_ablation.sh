#!/usr/bin/env bash
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.
#
# Boot the five same-substrate Companion Bench ablation endpoints, all sharing
# ONE frozen Qwen so any score delta is attributable to the layer over the
# substrate, not the substrate itself.
#
#   :8000  lifeform-serve --vertical companion        -> mode=lifeform (Volvence full)
#                                                      -> mode=raw      (bare Qwen, the `raw` track)
#   :8001  lifeform-serve --vertical companion-cold   -> mode=lifeform (Volvence, no trained bootstraps)
#   :8500  companion-ref-harness                      -> upstream :8000/v1?mode=raw  (standard memory wrapper)
#   :8600  companion-camel-baseline                   -> upstream :8000/v1?mode=raw  (CAMEL agent framework)
#
# The ref-harness + camel backends point their upstream at :8000's mode=raw path,
# guaranteeing byte-identical weights with the `raw` track. The lifeform +
# companion-cold tracks load the SAME --substrate-model-id in-process.
#
# Required env:
#   VZ_SUBSTRATE_MODEL_ID   HF model id, e.g. Qwen/Qwen2.5-7B-Instruct
#   LIFEFORM_LOCAL_API_KEY  bearer token the OpenAI-compat endpoints require
#
# Optional env:
#   VZ_SUBSTRATE_DEVICE     default: cuda
#   ARTIFACT_DIR            default: artifacts/companion-ablation/<date>
#   REFH_EXTRACTOR_BASE_URL / REFH_EXTRACTOR_MODEL / REFH_EXTRACTOR_KEY_ENV /
#   REFH_EXTRACTOR_FAMILY   cross-family memory extractor for the ref-harness
#                           (recommended: a NON-Qwen family, e.g. Anthropic).
#   CAMEL_BACKEND           default: camel  (set to `echo` for a no-GPU dry run)
#
# This launches background processes and writes per-track substrate fingerprints,
# then runs assert_same_substrate.py to fail loud if anything diverged. Use
# stop_same_substrate_ablation.sh (PID file) to tear them down.

set -euo pipefail

: "${VZ_SUBSTRATE_MODEL_ID:?set VZ_SUBSTRATE_MODEL_ID (e.g. Qwen/Qwen2.5-7B-Instruct)}"
: "${LIFEFORM_LOCAL_API_KEY:?set LIFEFORM_LOCAL_API_KEY (bearer token for the OpenAI-compat endpoints)}"

VZ_SUBSTRATE_DEVICE="${VZ_SUBSTRATE_DEVICE:-cuda}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/companion-ablation/${DATE_TAG}}"
CAMEL_BACKEND="${CAMEL_BACKEND:-camel}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${ARTIFACT_DIR}/serve-logs"
PID_FILE="${ARTIFACT_DIR}/serve.pids"
mkdir -p "$LOG_DIR"
: > "$PID_FILE"

RAW_UPSTREAM="http://127.0.0.1:8000/v1?mode=raw"

echo "[serve] substrate=${VZ_SUBSTRATE_MODEL_ID} device=${VZ_SUBSTRATE_DEVICE}"
echo "[serve] artifacts -> ${ARTIFACT_DIR}"

start() {
  local name="$1"; shift
  echo "[serve] starting ${name}: $*"
  ( "$@" >"${LOG_DIR}/${name}.log" 2>&1 & echo "$!" >>"$PID_FILE" )
}

# --- :8000 companion (mode=lifeform = volvence full, mode=raw = bare Qwen) ---
start lifeform-companion \
  lifeform-serve \
    --vertical companion \
    --port 8000 \
    --substrate-mode hf-shared \
    --substrate-model-id "$VZ_SUBSTRATE_MODEL_ID" \
    --substrate-device "$VZ_SUBSTRATE_DEVICE" \
    --enable-openai-compat

# --- :8001 companion-cold (volvence, no trained bootstraps) ---
start lifeform-companion-cold \
  lifeform-serve \
    --vertical companion-cold \
    --port 8001 \
    --substrate-mode hf-shared \
    --substrate-model-id "$VZ_SUBSTRATE_MODEL_ID" \
    --substrate-device "$VZ_SUBSTRATE_DEVICE" \
    --enable-openai-compat

# --- :8500 ref-harness (standard memory wrapper, all four components) ---
REFH_ARGS=(
  companion-ref-harness serve
    --port 8500
    --upstream-base-url "$RAW_UPSTREAM"
    --upstream-model lifeform-raw
    --upstream-key-env LIFEFORM_LOCAL_API_KEY
    --components summary,embed,user_model,episodic
    --store-mode sqlite
    --store-path "${ARTIFACT_DIR}/ref-harness.sqlite3"
)
if [[ -n "${REFH_EXTRACTOR_MODEL:-}" ]]; then
  REFH_ARGS+=(
    --summary-extractor-base-url "${REFH_EXTRACTOR_BASE_URL:?set REFH_EXTRACTOR_BASE_URL}"
    --summary-extractor-model "$REFH_EXTRACTOR_MODEL"
    --summary-extractor-key-env "${REFH_EXTRACTOR_KEY_ENV:?set REFH_EXTRACTOR_KEY_ENV}"
    --summary-extractor-family "${REFH_EXTRACTOR_FAMILY:-anthropic}"
  )
else
  echo "[serve] WARNING: no REFH_EXTRACTOR_MODEL set -> ref-harness extractor falls back to the Qwen upstream (same-family 'crib-notes' risk). Set a NON-Qwen extractor for a fair run." >&2
fi
start ref-harness "${REFH_ARGS[@]}"

# --- :8600 camel baseline (CAMEL agent framework on the same Qwen) ---
start camel-baseline \
  companion-camel-baseline serve \
    --port 8600 \
    --backend "$CAMEL_BACKEND" \
    --upstream-base-url "$RAW_UPSTREAM" \
    --upstream-model lifeform-raw \
    --upstream-key-env LIFEFORM_LOCAL_API_KEY \
    --store-mode sqlite \
    --store-path "${ARTIFACT_DIR}/camel-baseline.sqlite3"

# --- record per-track substrate fingerprints ---
write_fp() {
  local track="$1"
  local dir="${ARTIFACT_DIR}/${track}"
  mkdir -p "$dir"
  cat >"${dir}/substrate_fingerprint.json" <<JSON
{
  "track": "${track}",
  "substrate_model_id": "${VZ_SUBSTRATE_MODEL_ID}",
  "served_at": "${DATE_TAG}"
}
JSON
}
for track in raw ref-harness camel volvence-cold volvence; do
  write_fp "$track"
done

echo "[serve] waiting 20s for endpoints to come up..."
sleep 20

echo "[serve] asserting same-substrate invariant..."
python scripts/companion_bench/assert_same_substrate.py \
  --fingerprint-file raw="${ARTIFACT_DIR}/raw/substrate_fingerprint.json" \
  --fingerprint-file ref-harness="${ARTIFACT_DIR}/ref-harness/substrate_fingerprint.json" \
  --fingerprint-file camel="${ARTIFACT_DIR}/camel/substrate_fingerprint.json" \
  --fingerprint-file volvence-cold="${ARTIFACT_DIR}/volvence-cold/substrate_fingerprint.json" \
  --fingerprint-file volvence="${ARTIFACT_DIR}/volvence/substrate_fingerprint.json"

echo "[serve] all endpoints launched. PIDs in ${PID_FILE}. Logs in ${LOG_DIR}."
echo "[serve] tear down with: bash scripts/companion_bench/stop_same_substrate_ablation.sh ${PID_FILE}"
