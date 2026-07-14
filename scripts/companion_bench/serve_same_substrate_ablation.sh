#!/usr/bin/env bash
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.
#
# Boot the nine same-substrate Companion Bench ablation endpoints, all sharing
# ONE frozen Qwen so any score delta is attributable to the layer over the
# substrate, not the substrate itself.
#
#   :8000  lifeform-serve --vertical companion        -> mode=lifeform (Volvence full)
#                                                      -> mode=raw      (bare Qwen, the `raw` track)
#   :8001  lifeform-serve --vertical companion-cold   -> mode=lifeform (Volvence, no trained bootstraps)
#   :8002  lifeform-serve --vertical companion-pe-drive-off
#   :8003  lifeform-serve --vertical companion-eta-off
#   :8004  lifeform-serve --vertical companion-active-learning-off
#   :8005  lifeform-serve --vertical companion-lora-adapter
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
#   VZ_SUBSTRATE_DEVICE     default: cuda (use mps on Apple silicon, cpu as last resort)
#   ARTIFACT_DIR            default: artifacts/companion-ablation/<date>
#   REFH_EXTRACTOR_BASE_URL / REFH_EXTRACTOR_MODEL / REFH_EXTRACTOR_KEY_ENV /
#   REFH_EXTRACTOR_FAMILY   cross-family memory extractor for the ref-harness
#                           (REQUIRED for backend parity with the .ps1 launcher;
#                           must be a NON-Qwen family).
#   REFH_EMBEDDER           default: bge-m3 (real semantic embedder for H-B)
#   CAMEL_COMPACTION_BASE_URL / CAMEL_COMPACTION_MODEL / CAMEL_COMPACTION_KEY_ENV
#                           cross-family memory compaction for the camel track
#                           (REQUIRED when CAMEL_BACKEND=camel).
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

# --- component-causal arms (same substrate, altered controller layer only) ---
start lifeform-companion-pe-drive-off \
  lifeform-serve \
    --vertical companion-pe-drive-off \
    --port 8002 \
    --substrate-mode hf-shared \
    --substrate-model-id "$VZ_SUBSTRATE_MODEL_ID" \
    --substrate-device "$VZ_SUBSTRATE_DEVICE" \
    --enable-openai-compat

start lifeform-companion-eta-off \
  lifeform-serve \
    --vertical companion-eta-off \
    --port 8003 \
    --substrate-mode hf-shared \
    --substrate-model-id "$VZ_SUBSTRATE_MODEL_ID" \
    --substrate-device "$VZ_SUBSTRATE_DEVICE" \
    --enable-openai-compat

start lifeform-companion-active-learning-off \
  lifeform-serve \
    --vertical companion-active-learning-off \
    --port 8004 \
    --substrate-mode hf-shared \
    --substrate-model-id "$VZ_SUBSTRATE_MODEL_ID" \
    --substrate-device "$VZ_SUBSTRATE_DEVICE" \
    --enable-openai-compat

start lifeform-companion-lora-adapter \
  lifeform-serve \
    --vertical companion-lora-adapter \
    --port 8005 \
    --substrate-mode hf-shared \
    --substrate-model-id "$VZ_SUBSTRATE_MODEL_ID" \
    --substrate-device "$VZ_SUBSTRATE_DEVICE" \
    --enable-openai-compat

# --- :8500 ref-harness (standard memory wrapper, all four components) ---
REFH_EMBEDDER="${REFH_EMBEDDER:-bge-m3}"
REFH_ARGS=(
  companion-ref-harness serve
    --port 8500
    --upstream-base-url "$RAW_UPSTREAM"
    --upstream-model lifeform-raw
    --upstream-key-env LIFEFORM_LOCAL_API_KEY
    --components summary,embed,user_model,episodic
    --embedder "$REFH_EMBEDDER"
    --store-mode sqlite
    --store-path "${ARTIFACT_DIR}/ref-harness.sqlite3"
)
if [[ -n "${REFH_EXTRACTOR_MODEL:-}" ]]; then
  REFH_ARGS+=(
    --summary-extractor-base-url "${REFH_EXTRACTOR_BASE_URL:?set REFH_EXTRACTOR_BASE_URL}"
    --summary-extractor-model "$REFH_EXTRACTOR_MODEL"
    --summary-extractor-key-env "${REFH_EXTRACTOR_KEY_ENV:?set REFH_EXTRACTOR_KEY_ENV}"
    --summary-extractor-family "${REFH_EXTRACTOR_FAMILY:-openai-compat}"
  )
else
  echo "[serve] ERROR: ref-harness memory components require a cross-family extractor: set REFH_EXTRACTOR_MODEL / REFH_EXTRACTOR_BASE_URL / REFH_EXTRACTOR_KEY_ENV (a NON-Qwen family)." >&2
  exit 2
fi
start ref-harness "${REFH_ARGS[@]}"

# --- :8600 camel baseline (CAMEL agent framework on the same Qwen) ---
CAMEL_ARGS=(
  companion-camel-baseline serve
    --port 8600
    --backend "$CAMEL_BACKEND"
    --upstream-base-url "$RAW_UPSTREAM"
    --upstream-model lifeform-raw
    --upstream-key-env LIFEFORM_LOCAL_API_KEY
    --store-mode sqlite
    --store-path "${ARTIFACT_DIR}/camel-baseline.sqlite3"
)
if [[ "$CAMEL_BACKEND" == "camel" ]]; then
  if [[ -z "${CAMEL_COMPACTION_MODEL:-}" ]]; then
    echo "[serve] ERROR: camel backend memory compaction requires a cross-family extractor: set CAMEL_COMPACTION_MODEL / CAMEL_COMPACTION_BASE_URL / CAMEL_COMPACTION_KEY_ENV (a NON-Qwen family)." >&2
    exit 2
  fi
  CAMEL_ARGS+=(
    --compaction-base-url "${CAMEL_COMPACTION_BASE_URL:?set CAMEL_COMPACTION_BASE_URL}"
    --compaction-model "$CAMEL_COMPACTION_MODEL"
    --compaction-key-env "${CAMEL_COMPACTION_KEY_ENV:?set CAMEL_COMPACTION_KEY_ENV}"
  )
fi
start camel-baseline "${CAMEL_ARGS[@]}"

# --- record per-track substrate fingerprints ---
# Do NOT clobber fingerprints already written by preflight_llm.py: those carry
# weights_sha256 and are what assert_same_substrate.py --require-weights-sha256
# (the P1 runner gate) validates against.
write_fp() {
  local track="$1"
  local dir="${ARTIFACT_DIR}/${track}"
  mkdir -p "$dir"
  if [[ -f "${dir}/substrate_fingerprint.json" ]]; then
    echo "[serve] keeping existing fingerprint for ${track} (preflight-written)"
    return 0
  fi
  cat >"${dir}/substrate_fingerprint.json" <<JSON
{
  "track": "${track}",
  "substrate_model_id": "${VZ_SUBSTRATE_MODEL_ID}",
  "served_at": "${DATE_TAG}"
}
JSON
}
for track in raw ref-harness camel volvence-cold volvence pe-off eta-off active-learning-off lora-adapter; do
  write_fp "$track"
done

# --- wait for every endpoint to become healthy (model load can be slow) ---
wait_healthy() {
  local name="$1" url="$2" timeout="${3:-900}"
  local deadline=$(( $(date +%s) + timeout ))
  while (( $(date +%s) < deadline )); do
    if curl -sf -o /dev/null --max-time 5 "$url"; then
      echo "[serve] healthy ${name}: ${url}"
      return 0
    fi
    sleep 5
  done
  echo "[serve] ERROR: timed out waiting for ${name}: ${url}" >&2
  return 1
}
wait_healthy lifeform-companion "http://127.0.0.1:8000/v1/health"
wait_healthy lifeform-companion-cold "http://127.0.0.1:8001/v1/health"
wait_healthy lifeform-companion-pe-drive-off "http://127.0.0.1:8002/v1/health"
wait_healthy lifeform-companion-eta-off "http://127.0.0.1:8003/v1/health"
wait_healthy lifeform-companion-active-learning-off "http://127.0.0.1:8004/v1/health"
wait_healthy lifeform-companion-lora-adapter "http://127.0.0.1:8005/v1/health"
wait_healthy ref-harness "http://127.0.0.1:8500/healthz"
wait_healthy camel-baseline "http://127.0.0.1:8600/healthz"

echo "[serve] asserting same-substrate invariant..."
python scripts/companion_bench/assert_same_substrate.py \
  --fingerprint-file raw="${ARTIFACT_DIR}/raw/substrate_fingerprint.json" \
  --fingerprint-file ref-harness="${ARTIFACT_DIR}/ref-harness/substrate_fingerprint.json" \
  --fingerprint-file camel="${ARTIFACT_DIR}/camel/substrate_fingerprint.json" \
  --fingerprint-file volvence-cold="${ARTIFACT_DIR}/volvence-cold/substrate_fingerprint.json" \
  --fingerprint-file volvence="${ARTIFACT_DIR}/volvence/substrate_fingerprint.json" \
  --fingerprint-file pe-off="${ARTIFACT_DIR}/pe-off/substrate_fingerprint.json" \
  --fingerprint-file eta-off="${ARTIFACT_DIR}/eta-off/substrate_fingerprint.json" \
  --fingerprint-file active-learning-off="${ARTIFACT_DIR}/active-learning-off/substrate_fingerprint.json" \
  --fingerprint-file lora-adapter="${ARTIFACT_DIR}/lora-adapter/substrate_fingerprint.json"

echo "[serve] all endpoints launched. PIDs in ${PID_FILE}. Logs in ${LOG_DIR}."
echo "[serve] tear down with: bash scripts/companion_bench/stop_same_substrate_ablation.sh ${PID_FILE}"
