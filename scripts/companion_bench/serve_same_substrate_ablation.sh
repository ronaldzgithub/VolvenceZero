#!/usr/bin/env bash
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.
#
# Boot the nine same-substrate Companion Bench ablation endpoints, all sharing
# ONE frozen Qwen so any score delta is attributable to the layer over the
# substrate, not the substrate itself.
#
#   :8000  lifeform-serve --ablation-bundle           -> mode=lifeform + ?vertical=<track>
#                                                      -> mode=raw      (bare Qwen, the `raw` track)
#   :8500  companion-ref-harness                      -> upstream :8000/v1?mode=raw  (standard memory wrapper)
#   :8600  companion-camel-baseline                   -> upstream :8000/v1?mode=raw  (CAMEL agent framework)
#
# The ref-harness + camel backends point their upstream at :8000's mode=raw path,
# guaranteeing byte-identical weights with the `raw` track. All Volvence
# ablation arms share the same in-process frozen runtime behind :8000.
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
export ARTIFACT_DIR

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

# --- :8000 unified ablation bundle (mode=raw + six lifeform verticals) ---
start lifeform-ablation-bundle \
  lifeform-serve \
    --ablation-bundle \
    --port 8000 \
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

# --- :8501 memory-only arm (registry claim 2: standard memory wrapper WITHOUT
# retrieval — summary + user_model + episodic; H-A/H-C subset) ---
MEMORY_ONLY_ARGS=(
  companion-ref-harness serve
    --port 8501
    --upstream-base-url "$RAW_UPSTREAM"
    --upstream-model lifeform-raw
    --upstream-key-env LIFEFORM_LOCAL_API_KEY
    --components summary,user_model,episodic
    --store-mode sqlite
    --store-path "${ARTIFACT_DIR}/memory-only.sqlite3"
    --summary-extractor-base-url "${REFH_EXTRACTOR_BASE_URL}"
    --summary-extractor-model "$REFH_EXTRACTOR_MODEL"
    --summary-extractor-key-env "${REFH_EXTRACTOR_KEY_ENV}"
    --summary-extractor-family "${REFH_EXTRACTOR_FAMILY:-openai-compat}"
)
start memory-only "${MEMORY_ONLY_ARGS[@]}"

# --- :8502 RAG arm (registry claim 2: embed retrieval ONLY — H-B) ---
RAG_ARGS=(
  companion-ref-harness serve
    --port 8502
    --upstream-base-url "$RAW_UPSTREAM"
    --upstream-model lifeform-raw
    --upstream-key-env LIFEFORM_LOCAL_API_KEY
    --components embed
    --embedder "$REFH_EMBEDDER"
    --store-mode sqlite
    --store-path "${ARTIFACT_DIR}/rag.sqlite3"
)
start rag "${RAG_ARGS[@]}"

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
for track in raw ref-harness memory-only rag camel volvence-cold volvence pe-off eta-off active-learning-off lora-adapter; do
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
wait_healthy lifeform-ablation-bundle "http://127.0.0.1:8000/v1/health"
wait_healthy ref-harness "http://127.0.0.1:8500/healthz"
wait_healthy memory-only "http://127.0.0.1:8501/healthz"
wait_healthy rag "http://127.0.0.1:8502/healthz"
wait_healthy camel-baseline "http://127.0.0.1:8600/healthz"

python - <<'PY'
import json
import os
from pathlib import Path

artifact_dir = Path(os.environ["ARTIFACT_DIR"])
pid_file = artifact_dir / "serve.pids"
pids = [line.strip() for line in pid_file.read_text(encoding="utf-8").splitlines() if line.strip()]
payload = {
    "schema_version": "companion-ablation-serving-topology.v1",
    "serving_topology": "single-lifeform-ablation-bundle",
    "lifeform_owner_pid": int(pids[0]) if pids else None,
    "process_count": len(pids),
    "ports": [8000, 8500, 8501, 8502, 8600],
    "ablation_verticals": [
        "companion",
        "companion-cold",
        "companion-pe-drive-off",
        "companion-eta-off",
        "companion-active-learning-off",
        "companion-lora-adapter",
    ],
}
(artifact_dir / "serve_topology.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "[serve] asserting same-substrate invariant..."
python scripts/companion_bench/assert_same_substrate.py \
  --fingerprint-file raw="${ARTIFACT_DIR}/raw/substrate_fingerprint.json" \
  --fingerprint-file ref-harness="${ARTIFACT_DIR}/ref-harness/substrate_fingerprint.json" \
  --fingerprint-file memory-only="${ARTIFACT_DIR}/memory-only/substrate_fingerprint.json" \
  --fingerprint-file rag="${ARTIFACT_DIR}/rag/substrate_fingerprint.json" \
  --fingerprint-file camel="${ARTIFACT_DIR}/camel/substrate_fingerprint.json" \
  --fingerprint-file volvence-cold="${ARTIFACT_DIR}/volvence-cold/substrate_fingerprint.json" \
  --fingerprint-file volvence="${ARTIFACT_DIR}/volvence/substrate_fingerprint.json" \
  --fingerprint-file pe-off="${ARTIFACT_DIR}/pe-off/substrate_fingerprint.json" \
  --fingerprint-file eta-off="${ARTIFACT_DIR}/eta-off/substrate_fingerprint.json" \
  --fingerprint-file active-learning-off="${ARTIFACT_DIR}/active-learning-off/substrate_fingerprint.json" \
  --fingerprint-file lora-adapter="${ARTIFACT_DIR}/lora-adapter/substrate_fingerprint.json"

echo "[serve] all endpoints launched. PIDs in ${PID_FILE}. Logs in ${LOG_DIR}."
echo "[serve] tear down with: bash scripts/companion_bench/stop_same_substrate_ablation.sh ${PID_FILE}"
