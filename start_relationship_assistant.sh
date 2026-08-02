#!/usr/bin/env bash
# Start the product relationship assistant with persistent per-user memory.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 is required to start the relationship assistant." >&2
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8877}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
DEVICE="${DEVICE:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
MEMORY_SCOPE_ROOT_DIR="${MEMORY_SCOPE_ROOT_DIR:-${HOME}/.volvence/alpha-memory}"
EVIDENCE_ROOT_DIR="${EVIDENCE_ROOT_DIR:-${HOME}/.volvence/alpha-evidence}"

if [[ "${LOCAL_FILES_ONLY}" != "0" && "${LOCAL_FILES_ONLY}" != "1" ]]; then
  echo "LOCAL_FILES_ONLY must be 0 or 1." >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1 \
  && lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use. Stop that service or set PORT to another port." >&2
  exit 1
fi

SOURCE_DIRS=()
for source_dir in "${ROOT_DIR}"/packages/*/src; do
  if [[ -d "${source_dir}" ]]; then
    SOURCE_DIRS+=("${source_dir}")
  fi
done
if [[ "${#SOURCE_DIRS[@]}" -eq 0 ]]; then
  echo "No workspace package source directories were found under ${ROOT_DIR}/packages." >&2
  exit 1
fi
PACKAGE_PATHS="$(IFS=:; printf '%s' "${SOURCE_DIRS[*]}")"
export PYTHONPATH="${PACKAGE_PATHS}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${MEMORY_SCOPE_ROOT_DIR}" "${EVIDENCE_ROOT_DIR}"

SERVICE_ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --vertical companion
  --alpha-enabled
  --memory-scope-root-dir "${MEMORY_SCOPE_ROOT_DIR}"
  --evidence-root-dir "${EVIDENCE_ROOT_DIR}"
  --substrate-mode hf-shared
  --substrate-model-id "${MODEL_ID}"
  --substrate-device "${DEVICE}"
)
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  SERVICE_ARGS+=(--substrate-local-files-only)
fi

echo "[relationship-assistant] python=${PYTHON_BIN}"
echo "[relationship-assistant] model=${MODEL_ID} device=${DEVICE}"
echo "[relationship-assistant] memory=${MEMORY_SCOPE_ROOT_DIR}"
echo "[relationship-assistant] chat=http://${HOST}:${PORT}/chat"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m lifeform_service.cli "${SERVICE_ARGS[@]}"
