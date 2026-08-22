#!/usr/bin/env bash
set -euo pipefail

VOLVENCE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOLVENCE_SOURCE_PATHS=()
for VOLVENCE_SOURCE_DIR in "${VOLVENCE_SCRIPT_DIR}"/packages/*/src; do
  if [[ -d "${VOLVENCE_SOURCE_DIR}" ]]; then
    VOLVENCE_SOURCE_PATHS+=("${VOLVENCE_SOURCE_DIR}")
  fi
done
if [[ ${#VOLVENCE_SOURCE_PATHS[@]} -eq 0 ]]; then
  echo "No workspace package sources found under ${VOLVENCE_SCRIPT_DIR}/packages." >&2
  exit 1
fi
VOLVENCE_WORKSPACE_PYTHONPATH="$(IFS=:; echo "${VOLVENCE_SOURCE_PATHS[*]}")"
if [[ -n "${PYTHONPATH:-}" ]]; then
  VOLVENCE_WORKSPACE_PYTHONPATH="${VOLVENCE_WORKSPACE_PYTHONPATH}:${PYTHONPATH}"
fi
export PYTHONPATH="${VOLVENCE_WORKSPACE_PYTHONPATH}"

if [[ $# -eq 0 ]]; then
  set -- --prepare
fi
exec "${VOLVENCE_PYTHON_BIN:-python}" \
  -m lifeform_domain_emogpt.lab.p4_canary_cli "$@"
