#!/usr/bin/env bash
# Start the local Forge → Praxist → SHADOW → ACTIVE Research Lab workbench.

set -euo pipefail

RESEARCH_LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RESEARCH_LAB_HOST="127.0.0.1"
RESEARCH_LAB_API_PORT="${RESEARCH_LAB_API_PORT:-8766}"
RESEARCH_LAB_WEB_PORT="${RESEARCH_LAB_WEB_PORT:-3000}"
RESEARCH_LAB_MODE="controlled"
RESEARCH_LAB_OPEN_BROWSER="0"
RESEARCH_LAB_AUTO_RESEARCH="${RESEARCH_LAB_AUTO_RESEARCH:-1}"
RESEARCH_LAB_DISCOVERY_INTERVAL="${RESEARCH_LAB_DISCOVERY_INTERVAL:-300}"
RESEARCH_LAB_DISCOVERY_MODEL="${RESEARCH_LAB_DISCOVERY_MODEL:-gpt-5.6-luna}"

usage() {
  cat <<'EOF'
Usage: ./start_research_lab.sh [options]

Starts the loopback Research Lab API, Vinext workbench, and bounded automatic
research worker in the foreground. The worker enforces registered Portfolio
dependency gates and never approves Binding or A0.

Options:
  --read-only         Disable all POST command delegation.
  --no-auto-research  Disable periodic Demand discovery and approved reconcile.
  --api-port PORT     API port (default: 8766).
  --web-port PORT     Web port (default: 3000).
  --open              Open the local workbench in the default browser.
  -h, --help          Show this help.

Optional environment:
  RESEARCH_LAB_PYTHON     Python >=3.11 executable (default: .venv/bin/python).
  RESEARCH_LAB_NODE       Node >=22.13 executable (Codex native is auto-detected).
  FORGE_PRAXIST_EXECUTABLE
                          Shared Forge/Lab Praxist executable override (highest priority).
  RESEARCH_LAB_PRAXIST    Compatible lab-only override when the shared override is unset.
  RESEARCH_LAB_FOUNDRY_ROOT  Read-only Foundry descriptor root (sibling checkout is auto-detected).
  RESEARCH_LAB_API_PORT   API port override.
  RESEARCH_LAB_WEB_PORT   Web port override.
  RESEARCH_LAB_AUTO_RESEARCH  1 to run the worker, 0 to disable it (default: 1).
  RESEARCH_LAB_DISCOVERY_INTERVAL  Seconds between bounded passes (default: 300).
  RESEARCH_LAB_DISCOVERY_MODEL  Exact Codex model (default: gpt-5.6-luna).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --read-only)
      RESEARCH_LAB_MODE="read-only"
      shift
      ;;
    --no-auto-research)
      RESEARCH_LAB_AUTO_RESEARCH="0"
      shift
      ;;
    --api-port)
      [[ $# -ge 2 ]] || { echo "--api-port requires a value" >&2; exit 2; }
      RESEARCH_LAB_API_PORT="$2"
      shift 2
      ;;
    --web-port)
      [[ $# -ge 2 ]] || { echo "--web-port requires a value" >&2; exit 2; }
      RESEARCH_LAB_WEB_PORT="$2"
      shift 2
      ;;
    --open)
      RESEARCH_LAB_OPEN_BROWSER="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

validate_port() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] \
    || (( 10#${value} < 1 || 10#${value} > 65535 )); then
    echo "${label} must be an integer between 1 and 65535: ${value}" >&2
    exit 2
  fi
}

validate_port "API port" "${RESEARCH_LAB_API_PORT}"
validate_port "Web port" "${RESEARCH_LAB_WEB_PORT}"
if [[ "${RESEARCH_LAB_API_PORT}" == "${RESEARCH_LAB_WEB_PORT}" ]]; then
  echo "API and Web ports must be different." >&2
  exit 2
fi
if [[ "${RESEARCH_LAB_AUTO_RESEARCH}" != "0" && "${RESEARCH_LAB_AUTO_RESEARCH}" != "1" ]]; then
  echo "RESEARCH_LAB_AUTO_RESEARCH must be 0 or 1." >&2
  exit 2
fi
if [[ ! "${RESEARCH_LAB_DISCOVERY_INTERVAL}" =~ ^[0-9]+$ ]] \
  || (( 10#${RESEARCH_LAB_DISCOVERY_INTERVAL} < 10 )); then
  echo "RESEARCH_LAB_DISCOVERY_INTERVAL must be an integer of at least 10 seconds." >&2
  exit 2
fi
if [[ -z "${RESEARCH_LAB_DISCOVERY_MODEL}" ]]; then
  echo "RESEARCH_LAB_DISCOVERY_MODEL must be non-empty." >&2
  exit 2
fi

if [[ -n "${RESEARCH_LAB_PYTHON:-}" ]]; then
  RESEARCH_LAB_PYTHON_BIN="${RESEARCH_LAB_PYTHON}"
elif [[ -x "${RESEARCH_LAB_ROOT}/.venv/bin/python" ]]; then
  RESEARCH_LAB_PYTHON_BIN="${RESEARCH_LAB_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  RESEARCH_LAB_PYTHON_BIN="$(command -v python3)"
else
  echo "Python >=3.11 is required. Set RESEARCH_LAB_PYTHON." >&2
  exit 1
fi

if [[ ! -x "${RESEARCH_LAB_PYTHON_BIN}" ]]; then
  echo "Research Lab Python is not executable: ${RESEARCH_LAB_PYTHON_BIN}" >&2
  exit 1
fi
if ! "${RESEARCH_LAB_PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Research Lab requires Python >=3.11: ${RESEARCH_LAB_PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${RESEARCH_LAB_MODE}" == "controlled" && "${RESEARCH_LAB_AUTO_RESEARCH}" == "1" ]] \
  && ! "${RESEARCH_LAB_PYTHON_BIN}" -c 'import openai_codex'; then
  echo "Automatic research requires the openai-codex SDK in RESEARCH_LAB_PYTHON." >&2
  exit 1
fi

RESEARCH_LAB_CODEX_NODE=""
IFS=':' read -r -a RESEARCH_LAB_PATH_ENTRIES <<< "${PATH}"
for path_entry in "${RESEARCH_LAB_PATH_ENTRIES[@]}"; do
  case "${path_entry}" in
    */codex-primary-runtime/dependencies/bin/override|*/codex-primary-runtime/dependencies/bin/fallback)
      node_candidate="${path_entry}/../../node/bin/node"
      if [[ -x "${node_candidate}" ]]; then
        RESEARCH_LAB_CODEX_NODE="$(cd "$(dirname "${node_candidate}")" && pwd -P)/node"
        break
      fi
      ;;
  esac
done

if [[ -n "${RESEARCH_LAB_NODE:-}" ]]; then
  RESEARCH_LAB_NODE_BIN="${RESEARCH_LAB_NODE}"
elif [[ -n "${RESEARCH_LAB_CODEX_NODE}" ]]; then
  RESEARCH_LAB_NODE_BIN="${RESEARCH_LAB_CODEX_NODE}"
elif command -v node >/dev/null 2>&1; then
  RESEARCH_LAB_NODE_BIN="$(command -v node)"
else
  echo "Node >=22.13 is required. Set RESEARCH_LAB_NODE." >&2
  exit 1
fi

if [[ ! -x "${RESEARCH_LAB_NODE_BIN}" ]]; then
  echo "Research Lab Node is not executable: ${RESEARCH_LAB_NODE_BIN}" >&2
  exit 1
fi
RESEARCH_LAB_NODE_VERSION="$("${RESEARCH_LAB_NODE_BIN}" -p 'process.versions.node')"
RESEARCH_LAB_NODE_MAJOR="${RESEARCH_LAB_NODE_VERSION%%.*}"
RESEARCH_LAB_NODE_REMAINDER="${RESEARCH_LAB_NODE_VERSION#*.}"
RESEARCH_LAB_NODE_MINOR="${RESEARCH_LAB_NODE_REMAINDER%%.*}"
if (( RESEARCH_LAB_NODE_MAJOR < 22 )) \
  || (( RESEARCH_LAB_NODE_MAJOR == 22 && RESEARCH_LAB_NODE_MINOR < 13 )); then
  echo "Research Lab requires Node >=22.13, got ${RESEARCH_LAB_NODE_VERSION}." >&2
  echo "Set RESEARCH_LAB_NODE to a compatible executable." >&2
  exit 1
fi
RESEARCH_LAB_NODE_DIR="$(cd "$(dirname "${RESEARCH_LAB_NODE_BIN}")" && pwd -P)"
RESEARCH_LAB_NPM_BIN="${RESEARCH_LAB_NODE_DIR}/npm"
if [[ ! -x "${RESEARCH_LAB_NPM_BIN}" ]]; then
  if command -v npm >/dev/null 2>&1; then
    RESEARCH_LAB_NPM_BIN="$(command -v npm)"
  else
    echo "npm was not found for the selected Node runtime." >&2
    exit 1
  fi
fi

if [[ -n "${FORGE_PRAXIST_EXECUTABLE:-}" ]]; then
  RESEARCH_LAB_PRAXIST_BIN="${FORGE_PRAXIST_EXECUTABLE}"
elif [[ -n "${RESEARCH_LAB_PRAXIST:-}" ]]; then
  RESEARCH_LAB_PRAXIST_BIN="${RESEARCH_LAB_PRAXIST}"
elif [[ -x "${RESEARCH_LAB_ROOT}/../PRAXIST/.venv/bin/praxist" ]]; then
  RESEARCH_LAB_PRAXIST_BIN="${RESEARCH_LAB_ROOT}/../PRAXIST/.venv/bin/praxist"
elif command -v praxist >/dev/null 2>&1; then
  RESEARCH_LAB_PRAXIST_BIN="$(command -v praxist)"
elif [[ -x "${HOME}/.venvs/praxist/bin/praxist" ]]; then
  RESEARCH_LAB_PRAXIST_BIN="${HOME}/.venvs/praxist/bin/praxist"
else
  echo "Praxist executable was not found. Set FORGE_PRAXIST_EXECUTABLE or RESEARCH_LAB_PRAXIST." >&2
  exit 1
fi
if [[ ! -x "${RESEARCH_LAB_PRAXIST_BIN}" ]]; then
  echo "Praxist is not executable: ${RESEARCH_LAB_PRAXIST_BIN}" >&2
  exit 1
fi

RESEARCH_LAB_WEB_ROOT="${RESEARCH_LAB_ROOT}/research/labs/web"
if [[ ! -x "${RESEARCH_LAB_WEB_ROOT}/node_modules/.bin/vinext" ]]; then
  echo "Research Lab web dependencies are missing." >&2
  echo "Run npm install in ${RESEARCH_LAB_WEB_ROOT}." >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for bounded local readiness checks." >&2
  exit 1
fi

RESEARCH_LAB_FOUNDRY_ROOT_RESOLVED=""
if [[ -n "${RESEARCH_LAB_FOUNDRY_ROOT:-}" ]]; then
  if [[ ! -d "${RESEARCH_LAB_FOUNDRY_ROOT}" ]]; then
    echo "Foundry root is not a directory: ${RESEARCH_LAB_FOUNDRY_ROOT}" >&2
    exit 1
  fi
  RESEARCH_LAB_FOUNDRY_ROOT_RESOLVED="$(cd "${RESEARCH_LAB_FOUNDRY_ROOT}" && pwd -P)"
elif [[ -f "${RESEARCH_LAB_ROOT}/../foundry/schemas/research_lab_intent.schema.json" ]]; then
  RESEARCH_LAB_FOUNDRY_ROOT_RESOLVED="$(cd "${RESEARCH_LAB_ROOT}/../foundry" && pwd -P)"
fi

assert_port_free() {
  local port="$1"
  local label="$2"
  if command -v lsof >/dev/null 2>&1 \
    && lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "${label} port ${port} is already in use; refusing to stack another Lab process." >&2
    exit 1
  fi
}

assert_port_free "${RESEARCH_LAB_API_PORT}" "API"
assert_port_free "${RESEARCH_LAB_WEB_PORT}" "Web"

RESEARCH_LAB_API_PID=""
RESEARCH_LAB_WEB_PID=""
RESEARCH_LAB_WORKER_PID=""
cleanup() {
  trap - EXIT INT TERM
  for child_pid in "${RESEARCH_LAB_WORKER_PID}" "${RESEARCH_LAB_WEB_PID}" "${RESEARCH_LAB_API_PID}"; do
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" >/dev/null 2>&1; then
      kill "${child_pid}" >/dev/null 2>&1 || true
    fi
  done
  for child_pid in "${RESEARCH_LAB_WORKER_PID}" "${RESEARCH_LAB_WEB_PID}" "${RESEARCH_LAB_API_PID}"; do
    if [[ -n "${child_pid}" ]]; then
      wait "${child_pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

RESEARCH_LAB_API_ARGS=(
  -m volvence_labs.cli
  lab-server
  --repo-root "${RESEARCH_LAB_ROOT}"
  --host "${RESEARCH_LAB_HOST}"
  --port "${RESEARCH_LAB_API_PORT}"
  --praxist-executable "${RESEARCH_LAB_PRAXIST_BIN}"
)
if [[ "${RESEARCH_LAB_MODE}" == "controlled" ]]; then
  RESEARCH_LAB_API_ARGS+=(
    --enable-mutations
    --forge-python "${RESEARCH_LAB_PYTHON_BIN}"
    --ui-origin "http://localhost:${RESEARCH_LAB_WEB_PORT}"
    --ui-origin "http://127.0.0.1:${RESEARCH_LAB_WEB_PORT}"
  )
  if [[ -n "${RESEARCH_LAB_FOUNDRY_ROOT_RESOLVED}" ]]; then
    RESEARCH_LAB_API_ARGS+=(
      --external-domain-root "foundry=${RESEARCH_LAB_FOUNDRY_ROOT_RESOLVED}"
    )
  fi
fi

echo "[research-lab] mode=${RESEARCH_LAB_MODE}"
echo "[research-lab] python=${RESEARCH_LAB_PYTHON_BIN}"
echo "[research-lab] node=${RESEARCH_LAB_NODE_BIN} (${RESEARCH_LAB_NODE_VERSION})"
echo "[research-lab] praxist=${RESEARCH_LAB_PRAXIST_BIN}"
if [[ -n "${RESEARCH_LAB_FOUNDRY_ROOT_RESOLVED}" ]]; then
  echo "[research-lab] external foundry root=${RESEARCH_LAB_FOUNDRY_ROOT_RESOLVED} (read-only ingress)"
fi
echo "[research-lab] no Praxist start occurs until an exact A0 review and Forge reconcile are submitted"

PYTHONUNBUFFERED=1 \
PYTHONPATH="${RESEARCH_LAB_ROOT}/research/labs/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${RESEARCH_LAB_PYTHON_BIN}" "${RESEARCH_LAB_API_ARGS[@]}" &
RESEARCH_LAB_API_PID="$!"

(
  cd "${RESEARCH_LAB_WEB_ROOT}"
  PATH="${RESEARCH_LAB_NODE_DIR}:${PATH}" \
  RESEARCH_LAB_API_ORIGIN="http://${RESEARCH_LAB_HOST}:${RESEARCH_LAB_API_PORT}" \
    "${RESEARCH_LAB_NPM_BIN}" run dev -- --host "${RESEARCH_LAB_HOST}" --port "${RESEARCH_LAB_WEB_PORT}"
) &
RESEARCH_LAB_WEB_PID="$!"

wait_for_url() {
  local url="$1"
  local label="$2"
  local request_timeout="$3"
  local request_method="$4"
  local attempt
  for attempt in $(seq 1 60); do
    if [[ "${request_method}" == "HEAD" ]]; then
      if curl --head --fail --silent --show-error \
        --max-time "${request_timeout}" "${url}" >/dev/null 2>&1; then
        return 0
      fi
    elif curl --fail --silent --show-error \
      --max-time "${request_timeout}" "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "${label} did not become ready: ${url}" >&2
  return 1
}

wait_for_url "http://${RESEARCH_LAB_HOST}:${RESEARCH_LAB_API_PORT}/healthz" "Research Lab API" 2 GET
wait_for_url "http://localhost:${RESEARCH_LAB_WEB_PORT}/" "Research Lab Web" 15 HEAD

if [[ "${RESEARCH_LAB_MODE}" == "controlled" && "${RESEARCH_LAB_AUTO_RESEARCH}" == "1" ]]; then
  (
    while true; do
      echo "[research-lab] automatic managed research bounded pass"
      if ! PYTHONPATH="${RESEARCH_LAB_ROOT}/forge/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${RESEARCH_LAB_PYTHON_BIN}" -m volvence_forge.cli \
          --repo-root "${RESEARCH_LAB_ROOT}" \
          research-managed-loop \
          --once \
          --backend codex_sdk \
          --model "${RESEARCH_LAB_DISCOVERY_MODEL}"; then
        echo "[research-lab] automatic research pass failed closed; retrying after interval" >&2
      fi
      sleep "${RESEARCH_LAB_DISCOVERY_INTERVAL}"
    done
  ) &
  RESEARCH_LAB_WORKER_PID="$!"
  echo "[research-lab] automatic research enabled: model=${RESEARCH_LAB_DISCOVERY_MODEL} interval=${RESEARCH_LAB_DISCOVERY_INTERVAL}s"
else
  echo "[research-lab] automatic research disabled"
fi

echo "[research-lab] ready: http://localhost:${RESEARCH_LAB_WEB_PORT}/"
echo "[research-lab] press Ctrl-C to stop both local processes"
if [[ "${RESEARCH_LAB_OPEN_BROWSER}" == "1" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "http://localhost:${RESEARCH_LAB_WEB_PORT}/"
  else
    echo "[research-lab] browser open command is unavailable; use the printed URL" >&2
  fi
fi

while kill -0 "${RESEARCH_LAB_API_PID}" >/dev/null 2>&1 \
  && kill -0 "${RESEARCH_LAB_WEB_PID}" >/dev/null 2>&1 \
  && { [[ -z "${RESEARCH_LAB_WORKER_PID}" ]] || kill -0 "${RESEARCH_LAB_WORKER_PID}" >/dev/null 2>&1; }; do
  sleep 1
done

RESEARCH_LAB_EXIT_STATUS=1
if [[ -n "${RESEARCH_LAB_WORKER_PID}" ]] \
  && ! kill -0 "${RESEARCH_LAB_WORKER_PID}" >/dev/null 2>&1; then
  wait "${RESEARCH_LAB_WORKER_PID}" || RESEARCH_LAB_EXIT_STATUS="$?"
  echo "[research-lab] automatic research worker exited unexpectedly with status ${RESEARCH_LAB_EXIT_STATUS}" >&2
elif ! kill -0 "${RESEARCH_LAB_API_PID}" >/dev/null 2>&1; then
  wait "${RESEARCH_LAB_API_PID}" || RESEARCH_LAB_EXIT_STATUS="$?"
  echo "[research-lab] API exited unexpectedly with status ${RESEARCH_LAB_EXIT_STATUS}" >&2
else
  wait "${RESEARCH_LAB_WEB_PID}" || RESEARCH_LAB_EXIT_STATUS="$?"
  echo "[research-lab] Web exited unexpectedly with status ${RESEARCH_LAB_EXIT_STATUS}" >&2
fi
exit "${RESEARCH_LAB_EXIT_STATUS}"
