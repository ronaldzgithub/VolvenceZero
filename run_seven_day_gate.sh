#!/usr/bin/env bash
# Gate 8/11 七日陪伴证据的一键启动器。
#
# 首次运行：生成新的 preregistration、冻结只读 execution root，然后执行
# preflight -> one-run smoke -> 36-run formal -> independent audit。
#
# 续跑：必须显式传入同一份 preregistration、execution root 和 output root，
# 只允许从同一物证链上 --resume；旧 halt_record 目录会由控制面拒绝。
#
# 用法：
#   bash run_seven_day_gate.sh
#   bash run_seven_day_gate.sh --resume \
#     --preregistration artifacts/preregistrations/<name>.json \
#     --execution-root /private/tmp/volvence-seven-day-<name> \
#     --output-dir artifacts/seven-day-formal-<name>
#   bash run_seven_day_gate.sh --no-caffeinate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RESUME=0
CAFFEINATE=1
PREREGISTRATION=""
EXECUTION_ROOT=""
OUTPUT_DIR=""
MPS_LOCK="${REPO_ROOT}/artifacts/.companion-evidence-mps.lock"

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

options:
  --resume                 Resume an existing formal output root.
  --preregistration PATH   Preregistration JSON (required with --resume).
  --execution-root PATH   Frozen execution root (required with --resume).
  --output-dir PATH       Formal output root (required with --resume).
  --mps-lock PATH         Shared MPS lock path.
  --no-caffeinate          Do not keep the Mac awake during the run.
  -h, --help               Show this help.
EOF
}

resolve_repo_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME=1
      shift
      ;;
    --preregistration)
      [[ $# -ge 2 ]] || { echo "error: --preregistration requires a path" >&2; exit 2; }
      PREREGISTRATION="$2"
      shift 2
      ;;
    --execution-root)
      [[ $# -ge 2 ]] || { echo "error: --execution-root requires a path" >&2; exit 2; }
      EXECUTION_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "error: --output-dir requires a path" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --mps-lock)
      [[ $# -ge 2 ]] || { echo "error: --mps-lock requires a path" >&2; exit 2; }
      MPS_LOCK="$2"
      shift 2
      ;;
    --no-caffeinate)
      CAFFEINATE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

PREREG_SCRIPT="${REPO_ROOT}/scripts/preregister_seven_day_companion_simulated.py"
FREEZER_SCRIPT="${REPO_ROOT}/scripts/freeze_seven_day_execution_root.py"
if [[ ! -f "$PREREG_SCRIPT" || ! -f "$FREEZER_SCRIPT" ]]; then
  echo "error: seven-day preregistration or execution-root freezer is missing" >&2
  exit 2
fi

if [[ "$RESUME" -eq 1 ]]; then
  if [[ -z "$PREREGISTRATION" || -z "$EXECUTION_ROOT" || -z "$OUTPUT_DIR" ]]; then
    echo "error: --resume requires --preregistration, --execution-root, and --output-dir" >&2
    exit 2
  fi
  PREREGISTRATION="$(resolve_repo_path "$PREREGISTRATION")"
  EXECUTION_ROOT="$(resolve_repo_path "$EXECUTION_ROOT")"
  OUTPUT_DIR="$(resolve_repo_path "$OUTPUT_DIR")"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  PREREGISTRATION="${REPO_ROOT}/artifacts/preregistrations/seven-day-v3-${RUN_ID}.json"
  EXECUTION_ROOT="/private/tmp/volvence-seven-day-${RUN_ID}"
  OUTPUT_DIR="${REPO_ROOT}/artifacts/seven-day-formal-${RUN_ID}"
  mkdir -p "$(dirname "$PREREGISTRATION")" "$(dirname "$OUTPUT_DIR")"

  echo "[seven-day] creating preregistration: ${PREREGISTRATION}"
  "$PYTHON_BIN" "$PREREG_SCRIPT" \
    --repo-root "$REPO_ROOT" \
    --output "$PREREGISTRATION"

  echo "[seven-day] freezing execution root: ${EXECUTION_ROOT}"
  "$PYTHON_BIN" "$FREEZER_SCRIPT" \
    --repo-root "$REPO_ROOT" \
    --preregistration "$PREREGISTRATION" \
    --output-root "$EXECUTION_ROOT"
fi

if [[ ! -f "$PREREGISTRATION" ]]; then
  echo "error: preregistration does not exist: $PREREGISTRATION" >&2
  exit 2
fi
if [[ ! -d "$EXECUTION_ROOT" ]]; then
  echo "error: frozen execution root does not exist: $EXECUTION_ROOT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" && "$RESUME" -eq 0 ]]; then
  echo "error: output directory already exists; use --resume only for an interrupted run: $OUTPUT_DIR" >&2
  exit 2
fi

CONTROLLER="${EXECUTION_ROOT}/scripts/run_seven_day_companion_test_plan.py"
if [[ ! -f "$CONTROLLER" ]]; then
  echo "error: frozen execution root lacks the unified controller: $CONTROLLER" >&2
  exit 2
fi

echo "[seven-day] campaign=Gate-8/11"
echo "[seven-day] preregistration=${PREREGISTRATION}"
echo "[seven-day] execution_root=${EXECUTION_ROOT}"
echo "[seven-day] output_dir=${OUTPUT_DIR}"
echo "[seven-day] mps_lock=${MPS_LOCK}"
if [[ "$RESUME" -eq 1 ]]; then
  echo "[seven-day] mode=resume"
else
  echo "[seven-day] mode=new-run"
fi

controller_args=(
  "$PYTHON_BIN"
  "$CONTROLLER"
  all
  --execution-root "$EXECUTION_ROOT"
  --preregistration "$PREREGISTRATION"
  --output-dir "$OUTPUT_DIR"
  --mps-lock "$MPS_LOCK"
)
if [[ "$RESUME" -eq 1 ]]; then
  controller_args+=(--resume)
fi

if [[ "$CAFFEINATE" -eq 1 ]] && command -v caffeinate >/dev/null 2>&1; then
  echo "[seven-day] caffeinate=enabled"
  exec caffeinate -dimsu "${controller_args[@]}"
fi

echo "[seven-day] caffeinate=disabled"
exec "${controller_args[@]}"
