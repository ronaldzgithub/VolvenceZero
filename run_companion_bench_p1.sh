#!/usr/bin/env bash
# macOS / Apple-silicon 一键启动 CompanionBench P1 same-substrate directional 跑分。
#
# 仓库根目录薄封装，委托给 SSOT 编排器：
#   scripts/companion_bench/run_p1_apple.sh
#
# 前置条件（首次使用前）：
#   1. .local/llm.env  — OPENROUTER_API_KEY 与 ABLATION_* judge/user-sim 配置
#   2. 本地 HF 缓存    — 默认 Qwen/Qwen2.5-1.5B-Instruct（可用 VZ_SUBSTRATE_MODEL_ID 覆盖）
#   3. companion bootstrap — packages/lifeform-domain-emogpt/.../bootstraps/*.snap|*.bs
#   4. Apple silicon + torch MPS（默认；可用 VZ_SUBSTRATE_DEVICE=cpu）
#   5. editable install：bash install.sh 且 VOLVENCE_EXTRAS=hf（或 scripts/launch_evidence_runs_m2.sh setup）
#
# 产物目录：artifacts/companion-ablation/<run-id>/
# 完成后查看：verdict_p1.json 与各轨 scores/*/summary.json
#
# Usage:
#   bash run_companion_bench_p1.sh
#   bash run_companion_bench_p1.sh --resume
#   bash run_companion_bench_p1.sh --dry-run
#   bash run_companion_bench_p1.sh --stop
#   bash run_companion_bench_p1.sh --artifact-dir artifacts/companion-ablation/<tag> --resume
#   bash run_companion_bench_p1.sh --keep-services
#
# run_p1_apple.sh 已内置 caffeinate -dimsu；运行期间请接电并保持开盖（合盖仍会睡眠）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RUNNER="${REPO_ROOT}/scripts/companion_bench/run_p1_apple.sh"
if [[ ! -f "$RUNNER" ]]; then
  echo "error: missing orchestrator: $RUNNER" >&2
  exit 2
fi

ARTIFACT_DIR=""
DRY_RUN=0
RESUME=0
KEEP_SERVICES=0
STOP=0

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      [[ $# -ge 2 ]] || { echo "error: --artifact-dir requires a path" >&2; exit 2; }
      ARTIFACT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --keep-services)
      KEEP_SERVICES=1
      shift
      ;;
    --stop)
      STOP=1
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

latest_ablation_run_dir() {
  local base="${REPO_ROOT}/artifacts/companion-ablation"
  [[ -d "$base" ]] || return 1
  # Prefer newest directory by mtime (GNU/BSD find portable via python).
  python - "$base" <<'PY'
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
dirs = [p for p in base.iterdir() if p.is_dir()]
if not dirs:
    raise SystemExit(1)
latest = max(dirs, key=lambda p: p.stat().st_mtime)
print(latest.resolve())
PY
}

stop_ablation_run() {
  local run_dir="$1"
  local pid_file="${run_dir}/serve.pids"
  if [[ ! -f "$pid_file" ]]; then
    echo "[p1] no serve.pids under ${run_dir}"
    return 0
  fi
  bash "${REPO_ROOT}/scripts/companion_bench/stop_same_substrate_ablation.sh" "$pid_file"
  echo "[p1] stopped services from ${pid_file}"
}

if [[ "$STOP" -eq 1 ]]; then
  if [[ -z "$ARTIFACT_DIR" ]]; then
    if ! ARTIFACT_DIR="$(latest_ablation_run_dir)"; then
      echo "error: no artifacts/companion-ablation run found to stop" >&2
      exit 2
    fi
  elif [[ "$ARTIFACT_DIR" != /* ]]; then
    ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_DIR}"
  fi
  stop_ablation_run "$ARTIFACT_DIR"
  exit 0
fi

if [[ "$RESUME" -eq 1 && -z "$ARTIFACT_DIR" ]]; then
  if ARTIFACT_DIR="$(latest_ablation_run_dir)"; then
    echo "[p1] --resume without --artifact-dir; using latest run: ${ARTIFACT_DIR}"
  fi
fi

invoke_args=()
if [[ -n "$ARTIFACT_DIR" ]]; then
  if [[ "$ARTIFACT_DIR" != /* ]]; then
    ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_DIR}"
  fi
  invoke_args+=(--artifact-dir "$ARTIFACT_DIR")
fi
[[ "$DRY_RUN" -eq 1 ]] && invoke_args+=(--dry-run)
[[ "$RESUME" -eq 1 ]] && invoke_args+=(--resume)
[[ "$KEEP_SERVICES" -eq 1 ]] && invoke_args+=(--keep-services)

echo "[p1] repo=${REPO_ROOT}"
echo "[p1] orchestrator=${RUNNER}"
if [[ -n "$ARTIFACT_DIR" ]]; then
  echo "[p1] artifact-dir=${ARTIFACT_DIR}"
fi

# bash 3.2 (macOS /bin/bash) + set -u treats empty "${arr[@]}" as unbound.
bash "$RUNNER" ${invoke_args[@]+"${invoke_args[@]}"}
