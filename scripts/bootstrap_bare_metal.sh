#!/usr/bin/env bash
# Volvence Zero — bare-metal bootstrap (Linux)
#
# One-shot installer for a fresh Linux server: system packages, Python venv,
# full workspace wheels, smoke test, optional systemd unit.
#
# Typical usage (already cloned repo):
#   sudo ./scripts/bootstrap_bare_metal.sh --profile hf --install-systemd
#
# Clone + install to /opt:
#   sudo VOLVENCE_REPO_URL=https://github.com/you/VolvenceZero.git \
#        ./scripts/bootstrap_bare_metal.sh --install-dir /opt/volvence-zero
#
# Profiles:
#   synthetic  — default runtime, no torch (fast CI / API without GPU)
#   hf         — torch + transformers for Qwen / hf-shared production
#   full       — hf + torch training extras (SSL/RL, peft)
#   dev        — synthetic + pytest + ruff
#
# Environment overrides (all optional):
#   VOLVENCE_REPO_URL          git remote to clone when --install-dir is empty
#   VOLVENCE_REPO_REF          branch/tag/commit (default: current branch or main)
#   VOLVENCE_INSTALL_DIR       checkout root (default: repo root containing this script)
#   VOLVENCE_VENV_DIR          venv path (default: $INSTALL_DIR/.venv)
#   VOLVENCE_PROFILE           synthetic|hf|full|dev
#   VOLVENCE_PYTHON            explicit python binary
#   VOLVENCE_SKIP_SYSTEM=1     skip apt/dnf system package step
#   VOLVENCE_SKIP_SMOKE=1      skip post-install smoke test
#   VOLVENCE_INSTALL_SYSTEMD=1 install lifeform-serve systemd unit
#   VOLVENCE_SYSTEMD_USER=1    user systemd unit instead of system unit
#   VOLVENCE_SERVICE_USER       unix user for system unit (default: volvence)
#   VOLVENCE_SUBSTRATE_MODE     synthetic|hf-shared (for systemd unit)
#   VOLVENCE_SUBSTRATE_MODEL    HF model id (default: Qwen/Qwen2.5-0.5B-Instruct)
#   VOLVENCE_SERVICE_PORT       bind port (default: 8765)
#   VOLVENCE_SERVICE_HOST       bind host (default: 0.0.0.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROFILE="${VOLVENCE_PROFILE:-synthetic}"
INSTALL_DIR="${VOLVENCE_INSTALL_DIR:-${DEFAULT_REPO_ROOT}}"
VENV_DIR="${VOLVENCE_VENV_DIR:-${INSTALL_DIR}/.venv}"
REPO_URL="${VOLVENCE_REPO_URL:-}"
REPO_REF="${VOLVENCE_REPO_REF:-}"
PYTHON_BIN="${VOLVENCE_PYTHON:-}"
SKIP_SYSTEM=0
SKIP_SMOKE=0
INSTALL_SYSTEMD=0
SYSTEMD_USER=0
SERVICE_USER="${VOLVENCE_SERVICE_USER:-volvence}"
SUBSTRATE_MODE="${VOLVENCE_SUBSTRATE_MODE:-synthetic}"
SUBSTRATE_MODEL="${VOLVENCE_SUBSTRATE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVICE_PORT="${VOLVENCE_SERVICE_PORT:-8765}"
SERVICE_HOST="${VOLVENCE_SERVICE_HOST:-0.0.0.0}"
DRY_RUN=0

usage() {
  sed -n '2,36p' "$0" | sed 's/^# \?//'
  cat <<'EOF'

Options:
  --profile PROFILE       synthetic | hf | full | dev  (default: synthetic)
  --install-dir PATH      checkout / install root
  --venv PATH             Python venv directory
  --repo-url URL          clone into --install-dir when it has no install.sh
  --repo-ref REF          branch, tag, or commit for clone
  --python BIN            python executable (must be >= 3.11)
  --skip-system-deps      do not install OS packages (requires python >= 3.11 present)
  --skip-smoke-test       skip post-install brain smoke test
  --install-systemd       register lifeform-serve systemd unit
  --systemd-user          install a user unit (~/.config/systemd/user) instead of /etc/systemd/system
  --service-user NAME     unix account for system unit (default: volvence)
  --substrate-mode MODE   synthetic | hf-shared (systemd only)
  --substrate-model ID    Hugging Face model id (systemd, hf-shared only)
  --service-port PORT     HTTP bind port (default: 8765)
  --service-host HOST     HTTP bind host (default: 0.0.0.0)
  --dry-run               print planned actions without executing
  -h, --help              show this help
EOF
}

log() { printf '[bootstrap] %s\n' "$*"; }
warn() { printf '[bootstrap] WARN: %s\n' "$*" >&2; }
die() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

run() {
  log "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

maybe_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    if command -v sudo >/dev/null 2>&1; then
      sudo "$@"
    else
      die "root privileges required for: $*"
    fi
  fi
}

python_version_ok() {
  local bin="$1"
  "$bin" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

detect_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    python_version_ok "$PYTHON_BIN" || die "$PYTHON_BIN is not Python >= 3.11"
    return 0
  fi
  local candidate
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && python_version_ok "$candidate"; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done
  return 1
}

detect_os_id() {
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "${ID:-unknown}"
    return 0
  fi
  echo "unknown"
}

install_system_packages() {
  local os_id
  os_id="$(detect_os_id)"
  log "Installing OS packages for: ${os_id}"

  case "$os_id" in
    ubuntu|debian)
      maybe_sudo apt-get update -y
      local pkgs=(git curl ca-certificates build-essential pkg-config \
        libxml2-dev libxslt1-dev zlib1g-dev)
      if command -v python3.12 >/dev/null 2>&1; then
        pkgs+=(python3.12 python3.12-venv python3.12-dev)
      elif command -v python3.11 >/dev/null 2>&1; then
        pkgs+=(python3.11 python3.11-venv python3.11-dev)
      else
        # Ubuntu 22.04 ships 3.10 by default — pull 3.11 from deadsnakes.
        maybe_sudo apt-get install -y software-properties-common
        maybe_sudo add-apt-repository -y ppa:deadsnakes/ppa
        maybe_sudo apt-get update -y
        pkgs+=(python3.11 python3.11-venv python3.11-dev)
      fi
      maybe_sudo apt-get install -y "${pkgs[@]}"
      ;;
    rhel|centos|rocky|almalinux|fedora)
      maybe_sudo dnf install -y git curl gcc gcc-c++ make pkgconfig \
        libxml2-devel libxslt-devel zlib-devel
      if command -v python3.12 >/dev/null 2>&1; then
        maybe_sudo dnf install -y python3.12 python3.12-devel
      elif command -v python3.11 >/dev/null 2>&1; then
        maybe_sudo dnf install -y python3.11 python3.11-devel
      else
        maybe_sudo dnf install -y python3.11 python3.11-devel || \
          die "Could not install Python 3.11 via dnf; set VOLVENCE_PYTHON manually."
      fi
      ;;
    *)
      warn "Unknown distro '${os_id}'. Skipping automatic OS package install."
      warn "Ensure Python >= 3.11, git, curl, and libxml2/libxslt dev headers are present."
      return 0
      ;;
  esac
}

ensure_repo() {
  if [[ -f "${INSTALL_DIR}/install.sh" ]]; then
    log "Using existing checkout: ${INSTALL_DIR}"
    return 0
  fi

  [[ -n "$REPO_URL" ]] || die \
    "${INSTALL_DIR} is not a Volvence Zero checkout and VOLVENCE_REPO_URL / --repo-url was not set."

  log "Cloning ${REPO_URL} -> ${INSTALL_DIR}"
  run mkdir -p "$(dirname "${INSTALL_DIR}")"
  if [[ -d "${INSTALL_DIR}/.git" ]]; then
    run git -C "${INSTALL_DIR}" fetch --all --tags
  else
    run git clone "${REPO_URL}" "${INSTALL_DIR}"
  fi

  local ref="${REPO_REF:-main}"
  run git -C "${INSTALL_DIR}" checkout "${ref}"
}

create_venv() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    log "Reusing venv: ${VENV_DIR}"
  else
    log "Creating venv: ${VENV_DIR}"
    run "$PYTHON_BIN" -m venv "${VENV_DIR}"
  fi
  PYTHON_BIN="${VENV_DIR}/bin/python"
  run "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
}

resolve_extras() {
  case "$PROFILE" in
    synthetic) VOLVENCE_EXTRAS="" ;;
    hf)        VOLVENCE_EXTRAS="hf" ;;
    full)      VOLVENCE_EXTRAS="hf,torch" ;;
    dev)       VOLVENCE_EXTRAS="" ;;
    *) die "Unknown profile: ${PROFILE} (expected synthetic|hf|full|dev)" ;;
  esac
}

install_workspace() {
  log "Installing workspace wheels (profile=${PROFILE})"
  cd "${INSTALL_DIR}"
  export PYTHON="$PYTHON_BIN"
  export VOLVENCE_EXTRAS
  run bash ./install.sh

  if [[ "$PROFILE" == "dev" ]]; then
    run "$PYTHON_BIN" -m pip install -e ".[dev]"
  fi
}

run_smoke_test() {
  log "Running brain kernel smoke test"
  run "$PYTHON_BIN" - <<'PY'
from volvence_zero.brain import Brain, BrainConfig

session = Brain(BrainConfig()).create_session(session_id="bootstrap-smoke")
result = session.run_turn("I need help making a careful decision.")
text = (result.response.text or "").strip()
if not text:
    raise SystemExit("smoke test returned empty response")
print(text[:200])
PY
}

verify_console_scripts() {
  local scripts=(lifeform-serve volvence-zero)
  local name
  for name in "${scripts[@]}"; do
    if ! command -v "${VENV_DIR}/bin/${name}" >/dev/null 2>&1; then
      die "Expected console script missing: ${name}"
    fi
  done
  log "Console scripts OK: ${scripts[*]}"
}

gpu_hint() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    if [[ "$PROFILE" == "synthetic" ]]; then
      warn "GPU present but profile=synthetic. Use --profile hf for Qwen/hf-shared production."
    fi
  elif [[ "$PROFILE" != "synthetic" ]]; then
    warn "Profile '${PROFILE}' expects torch/transformers but nvidia-smi was not found."
    warn "CPU inference works but will be slow; install NVIDIA drivers + CUDA for production GPU use."
  fi
}

write_env_example() {
  local env_file="${INSTALL_DIR}/.env.production.example"
  if [[ -f "$env_file" ]]; then
    return 0
  fi
  log "Writing ${env_file}"
  cat >"$env_file" <<EOF
# Volvence Zero production environment (example)
# Copy to .env.production and adjust before enabling systemd.

VOLVENCE_PROFILE=${PROFILE}
VOLVENCE_VENV_DIR=${VENV_DIR}

# lifeform-serve defaults
LIFEFORM_VERTICAL=companion
LIFEFORM_SUBSTRATE_MODE=${SUBSTRATE_MODE}
LIFEFORM_SUBSTRATE_MODEL=${SUBSTRATE_MODEL}
LIFEFORM_HOST=${SERVICE_HOST}
LIFEFORM_PORT=${SERVICE_PORT}

# Hugging Face cache (optional — pre-download models here)
# HF_HOME=/var/lib/volvence/huggingface
# TRANSFORMERS_CACHE=/var/lib/volvence/huggingface

# External judge keys for benchmark harnesses only (not required for serving)
# JUDGE_API_KEY=
EOF
}

install_systemd_unit() {
  local unit_name="lifeform-serve.service"
  local unit_path
  local unit_dir
  local exec_start="${VENV_DIR}/bin/lifeform-serve"
  local extra_args=(--host "${SERVICE_HOST}" --port "${SERVICE_PORT}" --vertical companion)

  case "$SUBSTRATE_MODE" in
    synthetic)
      extra_args+=(--substrate-mode synthetic)
      ;;
    hf-shared)
      extra_args+=(--substrate-mode hf-shared --substrate-model-id "${SUBSTRATE_MODEL}" --substrate-device auto)
      ;;
    *)
      die "Unsupported --substrate-mode for systemd: ${SUBSTRATE_MODE}"
      ;;
  esac

  if [[ "$SYSTEMD_USER" -eq 1 ]]; then
    unit_dir="${HOME}/.config/systemd/user"
    run mkdir -p "${unit_dir}"
    unit_path="${unit_dir}/${unit_name}"
  else
    unit_dir="/etc/systemd/system"
    unit_path="${unit_dir}/${unit_name}"
    if ! id "$SERVICE_USER" >/dev/null 2>&1; then
      log "Creating service user: ${SERVICE_USER}"
      run maybe_sudo useradd --system --create-home --shell /usr/sbin/nologin "${SERVICE_USER}" || true
    fi
    run maybe_sudo mkdir -p /var/lib/volvence /var/log/volvence
    run maybe_sudo chown -R "${SERVICE_USER}:${SERVICE_USER}" /var/lib/volvence /var/log/volvence
  fi

  log "Writing systemd unit: ${unit_path}"
  local unit_body
  unit_body="$(cat <<EOF
[Unit]
Description=Volvence Zero lifeform-serve
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
$( [[ "$SYSTEMD_USER" -eq 0 ]] && echo "User=${SERVICE_USER}" )
$( [[ "$SYSTEMD_USER" -eq 0 ]] && echo "Group=${SERVICE_USER}" )
WorkingDirectory=${INSTALL_DIR}
Environment=PATH=${VENV_DIR}/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${exec_start} ${extra_args[*]}
Restart=on-failure
RestartSec=5
# Allow long model load on first start (hf-shared)
TimeoutStartSec=600

[Install]
WantedBy=$( [[ "$SYSTEMD_USER" -eq 1 ]] && echo "default.target" || echo "multi-user.target" )
EOF
)"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\n' "$unit_body"
    return 0
  fi

  if [[ "$SYSTEMD_USER" -eq 1 ]]; then
    printf '%s\n' "$unit_body" >"${unit_path}"
    systemctl --user daemon-reload
    systemctl --user enable --now "${unit_name}"
    log "Started user service: systemctl --user status ${unit_name}"
  else
    printf '%s\n' "$unit_body" | maybe_sudo tee "${unit_path}" >/dev/null
    maybe_sudo systemctl daemon-reload
    maybe_sudo systemctl enable --now "${unit_name}"
    log "Started system service: systemctl status ${unit_name}"
  fi
}

print_summary() {
  cat <<EOF

======================================================================
Volvence Zero bootstrap complete
======================================================================
Install dir : ${INSTALL_DIR}
Python      : ${PYTHON_BIN}
Venv        : ${VENV_DIR}
Profile     : ${PROFILE}

Activate:
  source ${VENV_DIR}/bin/activate

Quick smoke:
  ${VENV_DIR}/bin/python -c "from volvence_zero.brain import Brain, BrainConfig; print(Brain(BrainConfig()).create_session('demo').run_turn('hello').response.text)"

Start HTTP service (foreground):
  ${VENV_DIR}/bin/lifeform-serve --vertical companion --substrate-mode synthetic --host 127.0.0.1 --port ${SERVICE_PORT}

Production (one shared Qwen on GPU):
  ${VENV_DIR}/bin/lifeform-serve \\
    --vertical companion \\
    --substrate-mode hf-shared \\
    --substrate-model-id ${SUBSTRATE_MODEL} \\
    --substrate-device auto \\
    --host ${SERVICE_HOST} \\
    --port ${SERVICE_PORT}

List installed verticals:
  ${VENV_DIR}/bin/lifeform-serve --list-verticals

Optional private git submodules (benchmark held-out / research) are NOT
required for core runtime. Init only if you have credentials:
  git submodule update --init --recursive

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; VENV_DIR="${VOLVENCE_VENV_DIR:-${INSTALL_DIR}/.venv}"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --repo-ref) REPO_REF="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --skip-system-deps) SKIP_SYSTEM=1; shift ;;
    --skip-smoke-test) SKIP_SMOKE=1; shift ;;
    --install-systemd) INSTALL_SYSTEMD=1; shift ;;
    --systemd-user) SYSTEMD_USER=1; shift ;;
    --service-user) SERVICE_USER="$2"; shift 2 ;;
    --substrate-mode) SUBSTRATE_MODE="$2"; shift 2 ;;
    --substrate-model) SUBSTRATE_MODEL="$2"; shift 2 ;;
    --service-port) SERVICE_PORT="$2"; shift 2 ;;
    --service-host) SERVICE_HOST="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1 (try --help)" ;;
  esac
done

[[ "${VOLVENCE_SKIP_SYSTEM:-0}" == "1" ]] && SKIP_SYSTEM=1
[[ "${VOLVENCE_SKIP_SMOKE:-0}" == "1" ]] && SKIP_SMOKE=1
[[ "${VOLVENCE_INSTALL_SYSTEMD:-0}" == "1" ]] && INSTALL_SYSTEMD=1
[[ "${VOLVENCE_SYSTEMD_USER:-0}" == "1" ]] && SYSTEMD_USER=1

log "Volvence Zero bare-metal bootstrap"
log "install_dir=${INSTALL_DIR} profile=${PROFILE} venv=${VENV_DIR}"

if [[ "$SKIP_SYSTEM" -eq 0 ]]; then
  install_system_packages
else
  log "Skipping OS package install (--skip-system-deps)"
fi

detect_python || die "Python >= 3.11 not found. Install it or pass --python / VOLVENCE_PYTHON."
log "Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

ensure_repo
create_venv
resolve_extras
install_workspace
verify_console_scripts
gpu_hint
write_env_example

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  run_smoke_test
else
  log "Skipping smoke test"
fi

if [[ "$INSTALL_SYSTEMD" -eq 1 ]]; then
  install_systemd_unit
fi

print_summary
