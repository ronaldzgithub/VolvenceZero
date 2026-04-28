#!/usr/bin/env bash
# Install every wheel in the workspace as editable.
#
# By default this installs the synthetic substrate path. Pass
# VOLVENCE_EXTRAS=hf to also pull torch / transformers, or
# VOLVENCE_EXTRAS=torch for SSL/RL training deps.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
VOLVENCE_EXTRAS="${VOLVENCE_EXTRAS:-}"

cd "$ROOT_DIR"

echo "Installing Volvence Zero workspace into the current Python environment..."
echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

# Order matters: dependencies must be installed before dependents.
PACKAGES=(
  packages/vz-contracts
  packages/vz-substrate
  packages/vz-memory
  packages/vz-cognition
  packages/vz-application
  packages/vz-temporal
  packages/vz-runtime
  packages/lifeform-core
  packages/lifeform-expression
  packages/lifeform-domain-emogpt
  packages/lifeform-service
  packages/lifeform-evolution
)

for pkg in "${PACKAGES[@]}"; do
  if [[ -d "${pkg}" ]]; then
    echo "==> pip install -e ${pkg}"
    "$PYTHON_BIN" -m pip install -e "${pkg}" --no-deps "$@"
  fi
done

if [[ -n "$VOLVENCE_EXTRAS" ]]; then
  echo "==> pip install vz-runtime[${VOLVENCE_EXTRAS}] (extras only)"
  "$PYTHON_BIN" -m pip install "vz-runtime[${VOLVENCE_EXTRAS}]"
fi

echo "Volvence Zero workspace installed successfully."
