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
  packages/lifeform-thinking
  packages/lifeform-ingestion
  packages/lifeform-affordance
  packages/lifeform-expression
  packages/lifeform-domain-character
  packages/lifeform-domain-emogpt
  packages/lifeform-domain-coding
  packages/lifeform-domain-figure
  packages/lifeform-domain-growth-advisor
  packages/companion-bench
  packages/lifeform-service
  packages/lifeform-evolution
  packages/lifeform-openai-compat
  packages/lifeform-protocol-runtime
  packages/lifeform-mcp-bridge
  packages/dlaas-platform-contracts
  packages/dlaas-platform-registry
  packages/dlaas-platform-launcher
  packages/dlaas-platform-ops
  packages/dlaas-platform-eval
  packages/dlaas-platform-api
)

# Pass 1: register every workspace sibling editably with --no-deps so the
# circular workspace dependency cluster (lifeform-openai-compat <->
# lifeform-service <-> lifeform-protocol-runtime) does not cause pip's
# resolver to try to fetch unpublished `==0.1.*` siblings from PyPI.
for pkg in "${PACKAGES[@]}"; do
  if [[ -d "${pkg}" ]]; then
    echo "==> [pass 1] pip install -e ${pkg} --no-deps"
    "$PYTHON_BIN" -m pip install -e "${pkg}" --no-deps "$@"
  fi
done

# Pass 2: re-run with full dep resolution. By now every workspace sibling
# is editable and satisfies its ==0.1.* constraint, so pip only fetches
# the *external* PyPI deps declared in each wheel's pyproject.toml
# (aiohttp / pypdf / beautifulsoup4 / mwparserfromhell / lxml / requests
# / PyYAML / ...). This keeps pyproject.toml as the single source of
# truth for runtime deps — install.sh never has to mirror the list.
for pkg in "${PACKAGES[@]}"; do
  if [[ -d "${pkg}" ]]; then
    echo "==> [pass 2] pip install -e ${pkg}"
    "$PYTHON_BIN" -m pip install -e "${pkg}" "$@"
  fi
done

if [[ -n "$VOLVENCE_EXTRAS" ]]; then
  echo "==> pip install vz-runtime[${VOLVENCE_EXTRAS}] (extras only)"
  "$PYTHON_BIN" -m pip install "vz-runtime[${VOLVENCE_EXTRAS}]"
fi

echo "Volvence Zero workspace installed successfully."
