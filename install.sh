#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
VOLVENCE_EXTRAS="${VOLVENCE_EXTRAS:-hf}"

if [[ -n "$VOLVENCE_EXTRAS" ]]; then
  PACKAGE_SPEC=".[${VOLVENCE_EXTRAS}]"
else
  PACKAGE_SPEC="."
fi

cd "$ROOT_DIR"

echo "Installing volvence-zero into the current Python environment..."
echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Package spec: ${PACKAGE_SPEC}"

"$PYTHON_BIN" -m pip install -e "$PACKAGE_SPEC" "$@"

echo "volvence-zero installed successfully."
