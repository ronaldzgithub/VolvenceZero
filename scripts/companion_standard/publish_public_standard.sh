#!/usr/bin/env bash
# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.
#
# Publish the public slice of the Relationship Representation Standard to
# its read-only mirror repository.
#
# Source of truth is THIS monorepo (private). The public mirror contains:
#
#   packages/companion-standard/             Apache-2.0 standard wheel
#   packages/companion-trajgen/              Apache-2.0 data pipeline wheel
#   docs/external/relationship-representation-rfc-v0.md
#   docs/external/relationship-representation-trajectory.schema.json
#   tests/contracts/test_companion_standard_*  Standard-only contract tests
#   tests/contracts/test_companion_trajgen_*   Trajgen-only contract tests
#   tests/test_companion_trajgen_pipeline.py   Pipeline functional tests
#   scripts/companion_standard/              This publisher
#   LICENSE / README.md / .gitignore         Repo polish
#
# Anything outside this allow-list is NOT published; `vz-*`, `lifeform-*`,
# `research/`, `docs/business/`, and internal specs never leave the
# private monorepo. companion-bench mirrors separately via
# scripts/companion_bench/publish_public_bench.sh; the trajgen wheel
# depends on the published companion-bench PyPI wheel, not on repo paths.
#
# Modes:
#   --dry-run   stage into a temp dir and print the would-be tree (default)
#   --push      stage + git init + commit + force-push to GH_REMOTE
#
# Env:
#   GH_REMOTE   default git@github.com:companionbench/standard.git
#   GH_BRANCH   default main

set -euo pipefail

MODE="${1:---dry-run}"
GH_REMOTE="${GH_REMOTE:-git@github.com:companionbench/standard.git}"
GH_BRANCH="${GH_BRANCH:-main}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ALLOWLIST=(
  "packages/companion-standard"
  "packages/companion-trajgen"
  "scripts/companion_standard"
  "docs/external/relationship-representation-rfc-v0.md"
  "docs/external/relationship-representation-trajectory.schema.json"
  "tests/contracts/test_companion_standard_no_internal_imports.py"
  "tests/contracts/test_companion_standard_conformance.py"
  "tests/contracts/test_companion_trajgen_boundaries.py"
  "tests/test_companion_trajgen_pipeline.py"
)

DENYLIST_GLOBS=(
  "**/__pycache__"
  "**/.pytest_cache"
  "**/.ruff_cache"
  "**/.DS_Store"
  "**/*.egg-info"
  "**/.coverage"
)

STAGE_DIR="$(mktemp -d -t companion-standard-XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "[publish] staging into $STAGE_DIR"

for rel in "${ALLOWLIST[@]}"; do
  src="$REPO_ROOT/$rel"
  if [[ ! -e "$src" ]]; then
    echo "[publish] WARNING: allow-list entry missing: $rel" >&2
    continue
  fi
  dst="$STAGE_DIR/$rel"
  mkdir -p "$(dirname "$dst")"
  if [[ -d "$src" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync_excludes=()
      for glob in "${DENYLIST_GLOBS[@]}"; do
        rsync_excludes+=(--exclude "$glob")
      done
      rsync -a "${rsync_excludes[@]}" "$src/" "$dst/"
    else
      cp -r "$src" "$dst"
      find "$dst" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
      find "$dst" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
      find "$dst" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    fi
  else
    cp "$src" "$dst"
  fi
done

# Top-level repo polish.
cp "$REPO_ROOT/packages/companion-standard/README.md" "$STAGE_DIR/README.md"
cat > "$STAGE_DIR/LICENSE" <<'EOF'
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Copyright 2026 Companion Standard Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
EOF

cat > "$STAGE_DIR/.gitignore" <<'EOF'
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
*.egg-info/
.coverage
htmlcov/
.DS_Store
artifacts/
EOF

echo "[publish] staged tree:"
( cd "$STAGE_DIR" && find . -maxdepth 3 -type d -not -path "*/.*" | sort )

# Sanity 1: no internal wheel references in staged sources.
echo "[publish] internal-reference check:"
offenders=$(grep -rl "volvence_zero\|lifeform_\|dlaas_platform_" "$STAGE_DIR/packages" 2>/dev/null || true)
if [[ -n "$offenders" ]]; then
  # The guard tests legitimately NAME the forbidden roots; source files may not.
  offenders=$(echo "$offenders" | grep -v "test_" || true)
fi
if [[ -n "$offenders" ]]; then
  echo "[publish] FAILED: staged packages reference internal wheels:" >&2
  echo "$offenders" >&2
  exit 1
fi
echo "[publish] OK: no internal wheel references in staged packages"

# Sanity 2: no business / customer material. This script itself names the
# forbidden patterns (that is its job), so it is excluded from the sweep.
echo "[publish] business-material check:"
offenders=$(grep -rli "docs/business\|谌" "$STAGE_DIR" --exclude-dir=".git" \
  | grep -v "publish_public_standard.sh" || true)
if [[ -n "$offenders" ]]; then
  echo "[publish] FAILED: staged tree contains business material references:" >&2
  echo "$offenders" >&2
  exit 1
fi
echo "[publish] OK: no business material in staged tree"

case "$MODE" in
  --dry-run)
    echo
    echo "[publish] dry-run complete; staged at $STAGE_DIR (deleted on exit)."
    echo "[publish] to publish for real: $0 --push"
    ;;
  --push)
    echo "[publish] pushing to $GH_REMOTE (branch $GH_BRANCH)..."
    pushd "$STAGE_DIR" >/dev/null
    git init -q -b "$GH_BRANCH"
    git add .
    git -c user.email="bot@companionbench.com" \
        -c user.name="Companion Standard Publisher" \
        commit -q -m "publish $(date -u +%Y-%m-%dT%H:%M:%SZ) from monorepo@$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
    git remote add origin "$GH_REMOTE"
    git push --force origin "$GH_BRANCH"
    popd >/dev/null
    echo "[publish] done."
    ;;
  *)
    echo "usage: $0 [--dry-run | --push]" >&2
    exit 2
    ;;
esac
