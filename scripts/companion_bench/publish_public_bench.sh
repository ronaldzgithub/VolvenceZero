#!/usr/bin/env bash
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.
#
# Publish the public slice of this monorepo to companionbench/bench.
#
# Source of truth is THIS monorepo (VolvenceZero/VolvenceZero, private).
# The public companionbench/bench repo is a read-only mirror containing:
#
#   packages/companion-bench/                Apache-2.0 reference wheel
#   site/                                    Public static site
#   docs/external/companion-bench-*.md       Public RFC + protocol + crosswalk
#   docs/external/companion-bench-public-scenario-hashes.txt
#   docs/external/eqbench3-*                 Cross-walk + checklist
#   scripts/companion_bench/                 Build / smoke / scoring helpers
#   .github/ISSUE_TEMPLATE/                  Submission request templates
#   .github/workflows/companion-bench-*.yml  Public CI workflows
#   tests/contracts/test_companion_bench_*   Companion-bench-only contract tests
#   tests/contracts/test_no_lscb_strings.py  Brand-consistency guard
#   LICENSE                                  Apache 2.0
#   README.md                                Repo entry point
#
# Anything outside this allow-list is NOT published; `lifeform-*`, `vz-*`,
# `research/`, `external/companionbench-heldout/`, `external/vz-bundle/`,
# and `docs/business/` never leave the private monorepo.
#
# Modes:
#   --dry-run   stage into a temp dir and print the would-be tree; do NOT
#               touch any git remote
#   --push      stage + git init + commit + force-push to GH_REMOTE
#
# Defaults to --dry-run for safety.
#
# Env:
#   GH_REMOTE   default git@github.com:companionbench/bench.git
#   GH_BRANCH   default main

set -euo pipefail

MODE="${1:---dry-run}"
GH_REMOTE="${GH_REMOTE:-git@github.com:companionbench/bench.git}"
GH_BRANCH="${GH_BRANCH:-main}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Allow-list of paths to copy. Each line is a path relative to REPO_ROOT;
# files OR directories. Globs are NOT expanded — keep entries explicit.
ALLOWLIST=(
  "packages/companion-bench"
  "site"
  "scripts/companion_bench"
  "docs/external/companion-bench-rfc-v0.md"
  "docs/external/companion-bench-submission-protocol.md"
  "docs/external/companion-bench-governance-charter-draft.md"
  "docs/external/companion-bench-heldout-bootstrap.md"
  "docs/external/companion-bench-public-scenario-hashes.txt"
  "docs/external/eqbench3-submission-protocol.md"
  "docs/external/eqbench3-public-submission-checklist.md"
  "docs/external/eqbench3-results-internal.md"
  ".github/ISSUE_TEMPLATE"
  ".github/workflows/companion-bench-ci-smoke.yml"
  ".github/workflows/companion-bench-paper-suite-small.yml"
  ".github/workflows/companion-bench-paper-suite-full.yml"
  ".github/workflows/companion-bench-publish.yml"
  "tests/contracts/test_companion_bench_no_internal_imports.py"
  "tests/contracts/test_companion_bench_g2_site_cleanup.py"
  "tests/contracts/test_companion_bench_judge_family_rotation.py"
  "tests/contracts/test_no_lscb_strings.py"
)

# Files that must NEVER ship even if they sit inside an allow-listed dir.
DENYLIST_GLOBS=(
  "**/__pycache__"
  "**/.pytest_cache"
  "**/.ruff_cache"
  "**/.DS_Store"
  "**/*.egg-info"
  "**/.coverage"
)

STAGE_DIR="$(mktemp -d -t companionbench-bench-XXXXXX)"
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
    # rsync preferred; fall back to cp -r if rsync absent.
    if command -v rsync >/dev/null 2>&1; then
      rsync_excludes=()
      for glob in "${DENYLIST_GLOBS[@]}"; do
        rsync_excludes+=(--exclude "$glob")
      done
      rsync -a "${rsync_excludes[@]}" "$src/" "$dst/"
    else
      cp -r "$src" "$dst"
      # Best-effort deny: prune common cache dirs after copy.
      find "$dst" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
      find "$dst" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
      find "$dst" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    fi
  else
    cp "$src" "$dst"
  fi
done

# Top-level repo polish (LICENSE / README / .gitignore for the public repo).
if [[ -f "$REPO_ROOT/packages/companion-bench/README.md" ]]; then
  cp "$REPO_ROOT/packages/companion-bench/README.md" "$STAGE_DIR/README.md"
fi
cat > "$STAGE_DIR/LICENSE" <<'EOF'
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Copyright 2026 Companion Bench Contributors

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

# Sanity: the staged tree must be brand-consistent.
# Files where the legacy token is structurally required (the guard test
# names it; the G2 cleanup test asserts legacy stub paths are absent;
# this publish script itself references the token in its log messages).
ALLOWED_LEGACY_FILES=(
  "test_no_lscb_strings.py"
  "test_companion_bench_g2_site_cleanup.py"
  "publish_public_bench.sh"
)

echo "[publish] brand consistency check (no 'lscb' tokens):"
filter='cat'
for f in "${ALLOWED_LEGACY_FILES[@]}"; do
  filter+=" | grep -v '$f'"
done
offenders=$(
  grep -rli "lscb" "$STAGE_DIR" \
       --exclude-dir=".git" \
       --exclude-dir="__pycache__" \
       --exclude-dir="*.egg-info" \
    | eval "$filter"
)
if [[ -n "$offenders" ]]; then
  echo "[publish] FAILED: staged tree contains lscb token in:" >&2
  echo "$offenders" >&2
  exit 1
fi
echo "[publish] OK: no lscb tokens in staged tree (legacy-allowed files excepted)"

case "$MODE" in
  --dry-run)
    echo
    echo "[publish] dry-run complete; staged at $STAGE_DIR (will be deleted on exit)."
    echo "[publish] to publish for real: $0 --push"
    ;;
  --push)
    echo "[publish] pushing to $GH_REMOTE (branch $GH_BRANCH)..."
    pushd "$STAGE_DIR" >/dev/null
    git init -q -b "$GH_BRANCH"
    git add .
    git -c user.email="bot@companionbench.com" \
        -c user.name="Companion Bench Publisher" \
        commit -q -m "publish $(date -u +%Y-%m-%dT%H:%M:%SZ) from VolvenceZero/VolvenceZero@$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
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
