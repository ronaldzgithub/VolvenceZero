#!/usr/bin/env bash
# Companion Bench smoke run — full real-API end-to-end pipeline validation.
#
# What this does (in order):
#   1. Source .local/llm.env to load OPENROUTER_API_KEY + LIFEFORM_LOCAL_API_KEY
#   2. Start local VZ SUT (lifeform-serve --enable-openai-compat port 8000)
#   3. Run score_reference_systems.py against scripts/companion_bench/
#      reference_systems.smoke.yaml restricted to family F1 (4 scenarios)
#      × 2 SUT × 1 paraphrase seed = 8 arcs total
#   4. Stop local VZ SUT
#   5. Build site/data/* from artifacts
#   6. Print aggregate summary
#
# Cost estimate: ~$1-3 USD = ¥7-21
# Wallclock: ~10-20 min
#
# Prerequisites in .local/llm.env:
#   OPENROUTER_API_KEY=sk-or-v1-...
#   LIFEFORM_LOCAL_API_KEY=any-non-empty-string
# Optional:
#   OPENROUTER_HTTP_REFERER=https://volvence.zero
#   OPENROUTER_X_TITLE=VolvenceZero CompanionBench
#
# Refs:
#   docs/external/companion-bench-openrouter-setup.md
#   docs/moving forward/companion-bench-public-launch-packet.md §3

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="${REPO_ROOT}/.local/llm.env"
OUTPUT_DIR="${REPO_ROOT}/artifacts/companion_bench_smoke"
SUT_HELPER="${REPO_ROOT}/scripts/companion_bench/start_vz_sut.sh"
SCORE_CMD="${REPO_ROOT}/scripts/companion_bench/score_reference_systems.py"
BUILD_SITE_CMD="${REPO_ROOT}/scripts/companion_bench/build_site.py"

FAMILY="${SMOKE_FAMILY:-F1}"
SKIP_VZ_SUT="${SKIP_VZ_SUT:-0}"
SKIP_BUILD_SITE="${SKIP_BUILD_SITE:-0}"
# SMOKE_PROVIDER: openrouter (default — cross-vendor) | qwen (DashScope-only fallback)
SMOKE_PROVIDER="${SMOKE_PROVIDER:-openrouter}"

case "$SMOKE_PROVIDER" in
  openrouter)
    ROSTER="${REPO_ROOT}/scripts/companion_bench/reference_systems.smoke.yaml"
    USER_SIM_BASE_URL="https://openrouter.ai/api/v1"
    USER_SIM_MODEL="openai/gpt-5-mini"
    USER_SIM_KEY_ENV="OPENROUTER_API_KEY"
    PERTURN_BASE_URL="https://openrouter.ai/api/v1"
    PERTURN_MODEL="openai/gpt-5-mini"
    PERTURN_KEY_ENV="OPENROUTER_API_KEY"
    ARC_BASE_URL="https://openrouter.ai/api/v1"
    ARC_MODEL="anthropic/claude-3.7-sonnet"
    ARC_KEY_ENV="OPENROUTER_API_KEY"
    REQUIRED_KEY_VAR="OPENROUTER_API_KEY"
    ;;
  qwen)
    ROSTER="${REPO_ROOT}/scripts/companion_bench/reference_systems.smoke_qwen.yaml"
    USER_SIM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    USER_SIM_MODEL="qwen3-max"
    USER_SIM_KEY_ENV="PROTOCOL_LLM_API_KEY"
    PERTURN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    PERTURN_MODEL="qwen3-max"
    PERTURN_KEY_ENV="PROTOCOL_LLM_API_KEY"
    ARC_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ARC_MODEL="qwen-plus"
    ARC_KEY_ENV="PROTOCOL_LLM_API_KEY"
    REQUIRED_KEY_VAR="PROTOCOL_LLM_API_KEY"
    ;;
  *)
    echo "ERROR: SMOKE_PROVIDER=$SMOKE_PROVIDER not recognised (use 'openrouter' or 'qwen')"
    exit 1
    ;;
esac

# ----- Step 1: Source env -----------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found. Create it with $REQUIRED_KEY_VAR + LIFEFORM_LOCAL_API_KEY."
  exit 1
fi
# shellcheck source=/dev/null
set -a
source "$ENV_FILE"
set +a

if [ -z "${!REQUIRED_KEY_VAR:-}" ]; then
  echo "ERROR: $REQUIRED_KEY_VAR not set. Add to $ENV_FILE."
  exit 1
fi
if [ -z "${LIFEFORM_LOCAL_API_KEY:-}" ]; then
  echo "WARN: LIFEFORM_LOCAL_API_KEY not set; defaulting to 'local-dev-no-auth'"
  export LIFEFORM_LOCAL_API_KEY="local-dev-no-auth"
fi

mkdir -p "$OUTPUT_DIR"
echo "=== Companion Bench smoke run ==="
echo "Provider:   $SMOKE_PROVIDER"
echo "Roster:     $ROSTER"
echo "Family:     $FAMILY"
echo "Output:     $OUTPUT_DIR"
echo "User-sim:   $USER_SIM_MODEL @ $USER_SIM_BASE_URL"
echo "Per-turn:   $PERTURN_MODEL @ $PERTURN_BASE_URL"
echo "Arc judge:  $ARC_MODEL @ $ARC_BASE_URL"
echo ""

# ----- Step 2: Start VZ SUT ---------------------------------------------------
if [ "$SKIP_VZ_SUT" = "1" ]; then
  echo "[1/4] Skipping VZ SUT start (SKIP_VZ_SUT=1)"
else
  echo "[1/4] Starting VZ SUT..."
  bash "$SUT_HELPER" start
fi

cleanup() {
  if [ "$SKIP_VZ_SUT" = "0" ]; then
    echo "[cleanup] Stopping VZ SUT..."
    bash "$SUT_HELPER" stop || true
  fi
}
trap cleanup EXIT

# ----- Step 3: Run scoring ----------------------------------------------------
echo ""
echo "[2/4] Running score_reference_systems on family $FAMILY..."
python "$SCORE_CMD" \
  --roster "$ROSTER" \
  --output-dir "$OUTPUT_DIR" \
  --user-sim-base-url "$USER_SIM_BASE_URL" \
  --user-sim-model "$USER_SIM_MODEL" \
  --user-sim-key-env "$USER_SIM_KEY_ENV" \
  --perturn-base-url "$PERTURN_BASE_URL" \
  --perturn-model "$PERTURN_MODEL" \
  --perturn-key-env "$PERTURN_KEY_ENV" \
  --arc-base-url "$ARC_BASE_URL" \
  --arc-model "$ARC_MODEL" \
  --arc-key-env "$ARC_KEY_ENV" \
  --paraphrase-seeds 0 \
  --family "$FAMILY"

# ----- Step 4: Build site -----------------------------------------------------
echo ""
if [ "$SKIP_BUILD_SITE" = "1" ]; then
  echo "[3/4] Skipping build_site (SKIP_BUILD_SITE=1)"
else
  echo "[3/4] Building site/data..."
  python "$BUILD_SITE_CMD" \
    --artifact-dir "$OUTPUT_DIR" \
    --site-dir "${REPO_ROOT}/site"
fi

# ----- Step 5: Print summary --------------------------------------------------
echo ""
echo "[4/4] Aggregate summary:"
AGG_PATH="$OUTPUT_DIR/aggregate_results.json"
if [ -f "$AGG_PATH" ]; then
  python -c "
import json, pathlib
data = json.loads(pathlib.Path('$AGG_PATH').read_text(encoding='utf-8'))
print(f\"systems: {len(data.get('systems', []))}\")
for sys_row in data.get('systems', []):
    sid = sys_row.get('submission_id', '?')
    summary = sys_row.get('summary', {})
    agg = summary.get('aggregate', {})
    final = agg.get('final_mean')
    arc_count = agg.get('arc_count', 0)
    cost = summary.get('cost', {})
    total_usd = cost.get('total_usd')
    final_str = f'{final:.2f}' if isinstance(final, (int, float)) else 'n/a'
    cost_str = f'\${total_usd:.4f}' if isinstance(total_usd, (int, float)) else 'n/a'
    print(f'  {sid:40s}  final={final_str:>6s}  arcs={arc_count}  cost={cost_str}')
"
else
  echo "WARNING: $AGG_PATH not found"
fi

echo ""
echo "Done. Site preview: open ${REPO_ROOT}/site/index.html"
echo "Bundle inspection: ls $OUTPUT_DIR/*/arcs/"
