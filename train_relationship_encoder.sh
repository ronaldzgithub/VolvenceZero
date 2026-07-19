#!/usr/bin/env bash
# Relationship-encoder training, one command, OpenRouter end to end:
#
#   export OPENROUTER_API_KEY=sk-or-...
#   ./train_relationship_encoder.sh          # generate LLM data -> train -> G2 report
#
#   ./train_relationship_encoder.sh --fsm    # free deterministic dry-run (no key needed)
#
# One OpenRouter key drives all three LLM roles. Deliberately different
# models per role (reduces simulator style fingerprints and keeps the
# zero-shot baseline from grading its own homework):
#   user simulator     qwen/qwen3-235b-a22b
#   SUT (assistant)    anthropic/claude-3.5-haiku
#   zero-shot baseline openai/gpt-4o-mini
#
# No flags beyond --fsm / --regenerate. Occasional overrides go through env:
#   SIM_MODEL / SUT_MODEL / ZERO_SHOT_MODEL   OpenRouter model slugs
#   BACKBONE  (default hf:Qwen/Qwen2.5-0.5B; fsm mode uses tiny)
#   EPOCHS / BATCH_SIZE / LEARNING_RATE / DEVICE / RUN_NAME
#
# Data scale note: volume = 30 public scenarios x 3 paraphrase seeds = 90
# trajectories max. Scaling to 10^4 needs more scenarios upstream in
# companion-bench, not a knob here.

set -euo pipefail
cd "$(dirname "$0")"

OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
SIM_MODEL="${SIM_MODEL:-qwen/qwen3-235b-a22b}"
SUT_MODEL="${SUT_MODEL:-anthropic/claude-3.5-haiku}"
ZERO_SHOT_MODEL="${ZERO_SHOT_MODEL:-openai/gpt-4o-mini}"

MODE="llm"
REGENERATE=0
for arg in "$@"; do
  case "$arg" in
    --fsm)        MODE="fsm" ;;
    --regenerate) REGENERATE=1 ;;
    -h|--help)    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $arg (only --fsm / --regenerate; overrides go via env, see --help)" >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == "llm" ]]; then
  BACKBONE="${BACKBONE:-hf:Qwen/Qwen2.5-0.5B}"
  EPOCHS="${EPOCHS:-3}"
  BATCH_SIZE="${BATCH_SIZE:-4}"
  LEARNING_RATE="${LEARNING_RATE:-2e-5}"
  MAX_INPUT_BYTES="${MAX_INPUT_BYTES:-4096}"  # hf tokenizer cap = bytes/4 tokens
  [[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is not set (use --fsm for the free dry-run)" >&2; exit 2; }
  export TRAJGEN_SUT_API_KEY="$OPENROUTER_API_KEY"
  export TRAJGEN_SIM_API_KEY="$OPENROUTER_API_KEY"
  export OPENAI_API_KEY="$OPENROUTER_API_KEY"
else
  # fsm filler text carries no linguistic signal: tiny backbone + no
  # zero-shot column; this run validates the pipeline, not the model.
  BACKBONE="${BACKBONE:-tiny}"
  EPOCHS="${EPOCHS:-4}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  LEARNING_RATE="${LEARNING_RATE:-3e-4}"
  MAX_INPUT_BYTES="${MAX_INPUT_BYTES:-2048}"  # byte-level attention: 4096 OOMs on MPS
fi

DATA_DIR="data/encoder/traj-${MODE}"
RUN_NAME="${RUN_NAME:-${BACKBONE//[:\/]/-}-${MODE}-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="runs/encoder/${RUN_NAME}"
DEVICE="${DEVICE:-$(python -c "import torch; print('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))")}"

echo "== relationship-encoder pipeline =="
echo "   mode=${MODE} backbone=${BACKBONE} device=${DEVICE}"
echo "   data=${DATA_DIR} out=${OUT_DIR}"

# ---------------------------------------------------------- [1/3] data
if [[ -d "$DATA_DIR/train" && "$REGENERATE" -eq 0 ]]; then
  echo "== [1/3] data exists, skipping generation (--regenerate to force)"
else
  echo "== [1/3] generating trajectories (${MODE} mode)"
  GEN_ARGS=(generate --out-dir "$DATA_DIR" --mode "$MODE")
  if [[ "$MODE" == "llm" ]]; then
    GEN_ARGS+=(--sim-base-url "$OPENROUTER_BASE_URL" --sim-model "$SIM_MODEL"
               --sut-base-url "$OPENROUTER_BASE_URL" --sut-model "$SUT_MODEL")
  fi
  companion-trajgen "${GEN_ARGS[@]}"
fi

# --------------------------------------------------------- [2/3] train
echo "== [2/3] training (${BACKBONE}, ${EPOCHS} epochs, lr=${LEARNING_RATE}, ${DEVICE})"
companion-encoder train \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --max-input-bytes "$MAX_INPUT_BYTES" \
  --device "$DEVICE"

# ------------------------------------------------------ [3/3] G2 report
echo "== [3/3] evaluating (G2 report: encoder vs baselines)"
EVAL_ARGS=(evaluate
  --data-dir "$DATA_DIR"
  --checkpoint "$OUT_DIR/encoder.pt"
  --report "$OUT_DIR/g2-report.json"
  --device "$DEVICE")
if [[ "$MODE" == "llm" ]]; then
  EVAL_ARGS+=(--llm-base-url "$OPENROUTER_BASE_URL" --llm-model "$ZERO_SHOT_MODEL")
fi
companion-encoder "${EVAL_ARGS[@]}" > /dev/null

echo "== done"
echo "   checkpoint:   $OUT_DIR/encoder.pt"
echo "   train report: $OUT_DIR/train-report.json"
echo "   G2 report:    $OUT_DIR/g2-report.json"
python - "$OUT_DIR/g2-report.json" <<'EOF'
import json, sys
report = json.load(open(sys.argv[1]))
print("   -- G2 summary (val) --")
for name, column in report["structured_prediction"].items():
    print(f"   {name:24s} acc={column['accuracy']:.3f} macro_f1={column['macro_f1']:.3f} mean_mae={column['mean_mae']:.3f} ece={column['ece']:.3f}")
for name, score in report["retrieval_family_top1"].items():
    print(f"   retrieval/{name:14s} family_top1={score:.3f}")
EOF
