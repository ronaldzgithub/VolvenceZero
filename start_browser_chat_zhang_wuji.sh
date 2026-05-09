#!/usr/bin/env bash
# Convenience launcher: real Qwen substrate + 张无忌 vertical
# (with the lived LifeformTemplate from the demo arc replay).
#
# Pipeline this script automates:
#
#   1. If artifacts/lifeform-templates/zhang-wuji-demo.json does not
#      exist, run examples/train_zhang_wuji_template.py first to
#      produce the template (~1-2 min on a synthetic substrate).
#      Set ZHANG_WUJI_RETRAIN=1 to retrain even when one exists, or
#      ZHANG_WUJI_SKIP_TEMPLATE=1 to fall back to the base profile.
#   2. Export VERTICAL=zhang_wuji and ZHANG_WUJI_TEMPLATE_PATH so the
#      service-side lifeform_service.verticals.discover_verticals()
#      router builds a 张无忌 Lifeform via give_birth() under
#      alpha mode (per-user filesystem-scoped memory + saved drives).
#   3. Defer to start_browser_chat_qwen.sh, which actually loads the
#      real HF Qwen runtime, wires the LLM expression synthesizer,
#      and starts the aiohttp service. Every Qwen-related env var
#      (MODEL_ID, DEVICE, PORT, ALPHA_MODE, …) honored by the
#      delegate is honored here too.
#
# Why this is a separate script:
#
# start_browser_chat_qwen.sh is vertical-agnostic by design — it
# discovers all installed lifeform-domain-* wheels and routes by
# VERTICAL=. This wrapper is the canonical "experience the trained
# 张无忌 character" entry point and bundles the trainer step so a
# user only has to run one command.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

# Honor existing env values; only set defaults when missing.
export VERTICAL="${VERTICAL:-zhang_wuji}"
export ZHANG_WUJI_TEMPLATE_PATH="${ZHANG_WUJI_TEMPLATE_PATH:-${ROOT_DIR}/artifacts/lifeform-templates/zhang-wuji-demo.json}"

if [[ "${ZHANG_WUJI_SKIP_TEMPLATE:-0}" == "1" ]]; then
  echo "[start-browser-chat-zhang-wuji] ZHANG_WUJI_SKIP_TEMPLATE=1 — using base profile (no give_birth)."
  unset ZHANG_WUJI_TEMPLATE_PATH
else
  if [[ "${ZHANG_WUJI_RETRAIN:-0}" == "1" || ! -f "${ZHANG_WUJI_TEMPLATE_PATH}" ]]; then
    if [[ "${ZHANG_WUJI_RETRAIN:-0}" == "1" ]]; then
      echo "[start-browser-chat-zhang-wuji] ZHANG_WUJI_RETRAIN=1 — retraining template."
    else
      echo "[start-browser-chat-zhang-wuji] template ${ZHANG_WUJI_TEMPLATE_PATH} not found — running trainer."
    fi
    PACKAGE_PATHS="$(printf '%s:' "${ROOT_DIR}/packages"/*/src)"
    if [[ -n "${PYTHONPATH:-}" ]]; then
      PYTHONPATH="${PACKAGE_PATHS}${PYTHONPATH}" "$PYTHON_BIN" \
        "${ROOT_DIR}/examples/train_zhang_wuji_template.py" \
        --output "${ZHANG_WUJI_TEMPLATE_PATH}" \
        ${ZHANG_WUJI_RETRAIN:+--force}
    else
      PYTHONPATH="${PACKAGE_PATHS}" "$PYTHON_BIN" \
        "${ROOT_DIR}/examples/train_zhang_wuji_template.py" \
        --output "${ZHANG_WUJI_TEMPLATE_PATH}" \
        ${ZHANG_WUJI_RETRAIN:+--force}
    fi
  else
    echo "[start-browser-chat-zhang-wuji] template found at ${ZHANG_WUJI_TEMPLATE_PATH}; skipping trainer."
    echo "[start-browser-chat-zhang-wuji] (set ZHANG_WUJI_RETRAIN=1 to retrain.)"
  fi
fi

echo "[start-browser-chat-zhang-wuji] vertical=${VERTICAL} template=${ZHANG_WUJI_TEMPLATE_PATH:-<none>}"

exec "${ROOT_DIR}/start_browser_chat_qwen.sh"
