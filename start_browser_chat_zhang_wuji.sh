#!/usr/bin/env bash
# Start browser chat with the baked 张无忌 life-through template on Qwen 1.5B.
#
# This is a thin convenience wrapper around start_browser_chat_qwen.sh. It keeps
# all service behavior in the shared Qwen startup script, but pins the character
# vertical and the already-baked template artifact by default. ALPHA_MODE defaults
# to 0 here because the alpha path intentionally skips template memory restore;
# live-through QA should restore the template checkpoint.
#
# Usage:
#   bash start_browser_chat_zhang_wuji.sh
#   MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct bash start_browser_chat_zhang_wuji.sh
#   CHARACTER_PACKAGE_MODE=active bash start_browser_chat_zhang_wuji.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

default_template_path="${ROOT_DIR}/artifacts/lifeform-templates/zhang_wuji/zhang-wuji-live-through.json"
default_character_package_path="${ROOT_DIR}/artifacts/character-packages/zhang_wuji/zhang-wuji-qwen2.5-1.5b.character-prefix.json"
default_character_residual_path="${ROOT_DIR}/artifacts/character-packages/zhang_wuji/zhang-wuji-qwen2.5-1.5b.character-residual.json"
default_character_manifest_path="${ROOT_DIR}/artifacts/character-packages/zhang_wuji/character-package-manifest.json"
default_common_adapter_path="${ROOT_DIR}/artifacts/common-adapters/qwen2.5-1.5b/common-adapter-bundle.json"
export VERTICAL="${VERTICAL:-zhang_wuji}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
export ALPHA_MODE="${ALPHA_MODE:-0}"
export ZHANG_WUJI_CHARACTER_PREFIX_MODE="${ZHANG_WUJI_CHARACTER_PREFIX_MODE:-shadow}"
export CHARACTER_PACKAGE_MODE="${CHARACTER_PACKAGE_MODE:-shadow}"
export TEMPLATES_ROOT_DIR="${TEMPLATES_ROOT_DIR:-${ROOT_DIR}/artifacts/lifeform-templates}"
export ZHANG_WUJI_TEMPLATE_PATH="${ZHANG_WUJI_TEMPLATE_PATH:-${default_template_path}}"
if [[ -z "${CHARACTER_PACKAGE_MANIFESTS:-}" && "${MODEL_ID}" == "Qwen/Qwen2.5-1.5B-Instruct" ]]; then
  if [[ -f "${default_character_manifest_path}" && -f "${default_common_adapter_path}" ]]; then
    export CHARACTER_PACKAGE_MANIFESTS="${default_character_manifest_path}"
  elif [[ -z "${ZHANG_WUJI_CHARACTER_PACKAGE_PATH:-}" && -f "${default_character_package_path}" ]]; then
    export ZHANG_WUJI_CHARACTER_PACKAGE_PATH="${default_character_package_path}"
  fi
fi
if [[ -z "${COMMON_ADAPTER_BUNDLE_PATH:-}" && "${MODEL_ID}" == "Qwen/Qwen2.5-1.5B-Instruct" ]]; then
  export COMMON_ADAPTER_BUNDLE_PATH="${default_common_adapter_path}"
fi
if [[ -z "${ZHANG_WUJI_CHARACTER_RESIDUAL_PATH:-}" && "${MODEL_ID}" == "Qwen/Qwen2.5-1.5B-Instruct" && -f "${default_character_residual_path}" ]]; then
  export ZHANG_WUJI_CHARACTER_RESIDUAL_PATH="${default_character_residual_path}"
fi

if [[ ! -f "${ZHANG_WUJI_TEMPLATE_PATH}" ]]; then
  cat >&2 <<EOF
Cannot find the baked 张无忌 template:
  ZHANG_WUJI_TEMPLATE_PATH=${ZHANG_WUJI_TEMPLATE_PATH}

Rebuild it with:
  ${ROOT_DIR}/.venv/bin/python examples/bake_zhang_wuji_live_through.py --save-template

Or point ZHANG_WUJI_TEMPLATE_PATH at another saved LifeformTemplate JSON.
EOF
  exit 1
fi

if [[ -n "${CHARACTER_PACKAGE_MANIFESTS:-}" && ! -f "${CHARACTER_PACKAGE_MANIFESTS}" ]]; then
  cat >&2 <<EOF
Cannot find the 张无忌 character package manifest:
  CHARACTER_PACKAGE_MANIFESTS=${CHARACTER_PACKAGE_MANIFESTS}

Build it with:
  ${ROOT_DIR}/.venv/bin/python scripts/bake_zhang_wuji_character_package.py
EOF
  exit 1
fi

if [[ -n "${CHARACTER_PACKAGE_MANIFESTS:-}" && ! -f "${COMMON_ADAPTER_BUNDLE_PATH:-}" ]]; then
  cat >&2 <<EOF
Cannot find the shared common adapter bundle:
  COMMON_ADAPTER_BUNDLE_PATH=${COMMON_ADAPTER_BUNDLE_PATH:-<unset>}

Build it with:
  ${ROOT_DIR}/.venv/bin/python scripts/train_common_adapter_model.py --help
EOF
  exit 1
fi

if [[ "${ZHANG_WUJI_CHARACTER_RESIDUAL_MODE:-shadow}" == "active" ]]; then
  echo "CharacterResidualAdapterPackage is deprecated and cannot be ACTIVE; use the unified character manifest Prefix/KV + optional LoRA carriers." >&2
  exit 1
fi

exec "${ROOT_DIR}/start_browser_chat_qwen.sh"
