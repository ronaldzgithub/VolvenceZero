#!/usr/bin/env bash
# Start the browser chat service with a real Hugging Face Qwen substrate.
#
# Defaults target Qwen2.5-3B-Instruct as the smallest base model that
# can both follow VZ's structured system prompt and keep multi-turn
# coherence on short follow-ups (the 0.5B/1.5B variants tend to
# collapse into single-character or off-topic replies under the
# kernel's plan/ordering instructions).
#
# Model size sanity (FP16 weights, single device):
#   * Qwen2.5-1.5B-Instruct  ~3 GB on disk, ~4 GB resident
#   * Qwen2.5-3B-Instruct    ~6 GB on disk, ~8 GB resident   (default)
#   * Qwen2.5-7B-Instruct   ~15 GB on disk, ~18 GB resident  (24 GB Mac OK if
#                                                             the disk has it)
#
# Usage:
#   bash start_browser_chat_qwen.sh
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct bash start_browser_chat_qwen.sh
#   MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct bash start_browser_chat_qwen.sh
#
# If HuggingFace is slow / blocked, route through the mirror:
#   HF_ENDPOINT=https://hf-mirror.com bash start_browser_chat_qwen.sh
#
# Useful env:
#   HOST=127.0.0.1
#   PORT=8765
#   MODEL_ID=Qwen/Qwen2.5-3B-Instruct
#   DEVICE=auto
#   LOCAL_FILES_ONLY=0
#   OPEN_BROWSER=1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8765}"
export VERTICAL="${VERTICAL:-companion}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export DEVICE="${DEVICE:-auto}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
export MAX_SESSIONS="${MAX_SESSIONS:-256}"
export IDLE_EVICTION_SECONDS="${IDLE_EVICTION_SECONDS:-1800}"
export OPEN_BROWSER="${OPEN_BROWSER:-1}"

cd "$ROOT_DIR"

PACKAGE_PATHS="$(printf '%s:' packages/*/src)"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PACKAGE_PATHS}${PYTHONPATH}"
else
  export PYTHONPATH="${PACKAGE_PATHS}"
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use. Stop the existing service or set PORT=another_port." >&2
  exit 1
fi

CHAT_URL="http://${HOST}:${PORT}/chat"

echo "[start-browser-chat-qwen] model=${MODEL_ID}"
echo "[start-browser-chat-qwen] device=${DEVICE} local_files_only=${LOCAL_FILES_ONLY}"
echo "[start-browser-chat-qwen] url=${CHAT_URL}"

if [[ "$OPEN_BROWSER" == "1" ]] && command -v open >/dev/null 2>&1; then
  (sleep 3; open "$CHAT_URL" >/dev/null 2>&1 || true) &
fi

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
import sys

from aiohttp import web
from lifeform_service.app import create_app
from lifeform_service.verticals import default_vertical_name, discover_verticals
from volvence_zero.substrate import SubstrateFallbackMode, build_transformers_runtime_with_fallback


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    host = os.environ["HOST"]
    port = int(os.environ["PORT"])
    requested_vertical = os.environ.get("VERTICAL")
    model_id = os.environ["MODEL_ID"]
    device = os.environ["DEVICE"]
    local_files_only = _env_bool("LOCAL_FILES_ONLY")
    max_sessions = int(os.environ["MAX_SESSIONS"])
    idle_eviction_seconds = float(os.environ["IDLE_EVICTION_SECONDS"])

    verticals = discover_verticals()
    if not verticals:
        print(
            "No verticals available. Install lifeform-domain-emogpt or another lifeform-domain-* package.",
            file=sys.stderr,
        )
        return 1

    vertical_name = requested_vertical or default_vertical_name()
    if vertical_name not in verticals:
        print(f"Unknown vertical {vertical_name!r}. Available: {sorted(verticals)}", file=sys.stderr)
        return 1

    print("[start-browser-chat-qwen] loading real Qwen substrate; fallback is disabled", flush=True)
    runtime = build_transformers_runtime_with_fallback(
        model_id=model_id,
        device=device,
        local_files_only=local_files_only,
        fallback_mode=SubstrateFallbackMode.DENY,
        allow_live_substrate_mutation=False,
    )
    runtime_origin = getattr(runtime, "runtime_origin")
    if runtime_origin == "builtin-fallback":
        raise RuntimeError("Expected a real HF Qwen runtime, got builtin-fallback.")

    app = create_app(
        vertical=verticals[vertical_name],
        max_sessions=max_sessions,
        idle_eviction_seconds=idle_eviction_seconds,
        substrate_runtime=runtime,
    )
    print(
        "[start-browser-chat-qwen] ready "
        f"vertical={vertical_name} model_id={model_id} runtime_origin={runtime_origin}",
        flush=True,
    )
    print(f"[start-browser-chat-qwen] listening on http://{host}:{port}/chat", flush=True)
    web.run_app(app, host=host, port=port, print=lambda *_: None)
    return 0


raise SystemExit(main())
PY
