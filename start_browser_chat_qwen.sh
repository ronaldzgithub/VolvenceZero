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
# Cross-session memory: ALPHA_MODE defaults to 1 so the kernel binds each
# session to the ``userId`` typed in the chat UI (sent as X-Alpha-User) and
# persists per-user durable memory under MEMORY_SCOPE_ROOT_DIR. Set
# ALPHA_MODE=0 to fall back to the previous anonymous, in-memory-only
# behavior.
#
# Usage:
#   bash start_browser_chat_qwen.sh
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct bash start_browser_chat_qwen.sh
#   MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct bash start_browser_chat_qwen.sh
#
# If HuggingFace is slow / blocked, route through the mirror:
#   HF_ENDPOINT=https://hf-mirror.com bash start_browser_chat_qwen.sh
#   ALPHA_MODE=0 bash start_browser_chat_qwen.sh   # anonymous, no persistence
#
# Useful env:
#   HOST=127.0.0.1
#   PORT=8765
#   MODEL_ID=Qwen/Qwen2.5-3B-Instruct
#   DEVICE=auto
#   LOCAL_FILES_ONLY=0
#   OPEN_BROWSER=1
#   ALPHA_MODE=1                             # 1 = scoped memory, 0 = anonymous
#   MEMORY_SCOPE_ROOT_DIR=$ROOT_DIR/.local/browser_chat_memory
#   ALPHA_USERS_FILE=                        # optional JSON allowlist
#   EVIDENCE_ROOT_DIR=                       # optional alpha evidence dir

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
export ALPHA_MODE="${ALPHA_MODE:-1}"
export MEMORY_SCOPE_ROOT_DIR="${MEMORY_SCOPE_ROOT_DIR:-${ROOT_DIR}/.local/browser_chat_memory}"
export ALPHA_USERS_FILE="${ALPHA_USERS_FILE:-}"
export EVIDENCE_ROOT_DIR="${EVIDENCE_ROOT_DIR:-}"

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
from pathlib import Path

from aiohttp import web
from lifeform_service.alpha import (
    AlphaServiceConfig,
    load_alpha_users,
)
from lifeform_service.app import create_app
from lifeform_service.verticals import default_vertical_name, discover_verticals
from volvence_zero.substrate import SubstrateFallbackMode, build_transformers_runtime_with_fallback


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str_or_none(name: str) -> str | None:
    raw = os.environ.get(name, "").strip()
    return raw or None


def _build_alpha_config() -> AlphaServiceConfig:
    """Compose the AlphaServiceConfig from environment variables.

    Defaults: alpha mode ON, no allowlist (empty alpha_users = accept any
    user_id), memory_scope_root_dir under ./.local/browser_chat_memory.
    These defaults give the browser path cross-session memory: the
    chat UI binds the typed ``userId`` to a UserIdentity, which the
    kernel uses to build a filesystem-backed scoped MemoryStore so
    rupture-repair durable entries survive across sessions.
    """

    if not _env_bool("ALPHA_MODE", default=True):
        return AlphaServiceConfig()
    memory_dir = _env_str_or_none("MEMORY_SCOPE_ROOT_DIR")
    if memory_dir is None:
        raise RuntimeError(
            "ALPHA_MODE=1 requires MEMORY_SCOPE_ROOT_DIR to be set."
        )
    Path(memory_dir).mkdir(parents=True, exist_ok=True)
    evidence_dir = _env_str_or_none("EVIDENCE_ROOT_DIR")
    if evidence_dir is not None:
        Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    alpha_users_file = _env_str_or_none("ALPHA_USERS_FILE")
    return AlphaServiceConfig(
        enabled=True,
        memory_scope_root_dir=memory_dir,
        evidence_root_dir=evidence_dir,
        alpha_users=load_alpha_users(alpha_users_file),
    )


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

    alpha_config = _build_alpha_config()
    if alpha_config.enabled and verticals[vertical_name].alpha_factory is None:
        print(
            f"vertical {vertical_name!r} does not support alpha mode; "
            "set ALPHA_MODE=0 to fall back to anonymous in-memory sessions.",
            file=sys.stderr,
        )
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
        alpha_config=alpha_config,
    )
    if alpha_config.enabled:
        allowlist_size = len(alpha_config.alpha_users)
        allowlist_label = (
            f"allowlist={allowlist_size}-users"
            if allowlist_size
            else "allowlist=open"
        )
        print(
            "[start-browser-chat-qwen] cross-session memory ENABLED "
            f"memory_scope_root_dir={alpha_config.memory_scope_root_dir} "
            f"{allowlist_label}",
            flush=True,
        )
        print(
            "[start-browser-chat-qwen] type a 'userId' in the chat UI to "
            "identify yourself; the kernel binds it to a per-user MemoryStore.",
            flush=True,
        )
    else:
        print(
            "[start-browser-chat-qwen] anonymous mode (ALPHA_MODE=0); "
            "no cross-session memory, no per-user scope.",
            flush=True,
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
