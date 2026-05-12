#!/usr/bin/env bash
# Start the browser chat service with a real Hugging Face Qwen substrate.
#
# Defaults target Qwen2.5-7B-Instruct: the smallest base model that
# reliably follows VZ's structured system prompt AND keeps multi-turn
# coherence on short follow-ups. The 0.5B / 1.5B / 3B variants tend
# to collapse into single-character or off-topic replies once the
# kernel's plan/ordering instructions are stacked on top of the user
# turn.
#
# ---------------------------------------------------------------------
# What you can actually run locally
# ---------------------------------------------------------------------
# Reference target: MacBook Air M4, 24 GB unified memory, ~30 GB free disk.
# bf16 = transformers default, Q4 = GGUF / llama.cpp 4-bit quantization.
#
#   Model                       bf16 RAM   Q4 RAM   Disk    Verdict on M4 24GB
#   --------------------------- ---------- -------- ------- ------------------
#   Qwen2.5-1.5B-Instruct       ~ 4 GB     ~ 1 GB    3 GB   too weak for VZ prompt
#   Qwen2.5-3B-Instruct         ~ 8 GB     ~ 2 GB    6 GB   borderline coherent
#   Qwen2.5-7B-Instruct         ~16 GB     ~ 5 GB   15 GB   recommended (default)
#   Qwen2.5-14B-Instruct        ~28 GB     ~ 9 GB   28 GB   bf16 NO; Q4 OK
#   Qwen2.5-32B-Instruct        ~64 GB     ~18 GB   62 GB   bf16 NO; Q4 tight
#   Qwen2.5-72B-Instruct       ~145 GB     ~40 GB  145 GB   NOT runnable locally
#   Qwen3-235B-A22B (MoE)      ~470 GB    ~120 GB  470 GB   NOT runnable locally
#   Qwen3-Coder-480B-A35B (MoE)~960 GB    ~150 GB  960 GB   NOT runnable locally
#
# So on this hardware the practical "biggest that still fits" is
# Qwen2.5-32B-Instruct at Q4 (use a llama.cpp / mlx-lm runtime, not the
# transformers backend). Anything bigger requires a cloud GPU box.
#
# ---------------------------------------------------------------------
# Largest open Qwen models on Hugging Face (Qwen org, 2026-05)
# ---------------------------------------------------------------------
# Reference only — these CANNOT be served from this script directly.
#   * Qwen/Qwen3-Coder-480B-A35B-Instruct      MoE, 480B total / 35B active, code
#   * Qwen/Qwen3.5-397B-A17B                   MoE, 397B / 17B, general flagship
#   * Qwen/Qwen3-VL-235B-A22B-Instruct         MoE, 235B / 22B, multimodal
#   * Qwen/Qwen3-235B-A22B-Instruct-2507       MoE, 235B / 22B, general
#   * Qwen/Qwen2.5-72B-Instruct                72B dense (largest dense Qwen)
# Use these via remote inference endpoints (DashScope / vLLM cluster),
# not on a 24 GB Mac.
#
# ---------------------------------------------------------------------
# Cross-session memory
# ---------------------------------------------------------------------
# ALPHA_MODE defaults to 1 so the kernel binds each session to the
# ``userId`` typed in the chat UI (sent as X-Alpha-User) and persists
# per-user durable memory under MEMORY_SCOPE_ROOT_DIR. Set
# ALPHA_MODE=0 to fall back to the previous anonymous, in-memory-only
# behavior.
#
# ---------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------
#   bash start_browser_chat_qwen.sh                                # 7B default
#   MODEL_ID=Qwen/Qwen2.5-3B-Instruct bash start_browser_chat_qwen.sh
#   MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct bash start_browser_chat_qwen.sh
#
# If HuggingFace is slow / blocked, route through the mirror:
#   HF_ENDPOINT=https://hf-mirror.com bash start_browser_chat_qwen.sh
#   ALPHA_MODE=0 bash start_browser_chat_qwen.sh   # anonymous, no persistence
#
# Useful env:
#   HOST=127.0.0.1
#   PORT=8765
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct        # default; see sizing table above
#   DEVICE=auto
#   LOCAL_FILES_ONLY=0
#   OPEN_BROWSER=1
#   ALPHA_MODE=1                             # 1 = scoped memory, 0 = anonymous
#   MEMORY_SCOPE_ROOT_DIR=$ROOT_DIR/.local/browser_chat_memory
#   ALPHA_USERS_FILE=                        # optional JSON allowlist
#   EVIDENCE_ROOT_DIR=                       # optional alpha evidence dir
#   TEMPLATES_ROOT_DIR=$ROOT_DIR/artifacts/lifeform-templates
#                                            # chat UI lists/saves templates
#                                            # under <root>/<vertical>/*.json
#   MODEL_ID_ALLOWLIST=                      # comma-separated extra Qwen ids
#                                            # for the chat UI's "Switch Model"
#                                            # dropdown; empty = curated default
#   PROTOCOL_AUTOLOAD_DIR=                   # optional dir scanned at startup
#                                            # for PDFs/MDs to extract into
#                                            # pending behavior-protocol candidates
#                                            # (requires PROTOCOL_LLM_* below)
#   PROTOCOL_AUTOLOAD_FORCE_APPROVE=0        # DEV: 1 = auto-approve scanned
#                                            # candidates (skip review). Use only
#                                            # for local dev / smoke tests.
#   PROTOCOL_LLM_BASE_URL=                   # OpenAI-compatible endpoint for
#                                            # protocol uptake LLM (PDF / MD /
#                                            # description routes). Examples:
#                                            #   https://api.openai.com/v1
#                                            #   http://localhost:8000/v1 (vLLM)
#   PROTOCOL_LLM_API_KEY=                    # API key for the above
#   PROTOCOL_LLM_MODEL=gpt-4o-mini           # model_id for extraction calls
#   PROTOCOL_LLM_TIMEOUT_SECONDS=60
#   HF_HOME=$ROOT_DIR/.local/hf-cache        # where HuggingFace caches model
#                                            # weights / tokenizers / datasets.
#                                            # Default lives next to the repo so
#                                            # the cache lands on whichever drive
#                                            # the checkout sits on (15+ GB per
#                                            # 7B Qwen — set this to a drive
#                                            # with headroom). Set HF_HOME=""
#                                            # to fall back to the system default
#                                            # (~/.cache/huggingface).

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
# Template surface: when set, the chat UI's template <select> is
# populated from <root>/<vertical_subdir>/*.json and "Save as Template"
# writes back to that directory. Defaults to the same location used
# by examples/train_zhang_wuji_template.py so existing zhang-wuji
# templates show up automatically.
export TEMPLATES_ROOT_DIR="${TEMPLATES_ROOT_DIR:-${ROOT_DIR}/artifacts/lifeform-templates}"
# Substrate hot-swap: comma-separated list of additional Qwen variants
# the chat UI's "Switch Model" dropdown advertises. Empty = the
# curated default lineup (1.5B / 3B / 7B / 14B Qwen2.5 Instruct).
# MODEL_ID is always added in addition to this list so the startup
# model is never locked out of its own dropdown.
export MODEL_ID_ALLOWLIST="${MODEL_ID_ALLOWLIST:-}"

# Behavior Protocol Runtime engineering wrap.
# Optional uptake (PDF / MD / task description / API injection) routes. Mounted
# unconditionally in create_app; LLM-backed routes self-disable (HTTP 503)
# when PROTOCOL_LLM_BASE_URL / PROTOCOL_LLM_API_KEY are unset, so it is safe
# to leave these blank for default startup. ``from-payload`` (no LLM) still
# works.
export PROTOCOL_AUTOLOAD_DIR="${PROTOCOL_AUTOLOAD_DIR:-}"
export PROTOCOL_AUTOLOAD_FORCE_APPROVE="${PROTOCOL_AUTOLOAD_FORCE_APPROVE:-0}"
export PROTOCOL_LLM_BASE_URL="${PROTOCOL_LLM_BASE_URL:-}"
export PROTOCOL_LLM_API_KEY="${PROTOCOL_LLM_API_KEY:-}"
export PROTOCOL_LLM_MODEL="${PROTOCOL_LLM_MODEL:-gpt-4o-mini}"
export PROTOCOL_LLM_TIMEOUT_SECONDS="${PROTOCOL_LLM_TIMEOUT_SECONDS:-60}"

# HuggingFace cache location. Default to a per-repo cache so model
# weights land on whatever drive the checkout sits on (typical 7B
# Qwen download is ~15 GB). Setting HF_HOME="" explicitly opts into
# the system default (``~/.cache/huggingface``); leaving it unset
# uses our per-repo default below. ``HF_HUB_CACHE`` is derived from
# ``HF_HOME`` automatically by huggingface_hub.
if [[ -z "${HF_HOME+x}" ]]; then
  export HF_HOME="${ROOT_DIR}/.local/hf-cache"
fi
if [[ -n "${HF_HOME}" ]]; then
  mkdir -p "${HF_HOME}"
fi

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
echo "[start-browser-chat-qwen] hf_home=${HF_HOME:-<system default>}"
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
from lifeform_service.openai_compat_client import build_client_from_env
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from lifeform_service.substrate_registry import (
    build_qwen_runtime_loader,
    build_substrate_provider_from_env,
)
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

    default_vertical = requested_vertical or default_vertical_name()
    if default_vertical not in verticals:
        print(
            f"Unknown VERTICAL={default_vertical!r}. Available: {sorted(verticals)}",
            file=sys.stderr,
        )
        return 1

    alpha_config = _build_alpha_config()
    if alpha_config.enabled and verticals[default_vertical].alpha_factory is None:
        print(
            f"default vertical {default_vertical!r} does not support alpha mode; "
            "pick a different VERTICAL or set ALPHA_MODE=0.",
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

    # Hot-swap-capable provider so the chat UI can switch Qwen
    # variants at runtime. The loader closure captures device /
    # local-files / fallback-mode once so swap requests don't need
    # to re-thread those parameters.
    runtime_loader = build_qwen_runtime_loader(
        device=device,
        local_files_only=local_files_only,
        fallback_mode=SubstrateFallbackMode.DENY,
    )
    substrate_provider = build_substrate_provider_from_env(
        initial_runtime=runtime,
        initial_model_id=model_id,
        runtime_loader=runtime_loader,
        allowlist_env=_env_str_or_none("MODEL_ID_ALLOWLIST"),
    )

    templates_root_dir = _env_str_or_none("TEMPLATES_ROOT_DIR")

    autoload_dir_raw = _env_str_or_none("PROTOCOL_AUTOLOAD_DIR")
    autoload_force_approve = _env_bool("PROTOCOL_AUTOLOAD_FORCE_APPROVE", default=False)
    uptake_service = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=Path(autoload_dir_raw) if autoload_dir_raw else None,
            autoload_force_approve=autoload_force_approve,
            llm_client_factory=lambda: build_client_from_env(),
        ),
    )

    app = create_app(
        verticals=verticals,
        default_vertical=default_vertical,
        max_sessions=max_sessions,
        idle_eviction_seconds=idle_eviction_seconds,
        substrate_provider=substrate_provider,
        alpha_config=alpha_config,
        templates_root_dir=templates_root_dir,
        protocol_uptake_service=uptake_service,
    )

    if uptake_service._config.autoload_dir is not None:
        async def _run_autoload(_: web.Application) -> None:
            results = await uptake_service.autoload_directory()
            ok = sum(1 for r in results if r.status == "ok")
            err = sum(1 for r in results if r.status == "error")
            print(
                f"[start-browser-chat-qwen] protocol autoload: "
                f"dir={uptake_service._config.autoload_dir} "
                f"ok={ok} error={err} "
                f"(force_approve={uptake_service._config.autoload_force_approve})",
                flush=True,
            )
        app.on_startup.append(_run_autoload)
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
    if templates_root_dir is not None:
        print(
            "[start-browser-chat-qwen] templates ENABLED "
            f"templates_root_dir={templates_root_dir}",
            flush=True,
        )
    else:
        print(
            "[start-browser-chat-qwen] templates DISABLED "
            "(set TEMPLATES_ROOT_DIR to enable list/save in chat UI)",
            flush=True,
        )
    available_models = ", ".join(
        spec.model_id for spec in substrate_provider.available
    )
    print(
        "[start-browser-chat-qwen] substrate-swap ENABLED "
        f"current={substrate_provider.current_model_id} "
        f"allowlist=[{available_models}]",
        flush=True,
    )
    if uptake_service.llm_client is not None:
        print(
            "[start-browser-chat-qwen] protocol uptake routes ENABLED "
            "(PDF / Markdown / task-description / API-injection)",
            flush=True,
        )
    else:
        print(
            "[start-browser-chat-qwen] protocol uptake routes mounted "
            "but extraction is disabled (no PROTOCOL_LLM_BASE_URL / "
            "PROTOCOL_LLM_API_KEY); from-payload route still works",
            flush=True,
        )
    available_verticals = ", ".join(sorted(verticals))
    print(
        "[start-browser-chat-qwen] ready "
        f"default_vertical={default_vertical} "
        f"verticals=[{available_verticals}] "
        f"model_id={model_id} runtime_origin={runtime_origin}",
        flush=True,
    )
    print(f"[start-browser-chat-qwen] listening on http://{host}:{port}/chat", flush=True)
    web.run_app(app, host=host, port=port, print=lambda *_: None)
    return 0


raise SystemExit(main())
PY
