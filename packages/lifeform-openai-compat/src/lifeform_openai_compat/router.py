"""aiohttp router that exposes ``POST /v1/chat/completions``.

This module wires the OpenAI-compat surface onto an existing
``aiohttp.web.Application`` produced by
:func:`lifeform_service.create_app`. The router is **additive**: it
does not modify or replace any existing route and does not require
any change in lifeform-service except a one-line ``add_openai_routes(app)``
call (Packet 5).

Three-mode dispatch:

* ``mode=lifeform`` (default) — full lifeform pipeline
  (:func:`lifeform_complete`). Sticky session reuse via
  :func:`derive_session_id`.
* ``mode=raw`` — bypass the lifeform; call ``runtime.generate(...)``
  directly (:func:`raw_substrate_complete`). For ablation track 3
  on EQ-Bench / EmpathyBench.

Selection precedence:

1. ``X-Compat-Mode`` request header
2. ``?mode=`` query param
3. Default = ``lifeform``

Error mapping:

* ``ValueError`` from DTO parsing or session-bridge validation →
  HTTP 400 ``invalid_*`` (the message prefix is the stable error code).
* :class:`RawSubstrateUnavailable` → HTTP 503 ``raw_substrate_unavailable``.
* Any other unexpected exception → HTTP 500 ``internal_error``
  (body has stable code; the exception is logged but its message is
  NOT echoed back to the client, since it could leak stack details).

Streaming: not yet supported. ``stream=true`` requests are rejected
with 501 ``streaming_not_supported`` so callers can fall back to
non-streaming mode rather than getting a malformed response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web

from lifeform_openai_compat.dto import ChatCompletionRequest
from lifeform_openai_compat.raw_substrate import (
    RawSubstrateUnavailable,
    raw_substrate_complete,
)
from lifeform_openai_compat.session_bridge import (
    LifeformCompletionResult,
    lifeform_complete,
)

_LOG = logging.getLogger("lifeform_openai_compat")

_DEFAULT_MODE: str = "lifeform"
_VALID_MODES: frozenset[str] = frozenset({"lifeform", "raw"})

_OPENAI_CHAT_ROUTE: str = "/v1/chat/completions"


def add_openai_routes(app: web.Application) -> None:
    """Attach the OpenAI-compat routes to ``app``.

    Mounts:

    * ``POST /v1/chat/completions`` — the only route external chat
      benchmarks (EQ-Bench 3, EmpathyBench, OpenAI Python client)
      strictly require. Idempotent at the app level: calling
      ``add_openai_routes`` twice on the same app raises so the
      host cli surfaces a clear error instead of silently
      double-mounting.

    Why no ``GET /v1/models``: lifeform-service has its own
    ``/v1/models`` route with a vertical-specific schema (substrate
    provider + swap capability flags), so a duplicate openai-shape
    ``/v1/models`` would collide. External clients that strictly
    need an OpenAI-shape model list can post-process the
    lifeform-service response, or hit the adapter through a proxy
    that injects the OpenAI shape. EQ-Bench 3 itself does not call
    ``/v1/models`` — it only POSTs to ``/v1/chat/completions``.
    """

    if app.get("openai_compat_mounted"):
        raise RuntimeError(
            "OpenAI-compat routes are already mounted on this app. "
            "add_openai_routes(app) must only be called once per app."
        )
    app.router.add_post(_OPENAI_CHAT_ROUTE, _handle_chat_completions)
    app["openai_compat_mounted"] = True


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


async def _handle_chat_completions(request: web.Request) -> web.Response:
    """Dispatch a chat-completions request to lifeform / raw mode."""

    payload = await _read_json_body(request)
    if payload is None:
        return _error(
            status=400,
            error="invalid_body",
            detail="POST body must be a non-empty JSON object",
        )

    try:
        parsed = ChatCompletionRequest.from_payload(payload)
    except ValueError as exc:
        return _error_from_value_error(exc)

    if parsed.generation.stream:
        return _error(
            status=501,
            error="streaming_not_supported",
            detail=(
                "stream=true is not supported by this adapter yet. "
                "Set stream=false to receive a single-shot response."
            ),
        )

    mode = _resolve_mode(request)
    if mode not in _VALID_MODES:
        return _error(
            status=400,
            error="invalid_mode",
            detail=(
                f"mode must be one of {sorted(_VALID_MODES)}; got {mode!r}. "
                "Set X-Compat-Mode header or ?mode= query param."
            ),
        )

    manager = request.app.get("session_manager")
    if manager is None:
        return _error(
            status=500,
            error="session_manager_missing",
            detail=(
                "lifeform-service did not register a SessionManager on "
                "this app. The OpenAI-compat router must be mounted on "
                "an app produced by lifeform_service.create_app(...)."
            ),
        )

    if mode == "raw":
        return await _dispatch_raw(parsed=parsed, manager=manager)
    # default lifeform path
    return await _dispatch_lifeform(parsed=parsed, manager=manager)


async def _dispatch_raw(*, parsed: ChatCompletionRequest, manager: Any) -> web.Response:
    runtime = manager.substrate_runtime
    try:
        response = raw_substrate_complete(request=parsed, runtime=runtime)
    except RawSubstrateUnavailable as exc:
        return _error(
            status=503,
            error="raw_substrate_unavailable",
            detail=str(exc),
        )
    except ValueError as exc:
        return _error_from_value_error(exc)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.exception("raw substrate path failed: %s", exc)
        return _error(
            status=500,
            error="internal_error",
            detail="raw substrate generate call failed; check service logs.",
        )
    headers = {
        "x-lifeform-mode": "raw",
        "x-lifeform-fingerprint": response.system_fingerprint,
    }
    return web.json_response(response.to_json(), status=200, headers=headers)


async def _dispatch_lifeform(*, parsed: ChatCompletionRequest, manager: Any) -> web.Response:
    try:
        result = await lifeform_complete(request=parsed, manager=manager)
    except ValueError as exc:
        return _error_from_value_error(exc)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.exception("lifeform path failed: %s", exc)
        return _error(
            status=500,
            error="internal_error",
            detail="lifeform run_turn failed; check service logs.",
        )
    headers = _lifeform_telemetry_headers(result)
    return web.json_response(result.response.to_json(), status=200, headers=headers)


def _lifeform_telemetry_headers(result: LifeformCompletionResult) -> dict[str, str]:
    """Surface lifeform-side telemetry on response headers.

    Headers are read by ``scripts/external_bench/run_eqbench3.py`` so
    the ablation-comparison report can correlate per-turn EQ-Bench
    rubric scores with the active regime / abstract action / PE
    magnitude. The body itself stays OpenAI-compatible.
    """
    headers = {
        "x-lifeform-mode": "lifeform",
        "x-lifeform-fingerprint": result.response.system_fingerprint,
        "x-lifeform-session-resolution": result.resolution.kind,
        "x-lifeform-pe-magnitude": f"{result.pe_magnitude:.4f}",
    }
    if result.active_regime:
        headers["x-lifeform-regime"] = result.active_regime
    if result.active_abstract_action:
        headers["x-lifeform-abstract-action"] = result.active_abstract_action
    if result.rationale_tags:
        # Header values must be ASCII; rationale tags are already ASCII
        # (snake-case identifiers) but we comma-join for terseness.
        headers["x-lifeform-rationale-tags"] = ",".join(result.rationale_tags)
    return headers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _read_json_body(request: web.Request) -> Any:
    """Best-effort parse: returns dict on success, None on missing/invalid."""
    if not request.body_exists:
        return None
    try:
        text = await request.text()
    except Exception:  # pragma: no cover - aiohttp internal
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _resolve_mode(request: web.Request) -> str:
    """Header → query param → default."""
    header = request.headers.get("X-Compat-Mode", "").strip().lower()
    if header:
        return header
    query = request.query.get("mode", "").strip().lower()
    if query:
        return query
    return _DEFAULT_MODE


def _error_from_value_error(exc: ValueError) -> web.Response:
    """Translate a typed ``ValueError`` (``invalid_*: detail``) into a 400."""
    msg = str(exc)
    if ":" in msg:
        code, _, detail = msg.partition(":")
        code = code.strip() or "invalid_request"
        detail = detail.strip()
    else:
        code = "invalid_request"
        detail = msg
    if not code.startswith("invalid_"):
        # Defensive: only translate codes that follow the convention
        # — anything else is a programmer error and must surface as 500.
        return _error(
            status=500,
            error="internal_error",
            detail="adapter raised non-conventional ValueError",
        )
    return _error(status=400, error=code, detail=detail)


def _error(*, status: int, error: str, detail: str) -> web.Response:
    body = {
        "error": {
            "code": error,
            "message": detail,
            "type": "invalid_request_error" if status == 400 else "service_error",
        }
    }
    return web.json_response(body, status=status)
