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

Streaming (debt #12 / #31): SSE streaming is implemented as a
post-hoc chunked re-emission of the non-streaming response — the
adapter completes the lifeform run synchronously, then frames the
result into one ``role:assistant`` content delta, optional
fingerprint delta, and a final ``[DONE]`` sentinel. This is
deliberately simpler than per-token streaming because the
underlying lifeform pipeline produces the answer atomically (so
any "chunked" framing here is presentational, not generative).
Real per-token streaming requires substrate-level streaming hooks
(DLaaS Slice 5.4 substrate streaming additive interface) and lands
when those hooks are wired.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
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

    streaming_requested = parsed.generation.stream

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
        return await _dispatch_raw(
            parsed=parsed,
            manager=manager,
            streaming=streaming_requested,
            request=request,
        )
    # default lifeform path
    return await _dispatch_lifeform(
        parsed=parsed,
        manager=manager,
        streaming=streaming_requested,
        request=request,
    )


async def _dispatch_raw(
    *,
    parsed: ChatCompletionRequest,
    manager: Any,
    streaming: bool,
    request: web.Request,
) -> web.StreamResponse:
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
    if streaming:
        return await _emit_sse(
            payload=response.to_json(),
            headers=headers,
            model=parsed.model,
            request=request,
        )
    return web.json_response(response.to_json(), status=200, headers=headers)


async def _dispatch_lifeform(
    *,
    parsed: ChatCompletionRequest,
    manager: Any,
    streaming: bool,
    request: web.Request,
) -> web.StreamResponse:
    manager_or_response = _resolve_lifeform_manager_for_request(
        request=request,
        parsed=parsed,
        default_manager=manager,
    )
    if isinstance(manager_or_response, web.Response):
        return manager_or_response
    resolved_manager = manager_or_response
    try:
        result = await lifeform_complete(request=parsed, manager=resolved_manager)
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
    payload = result.response.to_json()
    if streaming:
        return await _emit_sse(
            payload=payload,
            headers=headers,
            model=parsed.model,
            request=request,
        )
    return web.json_response(payload, status=200, headers=headers)


def _resolve_lifeform_manager_for_request(
    *,
    request: web.Request,
    parsed: ChatCompletionRequest,
    default_manager: Any,
) -> Any | web.Response:
    """Resolve optional DLaaS ``ai_id`` metadata to a SessionManager.

    The OpenAI body stays compatible: DLaaS routing is opt-in through
    ``metadata["dlaas.ai_id"]``. We deliberately avoid importing the
    launcher package here so the OpenAI adapter remains installable as
    a thin lifeform-service facade; the app key is owned by
    ``dlaas-platform-launcher`` and only present in full-stack DLaaS
    apps.
    """

    ai_id = parsed.metadata.get("dlaas.ai_id", "").strip()
    if not ai_id:
        return default_manager
    launcher = request.app.get("dlaas_instance_manager")
    if launcher is None:
        return _error(
            status=404,
            error="dlaas_instance_manager_missing",
            detail=(
                "metadata['dlaas.ai_id'] was provided, but this service "
                "is not running the DLaaS InstanceManager."
            ),
        )
    try:
        return launcher.get(ai_id)
    except LookupError:
        return _error(
            status=404,
            error="ai_id_not_found",
            detail=f"metadata['dlaas.ai_id']={ai_id!r} is not adopted.",
        )


async def _emit_sse(
    *,
    payload: dict,
    headers: dict[str, str],
    model: str,
    request: web.Request,
) -> web.StreamResponse:
    """Stream a non-streaming OpenAI completion as SSE chunks (debt #12 / #31).

    Frame order (matches OpenAI's ``stream=true`` wire format):

    1. ``role:assistant`` opening delta (no content yet).
    2. ``content`` delta carrying the full assistant text.
    3. final chunk with ``finish_reason: "stop"`` + usage echo.
    4. ``data: [DONE]`` sentinel.

    Lifeform telemetry headers are preserved on the SSE response so
    harnesses still get ``x-lifeform-*`` even when streaming.
    """

    response = web.StreamResponse(
        status=200,
        headers={
            **headers,
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)
    completion_id = f"chatcmpl-stream-{uuid.uuid4().hex[:24]}"
    created_ts = int(time.time())

    def _sse(chunk: dict) -> bytes:
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")

    # Frame 1: role delta
    await response.write(_sse({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }))

    # Frame 2: content delta — extract assistant text from the
    # OpenAI-shaped non-streaming payload.
    choices = payload.get("choices") or []
    assistant_text = ""
    finish_reason = "stop"
    if choices:
        first = choices[0]
        message = first.get("message") or {}
        assistant_text = str(message.get("content") or "")
        finish_reason = str(first.get("finish_reason") or "stop")
    await response.write(_sse({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": assistant_text},
            "finish_reason": None,
        }],
    }))

    # Frame 3: final chunk with finish_reason + usage echo.
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    }
    if "usage" in payload:
        final_chunk["usage"] = payload["usage"]
    await response.write(_sse(final_chunk))

    # Frame 4: [DONE] sentinel per OpenAI streaming convention.
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


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
