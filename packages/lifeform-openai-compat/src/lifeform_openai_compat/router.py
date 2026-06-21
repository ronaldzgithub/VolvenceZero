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
    SessionEndUserMismatchError,
    lifeform_complete,
)

_LOG = logging.getLogger("lifeform_openai_compat")

_DEFAULT_MODE: str = "lifeform"
_VALID_MODES: frozenset[str] = frozenset({"lifeform", "raw"})

_OPENAI_CHAT_ROUTE: str = "/v1/chat/completions"
_AUTH_APP_KEY: str = "openai_compat_api_keys"

# Optional post-turn observability hook. A full-stack DLaaS host registers
# a callable here (``dlaas_platform_api`` sets ``_record_openai_compat_
# snapshot``) so a successful lifeform ``/v1/chat/completions`` turn also
# writes a DLaaS cognition snapshot — aligning OpenAI-path observability
# with the native typed-envelope dispatch. The adapter stays decoupled:
# the value is an opaque callable, never an import of the recorder. Absent
# key = no-op (e.g. a bare lifeform-service host), so this is additive and
# reversible. Signature: ``hook(request, *, ai_id, session, session_id)``.
_ON_TURN_APP_KEY: str = "openai_compat_on_turn"


def add_openai_routes(
    app: web.Application,
    *,
    api_keys: tuple[str, ...] = (),
) -> None:
    """Attach the OpenAI-compat routes to ``app``.

    Mounts:

    * ``POST /v1/chat/completions`` — the only route external chat
      benchmarks (EQ-Bench 3, EmpathyBench, OpenAI Python client)
      strictly require. Idempotent at the app level: calling
      ``add_openai_routes`` twice on the same app raises so the
      host cli surfaces a clear error instead of silently
      double-mounting.

    ``api_keys`` is optional so in-process tests can mount the adapter
    without auth, while ``lifeform-serve --enable-openai-compat`` can
    require an OpenAI-style ``Authorization: Bearer ...`` header.

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
    app[_AUTH_APP_KEY] = frozenset(key.strip() for key in api_keys if key.strip())
    app.router.add_post(_OPENAI_CHAT_ROUTE, _handle_chat_completions)
    app["openai_compat_mounted"] = True


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


async def _handle_chat_completions(request: web.Request) -> web.Response:
    """Dispatch a chat-completions request to lifeform / raw mode."""

    auth_error = _authorize_request(request)
    if auth_error is not None:
        return auth_error

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
    except SessionEndUserMismatchError as exc:
        return _error(
            status=409,
            error="session_end_user_mismatch",
            detail=str(exc),
        )
    except ValueError as exc:
        return _error_from_value_error(exc)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.exception("lifeform path failed: %s", exc)
        return _error(
            status=500,
            error="internal_error",
            detail="lifeform run_turn failed; check service logs.",
        )
    _invoke_on_turn_hook(request=request, parsed=parsed, result=result)
    headers = _lifeform_telemetry_headers(result)
    payload = result.response.to_json()
    if streaming:
        return await _emit_sse(
            payload=payload,
            headers=headers,
            model=parsed.model,
            request=request,
            # U6: thread structured L3 evidence pointers (or empty
            # tuple when no figure bundle was bound) into the SSE
            # stream so the final ``event: evidence`` frame surfaces
            # them to the family-memorial client.
            evidence_pointers=result.evidence_pointers,
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
    evidence_pointers: tuple[dict, ...] = (),
) -> web.StreamResponse:
    """Stream a non-streaming OpenAI completion as SSE chunks (debt #12 / #31).

    Frame order (matches OpenAI's ``stream=true`` wire format):

    1. ``role:assistant`` opening delta (no content yet).
    2. ``content`` delta carrying the full assistant text.
    3. final chunk with ``finish_reason: "stop"`` + usage echo.
    4. **(U6, optional)** ``event: evidence`` frame carrying the
       structured L3 ``EvidencePointer`` list — only emitted when
       the caller supplied non-empty ``evidence_pointers``. This
       sits between the final chunk and ``[DONE]`` so OpenAI-only
       clients that ignore unknown ``event:`` lines still consume
       the standard sequence unchanged; clients that DO consume it
       (apps/family-memorial's CitationCard) render clickable
       citations linking back to the original family corpus.
    5. ``data: [DONE]`` sentinel.

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

    # Frame 2: content/tool_calls delta — extract assistant payload from the
    # OpenAI-shaped non-streaming response.
    choices = payload.get("choices") or []
    assistant_text = ""
    finish_reason = "stop"
    tool_calls: list[dict[str, Any]] = []
    if choices:
        first = choices[0]
        message = first.get("message") or {}
        assistant_text = str(message.get("content") or "")
        raw_tool_calls = message.get("tool_calls") or []
        if isinstance(raw_tool_calls, list):
            tool_calls = raw_tool_calls
        finish_reason = str(first.get("finish_reason") or "stop")
    delta: dict[str, Any] = {}
    if tool_calls:
        delta["tool_calls"] = [
            {
                "index": index,
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "function": call.get("function", {}),
            }
            for index, call in enumerate(tool_calls)
            if isinstance(call, dict)
        ]
    else:
        delta["content"] = assistant_text
    await response.write(_sse({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
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

    # Frame 4 (U6 — optional): structured L3 evidence frame. Sits
    # between the final completion chunk and the ``[DONE]`` sentinel
    # so OpenAI-only clients pick up the spec-mandated sequence
    # unchanged; clients that read ``event: evidence`` (apps/
    # family-memorial CitationCard) deserialize the JSON list and
    # render clickable citations.
    if evidence_pointers:
        evidence_payload = json.dumps(
            {"pointers": list(evidence_pointers), "schema_version": 1},
            ensure_ascii=False,
        )
        await response.write(
            f"event: evidence\ndata: {evidence_payload}\n\n".encode("utf-8")
        )

    # Frame 5: [DONE] sentinel per OpenAI streaming convention.
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


def _invoke_on_turn_hook(
    *,
    request: web.Request,
    parsed: ChatCompletionRequest,
    result: LifeformCompletionResult,
) -> None:
    """Best-effort post-turn observability hook (additive, decoupled).

    When the host registered ``app['openai_compat_on_turn']`` (a full-stack
    DLaaS app does), hand it the resolved session so it can record a
    cognition snapshot for this turn — making the OpenAI front door as
    observable as the native ``/interactions`` dispatch. Bare
    lifeform-service hosts register no hook, so this is a no-op there.

    Wrapped in a broad ``except``: the chat completion is the user-visible
    response and must never 500 because an observability sink failed.
    """
    hook = request.app.get(_ON_TURN_APP_KEY)
    if hook is None or result.session is None:
        return
    ai_id = parsed.metadata.get("dlaas.ai_id", "").strip()
    if not ai_id:
        # No adopted instance bound → nothing to attribute a snapshot to.
        return
    try:
        hook(
            request,
            ai_id=ai_id,
            session=result.session,
            session_id=result.resolution.session_id,
        )
    except Exception as exc:  # pragma: no cover - defensive sink
        _LOG.warning("openai_compat on-turn hook failed: %s", exc)


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
    if result.expression_intent:
        # Drives the presence avatar on OpenAI-compat consumers
        # (einstein / family-memorial); value is a snake/kebab
        # identifier, ASCII-safe for a header.
        headers["x-lifeform-expression-intent"] = result.expression_intent
    if result.rationale_tags:
        # Header values must be ASCII; rationale tags are already ASCII
        # (snake-case identifiers) but we comma-join for terseness.
        headers["x-lifeform-rationale-tags"] = ",".join(result.rationale_tags)
    if result.confidence is not None:
        # Kernel-PE calibrated confidence (deploy debt D-collab-pe):
        # absent header = the kernel published no PE snapshot this turn;
        # consumers must fall back honestly, never fabricate a value.
        headers["x-lifeform-confidence"] = f"{result.confidence:.4f}"
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


def _authorize_request(request: web.Request) -> web.Response | None:
    """Validate optional OpenAI-style Bearer auth for chat completions."""

    configured_keys = request.app[_AUTH_APP_KEY]
    if not configured_keys:
        return None
    header = request.headers.get("Authorization", "").strip()
    prefix = "Bearer "
    if not header.startswith(prefix):
        return _error(
            status=401,
            error="unauthorized",
            detail="Missing Authorization: Bearer token for OpenAI-compatible route.",
        )
    token = header[len(prefix):].strip()
    if token not in configured_keys:
        return _error(
            status=401,
            error="unauthorized",
            detail="Invalid bearer token for OpenAI-compatible route.",
        )
    return None


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
