"""aiohttp Application factory + route handlers for the lifeform service.

Routes live in this module rather than spread across packages because
keeping them together makes the API surface easy to audit. The handlers
are deliberately thin: each one (a) parses the request DTO, (b) calls into
``SessionManager`` / ``LifeformSession``, (c) maps the result back to a
DTO. No business logic.

JSON contract:

* All success responses are JSON objects.
* All error responses are ``ErrorResponse`` JSON with HTTP status >= 400.
* Status codes:
    * 200 OK \u2014 normal response
    * 201 Created \u2014 ``POST /v1/sessions``
    * 400 Bad Request \u2014 missing required fields, malformed JSON
    * 404 Not Found \u2014 unknown ``session_id``
    * 409 Conflict \u2014 ``session_id`` collision on explicit create
    * 500 Internal Server Error \u2014 kernel exception (logged, not leaked)

Substrate sharing: ``create_app(substrate_runtime=...)`` accepts a
single pre-built runtime that is shared across every session this
service hosts. The runtime is checked at construction time for the
"frozen substrate" invariant (R2): if its
``supports_live_substrate_mutation`` flag is True, ``create_app``
raises rather than silently letting one session's adapter-delta updates
corrupt every other session's weights. Sharing is the default deployment
mode for one-GPU, multi-tenant servers.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from volvence_zero.substrate import OpenWeightResidualRuntime

from lifeform_service.dto import (
    CreateSessionResponse,
    EndSceneResponse,
    ErrorResponse,
    HealthResponse,
    ServiceInfoResponse,
    SessionStateResponse,
    TurnResponse,
)
from lifeform_service.session_manager import (
    SessionAlreadyExistsError,
    SessionManager,
    SessionNotFoundError,
)
from lifeform_service.verticals import VerticalSpec


_LOG = logging.getLogger("lifeform_service")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    vertical: VerticalSpec,
    max_sessions: int = 256,
    idle_eviction_seconds: float | None = 60 * 30,
    substrate_runtime: "OpenWeightResidualRuntime | None" = None,
) -> web.Application:
    """Build the aiohttp Application that fronts a single vertical.

    Args:
        vertical: which vertical (and its factory) this service hosts.
        max_sessions: cap on concurrently live sessions before LRU
            eviction.
        idle_eviction_seconds: auto-close sessions idle longer than this;
            ``None`` disables idle eviction.
        substrate_runtime: pre-built runtime shared across every session.
            If ``None`` the vertical's factory builds a fresh runtime per
            session (synthetic mode \u2014 fine for tests, wasteful for HF).
            When supplied, the runtime MUST be frozen
            (``supports_live_substrate_mutation == False``) \u2014 otherwise
            this constructor raises ``ValueError``.
    """
    if substrate_runtime is not None:
        _enforce_frozen_for_sharing(substrate_runtime)
    manager = SessionManager(
        lifeform_factory=vertical.factory,
        vertical_name=vertical.name,
        max_sessions=max_sessions,
        idle_eviction_seconds=idle_eviction_seconds,
        substrate_runtime=substrate_runtime,
    )
    app = web.Application(middlewares=[_error_middleware])
    app["session_manager"] = manager
    app["vertical_spec"] = vertical
    app["substrate_runtime"] = substrate_runtime
    app.router.add_get("/v1/health", _handle_health)
    app.router.add_get("/v1/info", _handle_info)
    app.router.add_post("/v1/sessions", _handle_create_session)
    app.router.add_delete("/v1/sessions/{session_id}", _handle_close_session)
    app.router.add_get("/v1/sessions/{session_id}/state", _handle_session_state)
    app.router.add_post("/v1/sessions/{session_id}/turns", _handle_turn)
    app.router.add_post("/v1/sessions/{session_id}/end-scene", _handle_end_scene)
    return app


def _enforce_frozen_for_sharing(runtime: "OpenWeightResidualRuntime") -> None:
    """R2 invariant: a shared runtime must NOT permit live mutation.

    Per-session adapter deltas would corrupt every other session's
    weights when the underlying ``_model`` is the same Python object.
    If a deployment legitimately needs per-session adapters, the path
    is to refactor that mutable state out of the runtime and into the
    per-session ``SubstrateAdapter`` \u2014 not to flip this flag.
    """
    if getattr(runtime, "supports_live_substrate_mutation", False):
        raise ValueError(
            "Cannot share a runtime that has supports_live_substrate_mutation=True "
            "across sessions: per-session adapter-delta updates would corrupt other "
            "sessions' weights. Build the runtime with "
            "allow_live_substrate_mutation=False (the default) when sharing."
        )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@web.middleware
async def _error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except web.HTTPException:
        raise  # already a structured HTTP error
    except SessionNotFoundError as exc:
        return _json_error(
            status=404,
            error="session_not_found",
            detail=str(exc) or "session_id is unknown",
            extra={"session_id": str(exc)},
        )
    except SessionAlreadyExistsError as exc:
        return _json_error(
            status=409,
            error="session_already_exists",
            detail=str(exc),
        )
    except _BadRequest as exc:
        return _json_error(status=400, error=exc.code, detail=exc.detail)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.exception("Unhandled error in lifeform-service handler: %s", exc)
        return _json_error(
            status=500,
            error="internal_error",
            detail="An unexpected error occurred. Check service logs.",
        )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_health(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    body = HealthResponse(
        status="ok",
        session_count=manager.session_count(),
        vertical=manager.vertical_name,
    )
    return _json_ok(body.to_json())


async def _handle_info(request: web.Request) -> web.Response:
    spec: VerticalSpec = request.app["vertical_spec"]
    runtime = request.app.get("substrate_runtime")
    body = ServiceInfoResponse(
        vertical=spec.name,
        has_temporal_bootstrap=spec.has_temporal_bootstrap,
        has_regime_bootstrap=spec.has_regime_bootstrap,
        bootstraps_dir=spec.bootstraps_dir,
        scenarios_dir=spec.scenarios_dir,
        substrate_shared=runtime is not None,
        substrate_model_id=getattr(runtime, "model_id", None),
        substrate_runtime_origin=getattr(runtime, "runtime_origin", None),
    )
    return _json_ok(body.to_json())


async def _handle_create_session(request: web.Request) -> web.Response:
    payload = await _maybe_json(request)
    requested_id = None
    if isinstance(payload, dict):
        raw = payload.get("session_id")
        if raw is not None and not isinstance(raw, str):
            raise _BadRequest("invalid_session_id", "session_id must be a string")
        if isinstance(raw, str):
            requested_id = raw
    manager: SessionManager = request.app["session_manager"]
    spec: VerticalSpec = request.app["vertical_spec"]
    session = await manager.create_session(session_id=requested_id)
    body = CreateSessionResponse(
        session_id=session.session_id,
        vertical=spec.name,
        has_temporal_bootstrap=spec.has_temporal_bootstrap,
        has_regime_bootstrap=spec.has_regime_bootstrap,
    )
    return _json_ok(body.to_json(), status=201)


async def _handle_close_session(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    closed = await manager.close_session(session_id)
    if not closed:
        raise SessionNotFoundError(session_id)
    return _json_ok({"session_id": session_id, "closed": True})


async def _handle_session_state(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    session = await manager.get_session(session_id)
    open_scene = session.open_scene
    summaries = session.turn_summaries
    last_summary = summaries[-1] if summaries else None
    body = SessionStateResponse(
        session_id=session_id,
        open_scene_id=open_scene.scene_id if open_scene else None,
        open_scene_turn_count=open_scene.turn_count if open_scene else 0,
        closed_scene_count=len(session.closed_scenes),
        turn_count=len(summaries),
        pending_followup_count=len(session.all_pending_followups()),
        last_active_regime=last_summary.active_regime if last_summary else None,
        last_active_abstract_action=(
            last_summary.active_abstract_action if last_summary else None
        ),
        last_response_text=session.latest_response_text,
    )
    return _json_ok(body.to_json())


async def _handle_turn(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    payload = await _require_json(request)
    user_input = payload.get("user_input")
    if not isinstance(user_input, str) or not user_input.strip():
        raise _BadRequest("invalid_user_input", "user_input must be a non-empty string")
    session = await manager.get_session(session_id)
    result = await session.run_turn(user_input)

    summaries = session.turn_summaries
    summary = summaries[-1] if summaries else None
    expression_intent: str | None = None
    assembly = result.active_snapshots.get("response_assembly")
    if assembly is not None:
        expression_intent = getattr(assembly.value, "expression_intent", None)
    open_loop_count = 0
    open_loop = result.active_snapshots.get("open_loop")
    if open_loop is not None:
        open_loop_count = len(getattr(open_loop.value, "unresolved_loops", ()) or ())
    commitment_count = 0
    commitment = result.active_snapshots.get("commitment")
    if commitment is not None:
        commitment_count = len(getattr(commitment.value, "active_commitments", ()) or ())
    pe_magnitude = summary.pe_magnitude if summary is not None else 0.0

    open_scene = session.open_scene
    body = TurnResponse(
        session_id=session_id,
        scene_id=open_scene.scene_id if open_scene else "scene-?",
        turn_index=summary.turn_index if summary is not None else 0,
        response_text=result.response.text,
        active_regime=result.active_regime,
        active_abstract_action=result.active_abstract_action,
        expression_intent=expression_intent,
        pe_magnitude=pe_magnitude,
        open_loop_count=open_loop_count,
        commitment_count=commitment_count,
    )
    return _json_ok(body.to_json())


async def _handle_end_scene(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    payload = await _maybe_json(request)
    drain = True
    reason = "scene-end"
    if isinstance(payload, dict):
        if "drain_slow_loop" in payload:
            if not isinstance(payload["drain_slow_loop"], bool):
                raise _BadRequest(
                    "invalid_drain_flag", "drain_slow_loop must be boolean"
                )
            drain = payload["drain_slow_loop"]
        if "reason" in payload:
            if not isinstance(payload["reason"], str):
                raise _BadRequest("invalid_reason", "reason must be string")
            reason = payload["reason"]
    session = await manager.get_session(session_id)
    closed = await session.end_scene(reason=reason, drain_slow_loop=drain)
    body = EndSceneResponse(
        session_id=session_id,
        closed_scene_id=closed.scene_id if closed is not None else None,
        slow_loop_drained=drain and closed is not None,
    )
    return _json_ok(body.to_json())


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_ok(payload: dict[str, Any], *, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status)


def _json_error(
    *,
    status: int,
    error: str,
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> web.Response:
    body = ErrorResponse(error=error, detail=detail, extra=extra or {})
    return web.json_response(body.to_json(), status=status)


async def _require_json(request: web.Request) -> dict[str, Any]:
    body = await _maybe_json(request)
    if not isinstance(body, dict):
        raise _BadRequest(
            "invalid_request_body",
            "Expected a JSON object body.",
        )
    return body


async def _maybe_json(request: web.Request) -> Any:
    if not request.body_exists:
        return None
    try:
        text = await request.text()
    except Exception as exc:
        raise _BadRequest("invalid_body", f"Could not read body: {exc}") from exc
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _BadRequest("invalid_json", f"Body is not valid JSON: {exc}") from exc


class _BadRequest(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
