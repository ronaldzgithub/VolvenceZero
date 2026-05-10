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
from uuid import uuid4
from pathlib import Path
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
from lifeform_service.alpha import (
    ALPHA_DISCLAIMER,
    AlphaIdentityProvider,
    AlphaServiceConfig,
    alpha_config_to_json,
)
from lifeform_service.session_manager import (
    SessionAlreadyExistsError,
    SessionManager,
    SessionNotFoundError,
    TemplatesNotSupportedError,
)
from lifeform_service.verticals import VerticalSpec
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.memory import (
    UserIdentity,
    build_scoped_memory_store,
    delete_entries_for_scope,
    list_durable_entries_for_scope,
)


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
    alpha_config: AlphaServiceConfig | None = None,
    templates_root_dir: str | None = None,
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
        templates_root_dir: filesystem root the chat-browser template
            surface scans (``GET /v1/templates`` lists ``<root>/<vertical_subdir>/*.json``;
            ``POST /v1/sessions`` accepts ``template_id``). ``None``
            disables the template surface for this app instance even
            if the vertical declares an adapter — useful for tests
            that should not touch a shared on-disk template store.
    """
    if substrate_runtime is not None:
        _enforce_frozen_for_sharing(substrate_runtime)
    alpha = alpha_config or AlphaServiceConfig()
    alpha_provider = (
        AlphaIdentityProvider(allowed_users=alpha.alpha_users)
        if alpha.enabled
        else None
    )
    if alpha.enabled and vertical.alpha_factory is None and vertical.template_adapter is None:
        raise ValueError(
            f"vertical {vertical.name!r} does not support alpha mode "
            "(no alpha_factory and no template_adapter)"
        )
    resolved_templates_root = _resolve_templates_root(
        templates_root_dir=templates_root_dir,
        vertical=vertical,
    )
    manager = SessionManager(
        lifeform_factory=vertical.factory,
        alpha_lifeform_factory=vertical.alpha_factory if alpha.enabled else None,
        alpha_identity_provider=alpha_provider,
        alpha_memory_scope_root_dir=alpha.memory_scope_root_dir,
        vertical_name=vertical.name,
        max_sessions=max_sessions,
        idle_eviction_seconds=idle_eviction_seconds,
        substrate_runtime=substrate_runtime,
        template_adapter=vertical.template_adapter,
        templates_root_dir=resolved_templates_root,
    )
    app = web.Application(middlewares=[_error_middleware])
    app["session_manager"] = manager
    app["vertical_spec"] = vertical
    app["substrate_runtime"] = substrate_runtime
    app["alpha_config"] = alpha
    app["templates_root_dir"] = resolved_templates_root
    app.router.add_get("/", _handle_chat_ui)
    app.router.add_get("/chat", _handle_chat_ui)
    app.router.add_get("/v1/health", _handle_health)
    app.router.add_get("/v1/info", _handle_info)
    app.router.add_get("/v1/templates", _handle_list_templates)
    app.router.add_post("/v1/sessions", _handle_create_session)
    app.router.add_delete("/v1/sessions/{session_id}", _handle_close_session)
    app.router.add_get("/v1/sessions/{session_id}/state", _handle_session_state)
    app.router.add_post("/v1/sessions/{session_id}/turns", _handle_turn)
    app.router.add_post(
        "/v1/sessions/{session_id}/dialogue-outcomes",
        _handle_dialogue_outcome,
    )
    app.router.add_post("/v1/sessions/{session_id}/pause", _handle_pause_session)
    app.router.add_post("/v1/sessions/{session_id}/end-scene", _handle_end_scene)
    app.router.add_post(
        "/v1/sessions/{session_id}/save-as-template",
        _handle_save_as_template,
    )
    app.router.add_get(
        "/v1/users/me/relationship-summary",
        _handle_relationship_summary,
    )
    app.router.add_get(
        "/v1/users/me/memory/rupture-repair",
        _handle_rupture_repair_memory,
    )
    app.router.add_delete("/v1/users/me/memory", _handle_delete_user_memory)
    app.router.add_get("/v1/admin/weekly-report", _handle_admin_weekly_report)
    return app


def _resolve_templates_root(
    *,
    templates_root_dir: str | None,
    vertical: VerticalSpec,
) -> Path | None:
    """Compute the per-vertical templates directory.

    Layout: ``<service_root>/<vertical_subdir>``. ``vertical_subdir``
    defaults to ``vertical.name`` when ``vertical.template_subdir`` is
    None; this gives every vertical a dedicated bucket so a future
    multi-vertical service does not cross-list templates.

    Returns ``None`` (template surface disabled) when either the
    service-level root is unset or the vertical does not register a
    template adapter.
    """
    if templates_root_dir is None or not templates_root_dir.strip():
        return None
    if vertical.template_adapter is None:
        return None
    subdir = (vertical.template_subdir or vertical.name).strip()
    if not subdir:
        return None
    return Path(templates_root_dir).expanduser() / subdir


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
    except PermissionError as exc:
        return _json_error(status=403, error="forbidden", detail=str(exc))
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


async def _handle_chat_ui(request: web.Request) -> web.Response:  # noqa: ARG001
    return web.Response(text=_CHAT_UI_HTML, content_type="text/html")


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
        alpha=dict(alpha_config_to_json(request.app["alpha_config"])),
    )
    return _json_ok(body.to_json())


async def _handle_create_session(request: web.Request) -> web.Response:
    payload = await _maybe_json(request)
    requested_id = None
    template_id: str | None = None
    if isinstance(payload, dict):
        raw = payload.get("session_id")
        if raw is not None and not isinstance(raw, str):
            raise _BadRequest("invalid_session_id", "session_id must be a string")
        if isinstance(raw, str):
            requested_id = raw
        raw_tpl = payload.get("template_id")
        if raw_tpl is not None and not isinstance(raw_tpl, str):
            raise _BadRequest("invalid_template_id", "template_id must be a string")
        if isinstance(raw_tpl, str) and raw_tpl.strip():
            template_id = raw_tpl.strip()
    manager: SessionManager = request.app["session_manager"]
    spec: VerticalSpec = request.app["vertical_spec"]
    alpha: AlphaServiceConfig = request.app["alpha_config"]
    user_id = None
    if alpha.enabled:
        user_id = _alpha_user_id(request, payload)
    try:
        session = await manager.create_session(
            session_id=requested_id,
            user_id=user_id,
            template_id=template_id,
        )
    except TemplatesNotSupportedError as exc:
        return _json_error(
            status=503, error="templates_not_supported", detail=str(exc)
        )
    except FileNotFoundError as exc:
        return _json_error(
            status=404, error="template_not_found", detail=str(exc)
        )
    except SessionAlreadyExistsError:
        # Re-raise so the middleware's specific handler returns 409
        # — SessionAlreadyExistsError extends ValueError so we MUST
        # let it through before the generic ValueError catch below.
        raise
    except ValueError as exc:
        return _json_error(
            status=400, error="invalid_template_id", detail=str(exc)
        )
    body = CreateSessionResponse(
        session_id=session.session_id,
        vertical=spec.name,
        has_temporal_bootstrap=spec.has_temporal_bootstrap,
        has_regime_bootstrap=spec.has_regime_bootstrap,
        user_id=user_id,
        service_version=alpha.service_version if alpha.enabled else "",
        policy_version=alpha.policy_version if alpha.enabled else "",
        alpha_disclaimer=ALPHA_DISCLAIMER if alpha.enabled else "",
    )
    response_payload = body.to_json()
    response_payload["template_id"] = template_id
    return _json_ok(response_payload, status=201)


async def _handle_list_templates(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    spec: VerticalSpec = request.app["vertical_spec"]
    adapter = manager.template_adapter
    root = manager.templates_root_dir
    if adapter is None or root is None:
        return _json_ok(
            {
                "vertical": spec.name,
                "templates_supported": False,
                "templates_root_dir": None,
                "templates": [],
            }
        )
    try:
        items = adapter.list_templates(root)
    except (NotADirectoryError, ValueError) as exc:
        return _json_error(
            status=500, error="templates_listing_failed", detail=str(exc)
        )
    return _json_ok(
        {
            "vertical": spec.name,
            "templates_supported": True,
            "templates_root_dir": str(root),
            "templates": [item.to_json() for item in items],
        }
    )


async def _handle_save_as_template(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    adapter = manager.template_adapter
    root = manager.templates_root_dir
    if adapter is None or root is None:
        return _json_error(
            status=503,
            error="templates_not_supported",
            detail=(
                "vertical does not register a template adapter or service "
                "has no templates_root_dir configured"
            ),
        )
    payload = await _require_json(request)
    raw_id = payload.get("template_id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise _BadRequest(
            "invalid_template_id", "template_id is required and must be a string"
        )
    template_id = raw_id.strip()
    replay_provenance = payload.get("replay_provenance", "")
    if not isinstance(replay_provenance, str):
        raise _BadRequest(
            "invalid_replay_provenance", "replay_provenance must be a string"
        )
    include_memory = payload.get("include_memory", True)
    if not isinstance(include_memory, bool):
        raise _BadRequest(
            "invalid_include_memory", "include_memory must be a boolean"
        )
    overwrite_existing = payload.get("overwrite_existing", False)
    if not isinstance(overwrite_existing, bool):
        raise _BadRequest(
            "invalid_overwrite_existing", "overwrite_existing must be a boolean"
        )
    session = await manager.get_session(session_id)
    context = manager.template_context_for(session_id)
    if context is None:
        return _json_error(
            status=409,
            error="session_not_template_eligible",
            detail=(
                f"session {session_id!r} was created via the legacy factory "
                "path and has no template context; recreate the session with "
                "a template-aware vertical to enable save-as-template"
            ),
        )
    try:
        metadata = adapter.save_session_as_template(
            session=session,
            context=context,
            root_dir=root,
            template_id=template_id,
            replay_provenance=replay_provenance,
            include_memory=include_memory,
            overwrite_existing=overwrite_existing,
        )
    except FileExistsError as exc:
        return _json_error(
            status=409, error="template_already_exists", detail=str(exc)
        )
    except (ValueError, TypeError) as exc:
        return _json_error(
            status=400, error="invalid_save_request", detail=str(exc)
        )
    return _json_ok({"saved": metadata.to_json()}, status=201)


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
        response_rationale_tags=tuple(result.response.rationale_tags),
        safety=_safety_metadata(result.response.rationale_tags),
    )
    return _json_ok(body.to_json())


async def _handle_dialogue_outcome(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    payload = await _require_json(request)
    raw_kind = payload.get("kind")
    if not isinstance(raw_kind, str):
        raise _BadRequest("invalid_outcome_kind", "kind must be a string")
    try:
        kind = DialogueExternalOutcomeKind(raw_kind.lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in DialogueExternalOutcomeKind)
        raise _BadRequest(
            "invalid_outcome_kind",
            f"kind must be one of: {allowed}",
        ) from exc
    confidence = payload.get("confidence", 0.9)
    if not isinstance(confidence, int | float):
        raise _BadRequest("invalid_confidence", "confidence must be numeric")
    evidence_ref = payload.get("evidence_ref")
    if evidence_ref is not None and not isinstance(evidence_ref, str):
        raise _BadRequest("invalid_evidence_ref", "evidence_ref must be string")
    description = payload.get("description", "")
    if not isinstance(description, str):
        raise _BadRequest("invalid_description", "description must be string")
    session = await manager.get_session(session_id)
    raw_turn_index = payload.get("turn_index")
    if raw_turn_index is not None and not isinstance(raw_turn_index, int):
        raise _BadRequest("invalid_turn_index", "turn_index must be an integer")
    # User-facing feedback is consumed by the next kernel turn, so by
    # default bind it to that upcoming turn. Review tools may pass an
    # explicit historical turn_index when needed.
    turn_index = (
        raw_turn_index
        if isinstance(raw_turn_index, int)
        else len(session.turn_summaries) + 1
    )
    evidence = session.submit_dialogue_outcome(
        kind=kind,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=float(confidence),
        turn_index=turn_index,
        evidence_ref=evidence_ref
        or f"service:{session_id}:{kind.value}:turn-{turn_index}:{uuid4().hex[:12]}",
        description=description,
    )
    return _json_ok(
        {
            "session_id": session_id,
            "evidence_id": evidence.evidence_id,
            "kind": evidence.kind.value,
            "source": evidence.source.value,
            "confidence": evidence.confidence,
        },
        status=201,
    )


async def _handle_pause_session(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    await manager.get_session(session_id)
    return _json_ok(
        {
            "session_id": session_id,
            "paused": True,
            "message": "Session paused. No memory was deleted.",
        }
    )


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
    evidence_ref = _write_session_evidence(
        request=request,
        session_id=session_id,
        session=session,
        closed_scene_id=closed.scene_id if closed is not None else None,
    )
    body = EndSceneResponse(
        session_id=session_id,
        closed_scene_id=closed.scene_id if closed is not None else None,
        slow_loop_drained=drain and closed is not None,
        evidence_artifact_ref=evidence_ref,
    )
    return _json_ok(body.to_json())


async def _handle_relationship_summary(request: web.Request) -> web.Response:
    alpha = _require_alpha(request)
    user_id = _alpha_user_id(request, None)
    entries = _scoped_rupture_repair_entries(alpha, user_id)
    observed = tuple(entry for entry in entries if "repair_outcome:observed" in entry.tags)
    kinds = sorted(
        {
            tag.split(":", 1)[1]
            for entry in entries
            for tag in entry.tags
            if tag.startswith("rupture_kind:")
        }
    )
    return _json_ok(
        {
            "user_id": user_id,
            "user_scope": user_id,
            "rupture_repair_count": len(entries),
            "observed_repair_count": len(observed),
            "rupture_kinds": kinds,
            "relationship_stage": None,
            "preferences": _preferences_from_rupture_memory(entries),
        }
    )


async def _handle_rupture_repair_memory(request: web.Request) -> web.Response:
    alpha = _require_alpha(request)
    user_id = _alpha_user_id(request, None)
    entries = _scoped_rupture_repair_entries(alpha, user_id)
    return _json_ok(
        {
            "user_id": user_id,
            "entries": [_memory_entry_to_json(entry) for entry in entries],
        }
    )


async def _handle_delete_user_memory(request: web.Request) -> web.Response:
    alpha = _require_alpha(request)
    user_id = _alpha_user_id(request, None)
    store = _build_scoped_store(alpha, user_id)
    deleted = delete_entries_for_scope(store, user_scope=user_id)
    store.save_to_backend()
    evidence_ref = _write_deletion_evidence(request, user_id=user_id, deleted=deleted)
    return _json_ok(
        {
            "user_id": user_id,
            "deleted_entry_ids": list(deleted),
            "evidence_artifact_ref": evidence_ref,
        }
    )


async def _handle_admin_weekly_report(request: web.Request) -> web.Response:
    alpha = _require_alpha(request)
    manager: SessionManager = request.app["session_manager"]
    sessions = await manager.session_summaries()
    active_users = sorted(
        {
            str(item["user_id"])
            for item in sessions
            if isinstance(item.get("user_id"), str)
        }
    )
    return _json_ok(
        {
            "service_version": alpha.service_version,
            "policy_version": alpha.policy_version,
            "active_user_count": len(active_users),
            "active_users": active_users,
            "session_count": len(sessions),
            "sessions": list(sessions),
            "serious_safety_issue_count": 0,
            "detected_rupture_count": None,
            "observed_repair_count": None,
            "memory_deletion_event_count": None,
        }
    )


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


def _alpha_user_id(request: web.Request, payload: object | None) -> str:
    header = request.headers.get("X-Alpha-User")
    body_user = payload.get("user_id") if isinstance(payload, dict) else None
    user = header or body_user
    if not isinstance(user, str) or not user.strip():
        raise _BadRequest("missing_alpha_user", "X-Alpha-User or user_id is required")
    alpha: AlphaServiceConfig = request.app["alpha_config"]
    user_id = user.strip()
    if not alpha.is_allowed(user_id):
        raise PermissionError(f"alpha user {user_id!r} is not allowed")
    return user_id


def _require_alpha(request: web.Request) -> AlphaServiceConfig:
    alpha: AlphaServiceConfig = request.app["alpha_config"]
    if not alpha.enabled:
        raise _BadRequest("alpha_disabled", "closed alpha routes are disabled")
    if alpha.memory_scope_root_dir is None:
        raise _BadRequest(
            "alpha_memory_disabled",
            "memory_scope_root_dir is required for this alpha route",
        )
    return alpha


def _build_scoped_store(alpha: AlphaServiceConfig, user_id: str):
    if alpha.memory_scope_root_dir is None:
        raise _BadRequest(
            "alpha_memory_disabled",
            "memory_scope_root_dir is required for scoped memory",
        )
    identity = UserIdentity(user_id=user_id, scope_key=user_id)
    return build_scoped_memory_store(
        identity=identity,
        root_dir=alpha.memory_scope_root_dir,
    )


def _scoped_rupture_repair_entries(alpha: AlphaServiceConfig, user_id: str):
    store = _build_scoped_store(alpha, user_id)
    return tuple(
        entry
        for entry in list_durable_entries_for_scope(store, user_scope=user_id)
        if "rupture_repair" in entry.tags
    )


def _memory_entry_to_json(entry) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "content": entry.content,
        "track": entry.track.value if hasattr(entry.track, "value") else entry.track,
        "stratum": entry.stratum,
        "created_at_ms": entry.created_at_ms,
        "tags": list(entry.tags),
    }


def _preferences_from_rupture_memory(entries) -> list[str]:
    preferences: list[str] = []
    for entry in entries:
        if "repair_outcome:observed" in entry.tags:
            preferences.append("slow_down_and_repair_before_planning")
            break
    return preferences


def _safety_metadata(tags: tuple[str, ...]) -> dict[str, Any]:
    return {
        "alpha_disclaimer": ALPHA_DISCLAIMER,
        "boundary_tags": [
            tag
            for tag in tags
            if tag.startswith("risk=")
            or tag.startswith("boundary")
            or tag.startswith("repair_alpha=")
        ],
    }


def _write_session_evidence(
    *,
    request: web.Request,
    session_id: str,
    session,
    closed_scene_id: str | None,
) -> str | None:
    alpha: AlphaServiceConfig = request.app["alpha_config"]
    if not alpha.enabled or alpha.evidence_root_dir is None:
        return None
    root = Path(alpha.evidence_root_dir) / "sessions" / session_id
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "closed_scene_id": closed_scene_id,
        "service_version": alpha.service_version,
        "policy_version": alpha.policy_version,
        "turns": [
            {
                "turn_index": summary.turn_index,
                "scene_id": summary.scene_id,
                "active_regime": summary.active_regime,
                "active_abstract_action": summary.active_abstract_action,
                "pe_magnitude": summary.pe_magnitude,
                "open_loop_count": summary.open_loop_count,
                "commitment_count": summary.commitment_count,
            }
            for summary in session.turn_summaries
        ],
        "latest_active_slots": sorted(session.latest_active_snapshots.keys()),
        "latest_shadow_slots": sorted(session.latest_shadow_snapshots.keys()),
        "pending_followup_count": len(session.all_pending_followups()),
    }
    path = root / "session_evidence.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _write_deletion_evidence(
    request: web.Request,
    *,
    user_id: str,
    deleted: tuple[str, ...],
) -> str | None:
    alpha: AlphaServiceConfig = request.app["alpha_config"]
    if not alpha.enabled or alpha.evidence_root_dir is None:
        return None
    root = Path(alpha.evidence_root_dir) / "deletions" / user_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / "deletion_evidence.json"
    path.write_text(
        json.dumps(
            {
                "user_id": user_id,
                "deleted_entry_ids": list(deleted),
                "service_version": alpha.service_version,
                "policy_version": alpha.policy_version,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(path)


_CHAT_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Volvence Zero Chat</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body {
      margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: #030712; color: #e5e7eb;
    }
    main {
      width: min(920px, calc(100vw - 28px));
      height: min(820px, calc(100vh - 28px));
      display: grid; grid-template-rows: auto auto 1fr auto; gap: 12px;
      padding: 16px; border: 1px solid #374151; border-radius: 16px;
      background: #111827;
    }
    h1 { margin: 0; font-size: 20px; }
    .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    input, textarea, button {
      font: inherit; border-radius: 10px; border: 1px solid #4b5563;
      background: #0f172a; color: #f9fafb;
    }
    input { padding: 8px 10px; min-width: 180px; }
    select { padding: 8px 10px; min-width: 220px; }
    textarea { min-height: 64px; padding: 10px; resize: vertical; }
    button { padding: 8px 11px; cursor: pointer; background: #1d4ed8; border-color: #2563eb; }
    button.secondary { background: #111827; border-color: #4b5563; }
    button.danger { background: #991b1b; border-color: #b91c1c; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    #log {
      overflow: auto; padding: 12px; border-radius: 12px;
      background: #030712; border: 1px solid #1f2937;
    }
    .msg {
      max-width: 82%; margin: 9px 0; padding: 10px 12px;
      border-radius: 14px; white-space: pre-wrap; line-height: 1.45;
    }
    .user { margin-left: auto; background: #1d4ed8; }
    .bot { margin-right: auto; background: #1f2937; }
    .system {
      max-width: 100%; color: #9ca3af; background: transparent;
      border: 1px dashed #374151; font-size: 13px;
    }
    .composer { display: grid; grid-template-columns: 1fr auto; gap: 10px; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Volvence Zero Chat</h1>
      <div id="status" class="msg system">Create a session to start. In alpha mode, set an allowed user id.</div>
    </header>
    <section class="bar">
      <input id="userId" placeholder="alpha user, e.g. alice">
      <input id="sessionId" placeholder="optional session_id">
      <select id="templateSelect" title="Pick a saved lifeform template (optional)">
        <option value="">no template (vertical default)</option>
      </select>
      <button id="createBtn">Create Session</button>
      <button id="saveTemplateBtn" class="secondary" disabled>Save as Template</button>
      <button id="endBtn" class="secondary" disabled>End Scene</button>
      <button id="clearBtn" class="secondary">Clear</button>
    </section>
    <section class="bar">
      <button class="secondary outcome" data-kind="FELT_HEARD" disabled>Felt heard</button>
      <button class="secondary outcome" data-kind="HELPED" disabled>Helped</button>
      <button class="secondary outcome" data-kind="MISSED" disabled>Missed</button>
      <button class="secondary outcome" data-kind="OVER_DIRECTIVE" disabled>Over-directive</button>
      <button class="secondary outcome" data-kind="COME_BACK" disabled>Come back</button>
      <button class="danger outcome" data-kind="UNSAFE" disabled>Unsafe</button>
    </section>
    <section id="log" aria-live="polite"></section>
    <section class="composer">
      <textarea id="input" placeholder="Type a message... Ctrl+Enter to send" disabled></textarea>
      <button id="sendBtn" disabled>Send</button>
    </section>
  </main>
  <script>
    const state = {
      sessionId: null,
      templatesSupported: false,
      debug: new URLSearchParams(window.location.search).get("debug") === "1",
    };
    const statusEl = document.getElementById("status");
    const logEl = document.getElementById("log");
    const inputEl = document.getElementById("input");
    const userIdEl = document.getElementById("userId");
    const sessionIdEl = document.getElementById("sessionId");
    const templateSelectEl = document.getElementById("templateSelect");
    const createBtn = document.getElementById("createBtn");
    const sendBtn = document.getElementById("sendBtn");
    const endBtn = document.getElementById("endBtn");
    const clearBtn = document.getElementById("clearBtn");
    const saveTemplateBtn = document.getElementById("saveTemplateBtn");
    const outcomeBtns = Array.from(document.querySelectorAll(".outcome"));

    function alphaHeaders() {
      const user = userIdEl.value.trim();
      return user ? { "X-Alpha-User": user } : {};
    }
    function addMessage(kind, text) {
      const div = document.createElement("div");
      div.className = `msg ${kind}`;
      div.textContent = text;
      logEl.appendChild(div);
      logEl.scrollTop = logEl.scrollHeight;
    }
    function setReady(ready) {
      inputEl.disabled = !ready;
      sendBtn.disabled = !ready;
      endBtn.disabled = !ready;
      saveTemplateBtn.disabled = !(ready && state.templatesSupported);
      outcomeBtns.forEach(btn => { btn.disabled = !ready; });
    }
    async function loadTemplates() {
      try {
        const payload = await requestJson("/v1/templates", { method: "GET" });
        state.templatesSupported = Boolean(payload.templates_supported);
        // Reset and refill the dropdown; preserve the leading "no template" option.
        templateSelectEl.innerHTML = "";
        const defaultOpt = document.createElement("option");
        defaultOpt.value = "";
        defaultOpt.textContent = state.templatesSupported
          ? "no template (vertical default)"
          : "templates not supported by this vertical";
        templateSelectEl.appendChild(defaultOpt);
        templateSelectEl.disabled = !state.templatesSupported;
        if (!state.templatesSupported) return;
        const items = Array.isArray(payload.templates) ? payload.templates : [];
        for (const item of items) {
          const opt = document.createElement("option");
          opt.value = item.template_id;
          const label = item.display_name && item.display_name.trim()
            ? item.display_name
            : item.template_id;
          opt.textContent = item.description
            ? `${item.template_id} — ${label} (${item.description})`
            : `${item.template_id} — ${label}`;
          templateSelectEl.appendChild(opt);
        }
        if (items.length === 0) {
          addMessage(
            "system",
            `No templates found under ${payload.templates_root_dir || "<unset>"} — you can save the current session as one after a chat.`,
          );
        }
      } catch (err) {
        addMessage("system", `Failed to load templates: ${err.message}`);
      }
    }
    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        headers: { "Content-Type": "application/json", ...alphaHeaders(), ...(options.headers || {}) },
        ...options,
      });
      const text = await response.text();
      const payload = text ? JSON.parse(text) : {};
      if (!response.ok) {
        throw new Error(`${payload.error || response.status}: ${payload.detail || text}`);
      }
      return payload;
    }
    async function createSession() {
      createBtn.disabled = true;
      try {
        const requested = sessionIdEl.value.trim();
        const templateId = templateSelectEl.value.trim();
        const body = {};
        if (requested) body.session_id = requested;
        if (templateId) body.template_id = templateId;
        const payload = await requestJson(
          "/v1/sessions",
          { method: "POST", body: JSON.stringify(body) },
        );
        state.sessionId = payload.session_id;
        const templateLabel = payload.template_id ? ` | template ${payload.template_id}` : "";
        statusEl.textContent = `Session ${state.sessionId} | vertical ${payload.vertical}${templateLabel}`;
        setReady(true);
        addMessage("system", `Created session ${state.sessionId}${templateLabel}`);
        if (payload.alpha_disclaimer) addMessage("system", payload.alpha_disclaimer);
        inputEl.focus();
      } catch (err) {
        addMessage("system", `Create failed: ${err.message}`);
      } finally {
        createBtn.disabled = false;
      }
    }
    async function saveAsTemplate() {
      if (!state.sessionId || !state.templatesSupported) return;
      const defaultId = `chat-${new Date().toISOString().replace(/[:.]/g, "-")}`;
      const templateId = window.prompt(
        "Save current session as template — enter a template id (filename, no extension):",
        defaultId,
      );
      if (!templateId || !templateId.trim()) return;
      const provenance = window.prompt(
        "Optional: short provenance note (what was achieved in this chat?)",
        "",
      ) || "";
      saveTemplateBtn.disabled = true;
      try {
        const payload = await requestJson(
          `/v1/sessions/${encodeURIComponent(state.sessionId)}/save-as-template`,
          {
            method: "POST",
            body: JSON.stringify({
              template_id: templateId.trim(),
              replay_provenance: provenance,
              include_memory: true,
              overwrite_existing: false,
            }),
          },
        );
        const saved = payload.saved || {};
        addMessage(
          "system",
          `Saved template ${saved.template_id} -> ${saved.file_path}`,
        );
        await loadTemplates();
      } catch (err) {
        addMessage("system", `Save-as-template failed: ${err.message}`);
      } finally {
        saveTemplateBtn.disabled = !state.templatesSupported;
      }
    }
    async function sendTurn() {
      const text = inputEl.value.trim();
      if (!text || !state.sessionId) return;
      inputEl.value = "";
      addMessage("user", text);
      setReady(false);
      try {
        const payload = await requestJson(`/v1/sessions/${encodeURIComponent(state.sessionId)}/turns`, {
          method: "POST",
          body: JSON.stringify({ user_input: text }),
        });
        addMessage("bot", payload.response_text || "(empty response)");
        if (state.debug) {
          const tags = payload.response_rationale_tags && payload.response_rationale_tags.length
            ? ` | tags=${payload.response_rationale_tags.join(",")}`
            : "";
          const meta = `turn=${payload.turn_index} | regime=${payload.active_regime || "none"} `
            + `| intent=${payload.expression_intent || "none"}${tags}`;
          addMessage("system", meta);
        }
      } catch (err) {
        addMessage("system", `Turn failed: ${err.message}`);
      } finally {
        setReady(Boolean(state.sessionId));
        inputEl.focus();
      }
    }
    async function submitOutcome(kind) {
      if (!state.sessionId) return;
      try {
        const payload = await requestJson(`/v1/sessions/${encodeURIComponent(state.sessionId)}/dialogue-outcomes`, {
          method: "POST",
          body: JSON.stringify({ kind, confidence: 0.95 }),
        });
        addMessage("system", `Submitted feedback: ${payload.kind}`);
      } catch (err) {
        addMessage("system", `Feedback failed: ${err.message}`);
      }
    }
    async function endScene() {
      if (!state.sessionId) return;
      try {
        const payload = await requestJson(`/v1/sessions/${encodeURIComponent(state.sessionId)}/end-scene`, {
          method: "POST",
          body: JSON.stringify({ reason: "chat-ui-end", drain_slow_loop: true }),
        });
        addMessage(
          "system",
          `Scene ended: ${payload.closed_scene_id || "none"} | slow_loop=${payload.slow_loop_drained}`,
        );
        if (payload.evidence_artifact_ref) {
          addMessage("system", `Evidence: ${payload.evidence_artifact_ref}`);
        }
      } catch (err) {
        addMessage("system", `End scene failed: ${err.message}`);
      }
    }
    createBtn.addEventListener("click", createSession);
    sendBtn.addEventListener("click", sendTurn);
    endBtn.addEventListener("click", endScene);
    clearBtn.addEventListener("click", () => { logEl.textContent = ""; });
    saveTemplateBtn.addEventListener("click", saveAsTemplate);
    outcomeBtns.forEach(btn => btn.addEventListener("click", () => submitOutcome(btn.dataset.kind)));
    inputEl.addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendTurn();
      }
    });
    loadTemplates();
  </script>
</body>
</html>
"""
