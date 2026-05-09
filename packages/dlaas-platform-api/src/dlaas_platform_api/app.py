"""DLaaS aiohttp router and dispatch handler.

This module owns the ``/dlaas/instances/{ai_id}/interactions`` route.
It does NOT itself hold any cognitive state. The dispatch flow is:

1. Parse the JSON body into a typed :class:`InteractionEnvelope`.
2. Look up the target ``LifeformSession`` via ``lifeform-service``'s
   ``SessionManager`` (Slice 1 uses the existing single-instance
   manager; Slice 3 will introduce a per-``ai_id`` ``InstanceManager``
   as a router in front of multiple session managers).
3. Switch on :class:`InteractionType` and call the matching kernel
   entry point (``run_turn`` for chat; later slices add the rest).
4. Wrap the kernel response into one or more :class:`OutputAct`
   objects per the DLaaS wire format.

Slice 1 supports only ``InteractionType.CHAT``. Other types return
``501 not_implemented`` with a clear pointer to which later slice will
implement them. The router is deliberately registered alongside the
existing ``/v1/sessions/...`` routes (NOT replacing them), so the
service remains backwards compatible until the platform tier is fully
ACTIVE.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from aiohttp import web

from dlaas_platform_contracts import (
    DEFAULT_PROTOCOL_VERSION,
    InteractionEnvelope,
    InteractionType,
    OutputAct,
)
from lifeform_service.app import create_app as create_lifeform_app
from lifeform_service.session_manager import (
    SessionAlreadyExistsError,
    SessionManager,
    SessionNotFoundError,
)

_LOG = logging.getLogger("dlaas_platform_api")

DLAAS_APP_AI_ID_KEY = "dlaas_default_ai_id"
"""App key under which the Slice 1 hardcoded ``ai_id`` is stored.

Slice 3 replaces this with a real ``InstanceManager`` indexed by
``ai_id``; until then we accept any path ``ai_id`` and serve it from a
single shared ``SessionManager``.
"""


def attach_dlaas_routes(
    app: web.Application,
    *,
    default_ai_id: str = "ai_default",
) -> web.Application:
    """Register ``/dlaas/*`` routes on an existing aiohttp app.

    The app MUST already have a ``session_manager`` set up by
    ``lifeform_service.app.create_app`` — we read the existing manager
    rather than building a parallel one, so the kernel only sees one
    ``Lifeform`` per process during Slice 1.
    """
    if "session_manager" not in app:
        raise ValueError(
            "attach_dlaas_routes requires an aiohttp app produced by "
            "lifeform_service.app.create_app (session_manager missing)."
        )
    app[DLAAS_APP_AI_ID_KEY] = default_ai_id
    app.router.add_post(
        "/dlaas/instances/{ai_id}/interactions",
        _handle_interaction,
    )
    return app


def build_dlaas_app(*, default_ai_id: str = "ai_default", **service_kwargs: Any) -> web.Application:
    """Build a lifeform-service app with DLaaS routes already attached.

    Equivalent to:

        app = lifeform_service.app.create_app(**service_kwargs)
        attach_dlaas_routes(app, default_ai_id=...)

    Used by the ``dlaas-serve`` CLI in Slice 3; in Slice 1 it lets
    callers spin up the full surface in one call for smoke testing.
    """
    app = create_lifeform_app(**service_kwargs)
    attach_dlaas_routes(app, default_ai_id=default_ai_id)
    return app


async def _handle_interaction(request: web.Request) -> web.Response:
    """Dispatch a typed ``InteractionEnvelope`` to the kernel.

    Slice 1: only ``InteractionType.CHAT`` is wired. Other types return
    ``501 not_implemented`` with the slice name. The handler does NOT
    inspect ``human_brief`` to guess the type — interaction_type is the
    sole dispatch key.
    """
    try:
        envelope = await _parse_envelope(request)
    except _EnvelopeError as exc:
        return _json_error(status=400, error=exc.code, detail=exc.detail)

    ai_id = request.match_info.get("ai_id", "")
    if not ai_id:
        return _json_error(
            status=400,
            error="invalid_ai_id",
            detail="ai_id path segment is required",
        )

    manager: SessionManager = request.app["session_manager"]
    try:
        session = await _get_or_create_session(manager, envelope.session_id)
    except SessionAlreadyExistsError as exc:  # pragma: no cover - racy, defensive
        return _json_error(
            status=409,
            error="session_already_exists",
            detail=str(exc),
        )
    except SessionNotFoundError as exc:  # pragma: no cover - get-or-create above
        return _json_error(
            status=404,
            error="session_not_found",
            detail=str(exc),
        )

    if envelope.interaction_type is InteractionType.CHAT:
        return await _dispatch_chat(envelope=envelope, session=session, ai_id=ai_id)

    # Slice mapping for the remaining types (the responses make the
    # implementation roadmap visible to integrators while the wheels
    # are being built):
    later_slice = {
        InteractionType.FEEDBACK: "Slice 2.1",
        InteractionType.OBSERVE: "Slice 2.2",
        InteractionType.TEACH: "Slice 2.3",
        InteractionType.TASK: "Slice 2.3",
        InteractionType.REPORT: "Slice 2.4",
        InteractionType.COMMAND: "Slice 2.4",
    }[envelope.interaction_type]
    return _json_error(
        status=501,
        error="not_implemented",
        detail=(
            f"interaction_type={envelope.interaction_type.value!r} "
            f"is wired in {later_slice}; only 'chat' is live in Slice 1."
        ),
    )


async def _parse_envelope(request: web.Request) -> InteractionEnvelope:
    if not request.body_exists:
        raise _EnvelopeError("invalid_envelope", "Request body is required")
    try:
        text = await request.text()
    except Exception as exc:
        raise _EnvelopeError("invalid_body", f"Could not read body: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _EnvelopeError("invalid_json", f"Body is not valid JSON: {exc}") from exc
    try:
        return InteractionEnvelope.from_json(data)
    except ValueError as exc:
        raise _EnvelopeError("invalid_envelope", str(exc)) from exc


async def _get_or_create_session(manager: SessionManager, session_id: str):
    """Reuse an existing session if present; otherwise create with that id.

    DLaaS interactions are session-scoped from the client side — the
    integrator picks ``session_id`` (e.g. ``sess_math_20260505_001``).
    The platform reuses the session across turns and creates it lazily
    on first contact so the integrator does not have to call a separate
    bootstrap endpoint.
    """
    try:
        return await manager.get_session(session_id)
    except SessionNotFoundError:
        return await manager.create_session(session_id=session_id)


async def _dispatch_chat(
    *,
    envelope: InteractionEnvelope,
    session: Any,
    ai_id: str,
) -> web.Response:
    if not envelope.human_brief.strip():
        return _json_error(
            status=400,
            error="invalid_human_brief",
            detail="interaction_type=chat requires a non-empty human_brief",
        )
    result = await session.run_turn(envelope.human_brief)
    response_text = getattr(result.response, "text", "") or ""
    rationale_tags = tuple(getattr(result.response, "rationale_tags", ()) or ())

    primary_act = OutputAct(
        act_type="text",
        capability="text_streaming",
        payload={"content": response_text},
        degraded=False,
        original_capability="",
    )
    body: dict[str, Any] = {
        "status": "ok",
        "ai_id": ai_id,
        "contract_id": envelope.contract_id,
        "session_id": envelope.session_id,
        "response_id": f"resp_{uuid4().hex[:12]}",
        "protocol_version": envelope.protocol_version or DEFAULT_PROTOCOL_VERSION,
        "interaction_type": envelope.interaction_type.value,
        "output_acts": [primary_act.to_json()],
        "active_regime": getattr(result, "active_regime", None),
        "active_abstract_action": getattr(result, "active_abstract_action", None),
        "rationale_tags": list(rationale_tags),
    }
    return web.json_response(body)


def _json_error(
    *,
    status: int,
    error: str,
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> web.Response:
    payload: dict[str, Any] = {"status": "error", "error": error, "detail": detail}
    if extra:
        payload.update(extra)
    return web.json_response(payload, status=status)


class _EnvelopeError(Exception):
    """Raised by ``_parse_envelope`` for any 400-level parse failure.

    Carries an ``code`` slug (e.g. ``invalid_envelope``) and a
    human-readable ``detail`` string, matching the existing
    ``lifeform-service`` error format.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
