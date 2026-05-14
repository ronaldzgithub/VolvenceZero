"""aiohttp ops routes (Slice 5.1 + 5.2 + 5.3).

Wired by :func:`attach_ops_routes`:

* ``GET  /dlaas/admin/ops/conversations`` (filtered list)
* ``GET  /dlaas/admin/ops/conversations/overview`` (count summary)
* ``GET  /dlaas/admin/ops/conversations/{session_id}``
* ``POST /dlaas/admin/ops/conversations/{session_id}/pause``
* ``POST /dlaas/admin/ops/conversations/{session_id}/resume``
* ``POST /dlaas/admin/ops/conversations/{session_id}/operator-message``
* ``POST /dlaas/admin/ops/conversations/{session_id}/escalate-handoff``
* ``GET  /dlaas/admin/ops/conversations/stream`` (SSE)
* ``GET  /dlaas/instances/{ai_id}/handoff_queue``
* ``POST /dlaas/instances/{ai_id}/handoff_tickets``
* ``POST /dlaas/instances/{ai_id}/handoff/{ticket_id}/human_reply``

All admin endpoints accept either ``X-Control-Plane-Secret`` or
``X-Service-Secret``. Tenant-facing handoff endpoints under
``/dlaas/instances/{ai_id}`` require tenant credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import HandoffStatus
from dlaas_platform_launcher import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceManager,
    InstanceNotFound,
)
from dlaas_platform_registry import (
    HandoffTicketNotFound,
    HandoffTicketStore,
    REGISTRY_APP_KEY,
    Registry,
    require_control_plane_or_service,
    require_tenant_auth,
)

from dlaas_platform_ops.handoff_trigger import HandoffDecision, evaluate_session
from dlaas_platform_ops.ledger import LedgerBroker
from dlaas_platform_ops.pause_state import PauseStore

_LOG = logging.getLogger("dlaas_platform_ops")

OPS_BUNDLE_APP_KEY = "dlaas_ops_bundle"


class OpsBundle:
    """Container the api wheel reads to dispatch ops state."""

    __slots__ = ("pause_store", "ledger", "tickets")

    def __init__(self, *, registry: Registry) -> None:
        self.pause_store = PauseStore()
        self.ledger = LedgerBroker()
        self.tickets = HandoffTicketStore(registry)


def attach_ops_routes(
    app: web.Application, *, registry: Registry
) -> web.Application:
    """Register every ops route on the given aiohttp app."""
    if REGISTRY_APP_KEY not in app:
        raise ValueError(
            "attach_ops_routes requires app[REGISTRY_APP_KEY] "
            "(dlaas_platform_api.build_dlaas_app handles this)."
        )
    app[OPS_BUNDLE_APP_KEY] = OpsBundle(registry=registry)
    R = app.router

    R.add_get("/dlaas/admin/ops/conversations", _handle_list_conversations)
    R.add_get(
        "/dlaas/admin/ops/conversations/overview", _handle_conversations_overview
    )
    R.add_get(
        "/dlaas/admin/ops/conversations/stream", _handle_conversations_stream
    )
    R.add_get(
        "/dlaas/admin/ops/conversations/{session_id}", _handle_get_conversation
    )
    R.add_post(
        "/dlaas/admin/ops/conversations/{session_id}/pause",
        _handle_pause,
    )
    R.add_post(
        "/dlaas/admin/ops/conversations/{session_id}/resume",
        _handle_resume,
    )
    R.add_post(
        "/dlaas/admin/ops/conversations/{session_id}/operator-message",
        _handle_operator_message,
    )
    R.add_post(
        "/dlaas/admin/ops/conversations/{session_id}/escalate-handoff",
        _handle_escalate_handoff,
    )

    R.add_get(
        "/dlaas/instances/{ai_id}/handoff_queue", _handle_list_handoff_queue
    )
    R.add_post(
        "/dlaas/instances/{ai_id}/handoff_tickets", _handle_create_handoff_ticket
    )
    R.add_post(
        "/dlaas/instances/{ai_id}/handoff/{ticket_id}/human_reply",
        _handle_handoff_human_reply,
    )
    return app


# ---------------------------------------------------------------------------
# Admin: conversations
# ---------------------------------------------------------------------------


async def _handle_list_conversations(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    ai_id = request.query.get("ai_id")
    paused_raw = request.query.get("paused")
    paused: bool | None = None
    if paused_raw is not None:
        paused = paused_raw.lower() in ("1", "true", "yes")
    convos = await bundle.pause_store.overview(ai_id=ai_id, paused=paused)
    limit = _clamp_int(request.query.get("limit"), 1, 500, default=50)
    return web.json_response(
        {"status": "ok", "conversations": list(convos[:limit])}
    )


async def _handle_conversations_overview(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    convos = await bundle.pause_store.overview()
    paused = sum(1 for c in convos if c["paused"])
    return web.json_response(
        {
            "status": "ok",
            "total": len(convos),
            "paused": paused,
            "live": len(convos) - paused,
        }
    )


async def _handle_get_conversation(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    session_id = request.match_info["session_id"]
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    matches = [
        entry
        for entry in await bundle.pause_store.overview()
        if entry["session_id"] == session_id
    ]
    if not matches:
        return _error(404, "session_not_found", session_id)
    return web.json_response({"status": "ok", "conversations": matches})


async def _handle_pause(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    session_id = request.match_info["session_id"]
    data = await _read_json(request, allow_empty=True)
    operator_id = str(data.get("operator_id", "") or "")
    note = str(data.get("note", "") or "")
    ai_id = str(data.get("ai_id", "") or "")
    if not ai_id:
        return _error(
            400, "missing_ai_id", "pause body must include ai_id"
        )
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    state = await bundle.pause_store.pause(
        ai_id=ai_id,
        session_id=session_id,
        operator_id=operator_id,
        note=note,
    )
    await bundle.ledger.publish(
        event_type="pause",
        payload={
            "ai_id": ai_id,
            "session_id": session_id,
            "paused": True,
            "operator_id": operator_id,
            "note": note,
        },
    )
    return web.json_response(
        {
            "status": "ok",
            "ai_id": ai_id,
            "session_id": session_id,
            "paused": state.paused,
            "operator_id": state.pause_operator_id,
            "note": state.pause_note,
        }
    )


async def _handle_resume(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    session_id = request.match_info["session_id"]
    data = await _read_json(request, allow_empty=True)
    operator_id = str(data.get("operator_id", "") or "")
    note = str(data.get("note", "") or "")
    ai_id = str(data.get("ai_id", "") or "")
    if not ai_id:
        return _error(400, "missing_ai_id", "resume body must include ai_id")
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    prev = await bundle.pause_store.resume(
        ai_id=ai_id,
        session_id=session_id,
        operator_id=operator_id,
        note=note,
    )
    if prev is None:
        return _error(409, "not_paused", "session was not paused")
    await bundle.ledger.publish(
        event_type="resume",
        payload={
            "ai_id": ai_id,
            "session_id": session_id,
            "paused": False,
            "operator_id": operator_id,
            "note": note,
        },
    )
    return web.json_response(
        {
            "status": "ok",
            "ai_id": ai_id,
            "session_id": session_id,
            "paused": False,
            "operator_id": operator_id,
            "note": note,
        }
    )


async def _handle_operator_message(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    session_id = request.match_info["session_id"]
    data = await _read_json(request)
    operator_id = str(data.get("operator_id", "") or "")
    text = str(data.get("text", "") or "")
    inject = bool(data.get("inject_into_runtime", False))
    ai_id = str(data.get("ai_id", "") or "")
    if not ai_id:
        return _error(400, "missing_ai_id", "operator-message body must include ai_id")
    if not text.strip():
        return _error(400, "missing_text", "text is required")
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    msg = await bundle.pause_store.append_operator_message(
        ai_id=ai_id,
        session_id=session_id,
        operator_id=operator_id,
        text=text,
        inject_into_runtime=inject,
    )
    if inject:
        # Inject as an apprentice teach-style turn so the kernel
        # learns the operator's correction without contaminating
        # the user-input PE pathway. Slice 7 will swap this for a
        # typed kernel call once the apprentice ledger is wired.
        instance_manager: InstanceManager | None = request.app.get(
            INSTANCE_MANAGER_APP_KEY
        )
        if isinstance(instance_manager, InstanceManager):
            try:
                manager = instance_manager.get(ai_id)
            except InstanceNotFound:
                manager = None
            if manager is not None:
                from lifeform_core.types import TurnTriggerKind

                try:
                    session = await manager.get_session(session_id)
                except LookupError:
                    # ``SessionNotFoundError`` subclasses ``LookupError``;
                    # treat "session does not exist" as a soft skip
                    # (operator-message injection is best-effort), but
                    # let real failures (network / OOM / contract
                    # violations) surface.
                    _LOG.warning(
                        "operator-message: session %s/%s not found, "
                        "skipping injection",
                        ai_id,
                        session_id,
                    )
                    session = None
                if session is not None:
                    try:
                        await session.run_turn(
                            text, trigger_kind=TurnTriggerKind.APPRENTICE
                        )
                    except (RuntimeError, ValueError) as exc:
                        _LOG.warning(
                            "operator-message injection failed for %s/%s: %s",
                            ai_id,
                            session_id,
                            exc,
                        )
    await bundle.ledger.publish(
        event_type="operator_message",
        payload={
            "ai_id": ai_id,
            "session_id": session_id,
            "operator_id": operator_id,
            "text": text,
            "inject_into_runtime": inject,
        },
    )
    return web.json_response({"status": "ok", "message": msg.to_json()})


async def _handle_escalate_handoff(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    session_id = request.match_info["session_id"]
    data = await _read_json(request, allow_empty=True)
    ai_id = str(data.get("ai_id", "") or "")
    if not ai_id:
        return _error(400, "missing_ai_id", "escalate body must include ai_id")
    contract_id = str(data.get("contract_id", "") or "")
    end_user_ref = str(data.get("end_user_ref", "") or "")
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    instance_manager: InstanceManager | None = request.app.get(
        INSTANCE_MANAGER_APP_KEY
    )
    decision: HandoffDecision = HandoffDecision(
        should_escalate=True,
        rupture_kind="manual",
        trigger_reason="manual_escalation",
        trigger_details={"operator_initiated": True, **data},
    )
    if isinstance(instance_manager, InstanceManager):
        try:
            manager = instance_manager.get(ai_id)
            session = await manager.get_session(session_id)
            decision = evaluate_session(
                session=session,
                recent_response_ids=tuple(
                    str(r) for r in (data.get("recent_response_ids") or ())
                ),
                confidence_aggregate=data.get("confidence_aggregate"),
            )
        except (InstanceNotFound, Exception):  # noqa: BLE001 - best effort
            pass
    ticket = await bundle.tickets.create(
        ai_id=ai_id,
        contract_id=contract_id,
        end_user_ref=end_user_ref,
        session_id=session_id,
        trigger_reason=decision.trigger_reason or "manual_escalation",
        trigger_details=decision.trigger_details,
        confidence_aggregate=float(data.get("confidence_aggregate", 0.0) or 0.0),
        recent_response_ids=tuple(
            str(r) for r in (data.get("recent_response_ids") or ())
        ),
    )
    await bundle.ledger.publish(
        event_type="handoff_open",
        payload={"ticket_id": ticket.ticket_id, **ticket.to_json()},
    )
    return web.json_response({"status": "ok", **ticket.to_json()})


# ---------------------------------------------------------------------------
# Admin SSE
# ---------------------------------------------------------------------------


async def _handle_conversations_stream(request: web.Request) -> web.StreamResponse:
    require_control_plane_or_service(request)
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)
    await response.write(b": connected\n\n")
    sub = await bundle.ledger.subscribe()
    heartbeat_interval = 15.0
    try:
        last_heartbeat = asyncio.get_event_loop().time()
        while True:
            try:
                evt = await asyncio.wait_for(
                    sub.__anext__(), timeout=heartbeat_interval
                )
            except asyncio.TimeoutError:
                await response.write(b": ping\n\n")
                last_heartbeat = asyncio.get_event_loop().time()
                continue
            line = (
                f"event: {evt.event_type}\n"
                f"data: {json.dumps(evt.to_json(), ensure_ascii=False)}\n\n"
            )
            await response.write(line.encode("utf-8"))
            now = asyncio.get_event_loop().time()
            if now - last_heartbeat > heartbeat_interval:
                await response.write(b": ping\n\n")
                last_heartbeat = now
    except (asyncio.CancelledError, ConnectionResetError):
        return response
    finally:
        await sub.aclose()
    return response


# ---------------------------------------------------------------------------
# Tenant-facing handoff endpoints
# ---------------------------------------------------------------------------


async def _handle_list_handoff_queue(request: web.Request) -> web.Response:
    await require_tenant_auth(request)
    ai_id = request.match_info["ai_id"]
    status_raw = request.query.get("status", "open")
    try:
        status = HandoffStatus(status_raw.lower())
    except ValueError:
        allowed = ", ".join(s.value for s in HandoffStatus)
        return _error(400, "invalid_status", f"status must be one of: {allowed}")
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    tickets = await bundle.tickets.list_for_ai(ai_id=ai_id, status=status)
    return web.json_response(
        {
            "status": "ok",
            "ai_id": ai_id,
            "tickets": [t.to_json() for t in tickets],
        }
    )


async def _handle_create_handoff_ticket(request: web.Request) -> web.Response:
    await require_tenant_auth(request)
    ai_id = request.match_info["ai_id"]
    data = await _read_json(request)
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    ticket = await bundle.tickets.create(
        ai_id=ai_id,
        contract_id=str(data.get("contract_id", "") or ""),
        end_user_ref=str(data.get("end_user_ref", "") or ""),
        session_id=str(data.get("session_id", "") or ""),
        trigger_reason=str(data.get("trigger_reason", "") or ""),
        trigger_details=data.get("trigger_details") or {},
        confidence_aggregate=float(data.get("confidence_aggregate", 0.0) or 0.0),
        recent_response_ids=tuple(
            str(r) for r in (data.get("recent_response_ids") or ())
        ),
    )
    await bundle.ledger.publish(
        event_type="handoff_open",
        payload={"ticket_id": ticket.ticket_id, **ticket.to_json()},
    )
    return web.json_response({"status": "ok", **ticket.to_json()})


async def _handle_handoff_human_reply(request: web.Request) -> web.Response:
    await require_tenant_auth(request)
    ticket_id = request.match_info["ticket_id"]
    data = await _read_json(request)
    bundle: OpsBundle = request.app[OPS_BUNDLE_APP_KEY]
    try:
        ticket = await bundle.tickets.submit_human_reply(
            ticket_id=ticket_id,
            operator_id=str(data.get("operator_id", "") or ""),
            human_reply=str(data.get("human_reply", "") or ""),
            resolution_notes=str(data.get("resolution_notes", "") or ""),
        )
    except HandoffTicketNotFound:
        return _error(404, "ticket_not_found", ticket_id)
    await bundle.ledger.publish(
        event_type="handoff_resolved",
        payload={"ticket_id": ticket.ticket_id, **ticket.to_json()},
    )
    return web.json_response({"status": "ok", **ticket.to_json()})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


def _clamp_int(
    raw: str | None, low: int, high: int, *, default: int
) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


async def _read_json(
    request: web.Request, *, allow_empty: bool = False
) -> Mapping[str, Any]:
    if not request.body_exists:
        if allow_empty:
            return {}
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "missing_body",
                    "detail": "Body required",
                }
            ),
            content_type="application/json",
        )
    try:
        text = await request.text()
    except (web.HTTPException, OSError):
        return {}
    if not text.strip():
        if allow_empty:
            return {}
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "missing_body",
                    "detail": "Empty body",
                }
            ),
            content_type="application/json",
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "invalid_json",
                    "detail": f"Body is not valid JSON: {exc}",
                }
            ),
            content_type="application/json",
        ) from exc
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "invalid_envelope",
                    "detail": "Top-level body must be a JSON object",
                }
            ),
            content_type="application/json",
        )
    return data


__all__ = [
    "OPS_BUNDLE_APP_KEY",
    "OpsBundle",
    "attach_ops_routes",
]
