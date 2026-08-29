"""Thin HTTP projection for the Foundry-facing Venture Brain contracts."""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web

from lifeform_domain_venture import (
    VentureBrainConflictError,
    VentureBrainController,
    VentureBrainLineageError,
    VentureBrainReadOnlyError,
    VentureBrainSettlementPendingError,
    VentureContextRequest,
    VentureOutcomeReport,
)
from lifeform_service.dto import ErrorResponse
from lifeform_service.session_manager import SessionManager


_APP_KEY = "venture_brain_controller"


def register_venture_brain_routes(
    app: web.Application,
    *,
    controller: VentureBrainController | None = None,
) -> None:
    """Register the two v1 Venture Brain endpoints."""

    app[_APP_KEY] = controller or VentureBrainController()
    app.router.add_post(
        "/v1/sessions/{session_id}/venture/context-packs",
        _handle_context_pack,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/venture/outcomes",
        _handle_outcome,
    )


def venture_brain_controller(app: web.Application) -> VentureBrainController:
    return app[_APP_KEY]


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        ErrorResponse(error=code, detail=detail).to_json(),
        status=status,
    )


async def _json_object(request: web.Request) -> dict[str, Any]:
    if not request.body_exists:
        raise ValueError("Expected a JSON object body")
    text = await request.text()
    if not text.strip():
        raise ValueError("Expected a JSON object body")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object body")
    return payload


async def _venture_session(request: web.Request):
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    vertical_name = manager.vertical_name_for(session_id)
    if vertical_name != "venture":
        return None, _error(
            409,
            "venture_vertical_required",
            f"session {session_id!r} uses vertical {vertical_name!r}, not 'venture'",
        )
    if manager.is_historical_readonly(session_id):
        return None, _error(
            409,
            "historical_session_readonly",
            "Venture Brain writes are disabled for historical read-only sessions",
        )
    return await manager.get_session(session_id), None


async def _handle_context_pack(request: web.Request) -> web.Response:
    session, guard_error = await _venture_session(request)
    if guard_error is not None:
        return guard_error
    try:
        context_request = VentureContextRequest.from_json(await _json_object(request))
        snapshot, created = await venture_brain_controller(request.app).publish_context_pack(
            session=session,
            request=context_request,
        )
    except VentureBrainConflictError as exc:
        return _error(409, "venture_idempotency_conflict", str(exc))
    except VentureBrainReadOnlyError as exc:
        return _error(409, "historical_session_readonly", str(exc))
    except ValueError as exc:
        return _error(400, "invalid_venture_context_request", str(exc))
    return web.json_response(snapshot.to_json(), status=201 if created else 200)


async def _handle_outcome(request: web.Request) -> web.Response:
    session, guard_error = await _venture_session(request)
    if guard_error is not None:
        return guard_error
    try:
        report = VentureOutcomeReport.from_json(await _json_object(request))
        receipt, created = await venture_brain_controller(request.app).publish_outcome(
            session=session,
            report=report,
        )
    except VentureBrainConflictError as exc:
        return _error(409, "venture_idempotency_conflict", str(exc))
    except VentureBrainLineageError as exc:
        return _error(409, "venture_context_lineage_error", str(exc))
    except VentureBrainSettlementPendingError as exc:
        return _error(409, "venture_settlement_pending", str(exc))
    except VentureBrainReadOnlyError as exc:
        return _error(409, "historical_session_readonly", str(exc))
    except ValueError as exc:
        return _error(400, "invalid_venture_outcome", str(exc))
    return web.json_response(receipt.to_json(), status=201 if created else 200)


__all__ = (
    "register_venture_brain_routes",
    "venture_brain_controller",
)
