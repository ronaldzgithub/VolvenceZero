"""Thin HTTP projection for the AutoCompany-facing Operations Brain contracts."""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web

from lifeform_domain_operations import (
    OperationsBrainConflictError,
    OperationsBrainController,
    OperationsBrainLineageError,
    OperationsBrainReadOnlyError,
    OperationsBrainSettlementPendingError,
    OperationsContextRequest,
    OperationsOutcomeReport,
)
from lifeform_service.dto import ErrorResponse
from lifeform_service.session_manager import SessionManager


_APP_KEY = "operations_brain_controller"


class OperationsRouteError(RuntimeError):
    """Stable service-layer error projection reusable by DLaaS adapters."""

    def __init__(self, *, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def register_operations_brain_routes(
    app: web.Application,
    *,
    controller: OperationsBrainController | None = None,
) -> None:
    """Register the two session-local Operations Brain v1 endpoints."""

    app[_APP_KEY] = controller or OperationsBrainController()
    app.router.add_post(
        "/v1/sessions/{session_id}/operations/context-packs",
        _handle_context_pack,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/operations/outcomes",
        _handle_outcome,
    )


def operations_brain_controller(app: web.Application) -> OperationsBrainController:
    return app[_APP_KEY]


async def publish_operations_context_payload(
    *,
    controller: OperationsBrainController,
    session: Any,
    payload: dict[str, Any],
) -> tuple[dict[str, object], bool]:
    """Validate and publish a context payload through the domain owner."""

    try:
        context_request = OperationsContextRequest.from_json(payload)
        snapshot, created = await controller.publish_context_pack(
            session=session,
            request=context_request,
        )
    except OperationsBrainConflictError as exc:
        raise OperationsRouteError(
            status=409,
            code="operations_idempotency_conflict",
            detail=str(exc),
        ) from exc
    except OperationsBrainReadOnlyError as exc:
        raise OperationsRouteError(
            status=409,
            code="historical_session_readonly",
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise OperationsRouteError(
            status=400,
            code="invalid_operations_context_request",
            detail=str(exc),
        ) from exc
    return snapshot.to_json(), created


async def publish_operations_outcome_payload(
    *,
    controller: OperationsBrainController,
    session: Any,
    payload: dict[str, Any],
) -> tuple[dict[str, object], bool]:
    """Validate and publish an outcome payload through the domain owner."""

    try:
        report = OperationsOutcomeReport.from_json(payload)
        receipt, created = await controller.publish_outcome(
            session=session,
            report=report,
        )
    except OperationsBrainConflictError as exc:
        raise OperationsRouteError(
            status=409,
            code="operations_idempotency_conflict",
            detail=str(exc),
        ) from exc
    except OperationsBrainLineageError as exc:
        raise OperationsRouteError(
            status=409,
            code="operations_context_lineage_error",
            detail=str(exc),
        ) from exc
    except OperationsBrainSettlementPendingError as exc:
        raise OperationsRouteError(
            status=409,
            code="operations_settlement_pending",
            detail=str(exc),
        ) from exc
    except OperationsBrainReadOnlyError as exc:
        raise OperationsRouteError(
            status=409,
            code="historical_session_readonly",
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise OperationsRouteError(
            status=400,
            code="invalid_operations_outcome",
            detail=str(exc),
        ) from exc
    return receipt.to_json(), created


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


async def _operations_session(request: web.Request):
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    vertical_name = manager.vertical_name_for(session_id)
    if vertical_name != "operations":
        return None, _error(
            409,
            "operations_vertical_required",
            f"session {session_id!r} uses vertical {vertical_name!r}, not 'operations'",
        )
    if manager.is_historical_readonly(session_id):
        return None, _error(
            409,
            "historical_session_readonly",
            "Operations Brain writes are disabled for historical read-only sessions",
        )
    return await manager.get_session(session_id), None


async def _handle_context_pack(request: web.Request) -> web.Response:
    session, guard_error = await _operations_session(request)
    if guard_error is not None:
        return guard_error
    try:
        payload = await _json_object(request)
        body, created = await publish_operations_context_payload(
            controller=operations_brain_controller(request.app),
            session=session,
            payload=payload,
        )
    except ValueError as exc:
        return _error(400, "invalid_json_body", str(exc))
    except OperationsRouteError as exc:
        return _error(exc.status, exc.code, exc.detail)
    return web.json_response(body, status=201 if created else 200)


async def _handle_outcome(request: web.Request) -> web.Response:
    session, guard_error = await _operations_session(request)
    if guard_error is not None:
        return guard_error
    try:
        payload = await _json_object(request)
        body, created = await publish_operations_outcome_payload(
            controller=operations_brain_controller(request.app),
            session=session,
            payload=payload,
        )
    except ValueError as exc:
        return _error(400, "invalid_json_body", str(exc))
    except OperationsRouteError as exc:
        return _error(exc.status, exc.code, exc.detail)
    return web.json_response(body, status=201 if created else 200)


__all__ = (
    "operations_brain_controller",
    "OperationsRouteError",
    "publish_operations_context_payload",
    "publish_operations_outcome_payload",
    "register_operations_brain_routes",
)
