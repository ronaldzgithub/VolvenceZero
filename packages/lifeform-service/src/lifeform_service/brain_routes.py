"""Uniform HTTP adapter for product-facing vertical Brains.

The cognitive owners remain in the kernel and each vertical continues to own
its typed state, action and outcome semantics.  This module only removes the
transport duplication: every supported vertical exposes the same two
operations and the live session selects the matching adapter.

Legacy ``/{vertical}/...`` routes remain registered by their original modules;
new clients should use ``/brain/...`` so Coding, Venture and Operations hosts
share one integration shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from aiohttp import web

from lifeform_domain_coding import (
    CodingBrainConflictError,
    CodingBrainController,
    CodingBrainLineageError,
    CodingBrainReadOnlyError,
    CodingContextRequest,
    CodingOutcomeReport,
)
from lifeform_domain_coding.coding_brain_contracts import (
    CONTEXT_REQUEST_SCHEMA_VERSION as CODING_CONTEXT_REQUEST_SCHEMA_VERSION,
    OUTCOME_REPORT_SCHEMA_VERSION as CODING_OUTCOME_REPORT_SCHEMA_VERSION,
)
from lifeform_domain_operations import (
    CONTEXT_REQUEST_SCHEMA_VERSION as OPERATIONS_CONTEXT_REQUEST_SCHEMA_VERSION,
    OUTCOME_REPORT_SCHEMA_VERSION as OPERATIONS_OUTCOME_REPORT_SCHEMA_VERSION,
    OperationsBrainConflictError,
    OperationsBrainController,
    OperationsBrainLineageError,
    OperationsBrainReadOnlyError,
    OperationsBrainSettlementPendingError,
    OperationsContextRequest,
    OperationsOutcomeReport,
)
from lifeform_domain_venture import (
    CONTEXT_REQUEST_SCHEMA_VERSION as VENTURE_CONTEXT_REQUEST_SCHEMA_VERSION,
    OUTCOME_REPORT_SCHEMA_VERSION as VENTURE_OUTCOME_REPORT_SCHEMA_VERSION,
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


_APP_KEY = "vertical_brain_adapters"
_KNOWN_ADAPTER_ERRORS = (
    CodingBrainConflictError,
    CodingBrainLineageError,
    CodingBrainReadOnlyError,
    VentureBrainConflictError,
    VentureBrainLineageError,
    VentureBrainReadOnlyError,
    VentureBrainSettlementPendingError,
    OperationsBrainConflictError,
    OperationsBrainLineageError,
    OperationsBrainReadOnlyError,
    OperationsBrainSettlementPendingError,
)


class _JsonContract(Protocol):
    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> Any: ...


class _VerticalBrainController(Protocol):
    async def publish_context_pack(
        self,
        *,
        session: Any,
        request: Any,
    ) -> tuple[Any, bool]: ...

    async def publish_outcome(
        self,
        *,
        session: Any,
        report: Any,
    ) -> tuple[Any, bool]: ...


class VerticalBrainRouteError(RuntimeError):
    """Stable transport error reusable by local and DLaaS projections."""

    def __init__(self, *, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class VerticalBrainAdapter:
    """Transport contract joining one vertical owner to the shared routes."""

    name: str
    controller: _VerticalBrainController
    context_request_type: type[_JsonContract]
    outcome_report_type: type[_JsonContract]
    context_request_schema_version: str
    outcome_report_schema_version: str
    conflict_errors: tuple[type[Exception], ...]
    lineage_errors: tuple[type[Exception], ...]
    readonly_errors: tuple[type[Exception], ...]
    settlement_errors: tuple[type[Exception], ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("VerticalBrainAdapter.name must be a stable identifier")
        for field_name in (
            "context_request_schema_version",
            "outcome_report_schema_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"VerticalBrainAdapter.{field_name} must be non-empty")

    async def publish_context_payload(
        self,
        *,
        session: Any,
        payload: Mapping[str, object],
    ) -> tuple[dict[str, object], bool]:
        request = self.context_request_type.from_json(payload)
        snapshot, created = await self.controller.publish_context_pack(
            session=session,
            request=request,
        )
        return snapshot.to_json(), created

    async def publish_outcome_payload(
        self,
        *,
        session: Any,
        payload: Mapping[str, object],
    ) -> tuple[dict[str, object], bool]:
        report = self.outcome_report_type.from_json(payload)
        receipt, created = await self.controller.publish_outcome(
            session=session,
            report=report,
        )
        return receipt.to_json(), created

    def discovery_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "context_request_schema_version": self.context_request_schema_version,
            "outcome_report_schema_version": self.outcome_report_schema_version,
            "context_path": "/v1/sessions/{session_id}/brain/context-packs",
            "outcome_path": "/v1/sessions/{session_id}/brain/outcomes",
        }


def default_vertical_brain_adapters(
    *,
    coding: CodingBrainController,
    venture: VentureBrainController,
    operations: OperationsBrainController,
) -> tuple[VerticalBrainAdapter, ...]:
    """Build the installed adapters without moving any domain ownership."""

    return (
        VerticalBrainAdapter(
            name="coding",
            controller=coding,
            context_request_type=CodingContextRequest,
            outcome_report_type=CodingOutcomeReport,
            context_request_schema_version=CODING_CONTEXT_REQUEST_SCHEMA_VERSION,
            outcome_report_schema_version=CODING_OUTCOME_REPORT_SCHEMA_VERSION,
            conflict_errors=(CodingBrainConflictError,),
            lineage_errors=(CodingBrainLineageError,),
            readonly_errors=(CodingBrainReadOnlyError,),
        ),
        VerticalBrainAdapter(
            name="venture",
            controller=venture,
            context_request_type=VentureContextRequest,
            outcome_report_type=VentureOutcomeReport,
            context_request_schema_version=VENTURE_CONTEXT_REQUEST_SCHEMA_VERSION,
            outcome_report_schema_version=VENTURE_OUTCOME_REPORT_SCHEMA_VERSION,
            conflict_errors=(VentureBrainConflictError,),
            lineage_errors=(VentureBrainLineageError,),
            readonly_errors=(VentureBrainReadOnlyError,),
            settlement_errors=(VentureBrainSettlementPendingError,),
        ),
        VerticalBrainAdapter(
            name="operations",
            controller=operations,
            context_request_type=OperationsContextRequest,
            outcome_report_type=OperationsOutcomeReport,
            context_request_schema_version=OPERATIONS_CONTEXT_REQUEST_SCHEMA_VERSION,
            outcome_report_schema_version=OPERATIONS_OUTCOME_REPORT_SCHEMA_VERSION,
            conflict_errors=(OperationsBrainConflictError,),
            lineage_errors=(OperationsBrainLineageError,),
            readonly_errors=(OperationsBrainReadOnlyError,),
            settlement_errors=(OperationsBrainSettlementPendingError,),
        ),
    )


def register_vertical_brain_routes(
    app: web.Application,
    *,
    adapters: tuple[VerticalBrainAdapter, ...],
) -> None:
    """Register the project-agnostic Brain discovery and command routes."""

    by_name = {adapter.name: adapter for adapter in adapters}
    if len(by_name) != len(adapters):
        raise ValueError("vertical Brain adapter names must be unique")
    if not by_name:
        raise ValueError("at least one vertical Brain adapter is required")
    app[_APP_KEY] = by_name
    app.router.add_get("/v1/brains", _handle_discovery)
    app.router.add_post(
        "/v1/sessions/{session_id}/brain/context-packs",
        _handle_context_pack,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/brain/outcomes",
        _handle_outcome,
    )


def vertical_brain_adapters(
    app: web.Application,
) -> Mapping[str, VerticalBrainAdapter]:
    return app[_APP_KEY]


def vertical_brain_adapter(
    app: web.Application,
    name: str,
) -> VerticalBrainAdapter:
    adapters = vertical_brain_adapters(app)
    try:
        return adapters[name]
    except KeyError as exc:
        raise KeyError(f"no vertical Brain adapter is registered for {name!r}") from exc


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        ErrorResponse(error=code, detail=detail).to_json(),
        status=status,
    )


async def _json_object(request: web.Request) -> dict[str, object]:
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


async def _session_and_adapter(
    request: web.Request,
) -> tuple[Any, VerticalBrainAdapter] | web.Response:
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    vertical_name = manager.vertical_name_for(session_id)
    adapter = vertical_brain_adapters(request.app).get(vertical_name)
    if adapter is None:
        return _error(
            409,
            "vertical_brain_unavailable",
            f"session {session_id!r} uses vertical {vertical_name!r}, which has no Brain adapter",
        )
    if manager.is_historical_readonly(session_id):
        return _error(
            409,
            "historical_session_readonly",
            "vertical Brain writes are disabled for historical read-only sessions",
        )
    return await manager.get_session(session_id), adapter


def _project_adapter_error(
    *,
    adapter: VerticalBrainAdapter,
    operation: str,
    exc: Exception,
) -> VerticalBrainRouteError:
    if isinstance(exc, adapter.conflict_errors):
        return VerticalBrainRouteError(
            status=409,
            code="brain_idempotency_conflict",
            detail=str(exc),
        )
    if isinstance(exc, adapter.lineage_errors):
        return VerticalBrainRouteError(
            status=409,
            code="brain_context_lineage_error",
            detail=str(exc),
        )
    if isinstance(exc, adapter.settlement_errors):
        return VerticalBrainRouteError(
            status=409,
            code="brain_settlement_pending",
            detail=str(exc),
        )
    if isinstance(exc, adapter.readonly_errors):
        return VerticalBrainRouteError(
            status=409,
            code="historical_session_readonly",
            detail=str(exc),
        )
    if isinstance(exc, ValueError):
        return VerticalBrainRouteError(
            status=400,
            code=f"invalid_brain_{operation}",
            detail=str(exc),
        )
    raise exc


async def publish_vertical_brain_payload(
    *,
    adapter: VerticalBrainAdapter,
    operation: str,
    session: Any,
    payload: Mapping[str, object],
) -> tuple[dict[str, object], bool]:
    """Publish one of the two closed Brain operations through its adapter."""

    if operation not in {"context-packs", "outcomes"}:
        raise ValueError(f"unsupported vertical Brain operation: {operation!r}")
    try:
        if operation == "context-packs":
            return await adapter.publish_context_payload(
                session=session,
                payload=payload,
            )
        return await adapter.publish_outcome_payload(
            session=session,
            payload=payload,
        )
    except ValueError as exc:
        error_operation = (
            "context_request" if operation == "context-packs" else "outcome"
        )
        raise _project_adapter_error(
            adapter=adapter,
            operation=error_operation,
            exc=exc,
        ) from exc
    except _KNOWN_ADAPTER_ERRORS as exc:
        error_operation = (
            "context_request" if operation == "context-packs" else "outcome"
        )
        raise _project_adapter_error(
            adapter=adapter,
            operation=error_operation,
            exc=exc,
        ) from exc


async def _dispatch(
    request: web.Request,
    *,
    operation: str,
) -> web.Response:
    resolved = await _session_and_adapter(request)
    if isinstance(resolved, web.Response):
        return resolved
    session, adapter = resolved
    try:
        payload = await _json_object(request)
    except ValueError as exc:
        return _error(400, "invalid_json_body", str(exc))
    try:
        body, created = await publish_vertical_brain_payload(
            adapter=adapter,
            operation=operation,
            session=session,
            payload=payload,
        )
    except VerticalBrainRouteError as exc:
        return _error(exc.status, exc.code, exc.detail)
    response = web.json_response(body, status=201 if created else 200)
    response.headers["X-Volvence-Brain"] = adapter.name
    return response


async def _handle_context_pack(request: web.Request) -> web.Response:
    return await _dispatch(
        request,
        operation="context-packs",
    )


async def _handle_outcome(request: web.Request) -> web.Response:
    return await _dispatch(
        request,
        operation="outcomes",
    )


async def _handle_discovery(request: web.Request) -> web.Response:
    adapters = vertical_brain_adapters(request.app)
    return web.json_response(
        {
            "schema_version": "vertical-brain-registry.v1",
            "brains": [
                adapters[name].discovery_payload()
                for name in sorted(adapters)
            ],
        }
    )


__all__ = (
    "VerticalBrainAdapter",
    "VerticalBrainRouteError",
    "default_vertical_brain_adapters",
    "publish_vertical_brain_payload",
    "register_vertical_brain_routes",
    "vertical_brain_adapter",
    "vertical_brain_adapters",
)
