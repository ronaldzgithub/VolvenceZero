"""DLaaS aiohttp router and dispatch entry point.

This module owns the ``/dlaas/instances/{ai_id}/interactions`` route
plus the wiring to attach the control-plane CRUD surface (Slice 3 +
4) and the multi-ai_id launcher.

Dispatch flow:

1. Parse the JSON body into a typed
   :class:`dlaas_platform_contracts.InteractionEnvelope`.
2. If the app carries an :class:`InstanceManager`, look up the
   ``SessionManager`` for the path ``ai_id``. Fall back to the
   single-instance ``app["session_manager"]`` (Slice 1) when no
   launcher is bound or when the ``ai_id`` is not adopted yet.
3. Hand off to :func:`dlaas_platform_api.dispatch.dispatch_envelope`,
   which switches on :class:`InteractionType` and calls the matching
   kernel sink.
4. Serialise the resulting JSON body back to the client.

The Slice 1 ``attach_dlaas_routes`` entry point continues to work
without a registry — it pins every request to the single shared
``SessionManager``. ``build_dlaas_app`` is the recommended Slice 3+
entry point: it builds a registry, a launcher, and the full
control-plane surface in one call.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import InteractionEnvelope
from dlaas_platform_launcher import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceManager,
    InstanceNotFound,
)
from dlaas_platform_launcher.instance_manager import default_vertical_resolver
from dlaas_platform_eval import attach_eval_routes
from dlaas_platform_ops import (
    OPS_BUNDLE_APP_KEY,
    OpsBundle,
    attach_ops_routes,
    operator_takeover_response_body,
)
from dlaas_platform_registry import (
    PlatformAuthBundle,
    PlatformAuthConfig,
    REGISTRY_APP_KEY,
    Registry,
    TenantStore,
)
from lifeform_service.app import create_app as create_lifeform_app
from lifeform_service.session_manager import (
    SessionAlreadyExistsError,
    SessionManager,
    SessionNotFoundError,
)

from dlaas_platform_api.control_plane import attach_control_plane_routes
from dlaas_platform_api.dispatch import DispatchError, dispatch_envelope

_LOG = logging.getLogger("dlaas_platform_api")

DLAAS_APP_AI_ID_KEY = "dlaas_default_ai_id"
"""``app[DLAAS_APP_AI_ID_KEY]`` — Slice 1 hardcoded ``ai_id`` fallback."""


def attach_dlaas_routes(
    app: web.Application,
    *,
    default_ai_id: str = "ai_default",
) -> web.Application:
    """Register only the runtime ``/dlaas/instances/{ai_id}/interactions``.

    Slice 1 entry point. The app MUST already have a
    ``session_manager`` set up by ``lifeform_service.app.create_app``;
    every request — regardless of the path ``ai_id`` — is served by
    that single SessionManager. Suitable for dev / smoke testing
    before the multi-tenant control plane is wired.
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


def attach_dlaas_full_stack(
    app: web.Application,
    *,
    registry: Registry,
    auth_config: PlatformAuthConfig,
    instance_manager: InstanceManager,
    default_ai_id: str = "ai_default",
    platform_endpoint: str = "",
) -> web.Application:
    """Wire registry + launcher + control plane onto an aiohttp app.

    Slice 3+ entry point. The app must already carry a
    ``session_manager`` (the Slice 1 fallback the dispatch reaches
    for when the path ``ai_id`` is not registered with the
    launcher). After this call the app exposes the runtime
    ``/dlaas/instances/{ai_id}/interactions`` route plus every
    control-plane CRUD endpoint listed in
    ``docs/specs/dlaas-platform.md``.
    """
    if "session_manager" not in app:
        raise ValueError(
            "attach_dlaas_full_stack requires an aiohttp app produced by "
            "lifeform_service.app.create_app (session_manager missing)."
        )
    app[REGISTRY_APP_KEY] = PlatformAuthBundle(
        tenant_store=TenantStore(registry),
        auth_config=auth_config,
    )
    app[INSTANCE_MANAGER_APP_KEY] = instance_manager
    app[DLAAS_APP_AI_ID_KEY] = default_ai_id
    if platform_endpoint:
        app["dlaas_platform_endpoint"] = platform_endpoint
    app.router.add_post(
        "/dlaas/instances/{ai_id}/interactions",
        _handle_interaction,
    )
    attach_control_plane_routes(app, registry=registry)
    attach_ops_routes(app, registry=registry)
    attach_eval_routes(app, registry=registry)
    return app


def build_dlaas_app(
    *,
    db_path: str | os.PathLike[str] = ":memory:",
    default_ai_id: str = "ai_default",
    control_plane_secret: str | None = None,
    service_secret: str | None = None,
    platform_endpoint: str = "",
    instance_manager: InstanceManager | None = None,
    **service_kwargs: Any,
) -> web.Application:
    """Build a lifeform-service app with the full DLaaS surface attached.

    Equivalent to:

        app = lifeform_service.app.create_app(**service_kwargs)
        registry = Registry(db_path=db_path)
        instance_manager = InstanceManager(
            vertical_resolver=default_vertical_resolver(),
            substrate_runtime=app["session_manager"].substrate_runtime,
        )
        attach_dlaas_full_stack(
            app,
            registry=registry,
            auth_config=PlatformAuthConfig(...),
            instance_manager=instance_manager,
            ...
        )

    ``control_plane_secret`` / ``service_secret`` default to
    ``$DLAAS_CONTROL_PLANE_SECRET`` / ``$DLAAS_SERVICE_SECRET``
    when the caller does not supply them explicitly. An empty value
    administratively disables that auth mode.
    """
    app = create_lifeform_app(**service_kwargs)
    registry = Registry(db_path=str(db_path))
    if instance_manager is None:
        instance_manager = InstanceManager(
            vertical_resolver=default_vertical_resolver(),
            substrate_runtime=app["session_manager"].substrate_runtime,
        )
    auth_config = PlatformAuthConfig(
        control_plane_secret=(
            control_plane_secret
            if control_plane_secret is not None
            else os.environ.get("DLAAS_CONTROL_PLANE_SECRET", "")
        ),
        service_secret=(
            service_secret
            if service_secret is not None
            else os.environ.get("DLAAS_SERVICE_SECRET", "")
        ),
    )
    attach_dlaas_full_stack(
        app,
        registry=registry,
        auth_config=auth_config,
        instance_manager=instance_manager,
        default_ai_id=default_ai_id,
        platform_endpoint=platform_endpoint,
    )
    return app


# ---------------------------------------------------------------------------
# Runtime dispatch
# ---------------------------------------------------------------------------


async def _handle_interaction(request: web.Request) -> web.Response:
    """Adapt the HTTP request to a typed dispatch call."""
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

    ops_bundle = request.app.get(OPS_BUNDLE_APP_KEY)
    if isinstance(ops_bundle, OpsBundle):
        if await ops_bundle.pause_store.is_paused(
            ai_id=ai_id, session_id=envelope.session_id
        ):
            body = operator_takeover_response_body(
                ai_id=ai_id,
                session_id=envelope.session_id,
                contract_id=envelope.contract_id,
                interaction_type=envelope.interaction_type.value,
            )
            return web.json_response(dict(body))

    try:
        manager = _resolve_session_manager(request, ai_id)
    except _AiIdNotFoundError as exc:
        return _json_error(
            status=404, error=exc.code, detail=exc.detail
        )

    try:
        session = await _get_or_create_session(manager, envelope.session_id)
    except SessionAlreadyExistsError as exc:  # pragma: no cover - racy
        return _json_error(
            status=409, error="session_already_exists", detail=str(exc)
        )
    except SessionNotFoundError as exc:  # pragma: no cover - get-or-create
        return _json_error(
            status=404, error="session_not_found", detail=str(exc)
        )

    try:
        body = await dispatch_envelope(
            envelope=envelope, session=session, ai_id=ai_id
        )
    except DispatchError as exc:
        return _json_error(status=exc.status, error=exc.code, detail=exc.detail)
    return web.json_response(body)


def _resolve_session_manager(
    request: web.Request, ai_id: str
) -> SessionManager:
    """Pick the SessionManager for ``ai_id``.

    Resolution order:

    1. If the launcher is bound and knows ``ai_id`` → return the
       launcher's per-ai_id ``SessionManager``.
    2. Else if the launcher is bound but does NOT know ``ai_id`` →
       reject with 404 ``ai_id_not_found``. This is the multi-tenant
       path: every ai_id must be adopted before traffic flows.
    3. Else (no launcher) fall back to ``app["session_manager"]``
       (Slice 1 single-instance path).
    """
    launcher = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if isinstance(launcher, InstanceManager):
        try:
            return launcher.get(ai_id)
        except InstanceNotFound as exc:
            raise _AiIdNotFoundError(
                code="ai_id_not_found",
                detail=(
                    f"ai_id={ai_id!r} is not adopted on this server. "
                    "Call POST /dlaas/adopt with a published, activated "
                    "template before sending interactions."
                ),
            ) from exc
    return request.app["session_manager"]


async def _parse_envelope(request: web.Request) -> InteractionEnvelope:
    if not request.body_exists:
        raise _EnvelopeError("invalid_envelope", "Request body is required")
    try:
        text = await request.text()
    except (web.HTTPException, OSError) as exc:
        raise _EnvelopeError(
            "invalid_body", f"Could not read body: {exc}"
        ) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _EnvelopeError(
            "invalid_json", f"Body is not valid JSON: {exc}"
        ) from exc
    try:
        return InteractionEnvelope.from_json(data)
    except ValueError as exc:
        raise _EnvelopeError("invalid_envelope", str(exc)) from exc


async def _get_or_create_session(manager: SessionManager, session_id: str):
    """Reuse an existing session if present; otherwise create with that id."""
    try:
        return await manager.get_session(session_id)
    except SessionNotFoundError:
        return await manager.create_session(session_id=session_id)


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
    """Raised by ``_parse_envelope`` for any 400-level parse failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class _AiIdNotFoundError(Exception):
    """Raised by :func:`_resolve_session_manager` when launcher is bound
    but the ai_id is unknown."""

    def __init__(self, *, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


__all__ = [
    "DLAAS_APP_AI_ID_KEY",
    "attach_dlaas_full_stack",
    "attach_dlaas_routes",
    "build_dlaas_app",
]
