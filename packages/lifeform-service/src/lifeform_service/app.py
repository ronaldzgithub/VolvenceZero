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
import pathlib
from collections.abc import Mapping
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
from lifeform_core import LlmJsonClient
from lifeform_service.protocol_routes import register_protocol_routes
from lifeform_service.protocol_uptake import ProtocolUptakeService
from lifeform_service.session_manager import (
    SessionAlreadyExistsError,
    SessionManager,
    SessionNotFoundError,
    TemplatesNotSupportedError,
)
from lifeform_service.substrate_registry import (
    SubstrateRuntimeProvider,
    SubstrateSwapError,
    UnknownSubstrateModelError,
    fixed_provider_from_runtime,
)
from lifeform_service.vertical_registry import (
    UnknownVerticalError,
    VerticalNotAlphaCapableError,
    VerticalRegistry,
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
    vertical: VerticalSpec | None = None,
    verticals: Mapping[str, VerticalSpec] | None = None,
    default_vertical: str | None = None,
    max_sessions: int = 256,
    idle_eviction_seconds: float | None = 60 * 30,
    substrate_runtime: "OpenWeightResidualRuntime | None" = None,
    substrate_provider: SubstrateRuntimeProvider | None = None,
    alpha_config: AlphaServiceConfig | None = None,
    templates_root_dir: str | None = None,
    protocol_uptake_service: ProtocolUptakeService | None = None,
    external_llm_client: "LlmJsonClient | None" = None,
) -> web.Application:
    """Build the aiohttp Application.

    Two ways to declare verticals:

    * **Multi-vertical (preferred for browser-chat)**: pass
      ``verticals={...}`` (the dict returned by
      :func:`lifeform_service.verticals.discover_verticals`) plus
      ``default_vertical=<name>``. Sessions can pick any registered
      vertical at creation time via
      ``POST /v1/sessions {"vertical": "..."}``; templates and
      save-as-template are scoped per-session-vertical.
    * **Single-vertical (legacy / DLaaS)**: pass ``vertical=<spec>``.
      The service still publishes ``/v1/verticals`` (with one
      entry) and accepts ``vertical=`` on session creation, but
      only that one name is accepted.

    Args:
        vertical: legacy single-vertical entry. Mutually exclusive
            with ``verticals``.
        verticals: name -> VerticalSpec map. Combined with
            ``default_vertical`` to build the registry.
        default_vertical: the vertical name used by sessions that
            do not declare ``vertical`` in their request body.
            Required when ``verticals`` is supplied; ignored when
            ``vertical`` is supplied (the single vertical's name is
            the default by definition).
        templates_root_dir: filesystem root the chat-browser template
            surface scans. ``GET /v1/templates`` lists
            ``<root>/<vertical_subdir>/*.json`` filtered by the
            ``?vertical=`` query (default = the registry's default
            vertical). Save-as-template writes back into the
            session's vertical subdir. ``None`` disables the
            template surface entirely.

    Substrate args (``substrate_runtime`` vs ``substrate_provider``,
    plus the rest) are unchanged from the prior release.
    """
    if substrate_runtime is not None and substrate_provider is not None:
        raise ValueError(
            "create_app: pass substrate_runtime OR substrate_provider, not both"
        )
    if substrate_runtime is not None:
        _enforce_frozen_for_sharing(substrate_runtime)
        substrate_provider = fixed_provider_from_runtime(substrate_runtime)
    alpha = alpha_config or AlphaServiceConfig()
    alpha_provider = (
        AlphaIdentityProvider(allowed_users=alpha.alpha_users)
        if alpha.enabled
        else None
    )
    registry = _build_vertical_registry(
        vertical=vertical,
        verticals=verticals,
        default_vertical=default_vertical,
        alpha_enabled=alpha.enabled,
    )
    if alpha.enabled and registry.default.alpha_factory is None and registry.default.template_adapter is None:
        raise ValueError(
            f"default vertical {registry.default_name!r} does not support "
            "alpha mode (no alpha_factory and no template_adapter)"
        )
    service_templates_root = (
        pathlib.Path(templates_root_dir).expanduser()
        if templates_root_dir and templates_root_dir.strip()
        else None
    )
    manager = SessionManager(
        vertical_registry=registry,
        alpha_identity_provider=alpha_provider,
        alpha_memory_scope_root_dir=alpha.memory_scope_root_dir,
        max_sessions=max_sessions,
        idle_eviction_seconds=idle_eviction_seconds,
        substrate_provider=substrate_provider,
        templates_root_dir=service_templates_root,
    )
    if substrate_provider is not None and substrate_provider.swap_supported:
        # Wire the SessionManager's session-clearer as the swap
        # pre-action so model-swap closes every active session before
        # the new runtime loads. Set after manager construction
        # because the callback is a bound method on the manager.
        substrate_provider.set_pre_swap_callback(manager.close_all_sessions_sync)
    app = web.Application(middlewares=[_error_middleware])
    app["session_manager"] = manager
    # ``vertical_spec`` continues to point at the default vertical so
    # existing routes (``_handle_info`` / ``_handle_health``) keep
    # working unchanged. Per-request vertical lookups go through
    # ``vertical_registry``.
    app["vertical_spec"] = registry.default
    app["vertical_registry"] = registry
    app["substrate_provider"] = substrate_provider
    app["substrate_runtime"] = (
        substrate_provider.current_runtime if substrate_provider is not None else None
    )
    app["alpha_config"] = alpha
    app["templates_root_dir"] = service_templates_root
    # Shared external LLM client. Any vertical / route handler that
    # wants LLM access reads ``app["external_llm_client"]`` (may be
    # ``None`` when unconfigured). Same instance the protocol
    # uptake routes use, so a single API key / quota covers all
    # opt-in consumers. Verticals that don't need the LLM simply
    # don't read this key — no behavior change.
    app["external_llm_client"] = external_llm_client
    app.router.add_get("/", _handle_chat_ui)
    app.router.add_get("/chat", _handle_chat_ui)
    app.router.add_get("/v1/health", _handle_health)
    app.router.add_get("/v1/info", _handle_info)
    app.router.add_get("/v1/verticals", _handle_list_verticals)
    app.router.add_get("/v1/models", _handle_list_models)
    app.router.add_post("/v1/admin/substrate", _handle_swap_substrate)
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
    if protocol_uptake_service is not None:
        register_protocol_routes(app, uptake_service=protocol_uptake_service)
    return app


def _build_vertical_registry(
    *,
    vertical: VerticalSpec | None,
    verticals: Mapping[str, VerticalSpec] | None,
    default_vertical: str | None,
    alpha_enabled: bool,
) -> VerticalRegistry:
    """Resolve the two ``create_app`` vertical-shape parameters.

    * ``vertical=`` (legacy single-vertical) → 1-entry registry
      with that vertical as default.
    * ``verticals=`` + ``default_vertical=`` (multi-vertical) →
      registry with the named default.
    * Both / neither → ``ValueError``.
    """
    if vertical is not None and verticals is not None:
        raise ValueError(
            "create_app: pass `vertical=` (single) OR `verticals=` "
            "(multi), not both"
        )
    if vertical is not None:
        return VerticalRegistry.single(vertical, alpha_enabled=alpha_enabled)
    if verticals is None or not verticals:
        raise ValueError(
            "create_app: at least one vertical is required (pass "
            "`vertical=` or `verticals=` with `default_vertical=`)"
        )
    if default_vertical is None or not default_vertical.strip():
        raise ValueError(
            "create_app: `default_vertical=` is required when "
            "`verticals=` is supplied"
        )
    return VerticalRegistry.from_mapping(
        verticals,
        default_name=default_vertical.strip(),
        alpha_enabled=alpha_enabled,
    )


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
    # Read through the provider so /v1/info reflects the current
    # post-swap runtime. The launcher / startup-time cache in
    # app["substrate_runtime"] is only the initial value and goes
    # stale after the first swap.
    provider: SubstrateRuntimeProvider | None = request.app.get("substrate_provider")
    runtime = (
        provider.current_runtime if provider is not None else request.app.get("substrate_runtime")
    )
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


async def _handle_list_verticals(request: web.Request) -> web.Response:
    """Report the registry contents and capability flags.

    Used by the chat UI's vertical dropdown. ``alpha_enabled`` is
    surfaced separately so the UI can grey out alpha-incompatible
    options when the service is running in alpha mode (rather than
    duplicating the gate logic on the client).
    """
    registry: VerticalRegistry = request.app["vertical_registry"]
    alpha: AlphaServiceConfig = request.app["alpha_config"]
    return _json_ok(
        {
            "default_vertical": registry.default_name,
            "alpha_enabled": alpha.enabled,
            "verticals": list(registry.summary_for_ui()),
        }
    )


async def _handle_create_session(request: web.Request) -> web.Response:
    payload = await _maybe_json(request)
    requested_id = None
    template_id: str | None = None
    requested_vertical: str | None = None
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
        raw_vertical = payload.get("vertical")
        if raw_vertical is not None and not isinstance(raw_vertical, str):
            raise _BadRequest("invalid_vertical", "vertical must be a string")
        if isinstance(raw_vertical, str) and raw_vertical.strip():
            requested_vertical = raw_vertical.strip()
    manager: SessionManager = request.app["session_manager"]
    alpha: AlphaServiceConfig = request.app["alpha_config"]
    user_id = None
    if alpha.enabled:
        user_id = _alpha_user_id(request, payload)
    try:
        session = await manager.create_session(
            session_id=requested_id,
            user_id=user_id,
            template_id=template_id,
            vertical_name=requested_vertical,
        )
    except UnknownVerticalError as exc:
        return _json_error(
            status=422, error="unknown_vertical", detail=str(exc)
        )
    except VerticalNotAlphaCapableError as exc:
        return _json_error(
            status=422, error="vertical_not_alpha_capable", detail=str(exc)
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
    bound_vertical_name = manager.vertical_name_for(session.session_id)
    bound_spec = manager.vertical_registry.get(bound_vertical_name)
    if bound_spec is None:
        bound_spec = manager.vertical_registry.default
    body = CreateSessionResponse(
        session_id=session.session_id,
        vertical=bound_vertical_name,
        has_temporal_bootstrap=bound_spec.has_temporal_bootstrap,
        has_regime_bootstrap=bound_spec.has_regime_bootstrap,
        user_id=user_id,
        service_version=alpha.service_version if alpha.enabled else "",
        policy_version=alpha.policy_version if alpha.enabled else "",
        alpha_disclaimer=ALPHA_DISCLAIMER if alpha.enabled else "",
    )
    response_payload = body.to_json()
    response_payload["template_id"] = template_id
    return _json_ok(response_payload, status=201)


async def _handle_list_models(request: web.Request) -> web.Response:
    """Report available substrate models + the current selection.

    Always returns 200, even when the service has no provider — UI
    consumers branch on ``swap_supported`` rather than hitting a
    different status. The shape mirrors ``/v1/templates``: a tuple
    of typed entries plus discoverable capability flags.
    """
    provider: SubstrateRuntimeProvider | None = request.app.get("substrate_provider")
    spec: VerticalSpec = request.app["vertical_spec"]
    if provider is None:
        return _json_ok(
            {
                "vertical": spec.name,
                "swap_supported": False,
                "current_model_id": None,
                "current_runtime_origin": None,
                "swap_count": 0,
                "last_swap_error": "",
                "models": [],
            }
        )
    runtime = provider.current_runtime
    return _json_ok(
        {
            "vertical": spec.name,
            "swap_supported": provider.swap_supported,
            "current_model_id": provider.current_model_id or None,
            "current_runtime_origin": (
                getattr(runtime, "runtime_origin", None) if runtime is not None else None
            ),
            "swap_count": provider.swap_count,
            "last_swap_error": provider.last_swap_error,
            "models": [spec.to_json() for spec in provider.available],
        }
    )


async def _handle_swap_substrate(request: web.Request) -> web.Response:
    """Hot-swap the shared substrate runtime to a different model id.

    Only available when the service was started with a swap-capable
    :class:`SubstrateRuntimeProvider`. Production paths that pass a
    fixed ``substrate_runtime`` (DLaaS) report ``503``: hot swapping
    a multi-tenant deployment's base model from a single HTTP route
    would violate the production contract that all tenants get the
    same R2 substrate for the contract's lifetime.

    The route is intentionally **not** behind alpha auth: the model
    selector is a developer affordance on the local browser-chat
    process. Multi-user deployments must keep this surface internal
    (firewall / reverse-proxy ACL). We surface that posture in the
    URL prefix (``/v1/admin/...``) so misuse is loud rather than
    silent.
    """
    provider: SubstrateRuntimeProvider | None = request.app.get("substrate_provider")
    if provider is None or not provider.swap_supported:
        return _json_error(
            status=503,
            error="substrate_swap_not_supported",
            detail=(
                "this service was started without a swap-capable substrate "
                "provider (typical for DLaaS / fixed-runtime deployments)"
            ),
        )
    payload = await _require_json(request)
    raw = payload.get("model_id")
    if not isinstance(raw, str) or not raw.strip():
        raise _BadRequest(
            "invalid_model_id", "model_id is required and must be a string"
        )
    model_id = raw.strip()
    try:
        result = await provider.swap(model_id)
    except UnknownSubstrateModelError as exc:
        return _json_error(
            status=400, error="unknown_model_id", detail=str(exc)
        )
    except SubstrateSwapError as exc:
        # Loader failure: respond 503 with the cause so the operator
        # can decide whether to retry. Provider has already cleared
        # the current runtime; UI must call swap again with a known-
        # good model_id (or restart the service).
        return _json_error(
            status=503,
            error="substrate_load_failed",
            detail=str(exc),
            extra={
                "target_model_id": exc.target_model_id,
                "previous_model_id": exc.previous_model_id,
            },
        )
    # Swap succeeded. We deliberately do NOT mutate
    # ``request.app["substrate_runtime"]`` here — aiohttp
    # deprecates writing into a started app's state, and the right
    # consumers (``/v1/info``, ``SessionManager.substrate_runtime``)
    # already read through the provider so they see the new
    # runtime without a mutable cache.
    return _json_ok(
        {
            "swapped": True,
            "model_id": result.model_id,
            "previous_model_id": result.previous_model_id,
            "runtime_origin": result.runtime_origin,
            "closed_session_count": result.closed_session_count,
            "duration_seconds": round(result.duration_seconds, 3),
        }
    )


async def _handle_list_templates(request: web.Request) -> web.Response:
    """List templates for a specific vertical.

    Filtering rule: ``?vertical=<name>`` selects which vertical's
    template subdirectory to scan; default = the registry's
    default vertical. Unknown vertical names return 422 so the
    UI can surface a typo / wrong-deployment error rather than
    silently falling back to the default.
    """
    manager: SessionManager = request.app["session_manager"]
    registry: VerticalRegistry = request.app["vertical_registry"]
    requested = request.query.get("vertical", "").strip()
    target_name = requested or registry.default_name
    if registry.get(target_name) is None:
        return _json_error(
            status=422,
            error="unknown_vertical",
            detail=(
                f"vertical {target_name!r} is not registered; available: "
                f"{sorted(registry.names)!r}"
            ),
        )
    adapter = manager.template_adapter_for(target_name)
    root = manager.templates_dir_for(target_name)
    if adapter is None or root is None:
        return _json_ok(
            {
                "vertical": target_name,
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
            "vertical": target_name,
            "templates_supported": True,
            "templates_root_dir": str(root),
            "templates": [item.to_json() for item in items],
        }
    )


async def _handle_save_as_template(request: web.Request) -> web.Response:
    manager: SessionManager = request.app["session_manager"]
    session_id = request.match_info["session_id"]
    # Save into the SESSION's vertical subdir, not the service-level
    # default vertical's. A session created with vertical=einstein
    # must save under <root>/einstein/, regardless of which vertical
    # the chat UI happens to be displaying right now.
    try:
        session_vertical = manager.vertical_name_for(session_id)
    except SessionNotFoundError:
        raise  # middleware → 404
    adapter = manager.template_adapter_for(session_vertical)
    root = manager.templates_dir_for(session_vertical)
    if adapter is None or root is None:
        return _json_error(
            status=503,
            error="templates_not_supported",
            detail=(
                f"vertical {session_vertical!r} does not register a template "
                "adapter or service has no templates_root_dir configured"
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
    /* The protocolsPanel modal carries inline ``display: grid`` to center
       its inner card; without this rule that inline style wins over the
       UA stylesheet's ``[hidden] { display: none }`` and the Close button
       cannot hide the modal. */
    [hidden] { display: none !important; }
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
      <select id="verticalSelect" title="Pick which vertical (lifeform domain) builds the session"></select>
      <select id="templateSelect" title="Pick a saved lifeform template (optional)">
        <option value="">no template (vertical default)</option>
      </select>
      <button id="createBtn">Create Session</button>
      <button id="saveTemplateBtn" class="secondary" disabled>Save as Template</button>
      <button id="endBtn" class="secondary" disabled>End Scene</button>
      <button id="clearBtn" class="secondary">Clear</button>
      <button id="protocolsBtn" class="secondary" title="Upload PDF / Markdown / task description and review extracted protocols">Protocols</button>
    </section>
    <section class="bar">
      <span id="modelStatus" class="msg system" style="margin: 0; padding: 4px 8px;">substrate: loading...</span>
      <select id="modelSelect" title="Switch the shared base model (closes all sessions)" disabled></select>
      <button id="switchModelBtn" class="secondary" disabled>Switch Model</button>
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
  <div id="protocolsPanel" hidden style="
    position: fixed; inset: 0; background: rgba(2,6,23,0.75);
    display: grid; place-items: center; z-index: 50;
  ">
    <div style="
      width: min(720px, calc(100vw - 32px));
      max-height: calc(100vh - 64px); overflow: auto;
      background: #111827; border: 1px solid #4b5563; border-radius: 14px;
      padding: 18px; color: #e5e7eb;
    ">
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
        <h2 style="margin: 0; font-size: 18px;">Behavior Protocols</h2>
        <button id="protocolsCloseBtn" class="secondary">Close</button>
      </div>
      <div id="protocolsStatus" class="msg system" style="margin-top: 12px;">Loading...</div>
      <h3 style="font-size: 14px; margin: 16px 0 6px;">Upload PDF / Markdown</h3>
      <div class="bar" style="gap: 6px;">
        <input type="file" id="protocolUploadFile" accept=".pdf,.md,.markdown,.txt" style="flex: 1; min-width: 220px;">
        <input id="protocolUploadSeed" placeholder="optional protocol_id seed" style="flex: 1; min-width: 180px;">
        <button id="protocolUploadBtn" class="secondary">Upload</button>
      </div>
      <h3 style="font-size: 14px; margin: 16px 0 6px;">From Description</h3>
      <div class="bar" style="gap: 6px; flex-wrap: wrap;">
        <input id="protocolDescId" placeholder="protocol_id (e.g. spa-bot)" style="flex: 1; min-width: 200px;">
        <input id="protocolDescAdvisor" placeholder="advisor_name" style="flex: 1; min-width: 160px;">
      </div>
      <textarea id="protocolDescText" placeholder="Free-text role description..." style="margin-top: 6px; min-height: 60px;"></textarea>
      <button id="protocolDescBtn" class="secondary" style="margin-top: 6px;">Submit Description</button>
      <h3 style="font-size: 14px; margin: 16px 0 6px;">Pending Candidates</h3>
      <div id="protocolsCandidates"></div>
      <h3 style="font-size: 14px; margin: 16px 0 6px;">Approved Protocols</h3>
      <div id="protocolsApproved"></div>
    </div>
  </div>
  <script>
    const state = {
      sessionId: null,
      sessionVertical: null,
      templatesSupported: false,
      swapSupported: false,
      currentModelId: null,
      defaultVertical: null,
      alphaEnabled: false,
      verticalsByName: {},
      debug: new URLSearchParams(window.location.search).get("debug") === "1",
    };
    const statusEl = document.getElementById("status");
    const logEl = document.getElementById("log");
    const inputEl = document.getElementById("input");
    const userIdEl = document.getElementById("userId");
    const sessionIdEl = document.getElementById("sessionId");
    const verticalSelectEl = document.getElementById("verticalSelect");
    const templateSelectEl = document.getElementById("templateSelect");
    const createBtn = document.getElementById("createBtn");
    const sendBtn = document.getElementById("sendBtn");
    const endBtn = document.getElementById("endBtn");
    const clearBtn = document.getElementById("clearBtn");
    const saveTemplateBtn = document.getElementById("saveTemplateBtn");
    const modelStatusEl = document.getElementById("modelStatus");
    const modelSelectEl = document.getElementById("modelSelect");
    const switchModelBtn = document.getElementById("switchModelBtn");
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
    async function loadModels() {
      try {
        const payload = await requestJson("/v1/models", { method: "GET" });
        state.swapSupported = Boolean(payload.swap_supported);
        state.currentModelId = payload.current_model_id || null;
        modelSelectEl.innerHTML = "";
        const models = Array.isArray(payload.models) ? payload.models : [];
        if (models.length === 0) {
          const opt = document.createElement("option");
          opt.value = "";
          opt.textContent = "no models advertised";
          modelSelectEl.appendChild(opt);
        }
        for (const item of models) {
          const opt = document.createElement("option");
          opt.value = item.model_id;
          const sizeLabel = item.size_label ? ` (${item.size_label})` : "";
          opt.textContent = item.display_name
            ? `${item.display_name}${sizeLabel}`
            : item.model_id;
          if (item.notes) opt.title = item.notes;
          if (item.model_id === state.currentModelId) opt.selected = true;
          modelSelectEl.appendChild(opt);
        }
        modelSelectEl.disabled = !state.swapSupported || models.length === 0;
        switchModelBtn.disabled = !state.swapSupported || models.length === 0;
        const statusBits = [`substrate: ${state.currentModelId || "<none>"}`];
        if (payload.current_runtime_origin) {
          statusBits.push(payload.current_runtime_origin);
        }
        if (!state.swapSupported) statusBits.push("swap disabled");
        if (payload.last_swap_error) {
          statusBits.push(`last error: ${payload.last_swap_error}`);
        }
        modelStatusEl.textContent = statusBits.join(" | ");
      } catch (err) {
        modelStatusEl.textContent = `substrate: failed to load (${err.message})`;
      }
    }
    async function switchModel() {
      if (!state.swapSupported) return;
      const targetId = modelSelectEl.value.trim();
      if (!targetId) return;
      if (targetId === state.currentModelId) {
        addMessage("system", `Already running ${targetId}; nothing to do.`);
        return;
      }
      const ok = window.confirm(
        `Switch substrate to ${targetId}?\n\n`
          + `This will close all active sessions and load the new model `
          + `(may take 30s+ depending on size). Existing chats will be lost.`,
      );
      if (!ok) return;
      switchModelBtn.disabled = true;
      modelSelectEl.disabled = true;
      const previousStatus = modelStatusEl.textContent;
      modelStatusEl.textContent = `substrate: switching to ${targetId}...`;
      try {
        const payload = await requestJson("/v1/admin/substrate", {
          method: "POST",
          body: JSON.stringify({ model_id: targetId }),
        });
        addMessage(
          "system",
          `Substrate swapped: ${payload.previous_model_id || "(none)"} -> `
            + `${payload.model_id} | closed ${payload.closed_session_count} sessions `
            + `| ${payload.duration_seconds.toFixed(2)}s`,
        );
        // Active session was killed by the swap.
        state.sessionId = null;
        statusEl.textContent = `Substrate ${payload.model_id} ready. Create a new session.`;
        setReady(false);
        await loadModels();
      } catch (err) {
        addMessage("system", `Switch failed: ${err.message}`);
        modelStatusEl.textContent = previousStatus;
        modelSelectEl.disabled = false;
        switchModelBtn.disabled = false;
      }
    }
    async function loadVerticals() {
      try {
        const payload = await requestJson("/v1/verticals", { method: "GET" });
        state.defaultVertical = payload.default_vertical || null;
        state.alphaEnabled = Boolean(payload.alpha_enabled);
        state.verticalsByName = {};
        verticalSelectEl.innerHTML = "";
        const items = Array.isArray(payload.verticals) ? payload.verticals : [];
        for (const item of items) {
          state.verticalsByName[item.name] = item;
          const opt = document.createElement("option");
          opt.value = item.name;
          // Build label: name + alpha-incompat hint when applicable.
          let label = item.name;
          const tags = [];
          if (item.is_default) tags.push("default");
          if (item.templates_supported) tags.push("templates");
          if (state.alphaEnabled && !item.alpha_supported) {
            tags.push("no-alpha");
            opt.disabled = true;
          }
          if (tags.length > 0) label += ` [${tags.join(", ")}]`;
          opt.textContent = label;
          if (item.is_default) opt.selected = true;
          verticalSelectEl.appendChild(opt);
        }
        await loadTemplates();
      } catch (err) {
        addMessage("system", `Failed to load verticals: ${err.message}`);
      }
    }
    function selectedVerticalName() {
      return verticalSelectEl.value || state.defaultVertical || "";
    }
    async function loadTemplates() {
      const vertical = selectedVerticalName();
      try {
        const url = vertical
          ? `/v1/templates?vertical=${encodeURIComponent(vertical)}`
          : "/v1/templates";
        const payload = await requestJson(url, { method: "GET" });
        state.templatesSupported = Boolean(payload.templates_supported);
        // Reset and refill the dropdown; preserve the leading "no template" option.
        templateSelectEl.innerHTML = "";
        const defaultOpt = document.createElement("option");
        defaultOpt.value = "";
        defaultOpt.textContent = state.templatesSupported
          ? "no template (vertical default)"
          : `templates not supported by ${payload.vertical || "this vertical"}`;
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
        const verticalName = selectedVerticalName();
        const body = {};
        if (requested) body.session_id = requested;
        if (templateId) body.template_id = templateId;
        if (verticalName) body.vertical = verticalName;
        const payload = await requestJson(
          "/v1/sessions",
          { method: "POST", body: JSON.stringify(body) },
        );
        state.sessionId = payload.session_id;
        state.sessionVertical = payload.vertical;
        const templateLabel = payload.template_id ? ` | template ${payload.template_id}` : "";
        statusEl.textContent = `Session ${state.sessionId} | vertical ${payload.vertical}${templateLabel}`;
        setReady(true);
        addMessage("system", `Created session ${state.sessionId} on vertical ${payload.vertical}${templateLabel}`);
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
        // If the dropdown is currently pointing at the session's
        // vertical, refresh; otherwise the new template lives in
        // a different vertical's subdir and the operator can
        // switch the dropdown to see it.
        if (selectedVerticalName() === state.sessionVertical) {
          await loadTemplates();
        }
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
    switchModelBtn.addEventListener("click", switchModel);
    verticalSelectEl.addEventListener("change", () => { loadTemplates(); });
    outcomeBtns.forEach(btn => btn.addEventListener("click", () => submitOutcome(btn.dataset.kind)));
    inputEl.addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendTurn();
      }
    });
    // ----- Protocol uptake panel -----
    const protocolsBtn = document.getElementById("protocolsBtn");
    const protocolsPanel = document.getElementById("protocolsPanel");
    const protocolsCloseBtn = document.getElementById("protocolsCloseBtn");
    const protocolsStatus = document.getElementById("protocolsStatus");
    const protocolsCandidates = document.getElementById("protocolsCandidates");
    const protocolsApproved = document.getElementById("protocolsApproved");
    const protocolUploadFile = document.getElementById("protocolUploadFile");
    const protocolUploadSeed = document.getElementById("protocolUploadSeed");
    const protocolUploadBtn = document.getElementById("protocolUploadBtn");
    const protocolDescId = document.getElementById("protocolDescId");
    const protocolDescAdvisor = document.getElementById("protocolDescAdvisor");
    const protocolDescText = document.getElementById("protocolDescText");
    const protocolDescBtn = document.getElementById("protocolDescBtn");

    function setProtocolStatus(msg) {
      protocolsStatus.textContent = msg;
    }

    function renderCandidate(item, container) {
      const div = document.createElement("div");
      div.className = "msg system";
      div.style.maxWidth = "100%";
      div.style.marginRight = "0";
      div.style.marginLeft = "0";
      const counts = `boundaries=${item.boundary_count} strategies=${item.strategy_count} `
        + `seeds=${item.knowledge_seed_count} cases=${item.signature_case_count}`;
      div.innerHTML = `<strong>${item.protocol_id}</strong> &middot; ${item.advisor_name || "(no name)"} &middot; `
        + `<em>${item.review_status}</em><br>`
        + `<span style="opacity:.8">${item.description || "(no description)"}</span><br>`
        + `<span style="font-size:12px;opacity:.7">${counts} &middot; src=${item.provenance ? item.provenance.source_kind : item.source_kind}</span>`;
      const btnRow = document.createElement("div");
      btnRow.style.marginTop = "6px";
      btnRow.style.display = "flex";
      btnRow.style.gap = "6px";
      div.appendChild(btnRow);
      container.appendChild(div);
      return btnRow;
    }

    async function loadProtocols() {
      setProtocolStatus("Loading...");
      protocolsCandidates.innerHTML = "";
      protocolsApproved.innerHTML = "";
      try {
        const [pendingPayload, approvedPayload] = await Promise.all([
          requestJson("/v1/protocols/candidates", { method: "GET" }),
          requestJson("/v1/protocols", { method: "GET" }),
        ]);
        const pending = pendingPayload.candidates || [];
        const approved = approvedPayload.protocols || [];
        if (pending.length === 0) {
          protocolsCandidates.innerHTML = "<div class='msg system'>(no pending candidates)</div>";
        } else {
          for (const c of pending) {
            const row = renderCandidate(c, protocolsCandidates);
            const approveBtn = document.createElement("button");
            approveBtn.textContent = "Approve";
            approveBtn.className = "secondary";
            approveBtn.addEventListener("click", () => approveCandidate(c.protocol_id));
            const rejectBtn = document.createElement("button");
            rejectBtn.textContent = "Reject";
            rejectBtn.className = "danger";
            rejectBtn.addEventListener("click", () => rejectCandidate(c.protocol_id));
            row.appendChild(approveBtn);
            row.appendChild(rejectBtn);
          }
        }
        if (approved.length === 0) {
          protocolsApproved.innerHTML = "<div class='msg system'>(no protocols loaded)</div>";
        } else {
          for (const p of approved) {
            const row = renderCandidate(p, protocolsApproved);
            const unloadBtn = document.createElement("button");
            unloadBtn.textContent = "Unload";
            unloadBtn.className = "danger";
            unloadBtn.addEventListener("click", () => unloadProtocol(p.protocol_id));
            row.appendChild(unloadBtn);
          }
        }
        setProtocolStatus(
          `Pending: ${pending.length} | Approved: ${approved.length}. `
          + `Note: approved protocols apply to NEW sessions — restart your session for them to take effect.`
        );
      } catch (err) {
        setProtocolStatus(`Failed to load: ${err.message}`);
      }
    }

    async function approveCandidate(pid) {
      try {
        await requestJson(`/v1/protocols/candidates/${encodeURIComponent(pid)}/approve`, { method: "POST" });
        setProtocolStatus(`Approved ${pid}`);
        await loadProtocols();
      } catch (err) {
        setProtocolStatus(`Approve failed: ${err.message}`);
      }
    }
    async function rejectCandidate(pid) {
      const reason = window.prompt(`Reject reason for ${pid}:`, "") || "";
      try {
        await requestJson(`/v1/protocols/candidates/${encodeURIComponent(pid)}/reject`, {
          method: "POST",
          body: JSON.stringify({ reason }),
        });
        await loadProtocols();
      } catch (err) {
        setProtocolStatus(`Reject failed: ${err.message}`);
      }
    }
    async function unloadProtocol(pid) {
      if (!window.confirm(`Unload ${pid}? This removes its compiled artifacts.`)) return;
      try {
        await requestJson(`/v1/protocols/${encodeURIComponent(pid)}`, { method: "DELETE" });
        await loadProtocols();
      } catch (err) {
        setProtocolStatus(`Unload failed: ${err.message}`);
      }
    }
    async function uploadProtocolFile() {
      const file = protocolUploadFile.files && protocolUploadFile.files[0];
      if (!file) {
        setProtocolStatus("Pick a .pdf / .md / .markdown / .txt file first.");
        return;
      }
      const seed = protocolUploadSeed.value.trim();
      const form = new FormData();
      form.append("file", file, file.name);
      if (seed) form.append("protocol_id_seed", seed);
      const isMarkdown = /\.(md|markdown|txt)$/i.test(file.name);
      const url = isMarkdown ? "/v1/protocols/upload-markdown" : "/v1/protocols/upload-pdf";
      protocolUploadBtn.disabled = true;
      setProtocolStatus(`Uploading ${file.name} ...`);
      try {
        const response = await fetch(url, {
          method: "POST",
          body: form,
          headers: { ...alphaHeaders() },
        });
        const text = await response.text();
        const payload = text ? JSON.parse(text) : {};
        if (!response.ok) {
          throw new Error(`${payload.error || response.status}: ${payload.detail || text}`);
        }
        setProtocolStatus(`Uploaded ${file.name} -> ${payload.protocol_id}`);
        await loadProtocols();
      } catch (err) {
        setProtocolStatus(`Upload failed: ${err.message}`);
      } finally {
        protocolUploadBtn.disabled = false;
      }
    }
    async function submitDescription() {
      const id = protocolDescId.value.trim();
      const advisor = protocolDescAdvisor.value.trim();
      const desc = protocolDescText.value.trim();
      if (!id || !advisor || !desc) {
        setProtocolStatus("Description, protocol_id, advisor_name are all required.");
        return;
      }
      protocolDescBtn.disabled = true;
      setProtocolStatus(`Submitting description for ${id} ...`);
      try {
        const payload = await requestJson("/v1/protocols/from-description", {
          method: "POST",
          body: JSON.stringify({ description: desc, protocol_id: id, advisor_name: advisor }),
        });
        setProtocolStatus(`Submitted ${payload.protocol_id}`);
        protocolDescText.value = "";
        await loadProtocols();
      } catch (err) {
        setProtocolStatus(`Submit failed: ${err.message}`);
      } finally {
        protocolDescBtn.disabled = false;
      }
    }
    if (protocolsBtn) {
      protocolsBtn.addEventListener("click", () => {
        protocolsPanel.hidden = false;
        loadProtocols();
      });
      protocolsCloseBtn.addEventListener("click", () => { protocolsPanel.hidden = true; });
      protocolUploadBtn.addEventListener("click", uploadProtocolFile);
      protocolDescBtn.addEventListener("click", submitDescription);
    }

    loadModels();
    // loadVerticals() chains into loadTemplates() so the template
    // dropdown is filled for whatever vertical was auto-selected.
    loadVerticals();
  </script>
</body>
</html>
"""
