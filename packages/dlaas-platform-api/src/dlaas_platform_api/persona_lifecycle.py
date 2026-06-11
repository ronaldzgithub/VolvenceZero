"""aiohttp routes for the unified persona cognitive-training lifecycle.

Endpoint summary::

    POST /dlaas/v1/personas/{template_id}/lifecycle           — create
    GET  /dlaas/v1/personas/{template_id}/lifecycle           — get + events
    POST /dlaas/v1/personas/{template_id}/lifecycle/advance   — forward move
    POST /dlaas/v1/personas/{template_id}/lifecycle/rollback  — backwards move
    GET  /dlaas/v1/personas/lifecycles                        — list

Auth: operator credentials (``X-Control-Plane-Secret`` or
``X-Service-Secret``) act cross-tenant; tenant credentials
(``X-Tenant-Api-Key`` + ``X-Tenant-Api-Secret``) act only on templates
their tenant owns.

Layering (R8 / R12 / R15): the lifecycle is platform governance — it
records pointers to evidence artifacts (figure bundles, cultivation
rows, exam/interview runs) and gate outcomes. Cognition never lives
here, no learning signal flows back from this surface, and every
transition (including rollback) is an immutable audited event.

Gate evidence is verified, not trusted: advancing to ``exam`` /
``interview`` cross-checks ``exam_run_id`` / ``interview_run_id``
against the registry's eval (``/dlaas/exam_runs``) and interview
(``/dlaas/interview_runs``) persistence — the run must exist, belong
to the same template, be completed, and its recorded outcome must
match the claimed ``passed`` flag (enforced in
``dlaas_platform_registry.persona_lifecycle_store``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    LifecycleTransitionError,
    PersonaLifecycleStage,
    gate_summary_from_events,
)
from dlaas_platform_registry import (
    PersonaLifecycleConflict,
    PersonaLifecycleNotFound,
    PersonaLifecycleStore,
    REGISTRY_APP_KEY,
    Registry,
    TemplateNotFound,
    TemplateStore,
    require_control_plane_or_service,
    require_tenant_auth,
)

_LOG = logging.getLogger("dlaas_platform_api.persona_lifecycle")

PERSONA_LIFECYCLE_BUNDLE_APP_KEY = "dlaas_persona_lifecycle_bundle"


class PersonaLifecycleBundle:
    """Container the api wheel reads to dispatch lifecycle state."""

    __slots__ = ("lifecycles", "templates")

    def __init__(self, *, registry: Registry) -> None:
        self.lifecycles = PersonaLifecycleStore(registry)
        self.templates = TemplateStore(registry)


def attach_persona_lifecycle_routes(
    app: web.Application,
    *,
    registry: Registry,
) -> web.Application:
    if REGISTRY_APP_KEY not in app:
        raise ValueError(
            "attach_persona_lifecycle_routes requires app[REGISTRY_APP_KEY] "
            "(dlaas_platform_api.build_dlaas_app handles this)."
        )
    app[PERSONA_LIFECYCLE_BUNDLE_APP_KEY] = PersonaLifecycleBundle(
        registry=registry
    )
    R = app.router
    # `lifecycles` is registered BEFORE the `{template_id}` wildcard so
    # the list path is not captured as a template id.
    R.add_get("/dlaas/v1/personas/lifecycles", _handle_list)
    R.add_post(
        "/dlaas/v1/personas/{template_id}/lifecycle", _handle_create
    )
    R.add_get("/dlaas/v1/personas/{template_id}/lifecycle", _handle_get)
    R.add_post(
        "/dlaas/v1/personas/{template_id}/lifecycle/advance",
        _handle_advance,
    )
    R.add_post(
        "/dlaas/v1/personas/{template_id}/lifecycle/rollback",
        _handle_rollback,
    )
    return app


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_create(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    template_id = request.match_info["template_id"]
    template = await _resolve_template(bundle, template_id)
    if isinstance(template, web.Response):
        return template
    authorized = await _authorize(request, template_tenant_id=template.tenant_id)
    if isinstance(authorized, web.Response):
        return authorized
    data = await _read_json(request, allow_empty=True)
    try:
        record = await bundle.lifecycles.create(
            template_id=template_id,
            tenant_id=template.tenant_id,
            ai_id=str(data.get("ai_id", "") or ""),
            display_name=str(
                data.get("display_name", "") or template.template_name
            ),
            app_id=str(data.get("app_id", "") or ""),
            notes=str(data.get("notes", "") or ""),
            actor=authorized,
        )
    except PersonaLifecycleConflict as exc:
        return _error(409, "lifecycle_exists", str(exc))
    return web.json_response({"status": "ok", **record.to_json()})


async def _handle_get(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    template_id = request.match_info["template_id"]
    template = await _resolve_template(bundle, template_id)
    if isinstance(template, web.Response):
        return template
    authorized = await _authorize(request, template_tenant_id=template.tenant_id)
    if isinstance(authorized, web.Response):
        return authorized
    try:
        record = await bundle.lifecycles.get_by_template(template_id)
    except PersonaLifecycleNotFound:
        return _error(404, "lifecycle_not_found", template_id)
    events = await bundle.lifecycles.list_events(record.lifecycle_id)
    return web.json_response(
        {
            "status": "ok",
            **record.to_json(),
            "events": [e.to_json() for e in events],
            "gates": gate_summary_from_events(events),
        }
    )


async def _handle_advance(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    template_id = request.match_info["template_id"]
    template = await _resolve_template(bundle, template_id)
    if isinstance(template, web.Response):
        return template
    authorized = await _authorize(request, template_tenant_id=template.tenant_id)
    if isinstance(authorized, web.Response):
        return authorized
    data = await _read_json(request)
    target = _parse_stage(data.get("to_stage"))
    if isinstance(target, web.Response):
        return target
    evidence = data.get("evidence") or {}
    if not isinstance(evidence, Mapping):
        return _error(400, "invalid_evidence", "evidence must be a JSON object")
    try:
        record = await bundle.lifecycles.get_by_template(template_id)
    except PersonaLifecycleNotFound:
        return _error(404, "lifecycle_not_found", template_id)
    try:
        record = await bundle.lifecycles.advance(
            lifecycle_id=record.lifecycle_id,
            target=target,
            evidence=dict(evidence),
            actor=authorized,
        )
    except LifecycleTransitionError as exc:
        return _error(409, "invalid_transition", str(exc))
    events = await bundle.lifecycles.list_events(record.lifecycle_id)
    return web.json_response(
        {
            "status": "ok",
            **record.to_json(),
            "gates": gate_summary_from_events(events),
        }
    )


async def _handle_rollback(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    template_id = request.match_info["template_id"]
    template = await _resolve_template(bundle, template_id)
    if isinstance(template, web.Response):
        return template
    authorized = await _authorize(request, template_tenant_id=template.tenant_id)
    if isinstance(authorized, web.Response):
        return authorized
    data = await _read_json(request)
    target = _parse_stage(data.get("to_stage"))
    if isinstance(target, web.Response):
        return target
    reason = str(data.get("reason", "") or "")
    try:
        record = await bundle.lifecycles.get_by_template(template_id)
    except PersonaLifecycleNotFound:
        return _error(404, "lifecycle_not_found", template_id)
    try:
        record = await bundle.lifecycles.rollback(
            lifecycle_id=record.lifecycle_id,
            target=target,
            reason=reason,
            actor=authorized,
        )
    except LifecycleTransitionError as exc:
        return _error(409, "invalid_transition", str(exc))
    return web.json_response({"status": "ok", **record.to_json()})


async def _handle_list(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    headers = request.headers
    if "X-Control-Plane-Secret" in headers or "X-Service-Secret" in headers:
        require_control_plane_or_service(request)
        tenant_filter = str(request.query.get("tenant_id", "") or "")
    else:
        tenant = await require_tenant_auth(request)
        tenant_filter = tenant.tenant_id
    records = await bundle.lifecycles.list_all(tenant_id=tenant_filter)
    return web.json_response(
        {"status": "ok", "lifecycles": [r.to_json() for r in records]}
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bundle(request: web.Request) -> PersonaLifecycleBundle:
    return request.app[PERSONA_LIFECYCLE_BUNDLE_APP_KEY]


async def _resolve_template(bundle: PersonaLifecycleBundle, template_id: str):
    try:
        return await bundle.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)


async def _authorize(
    request: web.Request, *, template_tenant_id: str
) -> str | web.Response:
    """Authorize the caller; return the actor label for the audit trail.

    Operator secrets act cross-tenant. Tenant credentials are scoped to
    the template's owning tenant — any mismatch is a 403, never a
    silent read.
    """

    headers = request.headers
    if "X-Control-Plane-Secret" in headers or "X-Service-Secret" in headers:
        require_control_plane_or_service(request)
        return "operator"
    tenant = await require_tenant_auth(request)
    if tenant.tenant_id != template_tenant_id:
        return _error(
            403,
            "tenant_mismatch",
            (
                f"authenticated tenant_id={tenant.tenant_id!r} cannot act "
                f"on a template owned by tenant_id={template_tenant_id!r}"
            ),
        )
    return f"tenant:{tenant.tenant_id}"


def _parse_stage(raw: Any) -> PersonaLifecycleStage | web.Response:
    if not isinstance(raw, str) or not raw.strip():
        return _error(400, "missing_to_stage", "to_stage must be a stage name")
    try:
        return PersonaLifecycleStage(raw.strip())
    except ValueError:
        valid = ", ".join(s.value for s in PersonaLifecycleStage)
        return _error(
            400, "invalid_to_stage", f"unknown stage {raw!r}; expected one of: {valid}"
        )


async def _read_json(
    request: web.Request, *, allow_empty: bool = False
) -> Mapping[str, Any]:
    if not request.body_exists:
        if allow_empty:
            return {}
        raise _bad_request("missing_body", "Body required")
    text = await request.text()
    if not text.strip():
        if allow_empty:
            return {}
        raise _bad_request("missing_body", "Empty body")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _bad_request("invalid_json", f"Body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _bad_request("invalid_envelope", "Top-level body must be a JSON object")
    return data


def _bad_request(code: str, detail: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(
        text=json.dumps({"status": "error", "error": code, "detail": detail}),
        content_type="application/json",
    )


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


__all__ = [
    "PERSONA_LIFECYCLE_BUNDLE_APP_KEY",
    "PersonaLifecycleBundle",
    "attach_persona_lifecycle_routes",
]
