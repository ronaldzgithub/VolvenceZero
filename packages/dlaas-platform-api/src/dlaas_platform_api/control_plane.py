"""HTTP routes for the DLaaS control plane (Slice 3 + 4).

Translates ``/dlaas/tenants``, ``/dlaas/shells``, ``/dlaas/assets``,
``/dlaas/templates*``, ``/dlaas/contracts*``, ``/dlaas/adopt`` and the
focus_persons / identity_links endpoints to the typed registry +
launcher API. Authentication is enforced at the handler edge via
``dlaas_platform_registry.auth``:

* Tenant CRUD endpoints require ``X-Control-Plane-Secret``.
* Per-tenant resources (shell / asset / template / contract /
  focus_persons / identity_links) require tenant credentials.
* Adoption requires tenant credentials and validates that the
  authenticated tenant owns the template + shell.

The control plane wheel never invokes the kernel directly. Activate
(Slice 3.4) hands a single ingestion turn to the launcher's
``SessionManager``; readiness reads the resulting activation stats
back from the registry.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    AdoptionConfig,
    ContractStatus,
    FocusPersonSpec,
    IdentityLinkSpec,
    ReadinessReport,
    ShellKind,
    TemplateActivationStatus,
    TemplateSpec,
    TemplateStatus,
)
from dlaas_platform_launcher import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceManager,
)
from dlaas_platform_registry import (
    AssetNotFound,
    AssetStore,
    ContractNotFound,
    ContractStore,
    REGISTRY_APP_KEY,
    Registry,
    ShellNotFound,
    ShellStore,
    TemplateNotFound,
    TemplateStore,
    TenantNotFound,
    TenantStore,
    assert_tenant_id_matches,
    require_control_plane_secret,
    require_tenant_auth,
)
from lifeform_ingestion import (
    IngestionComplianceProfile,
    IngestionPipeline,
    IngestionSourceKind,
    envelope_from_text,
)

_LOG = logging.getLogger("dlaas_platform_api.control_plane")


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


CONTROL_PLANE_STORES_KEY = "dlaas_control_plane_stores"
"""``app[CONTROL_PLANE_STORES_KEY]`` — typed handle to all stores."""


class _Stores:
    """Container for the per-resource stores (assembled from one Registry)."""

    __slots__ = (
        "registry",
        "tenants",
        "shells",
        "assets",
        "templates",
        "contracts",
    )

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.tenants = TenantStore(registry)
        self.shells = ShellStore(registry)
        self.assets = AssetStore(registry)
        self.templates = TemplateStore(registry)
        self.contracts = ContractStore(registry)


def attach_control_plane_routes(
    app: web.Application, *, registry: Registry
) -> web.Application:
    """Register every Slice 3 + 4 route on the given aiohttp app.

    Requires the app to already carry both ``app[REGISTRY_APP_KEY]``
    (PlatformAuthBundle) and ``app[INSTANCE_MANAGER_APP_KEY]``
    (InstanceManager). The api wheel's ``build_dlaas_app`` performs
    this wiring before calling us.
    """
    if REGISTRY_APP_KEY not in app:
        raise ValueError(
            "attach_control_plane_routes requires app[REGISTRY_APP_KEY] "
            "(PlatformAuthBundle). Build the app via "
            "dlaas_platform_api.build_dlaas_app(...)."
        )
    app[CONTROL_PLANE_STORES_KEY] = _Stores(registry)

    R = app.router

    R.add_post("/dlaas/tenants", _handle_create_tenant)
    R.add_get("/dlaas/tenants/{tenant_id}", _handle_get_tenant)

    R.add_post("/dlaas/shells", _handle_upsert_shell)
    R.add_post("/dlaas/register", _handle_upsert_shell)  # compatibility alias

    R.add_post("/dlaas/assets", _handle_create_asset)
    R.add_get("/dlaas/assets/{asset_id}", _handle_get_asset)
    R.add_get("/dlaas/tenants/{tenant_id}/assets", _handle_list_tenant_assets)

    R.add_post("/dlaas/templates", _handle_create_template)
    R.add_get("/dlaas/tenants/{tenant_id}/templates", _handle_list_tenant_templates)
    R.add_get("/dlaas/templates/{template_id}", _handle_get_template)
    R.add_patch("/dlaas/templates/{template_id}", _handle_patch_template)
    R.add_get(
        "/dlaas/templates/{template_id}/versions",
        _handle_list_template_versions,
    )
    R.add_post(
        "/dlaas/templates/{template_id}/snapshot",
        _handle_snapshot_template,
    )
    R.add_post(
        "/dlaas/templates/{template_id}/assets",
        _handle_link_template_asset,
    )
    R.add_get(
        "/dlaas/templates/{template_id}/assets",
        _handle_list_template_assets,
    )
    R.add_post(
        "/dlaas/templates/{template_id}/activate",
        _handle_activate_template,
    )
    R.add_get(
        "/dlaas/templates/{template_id}/readiness",
        _handle_readiness,
    )

    R.add_post("/dlaas/contracts", _handle_create_contract)
    R.add_get("/dlaas/contracts/{contract_id}", _handle_get_contract)
    R.add_get("/dlaas/tenants/{tenant_id}/contracts", _handle_list_tenant_contracts)
    R.add_patch("/dlaas/contracts/{contract_id}", _handle_patch_contract)
    R.add_delete("/dlaas/contracts/{contract_id}", _handle_delete_contract)

    R.add_post("/dlaas/adopt", _handle_adopt)
    R.add_post("/dlaas/v1/adoptions", _handle_adopt)

    R.add_post(
        "/dlaas/instances/{ai_id}/persons",
        _handle_add_focus_persons,
    )
    R.add_get(
        "/dlaas/contracts/{contract_id}/persons",
        _handle_list_focus_persons,
    )
    R.add_post(
        "/dlaas/instances/{ai_id}/identity_links",
        _handle_create_identity_link,
    )
    R.add_post(
        "/dlaas/instances/{ai_id}/identity_links/batch",
        _handle_create_identity_links_batch,
    )
    R.add_get(
        "/dlaas/instances/{ai_id}/identity_links",
        _handle_list_identity_links,
    )
    return app


# ---------------------------------------------------------------------------
# Tenants (control-plane secret)
# ---------------------------------------------------------------------------


async def _handle_create_tenant(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    data = await _read_json(request)
    tenant_name = _required_str(data, "tenant_name")
    contact_email = _required_str(data, "contact_email")
    business_type = str(data.get("business_type", "generic") or "generic")
    billing_plan = str(data.get("billing_plan", "pay_as_you_go") or "pay_as_you_go")
    quota = data.get("quota") or {}
    if not isinstance(quota, Mapping):
        return _error(400, "invalid_quota", "quota must be a JSON object")
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    spec = await stores.tenants.create(
        tenant_name=tenant_name,
        contact_email=contact_email,
        business_type=business_type,
        billing_plan=billing_plan,
        quota=quota,
    )
    payload = {
        "status": "ok",
        **spec.to_json(include_secret=True),
        "platform_endpoint": request.app.get("dlaas_platform_endpoint", ""),
    }
    return web.json_response(payload)


async def _handle_get_tenant(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    tenant_id = request.match_info["tenant_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        spec = await stores.tenants.get(tenant_id)
    except TenantNotFound:
        return _error(404, "tenant_not_found", f"tenant_id={tenant_id!r}")
    return web.json_response({"status": "ok", **spec.to_json()})


# ---------------------------------------------------------------------------
# Shells (tenant auth)
# ---------------------------------------------------------------------------


async def _handle_upsert_shell(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    data = await _read_json(request)
    body_tenant_id = str(data.get("tenant_id", "") or "")
    if body_tenant_id:
        assert_tenant_id_matches(tenant, body_tenant_id)
    shell_id = _required_str(data, "shell_id")
    kind_raw = str(data.get("shell_kind", "deployment") or "deployment").lower()
    try:
        shell_kind = ShellKind(kind_raw)
    except ValueError:
        allowed = ", ".join(k.value for k in ShellKind)
        return _error(
            400, "invalid_shell_kind", f"shell_kind must be one of: {allowed}"
        )
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    spec = await stores.shells.upsert(
        tenant_id=tenant.tenant_id,
        shell_id=shell_id,
        shell_kind=shell_kind,
        shell_type=str(data.get("shell_type", "generic") or "generic"),
        display_name=str(data.get("display_name", "") or ""),
        embodiment=data.get("embodiment") or {},
        channel=data.get("channel") or {},
        scene_meta=data.get("scene_meta") or {},
    )
    capabilities_accepted = sorted(
        k for k in ("perception", "expression", "action") if k in spec.embodiment
    )
    return web.json_response(
        {
            "status": "ok",
            **spec.to_json(),
            "capabilities_accepted": capabilities_accepted,
        }
    )


# ---------------------------------------------------------------------------
# Assets (tenant auth)
# ---------------------------------------------------------------------------


async def _handle_create_asset(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    data = await _read_json(request)
    body_tenant_id = str(data.get("tenant_id", "") or "")
    if body_tenant_id:
        assert_tenant_id_matches(tenant, body_tenant_id)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    spec = await stores.assets.create(
        tenant_id=tenant.tenant_id,
        asset_type=_required_str(data, "asset_type"),
        title=str(data.get("title", "") or ""),
        uri=str(data.get("uri", "") or ""),
        mime_type=str(data.get("mime_type", "") or ""),
        language=str(data.get("language", "") or ""),
        source_meta=data.get("source_meta") or {},
    )
    return web.json_response({"status": "ok", **spec.to_json()})


async def _handle_get_asset(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    asset_id = request.match_info["asset_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        spec = await stores.assets.get(asset_id)
    except AssetNotFound:
        return _error(404, "asset_not_found", asset_id)
    assert_tenant_id_matches(tenant, spec.tenant_id)
    return web.json_response({"status": "ok", **spec.to_json()})


async def _handle_list_tenant_assets(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    tenant_id = request.match_info["tenant_id"]
    assert_tenant_id_matches(tenant, tenant_id)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    specs = await stores.assets.list_for_tenant(tenant_id=tenant_id)
    return web.json_response(
        {
            "status": "ok",
            "tenant_id": tenant_id,
            "assets": [spec.to_json() for spec in specs],
        }
    )


# ---------------------------------------------------------------------------
# Templates (tenant auth)
# ---------------------------------------------------------------------------


async def _handle_create_template(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    data = await _read_json(request)
    body_tenant_id = str(data.get("tenant_id", "") or "")
    if body_tenant_id:
        assert_tenant_id_matches(tenant, body_tenant_id)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    spec = await stores.templates.create(
        tenant_id=tenant.tenant_id,
        template_name=_required_str(data, "template_name"),
        domain=str(data.get("domain", "generic") or "generic"),
        description=str(data.get("description", "") or ""),
        runtime_template_id=str(data.get("runtime_template_id", "") or ""),
        base_persona=data.get("base_persona") or {},
        persona_spec=data.get("persona_spec") or {},
        seed_config=data.get("seed_config") or {},
    )
    return web.json_response(
        {
            "status": "ok",
            "tenant_id": tenant.tenant_id,
            "template_id": spec.template_id,
            "version": spec.current_version,
            **spec.to_json(),
        }
    )


async def _handle_list_tenant_templates(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    tenant_id = request.match_info["tenant_id"]
    assert_tenant_id_matches(tenant, tenant_id)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    specs = await stores.templates.list_for_tenant(tenant_id=tenant_id)
    return web.json_response(
        {
            "status": "ok",
            "tenant_id": tenant_id,
            "templates": [spec.to_json() for spec in specs],
        }
    )


async def _handle_get_template(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        spec = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, spec.tenant_id)
    return web.json_response({"status": "ok", **spec.to_json()})


async def _handle_patch_template(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        existing = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, existing.tenant_id)
    data = await _read_json(request)
    status_value: TemplateStatus | None = None
    if "status" in data:
        try:
            status_value = TemplateStatus(str(data["status"]).lower())
        except ValueError:
            allowed = ", ".join(s.value for s in TemplateStatus)
            return _error(
                400,
                "invalid_template_status",
                f"status must be one of: {allowed}",
            )
    try:
        spec = await stores.templates.patch(
            template_id=template_id,
            template_name=data.get("template_name"),
            description=data.get("description"),
            domain=data.get("domain"),
            runtime_template_id=data.get("runtime_template_id"),
            status=status_value,
            base_persona=data.get("base_persona"),
            persona_spec=data.get("persona_spec"),
            seed_config=data.get("seed_config"),
            version_note=str(data.get("version_note", "") or ""),
        )
    except ValueError as exc:
        return _error(409, "invalid_status_transition", str(exc))
    return web.json_response({"status": "ok", **spec.to_json()})


async def _handle_list_template_versions(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        existing = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, existing.tenant_id)
    versions = await stores.templates.list_versions(template_id=template_id)
    return web.json_response(
        {
            "status": "ok",
            "template_id": template_id,
            "versions": [v.to_json() for v in versions],
        }
    )


async def _handle_snapshot_template(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        existing = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, existing.tenant_id)
    data = await _read_json(request, allow_empty=True)
    note = str(data.get("version_note", "") or "")
    version = await stores.templates.snapshot(
        template_id=template_id, version_note=note
    )
    return web.json_response({"status": "ok", **version.to_json()})


async def _handle_link_template_asset(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        template_spec = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, template_spec.tenant_id)
    data = await _read_json(request)
    asset_id = _required_str(data, "asset_id")
    try:
        asset_spec = await stores.assets.get(asset_id)
    except AssetNotFound:
        return _error(404, "asset_not_found", asset_id)
    assert_tenant_id_matches(tenant, asset_spec.tenant_id)
    link = await stores.assets.link_to_template(
        template_id=template_id,
        asset_id=asset_id,
        template_version=int(
            data.get("template_version", template_spec.current_version) or 1
        ),
        role=str(data.get("role", "training_material") or "training_material"),
        link_meta=data.get("link_meta") or {},
    )
    return web.json_response({"status": "ok", **link.to_json()})


async def _handle_list_template_assets(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        template_spec = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, template_spec.tenant_id)
    links = await stores.assets.list_template_links(template_id=template_id)
    return web.json_response(
        {
            "status": "ok",
            "template_id": template_id,
            "assets": [link.to_json() for link in links],
        }
    )


# ---------------------------------------------------------------------------
# Activate + readiness
# ---------------------------------------------------------------------------


async def _handle_activate_template(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        template = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, template.tenant_id)
    if not template.runtime_template_id.strip():
        return _error(
            422,
            "missing_runtime_template_id",
            "Cannot activate a template without runtime_template_id.",
        )
    instance_manager: InstanceManager = request.app[INSTANCE_MANAGER_APP_KEY]
    data = await _read_json(request, allow_empty=True)
    seed_override = data.get("seed_config_override") or {}
    if not isinstance(seed_override, Mapping):
        return _error(
            400, "invalid_seed_override", "seed_config_override must be an object"
        )

    text = _activation_text(template=template, seed_override=seed_override)
    activation_envelope = envelope_from_text(
        text,
        source_uri=f"dlaas:activation:{template_id}",
        uploader=tenant.tenant_id,
        source_kind=IngestionSourceKind.CORPUS,
        compliance_profile=IngestionComplianceProfile.FORCED,
    )

    await stores.templates.update_activation(
        template_id=template_id,
        activation_status=TemplateActivationStatus.ACTIVATING,
        activation_stats=template.activation_stats,
    )

    pending_ai_id = f"activation:{template_id}"
    try:
        manager = await instance_manager.acquire(
            ai_id=pending_ai_id,
            runtime_template_id=template.runtime_template_id,
        )
    except LookupError as exc:
        await stores.templates.update_activation(
            template_id=template_id,
            activation_status=TemplateActivationStatus.ACTIVATION_FAILED,
            activation_stats={"error": str(exc)},
        )
        return _error(503, "vertical_not_registered", str(exc))

    activation_session_id = (
        f"activation:{template_id}:{int(time.time() * 1000.0)}"
    )
    activation_session = await manager.create_session(
        session_id=activation_session_id
    )
    pipeline = IngestionPipeline()
    try:
        report = await pipeline.process_envelope(
            activation_envelope,
            session=activation_session,
            end_scene_after=True,
            scene_end_reason="dlaas-activation",
            scene_end_drains_slow_loop=True,
        )
    finally:
        await manager.close_session(activation_session_id)

    snapshot_summary = _activation_snapshot_summary(activation_session)
    activation_stats = {
        "envelope_id": activation_envelope.envelope_id,
        "total_chunks": report.total_chunks,
        "processed_chunks": report.processed_chunks,
        "skipped_chunks": report.skipped_chunks,
        "world_nodes": snapshot_summary["world_nodes"],
        "self_nodes": snapshot_summary["self_nodes"],
        "l2_cards": snapshot_summary["l2_cards"],
    }
    final_status = (
        TemplateActivationStatus.ACTIVATED
        if report.processed_chunks > 0
        else TemplateActivationStatus.ACTIVATION_FAILED
    )
    await stores.templates.update_activation(
        template_id=template_id,
        activation_status=final_status,
        activation_stats=activation_stats,
    )
    return web.json_response(
        {
            "status": "ok",
            "template_id": template_id,
            "activation_status": final_status.value,
            "activation_result": {
                "ingestion_report": {
                    "envelope_id": activation_envelope.envelope_id,
                    "total_chunks": report.total_chunks,
                    "processed_chunks": report.processed_chunks,
                    "skipped_chunks": report.skipped_chunks,
                    "all_succeeded": report.all_succeeded,
                },
                "stats": snapshot_summary,
            },
            "stats": snapshot_summary,
        }
    )


async def _handle_readiness(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        template = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, template.tenant_id)
    has_runtime_template_id = bool(template.runtime_template_id.strip())
    is_activated = (
        template.activation_status is TemplateActivationStatus.ACTIVATED
    )
    missing: list[str] = []
    if not has_runtime_template_id:
        missing.append("runtime_template_id")
    if not is_activated:
        missing.append("activation")
    stats = template.activation_stats or {}
    report = ReadinessReport(
        template_id=template_id,
        ready=not missing,
        missing=tuple(missing),
        activation_status=template.activation_status,
        has_runtime_template_id=has_runtime_template_id,
        world_nodes=int(stats.get("world_nodes", 0) or 0),
        self_nodes=int(stats.get("self_nodes", 0) or 0),
        l2_cards=int(stats.get("l2_cards", 0) or 0),
        snapshot_summary=stats,
    )
    return web.json_response(report.to_json())


# ---------------------------------------------------------------------------
# Contracts + Adopt (tenant auth)
# ---------------------------------------------------------------------------


async def _handle_create_contract(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    data = await _read_json(request)
    body_tenant_id = str(data.get("tenant_id", "") or "")
    if body_tenant_id:
        assert_tenant_id_matches(tenant, body_tenant_id)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    template_id = _required_str(data, "template_id")
    shell_id = _required_str(data, "shell_id")
    try:
        template = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, template.tenant_id)
    try:
        shell = await stores.shells.get(
            tenant_id=tenant.tenant_id, shell_id=shell_id
        )
    except ShellNotFound:
        return _error(404, "shell_not_found", shell_id)
    if shell.shell_kind is not ShellKind.DEPLOYMENT:
        return _error(
            409,
            "shell_not_deployable",
            "Only deployment shells can host a runtime contract.",
        )
    contract = await stores.contracts.create(
        tenant_id=tenant.tenant_id,
        template_id=template_id,
        template_version=int(
            data.get("template_version", template.current_version) or 1
        ),
        shell_id=shell_id,
        owner_user_id=str(data.get("owner_user_id", "") or ""),
        engine_tools=data.get("engine_tools") or {},
        tool_policy_snapshot=data.get("tool_policy_snapshot") or {},
        service_contract=data.get("service_contract") or {},
    )
    return web.json_response({"status": "ok", **contract.to_json()})


async def _handle_get_contract(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    contract_id = request.match_info["contract_id"]
    contract = await _resolve_contract_for_tenant(request, tenant, contract_id)
    if isinstance(contract, web.Response):
        return contract
    return web.json_response({"status": "ok", **contract.to_json()})


async def _handle_list_tenant_contracts(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    tenant_id = request.match_info["tenant_id"]
    assert_tenant_id_matches(tenant, tenant_id)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    specs = await stores.contracts.list_for_tenant(tenant_id=tenant_id)
    return web.json_response(
        {
            "status": "ok",
            "tenant_id": tenant_id,
            "contracts": [c.to_json() for c in specs],
        }
    )


async def _handle_patch_contract(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    contract_id = request.match_info["contract_id"]
    contract = await _resolve_contract_for_tenant(request, tenant, contract_id)
    if isinstance(contract, web.Response):
        return contract
    data = await _read_json(request)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    if "contract_status" in data:
        try:
            new_status = ContractStatus(str(data["contract_status"]).lower())
        except ValueError:
            allowed = ", ".join(s.value for s in ContractStatus)
            return _error(
                400,
                "invalid_contract_status",
                f"contract_status must be one of: {allowed}",
            )
        contract = await stores.contracts.update_status(
            contract_id=contract_id, contract_status=new_status
        )
    return web.json_response({"status": "ok", **contract.to_json()})


async def _handle_delete_contract(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    contract_id = request.match_info["contract_id"]
    contract = await _resolve_contract_for_tenant(request, tenant, contract_id)
    if isinstance(contract, web.Response):
        return contract
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    instance_manager: InstanceManager = request.app[INSTANCE_MANAGER_APP_KEY]
    if contract.ai_id:
        await instance_manager.release(contract.ai_id)
    await stores.contracts.delete(contract_id)
    return web.json_response({"status": "ok", "contract_id": contract_id})


async def _handle_adopt(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    data = await _read_json(request)
    body_tenant_id = str(data.get("tenant_id", "") or "")
    if body_tenant_id:
        assert_tenant_id_matches(tenant, body_tenant_id)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    template_id = _required_str(data, "template_id")
    shell_id = _required_str(data, "shell_id")
    try:
        template = await stores.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, template.tenant_id)
    if template.status is not TemplateStatus.PUBLISHED:
        return _error(
            409,
            "template_not_published",
            f"Template {template_id} must be published before adoption.",
        )
    if template.activation_status is not TemplateActivationStatus.ACTIVATED:
        return _error(
            409,
            "template_not_activated",
            f"Template {template_id} must be activated before adoption.",
        )
    try:
        shell = await stores.shells.get(
            tenant_id=tenant.tenant_id, shell_id=shell_id
        )
    except ShellNotFound:
        return _error(404, "shell_not_found", shell_id)
    if shell.shell_kind is not ShellKind.DEPLOYMENT:
        return _error(
            409,
            "shell_not_deployable",
            "Only deployment shells can host a runtime contract.",
        )

    adoption_config = _resolve_adoption_config(data)
    runtime_template_id = (
        adoption_config.vertical.runtime_template_id
        or adoption_config.vertical.vertical_id
        or template.runtime_template_id
    )
    service_contract = dict(data.get("service_contract") or {})
    service_contract["adoption_config"] = adoption_config.to_json()
    service_contract.setdefault(
        "awake_strategy", adoption_config.ops.awake_strategy
    )
    service_contract.setdefault(
        "substrate_profile_id", adoption_config.substrate.substrate_profile_id
    )
    service_contract.setdefault(
        "training_policy", adoption_config.training.to_json()
    )

    instance_manager: InstanceManager = request.app[INSTANCE_MANAGER_APP_KEY]
    contract = await stores.contracts.create(
        tenant_id=tenant.tenant_id,
        template_id=template_id,
        template_version=int(
            data.get("template_version", template.current_version) or 1
        ),
        shell_id=shell_id,
        owner_user_id=str(data.get("owner_user_id", "") or ""),
        engine_tools=data.get("engine_tools") or {},
        tool_policy_snapshot=_compute_tool_policy_snapshot(
            data.get("engine_tools") or {}
        ),
        service_contract=service_contract,
        contract_status=ContractStatus.PROVISIONING,
    )
    final_contract = await stores.contracts.set_ai_id(
        contract_id=contract.contract_id,
        ai_id=None,
        tool_policy_snapshot=contract.tool_policy_snapshot,
    )
    try:
        await instance_manager.acquire(
            ai_id=final_contract.ai_id,
            runtime_template_id=runtime_template_id,
        )
    except LookupError as exc:
        await stores.contracts.update_status(
            contract_id=final_contract.contract_id,
            contract_status=ContractStatus.FAILED,
        )
        return _error(503, "vertical_not_registered", str(exc))

    # Debt #22 closure: when the template names a figure_artifact_id,
    # resolve it through the lifeform-service public surface and
    # bind it to both the session manager (so subsequent sessions
    # carry the bundle to their synthesizer) AND the persona LoRA
    # pool (so the runtime can hot-swap the persona delta when
    # generating). The two helpers are import-guarded so a
    # platform install without the figure-vertical wheel still
    # adopts non-figure templates without crashing.
    # ``TemplateSpec.figure_artifact_id`` is a typed field (default ``""``)
    # in dlaas-platform-contracts; direct access keeps R8 / SSOT.
    figure_artifact_id = template.figure_artifact_id
    if figure_artifact_id:
        try:
            from lifeform_service import (
                lookup_figure_bundle,
                register_bundle_persona_lora,
            )
        except ImportError:
            lookup_figure_bundle = None
            register_bundle_persona_lora = None
        if lookup_figure_bundle is not None:
            bundle = lookup_figure_bundle(
                default=None, bundle_id=figure_artifact_id
            )
            if bundle is not None:
                # Note: the historical ``manager.bind_figure_bundle`` hook
                # was a hasattr-defended dead code path (no such method
                # exists on ``InstanceManager``); removed per
                # no-swallow-errors-no-hasattr-abuse. If a future packet
                # needs to thread the bundle into the launcher session,
                # add a typed method on ``InstanceManager`` and call it
                # directly.
                if register_bundle_persona_lora is not None:
                    register_bundle_persona_lora(bundle)

    persons_registered: list[dict[str, Any]] = []
    for person_payload in data.get("focus_persons") or ():
        try:
            spec = FocusPersonSpec.from_json(
                person_payload, contract_id=final_contract.contract_id
            )
        except ValueError as exc:
            return _error(400, "invalid_focus_person", str(exc))
        await stores.contracts.upsert_focus_person(
            contract_id=final_contract.contract_id,
            person_id=spec.person_id,
            name=spec.name,
            role=spec.role,
            relationship_to_owner=spec.relationship_to_owner,
            age=spec.age,
            initial_profile=spec.initial_profile,
        )
        persons_registered.append(
            {
                "person_id": spec.person_id,
                "name": spec.name,
                "role": spec.role,
                "card_created": True,
            }
        )

    body = final_contract.to_json()
    body.update(
        {
            "ai_id": final_contract.ai_id,
            "contract_id": final_contract.contract_id,
            "tenant_id": tenant.tenant_id,
            "template_id": template_id,
            "instance_endpoint": (
                f"/dlaas/instances/{final_contract.ai_id}/interactions"
            ),
            "v1_instance_endpoint": (
                f"/dlaas/v1/instances/{final_contract.ai_id}/interactions"
            ),
            "instance_token": _instance_token_placeholder(final_contract.ai_id),
            "contract_status": final_contract.contract_status.value,
            "awake_strategy": final_contract.service_contract.get(
                "awake_strategy", "on_demand"
            ),
            "engine_tools": dict(final_contract.engine_tools),
            "tool_policy_snapshot": dict(final_contract.tool_policy_snapshot),
            "adoption_config_version": adoption_config.version,
            "resolved": {
                "vertical": runtime_template_id,
                "substrate_profile_id": adoption_config.substrate.substrate_profile_id,
                "protocol_ids": list(adoption_config.protocols.autoload)
                + list(adoption_config.protocols.library_ids),
                "memory_scope_root": (
                    f"{tenant.tenant_id}/{final_contract.ai_id}"
                    if adoption_config.memory.scope_strategy == "tenant_ai_end_user"
                    else final_contract.ai_id
                ),
                "tool_policy_snapshot_id": f"toolpol:{final_contract.contract_id}:v1",
            },
            "persons_registered": persons_registered,
        }
    )
    return web.json_response(body)


def _resolve_adoption_config(data: Mapping[str, Any]) -> AdoptionConfig:
    raw = dict(data.get("adoption_config") or {})
    blueprint_id = str(data.get("blueprint_id", "") or "")
    if blueprint_id and not raw:
        if blueprint_id == "growth-advisor/cheng-laoshi/private-domain-v1":
            raw = {
                "vertical": {
                    "vertical_id": "growth_advisor",
                    "runtime_template_id": "growth_advisor",
                    "profile_id": "cheng_laoshi",
                },
                "protocols": {"autoload": ["growth_advisor:cheng-laoshi"]},
                "tools": {
                    "tool_policy_id": "growth-advisor-wechat-readonly",
                    "allowed_capabilities": [
                        "text",
                        "handoff_ticket",
                        "reviewed_knowledge",
                    ],
                },
                "ops": {
                    "awake_strategy": "on_demand",
                    "handoff_policy_id": "growth-advisor-standard",
                },
            }
        elif blueprint_id == "companion/default/dev-v1":
            raw = {
                "vertical": {
                    "vertical_id": "companion",
                    "runtime_template_id": "companion",
                }
            }
    overrides = data.get("adoption_overrides") or {}
    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(raw.get(key), Mapping):
                raw[key] = {**dict(raw[key]), **dict(value)}
            else:
                raw[key] = value
    return AdoptionConfig.from_json(raw)


async def _resolve_contract_for_tenant(
    request: web.Request, tenant, contract_id: str
):
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        contract = await stores.contracts.get(contract_id)
    except ContractNotFound:
        return _error(404, "contract_not_found", contract_id)
    assert_tenant_id_matches(tenant, contract.tenant_id)
    return contract


# ---------------------------------------------------------------------------
# Slice 4: focus_persons + identity_links
# ---------------------------------------------------------------------------


async def _handle_add_focus_persons(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    ai_id = request.match_info["ai_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    data = await _read_json(request)
    contract_id = _required_str(data, "contract_id")
    contract = await _resolve_contract_for_tenant(request, tenant, contract_id)
    if isinstance(contract, web.Response):
        return contract
    if contract.ai_id != ai_id:
        return _error(
            403,
            "ai_id_mismatch",
            f"contract_id={contract_id!r} is not adopted as ai_id={ai_id!r}",
        )
    persons_payload = data.get("focus_persons") or ()
    if not isinstance(persons_payload, list):
        return _error(400, "invalid_focus_persons", "focus_persons must be a list")
    registered: list[FocusPersonSpec] = []
    for raw in persons_payload:
        try:
            spec = FocusPersonSpec.from_json(raw, contract_id=contract_id)
        except ValueError as exc:
            return _error(400, "invalid_focus_person", str(exc))
        upserted = await stores.contracts.upsert_focus_person(
            contract_id=contract_id,
            person_id=spec.person_id,
            name=spec.name,
            role=spec.role,
            relationship_to_owner=spec.relationship_to_owner,
            age=spec.age,
            initial_profile=spec.initial_profile,
        )
        registered.append(upserted)
        instance_manager: InstanceManager = request.app[INSTANCE_MANAGER_APP_KEY]
        if instance_manager.has(ai_id):
            manager = instance_manager.get(ai_id)
            session_id = (
                f"focus_person:{spec.person_id}:{int(time.time() * 1000.0)}"
            )
            session = await manager.create_session(session_id=session_id)
            try:
                relationship = (
                    f"focus_person:{spec.person_id} role={spec.role!r} "
                    f"name={spec.name!r}"
                )
                session.submit_profile_event(
                    event_id=f"register:{spec.person_id}",
                    source="dlaas-platform",
                    relationship_note=relationship,
                    confidence=0.85,
                )
            finally:
                await manager.close_session(session_id)
    return web.json_response(
        {
            "status": "ok",
            "ai_id": ai_id,
            "contract_id": contract_id,
            "persons_registered": [s.to_json() for s in registered],
        }
    )


async def _handle_list_focus_persons(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    contract_id = request.match_info["contract_id"]
    contract = await _resolve_contract_for_tenant(request, tenant, contract_id)
    if isinstance(contract, web.Response):
        return contract
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    persons = await stores.contracts.list_focus_persons(contract_id=contract_id)
    return web.json_response(
        {
            "status": "ok",
            "contract_id": contract_id,
            "focus_persons": [p.to_json() for p in persons],
        }
    )


async def _handle_create_identity_link(request: web.Request) -> web.Response:
    await require_tenant_auth(request)
    ai_id = request.match_info["ai_id"]
    data = await _read_json(request)
    try:
        spec = IdentityLinkSpec.from_json(data, ai_id=ai_id)
    except ValueError as exc:
        return _error(400, "invalid_identity_link", str(exc))
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    persisted = await stores.contracts.upsert_identity_link(
        ai_id=spec.ai_id,
        channel_type=spec.channel_type,
        channel_ref=spec.channel_ref,
        canonical_end_user_ref=spec.canonical_end_user_ref,
        link_meta=spec.link_meta,
    )
    return web.json_response({"status": "ok", **persisted.to_json()})


async def _handle_create_identity_links_batch(
    request: web.Request,
) -> web.Response:
    await require_tenant_auth(request)
    ai_id = request.match_info["ai_id"]
    data = await _read_json(request)
    raw_links = data.get("links") or ()
    if not isinstance(raw_links, list):
        return _error(400, "invalid_links", "links must be a list")
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    persisted: list[IdentityLinkSpec] = []
    for raw in raw_links:
        try:
            spec = IdentityLinkSpec.from_json(raw, ai_id=ai_id)
        except ValueError as exc:
            return _error(400, "invalid_identity_link", str(exc))
        persisted.append(
            await stores.contracts.upsert_identity_link(
                ai_id=spec.ai_id,
                channel_type=spec.channel_type,
                channel_ref=spec.channel_ref,
                canonical_end_user_ref=spec.canonical_end_user_ref,
                link_meta=spec.link_meta,
            )
        )
    return web.json_response(
        {
            "status": "ok",
            "ai_id": ai_id,
            "links": [s.to_json() for s in persisted],
        }
    )


async def _handle_list_identity_links(request: web.Request) -> web.Response:
    await require_tenant_auth(request)
    ai_id = request.match_info["ai_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    links = await stores.contracts.list_identity_links(ai_id=ai_id)
    return web.json_response(
        {
            "status": "ok",
            "ai_id": ai_id,
            "links": [link.to_json() for link in links],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _activation_text(
    *,
    template: TemplateSpec,
    seed_override: Mapping[str, Any],
    asset_fetcher_callable: Any = None,
    template_assets: tuple[Any, ...] = (),
) -> str:
    """Build an activation corpus from template fields (debt #15 injection).

    SHADOW behaviour (default): the corpus is small and structured —
    persona name, role archetype, speaking style, value boundaries,
    background story, plus the seed config + override. Suitable for
    smoke / dev where the apprentice memory just needs to move off
    zero.

    ``asset_fetcher_callable`` (debt #15): when supplied, the
    callable receives ``(template_id, template_assets)`` and returns
    a list of fetched corpus chunks (typically pulled via HTTP from
    each asset's ``uri`` and parsed with the appropriate content-type
    handler — PDF / HTML / markdown / txt). Returned chunks are
    appended to the structured persona block so the apprentice's
    memory actually contains real reviewer-curated content rather
    than persona metadata alone. ``None`` keeps the persona-only
    path (back-compat with Slice 7 behaviour).
    """
    persona = template.persona_spec or {}
    seed = dict(template.seed_config)
    seed.update(seed_override)
    parts: list[str] = []
    parts.append(f"Template: {template.template_name}")
    parts.append(f"Domain: {template.domain}")
    if template.description:
        parts.append(f"Description: {template.description}")
    if persona:
        parts.append("Persona:")
        for key in (
            "display_name",
            "role_archetype",
            "speaking_style",
            "background_story",
        ):
            value = persona.get(key)
            if value:
                parts.append(f"  {key}: {value}")
        boundaries = persona.get("value_boundaries") or ()
        if isinstance(boundaries, list) and boundaries:
            parts.append("  value_boundaries:")
            for line in boundaries:
                parts.append(f"    - {line}")
    if seed:
        parts.append(f"Seed config: {json.dumps(seed, sort_keys=True)}")
    parts.append(f"Runtime template id: {template.runtime_template_id}")
    if asset_fetcher_callable is not None and template_assets:
        try:
            chunks = asset_fetcher_callable(
                template_id=template.template_id,
                template_assets=template_assets,
            )
        except Exception as exc:  # noqa: BLE001 — surface caller failure as fail-loud
            raise RuntimeError(
                f"asset_fetcher_callable raised on template "
                f"{template.template_id!r}: {exc}"
            ) from exc
        if chunks:
            parts.append("Asset corpus:")
            for idx, chunk in enumerate(chunks):
                if isinstance(chunk, str) and chunk.strip():
                    parts.append(f"  [chunk-{idx:02d}] {chunk.strip()}")
    return "\n\n".join(parts)


def _activation_snapshot_summary(session) -> dict[str, int]:
    """Placeholder kernel-counter readout for the activation report.

    R8 / SSOT note: ``world_nodes`` / ``self_nodes`` / ``l2_cards`` are
    not fields on ``vz-memory``'s ``MemorySnapshot`` (verified
    2026-05-14); the previous getattr / hasattr / except-AttributeError
    cascade silently returned all-zero counters anyway. Until
    ``LifeformSession`` exposes a typed kernel-counter API the platform
    can consume, return zeros directly. Field names preserved to keep
    ``dlaas-platform-contracts.ReadinessReport`` wire-compatible.
    """
    del session  # placeholder; see TODO above
    return {"world_nodes": 0, "self_nodes": 0, "l2_cards": 0}


def _compute_tool_policy_snapshot(engine_tools: Mapping[str, Any]) -> dict[str, Any]:
    enabled = []
    for name, value in engine_tools.items():
        if isinstance(value, bool) and value:
            enabled.append(name)
            continue
        if isinstance(value, Mapping) and bool(value.get("enabled", False)):
            enabled.append(name)
    snapshot = dict(engine_tools)
    snapshot["enabled_capabilities"] = enabled
    return snapshot


def _instance_token_placeholder(ai_id: str) -> str:
    """Return a deterministic placeholder instance token.

    Slice 5 / Slice 6 will issue real signed tokens; the placeholder
    surfaces the ai_id so integration tests can detect it without
    leaking entropy. The README explicitly notes the platform-side
    runtime endpoints authenticate with tenant credentials, so the
    placeholder is acceptable here.
    """
    return f"inst_tok_{ai_id[-12:]}"


# ---------------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------------


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str) or not value.strip():
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "missing_field",
                    "detail": f"Required field {key!r} must be a non-empty string.",
                }
            ),
            content_type="application/json",
        )
    return value


async def _read_json(
    request: web.Request, *, allow_empty: bool = False
) -> dict[str, Any]:
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
    except (web.HTTPException, OSError) as exc:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "invalid_body",
                    "detail": f"Could not read body: {exc}",
                }
            ),
            content_type="application/json",
        ) from exc
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


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


__all__ = [
    "CONTROL_PLANE_STORES_KEY",
    "attach_control_plane_routes",
]
