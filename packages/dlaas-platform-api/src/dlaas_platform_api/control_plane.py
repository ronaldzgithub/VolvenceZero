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
import os
import pathlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from aiohttp import web

from dlaas_platform_contracts import (
    AdoptionConfig,
    CitationPolicy,
    ContractStatus,
    CoveragePolicy,
    FocusPersonSpec,
    IdentityLinkSpec,
    PluginManifest,
    ReadinessReport,
    ShellKind,
    TemplateActivationStatus,
    TemplateSpec,
    TemplateStatus,
    compute_plugin_tool_policy_snapshot,
)
from dlaas_platform_launcher import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceManager,
)
from dlaas_platform_launcher.instance_manager import InstanceNotFound
from dlaas_platform_registry import (
    ApplicationNotFound,
    ApplicationStore,
    AssetNotFound,
    AssetStore,
    ContractNotFound,
    ContractStore,
    PersonaLifecycleStore,
    REGISTRY_APP_KEY,
    Registry,
    ShellNotFound,
    ShellStore,
    TemplateNotFound,
    TemplateStore,
    TenantNotFound,
    TenantStore,
    assert_tenant_id_matches,
    merge_plugins_from_applications,
    require_application_auth,
    require_control_plane_or_service,
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


BindReason = Literal[
    "ok",
    "empty_figure_artifact_id",
    "lifeform_service_absent",
    "bundle_not_found",
    "instance_not_found",
    "substrate_incompatible",
]


@dataclass(frozen=True)
class BindResult:
    """Outcome of :func:`bind_figure_artifact_to_ai_id`.

    The previous bool return forced the caller to either swallow
    failures (which defeated the trust contract — adopt would return
    200 even when the bundle was missing) or to mirror the helper's
    internal logic to figure out why the bool was False.

    This typed result lets adopt / wake map every documented branch
    to a typed HTTP error:

    * ``ok`` — figure_artifact_id resolved + bound; happy path.
    * ``empty_figure_artifact_id`` — template carries no bundle id;
      the session falls back to the global default bundle (also a
      happy path for non-figure templates).
    * ``lifeform_service_absent`` — figure-vertical wheel not
      installed on this dlaas-platform deploy; non-figure tenants
      continue to work, figure-tenants get a typed 503.
    * ``bundle_not_found`` — template names a bundle that the
      in-process FigureBundleStore has not registered. Caller surfaces
      as 503 ``figure_bundle_not_registered``; operator needs to run
      U8 rescan (``POST /dlaas/control/figure-bundles/rescan``) and
      check FIGURE_BUNDLE_ROOT mount.
    * ``instance_not_found`` — ``ai_id`` not present on the launcher;
      caller surfaces as 503 ``ai_id_not_acquired`` — adopt/wake
      should have acquired the instance moments ago, so this points
      at a launcher / lifecycle race condition.
    """

    bound: bool
    reason: BindReason


def bind_figure_artifact_to_ai_id(
    instance_manager: InstanceManager,
    ai_id: str,
    figure_artifact_id: str,
    *,
    adapter_policy: str = "persona_lora",
) -> BindResult:
    """Resolve ``figure_artifact_id`` + bind to ``ai_id``'s SessionManager.

    Shared by both the **adopt** path (U2 — full tenant onboarding) and
    the **wake** path (U5 — bake-worker / operator lazy materialize),
    so the bind contract is identical regardless of how the ai_id was
    brought up.

    Returns:
        BindResult — :attr:`BindResult.bound` is True only when a real
        bundle was resolved and bound. Every False branch carries a
        typed :attr:`BindResult.reason` enabling the caller to surface
        a typed HTTP error rather than collapsing them all to 200.

    Does not raise: ``FigureBundleNotFound`` / ``InstanceNotFound``
    are captured into BindResult reasons so the caller controls the
    HTTP mapping in one place.
    """

    if not figure_artifact_id:
        return BindResult(bound=False, reason="empty_figure_artifact_id")
    try:
        from lifeform_service import (  # noqa: PLC0415
            FigureBundleNotFound,
            lookup_figure_bundle,
        )
    except ImportError:
        return BindResult(bound=False, reason="lifeform_service_absent")
    try:
        bundle = lookup_figure_bundle(default=None, bundle_id=figure_artifact_id)
    except FigureBundleNotFound:
        # The defaulting overload above should never raise — but if
        # the lifeform-service API contract drifts, treat it the same
        # as bundle == None: a typed not-found rather than a silent
        # default.
        return BindResult(bound=False, reason="bundle_not_found")
    if bundle is None:
        return BindResult(bound=False, reason="bundle_not_found")
    try:
        session_manager = instance_manager.get(ai_id)
    except InstanceNotFound:
        return BindResult(bound=False, reason="instance_not_found")
    # Substrate-upgrade-protocol guard: a bundle whose L2 LoRA was baked
    # against specific substrate weights declares ``compatible_substrates``;
    # reject binding it onto a substrate it was not baked for (fail loud
    # rather than silently serving a mismatched adapter).
    if not _bundle_substrate_compatible(
        bundle=bundle, runtime=instance_manager.substrate_runtime
    ):
        return BindResult(bound=False, reason="substrate_incompatible")
    # ``adapter_policy`` is the resolved substrate profile policy; only
    # ``persona_lora`` permits the figure LoRA overlay. Anything else
    # (notably ``none``) binds the bundle for L1/L3/L4 enforcement but
    # disables the L2 persona-LoRA activation at the runtime (R10).
    #
    # ``bind_figure_bundle`` registers the bundle's LoRA into the
    # SessionManager's own *scoped* pool (isolated per ai_id) so two
    # tenants adopting different bundles for the same figure_id no
    # longer collide in the process-wide default pool.
    persona_lora_enabled = adapter_policy == "persona_lora"
    session_manager.bind_figure_bundle(
        bundle, persona_lora_enabled=persona_lora_enabled
    )
    return BindResult(bound=True, reason="ok")


_BIND_REASON_TO_ERROR_CODE: dict[BindReason, str] = {
    "ok": "ok",
    "empty_figure_artifact_id": "no_figure_artifact_id",
    "lifeform_service_absent": "figure_vertical_unavailable",
    "bundle_not_found": "figure_bundle_not_registered",
    "instance_not_found": "ai_id_not_acquired",
    "substrate_incompatible": "figure_bundle_substrate_incompatible",
}


def _bundle_substrate_compatible(*, bundle: object, runtime: object) -> bool:
    """Whether ``bundle`` may run on the currently-loaded substrate.

    Returns True when the bundle declares no ``compatible_substrates``
    (legacy / unconstrained), when the running substrate is synthetic
    (``runtime is None`` — dev path), when the bundle carries a legacy
    fingerprint sentinel, or when one declared fingerprint's ``model_id``
    matches the running substrate. Otherwise False (caller fails loud).
    """

    compatible = getattr(bundle, "compatible_substrates", ())
    if not compatible:
        return True
    if runtime is None:
        return True
    running_model_id = getattr(runtime, "model_id", "")
    for fingerprint in compatible:
        is_legacy = getattr(fingerprint, "is_legacy", None)
        if callable(is_legacy) and is_legacy():
            return True
        if getattr(fingerprint, "model_id", "") == running_model_id:
            return True
    return False


def _bind_failure_detail(
    figure_artifact_id: str, ai_id: str, reason: BindReason
) -> str:
    """Operator-friendly diagnostic for a failed bundle bind.

    Per-reason hint guides recovery: the U8 rescan endpoint for
    bundle_not_found; mount + module install check for
    lifeform_service_absent; launcher race for instance_not_found.
    """

    if reason == "bundle_not_found":
        return (
            f"figure_artifact_id={figure_artifact_id!r} is not registered "
            f"in the in-process FigureBundleStore. The bake-worker writes "
            f"bundles under FIGURE_BUNDLE_ROOT; run "
            f"POST /dlaas/control/figure-bundles/rescan to refresh the "
            f"in-process registry, and verify FIGURE_BUNDLE_ROOT is "
            f"mounted on this dlaas-platform pod."
        )
    if reason == "lifeform_service_absent":
        return (
            "lifeform-service wheel is not installed on this dlaas-platform; "
            "figure-vertical (and therefore figure-bundle binding) is "
            "unavailable. Operator: install lifeform-service or stop "
            "minting templates with figure_artifact_id."
        )
    if reason == "instance_not_found":
        return (
            f"ai_id={ai_id!r} is not registered on the InstanceManager. "
            "The lifecycle path that acquired this ai_id raced with bind; "
            "retry adopt/wake."
        )
    if reason == "substrate_incompatible":
        return (
            f"figure_artifact_id={figure_artifact_id!r} declares "
            "compatible_substrates that do not include the substrate this "
            "pod loaded. The L2 LoRA was baked against different substrate "
            "weights; re-bake the bundle against the running substrate "
            "(substrate-upgrade-protocol) or route this ai_id to a pod "
            "running a compatible substrate."
        )
    return f"reason={reason}"


class _Stores:
    """Container for the per-resource stores (assembled from one Registry)."""

    __slots__ = (
        "registry",
        "tenants",
        "shells",
        "assets",
        "templates",
        "contracts",
        "applications",
        "lifecycles",
    )

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.tenants = TenantStore(registry)
        self.shells = ShellStore(registry)
        self.assets = AssetStore(registry)
        self.templates = TemplateStore(registry)
        self.contracts = ContractStore(registry)
        self.applications = ApplicationStore(registry)
        self.lifecycles = PersonaLifecycleStore(registry)


# ---------------------------------------------------------------------------
# Applications (Packet 3: plugin foundation)
# ---------------------------------------------------------------------------


async def _handle_create_application(request: web.Request) -> web.Response:
    """Provision a new application bundle (control-plane operation).

    Body: ``{ "name": "...", "version": "...", "description": "...",
              "plugins": [PluginManifest...] }``

    Returns the freshly issued application_id + api_key + plaintext
    api_secret (the latter exactly once).
    """

    require_control_plane_secret(request)
    data = await _read_json(request)
    name = _required_str(data, "name")
    version = str(data.get("version", "0.0.0") or "0.0.0")
    description = str(data.get("description", "") or "")
    # Pre-allocate the application id so portal-authored safety manifests
    # can be materialised under a stable per-application directory before
    # the row is persisted.
    from dlaas_platform_registry.secrets import fresh_application_id

    application_id = fresh_application_id()
    try:
        plugins_payload = _materialize_inline_safety(
            data.get("plugins"), application_id=application_id
        )
        plugins = _parse_inline_plugins(plugins_payload)
    except ValueError as exc:
        return _error(400, "invalid_plugin_manifest", str(exc))
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    spec = await stores.applications.create(
        name=name,
        version=version,
        description=description,
        plugins=plugins,
        application_id=application_id,
    )
    return web.json_response({"status": "ok", **spec.to_json()})


async def _handle_get_application(request: web.Request) -> web.Response:
    """Return one application spec. Visible to tenants + control plane."""

    if "X-Control-Plane-Secret" in request.headers:
        require_control_plane_secret(request)
    else:
        await require_tenant_auth(request)
    application_id = request.match_info["application_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        spec = await stores.applications.get(application_id)
    except ApplicationNotFound:
        return _error(404, "application_not_found", application_id)
    return web.json_response({"status": "ok", **spec.to_json()})


async def _handle_list_applications(request: web.Request) -> web.Response:
    """List every registered application (catalog view).

    Tenants see the catalog to choose which ones to approve. The
    response does NOT include any approval status; tenants combine
    this with ``GET /dlaas/tenants/{id}/applications`` (Packet 4
    portal UI) to render an "approved / available" split.
    """

    if "X-Control-Plane-Secret" in request.headers:
        require_control_plane_secret(request)
    else:
        await require_tenant_auth(request)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    specs = await stores.applications.list()
    return web.json_response(
        {
            "status": "ok",
            "applications": [spec.to_json() for spec in specs],
        }
    )


async def _handle_update_application(request: web.Request) -> web.Response:
    """Application owner updates plugins / version (self-service).

    Authenticates via ``X-Application-Api-Key`` + ``X-Application-Api-Secret``.
    The request path's ``application_id`` MUST match the authenticated
    application; cross-app updates are 403.
    """

    application = await require_application_auth(request)
    application_id = request.match_info["application_id"]
    if application_id != application.application_id:
        return _error(
            403,
            "application_mismatch",
            (
                f"authenticated application_id="
                f"{application.application_id!r} cannot update "
                f"application_id={application_id!r}"
            ),
        )
    data = await _read_json(request)
    try:
        plugins = _parse_inline_plugins(data.get("plugins"))
    except ValueError as exc:
        return _error(400, "invalid_plugin_manifest", str(exc))
    version = data.get("version")
    description = data.get("description")
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    updated = await stores.applications.update_plugins(
        application_id=application_id,
        plugins=plugins,
        version=str(version) if version is not None else None,
        description=str(description) if description is not None else None,
    )
    return web.json_response({"status": "ok", **updated.to_json()})


async def _handle_approve_application(request: web.Request) -> web.Response:
    """Tenant approves an application's plugin bundle."""

    tenant = await require_tenant_auth(request)
    tenant_id = request.match_info["tenant_id"]
    assert_tenant_id_matches(tenant, tenant_id)
    application_id = request.match_info["application_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        await stores.applications.get(application_id)
    except ApplicationNotFound:
        return _error(404, "application_not_found", application_id)
    data = await _read_json(request, allow_empty=True)
    approved_by = str(data.get("approved_by_user_id", "") or "")
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None
    approval = await stores.applications.approve(
        tenant_id=tenant.tenant_id,
        application_id=application_id,
        approved_by_user_id=approved_by,
        metadata=metadata,
    )
    return web.json_response({"status": "ok", **approval.to_json()})


async def _handle_revoke_application_approval(
    request: web.Request,
) -> web.Response:
    """Tenant revokes a prior application approval.

    Already-adopted contracts keep their frozen
    :attr:`ContractSpec.plugins` snapshot; future adopts skip this
    application.
    """

    tenant = await require_tenant_auth(request)
    tenant_id = request.match_info["tenant_id"]
    assert_tenant_id_matches(tenant, tenant_id)
    application_id = request.match_info["application_id"]
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    removed = await stores.applications.revoke_approval(
        tenant_id=tenant.tenant_id, application_id=application_id
    )
    if not removed:
        return _error(404, "application_approval_not_found", application_id)
    return web.json_response(
        {
            "status": "ok",
            "tenant_id": tenant.tenant_id,
            "application_id": application_id,
        }
    )


async def _handle_list_tenant_applications(
    request: web.Request,
) -> web.Response:
    """List every application this tenant has approved."""

    tenant = await require_tenant_auth(request)
    tenant_id = request.match_info["tenant_id"]
    assert_tenant_id_matches(tenant, tenant_id)
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    approved = await stores.applications.list_approved_applications_for_tenant(
        tenant_id=tenant.tenant_id
    )
    return web.json_response(
        {
            "status": "ok",
            "tenant_id": tenant.tenant_id,
            "applications": [spec.to_json() for spec in approved],
        }
    )


async def _resolve_plugins_from_application_ids(
    stores: _Stores,
    *,
    tenant_id: str,
    application_ids: list[str],
) -> tuple[list[PluginManifest], web.Response | None]:
    """Resolve ``application_ids`` → merged plugin manifests.

    Validates each application exists, is approved by ``tenant_id``,
    and that no two applications declare the same plugin name. On
    any of those failures returns a (None, web.Response) pair that
    the caller propagates as the adopt 4xx error.
    """

    specs: list = []
    for app_id in application_ids:
        try:
            app_spec = await stores.applications.get(app_id)
        except ApplicationNotFound:
            return (
                [],
                _error(404, "application_not_found", app_id),
            )
        approval = await stores.applications.get_approval(
            tenant_id=tenant_id, application_id=app_id
        )
        if approval is None:
            return (
                [],
                _error(
                    409,
                    "application_not_approved",
                    (
                        f"application_id={app_id!r} is not approved by "
                        f"tenant_id={tenant_id!r}; call "
                        f"POST /dlaas/tenants/{tenant_id}/applications/{app_id}/approve "
                        "first."
                    ),
                ),
            )
        specs.append(app_spec)
    try:
        merged = merge_plugins_from_applications(specs)
    except ValueError as exc:
        return ([], _error(409, "plugin_name_conflict", str(exc)))
    return (list(merged), None)


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
    # U7 (family-memorial enabler): operator/control-plane variant of
    # template create. Uses control_plane_secret auth and accepts an
    # explicit ``tenant_id`` in the body. Operator services such as
    # ``family-bake-worker`` cannot mint tenant API keys for the
    # operator-owned tenant (e.g. ``family_memorial_operator``) and
    # would otherwise need to do a tenant bootstrap on every deploy.
    R.add_post(
        "/dlaas/control/templates",
        _handle_control_plane_template_create,
    )
    # U8 (family-memorial enabler): trigger an in-process re-scan of
    # FIGURE_BUNDLE_ROOT so a freshly-baked bundle is picked up by
    # the platform's FigureBundleStore without a process restart. The
    # operator alternative — restarting dlaas-platform after every
    # bake — drops live sessions and is not acceptable in production.
    R.add_post(
        "/dlaas/control/figure-bundles/rescan",
        _handle_control_plane_rescan_bundles,
    )
    # U9 (digital-employee enabler): read-only catalog of every
    # registered figure bundle so per-tenant apps (digital-employee
    # admin UI in particular) can render a "public persona library"
    # picker without forcing operators to embed the FIGURE_BUNDLE_ROOT
    # contents in env / config.
    R.add_get(
        "/dlaas/control/figure-bundles",
        _handle_control_plane_list_bundles,
    )
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

    R.add_post("/dlaas/applications", _handle_create_application)
    R.add_get("/dlaas/applications", _handle_list_applications)
    R.add_get(
        "/dlaas/applications/{application_id}", _handle_get_application
    )
    R.add_put(
        "/dlaas/applications/{application_id}", _handle_update_application
    )
    R.add_post(
        "/dlaas/tenants/{tenant_id}/applications/{application_id}/approve",
        _handle_approve_application,
    )
    R.add_delete(
        "/dlaas/tenants/{tenant_id}/applications/{application_id}/approve",
        _handle_revoke_application_approval,
    )
    R.add_get(
        "/dlaas/tenants/{tenant_id}/applications",
        _handle_list_tenant_applications,
    )

    # Save a LIVE adopted instance as a NEW DLaaS template ("另存为模板").
    # Operator-only, flag-gated, owner-clean compose of TemplateStore +
    # PersonaLifecycleStore (+ the instance's self-learned protocol bundle
    # when present). See `_handle_export_instance_template`.
    R.add_post(
        "/dlaas/v1/instances/{ai_id}/export-template",
        _handle_export_instance_template,
    )
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
    # U3 (family-memorial enabler): figure_artifact_id /
    # citation_policy / coverage_policy / figure_time_window
    # passthrough. The registry has supported these fields since
    # debt #22 baked them into TemplateSpec; the HTTP handler was
    # silently dropping them, forcing operators to use the registry
    # store directly. With this pass-through, family-memorial (and
    # any future figure-vertical product) can mint templates over
    # plain HTTP.
    try:
        citation_policy = _parse_citation_policy(data.get("citation_policy"))
        coverage_policy = _parse_coverage_policy(data.get("coverage_policy"))
    except ValueError as exc:
        return _error(400, "invalid_template_policy", str(exc))
    spec = await stores.templates.create(
        tenant_id=tenant.tenant_id,
        template_name=_required_str(data, "template_name"),
        domain=str(data.get("domain", "generic") or "generic"),
        description=str(data.get("description", "") or ""),
        runtime_template_id=str(data.get("runtime_template_id", "") or ""),
        base_persona=data.get("base_persona") or {},
        persona_spec=data.get("persona_spec") or {},
        seed_config=data.get("seed_config") or {},
        figure_artifact_id=str(data.get("figure_artifact_id", "") or ""),
        citation_policy=citation_policy,
        coverage_policy=coverage_policy,
        figure_time_window=str(data.get("figure_time_window", "") or ""),
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


async def _handle_control_plane_template_create(
    request: web.Request,
) -> web.Response:
    """U7: operator-side template create authorised by control-plane secret.

    Used by services (e.g. ``docker/family-bake-worker``) that need
    to mint templates against a static "operator" tenant without
    holding tenant API keys. The body MUST include ``tenant_id``
    explicitly; the registry's tenant store is consulted to confirm
    the tenant exists before delegating to
    ``stores.templates.create``.

    Apart from the auth surface and the explicit ``tenant_id``, the
    request shape and the response shape mirror
    :func:`_handle_create_template` so client code can share the
    payload-building logic.
    """

    require_control_plane_secret(request)
    data = await _read_json(request)
    tenant_id = _required_str(data, "tenant_id")
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        await stores.tenants.get(tenant_id)
    except TenantNotFound:
        return _error(404, "tenant_not_found", f"tenant_id={tenant_id!r}")
    try:
        citation_policy = _parse_citation_policy(data.get("citation_policy"))
        coverage_policy = _parse_coverage_policy(data.get("coverage_policy"))
    except ValueError as exc:
        return _error(400, "invalid_template_policy", str(exc))
    spec = await stores.templates.create(
        tenant_id=tenant_id,
        template_name=_required_str(data, "template_name"),
        domain=str(data.get("domain", "generic") or "generic"),
        description=str(data.get("description", "") or ""),
        runtime_template_id=str(data.get("runtime_template_id", "") or ""),
        base_persona=data.get("base_persona") or {},
        persona_spec=data.get("persona_spec") or {},
        seed_config=data.get("seed_config") or {},
        figure_artifact_id=str(data.get("figure_artifact_id", "") or ""),
        citation_policy=citation_policy,
        coverage_policy=coverage_policy,
        figure_time_window=str(data.get("figure_time_window", "") or ""),
    )
    return web.json_response(
        {
            "status": "ok",
            "tenant_id": tenant_id,
            "template_id": spec.template_id,
            "version": spec.current_version,
            **spec.to_json(),
        }
    )


def _instance_export_template_enabled() -> bool:
    """Whether ``POST /instances/{ai_id}/export-template`` is enabled.

    Default OFF (`DLAAS_INSTANCE_EXPORT_TEMPLATE_ENABLED`) so the new
    write surface is opt-in and reversible: an operator flips the flag to
    expose it, and clearing it makes the route fail closed (404).
    """

    raw = os.environ.get("DLAAS_INSTANCE_EXPORT_TEMPLATE_ENABLED", "").strip().lower()
    return raw in ("1", "true", "on", "yes")


def _capture_instance_learned_state(
    request: web.Request, *, ai_id: str, source: TemplateSpec
) -> tuple[dict[str, Any], str]:
    """Capture an adopted instance's learned state for save-as-template.

    Returns ``(seed_config, learned_state_kind)``. Owner-clean (R8) — we
    never re-derive cognition, we only export what the kernel already
    published:

    * ``protocol_bundle`` — the instance's self-learned school
      (the cultivation uptake bundle for this ``ai_id``) is exported so
      the new template re-hydrates the cultivated cognition on adoption.
      This is the path that makes "save the self-learned soul" real.
    * ``template_clone`` — no learned bundle is held in-process for this
      ai_id, so we clone the source template's ``seed_config`` (the
      baseline the live instance reincarnates from). Honest: the live
      in-session online-learning of an arbitrary adopted instance is not
      serialisable without a kernel export API (tracked debt), so this
      faithfully snapshots the adoptable baseline rather than fabricating
      a capture.
    """

    try:
        from dlaas_platform_api.cultivation import CULTIVATION_BUNDLE_APP_KEY
        from lifeform_service.cultivation_bundle import export_protocol_bundle

        bundle = request.app.get(CULTIVATION_BUNDLE_APP_KEY)
        services = getattr(bundle, "_uptake_services", None) if bundle else None
        svc = services.get(ai_id) if isinstance(services, dict) else None
        approved = svc.loaded_approved_snapshot() if svc is not None else ()
        if approved:
            return (
                {
                    "cultivation_protocol_bundle": export_protocol_bundle(
                        approved, source_ai_id=ai_id, cultivation_id=""
                    )
                },
                "protocol_bundle",
            )
    except Exception:  # noqa: BLE001 -- capture is best-effort, never 500
        _LOG.exception(
            "instance-export: protocol bundle capture failed for ai_id=%s", ai_id
        )

    seed = dict(source.seed_config or {})
    return seed, ("template_clone" if seed else "metadata_only")


async def _handle_export_instance_template(request: web.Request) -> web.Response:
    """Save a LIVE adopted instance as a NEW DLaaS template ("另存为模板").

    Operator-only (``X-Control-Plane-Secret`` / ``X-Service-Secret``),
    flag-gated, owner-clean compose: mints a fresh ``TemplateSpec`` from
    the instance's bound source template, capturing the instance's
    self-learned protocol school when present (else cloning the source
    seed_config), then activates + publishes it and opens a persona
    lifecycle so the new soul shows up in the soul console. R8: no
    cognition is re-derived; this composes existing owners only.

    Body: ``{ "source_template_id": "tpl_...", "template_name"?: str,
              "tenant_id"?: str, "notes"?: str }``. ``source_template_id``
    is the soul's current template (the soul console passes it).
    """

    if not _instance_export_template_enabled():
        return _error(
            404,
            "instance_export_disabled",
            "instance->template export is not enabled on this deployment "
            "(set DLAAS_INSTANCE_EXPORT_TEMPLATE_ENABLED=1).",
        )
    require_control_plane_or_service(request)
    ai_id = request.match_info.get("ai_id", "")
    if not ai_id:
        return _error(400, "missing_ai_id", "ai_id path segment is required.")
    data = await _read_json(request)
    source_template_id = _required_str(data, "source_template_id")
    stores: _Stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        source = await stores.templates.get(source_template_id)
    except TemplateNotFound:
        return _error(
            404,
            "source_template_not_found",
            f"source_template_id={source_template_id!r} does not exist.",
        )

    tenant_id = str(data.get("tenant_id", "") or "") or source.tenant_id
    saved_at_ms = int(time.time() * 1000.0)
    template_name = (
        str(data.get("template_name", "") or "").strip()
        or f"{source.template_name} (saved)"
    )
    seed_config, learned_state = _capture_instance_learned_state(
        request, ai_id=ai_id, source=source
    )
    persona_spec = dict(source.persona_spec or {})
    persona_spec["saved_from_instance"] = {
        "ai_id": ai_id,
        "source_template_id": source_template_id,
        "saved_at_ms": saved_at_ms,
        "learned_state": learned_state,
        "notes": str(data.get("notes", "") or ""),
    }

    spec = await stores.templates.create(
        tenant_id=tenant_id,
        template_name=template_name,
        domain=source.domain,
        description=(
            f"Saved from live instance {ai_id} (source {source_template_id})"
        ),
        runtime_template_id=source.runtime_template_id,
        base_persona=dict(source.base_persona or {}),
        persona_spec=persona_spec,
        seed_config=seed_config,
        figure_artifact_id=source.figure_artifact_id,
        citation_policy=source.citation_policy,
        coverage_policy=source.coverage_policy,
        figure_time_window=source.figure_time_window,
    )
    await stores.templates.update_activation(
        template_id=spec.template_id,
        activation_status=TemplateActivationStatus.ACTIVATED,
        activation_stats={
            "saved_from_ai_id": ai_id,
            "source_template_id": source_template_id,
            "learned_state": learned_state,
        },
    )
    spec = await stores.templates.patch(
        template_id=spec.template_id,
        status=TemplateStatus.PUBLISHED,
        version_note="instance-export",
    )
    # Open a persona lifecycle so the saved soul appears in the soul
    # console. Fail-soft: a lifecycle hiccup must not lose the minted
    # template (the template is already the durable artifact).
    try:
        await stores.lifecycles.create(
            template_id=spec.template_id,
            tenant_id=tenant_id,
            display_name=template_name,
            app_id=str(persona_spec.get("app_id", "") or ""),
            notes=(
                f"saved from instance ai_id={ai_id}; "
                f"source={source_template_id}; learned={learned_state}"
            ),
            actor="operator:instance-export",
        )
    except Exception:  # noqa: BLE001 -- lifecycle is additive, never blocks
        _LOG.exception(
            "instance-export: lifecycle create failed for template=%s",
            spec.template_id,
        )

    return web.json_response(
        {
            "status": "ok",
            "template_id": spec.template_id,
            "source_template_id": source_template_id,
            "ai_id": ai_id,
            "learned_state": learned_state,
            **spec.to_json(),
        }
    )


async def _handle_control_plane_list_bundles(
    request: web.Request,
) -> web.Response:
    """U9 (digital-employee enabler): list every figure bundle currently
    registered in the in-process :class:`FigureBundleStore`.

    Authentication: accepts either a tenant API key (so a per-tenant
    app — e.g. ``digital-employee`` — can render a picker without
    holding the control-plane secret) or ``X-Control-Plane-Secret``
    (for operator scripts). Mirrors the dual-auth pattern used by
    ``_handle_list_applications``.

    The catalog view is read-only and intentionally exposes ONLY the
    fields a downstream picker UI needs (display name, lifespan,
    coverage seed, available time windows, presence/LoRA flags). The
    underlying ``FigureArtifactBundle`` stays inside the process —
    callers reference a bundle by its returned ``bundle_id`` (or
    ``figure_id``) when minting a template.

    Returns::

        {
          "status": "ok",
          "figure_bundles": [
            {
              "bundle_id": "figure-bundle:einstein:<hash16>",
              "figure_id": "einstein",
              "figure_name": "Albert Einstein",
              "figure_lifespan": [1879, 1955],
              "profile_version": "v1+window:(0, 0)",
              "version_window": [0, 0],
              "description": "...",
              "domain_coverage_seed": ["physics", "philosophy_of_science"],
              "time_windows": [
                {"window_id": "early", "year_start": 1900, "year_end": 1925}
              ],
              "has_lora": false,
              "has_presence": false
            },
            ...
          ]
        }

    Empty list is a valid response (fresh install with no profiles
    shipped). 501 is returned when the ``lifeform-service`` wheel is
    absent on this deploy (same fail-loud contract as the rescan
    endpoint).
    """

    if "X-Control-Plane-Secret" in request.headers:
        require_control_plane_secret(request)
    else:
        await require_tenant_auth(request)
    try:
        from lifeform_service import default_figure_bundle_store  # noqa: PLC0415
    except ImportError:
        return _error(
            501,
            "lifeform_service_absent",
            "lifeform-service wheel is not installed on this dlaas-platform; "
            "figure-bundle catalog is not available.",
        )
    store = default_figure_bundle_store()
    seen_ids: set[int] = set()
    catalog: list[dict[str, Any]] = []
    for key in store.keys():
        try:
            bundle = store.lookup(key)
        except LookupError:
            continue
        if id(bundle) in seen_ids:
            # Each bundle is registered under both its bundle_id and
            # its figure_id (see FigureBundleStore.register); de-dupe
            # by object identity so the catalog has one row per bundle.
            continue
        seen_ids.add(id(bundle))
        catalog.append(_figure_bundle_summary(bundle))
    catalog.sort(key=lambda row: (row.get("figure_id", ""), row.get("bundle_id", "")))
    return web.json_response(
        {
            "status": "ok",
            "figure_bundles": catalog,
        }
    )


def _figure_bundle_summary(bundle: object) -> dict[str, Any]:
    """Project a ``FigureArtifactBundle`` to its picker-catalog view.

    All field reads go through ``getattr`` with documented defaults
    so a future bundle revision that adds fields does not break this
    endpoint, and so this module never imports
    ``lifeform_domain_figure`` directly (preserving the DLaaS
    allowlist invariant the rest of control_plane already obeys).
    """

    profile = getattr(bundle, "profile", None)
    figure_name = getattr(profile, "figure_name", "") if profile is not None else ""
    figure_lifespan_raw = (
        getattr(profile, "figure_lifespan", (0, 0)) if profile is not None else (0, 0)
    )
    description = (
        getattr(profile, "description", "") if profile is not None else ""
    )
    domain_coverage_seed = (
        tuple(getattr(profile, "domain_coverage_seed", ()) or ())
        if profile is not None
        else ()
    )
    time_windows_raw = (
        tuple(getattr(profile, "time_windows", ()) or ())
        if profile is not None
        else ()
    )
    time_windows: list[dict[str, Any]] = []
    for window in time_windows_raw:
        time_windows.append(
            {
                "window_id": str(getattr(window, "window_id", "") or ""),
                "year_start": int(getattr(window, "year_start", 0) or 0),
                "year_end": int(getattr(window, "year_end", 0) or 0),
            }
        )
    version_window_raw = getattr(bundle, "version_window", (0, 0)) or (0, 0)
    return {
        "bundle_id": str(getattr(bundle, "bundle_id", "") or ""),
        "figure_id": str(getattr(bundle, "figure_id", "") or ""),
        "figure_name": str(figure_name or ""),
        "figure_lifespan": [
            int(figure_lifespan_raw[0]) if len(figure_lifespan_raw) > 0 else 0,
            int(figure_lifespan_raw[1]) if len(figure_lifespan_raw) > 1 else 0,
        ],
        "profile_version": str(getattr(bundle, "profile_version", "") or ""),
        "version_window": [
            int(version_window_raw[0]) if len(version_window_raw) > 0 else 0,
            int(version_window_raw[1]) if len(version_window_raw) > 1 else 0,
        ],
        "description": str(description or ""),
        "domain_coverage_seed": [str(t) for t in domain_coverage_seed],
        "time_windows": time_windows,
        "has_lora": getattr(bundle, "lora", None) is not None,
        "has_presence": getattr(bundle, "presence", None) is not None,
    }


async def _handle_control_plane_rescan_bundles(
    request: web.Request,
) -> web.Response:
    """U8: rescan ``FIGURE_BUNDLE_ROOT`` and register any newly-baked
    bundles into the in-process :class:`FigureBundleStore`.

    Triggered by ``family-bake-worker`` after a successful bake +
    before the subsequent wake call. Idempotent — :func:`register`
    on the store overwrites by content-addressed ``bundle_id``.

    Request body (all optional):
        ``root_dir``: override the env-derived path. Tests use this
            to point the scanner at a temp dir; in production it
            should remain unset and the env var wins.
        ``figure_id``: pass-through filter to scanner.
        ``reason``: caller-supplied diagnostic string (logged).

    Returns a JSON ``BundleScanReport`` so the caller can confirm
    the freshly-baked bundle id appears in ``bundle_ids`` before
    issuing wake. ``registered_count == 0`` is a valid response and
    means the scanner found nothing new (or nothing at all on a
    fresh install).
    """

    require_control_plane_secret(request)
    try:
        # ``lifeform-service`` is the figure-vertical package and is
        # optional for non-figure deploys. Import lazily so the
        # control plane stays usable in deploys without it; return
        # a typed 501 instead of a generic 500 when missing.
        from lifeform_service import scan_and_register_bundles  # noqa: PLC0415
    except ImportError:
        return _error(
            501,
            "lifeform_service_absent",
            "lifeform-service wheel is not installed on this dlaas-platform; "
            "figure-bundle rescan is not available.",
        )
    try:
        data = await _read_json(request)
    except web.HTTPException:
        data = {}
    root_dir_raw = str(data.get("root_dir", "") or "").strip()
    figure_id_raw = str(data.get("figure_id", "") or "").strip()
    reason = str(data.get("reason", "") or "").strip()
    root_dir = root_dir_raw or os.environ.get("FIGURE_BUNDLE_ROOT", "").strip()
    if not root_dir:
        return _error(
            400,
            "figure_bundle_root_unset",
            "FIGURE_BUNDLE_ROOT is not set on the server and no root_dir "
            "override was supplied in the request body.",
        )
    try:
        report = scan_and_register_bundles(
            pathlib.Path(root_dir),
            figure_id=figure_id_raw or None,
        )
    except (FileNotFoundError, ValueError) as exc:
        return _error(
            422,
            "figure_bundle_scan_failed",
            f"scan_and_register_bundles({root_dir!r}) raised: {exc}",
        )
    _LOG.info(
        "[control_plane] figure-bundles rescan root=%s registered=%d already=%d "
        "figure_filter=%s reason=%s",
        report.root_dir,
        report.registered_count,
        report.already_registered_count,
        figure_id_raw or "(none)",
        reason or "(none)",
    )
    return web.json_response(
        {
            "status": "ok",
            "root_dir": str(report.root_dir),
            "registered_count": report.registered_count,
            "already_registered_count": report.already_registered_count,
            "bundle_ids": list(report.bundle_ids),
        }
    )


def _parse_citation_policy(value: object) -> CitationPolicy:
    """Convert request-body string to ``CitationPolicy``; default DISABLED.

    Raises :class:`ValueError` with the allowed values listed when the
    string is non-empty but unrecognised — fail-loud so misconfigured
    operator scripts get a typed 400 instead of silently dropping to
    the legacy default.
    """

    if value is None or value == "":
        return CitationPolicy.DISABLED
    if isinstance(value, CitationPolicy):
        return value
    try:
        return CitationPolicy(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(p.value for p in CitationPolicy)
        raise ValueError(
            f"citation_policy={value!r} not recognised; allowed: {allowed}"
        ) from exc


def _parse_coverage_policy(value: object) -> CoveragePolicy:
    if value is None or value == "":
        return CoveragePolicy.PASSTHROUGH
    if isinstance(value, CoveragePolicy):
        return value
    try:
        return CoveragePolicy(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(p.value for p in CoveragePolicy)
        raise ValueError(
            f"coverage_policy={value!r} not recognised; allowed: {allowed}"
        ) from exc


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
    # U3 (family-memorial enabler): figure_* field passthrough on
    # PATCH so operators can roll a memorial to a new bundle id (e.g.
    # after a re-bake) without dropping + recreating the template.
    try:
        figure_artifact_id_val: str | None = (
            None
            if "figure_artifact_id" not in data
            else str(data.get("figure_artifact_id", "") or "")
        )
        citation_policy_val = (
            _parse_citation_policy(data["citation_policy"])
            if "citation_policy" in data
            else None
        )
        coverage_policy_val = (
            _parse_coverage_policy(data["coverage_policy"])
            if "coverage_policy" in data
            else None
        )
        figure_time_window_val: str | None = (
            None
            if "figure_time_window" not in data
            else str(data.get("figure_time_window", "") or "")
        )
    except ValueError as exc:
        return _error(400, "invalid_template_policy", str(exc))
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
            figure_artifact_id=figure_artifact_id_val,
            citation_policy=citation_policy_val,
            coverage_policy=coverage_policy_val,
            figure_time_window=figure_time_window_val,
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
        session_id=activation_session_id,
        user_id="dlaas-template-activation",
        tenant_id=tenant.tenant_id,
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
    try:
        plugins = _parse_inline_plugins(data.get("plugins"))
    except ValueError as exc:
        return _error(400, "invalid_plugin_manifest", str(exc))
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
        plugins=plugins,
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
    profile_error, resolved_adapter_policy = _validate_substrate_profile(
        substrate=adoption_config.substrate,
        instance_manager=instance_manager,
    )
    if profile_error is not None:
        return profile_error
    # Persist the *resolved* adapter policy (from the registry profile)
    # so the bind path and the runtime activation gate read a single
    # authoritative value rather than re-deriving it from the contract.
    service_contract.setdefault("adapter_policy", resolved_adapter_policy)
    try:
        inline_plugins = _parse_inline_plugins(data.get("plugins"))
    except ValueError as exc:
        return _error(400, "invalid_plugin_manifest", str(exc))
    application_ids_raw = data.get("application_ids") or ()
    if not isinstance(application_ids_raw, (list, tuple)):
        return _error(
            400,
            "invalid_application_ids",
            "'application_ids' must be a JSON array of strings",
        )
    application_ids = [
        str(app_id) for app_id in application_ids_raw if str(app_id).strip()
    ]
    resolved_app_plugins, app_error = await _resolve_plugins_from_application_ids(
        stores,
        tenant_id=tenant.tenant_id,
        application_ids=application_ids,
    )
    if app_error is not None:
        return app_error
    plugins = tuple(list(inline_plugins) + resolved_app_plugins)
    try:
        # Reject a contract that ends up with two plugins of the same name
        # — whether the duplicate came from inline + approved app or two
        # approved apps, the contract surface has to be deduped or the
        # affordance registry would fail loudly at session start.
        _assert_plugin_names_unique(plugins)
    except ValueError as exc:
        return _error(409, "plugin_name_conflict", str(exc))
    engine_tools = dict(data.get("engine_tools") or {})
    for capability in adoption_config.tools.allowed_capabilities:
        if capability:
            engine_tools.setdefault(capability, True)
    contract = await stores.contracts.create(
        tenant_id=tenant.tenant_id,
        template_id=template_id,
        template_version=int(
            data.get("template_version", template.current_version) or 1
        ),
        shell_id=shell_id,
        owner_user_id=str(data.get("owner_user_id", "") or ""),
        engine_tools=engine_tools,
        tool_policy_snapshot=_compute_tool_policy_snapshot(
            engine_tools,
            plugins,
        ),
        service_contract=service_contract,
        contract_status=ContractStatus.PROVISIONING,
        plugins=plugins,
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
            plugins=final_contract.plugins,
            contract_id=final_contract.contract_id,
            tool_policy_snapshot=dict(final_contract.tool_policy_snapshot),
            tenant_id=tenant.tenant_id,
            scope_strategy=adoption_config.memory.scope_strategy,
            substrate_profile=adoption_config.substrate.substrate_profile_id,
        )
    except LookupError as exc:
        await stores.contracts.update_status(
            contract_id=final_contract.contract_id,
            contract_status=ContractStatus.FAILED,
        )
        return _error(503, "vertical_not_registered", str(exc))

    # Debt #22 closure + U2 (family-memorial enabler): when the
    # template names a ``figure_artifact_id``, resolve it through the
    # lifeform-service public surface and bind it to the
    # ``SessionManager`` of the freshly-acquired ``ai_id`` so
    # subsequent sessions carry the bundle into their synthesizer's
    # L1 / L3 / L4 enforcers. The shared
    # ``bind_figure_artifact_to_ai_id`` helper (above) is reused on
    # the **wake** path (U5) so adopt and wake produce the same
    # binding behaviour.
    bind_result = bind_figure_artifact_to_ai_id(
        instance_manager,
        final_contract.ai_id,
        template.figure_artifact_id,
        adapter_policy=resolved_adapter_policy,
    )
    if not bind_result.bound and bind_result.reason in (
        "bundle_not_found",
        "instance_not_found",
        "lifeform_service_absent",
        "substrate_incompatible",
    ):
        # Roll the contract back to FAILED so the operator's recovery
        # path is "fix the bundle / re-run adopt" rather than
        # "discover stale PROVISIONING".
        await stores.contracts.update_status(
            contract_id=final_contract.contract_id,
            contract_status=ContractStatus.FAILED,
        )
        return _error(
            503,
            _BIND_REASON_TO_ERROR_CODE[bind_result.reason],
            _bind_failure_detail(
                template.figure_artifact_id,
                final_contract.ai_id,
                bind_result.reason,
            ),
        )

    # Mirror the adopted ai_id onto the persona lifecycle (if one exists
    # for this template) so the operator-listable lifecycle becomes the
    # template->ai_id join a management console uses to attach a soul to
    # its live instance + health. Fail-soft: adoption must not break when
    # there is no lifecycle row (template adopted without offline bake).
    try:
        await stores.lifecycles.set_ai_id(
            template_id=template_id, ai_id=final_contract.ai_id
        )
    except Exception as exc:  # noqa: BLE001 - never break adopt on annotate
        _LOG.warning(
            "adopt: failed to link ai_id=%s onto lifecycle for template=%s: %s",
            final_contract.ai_id,
            template_id,
            exc,
        )

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


def _assert_plugin_names_unique(
    plugins: tuple[PluginManifest, ...],
) -> None:
    """Reject duplicate plugin names inside a single contract.

    Mirrors the contract-side rule in
    :func:`dlaas_platform_registry.merge_plugins_from_applications`
    but applied across inline + application-resolved manifests so
    the adopt path's "merge" step never produces a contract that
    breaks the affordance registry at session start.
    """

    seen: dict[str, str] = {}
    for plugin in plugins:
        if plugin.name in seen:
            raise ValueError(
                f"plugin name {plugin.name!r} is declared twice on this "
                f"contract; either drop one of the inline manifests or "
                f"revoke one of the contributing application's approvals."
            )
        seen[plugin.name] = plugin.name


def _managed_safety_dir() -> pathlib.Path:
    """Root directory where portal-authored safety manifests are written.

    Must be a persistent, lifeform-readable path in production (e.g. a
    mounted volume). Defaults under the platform working dir for dev.
    """
    raw = os.environ.get("DLAAS_MANAGED_SAFETY_DIR", "").strip()
    if raw:
        return pathlib.Path(raw)
    return pathlib.Path.cwd() / ".dlaas" / "managed-safety"


def _materialize_inline_safety(
    plugins_payload: Any, *, application_id: str
) -> Any:
    """Persist any per-plugin ``safety_manifest_yaml`` to disk.

    The portal LLM-authors HTTP plugins together with their reviewed
    ``.vzbridge.yaml`` content. Because the runtime only ever loads
    safety from a file (R10, no inline wire field), we materialise that
    content to ``DLAAS_MANAGED_SAFETY_DIR/{application_id}/{plugin}.vzbridge.yaml``
    and rewrite ``safety_manifest_path`` to point at it. The YAML is
    validated with the same ``load_safety_manifest`` the runtime uses, and for
    HTTP plugins every endpoint must have a matching tool entry — so a
    malformed pair is rejected with a 400 before the application exists.

    Plugins without ``safety_manifest_yaml`` pass through unchanged
    (backwards compatible with file-shipped bundles).
    """
    if not isinstance(plugins_payload, (list, tuple)):
        return plugins_payload
    from volvence_zero.mcp_safety_manifest import load_safety_manifest

    rewritten: list[Any] = []
    base_dir = _managed_safety_dir() / application_id
    for item in plugins_payload:
        if not isinstance(item, Mapping):
            rewritten.append(item)
            continue
        plugin = dict(item)
        safety_yaml = plugin.pop("safety_manifest_yaml", None)
        if safety_yaml is None:
            rewritten.append(plugin)
            continue
        if not isinstance(safety_yaml, str) or not safety_yaml.strip():
            raise ValueError(
                "safety_manifest_yaml must be a non-empty string when provided"
            )
        name = str(plugin.get("name", "")).strip()
        if not name:
            raise ValueError("plugin with safety_manifest_yaml must have a name")
        base_dir.mkdir(parents=True, exist_ok=True)
        target = base_dir / f"{name}.vzbridge.yaml"
        target.write_text(safety_yaml, encoding="utf-8")
        try:
            manifest = load_safety_manifest(path=target, expected_server_name=name)
        except Exception as exc:  # noqa: BLE001 - surfaced as 400
            target.unlink(missing_ok=True)
            raise ValueError(
                f"safety_manifest_yaml for plugin {name!r} is invalid: {exc}"
            ) from exc
        # For HTTP plugins, every declared endpoint needs a manifest entry.
        http_block = plugin.get("http")
        if plugin.get("kind") == "http" and isinstance(http_block, Mapping):
            endpoints = http_block.get("endpoints") or []
            for endpoint in endpoints:
                if not isinstance(endpoint, Mapping):
                    continue
                ep_name = str(endpoint.get("name", ""))
                if ep_name and ep_name not in manifest.tool_entries:
                    target.unlink(missing_ok=True)
                    raise ValueError(
                        f"plugin {name!r}: endpoint {ep_name!r} has no matching "
                        f"tool entry in safety_manifest_yaml"
                    )
        plugin["safety_manifest_path"] = str(target)
        rewritten.append(plugin)
    return rewritten


def _parse_inline_plugins(payload: Any) -> tuple[PluginManifest, ...]:
    """Parse a ``plugins`` array out of an HTTP request body.

    Accepts either an absent/None field (no plugins) or a JSON list
    of :class:`PluginManifest` objects in their ``to_json`` shape.
    Rejects any other type loudly so clients get a 400 instead of a
    silent empty plugin set. Packet 3 will replace this inline path
    with ``application_ids`` resolution; the inline acceptor stays
    available as the escape hatch for legacy clients and CLI smoke
    tests.
    """

    if payload is None:
        return ()
    if not isinstance(payload, (list, tuple)):
        raise ValueError(
            "'plugins' must be a JSON array of PluginManifest objects"
        )
    return tuple(PluginManifest.from_json(item) for item in payload)


def _compute_tool_policy_snapshot(
    engine_tools: Mapping[str, Any],
    plugins: tuple[PluginManifest, ...] = (),
) -> dict[str, Any]:
    """Compute the frozen tool-policy snapshot stored on a contract.

    Combines the legacy ``engine_tools`` bool flags with each
    declared plugin's
    :attr:`PluginManifest.declared_capabilities`. The resulting
    ``enabled_capabilities`` list is what the launcher hands to
    :meth:`AffordanceRegistry.set_contract_policy` at session time,
    so plugin-contributed affordances ride the same allowlist as
    legacy capabilities.
    """

    return compute_plugin_tool_policy_snapshot(engine_tools, plugins)


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


def _validate_substrate_profile(
    *,
    substrate: Any,
    instance_manager: InstanceManager,
) -> tuple[web.Response | None, str]:
    """Validate the requested substrate profile against the running base.

    Returns ``(error_response_or_None, resolved_adapter_policy)``.

    * An unknown ``substrate_profile_id`` -> 400 (fail loud; a typo must
      not silently fall through to the default profile).
    * A profile whose ``mode`` does not match the running substrate's
      mode -> 409 (e.g. adopting ``synthetic-dev`` on a GPU-frozen
      deployment, or ``shared-frozen`` on a synthetic dev process).

    On success the resolved profile's ``adapter_policy`` is returned so
    the caller can persist it on the service contract.
    """

    from dlaas_platform_api.substrate_profiles import (
        ADAPTER_POLICY_PERSONA_LORA,
        UnknownSubstrateProfile,
        default_substrate_profile_registry,
        running_substrate_backend,
        running_substrate_mode,
    )

    registry = default_substrate_profile_registry()
    profile_id = substrate.substrate_profile_id
    if not profile_id:
        # No explicit profile selection: stay mode-agnostic for
        # back-compat and keep the additive persona-LoRA behaviour
        # (permissive). Disabling adapters is opt-in by selecting a
        # profile whose adapter_policy is "none". Only an explicit
        # profile id is validated against the running substrate.
        return (None, ADAPTER_POLICY_PERSONA_LORA)
    try:
        profile = registry.get(profile_id)
    except UnknownSubstrateProfile as exc:
        return (
            _error(400, "unknown_substrate_profile", str(exc)),
            "",
        )
    running_runtime = getattr(instance_manager, "substrate_runtime", None)
    running_mode = running_substrate_mode(running_runtime)
    if profile.mode != running_mode:
        return (
            _error(
                409,
                "substrate_profile_mismatch",
                f"substrate_profile_id {profile.substrate_profile_id!r} "
                f"declares mode {profile.mode!r} but the running substrate "
                f"is {running_mode!r}.",
            ),
            "",
        )
    # Backend check only applies to a single-process launcher with a
    # loaded runtime; multi-pod placement routes by profile to a pod
    # whose substrate matches, so skip when there is no local runtime.
    running_backend = running_substrate_backend(running_runtime)
    if running_backend and profile.runtime_backend != running_backend:
        return (
            _error(
                409,
                "substrate_backend_mismatch",
                f"substrate_profile_id {profile.substrate_profile_id!r} "
                f"declares runtime_backend {profile.runtime_backend!r} but the "
                f"running substrate backend is {running_backend!r}.",
            ),
            "",
        )
    return (None, profile.adapter_policy)


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


__all__ = [
    "CONTROL_PLANE_STORES_KEY",
    "attach_control_plane_routes",
]
