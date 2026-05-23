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
from dlaas_platform_launcher.instance_manager import InstanceNotFound
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


BindReason = Literal[
    "ok",
    "empty_figure_artifact_id",
    "lifeform_service_absent",
    "bundle_not_found",
    "instance_not_found",
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
            register_bundle_persona_lora,
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
    session_manager.bind_figure_bundle(bundle)
    if register_bundle_persona_lora is not None:
        register_bundle_persona_lora(bundle)
    return BindResult(bound=True, reason="ok")


_BIND_REASON_TO_ERROR_CODE: dict[BindReason, str] = {
    "ok": "ok",
    "empty_figure_artifact_id": "no_figure_artifact_id",
    "lifeform_service_absent": "figure_vertical_unavailable",
    "bundle_not_found": "figure_bundle_not_registered",
    "instance_not_found": "ai_id_not_acquired",
}


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
        instance_manager, final_contract.ai_id, template.figure_artifact_id
    )
    if not bind_result.bound and bind_result.reason in (
        "bundle_not_found",
        "instance_not_found",
        "lifeform_service_absent",
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
