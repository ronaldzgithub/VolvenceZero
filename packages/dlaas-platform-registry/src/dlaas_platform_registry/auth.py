"""aiohttp auth helpers for the three DLaaS auth modes.

DLaaS public clients use one of three header sets:

* **Tenant auth** — ``X-Tenant-Api-Key`` + ``X-Tenant-Api-Secret``
  (compatibility aliases ``X-DLaaS-Tenant-Key`` /
  ``X-DLaaS-Tenant-Secret`` are also accepted).
* **Control-plane secret** — ``X-Control-Plane-Secret``. Used for
  tenant bootstrap and admin views.
* **Service secret** — ``X-Service-Secret``. Used by the runtime
  awake / diagnostics endpoints.

The helpers in this module raise :class:`aiohttp.web.HTTPUnauthorized`
or :class:`HTTPForbidden` directly so route handlers can stay
linear (``await require_tenant_auth(...)`` at the top, then proceed).
The errors carry a JSON body in the same shape as
``dlaas-platform-api`` produces for invalid envelopes.
"""

from __future__ import annotations

import json
import secrets as _stdlib_secrets
from dataclasses import dataclass

from aiohttp import web

from dlaas_platform_contracts import ApplicationSpec, TenantSpec

from dlaas_platform_registry.applications import (
    ApplicationCredentialError,
    ApplicationStore,
)
from dlaas_platform_registry.tenants import (
    TenantCredentialError,
    TenantStore,
)

REGISTRY_APP_KEY = "dlaas_registry"
"""``app[REGISTRY_APP_KEY]`` carries the :class:`PlatformAuthBundle`."""


# ---------------------------------------------------------------------------
# Configuration bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlatformAuthConfig:
    """Static configuration carried by the api wheel.

    ``control_plane_secret`` and ``service_secret`` are configured via
    environment / config files at startup. Empty values disable that
    auth mode entirely (every request with that header would be
    rejected as ``unauthorized`` unless explicitly stubbed).
    """

    control_plane_secret: str = ""
    service_secret: str = ""


@dataclass
class PlatformAuthBundle:
    """Per-app handle to all platform-tier auth surfaces.

    Stored under ``app[REGISTRY_APP_KEY]`` so handlers can pull both
    the typed :class:`TenantStore` and the static
    :class:`PlatformAuthConfig` in one access.

    ``application_store`` is optional only because Slice 3 + Slice 4
    apps existed before the plugin foundation (Packet 3); old
    deployments that have not migrated the schema yet pass ``None``
    and the new application routes return 503 until the migration
    runs.
    """

    tenant_store: TenantStore
    auth_config: PlatformAuthConfig
    application_store: ApplicationStore | None = None


# ---------------------------------------------------------------------------
# Tenant credentials (with compatibility aliases)
# ---------------------------------------------------------------------------


def _read_tenant_credentials(request: web.Request) -> tuple[str, str]:
    """Extract tenant key/secret from the request, honouring aliases."""
    headers = request.headers
    api_key = headers.get("X-Tenant-Api-Key") or headers.get("X-DLaaS-Tenant-Key")
    api_secret = headers.get("X-Tenant-Api-Secret") or headers.get("X-DLaaS-Tenant-Secret")
    return (api_key or "").strip(), (api_secret or "").strip()


async def require_tenant_auth(request: web.Request) -> TenantSpec:
    """Authenticate the caller as a tenant; return the typed spec.

    Behaviour:

    * Missing both header fields → ``401 missing_tenant_credentials``.
    * Provided but invalid → ``403 invalid_tenant_credentials``.
    * Body field ``tenant_id`` mismatching the authenticated tenant
      → ``403 tenant_mismatch`` (deferred to the handler; this helper
      only resolves the credential).
    """
    bundle: PlatformAuthBundle = request.app[REGISTRY_APP_KEY]
    api_key, api_secret = _read_tenant_credentials(request)
    if not api_key or not api_secret:
        raise _json_unauthorized(
            "missing_tenant_credentials",
            "X-Tenant-Api-Key + X-Tenant-Api-Secret are required",
        )
    try:
        return await bundle.tenant_store.authenticate(
            api_key=api_key, api_secret=api_secret
        )
    except TenantCredentialError as exc:
        raise _json_forbidden(
            "invalid_tenant_credentials", str(exc)
        ) from exc


async def require_application_auth(request: web.Request) -> ApplicationSpec:
    """Authenticate the caller as a registered application.

    Reads ``X-Application-Api-Key`` + ``X-Application-Api-Secret``
    headers; raises ``401`` when absent and ``403`` when invalid.
    Used by application self-service endpoints (``PUT
    /dlaas/applications/{id}``, version bumps, plugin updates) so
    application owners can manage their plugin bundle without
    needing the control-plane secret.
    """

    bundle: PlatformAuthBundle = request.app[REGISTRY_APP_KEY]
    if bundle.application_store is None:
        raise _json_unauthorized(
            "application_auth_disabled",
            "application authentication is not configured on this server",
        )
    headers = request.headers
    api_key = (headers.get("X-Application-Api-Key") or "").strip()
    api_secret = (headers.get("X-Application-Api-Secret") or "").strip()
    if not api_key or not api_secret:
        raise _json_unauthorized(
            "missing_application_credentials",
            "X-Application-Api-Key + X-Application-Api-Secret are required",
        )
    try:
        return await bundle.application_store.authenticate(
            api_key=api_key, api_secret=api_secret
        )
    except ApplicationCredentialError as exc:
        raise _json_forbidden(
            "invalid_application_credentials", str(exc)
        ) from exc


def assert_tenant_id_matches(
    tenant: TenantSpec, requested_tenant_id: str
) -> None:
    """Reject any cross-tenant access at the handler boundary.

    Raises ``403 tenant_mismatch`` when a request body / path declares
    a ``tenant_id`` that does not equal the authenticated tenant. This
    keeps tenant isolation enforced at the platform edge — the
    registry never returns rows for a tenant the caller cannot
    authenticate as.
    """
    if not requested_tenant_id:
        return
    if requested_tenant_id != tenant.tenant_id:
        raise _json_forbidden(
            "tenant_mismatch",
            (
                f"authenticated tenant_id={tenant.tenant_id!r} cannot "
                f"act on tenant_id={requested_tenant_id!r}"
            ),
        )


# ---------------------------------------------------------------------------
# Control-plane / service secrets
# ---------------------------------------------------------------------------


def require_control_plane_secret(request: web.Request) -> None:
    """Reject a request that does not present the control-plane secret.

    Used by tenant bootstrap and admin views (cross-tenant). The
    secret is configured via :class:`PlatformAuthConfig`; if the
    config has empty ``control_plane_secret``, all requests are
    rejected (the auth mode is administratively disabled).
    """
    bundle: PlatformAuthBundle = request.app[REGISTRY_APP_KEY]
    expected = bundle.auth_config.control_plane_secret
    provided = (request.headers.get("X-Control-Plane-Secret") or "").strip()
    if not expected:
        raise _json_unauthorized(
            "control_plane_secret_disabled",
            "control-plane secret auth is not configured on this server",
        )
    if not provided:
        raise _json_unauthorized(
            "missing_control_plane_secret",
            "X-Control-Plane-Secret header is required",
        )
    if not _stdlib_secrets.compare_digest(provided, expected):
        raise _json_forbidden(
            "invalid_control_plane_secret",
            "X-Control-Plane-Secret value does not match",
        )


def require_service_secret(request: web.Request) -> None:
    """Reject a request that does not present the service secret."""
    bundle: PlatformAuthBundle = request.app[REGISTRY_APP_KEY]
    expected = bundle.auth_config.service_secret
    provided = (request.headers.get("X-Service-Secret") or "").strip()
    if not expected:
        raise _json_unauthorized(
            "service_secret_disabled",
            "service secret auth is not configured on this server",
        )
    if not provided:
        raise _json_unauthorized(
            "missing_service_secret",
            "X-Service-Secret header is required",
        )
    if not _stdlib_secrets.compare_digest(provided, expected):
        raise _json_forbidden(
            "invalid_service_secret",
            "X-Service-Secret value does not match",
        )


def require_control_plane_or_service(request: web.Request) -> None:
    """Accept either the control-plane OR the service secret.

    Some admin endpoints accept either credential; this helper avoids
    duplicating the try/except logic in every handler.
    """
    headers = request.headers
    if "X-Control-Plane-Secret" in headers:
        require_control_plane_secret(request)
        return
    if "X-Service-Secret" in headers:
        require_service_secret(request)
        return
    raise _json_unauthorized(
        "missing_admin_credential",
        "X-Control-Plane-Secret or X-Service-Secret is required",
    )


# ---------------------------------------------------------------------------
# JSON-error helpers
# ---------------------------------------------------------------------------


def _json_unauthorized(code: str, detail: str) -> web.HTTPUnauthorized:
    return web.HTTPUnauthorized(
        text=json.dumps({"status": "error", "error": code, "detail": detail}),
        content_type="application/json",
    )


def _json_forbidden(code: str, detail: str) -> web.HTTPForbidden:
    return web.HTTPForbidden(
        text=json.dumps({"status": "error", "error": code, "detail": detail}),
        content_type="application/json",
    )


__all__ = [
    "PlatformAuthBundle",
    "PlatformAuthConfig",
    "REGISTRY_APP_KEY",
    "assert_tenant_id_matches",
    "require_application_auth",
    "require_control_plane_or_service",
    "require_control_plane_secret",
    "require_service_secret",
    "require_tenant_auth",
]
