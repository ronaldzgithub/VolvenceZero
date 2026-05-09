"""DLaaS platform-tier multi-tenant registry.

The registry is the single owner of all governance-state (tenant /
shell / asset / template / contract / focus_person / identity_link /
handoff_ticket). Cognitive state stays in the kernel and is read via
``vz-contracts`` snapshot types — see
``docs/specs/dlaas-platform.md`` invariants 5 & 6.

Public exports:

* :class:`Registry` — SQLite-backed connection + write lock.
* Per-resource stores: :class:`TenantStore`, :class:`ShellStore`,
  :class:`AssetStore`, :class:`TemplateStore`, :class:`ContractStore`,
  :class:`HandoffTicketStore`.
* Auth helpers: :class:`PlatformAuthBundle`,
  :class:`PlatformAuthConfig`, :func:`require_tenant_auth`,
  :func:`require_control_plane_secret`,
  :func:`require_service_secret`,
  :func:`require_control_plane_or_service`,
  :func:`assert_tenant_id_matches`,
  :data:`REGISTRY_APP_KEY`.
* Typed errors: :class:`TenantNotFound`,
  :class:`TenantCredentialError`, :class:`ShellNotFound`,
  :class:`AssetNotFound`, :class:`TemplateNotFound`,
  :class:`TemplateVersionNotFound`, :class:`ContractNotFound`,
  :class:`HandoffTicketNotFound`.
"""

from __future__ import annotations

from dlaas_platform_registry.assets import AssetNotFound, AssetStore
from dlaas_platform_registry.auth import (
    PlatformAuthBundle,
    PlatformAuthConfig,
    REGISTRY_APP_KEY,
    assert_tenant_id_matches,
    require_control_plane_or_service,
    require_control_plane_secret,
    require_service_secret,
    require_tenant_auth,
)
from dlaas_platform_registry.contracts import ContractNotFound, ContractStore
from dlaas_platform_registry.db import Registry, init_schema, open_connection
from dlaas_platform_registry.eval_store import (
    AudienceProfileNotFound,
    EvalStore,
    ExamQuestionNotFound,
    ExamRunNotFound,
    LaunchLicenseNotFound,
)
from dlaas_platform_registry.handoff import (
    HandoffTicketNotFound,
    HandoffTicketStore,
)
from dlaas_platform_registry.shells import ShellNotFound, ShellStore
from dlaas_platform_registry.templates import (
    TemplateNotFound,
    TemplateStore,
    TemplateVersionNotFound,
)
from dlaas_platform_registry.tenants import (
    TenantCredentialError,
    TenantNotFound,
    TenantStore,
)

__all__ = (
    "AssetNotFound",
    "AssetStore",
    "AudienceProfileNotFound",
    "ContractNotFound",
    "ContractStore",
    "EvalStore",
    "ExamQuestionNotFound",
    "ExamRunNotFound",
    "HandoffTicketNotFound",
    "HandoffTicketStore",
    "LaunchLicenseNotFound",
    "PlatformAuthBundle",
    "PlatformAuthConfig",
    "REGISTRY_APP_KEY",
    "Registry",
    "ShellNotFound",
    "ShellStore",
    "TemplateNotFound",
    "TemplateStore",
    "TemplateVersionNotFound",
    "TenantCredentialError",
    "TenantNotFound",
    "TenantStore",
    "assert_tenant_id_matches",
    "init_schema",
    "open_connection",
    "require_control_plane_or_service",
    "require_control_plane_secret",
    "require_service_secret",
    "require_tenant_auth",
)
