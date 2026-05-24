"""Application + ApplicationApproval — the "app as plugin bundle" model.

Packet 3 of the DLaaS plugin foundation introduces a third type of
DLaaS-managed entity alongside tenants and contracts:

* :class:`ApplicationSpec` — a business app (``apps/growth-advisor``,
  ``apps/einstein`` etc.) registered with the platform. Each
  application owns a versioned plugin bundle that the control plane
  can resolve at contract adopt time.
* :class:`ApplicationApprovalSpec` — a tenant's "I trust this app"
  decision. The control plane refuses to wire an application's
  plugins into a contract until the contract's tenant has approved
  the application.

Why a separate object rather than per-contract inline plugins:

* Lets the same plugin bundle apply to many contracts inside one
  org (one approval, N contracts).
* Concentrates safety review on the application boundary — an org
  admin reviews a manifest once at approval time rather than every
  adopt call.
* Plugin authors can ship plugin updates (new versions) without
  every tenant having to re-edit their contract metadata.

Identifiers:

* ``application_id`` is platform-issued (``app_<hex>``). The plaintext
  ``api_secret`` is returned exactly once at create time; subsequent
  reads always return ``""`` and the registry persists
  ``api_secret_hash`` (sha-256 of the plaintext).
* ``tenant_id`` on :class:`ApplicationApprovalSpec` refers to the
  approver tenant. Approvals are removable
  (``ApplicationStore.revoke_approval``); deletion does not affect
  contracts that already adopted the application — the
  :attr:`ContractSpec.plugins` snapshot is frozen at adopt time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dlaas_platform_contracts.plugins import PluginManifest


@dataclass(frozen=True)
class ApplicationSpec:
    """Registered business app with a typed plugin bundle.

    The frozen dataclass mirrors how :class:`TenantSpec` exposes
    secrets: ``api_secret`` is non-empty ONLY in the response of
    ``POST /dlaas/applications`` and ``POST /dlaas/applications/{id}:rotate``
    — every subsequent read returns ``""``.
    """

    application_id: str
    name: str
    version: str = "0.0.0"
    description: str = ""
    plugins: tuple[PluginManifest, ...] = ()
    api_key: str = ""
    api_secret: str = ""
    created_at_ms: int = 0
    updated_at_ms: int = 0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ApplicationSpec":
        if not isinstance(data, Mapping):
            raise ValueError("ApplicationSpec payload must be a JSON object")
        name = str(data.get("name", "") or "")
        if not name.strip():
            raise ValueError("ApplicationSpec.name must be non-empty")
        plugins_raw = data.get("plugins") or ()
        if not isinstance(plugins_raw, (list, tuple)):
            raise ValueError("ApplicationSpec.plugins must be a list of objects")
        plugins = tuple(
            PluginManifest.from_json(item) for item in plugins_raw
        )
        seen: set[str] = set()
        for plugin in plugins:
            if plugin.name in seen:
                raise ValueError(
                    f"ApplicationSpec(plugins=...): duplicate plugin name "
                    f"{plugin.name!r}; an application cannot expose the "
                    "same plugin under multiple manifests."
                )
            seen.add(plugin.name)
        return cls(
            application_id=str(data.get("application_id", "") or ""),
            name=name,
            version=str(data.get("version", "0.0.0") or "0.0.0"),
            description=str(data.get("description", "") or ""),
            plugins=plugins,
            api_key=str(data.get("api_key", "") or ""),
            api_secret=str(data.get("api_secret", "") or ""),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
            updated_at_ms=int(data.get("updated_at_ms", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "plugins": [plugin.to_json() for plugin in self.plugins],
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }


@dataclass(frozen=True)
class ApplicationApprovalSpec:
    """One tenant's approval of one registered application.

    The pair ``(tenant_id, application_id)`` is the primary key; one
    tenant approves an application at most once. Re-approving an
    application that's already approved is a no-op idempotent
    operation (the platform returns the existing approval row
    rather than failing).
    """

    tenant_id: str
    application_id: str
    approved_at_ms: int = 0
    approved_by_user_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ApplicationApprovalSpec":
        if not isinstance(data, Mapping):
            raise ValueError("ApplicationApprovalSpec payload must be a JSON object")
        for key in ("tenant_id", "application_id"):
            if not str(data.get(key, "") or "").strip():
                raise ValueError(
                    f"ApplicationApprovalSpec.{key} must be non-empty"
                )
        return cls(
            tenant_id=str(data["tenant_id"]),
            application_id=str(data["application_id"]),
            approved_at_ms=int(data.get("approved_at_ms", 0) or 0),
            approved_by_user_id=str(
                data.get("approved_by_user_id", "") or ""
            ),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "application_id": self.application_id,
            "approved_at_ms": self.approved_at_ms,
            "approved_by_user_id": self.approved_by_user_id,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "ApplicationApprovalSpec",
    "ApplicationSpec",
]
