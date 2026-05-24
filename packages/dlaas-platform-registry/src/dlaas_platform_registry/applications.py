"""Application + ApplicationApproval CRUD store.

Packet 3 of the DLaaS plugin foundation owns these two tables:

* ``applications`` — one row per registered business application.
  Plugins are serialised into ``plugins_json`` so they ride along
  with reads without extra joins; the typed
  :class:`PluginManifest` is rehydrated on every read.
* ``org_application_approvals`` — one row per ``(tenant, application)``
  approval. Approvals are the gate between "app declared its plugin
  bundle" and "this tenant's adopts can consume those plugins".

Concurrency follows the same pattern as :class:`TenantStore` — writes
hold :attr:`Registry.write_lock`, reads run lock-free on the WAL
snapshot.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from dlaas_platform_contracts import (
    ApplicationApprovalSpec,
    ApplicationSpec,
    PluginManifest,
)

from dlaas_platform_registry.db import Registry
from dlaas_platform_registry.secrets import (
    fresh_application_api_key,
    fresh_application_api_secret,
    fresh_application_id,
    hash_api_secret,
    verify_api_secret,
)


class ApplicationNotFound(LookupError):
    """No application row matches the supplied id."""


class ApplicationApprovalNotFound(LookupError):
    """No approval row matches the supplied ``(tenant, application)`` pair."""


class ApplicationCredentialError(PermissionError):
    """Application api_key / api_secret pair failed verification."""


class ApplicationStore:
    """Persistent store for :class:`ApplicationSpec` records."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Application CRUD
    # ------------------------------------------------------------------

    async def create(
        self,
        *,
        name: str,
        version: str = "0.0.0",
        description: str = "",
        plugins: Sequence[PluginManifest] = (),
        application_id: str | None = None,
    ) -> ApplicationSpec:
        """Create a new application; return the spec **with plaintext secret**.

        The plaintext ``api_secret`` is returned ONLY here; subsequent
        :meth:`get` calls always return ``api_secret=""`` (only the
        hash is persisted). Operators must surface the secret to the
        application owner immediately and never log it.
        """

        application_id = application_id or fresh_application_id()
        api_key = fresh_application_api_key()
        api_secret = fresh_application_api_secret()
        api_secret_hash = hash_api_secret(api_secret)
        created_at_ms = int(time.time() * 1000.0)
        plugins_tuple = tuple(plugins)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO applications (
                    application_id, name, version, description,
                    plugins_json, api_key, api_secret_hash,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    name,
                    version,
                    description,
                    json.dumps([plugin.to_json() for plugin in plugins_tuple]),
                    api_key,
                    api_secret_hash,
                    created_at_ms,
                    created_at_ms,
                ),
            )
        return ApplicationSpec(
            application_id=application_id,
            name=name,
            version=version,
            description=description,
            plugins=plugins_tuple,
            api_key=api_key,
            api_secret=api_secret,
            created_at_ms=created_at_ms,
            updated_at_ms=created_at_ms,
        )

    async def get(self, application_id: str) -> ApplicationSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM applications WHERE application_id = ?",
            (application_id,),
        ).fetchone()
        if row is None:
            raise ApplicationNotFound(application_id)
        return _row_to_spec(row)

    async def list(self) -> tuple[ApplicationSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM applications ORDER BY created_at_ms ASC"
        ).fetchall()
        return tuple(_row_to_spec(row) for row in rows)

    async def update_plugins(
        self,
        *,
        application_id: str,
        plugins: Sequence[PluginManifest],
        version: str | None = None,
        description: str | None = None,
    ) -> ApplicationSpec:
        current = await self.get(application_id)
        new_version = version if version is not None else current.version
        new_description = (
            description if description is not None else current.description
        )
        updated_at_ms = int(time.time() * 1000.0)
        plugins_tuple = tuple(plugins)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                UPDATE applications SET
                    plugins_json = ?,
                    version = ?,
                    description = ?,
                    updated_at_ms = ?
                WHERE application_id = ?
                """,
                (
                    json.dumps([plugin.to_json() for plugin in plugins_tuple]),
                    new_version,
                    new_description,
                    updated_at_ms,
                    application_id,
                ),
            )
        return await self.get(application_id)

    async def delete(self, application_id: str) -> bool:
        async with self._registry.write_lock:
            cur = self._registry.conn.execute(
                "DELETE FROM applications WHERE application_id = ?",
                (application_id,),
            )
            self._registry.conn.execute(
                "DELETE FROM org_application_approvals WHERE application_id = ?",
                (application_id,),
            )
            return cur.rowcount > 0

    async def authenticate(
        self, *, api_key: str, api_secret: str
    ) -> ApplicationSpec:
        """Validate an application credential pair; return the spec on success."""

        row = self._registry.conn.execute(
            "SELECT * FROM applications WHERE api_key = ?",
            (api_key,),
        ).fetchone()
        if row is None or not verify_api_secret(
            api_secret, row["api_secret_hash"]
        ):
            raise ApplicationCredentialError(
                "invalid application credentials"
            )
        return _row_to_spec(row)

    # ------------------------------------------------------------------
    # Approval CRUD
    # ------------------------------------------------------------------

    async def approve(
        self,
        *,
        tenant_id: str,
        application_id: str,
        approved_by_user_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ApplicationApprovalSpec:
        """Idempotently approve an application for a tenant.

        Re-approving an already-approved application returns the
        existing approval row unchanged (idempotent operation —
        admins commonly hit "approve" twice during onboarding).
        """

        existing = await self.get_approval(
            tenant_id=tenant_id, application_id=application_id
        )
        if existing is not None:
            return existing
        approved_at_ms = int(time.time() * 1000.0)
        meta_json = json.dumps(dict(metadata or {}))
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO org_application_approvals (
                    tenant_id, application_id, approved_at_ms,
                    approved_by_user_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    application_id,
                    approved_at_ms,
                    approved_by_user_id,
                    meta_json,
                ),
            )
        return ApplicationApprovalSpec(
            tenant_id=tenant_id,
            application_id=application_id,
            approved_at_ms=approved_at_ms,
            approved_by_user_id=approved_by_user_id,
            metadata=dict(metadata or {}),
        )

    async def revoke_approval(
        self, *, tenant_id: str, application_id: str
    ) -> bool:
        async with self._registry.write_lock:
            cur = self._registry.conn.execute(
                """
                DELETE FROM org_application_approvals
                WHERE tenant_id = ? AND application_id = ?
                """,
                (tenant_id, application_id),
            )
            return cur.rowcount > 0

    async def get_approval(
        self, *, tenant_id: str, application_id: str
    ) -> ApplicationApprovalSpec | None:
        row = self._registry.conn.execute(
            """
            SELECT * FROM org_application_approvals
            WHERE tenant_id = ? AND application_id = ?
            """,
            (tenant_id, application_id),
        ).fetchone()
        if row is None:
            return None
        return _row_to_approval_spec(row)

    async def list_approvals_for_tenant(
        self, *, tenant_id: str
    ) -> tuple[ApplicationApprovalSpec, ...]:
        rows = self._registry.conn.execute(
            """
            SELECT * FROM org_application_approvals
            WHERE tenant_id = ?
            ORDER BY approved_at_ms ASC
            """,
            (tenant_id,),
        ).fetchall()
        return tuple(_row_to_approval_spec(row) for row in rows)

    async def list_approved_applications_for_tenant(
        self, *, tenant_id: str
    ) -> tuple[ApplicationSpec, ...]:
        rows = self._registry.conn.execute(
            """
            SELECT applications.* FROM applications
            INNER JOIN org_application_approvals
                ON applications.application_id = org_application_approvals.application_id
            WHERE org_application_approvals.tenant_id = ?
            ORDER BY org_application_approvals.approved_at_ms ASC
            """,
            (tenant_id,),
        ).fetchall()
        return tuple(_row_to_spec(row) for row in rows)


def merge_plugins_from_applications(
    applications: Sequence[ApplicationSpec],
) -> tuple[PluginManifest, ...]:
    """Flatten + de-duplicate plugin manifests across applications.

    Plugin names are globally unique inside the contract; if two
    approved applications declare the same plugin name, the union is
    a conflict and the caller (adopt path) must surface a 409. We
    raise :class:`ValueError` with both contributing application ids
    so the operator can fix the duplicate.
    """

    merged: list[PluginManifest] = []
    owners: dict[str, str] = {}
    for app in applications:
        for plugin in app.plugins:
            previous_owner = owners.get(plugin.name)
            if previous_owner is not None and previous_owner != app.application_id:
                raise ValueError(
                    f"merge_plugins_from_applications: plugin "
                    f"{plugin.name!r} is declared by both "
                    f"application_id={previous_owner!r} and "
                    f"application_id={app.application_id!r}; "
                    "plugin names must be globally unique inside a "
                    "contract. Rename one of them or revoke one "
                    "application's approval."
                )
            owners[plugin.name] = app.application_id
            merged.append(plugin)
    return tuple(merged)


def _row_to_spec(row) -> ApplicationSpec:
    plugins_raw = row["plugins_json"] or "[]"
    try:
        decoded = json.loads(plugins_raw)
    except json.JSONDecodeError:
        decoded = []
    if isinstance(decoded, list):
        plugins = tuple(
            PluginManifest.from_json(item)
            for item in decoded
            if isinstance(item, Mapping)
        )
    else:
        plugins = ()
    return ApplicationSpec(
        application_id=row["application_id"],
        name=row["name"],
        version=row["version"],
        description=row["description"],
        plugins=plugins,
        api_key=row["api_key"],
        api_secret="",  # never returned past create / rotate
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )


def _row_to_approval_spec(row) -> ApplicationApprovalSpec:
    return ApplicationApprovalSpec(
        tenant_id=row["tenant_id"],
        application_id=row["application_id"],
        approved_at_ms=int(row["approved_at_ms"]),
        approved_by_user_id=row["approved_by_user_id"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


__all__ = [
    "ApplicationApprovalNotFound",
    "ApplicationCredentialError",
    "ApplicationNotFound",
    "ApplicationStore",
    "merge_plugins_from_applications",
]
