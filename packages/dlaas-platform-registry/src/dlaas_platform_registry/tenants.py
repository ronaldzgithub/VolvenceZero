"""Tenant CRUD store backed by :class:`dlaas_platform_registry.db.Registry`.

Every method is async because writes acquire ``Registry.write_lock``;
reads acquire it only to keep the spelling symmetrical and to avoid
surprises if the underlying SQLite driver ever changes. SQLite's WAL
mode means reads do not actually contend with writes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import TenantSpec

from dlaas_platform_registry.db import Registry
from dlaas_platform_registry.secrets import (
    fresh_api_key,
    fresh_api_secret,
    fresh_tenant_id,
    hash_api_secret,
    verify_api_secret,
)


class TenantNotFound(LookupError):
    """Raised when a tenant_id has no row."""


class TenantCredentialError(PermissionError):
    """Raised when an api_key / api_secret pair fails verification."""


class TenantStore:
    """Persistent store for :class:`TenantSpec` records."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def create(
        self,
        *,
        tenant_name: str,
        contact_email: str,
        business_type: str = "generic",
        billing_plan: str = "pay_as_you_go",
        quota: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> TenantSpec:
        """Create a new tenant and return the spec **with plaintext secret**.

        The plaintext ``api_secret`` is returned ONLY here; subsequent
        :meth:`get` calls return ``api_secret=""`` (only the hash is
        persisted). Callers MUST surface the secret to the integrator
        immediately and never log it.
        """
        tenant_id = tenant_id or fresh_tenant_id()
        api_key = fresh_api_key()
        api_secret = fresh_api_secret()
        api_secret_hash = hash_api_secret(api_secret)
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO tenants (
                    tenant_id, tenant_name, contact_email,
                    business_type, billing_plan, quota_json,
                    api_key, api_secret_hash, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    tenant_name,
                    contact_email,
                    business_type,
                    billing_plan,
                    json.dumps(dict(quota or {})),
                    api_key,
                    api_secret_hash,
                    created_at_ms,
                ),
            )
        return TenantSpec(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            contact_email=contact_email,
            business_type=business_type,
            billing_plan=billing_plan,
            quota=dict(quota or {}),
            api_key=api_key,
            api_secret=api_secret,
            created_at_ms=created_at_ms,
        )

    async def get(self, tenant_id: str) -> TenantSpec:
        """Return the tenant spec without the plaintext secret."""
        row = self._registry.conn.execute(
            "SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()
        if row is None:
            raise TenantNotFound(tenant_id)
        return _row_to_spec(row)

    async def list(self) -> tuple[TenantSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM tenants ORDER BY created_at_ms ASC"
        ).fetchall()
        return tuple(_row_to_spec(row) for row in rows)

    async def authenticate(
        self, *, api_key: str, api_secret: str
    ) -> TenantSpec:
        """Validate a tenant credential pair; return the spec on success.

        Raises :class:`TenantCredentialError` if either field is wrong.
        Constant-time comparison via :func:`verify_api_secret`.
        """
        row = self._registry.conn.execute(
            "SELECT * FROM tenants WHERE api_key = ?", (api_key,)
        ).fetchone()
        if row is None or not verify_api_secret(api_secret, row["api_secret_hash"]):
            raise TenantCredentialError("invalid tenant credentials")
        return _row_to_spec(row)


def _row_to_spec(row) -> TenantSpec:
    return TenantSpec(
        tenant_id=row["tenant_id"],
        tenant_name=row["tenant_name"],
        contact_email=row["contact_email"],
        business_type=row["business_type"],
        billing_plan=row["billing_plan"],
        quota=json.loads(row["quota_json"] or "{}"),
        api_key=row["api_key"],
        api_secret="",  # never returned past create
        created_at_ms=int(row["created_at_ms"]),
    )


__all__ = [
    "TenantCredentialError",
    "TenantNotFound",
    "TenantStore",
]
