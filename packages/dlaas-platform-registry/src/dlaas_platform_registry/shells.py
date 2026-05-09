"""Shell CRUD store.

Shells are the externally-visible "embodiments" the AI gets adopted
into. Their ``embodiment`` payload mirrors the four-Kind affordance
descriptor (``perception`` / ``expression`` / ``action`` / ``constraints``)
so capability degradation in the platform-api OutputAct path can match
shell.embodiment to the kernel-emitted capability.

Shell identifiers are unique per tenant — a tenant may own many
shells, and the same ``shell_id`` may be re-used across tenants
without conflict.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import ShellKind, ShellSpec

from dlaas_platform_registry.db import Registry


class ShellNotFound(LookupError):
    pass


class ShellStore:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def upsert(
        self,
        *,
        tenant_id: str,
        shell_id: str,
        shell_kind: ShellKind,
        shell_type: str = "generic",
        display_name: str = "",
        embodiment: Mapping[str, Any] | None = None,
        channel: Mapping[str, Any] | None = None,
        scene_meta: Mapping[str, Any] | None = None,
    ) -> ShellSpec:
        """Create or replace a shell record.

        DLaaS public clients re-declare a shell on every restart; the
        platform treats this as an upsert so integrators do not need
        to track existence separately.
        """
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO shells (
                    shell_id, tenant_id, shell_kind, shell_type,
                    display_name, embodiment_json, channel_json,
                    scene_meta_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, shell_id) DO UPDATE SET
                    shell_kind=excluded.shell_kind,
                    shell_type=excluded.shell_type,
                    display_name=excluded.display_name,
                    embodiment_json=excluded.embodiment_json,
                    channel_json=excluded.channel_json,
                    scene_meta_json=excluded.scene_meta_json
                """,
                (
                    shell_id,
                    tenant_id,
                    shell_kind.value,
                    shell_type,
                    display_name,
                    json.dumps(dict(embodiment or {})),
                    json.dumps(dict(channel or {})),
                    json.dumps(dict(scene_meta or {})),
                    created_at_ms,
                ),
            )
        return ShellSpec(
            shell_id=shell_id,
            tenant_id=tenant_id,
            shell_kind=shell_kind,
            shell_type=shell_type,
            display_name=display_name,
            embodiment=dict(embodiment or {}),
            channel=dict(channel or {}),
            scene_meta=dict(scene_meta or {}),
            created_at_ms=created_at_ms,
        )

    async def get(self, *, tenant_id: str, shell_id: str) -> ShellSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM shells WHERE tenant_id = ? AND shell_id = ?",
            (tenant_id, shell_id),
        ).fetchone()
        if row is None:
            raise ShellNotFound(f"{tenant_id}/{shell_id}")
        return _row_to_spec(row)

    async def list_for_tenant(self, *, tenant_id: str) -> tuple[ShellSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM shells WHERE tenant_id = ? ORDER BY created_at_ms ASC",
            (tenant_id,),
        ).fetchall()
        return tuple(_row_to_spec(row) for row in rows)


def _row_to_spec(row) -> ShellSpec:
    return ShellSpec(
        shell_id=row["shell_id"],
        tenant_id=row["tenant_id"],
        shell_kind=ShellKind(row["shell_kind"]),
        shell_type=row["shell_type"],
        display_name=row["display_name"],
        embodiment=json.loads(row["embodiment_json"] or "{}"),
        channel=json.loads(row["channel_json"] or "{}"),
        scene_meta=json.loads(row["scene_meta_json"] or "{}"),
        created_at_ms=int(row["created_at_ms"]),
    )


__all__ = ["ShellNotFound", "ShellStore"]
