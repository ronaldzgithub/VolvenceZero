"""Asset CRUD store.

Assets are content references (chat logs, persona kits, manuals, …)
owned by a tenant and linked to one or more templates. The registry
persists metadata only; the actual bytes live at ``uri``.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import AssetSpec, TemplateAssetLinkSpec

from dlaas_platform_registry.db import Registry


class AssetNotFound(LookupError):
    pass


def _fresh_asset_id() -> str:
    return f"ast_{secrets.token_hex(4)}"


class AssetStore:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def create(
        self,
        *,
        tenant_id: str,
        asset_type: str,
        title: str = "",
        uri: str = "",
        mime_type: str = "",
        language: str = "",
        source_meta: Mapping[str, Any] | None = None,
        asset_id: str | None = None,
    ) -> AssetSpec:
        asset_id = asset_id or _fresh_asset_id()
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO assets (
                    asset_id, tenant_id, asset_type, title, uri,
                    mime_type, language, source_meta_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    tenant_id,
                    asset_type,
                    title,
                    uri,
                    mime_type,
                    language,
                    json.dumps(dict(source_meta or {})),
                    created_at_ms,
                ),
            )
        return AssetSpec(
            asset_id=asset_id,
            tenant_id=tenant_id,
            asset_type=asset_type,
            title=title,
            uri=uri,
            mime_type=mime_type,
            language=language,
            source_meta=dict(source_meta or {}),
            created_at_ms=created_at_ms,
        )

    async def get(self, asset_id: str) -> AssetSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        if row is None:
            raise AssetNotFound(asset_id)
        return _row_to_spec(row)

    async def list_for_tenant(self, *, tenant_id: str) -> tuple[AssetSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM assets WHERE tenant_id = ? ORDER BY created_at_ms ASC",
            (tenant_id,),
        ).fetchall()
        return tuple(_row_to_spec(row) for row in rows)

    async def link_to_template(
        self,
        *,
        template_id: str,
        asset_id: str,
        template_version: int = 1,
        role: str = "training_material",
        link_meta: Mapping[str, Any] | None = None,
    ) -> TemplateAssetLinkSpec:
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO template_assets (
                    template_id, asset_id, template_version,
                    role, link_meta_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(template_id, asset_id, template_version) DO UPDATE SET
                    role=excluded.role,
                    link_meta_json=excluded.link_meta_json
                """,
                (
                    template_id,
                    asset_id,
                    template_version,
                    role,
                    json.dumps(dict(link_meta or {})),
                    created_at_ms,
                ),
            )
        return TemplateAssetLinkSpec(
            template_id=template_id,
            asset_id=asset_id,
            template_version=template_version,
            role=role,
            link_meta=dict(link_meta or {}),
        )

    async def list_template_links(
        self, *, template_id: str
    ) -> tuple[TemplateAssetLinkSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM template_assets WHERE template_id = ? "
            "ORDER BY created_at_ms ASC",
            (template_id,),
        ).fetchall()
        return tuple(
            TemplateAssetLinkSpec(
                template_id=row["template_id"],
                asset_id=row["asset_id"],
                template_version=int(row["template_version"]),
                role=row["role"],
                link_meta=json.loads(row["link_meta_json"] or "{}"),
            )
            for row in rows
        )


def _row_to_spec(row) -> AssetSpec:
    return AssetSpec(
        asset_id=row["asset_id"],
        tenant_id=row["tenant_id"],
        asset_type=row["asset_type"],
        title=row["title"],
        uri=row["uri"],
        mime_type=row["mime_type"],
        language=row["language"],
        source_meta=json.loads(row["source_meta_json"] or "{}"),
        created_at_ms=int(row["created_at_ms"]),
    )


__all__ = ["AssetNotFound", "AssetStore"]
