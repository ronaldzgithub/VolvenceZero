"""Contract / focus_person / identity_link CRUD store.

A :class:`ContractSpec` binds tenant × template × shell × tool policy
× ``ai_id``. ``ai_id`` is empty until adoption succeeds; the launcher
allocates it and writes back via :meth:`ContractStore.set_ai_id`.

Focus persons and identity links share this store because they are
both contract / ai_id-scoped governance metadata. Their cognitive
counterparts (belief / preference / role / memory user_id) remain
owned by the kernel — the registry stores indices only.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import (
    ContractSpec,
    ContractStatus,
    FocusPersonSpec,
    IdentityLinkSpec,
)

from dlaas_platform_registry.db import Registry


class ContractNotFound(LookupError):
    pass


def _fresh_contract_id() -> str:
    return f"ctr_{secrets.token_hex(4)}"


def _fresh_ai_id() -> str:
    return f"ai_{secrets.token_hex(4)}"


class ContractStore:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def create(
        self,
        *,
        tenant_id: str,
        template_id: str,
        shell_id: str,
        template_version: int = 1,
        owner_user_id: str = "",
        engine_tools: Mapping[str, Any] | None = None,
        tool_policy_snapshot: Mapping[str, Any] | None = None,
        service_contract: Mapping[str, Any] | None = None,
        contract_status: ContractStatus = ContractStatus.CREATED,
        contract_id: str | None = None,
    ) -> ContractSpec:
        contract_id = contract_id or _fresh_contract_id()
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO contracts (
                    contract_id, tenant_id, template_id, template_version,
                    shell_id, ai_id, owner_user_id, engine_tools_json,
                    tool_policy_snapshot_json, service_contract_json,
                    contract_status, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?)
                """,
                (
                    contract_id,
                    tenant_id,
                    template_id,
                    template_version,
                    shell_id,
                    owner_user_id,
                    json.dumps(dict(engine_tools or {})),
                    json.dumps(dict(tool_policy_snapshot or {})),
                    json.dumps(dict(service_contract or {})),
                    contract_status.value,
                    created_at_ms,
                ),
            )
        return ContractSpec(
            contract_id=contract_id,
            tenant_id=tenant_id,
            template_id=template_id,
            template_version=template_version,
            shell_id=shell_id,
            ai_id="",
            owner_user_id=owner_user_id,
            engine_tools=dict(engine_tools or {}),
            tool_policy_snapshot=dict(tool_policy_snapshot or {}),
            service_contract=dict(service_contract or {}),
            contract_status=contract_status,
            created_at_ms=created_at_ms,
        )

    async def get(self, contract_id: str) -> ContractSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM contracts WHERE contract_id = ?", (contract_id,)
        ).fetchone()
        if row is None:
            raise ContractNotFound(contract_id)
        return _row_to_spec(row)

    async def list_for_tenant(self, *, tenant_id: str) -> tuple[ContractSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM contracts WHERE tenant_id = ? ORDER BY created_at_ms ASC",
            (tenant_id,),
        ).fetchall()
        return tuple(_row_to_spec(row) for row in rows)

    async def update_status(
        self, *, contract_id: str, contract_status: ContractStatus
    ) -> ContractSpec:
        async with self._registry.write_lock:
            self._registry.conn.execute(
                "UPDATE contracts SET contract_status = ? WHERE contract_id = ?",
                (contract_status.value, contract_id),
            )
        return await self.get(contract_id)

    async def set_ai_id(
        self,
        *,
        contract_id: str,
        ai_id: str | None = None,
        tool_policy_snapshot: Mapping[str, Any] | None = None,
    ) -> ContractSpec:
        """Stamp the adopted ``ai_id`` (and final tool policy) on the contract.

        Called by the launcher after Adoption succeeds. The platform
        guarantees that an ``ai_id`` is unique across the registry —
        the launcher generates a fresh ID via :func:`_fresh_ai_id`
        when the caller does not provide one.
        """
        ai_id = ai_id or _fresh_ai_id()
        current = await self.get(contract_id)
        new_policy = (
            dict(tool_policy_snapshot)
            if tool_policy_snapshot is not None
            else dict(current.tool_policy_snapshot)
        )
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                UPDATE contracts SET
                    ai_id = ?,
                    tool_policy_snapshot_json = ?,
                    contract_status = ?
                WHERE contract_id = ?
                """,
                (
                    ai_id,
                    json.dumps(new_policy),
                    ContractStatus.ACTIVE.value,
                    contract_id,
                ),
            )
        return await self.get(contract_id)

    async def get_by_ai_id(self, ai_id: str) -> ContractSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM contracts WHERE ai_id = ?", (ai_id,)
        ).fetchone()
        if row is None:
            raise ContractNotFound(f"ai_id={ai_id!r}")
        return _row_to_spec(row)

    async def delete(self, contract_id: str) -> bool:
        async with self._registry.write_lock:
            cur = self._registry.conn.execute(
                "DELETE FROM contracts WHERE contract_id = ?", (contract_id,)
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Focus persons (per-contract index; cognitive state stays in kernel)
    # ------------------------------------------------------------------

    async def upsert_focus_person(
        self,
        *,
        contract_id: str,
        person_id: str,
        name: str = "",
        role: str = "user",
        relationship_to_owner: str = "",
        age: int | None = None,
        initial_profile: Mapping[str, Any] | None = None,
    ) -> FocusPersonSpec:
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO focus_persons (
                    contract_id, person_id, name, role,
                    relationship_to_owner, age, initial_profile_json,
                    created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(contract_id, person_id) DO UPDATE SET
                    name = excluded.name,
                    role = excluded.role,
                    relationship_to_owner = excluded.relationship_to_owner,
                    age = excluded.age,
                    initial_profile_json = excluded.initial_profile_json
                """,
                (
                    contract_id,
                    person_id,
                    name,
                    role,
                    relationship_to_owner,
                    age,
                    json.dumps(dict(initial_profile or {})),
                    created_at_ms,
                ),
            )
        return FocusPersonSpec(
            person_id=person_id,
            contract_id=contract_id,
            name=name,
            role=role,
            relationship_to_owner=relationship_to_owner,
            age=age,
            initial_profile=dict(initial_profile or {}),
            created_at_ms=created_at_ms,
        )

    async def list_focus_persons(
        self, *, contract_id: str
    ) -> tuple[FocusPersonSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM focus_persons WHERE contract_id = ? "
            "ORDER BY created_at_ms ASC",
            (contract_id,),
        ).fetchall()
        return tuple(
            FocusPersonSpec(
                person_id=row["person_id"],
                contract_id=row["contract_id"],
                name=row["name"],
                role=row["role"],
                relationship_to_owner=row["relationship_to_owner"],
                age=row["age"],
                initial_profile=json.loads(row["initial_profile_json"] or "{}"),
                created_at_ms=int(row["created_at_ms"]),
            )
            for row in rows
        )

    # ------------------------------------------------------------------
    # Identity links (ai_id-scoped channel mapping)
    # ------------------------------------------------------------------

    async def upsert_identity_link(
        self,
        *,
        ai_id: str,
        channel_type: str,
        channel_ref: str,
        canonical_end_user_ref: str,
        link_meta: Mapping[str, Any] | None = None,
    ) -> IdentityLinkSpec:
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO identity_links (
                    ai_id, channel_type, channel_ref,
                    canonical_end_user_ref, link_meta_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ai_id, channel_type, channel_ref) DO UPDATE SET
                    canonical_end_user_ref = excluded.canonical_end_user_ref,
                    link_meta_json = excluded.link_meta_json
                """,
                (
                    ai_id,
                    channel_type,
                    channel_ref,
                    canonical_end_user_ref,
                    json.dumps(dict(link_meta or {})),
                    created_at_ms,
                ),
            )
        return IdentityLinkSpec(
            ai_id=ai_id,
            channel_type=channel_type,
            channel_ref=channel_ref,
            canonical_end_user_ref=canonical_end_user_ref,
            link_meta=dict(link_meta or {}),
            created_at_ms=created_at_ms,
        )

    async def list_identity_links(
        self, *, ai_id: str
    ) -> tuple[IdentityLinkSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM identity_links WHERE ai_id = ? "
            "ORDER BY created_at_ms ASC",
            (ai_id,),
        ).fetchall()
        return tuple(
            IdentityLinkSpec(
                ai_id=row["ai_id"],
                channel_type=row["channel_type"],
                channel_ref=row["channel_ref"],
                canonical_end_user_ref=row["canonical_end_user_ref"],
                link_meta=json.loads(row["link_meta_json"] or "{}"),
                created_at_ms=int(row["created_at_ms"]),
            )
            for row in rows
        )

    async def resolve_canonical_end_user_ref(
        self,
        *,
        ai_id: str,
        channel_type: str,
        channel_ref: str,
    ) -> str | None:
        row = self._registry.conn.execute(
            "SELECT canonical_end_user_ref FROM identity_links WHERE "
            "ai_id = ? AND channel_type = ? AND channel_ref = ?",
            (ai_id, channel_type, channel_ref),
        ).fetchone()
        if row is None:
            return None
        return row["canonical_end_user_ref"]


def _row_to_spec(row) -> ContractSpec:
    return ContractSpec(
        contract_id=row["contract_id"],
        tenant_id=row["tenant_id"],
        template_id=row["template_id"],
        template_version=int(row["template_version"]),
        shell_id=row["shell_id"],
        ai_id=row["ai_id"],
        owner_user_id=row["owner_user_id"],
        engine_tools=json.loads(row["engine_tools_json"] or "{}"),
        tool_policy_snapshot=json.loads(row["tool_policy_snapshot_json"] or "{}"),
        service_contract=json.loads(row["service_contract_json"] or "{}"),
        contract_status=ContractStatus(row["contract_status"]),
        created_at_ms=int(row["created_at_ms"]),
    )


__all__ = ["ContractNotFound", "ContractStore"]
