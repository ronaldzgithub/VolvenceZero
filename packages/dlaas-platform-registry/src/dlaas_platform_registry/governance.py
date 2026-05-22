"""Generic governance record store.

The governance API owns several append/readout-oriented surfaces
(audit, artifacts, usage, billing, consent, data jobs). They share one
SQLite table keyed by ``(record_kind, record_id)`` so SHADOW API
surfaces can become persistent without introducing many premature
tables. The public DTO JSON remains the contract.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from dlaas_platform_registry.db import Registry


class GovernanceRecordNotFound(LookupError):
    pass


class GovernanceStore:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def upsert(
        self,
        *,
        record_kind: str,
        record_id: str,
        payload: Mapping[str, Any],
        ai_id: str = "",
        contract_id: str = "",
        session_id: str = "",
        created_at_ms: int | None = None,
    ) -> None:
        created = int(created_at_ms if created_at_ms is not None else time.time() * 1000)
        self._registry.conn.execute(
            """
            INSERT INTO governance_records (
                record_kind, record_id, ai_id, contract_id, session_id,
                payload_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_kind, record_id) DO UPDATE SET
                ai_id = excluded.ai_id,
                contract_id = excluded.contract_id,
                session_id = excluded.session_id,
                payload_json = excluded.payload_json,
                created_at_ms = excluded.created_at_ms
            """,
            (
                record_kind,
                record_id,
                ai_id,
                contract_id,
                session_id,
                json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
                created,
            ),
        )

    def get(self, *, record_kind: str, record_id: str) -> dict[str, Any]:
        row = self._registry.conn.execute(
            """
            SELECT payload_json FROM governance_records
            WHERE record_kind = ? AND record_id = ?
            """,
            (record_kind, record_id),
        ).fetchone()
        if row is None:
            raise GovernanceRecordNotFound(f"{record_kind}:{record_id}")
        return json.loads(row["payload_json"] or "{}")

    def list(
        self,
        *,
        record_kind: str,
        ai_id: str = "",
        session_id: str = "",
    ) -> tuple[dict[str, Any], ...]:
        clauses = ["record_kind = ?"]
        params: list[Any] = [record_kind]
        if ai_id:
            clauses.append("ai_id = ?")
            params.append(ai_id)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        rows = self._registry.conn.execute(
            f"""
            SELECT payload_json FROM governance_records
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at_ms ASC
            """,
            tuple(params),
        ).fetchall()
        return tuple(json.loads(row["payload_json"] or "{}") for row in rows)


__all__ = ["GovernanceRecordNotFound", "GovernanceStore"]
