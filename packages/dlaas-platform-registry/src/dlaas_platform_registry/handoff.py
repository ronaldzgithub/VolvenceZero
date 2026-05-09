"""Handoff ticket store (Slice 5.2).

Tickets are persisted at the platform layer and surfaced to operators
via the ops endpoints. The trigger signal — when to open a ticket
proactively — is computed from ``vz-cognition.rupture_state``
snapshots; this store does not hold any cognitive state.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import HandoffStatus, HandoffTicketSpec

from dlaas_platform_registry.db import Registry


class HandoffTicketNotFound(LookupError):
    pass


def _fresh_ticket_id() -> str:
    return f"hto_{secrets.token_hex(4)}"


class HandoffTicketStore:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def create(
        self,
        *,
        ai_id: str,
        contract_id: str,
        end_user_ref: str,
        session_id: str = "",
        trigger_reason: str = "",
        trigger_details: Mapping[str, Any] | None = None,
        confidence_aggregate: float = 0.0,
        recent_response_ids: tuple[str, ...] = (),
        ticket_id: str | None = None,
    ) -> HandoffTicketSpec:
        ticket_id = ticket_id or _fresh_ticket_id()
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO handoff_tickets (
                    ticket_id, ai_id, contract_id, end_user_ref,
                    session_id, trigger_reason, trigger_details_json,
                    confidence_aggregate, recent_response_ids_json,
                    status, operator_id, human_reply, resolution_notes,
                    created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', '', '', '', ?)
                """,
                (
                    ticket_id,
                    ai_id,
                    contract_id,
                    end_user_ref,
                    session_id,
                    trigger_reason,
                    json.dumps(dict(trigger_details or {})),
                    float(confidence_aggregate),
                    json.dumps(list(recent_response_ids)),
                    created_at_ms,
                ),
            )
        return HandoffTicketSpec(
            ticket_id=ticket_id,
            ai_id=ai_id,
            contract_id=contract_id,
            end_user_ref=end_user_ref,
            session_id=session_id,
            trigger_reason=trigger_reason,
            trigger_details=dict(trigger_details or {}),
            confidence_aggregate=float(confidence_aggregate),
            recent_response_ids=tuple(recent_response_ids),
            status=HandoffStatus.OPEN,
            created_at_ms=created_at_ms,
        )

    async def get(self, ticket_id: str) -> HandoffTicketSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM handoff_tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        if row is None:
            raise HandoffTicketNotFound(ticket_id)
        return _row_to_spec(row)

    async def list_for_ai(
        self, *, ai_id: str, status: HandoffStatus | None = None
    ) -> tuple[HandoffTicketSpec, ...]:
        if status is None:
            rows = self._registry.conn.execute(
                "SELECT * FROM handoff_tickets WHERE ai_id = ? "
                "ORDER BY created_at_ms ASC",
                (ai_id,),
            ).fetchall()
        else:
            rows = self._registry.conn.execute(
                "SELECT * FROM handoff_tickets WHERE ai_id = ? AND status = ? "
                "ORDER BY created_at_ms ASC",
                (ai_id, status.value),
            ).fetchall()
        return tuple(_row_to_spec(row) for row in rows)

    async def submit_human_reply(
        self,
        *,
        ticket_id: str,
        operator_id: str,
        human_reply: str,
        resolution_notes: str = "",
    ) -> HandoffTicketSpec:
        async with self._registry.write_lock:
            cur = self._registry.conn.execute(
                """
                UPDATE handoff_tickets SET
                    operator_id = ?,
                    human_reply = ?,
                    resolution_notes = ?,
                    status = ?
                WHERE ticket_id = ?
                """,
                (
                    operator_id,
                    human_reply,
                    resolution_notes,
                    HandoffStatus.RESOLVED.value,
                    ticket_id,
                ),
            )
            if cur.rowcount == 0:
                raise HandoffTicketNotFound(ticket_id)
        return await self.get(ticket_id)


def _row_to_spec(row) -> HandoffTicketSpec:
    return HandoffTicketSpec(
        ticket_id=row["ticket_id"],
        ai_id=row["ai_id"],
        contract_id=row["contract_id"],
        end_user_ref=row["end_user_ref"],
        session_id=row["session_id"],
        trigger_reason=row["trigger_reason"],
        trigger_details=json.loads(row["trigger_details_json"] or "{}"),
        confidence_aggregate=float(row["confidence_aggregate"]),
        recent_response_ids=tuple(
            json.loads(row["recent_response_ids_json"] or "[]")
        ),
        status=HandoffStatus(row["status"]),
        operator_id=row["operator_id"],
        human_reply=row["human_reply"],
        resolution_notes=row["resolution_notes"],
        created_at_ms=int(row["created_at_ms"]),
    )


__all__ = ["HandoffTicketNotFound", "HandoffTicketStore"]
