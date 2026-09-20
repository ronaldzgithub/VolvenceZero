"""Registry-owned durable ledger for keyed cognitive-turn dispatch.

This store owns reservation, lease renewal, compare-and-set completion, and
the irreversible ``OUTCOME_UNKNOWN`` terminal.  It stores no Lifeform state;
the only durable values are semantic request identity and the original HTTP
response needed for replay.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import (
    COGNITIVE_TURN_LEDGER_SCHEMA,
    COGNITIVE_TURN_LEDGER_SCHEMA_VERSION,
    CognitiveTurnLedgerRecord,
    CognitiveTurnLedgerStatus,
    CognitiveTurnRequest,
    CognitiveTurnReservation,
    CognitiveTurnReserveOutcome,
)

from dlaas_platform_registry.db import Registry


DEFAULT_COGNITIVE_TURN_LEASE_MS = 30_000


class CognitiveTurnLedgerTransitionError(RuntimeError):
    """Raised when a caller no longer owns the reserved ledger lease."""


def _row_to_record(row: object) -> CognitiveTurnLedgerRecord:
    response_json = row["response_body_json"]
    return CognitiveTurnLedgerRecord(
        schema_id=row["schema_id"],
        schema_version=int(row["schema_version"]),
        contract_id=row["contract_id"],
        ai_id=row["ai_id"],
        idempotency_key=row["idempotency_key"],
        request_sha256=row["request_sha256"],
        status=CognitiveTurnLedgerStatus(row["status"]),
        lease_token=row["lease_token"],
        lease_expires_at_ms=int(row["lease_expires_at_ms"]),
        response_status=(int(row["response_status"]) if row["response_status"] is not None else None),
        response_body=(json.loads(response_json) if response_json is not None else None),
        unknown_reason=row["unknown_reason"],
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )


class CognitiveTurnLedgerStore:
    """CAS store keyed by ``(contract_id, ai_id, idempotency_key)``."""

    def __init__(
        self,
        registry: Registry,
        *,
        lease_ms: int = DEFAULT_COGNITIVE_TURN_LEASE_MS,
    ) -> None:
        if lease_ms < 1:
            raise ValueError("cognitive-turn ledger lease_ms must be positive")
        self._registry = registry
        self._lease_ms = lease_ms

    def get(self, *, contract_id: str, ai_id: str, idempotency_key: str) -> CognitiveTurnLedgerRecord | None:
        row = self._registry.conn.execute(
            """
            SELECT * FROM cognitive_turn_ledger
            WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
            """,
            (contract_id, ai_id, idempotency_key),
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    @property
    def heartbeat_interval_seconds(self) -> float:
        return max(0.01, self._lease_ms / 3000.0)

    async def reserve(
        self,
        *,
        request: CognitiveTurnRequest,
        idempotency_key: str,
        now_ms: int | None = None,
    ) -> CognitiveTurnReservation:
        now = int(time.time() * 1000.0) if now_ms is None else now_ms
        if now < 1:
            raise ValueError("cognitive-turn ledger now_ms must be positive")
        lease_token = uuid.uuid4().hex
        lease_expires = now + self._lease_ms
        async with self._registry.write_lock:
            inserted = self._registry.conn.execute(
                """
                INSERT INTO cognitive_turn_ledger (
                    schema_id, schema_version, contract_id, ai_id,
                    idempotency_key, request_sha256, status, lease_token,
                    lease_expires_at_ms, response_status, response_body_json,
                    unknown_reason, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, '', ?, ?)
                ON CONFLICT (contract_id, ai_id, idempotency_key) DO NOTHING
                """,
                (
                    COGNITIVE_TURN_LEDGER_SCHEMA,
                    COGNITIVE_TURN_LEDGER_SCHEMA_VERSION,
                    request.contract_id,
                    request.ai_id,
                    idempotency_key,
                    request.request_sha256,
                    CognitiveTurnLedgerStatus.RESERVED.value,
                    lease_token,
                    lease_expires,
                    now,
                    now,
                ),
            )
            if inserted.rowcount == 1:
                record = self.get(
                    contract_id=request.contract_id,
                    ai_id=request.ai_id,
                    idempotency_key=idempotency_key,
                )
                if record is None:  # pragma: no cover - database contract failure
                    raise RuntimeError("inserted cognitive-turn reservation disappeared")
                return CognitiveTurnReservation(CognitiveTurnReserveOutcome.ACQUIRED, record)

            record = self.get(
                contract_id=request.contract_id,
                ai_id=request.ai_id,
                idempotency_key=idempotency_key,
            )
            if record is None:  # pragma: no cover - database contract failure
                raise RuntimeError("cognitive-turn conflict row disappeared")
            if record.request_sha256 != request.request_sha256:
                return CognitiveTurnReservation(CognitiveTurnReserveOutcome.CONFLICT, record)
            if record.status is CognitiveTurnLedgerStatus.COMPLETED:
                return CognitiveTurnReservation(CognitiveTurnReserveOutcome.REPLAY_COMPLETED, record)
            if record.status is CognitiveTurnLedgerStatus.OUTCOME_UNKNOWN:
                return CognitiveTurnReservation(CognitiveTurnReserveOutcome.OUTCOME_UNKNOWN, record)
            if record.lease_expires_at_ms > now:
                return CognitiveTurnReservation(CognitiveTurnReserveOutcome.IN_PROGRESS, record)

            self._registry.conn.execute(
                """
                UPDATE cognitive_turn_ledger
                SET status = ?, unknown_reason = ?, updated_at_ms = ?
                WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
                  AND request_sha256 = ? AND status = ? AND lease_token = ?
                  AND lease_expires_at_ms = ? AND lease_expires_at_ms <= ?
                """,
                (
                    CognitiveTurnLedgerStatus.OUTCOME_UNKNOWN.value,
                    "lease_expired",
                    now,
                    record.contract_id,
                    record.ai_id,
                    record.idempotency_key,
                    record.request_sha256,
                    CognitiveTurnLedgerStatus.RESERVED.value,
                    record.lease_token,
                    record.lease_expires_at_ms,
                    now,
                ),
            )
            terminal = self.get(
                contract_id=request.contract_id,
                ai_id=request.ai_id,
                idempotency_key=idempotency_key,
            )
            if terminal is None:  # pragma: no cover - database contract failure
                raise RuntimeError("expired cognitive-turn reservation disappeared")
            if terminal.status is CognitiveTurnLedgerStatus.COMPLETED:
                return CognitiveTurnReservation(CognitiveTurnReserveOutcome.REPLAY_COMPLETED, terminal)
            if terminal.status is CognitiveTurnLedgerStatus.RESERVED:
                return CognitiveTurnReservation(CognitiveTurnReserveOutcome.IN_PROGRESS, terminal)
            return CognitiveTurnReservation(CognitiveTurnReserveOutcome.OUTCOME_UNKNOWN, terminal)

    async def refresh_lease(
        self,
        *,
        reservation: CognitiveTurnLedgerRecord,
        now_ms: int | None = None,
    ) -> CognitiveTurnLedgerRecord:
        now = int(time.time() * 1000.0) if now_ms is None else now_ms
        new_expiry = now + self._lease_ms
        async with self._registry.write_lock:
            # Expiry never transfers execution to a new owner: a competing
            # reserve can only terminalize the row as OUTCOME_UNKNOWN.  Let
            # the original token renew late when its process-local event loop
            # was starved; the status/token CAS still makes that renewal race
            # safely against peer terminalization without double execution.
            updated = self._registry.conn.execute(
                """
                UPDATE cognitive_turn_ledger
                SET lease_expires_at_ms = ?, updated_at_ms = ?
                WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
                  AND request_sha256 = ? AND status = ? AND lease_token = ?
                """,
                (
                    new_expiry,
                    now,
                    reservation.contract_id,
                    reservation.ai_id,
                    reservation.idempotency_key,
                    reservation.request_sha256,
                    CognitiveTurnLedgerStatus.RESERVED.value,
                    reservation.lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise CognitiveTurnLedgerTransitionError(
                    "cognitive-turn lease cannot be refreshed after transition or ownership loss"
                )
            record = self.get(
                contract_id=reservation.contract_id,
                ai_id=reservation.ai_id,
                idempotency_key=reservation.idempotency_key,
            )
        if record is None:  # pragma: no cover - database contract failure
            raise RuntimeError("refreshed cognitive-turn ledger row disappeared")
        return record

    async def complete(
        self,
        *,
        reservation: CognitiveTurnLedgerRecord,
        response_status: int,
        response_body: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> CognitiveTurnLedgerRecord:
        if not (200 <= response_status < 300 or 400 <= response_status < 500):
            raise ValueError("only 2xx or deterministic 4xx responses can complete")
        now = int(time.time() * 1000.0) if now_ms is None else now_ms
        body_json = json.dumps(
            dict(response_body),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        async with self._registry.write_lock:
            # Completion follows the same ownership rule as late renewal.  A
            # peer can only terminalize an expired reservation, never acquire
            # it; status plus the original token therefore remain the
            # at-most-once CAS even when the wall-clock deadline has passed.
            updated = self._registry.conn.execute(
                """
                UPDATE cognitive_turn_ledger
                SET status = ?, response_status = ?, response_body_json = ?,
                    updated_at_ms = ?
                WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
                  AND request_sha256 = ? AND status = ? AND lease_token = ?
                """,
                (
                    CognitiveTurnLedgerStatus.COMPLETED.value,
                    response_status,
                    body_json,
                    now,
                    reservation.contract_id,
                    reservation.ai_id,
                    reservation.idempotency_key,
                    reservation.request_sha256,
                    CognitiveTurnLedgerStatus.RESERVED.value,
                    reservation.lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise CognitiveTurnLedgerTransitionError("cognitive-turn completion lost its reservation lease")
            record = self.get(
                contract_id=reservation.contract_id,
                ai_id=reservation.ai_id,
                idempotency_key=reservation.idempotency_key,
            )
        if record is None:  # pragma: no cover - database contract failure
            raise RuntimeError("completed cognitive-turn ledger row disappeared")
        return record

    async def mark_outcome_unknown(
        self,
        *,
        reservation: CognitiveTurnLedgerRecord,
        reason: str,
        now_ms: int | None = None,
    ) -> CognitiveTurnLedgerRecord:
        if not reason.strip():
            raise ValueError("cognitive-turn unknown reason must be non-empty")
        now = int(time.time() * 1000.0) if now_ms is None else now_ms
        async with self._registry.write_lock:
            updated = self._registry.conn.execute(
                """
                UPDATE cognitive_turn_ledger
                SET status = ?, unknown_reason = ?, updated_at_ms = ?
                WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
                  AND request_sha256 = ? AND status = ? AND lease_token = ?
                """,
                (
                    CognitiveTurnLedgerStatus.OUTCOME_UNKNOWN.value,
                    reason,
                    now,
                    reservation.contract_id,
                    reservation.ai_id,
                    reservation.idempotency_key,
                    reservation.request_sha256,
                    CognitiveTurnLedgerStatus.RESERVED.value,
                    reservation.lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise CognitiveTurnLedgerTransitionError("cognitive-turn unknown transition lost its reservation lease")
            record = self.get(
                contract_id=reservation.contract_id,
                ai_id=reservation.ai_id,
                idempotency_key=reservation.idempotency_key,
            )
        if record is None:  # pragma: no cover - database contract failure
            raise RuntimeError("unknown cognitive-turn ledger row disappeared")
        return record


__all__ = [
    "CognitiveTurnLedgerStore",
    "CognitiveTurnLedgerTransitionError",
    "DEFAULT_COGNITIVE_TURN_LEASE_MS",
]
