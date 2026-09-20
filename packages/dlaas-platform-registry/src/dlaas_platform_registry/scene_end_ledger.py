"""Registry-owned durable ledger for keyed destructive report dispatch.

The store owns reservation, compare-and-set completion, and the irreversible
``outcome_unknown`` terminal.  It deliberately stores only request identity
and the original HTTP response; cognitive/memory state remains with its owner.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import (
    REPORT_SCENE_END_LEDGER_SCHEMA,
    REPORT_SCENE_END_LEDGER_SCHEMA_VERSION,
    ReportSceneEndRequest,
    SceneEndLedgerRecord,
    SceneEndLedgerStatus,
    SceneEndReservation,
    SceneEndReserveOutcome,
)

from dlaas_platform_registry.db import Registry


DEFAULT_SCENE_END_LEASE_MS = 30_000


class SceneEndLedgerTransitionError(RuntimeError):
    """Raised when a caller no longer owns the reserved ledger lease."""


def _row_to_record(row: object) -> SceneEndLedgerRecord:
    response_json = row["response_body_json"]
    return SceneEndLedgerRecord(
        schema_id=row["schema_id"],
        schema_version=int(row["schema_version"]),
        contract_id=row["contract_id"],
        ai_id=row["ai_id"],
        idempotency_key=row["idempotency_key"],
        request_sha256=row["request_sha256"],
        status=SceneEndLedgerStatus(row["status"]),
        lease_token=row["lease_token"],
        lease_expires_at_ms=int(row["lease_expires_at_ms"]),
        response_status=(
            int(row["response_status"])
            if row["response_status"] is not None
            else None
        ),
        response_body=(json.loads(response_json) if response_json is not None else None),
        unknown_reason=row["unknown_reason"],
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )


class SceneEndLedgerStore:
    """CAS store keyed by ``(contract_id, ai_id, idempotency_key)``."""

    def __init__(
        self,
        registry: Registry,
        *,
        lease_ms: int = DEFAULT_SCENE_END_LEASE_MS,
    ) -> None:
        if lease_ms < 1:
            raise ValueError("scene-end ledger lease_ms must be positive")
        self._registry = registry
        self._lease_ms = lease_ms

    def get(
        self, *, contract_id: str, ai_id: str, idempotency_key: str
    ) -> SceneEndLedgerRecord | None:
        row = self._registry.conn.execute(
            """
            SELECT * FROM scene_end_ledger
            WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
            """,
            (contract_id, ai_id, idempotency_key),
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    @property
    def heartbeat_interval_seconds(self) -> float:
        """Cadence that renews well before the current lease expires."""

        return max(0.01, self._lease_ms / 3000.0)

    async def reserve(
        self,
        *,
        request: ReportSceneEndRequest,
        idempotency_key: str,
        now_ms: int | None = None,
    ) -> SceneEndReservation:
        now = int(time.time() * 1000.0) if now_ms is None else now_ms
        if now < 1:
            raise ValueError("scene-end ledger now_ms must be positive")
        lease_token = uuid.uuid4().hex
        lease_expires = now + self._lease_ms
        async with self._registry.write_lock:
            inserted = self._registry.conn.execute(
                """
                INSERT INTO scene_end_ledger (
                    schema_id, schema_version, contract_id, ai_id,
                    idempotency_key, request_sha256, status, lease_token,
                    lease_expires_at_ms, response_status, response_body_json,
                    unknown_reason,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, '', ?, ?)
                ON CONFLICT (contract_id, ai_id, idempotency_key) DO NOTHING
                """,
                (
                    REPORT_SCENE_END_LEDGER_SCHEMA,
                    REPORT_SCENE_END_LEDGER_SCHEMA_VERSION,
                    request.contract_id,
                    request.ai_id,
                    idempotency_key,
                    request.request_sha256,
                    SceneEndLedgerStatus.RESERVED.value,
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
                    raise RuntimeError("inserted scene-end reservation disappeared")
                return SceneEndReservation(SceneEndReserveOutcome.ACQUIRED, record)

            record = self.get(
                contract_id=request.contract_id,
                ai_id=request.ai_id,
                idempotency_key=idempotency_key,
            )
            if record is None:  # pragma: no cover - database contract failure
                raise RuntimeError("scene-end conflict row disappeared")
            if record.request_sha256 != request.request_sha256:
                return SceneEndReservation(SceneEndReserveOutcome.CONFLICT, record)
            if record.status is SceneEndLedgerStatus.COMPLETED:
                return SceneEndReservation(
                    SceneEndReserveOutcome.REPLAY_COMPLETED, record
                )
            if record.status is SceneEndLedgerStatus.OUTCOME_UNKNOWN:
                return SceneEndReservation(
                    SceneEndReserveOutcome.OUTCOME_UNKNOWN, record
                )
            if record.lease_expires_at_ms > now:
                return SceneEndReservation(SceneEndReserveOutcome.IN_PROGRESS, record)

            # Once the exclusive lease expires, execution may already have
            # crossed the destructive runtime boundary.  Fail closed forever;
            # never acquire a second lease and never call end_scene again.
            self._registry.conn.execute(
                """
                UPDATE scene_end_ledger
                SET status = ?, unknown_reason = ?, updated_at_ms = ?
                WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
                  AND request_sha256 = ? AND status = ? AND lease_token = ?
                  AND lease_expires_at_ms = ? AND lease_expires_at_ms <= ?
                """,
                (
                    SceneEndLedgerStatus.OUTCOME_UNKNOWN.value,
                    "lease_expired",
                    now,
                    record.contract_id,
                    record.ai_id,
                    record.idempotency_key,
                    record.request_sha256,
                    SceneEndLedgerStatus.RESERVED.value,
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
                raise RuntimeError("expired scene-end reservation disappeared")
            if terminal.status is SceneEndLedgerStatus.COMPLETED:
                return SceneEndReservation(
                    SceneEndReserveOutcome.REPLAY_COMPLETED, terminal
                )
            if terminal.status is SceneEndLedgerStatus.RESERVED:
                return SceneEndReservation(SceneEndReserveOutcome.IN_PROGRESS, terminal)
            return SceneEndReservation(SceneEndReserveOutcome.OUTCOME_UNKNOWN, terminal)

    async def refresh_lease(
        self,
        *,
        reservation: SceneEndLedgerRecord,
        now_ms: int | None = None,
    ) -> SceneEndLedgerRecord:
        """Extend the active lease without reviving an expired reservation."""

        now = int(time.time() * 1000.0) if now_ms is None else now_ms
        new_expiry = now + self._lease_ms
        async with self._registry.write_lock:
            updated = self._registry.conn.execute(
                """
                UPDATE scene_end_ledger
                SET lease_expires_at_ms = ?, updated_at_ms = ?
                WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
                  AND request_sha256 = ? AND status = ? AND lease_token = ?
                  AND lease_expires_at_ms > ?
                """,
                (
                    new_expiry,
                    now,
                    reservation.contract_id,
                    reservation.ai_id,
                    reservation.idempotency_key,
                    reservation.request_sha256,
                    SceneEndLedgerStatus.RESERVED.value,
                    reservation.lease_token,
                    now,
                ),
            )
            if updated.rowcount != 1:
                raise SceneEndLedgerTransitionError(
                    "scene-end lease cannot be refreshed after expiry or transition"
                )
            record = self.get(
                contract_id=reservation.contract_id,
                ai_id=reservation.ai_id,
                idempotency_key=reservation.idempotency_key,
            )
        if record is None:  # pragma: no cover - database contract failure
            raise RuntimeError("refreshed scene-end ledger row disappeared")
        return record

    async def complete(
        self,
        *,
        reservation: SceneEndLedgerRecord,
        response_status: int,
        response_body: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> SceneEndLedgerRecord:
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
            updated = self._registry.conn.execute(
                """
                UPDATE scene_end_ledger
                SET status = ?, response_status = ?, response_body_json = ?,
                    updated_at_ms = ?
                WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
                  AND request_sha256 = ? AND status = ? AND lease_token = ?
                """,
                (
                    SceneEndLedgerStatus.COMPLETED.value,
                    response_status,
                    body_json,
                    now,
                    reservation.contract_id,
                    reservation.ai_id,
                    reservation.idempotency_key,
                    reservation.request_sha256,
                    SceneEndLedgerStatus.RESERVED.value,
                    reservation.lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise SceneEndLedgerTransitionError(
                    "scene-end completion lost its reservation lease"
                )
            record = self.get(
                contract_id=reservation.contract_id,
                ai_id=reservation.ai_id,
                idempotency_key=reservation.idempotency_key,
            )
        if record is None:  # pragma: no cover - database contract failure
            raise RuntimeError("completed scene-end ledger row disappeared")
        return record

    async def mark_outcome_unknown(
        self,
        *,
        reservation: SceneEndLedgerRecord,
        reason: str,
        now_ms: int | None = None,
    ) -> SceneEndLedgerRecord:
        if not reason.strip():
            raise ValueError("scene-end unknown reason must be non-empty")
        now = int(time.time() * 1000.0) if now_ms is None else now_ms
        async with self._registry.write_lock:
            updated = self._registry.conn.execute(
                """
                UPDATE scene_end_ledger
                SET status = ?, unknown_reason = ?, updated_at_ms = ?
                WHERE contract_id = ? AND ai_id = ? AND idempotency_key = ?
                  AND request_sha256 = ? AND status = ? AND lease_token = ?
                """,
                (
                    SceneEndLedgerStatus.OUTCOME_UNKNOWN.value,
                    reason,
                    now,
                    reservation.contract_id,
                    reservation.ai_id,
                    reservation.idempotency_key,
                    reservation.request_sha256,
                    SceneEndLedgerStatus.RESERVED.value,
                    reservation.lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise SceneEndLedgerTransitionError(
                    "scene-end unknown transition lost its reservation lease"
                )
            record = self.get(
                contract_id=reservation.contract_id,
                ai_id=reservation.ai_id,
                idempotency_key=reservation.idempotency_key,
            )
        if record is None:  # pragma: no cover - database contract failure
            raise RuntimeError("unknown scene-end ledger row disappeared")
        return record


__all__ = [
    "DEFAULT_SCENE_END_LEASE_MS",
    "SceneEndLedgerStore",
    "SceneEndLedgerTransitionError",
]
