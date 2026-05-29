"""Persisted DLaaS training-job store (rare-heavy executor).

Backs the ``training_jobs`` table. The platform's training-job
governance API (create / get / cancel / promote) plus the background
:class:`TrainingJobExecutor` use this so job lifecycle
(``pending -> running -> succeeded/failed``) survives a process
restart, instead of living only in an in-memory dict.

Returns frozen :class:`dlaas_platform_contracts.TrainingJob` specs; the
platform never sees raw rows past this boundary (R8).
"""

from __future__ import annotations

import json
import time

from dlaas_platform_contracts import (
    TrainingJob,
    TrainingJobStatus,
    TrainingJobType,
)

from dlaas_platform_registry.db import Registry


class TrainingJobNotFound(LookupError):
    pass


def _row_to_job(row: object) -> TrainingJob:
    return TrainingJob(
        job_id=row["job_id"],
        ai_id=row["ai_id"],
        contract_id=row["contract_id"],
        job_type=TrainingJobType(row["job_type"]),
        status=TrainingJobStatus(row["status"]),
        created_by=row["created_by"],
        source_ref=row["source_ref"],
        promotion_gate=row["promotion_gate"],
        artifact_ref=row["artifact_ref"],
        gate_evidence=json.loads(row["gate_evidence_json"] or "{}"),
        notes=row["notes"],
    )


class TrainingJobStore:
    """Persisted ``training_jobs`` CRUD over the registry SQLite db."""

    def __init__(self, registry: Registry, *, tenant_id: str = "") -> None:
        self._registry = registry
        self._tenant_id = tenant_id

    async def put(self, job: TrainingJob, *, tenant_id: str = "") -> TrainingJob:
        """Insert or replace ``job`` (idempotent on (ai_id, job_id))."""

        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            existing = self._registry.conn.execute(
                "SELECT created_at_ms FROM training_jobs WHERE ai_id = ? AND job_id = ?",
                (job.ai_id, job.job_id),
            ).fetchone()
            created_at = existing["created_at_ms"] if existing is not None else now
            self._registry.conn.execute(
                """
                INSERT OR REPLACE INTO training_jobs (
                    job_id, ai_id, contract_id, tenant_id, job_type, status,
                    created_by, source_ref, promotion_gate, artifact_ref,
                    gate_evidence_json, notes, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.ai_id,
                    job.contract_id,
                    tenant_id or self._tenant_id,
                    job.job_type.value,
                    job.status.value,
                    job.created_by,
                    job.source_ref,
                    job.promotion_gate,
                    job.artifact_ref,
                    json.dumps(dict(job.gate_evidence)),
                    job.notes,
                    created_at,
                    now,
                ),
            )
        return job

    def get(self, *, ai_id: str, job_id: str) -> TrainingJob | None:
        row = self._registry.conn.execute(
            "SELECT * FROM training_jobs WHERE ai_id = ? AND job_id = ?",
            (ai_id, job_id),
        ).fetchone()
        if row is None:
            return None
        return _row_to_job(row)

    def list_by_status(self, status: TrainingJobStatus) -> tuple[TrainingJob, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM training_jobs WHERE status = ? ORDER BY created_at_ms",
            (status.value,),
        ).fetchall()
        return tuple(_row_to_job(row) for row in rows)

    def list_all(self) -> tuple[TrainingJob, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM training_jobs ORDER BY created_at_ms"
        ).fetchall()
        return tuple(_row_to_job(row) for row in rows)


__all__ = ["TrainingJobNotFound", "TrainingJobStore"]
