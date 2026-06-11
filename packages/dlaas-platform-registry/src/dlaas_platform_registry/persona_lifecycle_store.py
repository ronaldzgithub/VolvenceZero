"""Persona cognitive-training lifecycle persistence (schema v9).

One row per persona (keyed by ``template_id``) tracks the unified
training pipeline (draft → pretrained → studying → training → exam →
interview → inducted | retired). Every transition is persisted as an
immutable event row carrying its evidence, so the gate history is
audit-grade and the lifecycle is rollback-able (R15) without losing
which gates were actually passed.

Ownership boundary (R8 / R12): this store holds governance pointers
(bundle ids, cultivation ids, exam/interview run ids) — never
cognition. Stage validation lives in
``dlaas_platform_contracts.persona_lifecycle`` so every consumer sees
the same transition rules.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import (
    INDUCTION_GATE_STAGES,
    ExamRunStatus,
    LifecycleStageEvent,
    LifecycleTransitionError,
    PersonaLifecycleStage,
    PersonaTrainingLifecycle,
    stage_order_index,
    validate_stage_advance,
)

from dlaas_platform_registry.db import Registry
from dlaas_platform_registry.eval_store import EvalStore, ExamRunNotFound

#: ``run_type`` value that marks an eval run as an interactive interview.
#: Exact protocol constant (not keyword inference): callers recording an
#: interview through ``POST /dlaas/exam_runs`` must set this run_type.
INTERVIEW_RUN_TYPE = "interview"

#: Evidence key that must reference a persisted eval run per gate stage.
_GATE_RUN_EVIDENCE_KEY: Mapping[PersonaLifecycleStage, str] = {
    PersonaLifecycleStage.EXAM: "exam_run_id",
    PersonaLifecycleStage.INTERVIEW: "interview_run_id",
}


class PersonaLifecycleNotFound(LookupError):
    pass


class PersonaLifecycleConflict(ValueError):
    """Raised when a lifecycle already exists for the template."""


def _fresh_lifecycle_id() -> str:
    return f"plc_{secrets.token_hex(4)}"


def _fresh_event_id() -> str:
    return f"plce_{secrets.token_hex(4)}"


class PersonaLifecycleStore:
    """CRUD + transition log over ``persona_lifecycles`` tables.

    Mirrors the :class:`CultivationStore` shape: writes take the
    registry write-lock; reads run lock-free on WAL snapshots and
    return frozen contract dataclasses.
    """

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    # -- create / read -----------------------------------------------------

    async def create(
        self,
        *,
        template_id: str,
        tenant_id: str = "",
        ai_id: str = "",
        display_name: str = "",
        app_id: str = "",
        notes: str = "",
        actor: str = "",
    ) -> PersonaTrainingLifecycle:
        existing = self._registry.conn.execute(
            "SELECT lifecycle_id FROM persona_lifecycles WHERE template_id = ?",
            (template_id,),
        ).fetchone()
        if existing is not None:
            raise PersonaLifecycleConflict(
                f"lifecycle already exists for template_id={template_id!r} "
                f"(lifecycle_id={existing['lifecycle_id']!r})"
            )
        lifecycle_id = _fresh_lifecycle_id()
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO persona_lifecycles (
                    lifecycle_id, template_id, tenant_id, ai_id,
                    display_name, app_id, stage, notes,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?, ?)
                """,
                (
                    lifecycle_id,
                    template_id,
                    tenant_id,
                    ai_id,
                    display_name,
                    app_id,
                    notes,
                    now,
                    now,
                ),
            )
            self._insert_event_locked(
                lifecycle_id=lifecycle_id,
                event_kind="created",
                from_stage=PersonaLifecycleStage.DRAFT,
                to_stage=PersonaLifecycleStage.DRAFT,
                evidence={},
                actor=actor,
                now=now,
            )
        return await self.get(lifecycle_id)

    async def get(self, lifecycle_id: str) -> PersonaTrainingLifecycle:
        row = self._registry.conn.execute(
            "SELECT * FROM persona_lifecycles WHERE lifecycle_id = ?",
            (lifecycle_id,),
        ).fetchone()
        if row is None:
            raise PersonaLifecycleNotFound(lifecycle_id)
        return _row_to_lifecycle(row)

    async def get_by_template(self, template_id: str) -> PersonaTrainingLifecycle:
        row = self._registry.conn.execute(
            "SELECT * FROM persona_lifecycles WHERE template_id = ?",
            (template_id,),
        ).fetchone()
        if row is None:
            raise PersonaLifecycleNotFound(template_id)
        return _row_to_lifecycle(row)

    async def list_all(
        self, *, tenant_id: str = ""
    ) -> tuple[PersonaTrainingLifecycle, ...]:
        if tenant_id:
            rows = self._registry.conn.execute(
                "SELECT * FROM persona_lifecycles WHERE tenant_id = ? "
                "ORDER BY created_at_ms DESC",
                (tenant_id,),
            ).fetchall()
        else:
            rows = self._registry.conn.execute(
                "SELECT * FROM persona_lifecycles ORDER BY created_at_ms DESC"
            ).fetchall()
        return tuple(_row_to_lifecycle(row) for row in rows)

    async def list_events(
        self, lifecycle_id: str
    ) -> tuple[LifecycleStageEvent, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM persona_lifecycle_events WHERE lifecycle_id = ? "
            "ORDER BY recorded_at_ms ASC, event_id ASC",
            (lifecycle_id,),
        ).fetchall()
        return tuple(_row_to_event(row) for row in rows)

    # -- transitions -------------------------------------------------------

    async def advance(
        self,
        *,
        lifecycle_id: str,
        target: PersonaLifecycleStage,
        evidence: Mapping[str, Any],
        actor: str = "",
    ) -> PersonaTrainingLifecycle:
        """Move forward one or more stages with mandatory evidence.

        Raises :class:`LifecycleTransitionError` when the transition or
        evidence violates the contract — never degrades silently.
        """

        record = await self.get(lifecycle_id)
        validate_stage_advance(
            current=record.stage, target=target, evidence=evidence
        )
        if target is PersonaLifecycleStage.INDUCTED:
            await self._assert_induction_gates(
                lifecycle_id=lifecycle_id, evidence=evidence
            )
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                "UPDATE persona_lifecycles SET stage = ?, updated_at_ms = ? "
                "WHERE lifecycle_id = ?",
                (target.value, now, lifecycle_id),
            )
            self._insert_event_locked(
                lifecycle_id=lifecycle_id,
                event_kind="advance",
                from_stage=record.stage,
                to_stage=target,
                evidence=evidence,
                actor=actor,
                now=now,
            )
        return await self.get(lifecycle_id)

    async def rollback(
        self,
        *,
        lifecycle_id: str,
        target: PersonaLifecycleStage,
        reason: str,
        actor: str = "",
    ) -> PersonaTrainingLifecycle:
        """Explicit, audited backwards move (R15).

        The target must be strictly earlier on the forward pipeline and
        a non-empty reason is mandatory.
        """

        record = await self.get(lifecycle_id)
        if not reason.strip():
            raise LifecycleTransitionError("rollback requires a non-empty reason")
        if record.stage is PersonaLifecycleStage.RETIRED:
            raise LifecycleTransitionError(
                "retired lifecycles cannot be rolled back"
            )
        if target is PersonaLifecycleStage.RETIRED:
            raise LifecycleTransitionError(
                "cannot rollback to 'retired'; use advance with a reason"
            )
        if stage_order_index(target) >= stage_order_index(record.stage):
            raise LifecycleTransitionError(
                f"rollback must move backwards: {record.stage.value!r} -> "
                f"{target.value!r}"
            )
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                "UPDATE persona_lifecycles SET stage = ?, updated_at_ms = ? "
                "WHERE lifecycle_id = ?",
                (target.value, now, lifecycle_id),
            )
            self._insert_event_locked(
                lifecycle_id=lifecycle_id,
                event_kind="rollback",
                from_stage=record.stage,
                to_stage=target,
                evidence={"reason": reason},
                actor=actor,
                now=now,
            )
        return await self.get(lifecycle_id)

    # -- helpers -----------------------------------------------------------

    async def _assert_induction_gates(
        self, *, lifecycle_id: str, evidence: Mapping[str, Any]
    ) -> None:
        """Induction requires passing exam + interview evidence.

        An explicit ``waiver_reason`` in the induction evidence skips
        the check — the waiver itself is then part of the audit trail
        (no silent pass).
        """

        if str(evidence.get("waiver_reason", "") or "").strip():
            return
        events = await self.list_events(lifecycle_id)
        for gate in INDUCTION_GATE_STAGES:
            latest = None
            for event in events:
                if event.to_stage is gate and event.event_kind == "advance":
                    latest = event
            if latest is None:
                raise LifecycleTransitionError(
                    f"induction requires a recorded {gate.value!r} stage; "
                    "none found (supply waiver_reason to override explicitly)"
                )
            if not bool(latest.evidence.get("passed")):
                raise LifecycleTransitionError(
                    f"induction requires the latest {gate.value!r} evidence "
                    "to carry passed=true (supply waiver_reason to override "
                    "explicitly)"
                )

    def _insert_event_locked(
        self,
        *,
        lifecycle_id: str,
        event_kind: str,
        from_stage: PersonaLifecycleStage,
        to_stage: PersonaLifecycleStage,
        evidence: Mapping[str, Any],
        actor: str,
        now: int,
    ) -> None:
        self._registry.conn.execute(
            """
            INSERT INTO persona_lifecycle_events (
                event_id, lifecycle_id, event_kind, from_stage, to_stage,
                evidence_json, actor, recorded_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _fresh_event_id(),
                lifecycle_id,
                event_kind,
                from_stage.value,
                to_stage.value,
                json.dumps(dict(evidence), ensure_ascii=False),
                actor,
                now,
            ),
        )


def _row_to_lifecycle(row) -> PersonaTrainingLifecycle:
    return PersonaTrainingLifecycle(
        lifecycle_id=row["lifecycle_id"],
        template_id=row["template_id"],
        tenant_id=row["tenant_id"],
        ai_id=row["ai_id"],
        display_name=row["display_name"],
        app_id=row["app_id"],
        stage=PersonaLifecycleStage(row["stage"]),
        notes=row["notes"],
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )


def _row_to_event(row) -> LifecycleStageEvent:
    return LifecycleStageEvent(
        event_id=row["event_id"],
        lifecycle_id=row["lifecycle_id"],
        event_kind=row["event_kind"],
        from_stage=PersonaLifecycleStage(row["from_stage"]),
        to_stage=PersonaLifecycleStage(row["to_stage"]),
        evidence=json.loads(row["evidence_json"] or "{}"),
        actor=row["actor"],
        recorded_at_ms=int(row["recorded_at_ms"]),
    )


__all__ = [
    "PersonaLifecycleConflict",
    "PersonaLifecycleNotFound",
    "PersonaLifecycleStore",
]
