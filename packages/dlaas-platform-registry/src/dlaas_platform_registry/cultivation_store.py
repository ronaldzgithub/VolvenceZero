"""Autonomous expert-cultivation persistence (schema v6).

A *cultivation* is the platform-owned record for one industry expert
that the system is growing autonomously (the "行业专家自动养成" loop).
The cognitive state itself lives in the kernel (vz-*) and is reached
through the instance's ``ai_id``; this store only persists the
control-plane governance of the cultivation: its seed persona, its
curriculum, its lifecycle status, and the *readout* metrics
(cycles completed, school-coherence score, observed regime history)
that the operator console renders.

Ownership boundary (R8 / R12): the coherence score and regime history
are **readouts** computed from the kernel's published ``active_regime``
sequence — never a learning signal that flows back into the kernel.
The status machine is platform lifecycle, not cognition.

Status machine::

    seeding ──▶ studying ──▶ converging ──▶ exam ──▶ ready_for_review ──▶ inducted
        │           │             │           │              │
        └───────────┴─────────────┴───────────┴──────────────┴──▶ failed

* ``seeding``          — record created; seed charter not yet ingested.
* ``studying``         — autonomous self-study cycles running; school
                         not yet converged.
* ``converging``       — coherence over threshold but not yet exam-gated.
* ``exam``             — an eval/launch-license exam run is in flight.
* ``ready_for_review`` — exam passed; awaiting operator induction.
* ``inducted``         — promoted to a published default expert template.
* ``failed``           — cultivation abandoned (operator or hard error).
"""

from __future__ import annotations

import enum
import json
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dlaas_platform_registry.db import Registry


class CultivationNotFound(LookupError):
    pass


class CultivationStatus(str, enum.Enum):
    SEEDING = "seeding"
    STUDYING = "studying"
    CONVERGING = "converging"
    EXAM = "exam"
    READY_FOR_REVIEW = "ready_for_review"
    INDUCTED = "inducted"
    FAILED = "failed"


@dataclass(frozen=True)
class CultivationRecordSpec:
    """Immutable snapshot of one cultivation row."""

    cultivation_id: str
    ai_id: str
    slug: str
    display_name: str
    domain: str
    runtime_template_id: str
    seed_persona: Mapping[str, Any] = field(default_factory=dict)
    curriculum: Mapping[str, Any] = field(default_factory=dict)
    status: CultivationStatus = CultivationStatus.SEEDING
    cycles_completed: int = 0
    coherence_score: float = 0.0
    coherence_detail: Mapping[str, Any] = field(default_factory=dict)
    regime_history: tuple[str, ...] = ()
    dlaas_template_id: str = ""
    last_exam_run_id: str = ""
    inducted_template_id: str = ""
    notes: str = ""
    # Multi-direction package grouping (schema v8). ``package_id`` ties
    # sibling school tracks grown from one seed; ``track_id`` is this
    # track's stable id; ``direction`` is the per-track research schedule.
    # Empty package_id/track_id = legacy single-expert cultivation.
    package_id: str = ""
    track_id: str = ""
    direction: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "cultivation_id": self.cultivation_id,
            "ai_id": self.ai_id,
            "slug": self.slug,
            "display_name": self.display_name,
            "domain": self.domain,
            "runtime_template_id": self.runtime_template_id,
            "seed_persona": dict(self.seed_persona),
            "curriculum": dict(self.curriculum),
            "status": self.status.value,
            "cycles_completed": self.cycles_completed,
            "coherence_score": self.coherence_score,
            "coherence_detail": dict(self.coherence_detail),
            "regime_history": list(self.regime_history),
            "dlaas_template_id": self.dlaas_template_id,
            "last_exam_run_id": self.last_exam_run_id,
            "inducted_template_id": self.inducted_template_id,
            "notes": self.notes,
            "package_id": self.package_id,
            "track_id": self.track_id,
            "direction": dict(self.direction),
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }


def _fresh_cultivation_id() -> str:
    return f"cult_{secrets.token_hex(4)}"


class CultivationStore:
    """CRUD over the ``cultivations`` table.

    Mirrors the :class:`EvalStore` shape: writes take the registry
    write-lock; reads run lock-free on WAL snapshots and always return
    frozen :class:`CultivationRecordSpec` instances.
    """

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def create(
        self,
        *,
        ai_id: str,
        slug: str,
        display_name: str,
        domain: str,
        runtime_template_id: str,
        seed_persona: Mapping[str, Any],
        curriculum: Mapping[str, Any],
        cultivation_id: str | None = None,
        notes: str = "",
        package_id: str = "",
        track_id: str = "",
        direction: Mapping[str, Any] | None = None,
    ) -> CultivationRecordSpec:
        cultivation_id = cultivation_id or _fresh_cultivation_id()
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO cultivations (
                    cultivation_id, ai_id, slug, display_name, domain,
                    runtime_template_id, seed_persona_json, curriculum_json,
                    status, notes, package_id, track_id, direction_json,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'seeding', ?, ?, ?, ?, ?, ?)
                """,
                (
                    cultivation_id,
                    ai_id,
                    slug,
                    display_name,
                    domain,
                    runtime_template_id,
                    json.dumps(dict(seed_persona), ensure_ascii=False),
                    json.dumps(dict(curriculum), ensure_ascii=False),
                    notes,
                    package_id,
                    track_id,
                    json.dumps(dict(direction or {}), ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return await self.get(cultivation_id)

    async def get(self, cultivation_id: str) -> CultivationRecordSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM cultivations WHERE cultivation_id = ?",
            (cultivation_id,),
        ).fetchone()
        if row is None:
            raise CultivationNotFound(cultivation_id)
        return _row_to_cultivation(row)

    async def get_by_ai_id(self, ai_id: str) -> CultivationRecordSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM cultivations WHERE ai_id = ?", (ai_id,)
        ).fetchone()
        if row is None:
            raise CultivationNotFound(ai_id)
        return _row_to_cultivation(row)

    async def list_all(self) -> tuple[CultivationRecordSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM cultivations ORDER BY created_at_ms DESC"
        ).fetchall()
        return tuple(_row_to_cultivation(row) for row in rows)

    async def list_for_package(
        self, package_id: str
    ) -> tuple[CultivationRecordSpec, ...]:
        """Return every track row sharing ``package_id`` (creation order)."""

        rows = self._registry.conn.execute(
            "SELECT * FROM cultivations WHERE package_id = ? "
            "ORDER BY created_at_ms ASC",
            (package_id,),
        ).fetchall()
        return tuple(_row_to_cultivation(row) for row in rows)

    async def list_package_ids(self) -> tuple[str, ...]:
        """Return distinct non-empty package ids, newest first."""

        rows = self._registry.conn.execute(
            "SELECT package_id, MAX(created_at_ms) AS latest "
            "FROM cultivations WHERE package_id != '' "
            "GROUP BY package_id ORDER BY latest DESC"
        ).fetchall()
        return tuple(row["package_id"] for row in rows)

    async def update_progress(
        self,
        *,
        cultivation_id: str,
        status: CultivationStatus,
        cycles_completed: int,
        coherence_score: float,
        coherence_detail: Mapping[str, Any],
        regime_history: tuple[str, ...],
    ) -> CultivationRecordSpec:
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                UPDATE cultivations SET
                    status = ?,
                    cycles_completed = ?,
                    coherence_score = ?,
                    coherence_detail_json = ?,
                    regime_history_json = ?,
                    updated_at_ms = ?
                WHERE cultivation_id = ?
                """,
                (
                    status.value,
                    int(cycles_completed),
                    float(coherence_score),
                    json.dumps(dict(coherence_detail), ensure_ascii=False),
                    json.dumps(list(regime_history), ensure_ascii=False),
                    now,
                    cultivation_id,
                ),
            )
        return await self.get(cultivation_id)

    async def update_status(
        self,
        *,
        cultivation_id: str,
        status: CultivationStatus,
        notes: str | None = None,
    ) -> CultivationRecordSpec:
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            if notes is None:
                self._registry.conn.execute(
                    "UPDATE cultivations SET status = ?, updated_at_ms = ? "
                    "WHERE cultivation_id = ?",
                    (status.value, now, cultivation_id),
                )
            else:
                self._registry.conn.execute(
                    "UPDATE cultivations SET status = ?, notes = ?, "
                    "updated_at_ms = ? WHERE cultivation_id = ?",
                    (status.value, notes, now, cultivation_id),
                )
        return await self.get(cultivation_id)

    async def set_eval_template(
        self,
        *,
        cultivation_id: str,
        dlaas_template_id: str,
        last_exam_run_id: str = "",
    ) -> CultivationRecordSpec:
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                "UPDATE cultivations SET dlaas_template_id = ?, "
                "last_exam_run_id = ?, updated_at_ms = ? "
                "WHERE cultivation_id = ?",
                (dlaas_template_id, last_exam_run_id, now, cultivation_id),
            )
        return await self.get(cultivation_id)

    async def set_inducted(
        self,
        *,
        cultivation_id: str,
        inducted_template_id: str,
    ) -> CultivationRecordSpec:
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                "UPDATE cultivations SET status = ?, inducted_template_id = ?, "
                "updated_at_ms = ? WHERE cultivation_id = ?",
                (
                    CultivationStatus.INDUCTED.value,
                    inducted_template_id,
                    now,
                    cultivation_id,
                ),
            )
        return await self.get(cultivation_id)


def _row_to_cultivation(row) -> CultivationRecordSpec:
    keys = row.keys() if hasattr(row, "keys") else ()
    package_id = row["package_id"] if "package_id" in keys else ""
    track_id = row["track_id"] if "track_id" in keys else ""
    direction_json = row["direction_json"] if "direction_json" in keys else "{}"
    return CultivationRecordSpec(
        cultivation_id=row["cultivation_id"],
        ai_id=row["ai_id"],
        slug=row["slug"],
        display_name=row["display_name"],
        domain=row["domain"],
        runtime_template_id=row["runtime_template_id"],
        seed_persona=json.loads(row["seed_persona_json"] or "{}"),
        curriculum=json.loads(row["curriculum_json"] or "{}"),
        status=CultivationStatus(row["status"]),
        cycles_completed=int(row["cycles_completed"]),
        coherence_score=float(row["coherence_score"]),
        coherence_detail=json.loads(row["coherence_detail_json"] or "{}"),
        regime_history=tuple(json.loads(row["regime_history_json"] or "[]")),
        dlaas_template_id=row["dlaas_template_id"],
        last_exam_run_id=row["last_exam_run_id"],
        inducted_template_id=row["inducted_template_id"],
        notes=row["notes"],
        package_id=package_id or "",
        track_id=track_id or "",
        direction=json.loads(direction_json or "{}"),
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )


__all__ = [
    "CultivationNotFound",
    "CultivationRecordSpec",
    "CultivationStatus",
    "CultivationStore",
]
