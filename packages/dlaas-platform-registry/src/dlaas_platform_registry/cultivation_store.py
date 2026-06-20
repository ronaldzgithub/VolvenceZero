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
import sqlite3
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
    # Operator-held: ticks are gated until resume (self-learning
    # workshop supervision). A paused cultivation keeps all its
    # accumulated study progress; resume restores the prior runnable
    # status recorded in ``notes``-independent state.
    PAUSED = "paused"


@dataclass(frozen=True)
class CultivationRecordSpec:
    """Immutable snapshot of one cultivation row."""

    cultivation_id: str
    ai_id: str
    slug: str
    display_name: str
    domain: str
    runtime_template_id: str
    # Owning tenant. Empty string = system-owned (operator-created via
    # the control-plane secret); non-empty = the BFF tenant that seeded
    # this cultivation through tenant credentials.
    tenant_id: str = ""
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
    # Schema v11: when non-empty, this cultivation was seeded from an
    # existing baked persona/template (an "adopted seed") rather than an
    # empty seed; the adopted template's converged school is hydrated
    # into the study instance at acquire time. Provenance only.
    source_template_id: str = ""
    # Schema v12: adopted-seed role semantics + continuation provenance.
    # ``source_kind`` / ``source_angle`` preserve whether the source was a
    # character / author / interpreter / expert; ``continuation_mode`` is
    # ``protocol_bundle`` (true learned-state continuation) or
    # ``metadata_only`` (persona anchor only). Readout/audit only.
    provenance: Mapping[str, Any] = field(default_factory=dict)
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
            "tenant_id": self.tenant_id,
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
            "source_template_id": self.source_template_id,
            "provenance": dict(self.provenance),
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
        _ensure_tenant_column(registry)
        _ensure_source_template_column(registry)
        _ensure_provenance_column(registry)
        _ensure_events_table(registry)

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
        tenant_id: str = "",
        notes: str = "",
        package_id: str = "",
        track_id: str = "",
        direction: Mapping[str, Any] | None = None,
        source_template_id: str = "",
        provenance: Mapping[str, Any] | None = None,
    ) -> CultivationRecordSpec:
        cultivation_id = cultivation_id or _fresh_cultivation_id()
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO cultivations (
                    cultivation_id, ai_id, slug, display_name, domain,
                    runtime_template_id, tenant_id, seed_persona_json,
                    curriculum_json,
                    status, notes, package_id, track_id, direction_json,
                    source_template_id, provenance_json,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'seeding', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cultivation_id,
                    ai_id,
                    slug,
                    display_name,
                    domain,
                    runtime_template_id,
                    tenant_id,
                    json.dumps(dict(seed_persona), ensure_ascii=False),
                    json.dumps(dict(curriculum), ensure_ascii=False),
                    notes,
                    package_id,
                    track_id,
                    json.dumps(dict(direction or {}), ensure_ascii=False),
                    source_template_id,
                    json.dumps(dict(provenance or {}), ensure_ascii=False),
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

    async def list_all(
        self, *, tenant_id: str = ""
    ) -> tuple[CultivationRecordSpec, ...]:
        """List cultivations, newest first.

        Empty ``tenant_id`` returns every record (operator view);
        non-empty restricts to that tenant's own records (tenant view).
        """

        if tenant_id:
            rows = self._registry.conn.execute(
                "SELECT * FROM cultivations WHERE tenant_id = ? "
                "ORDER BY created_at_ms DESC",
                (tenant_id,),
            ).fetchall()
        else:
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

    # ------------------------------------------------------------------
    # Monitoring event log (self-learning workshop)
    # ------------------------------------------------------------------
    #
    # Append-only readout of what the autonomous loop did each cycle plus
    # operator teach corrections. This is governance/observability data,
    # never a learning signal that flows back into the kernel (R8 / R12):
    # the cognition owner is still the per-ai_id session.

    async def append_events(
        self,
        *,
        cultivation_id: str,
        kind: str,
        events: tuple[Mapping[str, Any], ...],
    ) -> int:
        """Append a batch of typed events for one cultivation.

        ``events`` are the engine's per-cycle ``CycleEvent.to_json()``
        payloads (kind=``"cycle"``) or a single operator-correction
        payload (kind=``"teach"``). The ``seq`` is read from each
        event's ``cycle_index`` when present, else a monotonic-by-time
        ordering via ``recorded_at_ms``. Returns the number written.
        """

        if not events:
            return 0
        now = int(time.time() * 1000.0)
        rows = []
        for event in events:
            seq_raw = event.get("cycle_index", 0)
            try:
                seq = int(seq_raw)
            except (TypeError, ValueError):
                seq = 0
            rows.append(
                (
                    f"cev_{secrets.token_hex(6)}",
                    cultivation_id,
                    seq,
                    kind,
                    json.dumps(dict(event), ensure_ascii=False),
                    now,
                )
            )
        async with self._registry.write_lock:
            self._registry.conn.executemany(
                """
                INSERT INTO cultivation_events (
                    event_id, cultivation_id, seq, kind, event_json,
                    recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    async def list_events(
        self,
        cultivation_id: str,
        *,
        limit: int = 500,
        kinds: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return the cultivation's event log, oldest first (capped).

        Each item is ``{event_id, seq, kind, recorded_at_ms, **payload}``
        so the operator console can render the per-cycle research/study
        trail, teach corrections, supervision actions and per-tick
        ``progress`` snapshots without reshaping on the client. When
        ``kinds`` is given, only those event kinds are returned (e.g.
        split the convergence ``timeline`` from the event log).
        """

        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            rows = self._registry.conn.execute(
                "SELECT * FROM cultivation_events WHERE cultivation_id = ? "
                f"AND kind IN ({placeholders}) "
                "ORDER BY recorded_at_ms ASC, seq ASC LIMIT ?",
                (cultivation_id, *kinds, int(limit)),
            ).fetchall()
        else:
            rows = self._registry.conn.execute(
                "SELECT * FROM cultivation_events WHERE cultivation_id = ? "
                "ORDER BY recorded_at_ms ASC, seq ASC LIMIT ?",
                (cultivation_id, int(limit)),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["event_json"] or "{}")
            out.append(
                {
                    "event_id": row["event_id"],
                    "seq": int(row["seq"]),
                    "kind": row["kind"],
                    "recorded_at_ms": int(row["recorded_at_ms"]),
                    **payload,
                }
            )
        return tuple(out)


def _ensure_tenant_column(registry: Registry) -> None:
    """Lazily add the ``tenant_id`` column to ``cultivations``.

    Forward-only schema delta owned by this store (the base table is
    created in :mod:`dlaas_platform_registry.db`). Mirrors the
    ``_apply_forward_migrations`` style: on SQLite the duplicate-column
    error is the documented no-op; Postgres supports ``IF NOT EXISTS``
    natively so re-runs are idempotent there too. Any other failure is
    re-raised — a half-migrated schema must fail loudly.
    """

    ddl = "ALTER TABLE cultivations ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''"
    if registry.backend == "postgres":
        registry.conn.execute(
            "ALTER TABLE cultivations "
            "ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''"
        )
        return
    try:
        registry.conn.execute(ddl)
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def _ensure_source_template_column(registry: Registry) -> None:
    """Lazily add the schema-v11 ``source_template_id`` column.

    Same forward-only delta contract as :func:`_ensure_tenant_column`
    (adopted-seed provenance). Empty default = legacy empty-seed
    cultivation.
    """

    if registry.backend == "postgres":
        registry.conn.execute(
            "ALTER TABLE cultivations "
            "ADD COLUMN IF NOT EXISTS source_template_id TEXT NOT NULL DEFAULT ''"
        )
        return
    try:
        registry.conn.execute(
            "ALTER TABLE cultivations "
            "ADD COLUMN source_template_id TEXT NOT NULL DEFAULT ''"
        )
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def _ensure_provenance_column(registry: Registry) -> None:
    """Lazily add the schema-v12 ``provenance_json`` column.

    Holds adopted-seed role semantics + continuation mode. Same
    forward-only delta contract as :func:`_ensure_tenant_column`.
    """

    if registry.backend == "postgres":
        registry.conn.execute(
            "ALTER TABLE cultivations "
            "ADD COLUMN IF NOT EXISTS provenance_json TEXT NOT NULL DEFAULT '{}'"
        )
        return
    try:
        registry.conn.execute(
            "ALTER TABLE cultivations "
            "ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{}'"
        )
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def _ensure_events_table(registry: Registry) -> None:
    """Lazily create the schema-v11 ``cultivation_events`` table.

    ``CREATE TABLE IF NOT EXISTS`` is idempotent on both backends, so a
    re-run is a no-op; any other failure is a real error and re-raised.
    """

    registry.conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cultivation_events (
            event_id TEXT PRIMARY KEY,
            cultivation_id TEXT NOT NULL,
            seq INTEGER NOT NULL DEFAULT 0,
            kind TEXT NOT NULL DEFAULT 'cycle',
            event_json TEXT NOT NULL DEFAULT '{}',
            recorded_at_ms INTEGER NOT NULL
        )
        """
    )


def _row_to_cultivation(row) -> CultivationRecordSpec:
    keys = row.keys() if hasattr(row, "keys") else ()
    package_id = row["package_id"] if "package_id" in keys else ""
    track_id = row["track_id"] if "track_id" in keys else ""
    direction_json = row["direction_json"] if "direction_json" in keys else "{}"
    tenant_id = row["tenant_id"] if "tenant_id" in keys else ""
    source_template_id = (
        row["source_template_id"] if "source_template_id" in keys else ""
    )
    provenance_json = (
        row["provenance_json"] if "provenance_json" in keys else "{}"
    )
    return CultivationRecordSpec(
        cultivation_id=row["cultivation_id"],
        ai_id=row["ai_id"],
        slug=row["slug"],
        display_name=row["display_name"],
        domain=row["domain"],
        runtime_template_id=row["runtime_template_id"],
        tenant_id=tenant_id or "",
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
        source_template_id=source_template_id or "",
        provenance=json.loads(provenance_json or "{}"),
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )


__all__ = [
    "CultivationNotFound",
    "CultivationRecordSpec",
    "CultivationStatus",
    "CultivationStore",
]
