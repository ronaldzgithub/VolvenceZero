"""CrossGenerationAggregator (architecture-uplift A2 step 3 final tier).

Implements the schema + module skeleton defined in
[`docs/specs/evaluation-cascade.md`](../../../../../../../docs/specs/evaluation-cascade.md)
§A2.4. The aggregator is the cascade's interface to ModificationGate via
``ModificationGateEvidence``; ``audit_evidence_id`` is the explicit link to
the A5 audit owner (see [`audit-owner.md`](../../../../../../../docs/specs/audit-owner.md)).

Phased implementation:

- **T9**: schemas + module skeleton.
- **C4 (2026-07-16)**: real multi-generation aggregation — the module
  keeps a bounded window of recent ``ExpensiveLayerSnapshot`` generations
  and aggregates all head-to-head rows across the window into one
  ``ModificationGateEvidence``. ``audit_evidence_id`` link stays ``None``
  until the audit owner is ACTIVE (OA-4). DISABLED-by-default unchanged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from volvence_zero.evaluation.cheap_layer import EvaluationCascadeRole
from volvence_zero.evaluation.expensive_layer import ExpensiveLayerSnapshot, HeadToHeadResult
from volvence_zero.runtime.kernel import (
    RuntimeModule,
    Snapshot,
    WiringLevel,
)

__all__ = [
    "ModificationGateEvidence",
    "CrossGenerationAggregateSnapshot",
    "CrossGenerationAggregatorModule",
    "build_cross_generation_aggregate_snapshot",
    "build_cross_generation_window_snapshot",
]

# Bounded generation window: rare-heavy cadence means a handful of
# generations is already a long horizon; older evidence should age out.
_GENERATION_WINDOW_SIZE = 5


@dataclasses.dataclass(frozen=True)
class ModificationGateEvidence:
    """Frozen evidence consumed by ModificationGate (spec §A2.4).

    Three-channel evidence aggregation:
    1. ``validation_score`` — calibrated evaluation readout (from cascade)
    2. ``head_to_head_aggregate_winrate`` — DM-7 / EVO-6 hard evidence
    3. ``audit_evidence_id`` — link to A5 ``AuditSnapshot.audit_id``
       (``None`` until OA-4 lands and audit owner becomes ACTIVE)

    ``rollback_evidence_present`` + ``capacity_within_cap`` are mirrors of
    structural evidence still passed via ``ModificationProposal`` (kept here
    so ModificationGate has one place to fetch all three categories of
    evidence after T11 lands).
    """

    evidence_id: str
    validation_score: float
    head_to_head_aggregate_winrate: float
    rollback_evidence_present: bool
    capacity_within_cap: bool
    audit_evidence_id: str | None
    notes: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class CrossGenerationAggregateSnapshot:
    """Cross-generation aggregator output."""

    aggregator_id: str
    timestamp_ms: int
    generation_id_window: tuple[str, ...]
    head_to_head_table: tuple[HeadToHeadResult, ...]
    modification_gate_evidence: ModificationGateEvidence
    description: str
    cascade_role: EvaluationCascadeRole = (
        EvaluationCascadeRole.CROSS_GENERATION_AGGREGATOR
    )


_EMPTY_GATE_EVIDENCE = ModificationGateEvidence(
    evidence_id="",
    validation_score=0.0,
    head_to_head_aggregate_winrate=0.0,
    rollback_evidence_present=False,
    capacity_within_cap=True,
    audit_evidence_id=None,
)

_EMPTY_AGGREGATE_SNAPSHOT = CrossGenerationAggregateSnapshot(
    aggregator_id="",
    timestamp_ms=0,
    generation_id_window=(),
    head_to_head_table=(),
    modification_gate_evidence=_EMPTY_GATE_EVIDENCE,
    description="cross_generation_aggregator skeleton (T9 packet)",
)


def build_cross_generation_aggregate_snapshot(
    *,
    expensive_snapshot: ExpensiveLayerSnapshot,
    timestamp_ms: int = 0,
    aggregator_id: str = "cross-generation-aggregate",
) -> CrossGenerationAggregateSnapshot:
    """Build a cross-generation aggregate from expensive-layer evidence.

    Empty head-to-head input preserves the old skeleton behaviour. LLM judge
    readouts remain ignored for gate evidence per R12 / Mind-Face isolation.
    """
    if not expensive_snapshot.head_to_head_results:
        return _EMPTY_AGGREGATE_SNAPSHOT

    winrates = tuple(
        result.winrate_a_vs_b
        for result in expensive_snapshot.head_to_head_results
    )
    aggregate_winrate = round(sum(winrates) / len(winrates), 4)
    validation_score = max(0.0, min(1.0, aggregate_winrate))
    generation_window = (
        (expensive_snapshot.generation_id,)
        if expensive_snapshot.generation_id
        else ()
    )
    evidence = ModificationGateEvidence(
        evidence_id=(
            f"xgen:{expensive_snapshot.generation_id or 'unknown'}:"
            f"{len(expensive_snapshot.head_to_head_results)}"
        ),
        validation_score=validation_score,
        head_to_head_aggregate_winrate=aggregate_winrate,
        rollback_evidence_present=False,
        capacity_within_cap=True,
        audit_evidence_id=None,
        notes=(
            "Built from ExpensiveLayerSnapshot.head_to_head_results only; "
            "LLM judge readouts ignored for gate evidence.",
        ),
    )
    return CrossGenerationAggregateSnapshot(
        aggregator_id=aggregator_id,
        timestamp_ms=timestamp_ms,
        generation_id_window=generation_window,
        head_to_head_table=expensive_snapshot.head_to_head_results,
        modification_gate_evidence=evidence,
        description=(
            f"Cross-generation aggregate over "
            f"{len(expensive_snapshot.head_to_head_results)} head-to-head rows; "
            f"winrate={aggregate_winrate:.3f}."
        ),
    )


def build_cross_generation_window_snapshot(
    *,
    window: Sequence[ExpensiveLayerSnapshot],
    timestamp_ms: int = 0,
    aggregator_id: str = "cross-generation-aggregate",
) -> CrossGenerationAggregateSnapshot:
    """Aggregate head-to-head evidence across a window of generations.

    All head-to-head rows across the window feed one aggregate winrate;
    LLM judge readouts stay excluded from gate evidence (R12 / Mind-Face
    isolation). Empty window / no rows preserves skeleton behaviour.
    """

    rows: list[HeadToHeadResult] = []
    generation_ids: list[str] = []
    for snapshot in window:
        if not snapshot.head_to_head_results:
            continue
        rows.extend(snapshot.head_to_head_results)
        if snapshot.generation_id:
            generation_ids.append(snapshot.generation_id)
    if not rows:
        return _EMPTY_AGGREGATE_SNAPSHOT

    aggregate_winrate = round(
        sum(row.winrate_a_vs_b for row in rows) / len(rows), 4
    )
    validation_score = max(0.0, min(1.0, aggregate_winrate))
    evidence = ModificationGateEvidence(
        evidence_id=f"xgen-window:{len(generation_ids)}gen:{len(rows)}rows",
        validation_score=validation_score,
        head_to_head_aggregate_winrate=aggregate_winrate,
        rollback_evidence_present=False,
        capacity_within_cap=True,
        audit_evidence_id=None,
        notes=(
            f"Aggregated over {len(generation_ids)} generations "
            f"({len(rows)} head-to-head rows); LLM judge readouts ignored "
            "for gate evidence.",
        ),
    )
    return CrossGenerationAggregateSnapshot(
        aggregator_id=aggregator_id,
        timestamp_ms=timestamp_ms,
        generation_id_window=tuple(generation_ids),
        head_to_head_table=tuple(rows),
        modification_gate_evidence=evidence,
        description=(
            f"Cross-generation window aggregate over {len(generation_ids)} "
            f"generations / {len(rows)} head-to-head rows; "
            f"winrate={aggregate_winrate:.3f}."
        ),
    )


class CrossGenerationAggregatorModule(RuntimeModule[CrossGenerationAggregateSnapshot]):
    """Final tier of the cascade. DISABLED by default.

    C4: keeps a bounded window of recent generations (keyed by
    ``generation_id``, oldest evicted first) and publishes the window
    aggregate instead of only mirroring the latest generation.
    """

    slot_name = "evaluation_cross_generation"
    owner = "CrossGenerationAggregatorModule"
    value_type = CrossGenerationAggregateSnapshot
    dependencies = ("evaluation_expensive",)
    default_wiring_level = WiringLevel.DISABLED

    def __init__(self, *, wiring_level: WiringLevel | None = None) -> None:
        super().__init__(wiring_level=wiring_level)
        self._generation_window: dict[str, ExpensiveLayerSnapshot] = {}

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[CrossGenerationAggregateSnapshot]:
        # fail-loudly on missing dependency
        expensive_snapshot = upstream["evaluation_expensive"]
        if not isinstance(expensive_snapshot.value, ExpensiveLayerSnapshot):
            return self.publish(_EMPTY_AGGREGATE_SNAPSHOT)
        expensive_value = expensive_snapshot.value
        if expensive_value.generation_id and expensive_value.head_to_head_results:
            self._generation_window[expensive_value.generation_id] = expensive_value
            while len(self._generation_window) > _GENERATION_WINDOW_SIZE:
                oldest = next(iter(self._generation_window))
                del self._generation_window[oldest]
        if not self._generation_window:
            return self.publish(_EMPTY_AGGREGATE_SNAPSHOT)
        return self.publish(
            build_cross_generation_window_snapshot(
                window=tuple(self._generation_window.values()),
                timestamp_ms=expensive_snapshot.timestamp_ms,
            )
        )
