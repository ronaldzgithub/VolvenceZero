"""CrossGenerationAggregator (architecture-uplift A2 step 3 final tier).

Implements the schema + module skeleton defined in
[`docs/specs/evaluation-cascade.md`](../../../../../../../docs/specs/evaluation-cascade.md)
§A2.4. The aggregator is the cascade's interface to ModificationGate via
``ModificationGateEvidence``; ``audit_evidence_id`` is the explicit link to
the A5 audit owner (see [`audit-owner.md`](../../../../../../../docs/specs/audit-owner.md)).

Phased implementation:

- **T9 (this packet)**: schemas + module skeleton.
  ``process()`` returns empty evidence; DISABLED-by-default.
- **Future**: aggregate across multiple ``ExpensiveLayerSnapshot`` generations
  + compute ``head_to_head_aggregate_winrate`` + link audit_evidence_id when
  audit owner publishes (T11 + OA-4 ACTIVE).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
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
]


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


class CrossGenerationAggregatorModule(RuntimeModule[CrossGenerationAggregateSnapshot]):
    """Final tier of the cascade. DISABLED by default."""

    slot_name = "evaluation_cross_generation"
    owner = "CrossGenerationAggregatorModule"
    value_type = CrossGenerationAggregateSnapshot
    dependencies = ("evaluation_expensive",)
    default_wiring_level = WiringLevel.DISABLED

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[CrossGenerationAggregateSnapshot]:
        # fail-loudly on missing dependency
        expensive_snapshot = upstream["evaluation_expensive"]
        if not isinstance(expensive_snapshot.value, ExpensiveLayerSnapshot):
            return self.publish(_EMPTY_AGGREGATE_SNAPSHOT)
        return self.publish(
            build_cross_generation_aggregate_snapshot(
                expensive_snapshot=expensive_snapshot.value,
                timestamp_ms=expensive_snapshot.timestamp_ms,
            )
        )
