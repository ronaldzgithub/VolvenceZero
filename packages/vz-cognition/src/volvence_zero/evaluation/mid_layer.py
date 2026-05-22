"""Mid layer of the evaluation cascade (architecture-uplift A2 step 2).

Implements the schema + module skeleton defined in
[`docs/specs/evaluation-cascade.md`](../../../../../../../docs/specs/evaluation-cascade.md)
§A2.2. The module is **DISABLED-by-default** at the wiring layer; SHADOW
or ACTIVE promotion is gated by the SHADOW-evidence flow described in spec
§迁移协议 Step 2.

Phased implementation:

- **T8 (this packet)**: declare ``MidLayerSnapshot``,
  ``CounterfactualContributionReadout``, ``MidLayerScore``, and the
  ``MidLayerModule`` skeleton with ``default_wiring_level=DISABLED``. The
  module's ``process()`` returns an empty snapshot until step 2 of the
  cascade rollout (which lives in a future packet). Contract tests pin the
  schema shape so COG-1 / OA-4 evidence can plug into a stable surface.

- **Future**: implement actual ablation aggregation + counterfactual readout
  extraction from ``CreditSnapshot.counterfactual_readouts``; wire into
  paper-suite-small benchmark; produce SHADOW evidence document.

Failure semantics (spec §跨层 failure semantics F2):
- Any internal failure must ``raise`` and propagate; consumers receive the
  resulting placeholder snapshot via the kernel's standard
  ``DependencyGuard`` path. No silent fallback.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from volvence_zero.evaluation.cheap_layer import EvaluationCascadeRole
from volvence_zero.evaluation.types import EvaluationScore
from volvence_zero.runtime.kernel import (
    RuntimeModule,
    Snapshot,
    WiringLevel,
)

__all__ = [
    "MidLayerScore",
    "CounterfactualContributionReadout",
    "MidLayerSnapshot",
    "MidLayerModule",
]


# ---------------------------------------------------------------------------
# Schema (spec §A2.2)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MidLayerScore:
    """Aggregated ablation / multi-seed score from the mid tier.

    Distinct from ``EvaluationScore`` (cheap tier) because mid tier always
    has a ``baseline_label`` reference for ablation deltas; cheap tier does
    not have that context.
    """

    family: str
    metric_name: str
    value: float
    confidence: float
    baseline_label: str
    delta_vs_baseline: float | None
    evidence: str


@dataclasses.dataclass(frozen=True)
class CounterfactualContributionReadout:
    """COG-1 starting surface readout (spec §A2.2).

    Mirrors records from ``CreditSnapshot.counterfactual_readouts``; mid_layer
    does NOT recompute (R8 SSOT). Only aggregates / re-emits the existing
    credit owner outputs into a tier where benchmark hooks can pick them up.
    """

    record_id: str
    counterfactual_contribution_learned: float
    counterfactual_contribution_baseline: float
    contribution_delta: float
    confidence: float


@dataclasses.dataclass(frozen=True)
class MidLayerSnapshot:
    """Mid-tier evaluation snapshot.

    Independent value type from ``EvaluationSnapshot`` (spec §关键不变量 2);
    downstream consumers opt-in via the new ``evaluation_mid`` slot.
    """

    scenario_id: str
    seeds: tuple[int, ...]
    profile_label: str
    baseline_label: str
    aggregated_scores: tuple[MidLayerScore, ...]
    counterfactual_readouts: tuple[CounterfactualContributionReadout, ...]
    acceptance_gate_passed: bool
    acceptance_gate_reasons: tuple[str, ...]
    description: str
    cascade_role: EvaluationCascadeRole = EvaluationCascadeRole.MID_LAYER


# ---------------------------------------------------------------------------
# Module skeleton (spec §A2.2)
# ---------------------------------------------------------------------------


_EMPTY_MID_SNAPSHOT = MidLayerSnapshot(
    scenario_id="",
    seeds=(),
    profile_label="",
    baseline_label="",
    aggregated_scores=(),
    counterfactual_readouts=(),
    acceptance_gate_passed=True,
    acceptance_gate_reasons=(),
    description="mid_layer skeleton (T8 packet): no aggregation yet",
)


class MidLayerModule(RuntimeModule[MidLayerSnapshot]):
    """Mid-tier evaluation publisher.

    DISABLED by default — promotion to SHADOW / ACTIVE requires the SHADOW
    evidence packet described in spec §迁移协议 Step 2.

    The module declares ``"evaluation"`` + ``"credit"`` dependencies:
    it consumes the cheap_layer's frozen ``EvaluationSnapshot`` and the
    credit owner-published COG-1 readouts. It does not recompute credit.
    """

    slot_name = "evaluation_mid"
    owner = "MidLayerModule"
    value_type = MidLayerSnapshot
    dependencies = ("evaluation", "credit")
    default_wiring_level = WiringLevel.DISABLED

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[MidLayerSnapshot]:
        """Emit COG-1 readout surfaces from the credit owner."""
        # Sanity: cheap_layer snapshot must be present per declared dependency.
        # Use direct dict-style consumption so missing slot raises (fail-loudly)
        # — kernel UpstreamView already enforces this for declared dependencies.
        cheap_snapshot = upstream["evaluation"]
        del cheap_snapshot
        credit_snapshot = upstream["credit"]
        credit_value = credit_snapshot.value
        from volvence_zero.credit.gate import CreditSnapshot

        if not isinstance(credit_value, CreditSnapshot):
            return self.publish(_EMPTY_MID_SNAPSHOT)
        counterfactual_readouts = tuple(
            CounterfactualContributionReadout(
                record_id=readout.source_event,
                counterfactual_contribution_learned=readout.learned_contribution,
                counterfactual_contribution_baseline=readout.historical_baseline,
                contribution_delta=(
                    readout.learned_contribution
                    - (readout.actual_outcome - readout.historical_baseline)
                ),
                confidence=0.70 if readout.update_count else 0.45,
            )
            for readout in credit_value.counterfactual_readouts
        )
        scores: list[MidLayerScore] = []
        least_control = credit_value.least_control_readout
        if least_control is not None:
            scores.append(
                MidLayerScore(
                    family="learning",
                    metric_name="least_control_score",
                    value=least_control.least_control_score,
                    confidence=0.60,
                    baseline_label="",
                    delta_vs_baseline=None,
                    evidence=least_control.description,
                )
            )
            scores.append(
                MidLayerScore(
                    family="learning",
                    metric_name="least_control_effort",
                    value=least_control.control_effort,
                    confidence=0.60,
                    baseline_label="",
                    delta_vs_baseline=None,
                    evidence=least_control.description,
                )
            )
        snapshot = MidLayerSnapshot(
            scenario_id="",
            seeds=(),
            profile_label="",
            baseline_label="",
            aggregated_scores=tuple(scores),
            counterfactual_readouts=counterfactual_readouts,
            acceptance_gate_passed=True,
            acceptance_gate_reasons=(),
            description=(
                "mid_layer COG-1 readout extraction: "
                f"{len(counterfactual_readouts)} counterfactual readouts, "
                f"{len(scores)} least-control scores."
            ),
        )
        return self.publish(snapshot)
