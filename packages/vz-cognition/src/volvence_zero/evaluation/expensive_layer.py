"""Expensive layer of the evaluation cascade (architecture-uplift A2 step 3).

Implements the schema + module skeleton defined in
[`docs/specs/evaluation-cascade.md`](../../../../../../../docs/specs/evaluation-cascade.md)
§A2.3.

Key invariants (spec §关键不变量 3):

- LLM-judge ``readouts`` are advisory only; ``is_gate_eligible`` is
  permanently ``False`` (enforced by ``tests/contracts/test_evaluation_cascade_readouts.py``)
- Plugged into the cascade via dependency on ``evaluation_mid``
- DISABLED by default; promotion gated by ETA strong-proof PASS

Phased implementation:

- **T9 (this packet)**: schemas + module skeleton + ``is_gate_eligible``
  constant; ``process()`` returns empty snapshot.
- **Future**: head-to-head winrate computation + LLM-judge readout via
  centralised prompt (`llm-prompt-centralization.mdc`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from volvence_zero.evaluation.cheap_layer import EvaluationCascadeRole
from volvence_zero.evaluation.mid_layer import MidLayerScore
from volvence_zero.runtime.kernel import (
    RuntimeModule,
    Snapshot,
    WiringLevel,
)

__all__ = [
    "HeadToHeadResult",
    "LlmJudgeReadout",
    "ExpensiveLayerSnapshot",
    "ExpensiveLayerModule",
]


@dataclasses.dataclass(frozen=True)
class HeadToHeadResult:
    """DM-7 / EVO-6 head-to-head winrate (spec §A2.3)."""

    profile_a: str
    profile_b: str
    case_count: int
    winrate_a_vs_b: float
    confidence_interval_low: float
    confidence_interval_high: float
    judge_kind: str  # "deterministic" / "llm-readout"
    notes: str


@dataclasses.dataclass(frozen=True)
class LlmJudgeReadout:
    """LLM-judge readout (spec §A2.3).

    ``is_gate_eligible`` is permanently ``False`` (R12 + OA-2 isolation).
    Contract test ``test_llm_judge_readout_never_gate_eligible`` enforces
    this at module load.
    """

    case_id: str
    judge_model_id: str
    naturalness_score: float
    coherence_score: float
    note: str
    is_gate_eligible: bool = False


@dataclasses.dataclass(frozen=True)
class ExpensiveLayerSnapshot:
    """Expensive-tier evaluation snapshot."""

    generation_id: str
    head_to_head_results: tuple[HeadToHeadResult, ...]
    llm_judge_readouts: tuple[LlmJudgeReadout, ...]
    aggregated_scores: tuple[MidLayerScore, ...]
    description: str
    cascade_role: EvaluationCascadeRole = EvaluationCascadeRole.EXPENSIVE_LAYER


_EMPTY_EXPENSIVE_SNAPSHOT = ExpensiveLayerSnapshot(
    generation_id="",
    head_to_head_results=(),
    llm_judge_readouts=(),
    aggregated_scores=(),
    description="expensive_layer skeleton (T9 packet): no aggregation yet",
)


class ExpensiveLayerModule(RuntimeModule[ExpensiveLayerSnapshot]):
    """Expensive-tier evaluation publisher.

    DISABLED by default; promotion gated by ETA strong-proof PASS per spec
    §迁移协议 Step 3.
    """

    slot_name = "evaluation_expensive"
    owner = "ExpensiveLayerModule"
    value_type = ExpensiveLayerSnapshot
    dependencies = ("evaluation_mid",)
    default_wiring_level = WiringLevel.DISABLED

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ExpensiveLayerSnapshot]:
        # fail-loudly on missing dependency (kernel UpstreamView guards this)
        mid_snapshot = upstream["evaluation_mid"]
        del mid_snapshot
        return self.publish(_EMPTY_EXPENSIVE_SNAPSHOT)
