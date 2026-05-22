"""Unified facade for the evaluation cascade.

The individual tier modules are intentionally small and independently
owned. This facade gives downstream experiment / gate code one stable import
surface without changing any runtime wiring behavior.
"""

from __future__ import annotations

from volvence_zero.evaluation.cheap_layer import (
    EVALUATION_SNAPSHOT_FIELD_NAMES,
    EvaluationCascadeRole,
    EvaluationCheapLayer,
)
from volvence_zero.evaluation.cross_generation_aggregator import (
    CrossGenerationAggregateSnapshot,
    CrossGenerationAggregatorModule,
    ModificationGateEvidence,
)
from volvence_zero.evaluation.expensive_layer import (
    ExpensiveLayerModule,
    ExpensiveLayerSnapshot,
    HeadToHeadResult,
    LlmJudgeReadout,
)
from volvence_zero.evaluation.mid_layer import (
    CounterfactualContributionReadout,
    MidLayerModule,
    MidLayerScore,
    MidLayerSnapshot,
)

__all__ = [
    "CounterfactualContributionReadout",
    "CrossGenerationAggregateSnapshot",
    "CrossGenerationAggregatorModule",
    "EVALUATION_SNAPSHOT_FIELD_NAMES",
    "EvaluationCascadeRole",
    "EvaluationCheapLayer",
    "ExpensiveLayerModule",
    "ExpensiveLayerSnapshot",
    "HeadToHeadResult",
    "LlmJudgeReadout",
    "MidLayerModule",
    "MidLayerScore",
    "MidLayerSnapshot",
    "ModificationGateEvidence",
]
