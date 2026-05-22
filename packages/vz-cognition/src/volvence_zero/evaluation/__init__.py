"""Evaluation package facade.

Re-exports the evaluation contract surface, statistical / semantic /
replay helpers, and the main ``EvaluationBackbone`` + ``EvaluationModule``.
External consumers keep using
``from volvence_zero.evaluation import X`` unchanged after Slice S.2.

Capability-domain files:

* :mod:`.types`              — Enums + ``Evaluation*`` dataclasses
* :mod:`.statistics`         — clamp / std / percentile / CI / pairwise effect
* :mod:`.semantic_readouts`  — char-level embedding + prototype vectors
* :mod:`.replay_scenarios`   — default replay cases + substrate builder
* :mod:`.report_helpers`     — small report read helpers
* :mod:`.backbone`           — ``EvaluationBackbone`` + ``EvaluationModule``
* :mod:`.cascade`            — cheap / mid / expensive / cross-generation facade
"""

from __future__ import annotations

from volvence_zero.evaluation.cascade import (
    CounterfactualContributionReadout,
    CrossGenerationAggregateSnapshot,
    CrossGenerationAggregatorModule,
    EvaluationCascadeRole,
    EvaluationCheapLayer,
    ExpensiveLayerModule,
    ExpensiveLayerSnapshot,
    HeadToHeadResult,
    LlmJudgeReadout,
    MidLayerModule,
    MidLayerScore,
    MidLayerSnapshot,
    ModificationGateEvidence,
)
from volvence_zero.evaluation.types import (
    CrossSessionBenchmarkSuite,
    CrossSessionGrowthReport,
    EvaluationAlert,
    EvaluationRecord,
    EvaluationReplayCase,
    EvaluationReplayCaseResult,
    EvaluationReplaySuiteResult,
    EvaluationReport,
    EvaluationScore,
    EvaluationSnapshot,
    EvaluationTrack,
    EvolutionDecision,
    EvolutionJudgement,
    JudgementCategory,
    LongitudinalReport,
    MetricIntervalSummary,
    PairwiseMetricEffect,
)
from volvence_zero.evaluation.statistics import (
    build_metric_interval_summaries,
    build_metric_interval_summary,
    build_pairwise_metric_effect,
)
from volvence_zero.evaluation.backbone import (
    EvaluationBackbone,
    EvaluationModule,
)

__all__ = [
    "CounterfactualContributionReadout",
    "CrossGenerationAggregateSnapshot",
    "CrossGenerationAggregatorModule",
    "CrossSessionBenchmarkSuite",
    "CrossSessionGrowthReport",
    "EvaluationBackbone",
    "EvaluationCascadeRole",
    "EvaluationCheapLayer",
    "EvolutionDecision",
    "EvolutionJudgement",
    "ExpensiveLayerModule",
    "ExpensiveLayerSnapshot",
    "HeadToHeadResult",
    "JudgementCategory",
    "EvaluationReplayCase",
    "EvaluationReplayCaseResult",
    "EvaluationReplaySuiteResult",
    "EvaluationModule",
    "EvaluationAlert",
    "EvaluationRecord",
    "EvaluationReport",
    "EvaluationScore",
    "EvaluationSnapshot",
    "EvaluationTrack",
    "LlmJudgeReadout",
    "LongitudinalReport",
    "MetricIntervalSummary",
    "MidLayerModule",
    "MidLayerScore",
    "MidLayerSnapshot",
    "ModificationGateEvidence",
    "PairwiseMetricEffect",
    "build_metric_interval_summaries",
    "build_metric_interval_summary",
    "build_pairwise_metric_effect",
]
