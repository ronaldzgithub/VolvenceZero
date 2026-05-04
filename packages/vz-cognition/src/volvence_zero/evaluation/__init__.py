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
"""

from __future__ import annotations

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
    "CrossSessionBenchmarkSuite",
    "CrossSessionGrowthReport",
    "EvaluationBackbone",
    "EvolutionDecision",
    "EvolutionJudgement",
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
    "LongitudinalReport",
    "MetricIntervalSummary",
    "PairwiseMetricEffect",
    "build_metric_interval_summaries",
    "build_metric_interval_summary",
    "build_pairwise_metric_effect",
]
