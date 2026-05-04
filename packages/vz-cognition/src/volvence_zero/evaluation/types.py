"""Evaluation contract surface (R8 / R11).

Pure data-only module. Holds ``EvaluationSnapshot`` / ``EvaluationReport``
and the evolution-judgment / replay / longitudinal / metric-interval /
pairwise-effect dataclasses consumed by :class:`EvaluationBackbone` and
downstream dialogue / paper-suite runners.

Slice S.2 (2026-05-04): extracted from the previous monolithic
``evaluation/backbone.py``. External consumers import these via the
``volvence_zero.evaluation`` package facade unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemorySnapshot
from volvence_zero.substrate import SubstrateSnapshot


class EvaluationTrack(str, Enum):
    WORLD = "world"
    SELF = "self"
    CROSS = "cross"


@dataclass(frozen=True)
class EvaluationScore:
    family: str
    metric_name: str
    value: float
    confidence: float
    evidence: str


@dataclass(frozen=True)
class EvaluationSnapshot:
    turn_scores: tuple[EvaluationScore, ...]
    session_scores: tuple[EvaluationScore, ...]
    alerts: tuple[str, ...]
    description: str
    reflection_accuracy: float = 0.0
    longitudinal_verdict: str = ""


@dataclass(frozen=True)
class EvaluationRecord:
    record_id: str
    session_id: str
    wave_id: str | None
    timestamp_ms: int
    timescale: str
    family: str
    metric_name: str
    value: float
    confidence: float
    track: str
    evidence: str
    signal_sources: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    report_id: str
    report_type: str
    timestamp_ms: int
    session_ids: tuple[str, ...]
    scores_by_family: tuple[tuple[str, tuple[EvaluationRecord, ...]], ...]
    alerts: tuple[tuple[str, str], ...]
    trends: tuple[tuple[str, str, float], ...]
    recommendations: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class EvaluationReplayCase:
    case_id: str
    session_id: str
    wave_id: str
    substrate_snapshot: SubstrateSnapshot | None
    memory_snapshot: MemorySnapshot | None
    dual_track_snapshot: DualTrackSnapshot | None
    metric_floors: tuple[tuple[str, float], ...] = ()
    max_alert_count: int = 0


@dataclass(frozen=True)
class EvaluationReplayCaseResult:
    case_id: str
    passed: bool
    observed_metrics: tuple[tuple[str, float], ...]
    alerts: tuple[str, ...]
    issues: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class EvaluationReplaySuiteResult:
    suite_name: str
    passed: bool
    case_results: tuple[EvaluationReplayCaseResult, ...]
    description: str


class EvolutionDecision(str, Enum):
    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


class JudgementCategory(str, Enum):
    REAL_IMPROVEMENT = "real_improvement"
    STYLE_DRIFT = "style_drift"
    UNSAFE_MUTATION = "unsafe_mutation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class EvolutionJudgement:
    decision: EvolutionDecision
    category: JudgementCategory
    replay_passed: bool
    abstraction_trend: float
    learning_trend: float
    relationship_trend: float
    reasons: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class CrossSessionGrowthReport:
    window_trends: tuple[tuple[int, tuple[tuple[str, float], ...]], ...]
    family_persistence: float
    regime_effectiveness_delta: float
    verdict: str
    description: str


@dataclass(frozen=True)
class LongitudinalReport:
    cross_session: CrossSessionGrowthReport
    dimension_trends: tuple[tuple[str, float, float], ...]
    family_survival_curves: tuple[tuple[str, tuple[float, ...]], ...]
    regime_effectiveness_curves: tuple[tuple[str, tuple[float, ...]], ...]
    verdict: str
    description: str


@dataclass(frozen=True)
class MetricIntervalSummary:
    metric_name: str
    sample_count: int
    mean: float
    std: float
    stderr: float
    ci_low: float
    ci_high: float
    min_value: float
    max_value: float
    description: str


@dataclass(frozen=True)
class PairwiseMetricEffect:
    metric_name: str
    candidate_label: str
    control_label: str
    sample_count: int
    mean_delta: float
    std_delta: float
    stderr_delta: float
    ci_low: float
    ci_high: float
    effect_size: float
    description: str


@dataclass(frozen=True)
class CrossSessionBenchmarkSuite:
    session_reports: tuple[EvaluationReport, ...]
    comparison_windows: tuple[int, ...] = (1, 3, 5)

