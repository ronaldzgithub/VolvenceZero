from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Mapping
from uuid import uuid4

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate import FeatureSignal, SubstrateSnapshot, SurfaceKind, feature_signal_value

if TYPE_CHECKING:
    from volvence_zero.joint_loop.runtime import ScheduledJointLoopResult
    from volvence_zero.reflection import ReflectionSnapshot
    from volvence_zero.regime.identity import RegimeSnapshot
    from volvence_zero.temporal.interface import TemporalAbstractionSnapshot
    from volvence_zero.temporal.interface import MetacontrollerRuntimeState


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


@dataclass(frozen=True)
class EvolutionJudgement:
    decision: EvolutionDecision
    replay_passed: bool
    abstraction_trend: float
    learning_trend: float
    relationship_trend: float
    reasons: tuple[str, ...]
    description: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _semantic_embedding(text: str, *, dim: int = 8) -> tuple[float, ...]:
    tokens = _semantic_tokens(text)
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        token_scale = max(len(token), 1)
        for index, char in enumerate(token):
            vector[(index + len(token)) % dim] += (ord(char) % 41) / 41.0 / token_scale
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def _semantic_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    ascii_buffer: list[str] = []
    compact = "".join(char for char in text.lower() if not char.isspace())
    for char in text.lower():
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            continue
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer.clear()
        if not char.isspace():
            tokens.append(char)
    if ascii_buffer:
        tokens.append("".join(ascii_buffer))
    tokens.extend(compact[index : index + 2] for index in range(len(compact) - 1))
    return tuple(tokens)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


TASK_PRESSURE_PROTOTYPE = _semantic_embedding(
    "decide priority execute plan concrete action urgency next step tradeoff schedule "
    "明确顺序 执行次序 取舍理由 直接判断 推进任务"
)
SUPPORT_PRESENCE_PROTOTYPE = _semantic_embedding(
    "support reassurance warmth steadiness emotional care trust repair safety "
    "先陪我稳住 情绪支持 别急着解决 温暖 安抚 慢一点"
)


def _goal_semantic_pressure(goals: tuple[str, ...], *, prototype: tuple[float, ...]) -> float:
    if not goals:
        return 0.0
    values = [
        _clamp((_cosine_similarity(_semantic_embedding(goal), prototype) + 1.0) / 2.0)
        for goal in goals
    ]
    return sum(values) / len(values)


def _relationship_relevant_alerts(alerts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        alert
        for alert in alerts
        if "cross-track stability" in alert.lower() or "rollback pressure" in alert.lower()
    )


def _feature_surface_snapshot(
    *,
    model_id: str,
    task_pull: float,
    support_pull: float,
    repair_pull: float,
    exploration_pull: float,
    directive_pull: float,
) -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id=model_id,
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(),
        feature_surface=(
            FeatureSignal(name="semantic_task_pull", values=(task_pull,), source="evolution-benchmark"),
            FeatureSignal(name="semantic_support_pull", values=(support_pull,), source="evolution-benchmark"),
            FeatureSignal(name="semantic_repair_pull", values=(repair_pull,), source="evolution-benchmark"),
            FeatureSignal(name="semantic_exploration_pull", values=(exploration_pull,), source="evolution-benchmark"),
            FeatureSignal(name="semantic_directive_pull", values=(directive_pull,), source="evolution-benchmark"),
            FeatureSignal(name="fallback_active", values=(0.0,), source="evolution-benchmark"),
        ),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="evolution benchmark substrate snapshot",
    )


def _default_evolution_benchmark_cases() -> tuple[EvaluationReplayCase, ...]:
    return (
        EvaluationReplayCase(
            case_id="task-dominant",
            session_id="benchmark-task",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-task",
                task_pull=0.88,
                support_pull=0.24,
                repair_pull=0.20,
                exploration_pull=0.18,
                directive_pull=0.72,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("task_pressure", 0.45), ("contract_integrity", 0.95)),
            max_alert_count=0,
        ),
        EvaluationReplayCase(
            case_id="support-dominant",
            session_id="benchmark-support",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-support",
                task_pull=0.22,
                support_pull=0.86,
                repair_pull=0.52,
                exploration_pull=0.20,
                directive_pull=0.12,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("support_presence", 0.40), ("warmth", 0.35)),
            max_alert_count=0,
        ),
        EvaluationReplayCase(
            case_id="mixed-conflict",
            session_id="benchmark-mixed",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-mixed",
                task_pull=0.58,
                support_pull=0.56,
                repair_pull=0.64,
                exploration_pull=0.44,
                directive_pull=0.28,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("cross_track_stability", 0.30), ("contract_integrity", 0.95)),
            max_alert_count=0,
        ),
        EvaluationReplayCase(
            case_id="continuity-long-horizon",
            session_id="benchmark-continuity",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-continuity",
                task_pull=0.44,
                support_pull=0.62,
                repair_pull=0.34,
                exploration_pull=0.52,
                directive_pull=0.18,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("info_integration", 0.10), ("warmth", 0.30)),
            max_alert_count=0,
        ),
    )


def _report_trend(
    report: EvaluationReport,
    *,
    family: str,
    metric_name: str,
) -> float:
    for trend_family, trend_metric, value in report.trends:
        if trend_family == family and trend_metric == metric_name:
            return value
    return 0.0


class EvaluationBackbone:
    """Minimal turn/session evaluator with evidence and alerts."""

    def __init__(self) -> None:
        self._records: list[EvaluationRecord] = []

    @property
    def records(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._records)

    def evaluate_turn(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
        temporal_snapshot: "TemporalAbstractionSnapshot | None" = None,
    ) -> EvaluationSnapshot:
        turn_scores = self._merge_turn_scores(
            self._build_turn_scores(
                substrate_snapshot=substrate_snapshot,
                memory_snapshot=memory_snapshot,
                dual_track_snapshot=dual_track_snapshot,
            ),
            self._temporal_public_scores(temporal_snapshot=temporal_snapshot),
        )
        alerts = self._build_alerts(turn_scores=turn_scores)
        self._append_records(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            timescale="turn",
            scores=turn_scores,
        )
        session_scores = self._session_scores_for(session_id=session_id)
        return EvaluationSnapshot(
            turn_scores=turn_scores,
            session_scores=session_scores,
            alerts=alerts,
            description=(
                f"Evaluation backbone produced {len(turn_scores)} turn scores and "
                f"{len(alerts)} alerts."
            ),
        )

    def family_signals(self, evaluation_snapshot: EvaluationSnapshot) -> dict[str, float]:
        """Extract per-family average signal from an evaluation snapshot.

        Returns signals for 6 families: task, interaction, relationship,
        learning, abstraction, safety.
        """
        families: dict[str, list[float]] = {}
        for score in evaluation_snapshot.turn_scores + evaluation_snapshot.session_scores:
            normalized_value = score.value
            if score.metric_name == "fallback_reliance":
                normalized_value = 1.0 - score.value
            families.setdefault(score.family, []).append(normalized_value)
        result: dict[str, float] = {}
        for family in ("task", "interaction", "relationship", "learning", "abstraction", "safety"):
            values = families.get(family, [])
            result[family] = sum(values) / len(values) if values else 0.5
        return result

    def build_session_report(self, *, session_id: str, timestamp_ms: int) -> EvaluationReport:
        session_records = tuple(record for record in self._records if record.session_id == session_id)
        scores_by_family: dict[str, list[EvaluationRecord]] = {}
        for record in session_records:
            scores_by_family.setdefault(record.family, []).append(record)

        alerts = self._alerts_from_records(session_records)
        recommendations = self._recommendations_from_alerts(alerts)
        return EvaluationReport(
            report_id=str(uuid4()),
            report_type="session",
            timestamp_ms=timestamp_ms,
            session_ids=(session_id,),
            scores_by_family=tuple(
                (family, tuple(records)) for family, records in sorted(scores_by_family.items())
            ),
            alerts=alerts,
            trends=self._longitudinal_trends(session_records),
            recommendations=recommendations,
            description=f"Session evaluation report with {len(session_records)} records.",
        )

    def run_replay_suite(
        self,
        *,
        suite_name: str,
        cases: tuple[EvaluationReplayCase, ...],
        timestamp_ms: int,
    ) -> EvaluationReplaySuiteResult:
        case_results: list[EvaluationReplayCaseResult] = []
        for index, case in enumerate(cases):
            snapshot = self.evaluate_turn(
                session_id=case.session_id,
                wave_id=case.wave_id,
                timestamp_ms=timestamp_ms + index,
                substrate_snapshot=case.substrate_snapshot,
                memory_snapshot=case.memory_snapshot,
                dual_track_snapshot=case.dual_track_snapshot,
            )
            metrics = {score.metric_name: score.value for score in snapshot.turn_scores}
            issues: list[str] = []
            for metric_name, floor in case.metric_floors:
                if metrics.get(metric_name, 0.0) < floor:
                    issues.append(f"{metric_name}<{floor:.2f}")
            if len(snapshot.alerts) > case.max_alert_count:
                issues.append(f"alerts>{case.max_alert_count}")
            case_results.append(
                EvaluationReplayCaseResult(
                    case_id=case.case_id,
                    passed=not issues,
                    observed_metrics=tuple(sorted(metrics.items())),
                    alerts=snapshot.alerts,
                    issues=tuple(issues),
                    description=(
                        f"Replay case {case.case_id} produced {len(snapshot.turn_scores)} scores and "
                        f"{len(snapshot.alerts)} alerts."
                    ),
                )
            )
        passed = all(result.passed for result in case_results)
        return EvaluationReplaySuiteResult(
            suite_name=suite_name,
            passed=passed,
            case_results=tuple(case_results),
            description=f"Replay suite {suite_name} {'passed' if passed else 'failed'} with {len(case_results)} cases.",
        )

    def run_default_evolution_benchmark(
        self,
        *,
        timestamp_ms: int,
    ) -> EvaluationReplaySuiteResult:
        return self.run_replay_suite(
            suite_name="default-evolution-benchmark",
            cases=_default_evolution_benchmark_cases(),
            timestamp_ms=timestamp_ms,
        )

    def judge_evolution_candidate(
        self,
        *,
        replay_suite_result: EvaluationReplaySuiteResult,
        session_report: EvaluationReport,
    ) -> EvolutionJudgement:
        abstraction_trend = _report_trend(
            session_report,
            family="abstraction",
            metric_name="abstraction_reuse",
        )
        learning_trend = _report_trend(
            session_report,
            family="learning",
            metric_name="learning_quality",
        )
        relationship_trend = _report_trend(
            session_report,
            family="relationship",
            metric_name="relationship_continuity",
        )
        reasons: list[str] = []
        high_alerts = [alert for _, alert in session_report.alerts if alert.startswith("HIGH") or alert.startswith("CRITICAL")]
        if not replay_suite_result.passed:
            reasons.append("replay-suite-failed")
        if high_alerts:
            reasons.append("high-alert-pressure")
        if abstraction_trend < -0.03 or learning_trend < -0.03:
            reasons.append("trend-regression")
        if reasons:
            decision = EvolutionDecision.ROLLBACK
        elif abstraction_trend > 0.03 and learning_trend > 0.03 and relationship_trend >= -0.02:
            decision = EvolutionDecision.PROMOTE
            reasons.append("replay-pass-with-positive-trends")
        else:
            decision = EvolutionDecision.HOLD
            reasons.append("insufficient-positive-evidence")
        return EvolutionJudgement(
            decision=decision,
            replay_passed=replay_suite_result.passed,
            abstraction_trend=abstraction_trend,
            learning_trend=learning_trend,
            relationship_trend=relationship_trend,
            reasons=tuple(reasons),
            description=(
                f"Evolution decision={decision.value} replay_passed={replay_suite_result.passed} "
                f"abstraction_trend={abstraction_trend:.3f} learning_trend={learning_trend:.3f} "
                f"relationship_trend={relationship_trend:.3f}."
            ),
        )

    def record_external_scores(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        base_snapshot: EvaluationSnapshot,
        scores: tuple[EvaluationScore, ...],
        description_suffix: str,
        timescale: str = "turn",
    ) -> EvaluationSnapshot:
        if not scores:
            return base_snapshot
        self._append_records(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            timescale=timescale,
            scores=scores,
        )
        turn_scores = self._merge_turn_scores(base_snapshot.turn_scores, scores)
        alerts = tuple(dict.fromkeys(base_snapshot.alerts + self._build_alerts(turn_scores=turn_scores)))
        return EvaluationSnapshot(
            turn_scores=turn_scores,
            session_scores=self._session_scores_for(session_id=session_id),
            alerts=alerts,
            description=f"{base_snapshot.description} {description_suffix}",
        )

    def record_temporal_public_evidence(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        base_snapshot: EvaluationSnapshot,
        temporal_snapshot: "TemporalAbstractionSnapshot | None",
    ) -> EvaluationSnapshot:
        temporal_scores = self._temporal_public_scores(temporal_snapshot=temporal_snapshot)
        return self.record_external_scores(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            base_snapshot=base_snapshot,
            scores=temporal_scores,
            description_suffix=f"Enriched with {len(temporal_scores)} temporal public scores.",
        )

    def record_metacontroller_evidence(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        metacontroller_state: "MetacontrollerRuntimeState | None",
        policy_objective: float,
        rollback_reasons: tuple[str, ...],
    ) -> tuple[EvaluationScore, ...]:
        if metacontroller_state is None:
            return ()
        adaptive_stability = _clamp(
            1.0
            - min(len(rollback_reasons) * 0.25, 0.75)
            - min(metacontroller_state.latest_ssl_loss * 0.15, 0.45)
        )
        posterior_stability = _clamp(1.0 - metacontroller_state.posterior_drift)
        switch_sparsity = _clamp(metacontroller_state.switch_sparsity)
        binary_gate_ratio = _clamp(metacontroller_state.binary_switch_rate)
        decoder_usefulness = _clamp(
            sum(metacontroller_state.decoder_applied_control) / max(len(metacontroller_state.decoder_applied_control), 1)
            - min(metacontroller_state.latest_ssl_loss * 0.08, 0.15)
            + 0.15
        )
        policy_replacement_quality = _clamp(
            0.5
            + metacontroller_state.policy_replacement_score * 0.35
            - min(len(rollback_reasons) * 0.08, 0.2)
        )
        abstract_action_usefulness = _clamp(
            0.5
            + policy_objective * 0.35
            - min(metacontroller_state.latest_ssl_loss * 0.1, 0.2)
        )
        active_family = metacontroller_state.active_family_summary
        family_reuse = (
            _clamp(min(active_family.support / 6.0, 1.0))
            if active_family is not None
            else 0.0
        )
        family_stability = active_family.stability if active_family is not None else 0.0
        family_diversity = _clamp(len(metacontroller_state.action_family_summaries) / 4.0)
        family_competition_score = _clamp(metacontroller_state.active_family_competition_score)
        family_monopoly_pressure = _clamp(metacontroller_state.action_family_monopoly_pressure)
        family_turnover_health = _clamp(metacontroller_state.action_family_turnover_health)
        family_collapse_risk = _clamp(
            family_monopoly_pressure * 0.50
            + (1.0 - family_turnover_health) * 0.30
            + (1.0 - family_competition_score) * 0.10
            + (1.0 - family_diversity) * 0.15
        )
        scores = (
            EvaluationScore(
                family="learning",
                metric_name="adaptive_stability",
                value=adaptive_stability,
                confidence=0.6,
                evidence=(
                    f"Derived from ssl_loss={metacontroller_state.latest_ssl_loss:.3f} "
                    f"and rollback_count={len(rollback_reasons)}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="posterior_stability",
                value=posterior_stability,
                confidence=0.57,
                evidence=(
                    f"Derived from posterior_drift={metacontroller_state.posterior_drift:.3f} "
                    f"and z_tilde={metacontroller_state.z_tilde}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="switch_sparsity",
                value=switch_sparsity,
                confidence=0.55,
                evidence=(
                    f"Derived from latest_switch_gate={metacontroller_state.latest_switch_gate:.3f} "
                    f"and switch_sparsity={metacontroller_state.switch_sparsity:.3f}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="binary_gate_ratio",
                value=binary_gate_ratio,
                confidence=0.55,
                evidence=(
                    f"Derived from beta_binary={metacontroller_state.beta_binary} "
                    f"and binary_switch_rate={metacontroller_state.binary_switch_rate:.3f}."
                ),
            ),
            EvaluationScore(
                family="learning",
                metric_name="decoder_usefulness",
                value=decoder_usefulness,
                confidence=0.56,
                evidence=(
                    f"Derived from decoder_applied_control={metacontroller_state.decoder_applied_control}."
                ),
            ),
            EvaluationScore(
                family="learning",
                metric_name="policy_replacement_quality",
                value=policy_replacement_quality,
                confidence=0.58,
                evidence=(
                    f"Derived from replacement_score={metacontroller_state.policy_replacement_score:.3f}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="abstract_action_usefulness",
                value=abstract_action_usefulness,
                confidence=0.58,
                evidence=(
                    f"Derived from policy_objective={policy_objective:.3f} and "
                    f"active_label={metacontroller_state.active_label}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="action_family_reuse",
                value=family_reuse,
                confidence=0.57,
                evidence=(
                    f"Derived from active_family={active_family.family_id if active_family is not None else 'none'} "
                    f"support={active_family.support if active_family is not None else 0}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="action_family_stability",
                value=family_stability,
                confidence=0.59,
                evidence=(
                    f"Derived from active_family={active_family.family_id if active_family is not None else 'none'} "
                    f"stability={active_family.stability if active_family is not None else 0.0:.3f}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="action_family_diversity",
                value=family_diversity,
                confidence=0.54,
                evidence=(
                    f"Derived from family_version={metacontroller_state.action_family_version} "
                    f"and family_count={len(metacontroller_state.action_family_summaries)}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="action_family_competition_score",
                value=family_competition_score,
                confidence=0.56,
                evidence=(
                    f"Derived from active_family={active_family.family_id if active_family is not None else 'none'} "
                    f"competition_score={family_competition_score:.3f} and "
                    f"family_version={metacontroller_state.action_family_version}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="action_family_monopoly_pressure",
                value=family_monopoly_pressure,
                confidence=0.58,
                evidence=(
                    f"Derived from active_family={active_family.family_id if active_family is not None else 'none'} "
                    f"monopoly_pressure={family_monopoly_pressure:.3f}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="action_family_turnover_health",
                value=family_turnover_health,
                confidence=0.57,
                evidence=(
                    f"Derived from family_version={metacontroller_state.action_family_version} "
                    f"and family_count={len(metacontroller_state.action_family_summaries)}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="action_family_collapse_risk",
                value=family_collapse_risk,
                confidence=0.59,
                evidence=(
                    f"Derived from monopoly_pressure={family_monopoly_pressure:.3f}, "
                    f"turnover_health={family_turnover_health:.3f}, diversity={family_diversity:.3f}."
                ),
            ),
        )
        self._append_records(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            timescale="turn",
            scores=scores,
        )
        return scores

    def record_learning_evidence(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        base_snapshot: EvaluationSnapshot,
        memory_snapshot: MemorySnapshot | None,
        reflection_snapshot: object | None,
        writeback_result: object | None,
        joint_loop_result: object | None,
        regime_snapshot: "RegimeSnapshot | None" = None,
    ) -> EvaluationSnapshot:
        scores = self._learning_evidence_scores(
            memory_snapshot=memory_snapshot,
            reflection_snapshot=reflection_snapshot,
            writeback_result=writeback_result,
            joint_loop_result=joint_loop_result,
            regime_snapshot=regime_snapshot,
        )
        if not scores:
            return base_snapshot
        return self.record_external_scores(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            base_snapshot=base_snapshot,
            scores=scores,
            description_suffix=f"Enriched with {len(scores)} learning-evidence scores.",
        )

    def _temporal_public_scores(
        self,
        *,
        temporal_snapshot: "TemporalAbstractionSnapshot | None",
    ) -> tuple[EvaluationScore, ...]:
        if temporal_snapshot is None:
            return ()
        controller_state = temporal_snapshot.controller_state
        code_energy = _clamp(
            sum(abs(value) for value in controller_state.code) / max(len(controller_state.code), 1)
        )
        persistence = min(controller_state.steps_since_switch / 3.0, 1.0)
        switch_commitment = _clamp(1.0 - abs(controller_state.switch_gate - (1.0 if controller_state.is_switching else 0.0)))
        named_action_bonus = (
            0.12
            if temporal_snapshot.active_abstract_action
            and temporal_snapshot.active_abstract_action != "unassigned_action"
            else 0.0
        )
        temporal_action_commitment = _clamp(
            0.28
            + code_energy * 0.32
            + persistence * 0.18
            + switch_commitment * 0.10
            + named_action_bonus
        )
        return (
            EvaluationScore(
                family="abstraction",
                metric_name="temporal_action_commitment",
                value=temporal_action_commitment,
                confidence=0.54,
                evidence=(
                    f"Derived from active_abstract_action={temporal_snapshot.active_abstract_action}, "
                    f"switch_gate={controller_state.switch_gate:.2f}, is_switching={controller_state.is_switching}, "
                    f"steps_since_switch={controller_state.steps_since_switch}, code_dim={controller_state.code_dim}."
                ),
            ),
        )

    def _build_turn_scores(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
    ) -> tuple[EvaluationScore, ...]:
        memory_count = len(memory_snapshot.retrieved_entries) if memory_snapshot else 0
        cross_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot else 0.0
        relationship_stability = _clamp(1.0 - cross_tension)
        semantic_task_pull = (
            feature_signal_value(substrate_snapshot.feature_surface, name="semantic_task_pull")
            if substrate_snapshot is not None
            else 0.0
        )
        semantic_support_pull = (
            feature_signal_value(substrate_snapshot.feature_surface, name="semantic_support_pull")
            if substrate_snapshot is not None
            else 0.0
        )
        semantic_repair_pull = (
            feature_signal_value(substrate_snapshot.feature_surface, name="semantic_repair_pull")
            if substrate_snapshot is not None
            else 0.0
        )
        semantic_exploration_pull = (
            feature_signal_value(substrate_snapshot.feature_surface, name="semantic_exploration_pull")
            if substrate_snapshot is not None
            else 0.0
        )
        semantic_directive_pull = (
            feature_signal_value(substrate_snapshot.feature_surface, name="semantic_directive_pull")
            if substrate_snapshot is not None
            else 0.0
        )
        fallback_active = (
            feature_signal_value(substrate_snapshot.feature_surface, name="fallback_active")
            if substrate_snapshot is not None
            else 0.0
        )
        world_drive = dual_track_snapshot.world_track.controller_code[0] if dual_track_snapshot else 0.0
        self_drive = dual_track_snapshot.self_track.controller_code[0] if dual_track_snapshot else 0.0
        shared_drive = (
            (
                (dual_track_snapshot.world_track.controller_code[1] if len(dual_track_snapshot.world_track.controller_code) > 1 else 0.0)
                + (dual_track_snapshot.self_track.controller_code[1] if len(dual_track_snapshot.self_track.controller_code) > 1 else 0.0)
            )
            / 2.0
            if dual_track_snapshot
            else 0.0
        )
        world_goal_count = len(dual_track_snapshot.world_track.active_goals) if dual_track_snapshot else 0
        self_goal_count = len(dual_track_snapshot.self_track.active_goals) if dual_track_snapshot else 0
        world_goal_semantics = _goal_semantic_pressure(
            dual_track_snapshot.world_track.active_goals if dual_track_snapshot else (),
            prototype=TASK_PRESSURE_PROTOTYPE,
        )
        self_goal_semantics = _goal_semantic_pressure(
            dual_track_snapshot.self_track.active_goals if dual_track_snapshot else (),
            prototype=SUPPORT_PRESENCE_PROTOTYPE,
        )
        task_pressure = _clamp(
            semantic_task_pull * 0.34
            + semantic_directive_pull * 0.20
            + semantic_repair_pull * 0.06
            + world_drive * 0.20
            + min(world_goal_count / 3.0, 1.0) * 0.08
            + memory_count / 5.0 * 0.10
            + world_goal_semantics * 0.12
        )
        support_presence = _clamp(
            semantic_support_pull * 0.34
            + semantic_repair_pull * 0.15
            + self_drive * 0.12
            + shared_drive * 0.08
            + min(self_goal_count / 3.0, 1.0) * 0.06
            + relationship_stability * 0.07
            + self_goal_semantics * 0.12
            - semantic_directive_pull * 0.10
        )
        info_integration = _clamp(
            memory_count / 5.0 * 0.45
            + world_drive * 0.20
            + min(world_goal_count / 3.0, 1.0) * 0.10
            + semantic_task_pull * 0.16
            + semantic_directive_pull * 0.10
            + semantic_exploration_pull * 0.05
        )
        warmth = _clamp(
            0.06
            + semantic_support_pull * 0.34
            + semantic_repair_pull * 0.12
            + self_drive * 0.08
            + shared_drive * 0.06
            + relationship_stability * 0.08
            + self_goal_semantics * 0.12
            - semantic_directive_pull * 0.08
        )
        placeholder_penalty = 0.12 if substrate_snapshot is None or substrate_snapshot.surface_kind is SurfaceKind.PLACEHOLDER else 0.0
        contract_integrity = _clamp(1.0 - placeholder_penalty)

        return (
            EvaluationScore(
                family="task",
                metric_name="info_integration",
                value=info_integration,
                confidence=0.7,
                evidence=f"{memory_count} retrieved memory entries available to the turn.",
            ),
            EvaluationScore(
                family="task",
                metric_name="task_pressure",
                value=task_pressure,
                confidence=0.6,
                evidence=(
                    f"Derived from semantic_task_pull={semantic_task_pull:.2f}, semantic_directive_pull={semantic_directive_pull:.2f}, world_drive={world_drive:.2f}, "
                    f"world_goal_count={world_goal_count}, retrieved_entries={memory_count}."
                ),
            ),
            EvaluationScore(
                family="interaction",
                metric_name="warmth",
                value=warmth,
                confidence=0.55,
                evidence=(
                    f"Derived from semantic_support_pull={semantic_support_pull:.2f}, "
                    f"semantic_repair_pull={semantic_repair_pull:.2f}, semantic_directive_pull={semantic_directive_pull:.2f}, self_drive={self_drive:.2f}, "
                    f"cross_track_stability={relationship_stability:.2f}."
                ),
            ),
            EvaluationScore(
                family="interaction",
                metric_name="support_presence",
                value=support_presence,
                confidence=0.58,
                evidence=(
                    f"Derived from semantic_support_pull={semantic_support_pull:.2f}, "
                    f"semantic_repair_pull={semantic_repair_pull:.2f}, semantic_exploration_pull={semantic_exploration_pull:.2f}, "
                    f"semantic_directive_pull={semantic_directive_pull:.2f}."
                ),
            ),
            EvaluationScore(
                family="relationship",
                metric_name="cross_track_stability",
                value=relationship_stability,
                confidence=0.65,
                evidence=f"Computed from cross_track_tension={cross_tension:.2f}.",
            ),
            EvaluationScore(
                family="safety",
                metric_name="fallback_reliance",
                value=fallback_active,
                confidence=0.9,
                evidence=(
                    f"Derived from substrate fallback_active={fallback_active:.2f} "
                    f"and surface_kind={substrate_snapshot.surface_kind.value if substrate_snapshot is not None else 'missing'}."
                ),
            ),
            EvaluationScore(
                family="safety",
                metric_name="contract_integrity",
                value=contract_integrity,
                confidence=0.9,
                evidence=(
                    f"Derived from fallback_active={fallback_active:.2f} "
                    f"and surface_kind={substrate_snapshot.surface_kind.value if substrate_snapshot is not None else 'missing'}."
                ),
            ),
        )

    def _learning_evidence_scores(
        self,
        *,
        memory_snapshot: MemorySnapshot | None,
        reflection_snapshot: object | None,
        writeback_result: object | None,
        joint_loop_result: object | None,
        regime_snapshot: "RegimeSnapshot | None",
    ) -> tuple[EvaluationScore, ...]:
        from volvence_zero.joint_loop.runtime import ScheduledJointLoopResult
        from volvence_zero.reflection import ReflectionSnapshot, WritebackResult

        scores: list[EvaluationScore] = []
        if memory_snapshot is not None:
            retrieval_quality = _clamp(
                len(memory_snapshot.retrieved_entries) / 5.0 + (0.1 if memory_snapshot.cms_state is not None else 0.0)
            )
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="retrieval_quality",
                    value=retrieval_quality,
                    confidence=0.58,
                    evidence=(
                        f"Derived from retrieved_entries={len(memory_snapshot.retrieved_entries)} "
                        f"and cms_state={'present' if memory_snapshot.cms_state is not None else 'missing'}."
                    ),
                )
            )
        if reflection_snapshot is not None and isinstance(reflection_snapshot, ReflectionSnapshot):
            confidence = reflection_snapshot.consolidation_score.confidence
            applied_count = len(writeback_result.applied_operations) if isinstance(writeback_result, WritebackResult) else 0
            blocked_count = len(writeback_result.blocked_operations) if isinstance(writeback_result, WritebackResult) else 0
            reflection_usefulness = _clamp(confidence + applied_count * 0.05 - blocked_count * 0.08)
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="reflection_usefulness",
                    value=reflection_usefulness,
                    confidence=0.57,
                    evidence=(
                        f"Derived from consolidation_confidence={confidence:.3f}, "
                        f"applied_operations={applied_count}, blocked_operations={blocked_count}."
                    ),
                )
            )
        if joint_loop_result is not None and isinstance(joint_loop_result, ScheduledJointLoopResult):
            schedule_action = joint_loop_result.schedule_action
            cycle_report = joint_loop_result.cycle_report
            if cycle_report is not None:
                joint_progress = _clamp(
                    0.5
                    + cycle_report.mean_transition_reward * 0.30
                    + cycle_report.policy_objective * 0.1
                    - len(cycle_report.rollback_reasons) * 0.08
                )
                evidence = (
                    f"Derived from action={schedule_action}, mean_transition_reward={cycle_report.mean_transition_reward:.3f}, "
                    f"total_reward={cycle_report.total_reward:.3f}, policy_objective={cycle_report.policy_objective:.3f}."
                )
            elif schedule_action == "ssl-only":
                joint_progress = _clamp(
                    0.65
                    - min(joint_loop_result.ssl_prediction_loss * 0.08, 0.2)
                    - min(joint_loop_result.ssl_kl_loss * 0.05, 0.1)
                )
                evidence = (
                    f"Derived from action=ssl-only, pred={joint_loop_result.ssl_prediction_loss:.3f}, "
                    f"kl={joint_loop_result.ssl_kl_loss:.3f}."
                )
            else:
                joint_progress = 0.5
                evidence = f"Derived from action={schedule_action} without optimizer update."
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="joint_learning_progress",
                    value=joint_progress,
                    confidence=0.56,
                    evidence=evidence,
                )
            )
            if cycle_report is not None:
                rollback_resilience = _clamp(1.0 - min(len(cycle_report.rollback_reasons) * 0.25, 0.75))
                scores.append(
                    EvaluationScore(
                        family="safety",
                        metric_name="rollback_resilience",
                        value=rollback_resilience,
                        confidence=0.62,
                        evidence=(
                            f"Derived from rollback_count={len(cycle_report.rollback_reasons)} "
                            f"and schedule_action={schedule_action}."
                        ),
                    )
                )
                scores.append(
                    EvaluationScore(
                        family="abstraction",
                        metric_name="residual_env_fidelity",
                        value=_clamp(cycle_report.backend_fidelity),
                        confidence=0.72,
                        evidence=(
                            f"Derived from backend={cycle_report.backend_name} "
                            f"fidelity={cycle_report.backend_fidelity:.3f}."
                        ),
                    )
                )
        if regime_snapshot is not None and regime_snapshot.delayed_outcomes:
            delayed_values = tuple(score for _, score in regime_snapshot.delayed_outcomes)
            delayed_alignment = _clamp(sum(delayed_values) / len(delayed_values))
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="delayed_regime_alignment",
                    value=delayed_alignment,
                    confidence=0.6,
                    evidence=(
                        f"Derived from delayed_outcomes={regime_snapshot.delayed_outcomes}."
                    ),
                )
            )
        if regime_snapshot is not None and regime_snapshot.delayed_attributions:
            delayed_action_alignment = _clamp(
                sum(item.outcome_score for item in regime_snapshot.delayed_attributions)
                / len(regime_snapshot.delayed_attributions)
            )
            unique_regimes = {item.regime_id for item in regime_snapshot.delayed_attributions}
            scores.extend(
                (
                    EvaluationScore(
                        family="abstraction",
                        metric_name="delayed_action_alignment",
                        value=delayed_action_alignment,
                        confidence=0.58,
                        evidence=(
                            f"Derived from delayed_attributions={len(regime_snapshot.delayed_attributions)} "
                            f"and abstract_actions="
                            f"{tuple(item.abstract_action for item in regime_snapshot.delayed_attributions)}."
                        ),
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="regime_sequence_payoff",
                        value=_clamp(
                            delayed_action_alignment
                            + min(len(unique_regimes) / max(len(regime_snapshot.delayed_attributions), 1), 1.0) * 0.12
                        ),
                        confidence=0.56,
                        evidence=(
                            f"Derived from delayed_regime_ids={tuple(sorted(unique_regimes))} "
                            f"and mean_outcome={delayed_action_alignment:.3f}."
                        ),
                    ),
                )
            )
        if regime_snapshot is not None and regime_snapshot.delayed_payoffs:
            rolling_alignment = _clamp(
                sum(item.rolling_payoff for item in regime_snapshot.delayed_payoffs)
                / len(regime_snapshot.delayed_payoffs)
            )
            rolling_sample_density = _clamp(
                sum(item.sample_count for item in regime_snapshot.delayed_payoffs)
                / max(len(regime_snapshot.delayed_payoffs) * 3.0, 1.0)
            )
            scores.extend(
                (
                    EvaluationScore(
                        family="learning",
                        metric_name="delayed_credit_horizon",
                        value=rolling_sample_density,
                        confidence=0.58,
                        evidence=(
                            f"Derived from delayed_payoff_sample_counts="
                            f"{tuple(item.sample_count for item in regime_snapshot.delayed_payoffs)}."
                        ),
                    ),
                    EvaluationScore(
                        family="abstraction",
                        metric_name="rolling_action_payoff",
                        value=rolling_alignment,
                        confidence=0.6,
                        evidence=(
                            f"Derived from delayed_payoffs="
                            f"{tuple((item.abstract_action, item.rolling_payoff) for item in regime_snapshot.delayed_payoffs)}."
                        ),
                    ),
                )
            )
        return tuple(scores)

    def _merge_turn_scores(
        self,
        base_scores: tuple[EvaluationScore, ...],
        new_scores: tuple[EvaluationScore, ...],
    ) -> tuple[EvaluationScore, ...]:
        ordered_scores = list(base_scores)
        existing_by_metric = {score.metric_name: index for index, score in enumerate(ordered_scores)}
        for score in new_scores:
            index = existing_by_metric.get(score.metric_name)
            if index is None:
                existing_by_metric[score.metric_name] = len(ordered_scores)
                ordered_scores.append(score)
                continue
            ordered_scores[index] = score
        return tuple(ordered_scores)

    def _build_alerts(self, *, turn_scores: tuple[EvaluationScore, ...]) -> tuple[str, ...]:
        alerts: list[str] = []
        for score in turn_scores:
            if score.family == "relationship" and score.value < 0.4:
                alerts.append("HIGH: cross-track stability is degraded")
            if score.metric_name == "contract_integrity" and score.value < 0.95:
                alerts.append("HIGH: contract integrity below threshold")
            if score.metric_name == "fallback_reliance" and score.value > 0.5:
                alerts.append("MEDIUM: substrate fallback is active")
            if score.metric_name == "rollback_resilience" and score.value < 0.6:
                alerts.append("MEDIUM: rollback pressure is elevated")
            if score.metric_name == "action_family_monopoly_pressure" and score.value > 0.72:
                alerts.append("MEDIUM: action-family monopoly pressure is elevated")
            if score.metric_name == "action_family_collapse_risk" and score.value > 0.68:
                alerts.append("HIGH: action-family collapse risk is elevated")
        return tuple(alerts)

    def _longitudinal_trends(
        self,
        records: tuple[EvaluationRecord, ...],
    ) -> tuple[tuple[str, str, float], ...]:
        return (
            ("relationship", "relationship_continuity", self._trend_for_metrics(records, ("cross_track_stability",))),
            (
                "learning",
                "learning_quality",
                self._trend_for_metrics(
                    records,
                    ("joint_learning_progress", "reflection_usefulness", "retrieval_quality", "rollback_resilience"),
                ),
            ),
            (
                "abstraction",
                "abstraction_reuse",
                self._trend_for_metrics(
                    records,
                    (
                        "abstract_action_usefulness",
                        "switch_sparsity",
                        "binary_gate_ratio",
                        "action_family_competition_score",
                        "action_family_reuse",
                        "action_family_stability",
                        "action_family_turnover_health",
                    ),
                ),
            ),
        )

    def _trend_for_metrics(
        self,
        records: tuple[EvaluationRecord, ...],
        metric_names: tuple[str, ...],
    ) -> float:
        values = [record.value for record in records if record.metric_name in metric_names]
        if len(values) < 2:
            return 0.0
        midpoint = max(len(values) // 2, 1)
        first_half = values[:midpoint]
        second_half = values[midpoint:]
        if not second_half:
            return 0.0
        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)
        return round(second_mean - first_mean, 4)

    def _append_records(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        timescale: str,
        scores: tuple[EvaluationScore, ...],
    ) -> None:
        for score in scores:
            self._records.append(
                EvaluationRecord(
                    record_id=str(uuid4()),
                    session_id=session_id,
                    wave_id=wave_id,
                    timestamp_ms=timestamp_ms,
                    timescale=timescale,
                    family=score.family,
                    metric_name=score.metric_name,
                    value=score.value,
                    confidence=score.confidence,
                    track=self._track_for_family(score.family).value,
                    evidence=score.evidence,
                    signal_sources=self._signal_sources_for_metric(score.metric_name),
                )
            )

    def _session_scores_for(self, *, session_id: str) -> tuple[EvaluationScore, ...]:
        session_records = [record for record in self._records if record.session_id == session_id]
        grouped: dict[tuple[str, str], list[EvaluationRecord]] = {}
        for record in session_records:
            grouped.setdefault((record.family, record.metric_name), []).append(record)
        session_scores: list[EvaluationScore] = []
        for (family, metric_name), records in sorted(grouped.items()):
            average = sum(record.value for record in records) / len(records)
            confidence = sum(record.confidence for record in records) / len(records)
            session_scores.append(
                EvaluationScore(
                    family=family,
                    metric_name=metric_name,
                    value=round(average, 4),
                    confidence=round(confidence, 4),
                    evidence=f"Session aggregate over {len(records)} turn records.",
                )
            )
        return tuple(session_scores)

    def _track_for_family(self, family: str) -> EvaluationTrack:
        if family == "task":
            return EvaluationTrack.WORLD
        if family == "relationship":
            return EvaluationTrack.SELF
        return EvaluationTrack.CROSS

    def _signal_sources_for_metric(self, metric_name: str) -> tuple[str, ...]:
        if metric_name == "info_integration":
            return (
                "memory.retrieved_entries",
                "substrate.feature_surface.semantic_task_pull",
                "substrate.feature_surface.semantic_directive_pull",
            )
        if metric_name == "task_pressure":
            return (
                "substrate.feature_surface.semantic_task_pull",
                "substrate.feature_surface.semantic_directive_pull",
                "dual_track.world_track.controller_code",
            )
        if metric_name == "warmth":
            return (
                "substrate.feature_surface.semantic_support_pull",
                "substrate.feature_surface.semantic_repair_pull",
                "substrate.feature_surface.semantic_directive_pull",
                "dual_track.self_track.controller_code",
            )
        if metric_name == "support_presence":
            return (
                "substrate.feature_surface.semantic_support_pull",
                "substrate.feature_surface.semantic_repair_pull",
                "substrate.feature_surface.semantic_directive_pull",
            )
        if metric_name == "cross_track_stability":
            return ("dual_track.cross_track_tension",)
        if metric_name == "fallback_reliance":
            return ("substrate.feature_surface.fallback_active",)
        if metric_name == "rollback_resilience":
            return ("joint_loop.rollback_reasons", "joint_loop.schedule_action")
        if metric_name == "residual_env_fidelity":
            return ("joint_loop.backend_name", "joint_loop.backend_fidelity")
        if metric_name == "delayed_regime_alignment":
            return ("regime.delayed_outcomes",)
        if metric_name in {"delayed_action_alignment", "regime_sequence_payoff"}:
            return ("regime.delayed_attributions",)
        if metric_name in {"delayed_credit_horizon", "rolling_action_payoff"}:
            return ("regime.delayed_payoffs",)
        if metric_name == "temporal_action_commitment":
            return ("temporal_abstraction.active_abstract_action", "temporal_abstraction.controller_state")
        if metric_name == "adaptive_stability":
            return ("temporal.metacontroller_state", "joint_loop.rollback_reasons")
        if metric_name == "posterior_stability":
            return ("temporal.metacontroller_state.posterior",)
        if metric_name in {"switch_sparsity", "binary_gate_ratio"}:
            return ("temporal.metacontroller_state.switch",)
        if metric_name == "decoder_usefulness":
            return ("temporal.metacontroller_state.decoder",)
        if metric_name in {
            "action_family_competition_score",
            "action_family_reuse",
            "action_family_stability",
            "action_family_diversity",
            "action_family_monopoly_pressure",
            "action_family_turnover_health",
            "action_family_collapse_risk",
        }:
            return ("temporal.metacontroller_state.action_families",)
        if metric_name in {"abstract_action_usefulness", "policy_replacement_quality"}:
            return ("temporal.metacontroller_state", "internal_rl.policy_objective")
        return ("runtime.contract_status",)

    def _alerts_from_records(
        self,
        records: tuple[EvaluationRecord, ...],
    ) -> tuple[tuple[str, str], ...]:
        alerts: list[tuple[str, str]] = []
        for record in records:
            if record.family == "relationship" and record.value < 0.4:
                alerts.append(("HIGH", f"{record.metric_name} below threshold"))
        return tuple(alerts)

    def _recommendations_from_alerts(
        self,
        alerts: tuple[tuple[str, str], ...],
    ) -> tuple[str, ...]:
        if not alerts:
            return ("Continue collecting turn-level evidence before widening scope.",)
        recommendations: list[str] = []
        for _, message in alerts:
            recommendations.append(f"Review alert: {message}")
            lowered = message.lower()
            if "action-family monopoly pressure" in lowered:
                recommendations.append("Reduce active-family monopoly before widening structure changes.")
            if "action-family collapse risk" in lowered:
                recommendations.append("Prefer bounded split/turnover proposals before promoting the current family.")
        return tuple(dict.fromkeys(recommendations))


class EvaluationModule(RuntimeModule[EvaluationSnapshot]):
    slot_name = "evaluation"
    owner = "EvaluationModule"
    value_type = EvaluationSnapshot
    dependencies = ("substrate", "memory", "dual_track")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        backbone: EvaluationBackbone | None = None,
        session_id: str = "runtime-session",
        wave_id: str = "wave-0",
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._backbone = backbone or EvaluationBackbone()
        self._session_id = session_id
        self._wave_id = wave_id

    @property
    def backbone(self) -> EvaluationBackbone:
        return self._backbone

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[EvaluationSnapshot]:
        substrate_snapshot = upstream["substrate"]
        memory_snapshot = upstream["memory"]
        dual_track_snapshot = upstream["dual_track"]
        substrate_value = (
            substrate_snapshot.value if isinstance(substrate_snapshot.value, SubstrateSnapshot) else None
        )
        memory_value = memory_snapshot.value if isinstance(memory_snapshot.value, MemorySnapshot) else None
        dual_track_value = (
            dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        )
        return self.publish(
            self._backbone.evaluate_turn(
                session_id=self._session_id,
                wave_id=self._wave_id,
                timestamp_ms=max(substrate_snapshot.timestamp_ms, memory_snapshot.timestamp_ms, dual_track_snapshot.timestamp_ms),
                substrate_snapshot=substrate_value,
                memory_snapshot=memory_value,
                dual_track_snapshot=dual_track_value,
            )
        )

    async def process_standalone(self, **kwargs: object) -> Snapshot[EvaluationSnapshot]:
        session_id = str(kwargs.get("session_id", self._session_id))
        wave_id = str(kwargs.get("wave_id", self._wave_id))
        timestamp_ms = int(kwargs.get("timestamp_ms", 1))
        substrate_snapshot = kwargs.get("substrate_snapshot")
        memory_snapshot = kwargs.get("memory_snapshot")
        dual_track_snapshot = kwargs.get("dual_track_snapshot")
        temporal_snapshot = kwargs.get("temporal_snapshot")
        from volvence_zero.temporal.interface import TemporalAbstractionSnapshot

        return self.publish(
            self._backbone.evaluate_turn(
                session_id=session_id,
                wave_id=wave_id,
                timestamp_ms=timestamp_ms,
                substrate_snapshot=substrate_snapshot if isinstance(substrate_snapshot, SubstrateSnapshot) else None,
                memory_snapshot=memory_snapshot if isinstance(memory_snapshot, MemorySnapshot) else None,
                dual_track_snapshot=(
                    dual_track_snapshot
                    if isinstance(dual_track_snapshot, DualTrackSnapshot)
                    else None
                ),
                temporal_snapshot=(
                    temporal_snapshot
                    if isinstance(temporal_snapshot, TemporalAbstractionSnapshot)
                    else None
                ),
            )
        )
