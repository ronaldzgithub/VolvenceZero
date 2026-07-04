from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Mapping
from uuid import uuid4

# Slice C (2026-05-03): the evaluation layer consumes application-tier
# snapshots through structural ``Protocol`` readouts published by
# ``vz-contracts``. Concrete dataclasses live in vz-application and
# satisfy these protocols by attribute presence, so no kernel-tier
# wheel needs to import application owner code.
from volvence_zero.application_readouts import (
    ApplicationOutcomeAttributionReadout,
    ApplicationSequencePayoffReadout,
    BoundaryReadout,
    CaseMemoryReadout,
    DomainKnowledgeReadout,
    ExperienceFastPriorReadout,
    ResponseAssemblyReadout,
    StrategyPlaybookReadout,
)
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

# Slice S.2 (2026-05-04): capability-domain helpers and data contracts
# previously inlined in this file now live in sibling modules. The main
# ``EvaluationBackbone`` class below is kept intact for this slice; a
# future slice may split it further once a clean API boundary emerges.
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
    _clamp,
    _percentile,
    _sample_std,
    build_metric_interval_summaries,
    build_metric_interval_summary,
    build_pairwise_metric_effect,
)
from volvence_zero.evaluation.semantic_readouts import (
    _cosine_similarity,
    _goal_semantic_pressure,
    _semantic_embedding,
    _semantic_tokens,
    support_presence_prototype,
    task_pressure_prototype,
)
from volvence_zero.evaluation.replay_scenarios import (
    _default_evolution_benchmark_cases,
    _feature_surface_snapshot,
)
from volvence_zero.evaluation.report_helpers import (
    _report_metric_mean,
    _report_trend,
)


class EvaluationBackbone:
    """Observability readout layer for the cognitive loop.

    This backbone produces evaluation scores for **monitoring, alerting,
    and evidence** purposes.  It is NOT the primary learning signal source.
    The primary learning signal is prediction error (R-PE), produced by
    ``PredictionErrorModule``.  Scores from this backbone are consumed by
    the PE module as **inputs to prediction construction**, and by credit
    as **lightweight readout evidence** — not as direct reward.
    """

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
        structured_alerts = self._build_structured_alerts(turn_scores=turn_scores)
        alerts = tuple(alert.legacy_text for alert in structured_alerts)
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
            structured_alerts=structured_alerts,
        )

    def family_signals(self, evaluation_snapshot: EvaluationSnapshot) -> dict[str, float]:
        """Extract per-family average readout signal from an evaluation snapshot.

        These signals are **readouts**, not primary learning signals.
        They are consumed by ``PredictionErrorModule`` for constructing
        predicted outcomes and by the joint loop for rollback gating.
        The actual learning primitive is prediction error (R-PE).

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

    def run_cross_session_benchmark(
        self,
        *,
        suite: CrossSessionBenchmarkSuite,
    ) -> CrossSessionGrowthReport:
        reports = suite.session_reports
        if not reports:
            return CrossSessionGrowthReport(
                window_trends=(),
                family_persistence=0.0,
                regime_effectiveness_delta=0.0,
                verdict="insufficient-data",
                description="No session reports to compare.",
            )
        metric_keys = (
            "relationship_continuity",
            "identity_continuity",
            "relationship_repair_continuity",
            "learning_quality",
            "abstraction_reuse",
            "scheduler_health",
            "async_robustness",
            "semantic_spine_readiness",
        )
        window_trends: list[tuple[int, tuple[tuple[str, float], ...]]] = []
        for window in suite.comparison_windows:
            if len(reports) < window + 1:
                continue
            recent = reports[-window:]
            earlier = reports[: max(len(reports) - window, 1)]
            trends: list[tuple[str, float]] = []
            for metric in metric_keys:
                recent_vals = [
                    v for r in recent for _, m, v in r.trends if m == metric
                ]
                earlier_vals = [
                    v for r in earlier for _, m, v in r.trends if m == metric
                ]
                recent_mean = sum(recent_vals) / max(len(recent_vals), 1) if recent_vals else 0.0
                earlier_mean = sum(earlier_vals) / max(len(earlier_vals), 1) if earlier_vals else 0.0
                trends.append((metric, round(recent_mean - earlier_mean, 4)))
            window_trends.append((window, tuple(trends)))
        all_family_ids: list[set[str]] = []
        for report in reports:
            family_ids: set[str] = set()
            for family, records in report.scores_by_family:
                if family == "abstraction":
                    for rec in records:
                        if rec.metric_name == "action_family_reuse":
                            family_ids.add(rec.evidence)
            all_family_ids.append(family_ids)
        if len(all_family_ids) >= 2 and all_family_ids[0]:
            first_families = all_family_ids[0]
            last_families = all_family_ids[-1]
            family_persistence = len(first_families & last_families) / max(len(first_families), 1)
        else:
            family_persistence = 0.0
        effectiveness_deltas: list[float] = []
        for report in reports:
            for _, metric, value in report.trends:
                if metric == "learning_quality":
                    effectiveness_deltas.append(value)
        regime_effectiveness_delta = (
            sum(effectiveness_deltas) / len(effectiveness_deltas) if effectiveness_deltas else 0.0
        )
        growth_signals = 0
        regression_signals = 0
        for _, trends in window_trends:
            for metric, delta in trends:
                if metric in (
                    "relationship_continuity",
                    "identity_continuity",
                    "relationship_repair_continuity",
                    "learning_quality",
                    "async_robustness",
                    "semantic_spine_readiness",
                ):
                    if delta > 0.01:
                        growth_signals += 1
                    elif delta < -0.01:
                        regression_signals += 1
        if growth_signals >= 2 and regression_signals == 0:
            verdict = "growing"
        elif regression_signals >= 2:
            verdict = "regressing"
        else:
            verdict = "stable"
        return CrossSessionGrowthReport(
            window_trends=tuple(window_trends),
            family_persistence=family_persistence,
            regime_effectiveness_delta=round(regime_effectiveness_delta, 4),
            verdict=verdict,
            description=(
                f"Cross-session benchmark over {len(reports)} sessions: "
                f"verdict={verdict}, effectiveness_delta={regime_effectiveness_delta:.4f}."
            ),
        )

    def build_longitudinal_report(
        self,
        *,
        suite: CrossSessionBenchmarkSuite,
    ) -> LongitudinalReport:
        """Build a rich longitudinal report on top of the cross-session benchmark."""
        cross_session = self.run_cross_session_benchmark(suite=suite)
        reports = suite.session_reports
        metric_keys = (
            "relationship_continuity",
            "identity_continuity",
            "relationship_repair_continuity",
            "learning_quality",
            "abstraction_reuse",
            "scheduler_health",
            "async_robustness",
            "semantic_spine_readiness",
        )
        dimension_trends: list[tuple[str, float, float]] = []
        for metric in metric_keys:
            values = [v for r in reports for _, m, v in r.trends if m == metric]
            if len(values) < 2:
                dimension_trends.append((metric, 0.0, 0.0))
                continue
            slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            confidence = max(0.0, min(1.0, 1.0 - variance))
            dimension_trends.append((metric, round(slope, 4), round(confidence, 4)))
        family_ids_per_session: list[dict[str, float]] = []
        for report in reports:
            fam_scores: dict[str, float] = {}
            for family, records in report.scores_by_family:
                if family == "abstraction":
                    for rec in records:
                        if rec.metric_name == "action_family_reuse":
                            fam_scores[rec.evidence] = rec.value
            family_ids_per_session.append(fam_scores)
        all_family_ids = set()
        for fam_scores in family_ids_per_session:
            all_family_ids.update(fam_scores.keys())
        family_survival_curves: list[tuple[str, tuple[float, ...]]] = []
        for fam_id in sorted(all_family_ids):
            curve = tuple(
                fam_scores.get(fam_id, 0.0)
                for fam_scores in family_ids_per_session
            )
            family_survival_curves.append((fam_id, curve))
        regime_curves: dict[str, list[float]] = {}
        for report in reports:
            for _, metric, value in report.trends:
                if metric == "learning_quality":
                    for fam, _ in report.scores_by_family:
                        regime_curves.setdefault(fam, []).append(value)
        regime_effectiveness_curves = tuple(
            (regime, tuple(curve)) for regime, curve in sorted(regime_curves.items())
        )
        return LongitudinalReport(
            cross_session=cross_session,
            dimension_trends=tuple(dimension_trends),
            family_survival_curves=tuple(family_survival_curves),
            regime_effectiveness_curves=regime_effectiveness_curves,
            verdict=cross_session.verdict,
            description=(
                f"Longitudinal report over {len(reports)} sessions: "
                f"verdict={cross_session.verdict}, "
                f"{len(family_survival_curves)} families tracked, "
                f"{len(regime_effectiveness_curves)} regime curves."
            ),
        )

    def judge_evolution_candidate(
        self,
        *,
        replay_suite_result: EvaluationReplaySuiteResult,
        session_report: EvaluationReport,
        cross_session_report: CrossSessionGrowthReport | None = None,
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
        semantic_spine_trend = _report_trend(
            session_report,
            family="learning",
            metric_name="semantic_spine_readiness",
        )
        delayed_mix_alignment = _report_metric_mean(
            session_report,
            family="learning",
            metric_name="delayed_retrieval_mix_alignment",
        )
        delayed_regime_alignment = _report_metric_mean(
            session_report,
            family="learning",
            metric_name="delayed_regime_alignment",
        )
        delayed_action_alignment = _report_metric_mean(
            session_report,
            family="abstraction",
            metric_name="delayed_abstract_action_alignment",
        )
        regime_sequence_payoff = _report_metric_mean(
            session_report,
            family="learning",
            metric_name="regime_sequence_payoff",
        )
        playbook_confidence = _report_metric_mean(
            session_report,
            family="learning",
            metric_name="playbook_confidence_mean",
        )
        positive_experience_payoff_values = tuple(
            value
            for value in (
                delayed_mix_alignment,
                delayed_regime_alignment,
                delayed_action_alignment,
                regime_sequence_payoff,
            )
            if value > 0.0
        )
        reasons: list[str] = []
        critical_alerts = [
            alert for alert in session_report.alerts if alert[0] == "CRITICAL"
        ]
        has_unsafe = any(severity == "CRITICAL" for severity, _ in session_report.alerts) or any(
            family == "safety" and any(record.value < 0.85 for record in records)
            for family, records in session_report.scores_by_family
        )
        if not replay_suite_result.passed:
            reasons.append("replay-suite-failed")
        if critical_alerts:
            reasons.append("high-alert-pressure")
        if abstraction_trend < -0.03 or learning_trend < -0.03:
            reasons.append("trend-regression")
        if semantic_spine_trend < -0.03:
            reasons.append("semantic-spine-regression")
        if cross_session_report is not None and cross_session_report.verdict == "regressing":
            reasons.append("cross-session-regression")
        if any(
            value > 0.0 and value < 0.46
            for value in (
                delayed_mix_alignment,
                delayed_regime_alignment,
                delayed_action_alignment,
                regime_sequence_payoff,
            )
        ):
            reasons.append("experience-payoff-weak")
        if reasons:
            decision = EvolutionDecision.ROLLBACK
            if has_unsafe:
                category = JudgementCategory.UNSAFE_MUTATION
            else:
                category = JudgementCategory.STYLE_DRIFT
        elif (
            abstraction_trend > 0.03
            and learning_trend > 0.03
            and relationship_trend >= -0.02
            and (
                max(
                    delayed_mix_alignment,
                    delayed_regime_alignment,
                    delayed_action_alignment,
                    regime_sequence_payoff,
                    playbook_confidence,
                )
                == 0.0
                or not positive_experience_payoff_values
                or min(positive_experience_payoff_values) >= 0.52
            )
        ):
            decision = EvolutionDecision.PROMOTE
            category = JudgementCategory.REAL_IMPROVEMENT
            if cross_session_report is not None and cross_session_report.verdict == "growing":
                reasons.append("cross-session-growth-confirmed")
            reasons.append("replay-pass-with-positive-trends")
            if delayed_mix_alignment > 0.0:
                reasons.append("experience-payoff-confirmed")
        else:
            decision = EvolutionDecision.HOLD
            category = JudgementCategory.INSUFFICIENT_EVIDENCE
            reasons.append("insufficient-positive-evidence")
        return EvolutionJudgement(
            decision=decision,
            category=category,
            replay_passed=replay_suite_result.passed,
            abstraction_trend=abstraction_trend,
            learning_trend=learning_trend,
            relationship_trend=relationship_trend,
            reasons=tuple(reasons),
            description=(
                f"Evolution decision={decision.value} category={category.value} "
                f"replay_passed={replay_suite_result.passed} "
                f"abstraction_trend={abstraction_trend:.3f} learning_trend={learning_trend:.3f} "
                f"relationship_trend={relationship_trend:.3f} "
                f"semantic_spine_trend={semantic_spine_trend:.3f} "
                f"delayed_mix_alignment={delayed_mix_alignment:.3f} "
                f"delayed_regime_alignment={delayed_regime_alignment:.3f} "
                f"delayed_action_alignment={delayed_action_alignment:.3f} "
                f"regime_sequence_payoff={regime_sequence_payoff:.3f}."
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
        structured_alerts = tuple(
            {
                (alert.code, alert.metric_name, alert.severity): alert
                for alert in (
                    base_snapshot.structured_alerts
                    + self._build_structured_alerts(turn_scores=turn_scores)
                )
            }.values()
        )
        alerts = tuple(alert.legacy_text for alert in structured_alerts)
        return EvaluationSnapshot(
            turn_scores=turn_scores,
            session_scores=self._session_scores_for(session_id=session_id),
            alerts=alerts,
            description=f"{base_snapshot.description} {description_suffix}",
            structured_alerts=structured_alerts,
            reflection_accuracy=base_snapshot.reflection_accuracy,
            longitudinal_verdict=base_snapshot.longitudinal_verdict,
        )

    def record_temporal_public_evidence(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        base_snapshot: EvaluationSnapshot,
        temporal_snapshot: "TemporalAbstractionSnapshot | None",
        metacontroller_state: "MetacontrollerRuntimeState | None" = None,
    ) -> EvaluationSnapshot:
        temporal_scores = self._temporal_public_scores(
            temporal_snapshot=temporal_snapshot,
            metacontroller_state=metacontroller_state,
        )
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
        fast_prior_strength = _clamp(metacontroller_state.fast_prior_strength)
        fast_prior_switch_pressure = _clamp(0.5 + metacontroller_state.fast_prior_switch_pressure_delta)
        encoder_slow_norm = (
            metacontroller_state.encoder_optimizer_state.mean_slow_norm
            if metacontroller_state.encoder_optimizer_state is not None
            else 0.0
        )
        decoder_slow_norm = (
            metacontroller_state.decoder_optimizer_state.mean_slow_norm
            if metacontroller_state.decoder_optimizer_state is not None
            else 0.0
        )
        optimizer_memory_drive = _clamp((encoder_slow_norm + decoder_slow_norm) / 2.0)
        updater_effective_lr = (
            _clamp(metacontroller_state.learned_update_rule_state.last_effective_learning_rate)
            if metacontroller_state.learned_update_rule_state is not None
            else 0.0
        )
        updater_confidence = (
            _clamp(metacontroller_state.learned_update_rule_state.last_confidence)
            if metacontroller_state.learned_update_rule_state is not None
            else 0.0
        )
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
            EvaluationScore(
                family="abstraction",
                metric_name="family_outcome_divergence",
                value=self._family_outcome_divergence(metacontroller_state),
                confidence=0.56,
                evidence=(
                    f"Derived from outcome histories across "
                    f"{len(metacontroller_state.action_family_summaries)} families."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="temporal_fast_prior_strength",
                value=fast_prior_strength,
                confidence=0.62,
                evidence=(
                    f"Derived from fast_prior_strength={metacontroller_state.fast_prior_strength:.3f}, "
                    f"action_bias={metacontroller_state.fast_prior_action_bias:.3f}, "
                    f"family_bias={metacontroller_state.fast_prior_family_bias:.3f}."
                ),
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="temporal_fast_prior_switch_pressure",
                value=fast_prior_switch_pressure,
                confidence=0.61,
                evidence=(
                    f"Derived from fast_prior_switch_pressure_delta="
                    f"{metacontroller_state.fast_prior_switch_pressure_delta:.3f}."
                ),
            ),
            EvaluationScore(
                family="learning",
                metric_name="optimizer_memory_drive",
                value=optimizer_memory_drive,
                confidence=0.65,
                evidence=(
                    f"Derived from encoder_slow_norm={encoder_slow_norm:.3f} "
                    f"and decoder_slow_norm={decoder_slow_norm:.3f}."
                ),
            ),
            EvaluationScore(
                family="learning",
                metric_name="temporal_updater_effective_lr",
                value=updater_effective_lr,
                confidence=0.66,
                evidence=(
                    metacontroller_state.learned_update_rule_state.description
                    if metacontroller_state.learned_update_rule_state is not None
                    else "No temporal updater state was published."
                ),
            ),
            EvaluationScore(
                family="learning",
                metric_name="temporal_updater_confidence",
                value=updater_confidence,
                confidence=0.64,
                evidence=(
                    metacontroller_state.learned_update_rule_state.description
                    if metacontroller_state.learned_update_rule_state is not None
                    else "No temporal updater state was published."
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
        domain_knowledge_snapshot: DomainKnowledgeReadout | None = None,
        case_memory_snapshot: CaseMemoryReadout | None = None,
        strategy_playbook_snapshot: StrategyPlaybookReadout | None = None,
        boundary_policy_snapshot: BoundaryReadout | None = None,
        experience_fast_prior_snapshot: ExperienceFastPriorReadout | None = None,
        response_assembly_snapshot: ResponseAssemblyReadout | None = None,
        semantic_state_snapshots: tuple[object, ...] = (),
        delayed_outcome_ledger: tuple[ApplicationOutcomeAttributionReadout, ...] = (),
        sequence_payoffs: tuple[ApplicationSequencePayoffReadout, ...] = (),
    ) -> EvaluationSnapshot:
        scores = self._learning_evidence_scores(
            memory_snapshot=memory_snapshot,
            reflection_snapshot=reflection_snapshot,
            writeback_result=writeback_result,
            joint_loop_result=joint_loop_result,
            regime_snapshot=regime_snapshot,
            domain_knowledge_snapshot=domain_knowledge_snapshot,
            case_memory_snapshot=case_memory_snapshot,
            strategy_playbook_snapshot=strategy_playbook_snapshot,
            boundary_policy_snapshot=boundary_policy_snapshot,
            experience_fast_prior_snapshot=experience_fast_prior_snapshot,
            response_assembly_snapshot=response_assembly_snapshot,
            semantic_state_snapshots=semantic_state_snapshots,
            delayed_outcome_ledger=delayed_outcome_ledger,
            sequence_payoffs=sequence_payoffs,
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

    def record_application_delayed_evidence(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        base_snapshot: EvaluationSnapshot,
        delayed_outcome_ledger: tuple[ApplicationOutcomeAttributionReadout, ...] = (),
        sequence_payoffs: tuple[ApplicationSequencePayoffReadout, ...] = (),
    ) -> EvaluationSnapshot:
        scores = self._learning_evidence_scores(
            memory_snapshot=None,
            reflection_snapshot=None,
            writeback_result=None,
            joint_loop_result=None,
            regime_snapshot=None,
            delayed_outcome_ledger=delayed_outcome_ledger,
            sequence_payoffs=sequence_payoffs,
        )
        if not scores:
            return base_snapshot
        return self.record_external_scores(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            base_snapshot=base_snapshot,
            scores=scores,
            description_suffix=f"Enriched with {len(scores)} application delayed-evidence scores.",
            timescale="session",
        )

    def record_prediction_error_evidence(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        base_snapshot: EvaluationSnapshot,
        prediction_error_snapshot: "PredictionErrorSnapshot | None",
    ) -> EvaluationSnapshot:
        scores = self._prediction_error_scores(prediction_error_snapshot=prediction_error_snapshot)
        if not scores:
            return base_snapshot
        return self.record_external_scores(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=timestamp_ms,
            base_snapshot=base_snapshot,
            scores=scores,
            description_suffix=f"Enriched with {len(scores)} prediction-error scores.",
        )

    def record_persona_geometry_evidence(
        self,
        *,
        session_id: str,
        wave_id: str,
        timestamp_ms: int,
        base_snapshot: EvaluationSnapshot,
        substrate_snapshot: SubstrateSnapshot | None,
        regime_snapshot: "RegimeSnapshot | None",
    ) -> EvaluationSnapshot:
        scores = self._persona_geometry_scores(
            substrate_snapshot=substrate_snapshot,
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
            description_suffix=f"Enriched with {len(scores)} persona-geometry scores.",
        )

    def _persona_geometry_scores(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot | None,
        regime_snapshot: "RegimeSnapshot | None",
    ) -> tuple[EvaluationScore, ...]:
        if substrate_snapshot is None or regime_snapshot is None:
            return ()
        regime_embedding = regime_snapshot.active_regime.embedding
        if not regime_embedding:
            return ()
        feature_values: list[float] = []
        for signal in substrate_snapshot.feature_surface:
            feature_values.extend(signal.values)
        if not feature_values and substrate_snapshot.residual_activations:
            feature_values.extend(substrate_snapshot.residual_activations)
        if not feature_values:
            return ()
        width = min(len(feature_values), len(regime_embedding))
        if width == 0:
            return ()
        numerator = sum(feature_values[index] * regime_embedding[index] for index in range(width))
        feature_norm = math.sqrt(sum(feature_values[index] ** 2 for index in range(width)))
        regime_norm = math.sqrt(sum(regime_embedding[index] ** 2 for index in range(width)))
        if feature_norm == 0.0 or regime_norm == 0.0:
            alignment = 0.0
        else:
            alignment = _clamp((numerator / (feature_norm * regime_norm) + 1.0) / 2.0)
        drift = _clamp(1.0 - alignment)
        return (
            EvaluationScore(
                family="safety",
                metric_name="persona_geometry_drift",
                value=drift,
                confidence=0.55,
                evidence=(
                    f"Read-only COG-3 geometry readout against regime="
                    f"{regime_snapshot.active_regime.regime_id}; "
                    f"alignment={alignment:.3f} width={width}."
                ),
            ),
            EvaluationScore(
                family="safety",
                metric_name="persona_regime_geometry_alignment",
                value=alignment,
                confidence=0.55,
                evidence=(
                    f"Read-only COG-3 geometry alignment from substrate features "
                    f"and regime embedding; width={width}."
                ),
            ),
        )

    def _prediction_error_scores(
        self,
        *,
        prediction_error_snapshot: "PredictionErrorSnapshot | None",
    ) -> tuple[EvaluationScore, ...]:
        if prediction_error_snapshot is None:
            return ()
        pe = prediction_error_snapshot.error
        if prediction_error_snapshot.bootstrap:
            return (
                EvaluationScore(
                    family="learning",
                    metric_name="prediction_error_bootstrap",
                    value=1.0,
                    confidence=0.95,
                    evidence=prediction_error_snapshot.description,
                ),
            )
        prediction_confidence = (
            prediction_error_snapshot.evaluated_prediction.confidence
            if prediction_error_snapshot.evaluated_prediction is not None
            else 0.0
        )
        magnitude_readout = _clamp(min(pe.magnitude / 4.0, 1.0))
        reward_readout = _clamp(0.5 + pe.signed_reward * 0.5)
        predictive_accuracy = _clamp((1.0 - magnitude_readout) * 0.8 + prediction_confidence * 0.2)
        scores: list[EvaluationScore] = [
            EvaluationScore(
                family="learning",
                metric_name="prediction_error_magnitude",
                value=magnitude_readout,
                confidence=0.82,
                evidence=f"PE-owner magnitude readout. {pe.description}",
            ),
            EvaluationScore(
                family="learning",
                metric_name="prediction_error_reward",
                value=reward_readout,
                confidence=0.82,
                evidence=f"PE-owner signed reward readout. {pe.description}",
            ),
            EvaluationScore(
                family="learning",
                metric_name="predictive_accuracy",
                value=predictive_accuracy,
                confidence=0.8,
                evidence=(
                    f"Derived from PE-owner magnitude={magnitude_readout:.3f} "
                    f"and prediction_confidence={prediction_confidence:.3f}."
                ),
            ),
            EvaluationScore(
                family="task",
                metric_name="task_prediction_alignment",
                value=_clamp(1.0 - abs(pe.task_error)),
                confidence=0.78,
                evidence=pe.description,
            ),
            EvaluationScore(
                family="relationship",
                metric_name="relationship_prediction_alignment",
                value=_clamp(1.0 - abs(pe.relationship_error)),
                confidence=0.78,
                evidence=pe.description,
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="action_prediction_alignment",
                value=_clamp(1.0 - abs(pe.action_error)),
                confidence=0.78,
                evidence=pe.description,
            ),
            EvaluationScore(
                family="learning",
                metric_name="primary_prediction_error",
                value=_clamp(max(abs(pe.task_error), abs(pe.relationship_error), abs(pe.regime_error), abs(pe.action_error))),
                confidence=0.8,
                evidence=pe.description,
            ),
        ]
        # Phase 1.B: Curiosity-Critic readout. Strictly report-only;
        # not consumed by any acceptance gate. Skipped when the PE
        # owner did not publish a decomposition (bootstrap turn or
        # Phase 1.B not yet in this code path).
        decomposition = prediction_error_snapshot.pe_decomposition
        if decomposition is not None:
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="pe_aleatoric_magnitude",
                    value=_clamp(decomposition.aleatoric_magnitude),
                    confidence=0.6,
                    evidence=(
                        f"PE-owner Curiosity-Critic readout. "
                        f"{decomposition.description}"
                    ),
                )
            )
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="pe_epistemic_magnitude",
                    value=_clamp(decomposition.epistemic_magnitude),
                    confidence=0.6,
                    evidence=(
                        f"PE-owner Curiosity-Critic readout. "
                        f"{decomposition.description}"
                    ),
                )
            )
        return tuple(scores)

    def _family_outcome_divergence(
        self,
        metacontroller_state: "MetacontrollerRuntimeState",
    ) -> float:
        summaries = metacontroller_state.action_family_summaries
        if len(summaries) < 2:
            return 0.0
        means: list[float] = []
        for summary in summaries:
            if summary.outcome_history:
                means.append(sum(summary.outcome_history) / len(summary.outcome_history))
        if len(means) < 2:
            return 0.0
        overall_mean = sum(means) / len(means)
        variance = sum((m - overall_mean) ** 2 for m in means) / len(means)
        return _clamp(variance ** 0.5 * 3.0)

    def _temporal_public_scores(
        self,
        *,
        temporal_snapshot: "TemporalAbstractionSnapshot | None",
        metacontroller_state: "MetacontrollerRuntimeState | None" = None,
    ) -> tuple[EvaluationScore, ...]:
        if temporal_snapshot is None:
            return ()
        controller_state = temporal_snapshot.controller_state
        snapshot_code_energy = _clamp(
            sum(abs(value) for value in controller_state.code) / max(len(controller_state.code), 1)
        )
        persistence = min(controller_state.steps_since_switch / 3.0, 1.0)
        switch_commitment = _clamp(1.0 - abs(controller_state.switch_gate - (1.0 if controller_state.is_switching else 0.0)))
        family_support = 0.0
        family_stability = 0.0
        family_competition_score = 0.0
        switch_sparsity = 1.0 - controller_state.switch_gate
        policy_replacement_quality = 0.0
        family_version = temporal_snapshot.action_family_version
        family_count = 0
        active_label = temporal_snapshot.active_abstract_action
        active_family = None
        fast_prior_strength = 0.0
        fast_prior_switch_commitment = 0.5
        if metacontroller_state is not None:
            active_label = metacontroller_state.active_label
            active_family = metacontroller_state.active_family_summary
            family_support = _clamp(
                min(active_family.support / 6.0, 1.0)
                if active_family is not None
                else 0.0
            )
            family_stability = _clamp(active_family.stability if active_family is not None else 0.0)
            family_competition_score = _clamp(metacontroller_state.active_family_competition_score)
            switch_sparsity = _clamp(metacontroller_state.switch_sparsity)
            policy_replacement_quality = _clamp(metacontroller_state.policy_replacement_score)
            family_version = metacontroller_state.action_family_version
            family_count = len(metacontroller_state.action_family_summaries)
            fast_prior_strength = _clamp(metacontroller_state.fast_prior_strength)
            fast_prior_switch_commitment = _clamp(0.5 - metacontroller_state.fast_prior_switch_pressure_delta)
        family_signal = _clamp(
            family_support * 0.35
            + family_stability * 0.35
            + family_competition_score * 0.30
        )
        temporal_action_commitment = _clamp(
            0.18
            + snapshot_code_energy * 0.18
            + persistence * 0.12
            + switch_commitment * 0.08
            + switch_sparsity * 0.14
            + family_signal * 0.20
            + policy_replacement_quality * 0.10
            + fast_prior_strength * 0.05
            + fast_prior_switch_commitment * 0.03
        )
        scores = [
            EvaluationScore(
                family="abstraction",
                metric_name="temporal_action_commitment",
                value=temporal_action_commitment,
                confidence=0.54,
                evidence=(
                    f"Derived from active_label={active_label}, "
                    f"switch_gate={controller_state.switch_gate:.2f}, switch_sparsity={switch_sparsity:.2f}, "
                    f"family_version={family_version}, family_count={family_count}, "
                    f"fast_prior_strength={fast_prior_strength:.2f}, "
                    f"fast_prior_switch_commitment={fast_prior_switch_commitment:.2f}, "
                    f"active_family={active_family.family_id if active_family is not None else 'none'}, "
                    f"steps_since_switch={controller_state.steps_since_switch}, code_dim={controller_state.code_dim}."
                ),
            ),
        ]
        if metacontroller_state is not None:
            scores.extend(
                (
                    EvaluationScore(
                        family="abstraction",
                        metric_name="temporal_fast_prior_strength",
                        value=fast_prior_strength,
                        confidence=0.6,
                        evidence=(
                            f"Derived from fast_prior_strength={metacontroller_state.fast_prior_strength:.3f}, "
                            f"action_bias={metacontroller_state.fast_prior_action_bias:.3f}, "
                            f"family_bias={metacontroller_state.fast_prior_family_bias:.3f}."
                        ),
                    ),
                    EvaluationScore(
                        family="abstraction",
                        metric_name="temporal_fast_prior_switch_pressure",
                        value=_clamp(0.5 + metacontroller_state.fast_prior_switch_pressure_delta),
                        confidence=0.6,
                        evidence=(
                            f"Derived from fast_prior_switch_pressure_delta="
                            f"{metacontroller_state.fast_prior_switch_pressure_delta:.3f}."
                        ),
                    ),
                )
            )
        return tuple(scores)

    def _build_turn_scores(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        dual_track_snapshot: DualTrackSnapshot | None,
    ) -> tuple[EvaluationScore, ...]:
        """Build per-turn observability readout scores.

        These scores measure what is happening in the current turn for
        **monitoring and evidence** purposes.  They are NOT reward signals.
        The primary learning signal is prediction error (R-PE).
        """
        memory_count = len(memory_snapshot.retrieved_entries) if memory_snapshot else 0
        cross_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot else 0.0
        relationship_stability = _clamp(1.0 - cross_tension)
        semantic_task_pull = (
            feature_signal_value(substrate_snapshot.feature_surface, name="semantic_task_pull")
            if substrate_snapshot is not None
            else 0.0
        )
        semantic_social_pull = (
            feature_signal_value(substrate_snapshot.feature_surface, name="semantic_social_pull")
            if substrate_snapshot is not None
            else 0.0
        )
        semantic_surface_active = (
            feature_signal_value(substrate_snapshot.feature_surface, name="semantic_surface_active")
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
        semantic_decision_delegation_pull = (
            feature_signal_value(
                substrate_snapshot.feature_surface,
                name="semantic_decision_delegation_pull",
            )
            if substrate_snapshot is not None
            else 0.0
        )
        fallback_active = (
            feature_signal_value(substrate_snapshot.feature_surface, name="fallback_active")
            if substrate_snapshot is not None
            else 0.0
        )
        hook_layer_coverage = (
            feature_signal_value(substrate_snapshot.feature_surface, name="hook_layer_coverage")
            if substrate_snapshot is not None
            else 0.0
        )
        semantic_residual_weight = (
            feature_signal_value(substrate_snapshot.feature_surface, name="semantic_residual_weight")
            if substrate_snapshot is not None
            else 0.0
        )
        top_logit_margin = (
            feature_signal_value(substrate_snapshot.feature_surface, name="top_logit_margin")
            if substrate_snapshot is not None
            else 0.0
        )
        top_logit_entropy = (
            feature_signal_value(substrate_snapshot.feature_surface, name="top_logit_entropy")
            if substrate_snapshot is not None
            else 0.0
        )
        real_path_quality = _clamp(
            (1.0 - fallback_active) * 0.45
            + hook_layer_coverage * 0.20
            + semantic_residual_weight * 0.20
            + top_logit_margin * 0.10
            + (1.0 - top_logit_entropy) * 0.05
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
            prototype=task_pressure_prototype(),
        )
        self_goal_semantics = _goal_semantic_pressure(
            dual_track_snapshot.self_track.active_goals if dual_track_snapshot else (),
            prototype=support_presence_prototype(),
        )
        task_pressure = _clamp(
            semantic_task_pull * 0.34
            + semantic_directive_pull * 0.20
            + semantic_repair_pull * 0.06
            + world_drive * 0.20
            + min(world_goal_count / 3.0, 1.0) * 0.08
            + memory_count / 5.0 * 0.10
            + world_goal_semantics * 0.12
            + real_path_quality * 0.05
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
            + real_path_quality * 0.06
        )
        info_integration = _clamp(
            memory_count / 5.0 * 0.45
            + world_drive * 0.20
            + min(world_goal_count / 3.0, 1.0) * 0.10
            + semantic_task_pull * 0.16
            + semantic_directive_pull * 0.10
            + semantic_exploration_pull * 0.05
            + real_path_quality * 0.06
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
            + real_path_quality * 0.05
        )
        relationship_stability = _clamp(
            relationship_stability
            + real_path_quality * 0.08
            + semantic_support_pull * 0.04
            - semantic_directive_pull * 0.03
        )
        placeholder_penalty = 0.12 if substrate_snapshot is None or substrate_snapshot.surface_kind is SurfaceKind.PLACEHOLDER else 0.0
        contract_integrity = _clamp(1.0 - placeholder_penalty)

        return (
            EvaluationScore(
                family="interaction",
                metric_name="social_pressure",
                value=semantic_social_pull,
                confidence=0.58,
                evidence=(
                    f"Read from substrate.semantic_social_pull={semantic_social_pull:.2f}; "
                    "used by regime selection to distinguish light social / acquaintance turns."
                ),
            ),
            EvaluationScore(
                family="interaction",
                metric_name="semantic_surface_active",
                value=semantic_surface_active,
                confidence=1.0,
                evidence="1.0 when substrate publishes the public semantic pull surface.",
            ),
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
                family="interaction",
                metric_name="repair_pressure",
                value=semantic_repair_pull,
                confidence=0.58,
                evidence=(
                    f"Read directly from substrate.semantic_repair_pull={semantic_repair_pull:.2f} "
                    "so regime selection can distinguish repair turns from low-pressure social turns."
                ),
            ),
            EvaluationScore(
                family="safety",
                metric_name="decision_delegation_pressure",
                value=semantic_decision_delegation_pull,
                confidence=0.62,
                evidence=(
                    "Read from substrate.semantic_decision_delegation_pull="
                    f"{semantic_decision_delegation_pull:.2f}; high values mean the user "
                    "is asking the system to carry a high-stakes decision rather than "
                    "just answer a bounded task."
                ),
            ),
            EvaluationScore(
                family="relationship",
                metric_name="cross_track_stability",
                value=relationship_stability,
                confidence=0.65,
                evidence=(
                    f"Computed from cross_track_tension={cross_tension:.2f}, "
                    f"real_path_quality={real_path_quality:.2f}, support_pull={semantic_support_pull:.2f}, "
                    f"directive_pull={semantic_directive_pull:.2f}."
                ),
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
            EvaluationScore(
                family="learning",
                metric_name="substrate_signal_quality",
                value=real_path_quality,
                confidence=0.72,
                evidence=(
                    f"Derived from fallback_active={fallback_active:.2f}, "
                    f"hook_layer_coverage={hook_layer_coverage:.2f}, semantic_residual_weight={semantic_residual_weight:.2f}, "
                    f"top_logit_margin={top_logit_margin:.2f}, top_logit_entropy={top_logit_entropy:.2f}."
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
        domain_knowledge_snapshot: DomainKnowledgeReadout | None = None,
        case_memory_snapshot: CaseMemoryReadout | None = None,
        strategy_playbook_snapshot: StrategyPlaybookReadout | None = None,
        boundary_policy_snapshot: BoundaryReadout | None = None,
        experience_fast_prior_snapshot: ExperienceFastPriorReadout | None = None,
        response_assembly_snapshot: ResponseAssemblyReadout | None = None,
        semantic_state_snapshots: tuple[object, ...] = (),
        delayed_outcome_ledger: tuple[ApplicationOutcomeAttributionReadout, ...] = (),
        sequence_payoffs: tuple[ApplicationSequencePayoffReadout, ...] = (),
    ) -> tuple[EvaluationScore, ...]:
        """Build readout evidence scores for the learning loop.

        These are descriptive observations, not primary learning signals.
        The actual learning driver is prediction error (R-PE).
        """
        from volvence_zero.joint_loop.runtime import ScheduledJointLoopResult
        from volvence_zero.reflection import ReflectionSnapshot, WritebackResult
        from volvence_zero.semantic_state import (
            BeliefAssumptionSnapshot,
            BoundaryConsentSnapshot,
            CommitmentSnapshot,
            ExecutionResultSnapshot,
            GoalValueSnapshot,
            OpenLoopSnapshot,
            PlanIntentSnapshot,
            RelationshipStateSnapshot,
            UserModelSnapshot,
        )

        scores: list[EvaluationScore] = []
        if memory_snapshot is not None:
            lifecycle_metrics = dict(memory_snapshot.lifecycle_metrics)
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
            nested_profile_active = lifecycle_metrics.get("nested_profile_active", 0.0)
            if nested_profile_active > 0.0:
                scores.extend(
                    (
                        EvaluationScore(
                            family="learning",
                            metric_name="nested_profile_active",
                            value=_clamp(nested_profile_active),
                            confidence=0.72,
                            evidence=(
                                f"Derived from nested_context_reset_count="
                                f"{lifecycle_metrics.get('nested_context_reset_count', 0.0):.0f}."
                            ),
                        ),
                        EvaluationScore(
                            family="learning",
                            metric_name="slow_to_fast_init_benefit",
                            value=_clamp(
                                lifecycle_metrics.get("slow_to_fast_init_benefit", 0.0)
                                + lifecycle_metrics.get("last_nested_reset_online_seed_strength", 0.0) * 0.35
                            ),
                            confidence=0.65,
                            evidence=(
                                f"Derived from transfer_strength="
                                f"{lifecycle_metrics.get('slow_to_fast_init_benefit', 0.0):.3f}, "
                                f"online_seed_strength="
                                f"{lifecycle_metrics.get('last_nested_reset_online_seed_strength', 0.0):.3f}."
                            ),
                        ),
                        EvaluationScore(
                            family="learning",
                            metric_name="nested_context_reuse",
                            value=_clamp(lifecycle_metrics.get("nested_context_reset_count", 0.0) / 3.0),
                            confidence=0.61,
                            evidence=(
                                f"Derived from nested_context_reset_count="
                                f"{lifecycle_metrics.get('nested_context_reset_count', 0.0):.0f}."
                            ),
                        ),
                    )
                )
            tower_depth = lifecycle_metrics.get("last_memory_tower_depth", 0.0)
            tower_alignment = lifecycle_metrics.get("last_memory_tower_alignment", 0.0)
            tower_consolidation_count = lifecycle_metrics.get("tower_consolidation_count", 0.0)
            runtime_backbone_observation_count = lifecycle_metrics.get(
                "runtime_backbone_observation_count", 0.0
            )
            runtime_backbone_signal_quality = lifecycle_metrics.get(
                "last_runtime_backbone_signal_quality", 0.0
            )
            runtime_backbone_signal_strength = lifecycle_metrics.get(
                "last_runtime_backbone_signal_strength", 0.0
            )
            runtime_backbone_hook_coverage = lifecycle_metrics.get(
                "last_runtime_backbone_hook_coverage", 0.0
            )
            fast_memory_signal_count = lifecycle_metrics.get("fast_memory_signal_count", 0.0)
            fast_memory_signal_norm = lifecycle_metrics.get("last_fast_memory_signal_norm", 0.0)
            fast_memory_runtime_alignment = lifecycle_metrics.get(
                "last_fast_memory_runtime_alignment", 0.0
            )
            if runtime_backbone_observation_count > 0.0:
                scores.extend(
                    (
                        EvaluationScore(
                            family="learning",
                            metric_name="runtime_backbone_signal_quality",
                            value=_clamp(runtime_backbone_signal_quality),
                            confidence=0.7,
                            evidence=(
                                f"Derived from runtime_backbone_observation_count="
                                f"{runtime_backbone_observation_count:.0f}, "
                                f"signal_quality={runtime_backbone_signal_quality:.3f}, "
                                f"hook_coverage={runtime_backbone_hook_coverage:.3f}."
                            ),
                        ),
                        EvaluationScore(
                            family="learning",
                            metric_name="runtime_backbone_signal_strength",
                            value=_clamp(runtime_backbone_signal_strength),
                            confidence=0.68,
                            evidence=(
                                f"Derived from runtime_backbone_strength={runtime_backbone_signal_strength:.3f} "
                                f"and hook_coverage={runtime_backbone_hook_coverage:.3f}."
                            ),
                        ),
                    )
                )
            if fast_memory_signal_count > 0.0:
                scores.extend(
                    (
                        EvaluationScore(
                            family="learning",
                            metric_name="fast_memory_signal_norm",
                            value=_clamp(fast_memory_signal_norm),
                            confidence=0.67,
                            evidence=(
                                f"Derived from fast_memory_signal_count={fast_memory_signal_count:.0f} "
                                f"and signal_norm={fast_memory_signal_norm:.3f}."
                            ),
                        ),
                        EvaluationScore(
                            family="learning",
                            metric_name="fast_memory_runtime_alignment",
                            value=_clamp(max(fast_memory_runtime_alignment, 0.0)),
                            confidence=0.66,
                            evidence=(
                                f"Derived from fast_memory_runtime_alignment="
                                f"{fast_memory_runtime_alignment:.3f}."
                            ),
                        ),
                    )
                )
            if updater_state := memory_snapshot.cms_state.update_rule_state if memory_snapshot.cms_state is not None else None:
                scores.extend(
                    (
                        EvaluationScore(
                            family="learning",
                            metric_name="memory_updater_effective_lr",
                            value=_clamp(updater_state.last_effective_learning_rate),
                            confidence=0.7,
                            evidence=updater_state.description,
                        ),
                        EvaluationScore(
                            family="learning",
                            metric_name="memory_updater_write_gate",
                            value=_clamp(updater_state.last_write_gate),
                            confidence=0.68,
                            evidence=updater_state.description,
                        ),
                        EvaluationScore(
                            family="learning",
                            metric_name="memory_updater_confidence",
                            value=_clamp(updater_state.last_confidence),
                            confidence=0.67,
                            evidence=updater_state.description,
                        ),
                    )
                )
            if tower_depth > 0.0:
                scores.extend(
                    (
                        EvaluationScore(
                            family="learning",
                            metric_name="memory_tower_depth",
                            value=_clamp(tower_depth / 6.0),
                            confidence=0.68,
                            evidence=(
                                f"Derived from memory tower depth={tower_depth:.1f} "
                                f"and nested_profile_active={nested_profile_active:.1f}."
                            ),
                        ),
                        EvaluationScore(
                            family="learning",
                            metric_name="memory_tower_alignment",
                            value=_clamp(max(tower_alignment, 0.0)),
                            confidence=0.64,
                            evidence=(
                                f"Derived from tower_alignment={tower_alignment:.3f} "
                                f"and retrieval_quality={retrieval_quality:.3f}."
                            ),
                        ),
                        EvaluationScore(
                            family="learning",
                            metric_name="tower_consolidation_activity",
                            value=_clamp(tower_consolidation_count / 3.0),
                            confidence=0.62,
                            evidence=(
                                f"Derived from tower_consolidation_count={tower_consolidation_count:.0f}."
                            ),
                        ),
                    )
                )
            continuum_band_count = lifecycle_metrics.get("continuum_band_count", 0.0)
            continuum_reconstruction_edge_count = lifecycle_metrics.get("continuum_reconstruction_edge_count", 0.0)
            continuum_frequency_span = lifecycle_metrics.get("continuum_frequency_span", 0.0)
            if continuum_band_count > 0.0:
                scores.extend(
                    (
                        EvaluationScore(
                            family="learning",
                            metric_name="continuum_frequency_coverage",
                            value=_clamp(continuum_band_count / 6.0),
                            confidence=0.66,
                            evidence=(
                                f"Derived from continuum_band_count={continuum_band_count:.0f} "
                                f"and frequency_span={continuum_frequency_span:.3f}."
                            ),
                        ),
                        EvaluationScore(
                            family="learning",
                            metric_name="continuum_reconstruction_capacity",
                            value=_clamp(continuum_reconstruction_edge_count / 8.0),
                            confidence=0.64,
                            evidence=(
                                f"Derived from reconstruction_edge_count={continuum_reconstruction_edge_count:.0f}."
                            ),
                        ),
                    )
                )
        if domain_knowledge_snapshot is not None:
            knowledge_hit_count = _clamp(len(domain_knowledge_snapshot.hits) / 3.0)
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="knowledge_hit_count",
                    value=knowledge_hit_count,
                    confidence=0.66,
                    evidence=(
                        f"Derived from domain_knowledge hits={len(domain_knowledge_snapshot.hits)} "
                        f"citation_required={domain_knowledge_snapshot.citation_required}."
                    ),
                )
            )
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="knowledge_conflict_exposed",
                    value=_clamp(len(domain_knowledge_snapshot.unresolved_conflicts)),
                    confidence=0.64,
                    evidence=(
                        f"Derived from unresolved_conflicts={len(domain_knowledge_snapshot.unresolved_conflicts)}."
                    ),
                )
            )
        if case_memory_snapshot is not None:
            scores.extend(
                (
                    EvaluationScore(
                        family="learning",
                        metric_name="case_hit_count",
                        value=_clamp(len(case_memory_snapshot.hits) / 3.0),
                        confidence=0.67,
                        evidence=(
                            f"Derived from case_memory hits={len(case_memory_snapshot.hits)} "
                            f"problem_patterns={len(case_memory_snapshot.active_problem_patterns)}."
                        ),
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="case_relevance_mean",
                        value=_clamp(
                            sum(hit.relevance_score for hit in case_memory_snapshot.hits)
                            / max(len(case_memory_snapshot.hits), 1)
                        ),
                        confidence=0.62,
                        evidence=(
                            f"Derived from case hit relevance over {len(case_memory_snapshot.hits)} hits."
                        ),
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="case_delayed_outcome_availability",
                        value=_clamp(
                            sum(hit.outcome.delayed_signal_count for hit in case_memory_snapshot.hits)
                            / max(len(case_memory_snapshot.hits), 1)
                            / 4.0
                        ),
                        confidence=0.59,
                        evidence=(
                            f"Derived from delayed_signal_count across {len(case_memory_snapshot.hits)} case hits."
                        ),
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="application_continuum_case_coverage",
                        value=_clamp(len(case_memory_snapshot.active_band_ids) / 4.0),
                        confidence=0.63,
                        evidence=(
                            f"Derived from case continuum bands={len(case_memory_snapshot.active_band_ids)} "
                            f"mean_position={case_memory_snapshot.mean_continuum_position:.3f}."
                        ),
                    ),
                )
            )
        if strategy_playbook_snapshot is not None:
            scores.extend(
                (
                    EvaluationScore(
                        family="learning",
                        metric_name="playbook_match_count",
                        value=_clamp(len(strategy_playbook_snapshot.matched_rules) / 3.0),
                        confidence=0.66,
                        evidence=(
                            f"Derived from strategy_playbook matched_rules="
                            f"{len(strategy_playbook_snapshot.matched_rules)}."
                        ),
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="playbook_confidence_mean",
                        value=_clamp(
                            sum(rule.confidence for rule in strategy_playbook_snapshot.matched_rules)
                            / max(len(strategy_playbook_snapshot.matched_rules), 1)
                        ),
                        confidence=0.61,
                        evidence=(
                            f"Derived from matched_problem_patterns="
                            f"{len(strategy_playbook_snapshot.matched_problem_patterns)}."
                        ),
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="application_continuum_playbook_transfer",
                        value=_clamp(len(strategy_playbook_snapshot.active_band_ids) / 4.0),
                        confidence=0.61,
                        evidence=(
                            f"Derived from playbook continuum bands={len(strategy_playbook_snapshot.active_band_ids)}."
                        ),
                    ),
                )
            )
        if experience_fast_prior_snapshot is not None:
            scores.extend(
                (
                    EvaluationScore(
                        family="learning",
                        metric_name="delayed_fast_prior_available",
                        value=1.0
                        if (
                            experience_fast_prior_snapshot.source_attribution_ids
                            or experience_fast_prior_snapshot.source_sequence_ids
                        )
                        else 0.0,
                        confidence=0.72,
                        evidence=experience_fast_prior_snapshot.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="delayed_regime_bias_applied",
                        value=_clamp(
                            sum(abs(item.bias) for item in experience_fast_prior_snapshot.regime_biases)
                            / max(len(experience_fast_prior_snapshot.regime_biases), 1)
                        )
                        if experience_fast_prior_snapshot.regime_biases
                        else 0.0,
                        confidence=0.68,
                        evidence=experience_fast_prior_snapshot.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="delayed_retrieval_mix_bias_applied",
                        value=_clamp(
                            (
                                abs(experience_fast_prior_snapshot.knowledge_weight_bias)
                                + abs(experience_fast_prior_snapshot.experience_weight_bias)
                            )
                            / 2.0
                        ),
                        confidence=0.68,
                        evidence=experience_fast_prior_snapshot.description,
                    ),
                    EvaluationScore(
                        family="abstraction",
                        metric_name="delayed_action_family_bias_applied",
                        value=_clamp(
                            sum(abs(item.continuation_bias) for item in experience_fast_prior_snapshot.family_biases)
                            / max(len(experience_fast_prior_snapshot.family_biases), 1)
                        )
                        if experience_fast_prior_snapshot.family_biases
                        else 0.0,
                        confidence=0.67,
                        evidence=experience_fast_prior_snapshot.description,
                    ),
                )
            )
        if delayed_outcome_ledger:
            mean_outcome = _clamp(
                sum(item.outcome_score for item in delayed_outcome_ledger) / len(delayed_outcome_ledger)
            )
            mean_mix_alignment = _clamp(
                sum(item.retrieval_mix_alignment for item in delayed_outcome_ledger) / len(delayed_outcome_ledger)
            )
            mean_regime_alignment = _clamp(
                sum(item.regime_alignment for item in delayed_outcome_ledger) / len(delayed_outcome_ledger)
            )
            mean_action_alignment = _clamp(
                sum(item.abstract_action_alignment for item in delayed_outcome_ledger) / len(delayed_outcome_ledger)
            )
            mean_continuum_alignment = _clamp(
                sum(item.continuum_alignment for item in delayed_outcome_ledger) / len(delayed_outcome_ledger)
            )
            scores.extend(
                (
                    EvaluationScore(
                        family="learning",
                        metric_name="delayed_retrieval_mix_alignment",
                        value=mean_mix_alignment,
                        confidence=0.7,
                        evidence=(
                            f"Derived from delayed ledger count={len(delayed_outcome_ledger)} "
                            f"mean_outcome={mean_outcome:.3f}."
                        ),
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="delayed_regime_alignment",
                        value=mean_regime_alignment,
                        confidence=0.7,
                        evidence=(
                            f"Derived from delayed ledger count={len(delayed_outcome_ledger)} "
                            f"mean_outcome={mean_outcome:.3f}."
                        ),
                    ),
                    EvaluationScore(
                        family="abstraction",
                        metric_name="delayed_abstract_action_alignment",
                        value=mean_action_alignment,
                        confidence=0.7,
                        evidence=(
                            f"Derived from delayed ledger count={len(delayed_outcome_ledger)} "
                            f"mean_outcome={mean_outcome:.3f}."
                        ),
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="application_continuum_alignment",
                        value=mean_continuum_alignment,
                        confidence=0.68,
                        evidence=(
                            f"Derived from delayed ledger count={len(delayed_outcome_ledger)} "
                            f"mean_continuum_alignment={mean_continuum_alignment:.3f}."
                        ),
                    ),
                )
            )
        if sequence_payoffs:
            mean_sequence_payoff = _clamp(
                sum(item.rolling_payoff for item in sequence_payoffs) / len(sequence_payoffs)
            )
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="regime_sequence_payoff",
                    value=mean_sequence_payoff,
                    confidence=0.68,
                    evidence=(
                        f"Derived from sequence payoff count={len(sequence_payoffs)} "
                        f"latest_outcome_mean={mean_sequence_payoff:.3f}."
                    ),
                )
            )
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="application_continuum_sequence_payoff",
                    value=_clamp(
                        sum(item.mean_continuum_position for item in sequence_payoffs) / len(sequence_payoffs)
                    ),
                    confidence=0.63,
                    evidence=(
                        f"Derived from sequence payoff count={len(sequence_payoffs)} with continuum positions."
                    ),
                )
            )
        if boundary_policy_snapshot is not None:
            scores.extend(
                (
                    EvaluationScore(
                        family="safety",
                        metric_name="boundary_clarification_triggered",
                        value=1.0 if boundary_policy_snapshot.active_decision.clarification_required else 0.0,
                        confidence=0.72,
                        evidence=boundary_policy_snapshot.description,
                    ),
                    EvaluationScore(
                        family="safety",
                        metric_name="boundary_referral_triggered",
                        value=1.0 if boundary_policy_snapshot.active_decision.refer_out_required else 0.0,
                        confidence=0.74,
                        evidence=boundary_policy_snapshot.description,
                    ),
                    EvaluationScore(
                        family="safety",
                        metric_name="boundary_citation_required",
                        value=1.0 if boundary_policy_snapshot.active_decision.citation_required else 0.0,
                        confidence=0.68,
                        evidence=boundary_policy_snapshot.description,
                    ),
                )
            )
        if response_assembly_snapshot is not None:
            scores.extend(
                (
                    EvaluationScore(
                        family="learning",
                        metric_name="response_depth_compliance",
                        value=1.0 if response_assembly_snapshot.answer_depth_limit in {"support-first", "standard", "high-level-only"} else 0.0,
                        confidence=0.68,
                        evidence=response_assembly_snapshot.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="clarification_compliance",
                        value=1.0 if response_assembly_snapshot.max_questions <= 1 else 0.0,
                        confidence=0.66,
                        evidence=response_assembly_snapshot.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="refer_out_compliance",
                        value=1.0 if (
                            not response_assembly_snapshot.refer_out_required
                            or bool(response_assembly_snapshot.required_disclaimer_phrases)
                        ) else 0.0,
                        confidence=0.65,
                        evidence=response_assembly_snapshot.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="ordering_plan_alignment",
                        value=_clamp(len(response_assembly_snapshot.ordering_plan) / 3.0),
                        confidence=0.63,
                        evidence=response_assembly_snapshot.description,
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="prompt_residue_ratio",
                        value=_clamp(response_assembly_snapshot.prompt_residue_ratio),
                        confidence=0.64,
                        evidence=response_assembly_snapshot.prompt_residue_summary,
                    ),
                    EvaluationScore(
                        family="interaction",
                        metric_name="support_before_decision_pressure",
                        value=_clamp(response_assembly_snapshot.support_before_decision_pressure),
                        confidence=0.66,
                        evidence=response_assembly_snapshot.description,
                    ),
                    EvaluationScore(
                        family="abstraction",
                        metric_name="emotional_decision_action_family",
                        value=1.0 if response_assembly_snapshot.eta_action_family else 0.0,
                        confidence=0.62,
                        evidence=response_assembly_snapshot.description,
                    ),
                )
            )
        for semantic_snapshot in semantic_state_snapshots:
            if isinstance(semantic_snapshot, PlanIntentSnapshot):
                scores.append(
                    EvaluationScore(
                        family="learning",
                        metric_name="plan_continuity",
                        value=_clamp(semantic_snapshot.continuity_score),
                        confidence=0.64,
                        evidence=semantic_snapshot.description,
                    )
                )
            elif isinstance(semantic_snapshot, CommitmentSnapshot):
                scores.append(
                    EvaluationScore(
                        family="relationship",
                        metric_name="commitment_honoring",
                        value=_clamp(semantic_snapshot.continuity_score),
                        confidence=0.62,
                        evidence=semantic_snapshot.description,
                    )
                )
            elif isinstance(semantic_snapshot, OpenLoopSnapshot):
                scores.append(
                    EvaluationScore(
                        family="interaction",
                        metric_name="open_loop_closure_pressure",
                        value=_clamp(semantic_snapshot.closure_pressure),
                        confidence=0.61,
                        evidence=semantic_snapshot.description,
                    )
                )
            elif isinstance(semantic_snapshot, UserModelSnapshot):
                scores.extend(
                    (
                        EvaluationScore(
                            family="relationship",
                            metric_name="user_model_stability",
                            value=_clamp(semantic_snapshot.stability_score),
                            confidence=0.60,
                            evidence=semantic_snapshot.description,
                        ),
                        EvaluationScore(
                            family="interaction",
                            metric_name="user_overwhelm_pattern_strength",
                            value=_clamp(semantic_snapshot.overwhelm_pattern_strength),
                            confidence=0.62,
                            evidence=semantic_snapshot.description,
                        ),
                    )
                )
            elif isinstance(semantic_snapshot, ExecutionResultSnapshot):
                scores.append(
                    EvaluationScore(
                        family="task",
                        metric_name="execution_grounding",
                        value=_clamp(semantic_snapshot.execution_grounding_score),
                        confidence=0.65,
                        evidence=semantic_snapshot.description,
                    )
                )
            elif isinstance(semantic_snapshot, BeliefAssumptionSnapshot):
                scores.append(
                    EvaluationScore(
                        family="safety",
                        metric_name="belief_verification",
                        value=_clamp(semantic_snapshot.mean_confidence),
                        confidence=0.63,
                        evidence=semantic_snapshot.description,
                    )
                )
            elif isinstance(semantic_snapshot, RelationshipStateSnapshot):
                scores.extend(
                    (
                        EvaluationScore(
                            family="relationship",
                            metric_name="relationship_continuity",
                            value=_clamp(semantic_snapshot.continuity_level),
                            confidence=0.62,
                            evidence=semantic_snapshot.description,
                        ),
                        EvaluationScore(
                            family="interaction",
                            metric_name="owner_emotional_load",
                            value=_clamp(semantic_snapshot.emotional_load),
                            confidence=0.64,
                            evidence=semantic_snapshot.description,
                        ),
                        EvaluationScore(
                            family="interaction",
                            metric_name="owner_stabilization_need",
                            value=_clamp(semantic_snapshot.stabilization_need),
                            confidence=0.64,
                            evidence=semantic_snapshot.description,
                        ),
                    )
                )
            elif isinstance(semantic_snapshot, GoalValueSnapshot):
                scores.extend(
                    (
                        EvaluationScore(
                            family="task",
                            metric_name="goal_alignment",
                            value=_clamp(semantic_snapshot.alignment_score),
                            confidence=0.62,
                            evidence=semantic_snapshot.description,
                        ),
                        EvaluationScore(
                            family="task",
                            metric_name="owner_value_conflict",
                            value=_clamp(semantic_snapshot.value_conflict),
                            confidence=0.63,
                            evidence=semantic_snapshot.description,
                        ),
                        EvaluationScore(
                            family="task",
                            metric_name="owner_decision_readiness",
                            value=_clamp(semantic_snapshot.decision_readiness),
                            confidence=0.63,
                            evidence=semantic_snapshot.description,
                        ),
                    )
                )
            elif isinstance(semantic_snapshot, BoundaryConsentSnapshot):
                scores.extend(
                    (
                        EvaluationScore(
                            family="safety",
                            metric_name="consent_compliance",
                            value=_clamp(semantic_snapshot.compliance_score),
                            confidence=0.66,
                            evidence=semantic_snapshot.description,
                        ),
                        EvaluationScore(
                            family="safety",
                            metric_name="owner_autonomy_risk",
                            value=_clamp(semantic_snapshot.autonomy_risk),
                            confidence=0.66,
                            evidence=semantic_snapshot.description,
                        ),
                        EvaluationScore(
                            family="safety",
                            metric_name="owner_overreach_risk",
                            value=_clamp(semantic_snapshot.overreach_risk),
                            confidence=0.66,
                            evidence=semantic_snapshot.description,
                        ),
                    )
                )
        scores.extend(
            self._semantic_spine_readiness_scores(
                semantic_state_snapshots=semantic_state_snapshots,
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
            schedule_telemetry = dict(joint_loop_result.schedule_telemetry)
            scheduler_pe_pressure = _clamp(schedule_telemetry.get("pe_pressure_x1000", 0) / 1000.0)
            scheduler_family_stability = _clamp(schedule_telemetry.get("family_stability_x1000", 0) / 1000.0)
            scheduler_rollback_risk = _clamp(schedule_telemetry.get("rollback_risk_x1000", 0) / 1000.0)
            scheduler_transition_pressure = _clamp(schedule_telemetry.get("transition_pressure_x1000", 0) / 1000.0)
            scheduler_substrate_pressure = _clamp(schedule_telemetry.get("substrate_pressure_x1000", 0) / 1000.0)
            scheduler_rare_heavy_pressure = _clamp(schedule_telemetry.get("rare_heavy_pressure_x1000", 0) / 1000.0)
            scheduler_experience_credit = _clamp(schedule_telemetry.get("experience_credit_x1000", 0) / 1000.0)
            scheduler_control_prior_strength = _clamp(
                schedule_telemetry.get("control_prior_strength_x1000", 0) / 1000.0
            )
            batch_target = max(schedule_telemetry.get("rl_batch_target", 1), 1)
            pending_batch_count = max(schedule_telemetry.get("pending_batch_count", 0), 0)
            scheduler_batch_fill_ratio = _clamp(pending_batch_count / batch_target)
            scheduler_risk_managed = _clamp(1.0 - scheduler_rollback_risk)
            timescale_contract_retained = _clamp(
                0.35
                + scheduler_risk_managed * 0.20
                + (1.0 - scheduler_transition_pressure) * 0.15
                + (1.0 - scheduler_substrate_pressure) * 0.10
                + (1.0 - scheduler_rare_heavy_pressure) * 0.10
                + float(joint_loop_result.substrate_online_fast_due or joint_loop_result.rare_heavy_review_recommended) * 0.10
            )
            scheduler_discipline = _clamp(
                scheduler_family_stability * 0.30
                + scheduler_risk_managed * 0.30
                + (1.0 - scheduler_transition_pressure) * 0.15
                + (1.0 - scheduler_rare_heavy_pressure) * 0.10
                + (1.0 - scheduler_substrate_pressure) * 0.05
                + scheduler_experience_credit * 0.05
                + scheduler_control_prior_strength * 0.05
                + scheduler_batch_fill_ratio * 0.10
            )
            scores.extend(
                (
                    EvaluationScore(
                        family="learning",
                        metric_name="scheduler_pe_pressure",
                        value=scheduler_pe_pressure,
                        confidence=0.62,
                        evidence=f"Derived from schedule_action={schedule_action} and joint schedule telemetry.",
                    ),
                    EvaluationScore(
                        family="abstraction",
                        metric_name="scheduler_family_stability",
                        value=scheduler_family_stability,
                        confidence=0.63,
                        evidence=f"Derived from schedule_action={schedule_action} and active family telemetry.",
                    ),
                    EvaluationScore(
                        family="safety",
                        metric_name="scheduler_rollback_risk",
                        value=scheduler_rollback_risk,
                        confidence=0.66,
                        evidence=f"Derived from schedule_action={schedule_action} and rollback-oriented telemetry.",
                    ),
                    EvaluationScore(
                        family="abstraction",
                        metric_name="scheduler_transition_pressure",
                        value=scheduler_transition_pressure,
                        confidence=0.61,
                        evidence=f"Derived from schedule_action={schedule_action} and takeover pressure telemetry.",
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="scheduler_substrate_pressure",
                        value=scheduler_substrate_pressure,
                        confidence=0.58,
                        evidence=f"Derived from schedule_action={schedule_action} and substrate pressure telemetry.",
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="scheduler_rare_heavy_pressure",
                        value=scheduler_rare_heavy_pressure,
                        confidence=0.6,
                        evidence=f"Derived from schedule_action={schedule_action} and rare-heavy pressure telemetry.",
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="scheduler_experience_credit",
                        value=scheduler_experience_credit,
                        confidence=0.61,
                        evidence=f"Derived from schedule_action={schedule_action} and delayed experience telemetry.",
                    ),
                    EvaluationScore(
                        family="abstraction",
                        metric_name="scheduler_control_prior_strength",
                        value=scheduler_control_prior_strength,
                        confidence=0.61,
                        evidence=f"Derived from schedule_action={schedule_action} and case/playbook prior telemetry.",
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="scheduler_batch_fill_ratio",
                        value=scheduler_batch_fill_ratio,
                        confidence=0.59,
                        evidence=f"Derived from pending_batch_count={pending_batch_count} and rl_batch_target={batch_target}.",
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="scheduler_risk_managed",
                        value=scheduler_risk_managed,
                        confidence=0.63,
                        evidence=f"Derived from scheduler_rollback_risk={scheduler_rollback_risk:.3f}.",
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="scheduler_discipline",
                        value=scheduler_discipline,
                        confidence=0.64,
                        evidence=f"Derived from schedule_action={schedule_action} and scheduler pressure balance.",
                    ),
                    EvaluationScore(
                        family="learning",
                        metric_name="timescale_contract_retained",
                        value=timescale_contract_retained,
                        confidence=0.67,
                        evidence=(
                            f"Derived from schedule_action={schedule_action}, "
                            f"substrate_due={int(joint_loop_result.substrate_online_fast_due)}, "
                            f"rare_heavy_due={int(joint_loop_result.rare_heavy_review_recommended)}."
                        ),
                    ),
                )
            )
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
            elif schedule_action in {"ssl-only", "ssl-only-pe"}:
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
                continual_surface = cycle_report.default_continual_learning_surface
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
                if continual_surface is not None:
                    owner_writeback_retained = float(
                        continual_surface.memory_regime_writeback_applied
                        or continual_surface.temporal_writeback_applied
                        or continual_surface.regime_evidence_applied
                    )
                    scores.extend(
                        (
                            EvaluationScore(
                                family="learning",
                                metric_name="default_continual_learning_active",
                                value=float(continual_surface.active),
                                confidence=0.72,
                                evidence=continual_surface.description,
                            ),
                            EvaluationScore(
                                family="learning",
                                metric_name="default_owner_writeback_retained",
                                value=owner_writeback_retained,
                                confidence=0.72,
                                evidence=(
                                    "Derived from owner-side memory/regime/temporal/reflection "
                                    f"surface {continual_surface.surface_id}."
                                ),
                            ),
                            EvaluationScore(
                                family="safety",
                                metric_name="default_substrate_live_mutation_suppressed",
                                value=float(
                                    continual_surface.substrate_review_only
                                    and not continual_surface.substrate_live_mutation_applied
                                ),
                                confidence=0.74,
                                evidence=(
                                    "Default continual learner retains substrate review-only doctrine; "
                                    f"rare_heavy_review={int(continual_surface.rare_heavy_review_recommended)}."
                                ),
                            ),
                            EvaluationScore(
                                family="safety",
                                metric_name="default_continual_rollback_clean",
                                value=1.0,
                                confidence=0.7,
                                evidence=(
                                    f"Surface rollback_applied={int(continual_surface.rollback_applied)} "
                                    f"blocked={continual_surface.blocked_operations}."
                                ),
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
            max_resolved_horizon = 0
            if regime_snapshot.delayed_attributions:
                max_resolved_horizon = max(
                    (a.resolved_turn_index - a.source_turn_index for a in regime_snapshot.delayed_attributions),
                    default=0,
                )
            horizon_quality = _clamp(max_resolved_horizon / 8.0)
            rolling_sample_density = _clamp(
                sum(item.sample_count for item in regime_snapshot.delayed_payoffs)
                / max(len(regime_snapshot.delayed_payoffs) * 3.0, 1.0)
            )
            delayed_credit_horizon = _clamp(horizon_quality * 0.6 + rolling_sample_density * 0.4)
            scores.extend(
                (
                    EvaluationScore(
                        family="learning",
                        metric_name="delayed_credit_horizon",
                        value=delayed_credit_horizon,
                        confidence=0.58,
                        evidence=(
                            f"Derived from max_resolved_horizon={max_resolved_horizon}, "
                            f"sample_counts={tuple(item.sample_count for item in regime_snapshot.delayed_payoffs)}."
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
        if regime_snapshot is not None and regime_snapshot.sequence_payoffs:
            sequence_alignment = _clamp(
                sum(item.rolling_payoff for item in regime_snapshot.sequence_payoffs)
                / len(regime_snapshot.sequence_payoffs)
            )
            scores.append(
                EvaluationScore(
                    family="learning",
                    metric_name="regime_sequence_alignment",
                    value=sequence_alignment,
                    confidence=0.58,
                    evidence=(
                        f"Derived from sequence_payoffs={len(regime_snapshot.sequence_payoffs)} entries."
                    ),
                )
            )
        return tuple(scores)

    def _semantic_spine_readiness_scores(
        self,
        *,
        semantic_state_snapshots: tuple[object, ...],
    ) -> tuple[EvaluationScore, ...]:
        """Read the narrow cognitive-loop spine from public owner snapshots."""
        from volvence_zero.semantic_state import (
            BoundaryConsentSnapshot,
            CommitmentSnapshot,
            ExecutionResultSnapshot,
            GoalValueSnapshot,
            RelationshipStateSnapshot,
        )

        relationship: RelationshipStateSnapshot | None = None
        goal: GoalValueSnapshot | None = None
        boundary: BoundaryConsentSnapshot | None = None
        commitment: CommitmentSnapshot | None = None
        execution: ExecutionResultSnapshot | None = None
        for snapshot in semantic_state_snapshots:
            if isinstance(snapshot, RelationshipStateSnapshot):
                relationship = snapshot
            elif isinstance(snapshot, GoalValueSnapshot):
                goal = snapshot
            elif isinstance(snapshot, BoundaryConsentSnapshot):
                boundary = snapshot
            elif isinstance(snapshot, CommitmentSnapshot):
                commitment = snapshot
            elif isinstance(snapshot, ExecutionResultSnapshot):
                execution = snapshot

        present = tuple(
            item
            for item in (relationship, goal, boundary, commitment, execution)
            if item is not None
        )
        if not present:
            return ()

        coverage = _clamp(len(present) / 5.0)
        relationship_readiness = relationship.continuity_level if relationship is not None else 0.0
        goal_readiness = goal.decision_readiness if goal is not None else 0.0
        boundary_readiness = (
            _clamp(boundary.compliance_score * (1.0 - boundary.overreach_risk))
            if boundary is not None
            else 0.0
        )
        commitment_readiness = (
            _clamp(
                commitment.continuity_score * 0.55
                + min(commitment.trust_obligation_count / 3.0, 1.0) * 0.25
                + min(commitment.outcome_completed_count / 2.0, 1.0) * 0.20
            )
            if commitment is not None
            else 0.0
        )
        execution_readiness = (
            execution.execution_grounding_score if execution is not None else 0.0
        )
        cognitive_loop_readiness = _clamp(
            (
                relationship_readiness
                + goal_readiness
                + boundary_readiness
                + commitment_readiness
                + execution_readiness
            )
            / 5.0
            * coverage
        )
        evidence = (
            "Derived from core semantic owner snapshots: "
            f"relationship={'present' if relationship is not None else 'missing'}, "
            f"goal={'present' if goal is not None else 'missing'}, "
            f"boundary={'present' if boundary is not None else 'missing'}, "
            f"commitment={'present' if commitment is not None else 'missing'}, "
            f"execution={'present' if execution is not None else 'missing'}."
        )
        return (
            EvaluationScore(
                family="learning",
                metric_name="semantic_spine_coverage",
                value=coverage,
                confidence=0.72,
                evidence=evidence,
            ),
            EvaluationScore(
                family="learning",
                metric_name="cognitive_loop_readiness",
                value=cognitive_loop_readiness,
                confidence=0.68,
                evidence=(
                    f"{evidence} readiness components: "
                    f"relationship={relationship_readiness:.3f}, "
                    f"goal={goal_readiness:.3f}, "
                    f"boundary={boundary_readiness:.3f}, "
                    f"commitment={commitment_readiness:.3f}, "
                    f"execution={execution_readiness:.3f}."
                ),
            ),
        )

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

    def _build_structured_alerts(self, *, turn_scores: tuple[EvaluationScore, ...]) -> tuple[EvaluationAlert, ...]:
        alerts: list[EvaluationAlert] = []
        for score in turn_scores:
            if score.family == "relationship" and score.value < 0.4:
                alerts.append(
                    EvaluationAlert(
                        code="cross_track_stability_degraded",
                        severity="HIGH",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="cross-track stability is degraded",
                    )
                )
            if score.metric_name == "contract_integrity" and score.value < 0.95:
                alerts.append(
                    EvaluationAlert(
                        code="contract_integrity_low",
                        severity="HIGH",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="contract integrity below threshold",
                    )
                )
            if score.metric_name == "fallback_reliance" and score.value > 0.5:
                alerts.append(
                    EvaluationAlert(
                        code="substrate_fallback_active",
                        severity="MEDIUM",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="substrate fallback is active",
                    )
                )
            if score.metric_name == "rollback_resilience" and score.value < 0.6:
                alerts.append(
                    EvaluationAlert(
                        code="rollback_pressure_elevated",
                        severity="MEDIUM",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="rollback pressure is elevated",
                    )
                )
            if score.metric_name == "scheduler_rollback_risk" and score.value > 0.7:
                alerts.append(
                    EvaluationAlert(
                        code="scheduler_rollback_risk_elevated",
                        severity="HIGH",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="scheduler rollback risk is elevated",
                    )
                )
            if score.metric_name == "scheduler_transition_pressure" and score.value > 0.75:
                alerts.append(
                    EvaluationAlert(
                        code="scheduler_transition_pressure_elevated",
                        severity="MEDIUM",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="scheduler transition pressure is elevated",
                    )
                )
            if score.metric_name == "scheduler_rare_heavy_pressure" and score.value > 0.8:
                alerts.append(
                    EvaluationAlert(
                        code="scheduler_rare_heavy_pressure_elevated",
                        severity="MEDIUM",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="scheduler rare-heavy pressure is elevated",
                    )
                )
            if score.metric_name == "action_family_monopoly_pressure" and score.value > 0.72:
                alerts.append(
                    EvaluationAlert(
                        code="action_family_monopoly_pressure_elevated",
                        severity="MEDIUM",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="action-family monopoly pressure is elevated",
                    )
                )
            if score.metric_name == "action_family_collapse_risk" and score.value > 0.68:
                alerts.append(
                    EvaluationAlert(
                        code="action_family_collapse_risk_elevated",
                        severity="HIGH",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="action-family collapse risk is elevated",
                    )
                )
            if score.metric_name == "family_outcome_divergence" and score.value < 0.05:
                alerts.append(
                    EvaluationAlert(
                        code="family_outcome_stagnation",
                        severity="MEDIUM",
                        family=score.family,
                        metric_name=score.metric_name,
                        description="family outcome stagnation - all families have indistinguishable outcomes",
                    )
                )
        return tuple(alerts)

    def _build_alerts(self, *, turn_scores: tuple[EvaluationScore, ...]) -> tuple[str, ...]:
        return tuple(alert.legacy_text for alert in self._build_structured_alerts(turn_scores=turn_scores))

    def _longitudinal_trends(
        self,
        records: tuple[EvaluationRecord, ...],
    ) -> tuple[tuple[str, str, float], ...]:
        return (
            ("relationship", "relationship_continuity", self._trend_for_metrics(records, ("cross_track_stability",))),
            (
                "relationship",
                "identity_continuity",
                self._trend_for_metrics(
                    records,
                    (
                        "cross_track_stability",
                        "delayed_regime_alignment",
                        "regime_sequence_payoff",
                    ),
                ),
            ),
            (
                "relationship",
                "relationship_repair_continuity",
                self._trend_for_metrics(
                    records,
                    (
                        "support_presence",
                        "warmth",
                        "delayed_regime_alignment",
                    ),
                ),
            ),
            (
                "learning",
                "learning_quality",
                self._trend_for_metrics(
                    records,
                    (
                        "joint_learning_progress",
                        "reflection_usefulness",
                        "retrieval_quality",
                        "slow_to_fast_init_benefit",
                        "nested_context_reuse",
                        "rollback_resilience",
                        "delayed_credit_horizon",
                        "delayed_retrieval_mix_alignment",
                        "delayed_regime_alignment",
                        "regime_sequence_payoff",
                        "regime_sequence_alignment",
                        "scheduler_discipline",
                        "scheduler_risk_managed",
                    ),
                ),
            ),
            (
                "learning",
                "scheduler_health",
                self._trend_for_metrics(
                    records,
                    (
                        "scheduler_discipline",
                        "scheduler_risk_managed",
                    ),
                ),
            ),
            (
                "safety",
                "async_robustness",
                self._trend_for_metrics(
                    records,
                    (
                        "rollback_resilience",
                        "scheduler_risk_managed",
                        "residual_env_fidelity",
                    ),
                ),
            ),
            (
                "learning",
                "semantic_spine_readiness",
                self._trend_for_metrics(
                    records,
                    ("cognitive_loop_readiness",),
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
                        "delayed_abstract_action_alignment",
                        "scheduler_family_stability",
                        "scheduler_transition_pressure",
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
        if metric_name in {
            "scheduler_pe_pressure",
            "scheduler_family_stability",
            "scheduler_rollback_risk",
            "scheduler_transition_pressure",
            "scheduler_substrate_pressure",
            "scheduler_rare_heavy_pressure",
            "scheduler_batch_fill_ratio",
            "scheduler_risk_managed",
            "scheduler_discipline",
        }:
            return ("joint_loop.schedule_action", "joint_loop.schedule_telemetry")
        if metric_name == "residual_env_fidelity":
            return ("joint_loop.backend_name", "joint_loop.backend_fidelity")
        if metric_name == "delayed_regime_alignment":
            return ("regime.delayed_outcomes",)
        if metric_name in {"delayed_action_alignment", "regime_sequence_payoff"}:
            return ("regime.delayed_attributions",)
        if metric_name in {"delayed_credit_horizon", "rolling_action_payoff"}:
            return ("regime.delayed_payoffs",)
        if metric_name == "regime_sequence_alignment":
            return ("regime.sequence_payoffs",)
        if metric_name in {"semantic_spine_coverage", "cognitive_loop_readiness"}:
            return (
                "semantic_state.relationship_state",
                "semantic_state.goal_value",
                "semantic_state.boundary_consent",
                "semantic_state.commitment",
                "semantic_state.execution_result",
            )
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
            "family_outcome_divergence",
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
            alert_code = EvaluationAlert.from_legacy_text(message).code
            if alert_code == "action_family_monopoly_pressure_elevated":
                recommendations.append("Reduce active-family monopoly before widening structure changes.")
            if alert_code == "action_family_collapse_risk_elevated":
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
