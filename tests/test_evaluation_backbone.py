from __future__ import annotations

import asyncio

from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import (
    CrossSessionBenchmarkSuite,
    EvaluationBackbone,
    EvaluationModule,
    EvaluationReplayCase,
    EvaluationReport,
    EvaluationScore,
    EvaluationSnapshot,
    EvolutionDecision,
)
from volvence_zero.evaluation.backbone import _feature_surface_snapshot
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.regime import (
    DelayedOutcomeAttribution,
    DelayedOutcomePayoff,
    RegimeIdentity,
    RegimeSnapshot,
)
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
    SubstrateSnapshot,
    SurfaceKind,
)
from volvence_zero.temporal import (
    ActionFamilyPublicSummary,
    ControllerState,
    FullLearnedTemporalPolicy,
    LearnedLiteTemporalPolicy,
    MetacontrollerRuntimeState,
    TemporalAbstractionSnapshot,
    TemporalModule,
)


def test_evaluation_backbone_produces_turn_scores_and_alerts():
    backbone = EvaluationBackbone()
    snapshot = asyncio.run(
        EvaluationModule(backbone=backbone, wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=10,
        )
    )

    assert len(snapshot.value.turn_scores) >= 4
    assert snapshot.value.session_scores
    assert snapshot.value.description.startswith("Evaluation backbone produced")


def test_evaluation_backbone_builds_session_report():
    backbone = EvaluationBackbone()
    module = EvaluationModule(backbone=backbone, wiring_level=WiringLevel.ACTIVE)
    asyncio.run(module.process_standalone(session_id="s1", wave_id="w1", timestamp_ms=10))
    asyncio.run(module.process_standalone(session_id="s1", wave_id="w2", timestamp_ms=20))

    report = backbone.build_session_report(session_id="s1", timestamp_ms=30)
    assert report.report_type == "session"
    assert report.session_ids == ("s1",)
    assert report.scores_by_family
    assert report.trends


def test_evaluation_backbone_uses_public_temporal_snapshot_fields():
    backbone = EvaluationBackbone()
    snapshot = asyncio.run(
        EvaluationModule(backbone=backbone, wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w-temporal",
            timestamp_ms=10,
            temporal_snapshot=TemporalAbstractionSnapshot(
                controller_state=ControllerState(
                    code=(0.35, 0.55, 0.25),
                    code_dim=3,
                    switch_gate=0.15,
                    is_switching=False,
                    steps_since_switch=3,
                ),
                active_abstract_action="task_controller",
                controller_params_hash="temporal-hash",
                description="stable temporal action",
            ),
        )
    )

    metrics = {score.metric_name: score.value for score in snapshot.value.turn_scores}

    assert "temporal_action_commitment" in metrics
    assert metrics["temporal_action_commitment"] > 0.5


def test_evaluation_backbone_records_delayed_action_and_regime_payoff():
    backbone = EvaluationBackbone()
    enriched = backbone.record_learning_evidence(
        session_id="s-delayed",
        wave_id="w-delayed",
        timestamp_ms=25,
        base_snapshot=EvaluationSnapshot(turn_scores=(), session_scores=(), alerts=(), description="base"),
        memory_snapshot=None,
        reflection_snapshot=None,
        writeback_result=None,
        joint_loop_result=None,
        regime_snapshot=RegimeSnapshot(
            active_regime=RegimeIdentity(
                regime_id="guided_exploration",
                name="guided exploration",
                embedding=(0.4, 0.4, 0.5),
                entry_conditions="balanced",
                exit_conditions="narrow",
                historical_effectiveness=0.6,
            ),
            previous_regime=None,
            switch_reason="test",
            candidate_regimes=(("guided_exploration", 0.8),),
            turns_in_current_regime=2,
            description="regime snapshot",
            delayed_outcomes=(("guided_exploration", 0.74),),
            delayed_attributions=(
                DelayedOutcomeAttribution(
                    regime_id="guided_exploration",
                    outcome_score=0.74,
                    source_turn_index=2,
                    source_wave_id="wave-2",
                    abstract_action="discovered_family_3",
                    action_family_version=5,
                ),
            ),
            delayed_payoffs=(
                DelayedOutcomePayoff(
                    regime_id="guided_exploration",
                    abstract_action="discovered_family_3",
                    action_family_version=5,
                    sample_count=3,
                    rolling_payoff=0.71,
                    latest_outcome=0.74,
                    last_source_wave_id="wave-2",
                ),
            ),
        ),
    )

    metrics = {score.metric_name for score in enriched.turn_scores}
    assert "delayed_regime_alignment" in metrics
    assert "delayed_action_alignment" in metrics
    assert "regime_sequence_payoff" in metrics
    assert "delayed_credit_horizon" in metrics
    assert "rolling_action_payoff" in metrics


def test_evaluation_module_consumes_runtime_chain_actively():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="ship a correct answer quickly",
            track=Track.WORLD,
            stratum=MemoryStratum.DURABLE,
            strength=0.75,
        ),
        timestamp_ms=10,
    )
    store.write(
        MemoryWriteRequest(
            content="keep trust and warmth stable",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.7,
        ),
        timestamp_ms=11,
    )

    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="eval-model",
            feature_surface=(FeatureSignal(name="response_context", values=(0.5,), source="adapter"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    memory = MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE)
    temporal = TemporalModule(policy=FullLearnedTemporalPolicy(), wiring_level=WiringLevel.ACTIVE)
    dual_track = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    evaluation = EvaluationModule(session_id="s1", wave_id="w1", wiring_level=WiringLevel.ACTIVE)

    result = asyncio.run(
        propagate(
            [substrate, memory, temporal, dual_track, evaluation],
            session_id="s1",
            wave_id="w1",
        )
    )

    assert "evaluation" in result
    evaluation_snapshot = result["evaluation"]
    assert evaluation_snapshot.value.turn_scores
    assert evaluation_snapshot.value.session_scores
    metric_names = {score.metric_name for score in evaluation_snapshot.value.turn_scores}
    assert "task_pressure" in metric_names
    assert "support_presence" in metric_names


def test_evaluation_backbone_uses_substrate_semantic_signals():
    backbone = EvaluationBackbone()
    substrate_snapshot = SubstrateSnapshot(
        model_id="semantic-substrate",
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(),
        feature_surface=(
            FeatureSignal(name="semantic_task_pull", values=(0.92,), source="test"),
            FeatureSignal(name="semantic_support_pull", values=(0.18,), source="test"),
            FeatureSignal(name="semantic_repair_pull", values=(0.10,), source="test"),
            FeatureSignal(name="semantic_exploration_pull", values=(0.22,), source="test"),
        ),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="task-dominant substrate",
    )

    snapshot = asyncio.run(
        EvaluationModule(backbone=backbone, wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=10,
            substrate_snapshot=substrate_snapshot,
        )
    )
    metrics = {score.metric_name: score.value for score in snapshot.value.turn_scores}

    assert metrics["task_pressure"] > metrics["support_presence"]


def test_evaluation_backbone_uses_directive_signal_to_strengthen_task_pressure():
    backbone = EvaluationBackbone()
    substrate_snapshot = SubstrateSnapshot(
        model_id="directive-substrate",
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(),
        feature_surface=(
            FeatureSignal(name="semantic_task_pull", values=(0.66,), source="test"),
            FeatureSignal(name="semantic_support_pull", values=(0.50,), source="test"),
            FeatureSignal(name="semantic_repair_pull", values=(0.20,), source="test"),
            FeatureSignal(name="semantic_exploration_pull", values=(0.15,), source="test"),
            FeatureSignal(name="semantic_directive_pull", values=(0.78,), source="test"),
        ),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="directive task substrate",
    )

    snapshot = asyncio.run(
        EvaluationModule(backbone=backbone, wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=10,
            substrate_snapshot=substrate_snapshot,
        )
    )
    metrics = {score.metric_name: score.value for score in snapshot.value.turn_scores}

    assert metrics["task_pressure"] > metrics["support_presence"]
    assert metrics["task_pressure"] > 0.35


def test_evaluation_backbone_surfaces_fallback_reliance():
    backbone = EvaluationBackbone()
    substrate_snapshot = SubstrateSnapshot(
        model_id="fallback-substrate",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(),
        feature_surface=(
            FeatureSignal(name="fallback_active", values=(1.0,), source="test"),
        ),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="builtin fallback active",
    )

    snapshot = asyncio.run(
        EvaluationModule(backbone=backbone, wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s-fallback",
            wave_id="w1",
            timestamp_ms=10,
            substrate_snapshot=substrate_snapshot,
        )
    )
    metrics = {score.metric_name: score.value for score in snapshot.value.turn_scores}

    assert metrics["fallback_reliance"] == 1.0
    assert metrics["contract_integrity"] == 1.0
    assert "MEDIUM: substrate fallback is active" in snapshot.value.alerts
    assert "HIGH: contract integrity below threshold" not in snapshot.value.alerts


def test_evaluation_backbone_records_metacontroller_evidence():
    backbone = EvaluationBackbone()
    runtime_state = MetacontrollerRuntimeState(
        mode="full-learned",
        temporal_parameters=LearnedLiteTemporalPolicy().export_parameters(),
        track_parameters=(("world", (0.7, 0.2, 0.1)), ("self", (0.2, 0.7, 0.1)), ("shared", (0.4, 0.4, 0.2))),
        encoder_weights=((0.7, 0.2, 0.1), (0.25, 0.55, 0.2), (0.15, 0.25, 0.6)),
        switch_weights=(0.45, 0.35, 0.2),
        decoder_matrix=((0.8, 0.15, 0.05), (0.2, 0.65, 0.15), (0.25, 0.25, 0.5)),
        persistence=0.65,
        learning_rate=0.08,
        clip_epsilon=0.2,
        update_steps=(("world", 1), ("self", 1), ("shared", 0)),
        latent_mean=(0.35, 0.55, 0.45),
        latent_scale=(0.12, 0.18, 0.09),
        decoder_control=(0.40, 0.58, 0.47),
        latest_switch_gate=0.30,
        sequence_length=5,
        latest_ssl_loss=0.10,
        latest_ssl_kl_loss=0.03,
        active_label="task_controller",
        posterior_mean=(0.35, 0.55, 0.45),
        posterior_std=(0.12, 0.18, 0.09),
        z_tilde=(0.40, 0.60, 0.52),
        posterior_hidden_state=(0.31, 0.48, 0.44),
        posterior_drift=0.14,
        beta_binary=1,
        switch_sparsity=0.70,
        binary_switch_rate=0.60,
        mean_persistence_window=2.0,
        decoder_applied_control=(0.42, 0.57, 0.50),
        policy_replacement_score=0.55,
        action_family_version=4,
        action_family_summaries=(
            ActionFamilyPublicSummary(
                family_id="discovered_family_0",
                dominant_axis="world",
                support=7,
                stability=0.81,
                switch_bias=0.62,
                mean_posterior_drift=0.12,
                mean_persistence_window=0.74,
                reuse_streak=5,
                stagnation_pressure=0.12,
                monopoly_pressure=0.78,
                competition_score=0.46,
                summary="competitive dominant family",
            ),
            ActionFamilyPublicSummary(
                family_id="discovered_family_1",
                dominant_axis="self",
                support=1,
                stability=0.42,
                switch_bias=0.28,
                mean_posterior_drift=0.19,
                mean_persistence_window=0.20,
                reuse_streak=0,
                stagnation_pressure=0.68,
                monopoly_pressure=0.10,
                competition_score=0.24,
                summary="idle challenger family",
            ),
        ),
        active_family_summary=ActionFamilyPublicSummary(
            family_id="discovered_family_0",
            dominant_axis="world",
            support=7,
            stability=0.81,
            switch_bias=0.62,
            mean_posterior_drift=0.12,
            mean_persistence_window=0.74,
            reuse_streak=5,
            stagnation_pressure=0.12,
            monopoly_pressure=0.78,
            competition_score=0.46,
            summary="competitive dominant family",
        ),
        active_family_competition_score=0.46,
        action_family_monopoly_pressure=0.78,
        action_family_turnover_health=0.34,
        description="Metacontroller runtime state mode=full-learned.",
    )

    scores = backbone.record_metacontroller_evidence(
        session_id="s1",
        wave_id="w3",
        timestamp_ms=30,
        metacontroller_state=runtime_state,
        policy_objective=0.2,
        rollback_reasons=(),
    )
    report = backbone.build_session_report(session_id="s1", timestamp_ms=31)

    assert len(scores) == 15
    assert any(record.metric_name == "adaptive_stability" for record in backbone.records)
    assert any(record.metric_name == "posterior_stability" for record in backbone.records)
    assert any(record.metric_name == "policy_replacement_quality" for record in backbone.records)
    assert any(record.metric_name == "family_outcome_divergence" for record in backbone.records)
    assert any(record.metric_name == "action_family_reuse" for record in backbone.records)
    assert any(record.metric_name == "action_family_stability" for record in backbone.records)
    assert any(record.metric_name == "action_family_diversity" for record in backbone.records)
    assert any(record.metric_name == "action_family_competition_score" for record in backbone.records)
    assert any(record.metric_name == "action_family_monopoly_pressure" for record in backbone.records)
    assert any(record.metric_name == "action_family_turnover_health" for record in backbone.records)
    assert any(record.metric_name == "action_family_collapse_risk" for record in backbone.records)
    assert any(family == "abstraction" for family, _ in report.scores_by_family)


def test_evaluation_backbone_raises_action_family_collapse_alerts():
    backbone = EvaluationBackbone()
    snapshot = backbone.record_external_scores(
        session_id="collapse-s1",
        wave_id="collapse-w1",
        timestamp_ms=10,
        base_snapshot=EvaluationSnapshot(turn_scores=(), session_scores=(), alerts=(), description="base"),
        scores=(
            EvaluationScore(
                family="abstraction",
                metric_name="action_family_monopoly_pressure",
                value=0.79,
                confidence=0.6,
                evidence="dominant family repeated too often",
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="action_family_collapse_risk",
                value=0.74,
                confidence=0.6,
                evidence="collapse risk elevated",
            ),
        ),
        description_suffix="added collapse metrics",
    )

    assert "MEDIUM: action-family monopoly pressure is elevated" in snapshot.alerts
    assert "HIGH: action-family collapse risk is elevated" in snapshot.alerts


def test_evaluation_backbone_runs_replay_suite_gate():
    backbone = EvaluationBackbone()
    substrate_snapshot = SubstrateSnapshot(
        model_id="replay-substrate",
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(),
        feature_surface=(
            FeatureSignal(name="semantic_task_pull", values=(0.82,), source="test"),
            FeatureSignal(name="semantic_support_pull", values=(0.64,), source="test"),
            FeatureSignal(name="fallback_active", values=(0.0,), source="test"),
        ),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="balanced replay substrate",
    )

    result = backbone.run_replay_suite(
        suite_name="widening-gate",
        timestamp_ms=100,
        cases=(
            EvaluationReplayCase(
                case_id="balanced-case",
                session_id="replay-s1",
                wave_id="replay-w1",
                substrate_snapshot=substrate_snapshot,
                memory_snapshot=None,
                dual_track_snapshot=None,
                metric_floors=(("contract_integrity", 0.95),),
                max_alert_count=0,
            ),
        ),
    )

    assert result.passed is True
    assert result.case_results[0].passed is True


def test_evaluation_backbone_can_judge_evolution_candidate():
    backbone = EvaluationBackbone()
    benchmark = backbone.run_default_evolution_benchmark(timestamp_ms=100)
    substrate_snapshot = SubstrateSnapshot(
        model_id="judge-substrate",
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(),
        feature_surface=(
            FeatureSignal(name="semantic_task_pull", values=(0.82,), source="test"),
            FeatureSignal(name="semantic_support_pull", values=(0.28,), source="test"),
            FeatureSignal(name="fallback_active", values=(0.0,), source="test"),
        ),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="judge substrate",
    )
    for turn in range(4):
        backbone.evaluate_turn(
            session_id="judge-session",
            wave_id=f"wave-{turn + 1}",
            timestamp_ms=200 + turn,
            substrate_snapshot=substrate_snapshot,
            memory_snapshot=None,
            dual_track_snapshot=None,
        )
    report = backbone.build_session_report(session_id="judge-session", timestamp_ms=300)

    judgement = backbone.judge_evolution_candidate(
        replay_suite_result=benchmark,
        session_report=report,
    )

    assert judgement.decision in {EvolutionDecision.PROMOTE, EvolutionDecision.HOLD}
    assert judgement.replay_passed is True
    assert judgement.reasons
    assert judgement.category


def test_cross_session_benchmark_computes_growth_verdict():
    from uuid import uuid4

    def _make_report(session_id: str, learning_trend: float, relationship_trend: float) -> EvaluationReport:
        return EvaluationReport(
            report_id=str(uuid4()),
            report_type="session",
            timestamp_ms=1,
            session_ids=(session_id,),
            scores_by_family=(),
            alerts=(),
            trends=(
                ("relationship", "relationship_continuity", relationship_trend),
                ("learning", "learning_quality", learning_trend),
                ("abstraction", "abstraction_reuse", 0.0),
            ),
            recommendations=(),
            description=f"session {session_id}",
        )

    backbone = EvaluationBackbone()
    reports = (
        _make_report("s1", 0.01, 0.00),
        _make_report("s2", 0.02, 0.01),
        _make_report("s3", 0.03, 0.02),
        _make_report("s4", 0.04, 0.03),
    )
    suite = CrossSessionBenchmarkSuite(session_reports=reports, comparison_windows=(1, 3))
    growth = backbone.run_cross_session_benchmark(suite=suite)

    assert growth.verdict in ("growing", "stable", "regressing")
    assert growth.window_trends
    assert growth.description


def test_cross_session_benchmark_detects_regression():
    from uuid import uuid4

    def _make_report(session_id: str, learning_trend: float, relationship_trend: float) -> EvaluationReport:
        return EvaluationReport(
            report_id=str(uuid4()),
            report_type="session",
            timestamp_ms=1,
            session_ids=(session_id,),
            scores_by_family=(),
            alerts=(),
            trends=(
                ("relationship", "relationship_continuity", relationship_trend),
                ("learning", "learning_quality", learning_trend),
                ("abstraction", "abstraction_reuse", 0.0),
            ),
            recommendations=(),
            description=f"session {session_id}",
        )

    backbone = EvaluationBackbone()
    reports = (
        _make_report("s1", 0.05, 0.04),
        _make_report("s2", 0.02, 0.02),
        _make_report("s3", -0.02, -0.01),
        _make_report("s4", -0.04, -0.03),
    )
    suite = CrossSessionBenchmarkSuite(session_reports=reports, comparison_windows=(1, 3))
    growth = backbone.run_cross_session_benchmark(suite=suite)

    assert growth.verdict == "regressing"


def test_cross_session_regression_triggers_judge_rollback():
    from uuid import uuid4

    backbone = EvaluationBackbone()
    substrate = _feature_surface_snapshot(
        model_id="cross-judge",
        task_pull=0.6, support_pull=0.4, repair_pull=0.3,
        exploration_pull=0.3, directive_pull=0.4,
    )
    benchmark = backbone.run_default_evolution_benchmark(timestamp_ms=100)
    for turn in range(4):
        backbone.evaluate_turn(
            session_id="judge-s", wave_id=f"w-{turn}",
            timestamp_ms=200 + turn,
            substrate_snapshot=substrate,
            memory_snapshot=None, dual_track_snapshot=None,
        )
    report = backbone.build_session_report(session_id="judge-s", timestamp_ms=300)

    from volvence_zero.evaluation import CrossSessionGrowthReport
    regression_report = CrossSessionGrowthReport(
        window_trends=((1, (("learning_quality", -0.05), ("relationship_continuity", -0.03))),),
        family_persistence=0.5,
        regime_effectiveness_delta=-0.03,
        verdict="regressing",
        description="regression",
    )
    judgement = backbone.judge_evolution_candidate(
        replay_suite_result=benchmark,
        session_report=report,
        cross_session_report=regression_report,
    )
    assert judgement.decision == EvolutionDecision.ROLLBACK
    assert "cross-session-regression" in judgement.reasons


def test_longitudinal_report_from_session_sequence():
    from volvence_zero.evaluation import (
        CrossSessionBenchmarkSuite,
        EvaluationBackbone,
        LongitudinalReport,
    )

    backbone = EvaluationBackbone()
    reports = []
    for i in range(5):
        report = backbone.build_session_report(
            session_id=f"session-{i}",
            timestamp_ms=i * 1000,
        )
        reports.append(report)

    suite = CrossSessionBenchmarkSuite(session_reports=tuple(reports))
    longitudinal = backbone.build_longitudinal_report(suite=suite)

    assert isinstance(longitudinal, LongitudinalReport)
    assert longitudinal.cross_session is not None
    assert longitudinal.verdict in ("growing", "stable", "regressing", "insufficient-data")
    assert len(longitudinal.dimension_trends) > 0
    assert longitudinal.description
