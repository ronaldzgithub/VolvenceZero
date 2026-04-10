from __future__ import annotations

import asyncio

from volvence_zero.credit import (
    CreditModule,
    GateDecision,
    ModificationGate,
    ModificationProposal,
    SelfModificationRecord,
)
from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import EvaluationModule, EvaluationScore, EvaluationSnapshot
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.prediction import PredictionError, PredictionErrorSnapshot, PredictedOutcome, ActualOutcome
from volvence_zero.regime import DelayedOutcomeAttribution, DelayedOutcomePayoff, RegimeModule
from volvence_zero.reflection import ReflectionEngine, ReflectionModule, WritebackMode
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
)


def test_reflection_module_builds_proposal_first_snapshot():
    reflection = ReflectionModule(wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(reflection.process_standalone(timestamp_ms=10))

    assert snapshot.value.writeback_mode == WritebackMode.PROPOSAL_ONLY.value
    assert snapshot.value.review_required is True
    assert snapshot.value.description.startswith("Reflection generated")


def test_reflection_module_consolidates_memory_and_policy_from_inputs():
    store = MemoryStore()
    memory_snapshot = asyncio.run(
        MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE).process_standalone(
            user_text="remember that the user values warmth and clarity",
            timestamp_ms=10,
        )
    ).value
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(world_entries=(), self_entries=())
    ).value
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=11,
        )
    ).value
    credit_snapshot = asyncio.run(
        CreditModule(
            wiring_level=WiringLevel.ACTIVE,
            pending_proposals=(
                ModificationProposal(
                    target="retrieval_weight",
                    desired_gate=ModificationGate.ONLINE,
                    old_value_hash="old",
                    new_value_hash="new",
                    justification="use audit trail in reflection",
                ),
            ),
        ).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=12,
        )
    ).value

    reflection_snapshot = asyncio.run(
        ReflectionModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            credit_snapshot=credit_snapshot,
            timestamp_ms=13,
        )
    ).value

    assert reflection_snapshot.memory_consolidation.promoted_entries or reflection_snapshot.lessons_extracted
    assert reflection_snapshot.policy_consolidation.controller_updates or reflection_snapshot.policy_consolidation.strategy_priors_updated
    assert reflection_snapshot.consolidation_score.confidence >= 0.0
    assert reflection_snapshot.lessons_extracted
    assert reflection_snapshot.tensions_identified


def test_reflection_consumes_prediction_error_snapshot():
    reflection = ReflectionModule(wiring_level=WiringLevel.ACTIVE)
    prediction_error_snapshot = PredictionErrorSnapshot(
        evaluated_prediction=PredictedOutcome(
            source_turn_index=1,
            target_turn_index=2,
            predicted_task_progress=0.8,
            predicted_relationship_delta=0.8,
            predicted_regime_stability=0.7,
            predicted_action_payoff=0.6,
            confidence=0.8,
            description="pred",
        ),
        actual_outcome=ActualOutcome(
            observed_turn_index=2,
            task_progress=0.2,
            relationship_delta=0.3,
            regime_stability=0.4,
            action_payoff=0.1,
            description="actual",
        ),
        next_prediction=PredictedOutcome(
            source_turn_index=2,
            target_turn_index=3,
            predicted_task_progress=0.5,
            predicted_relationship_delta=0.5,
            predicted_regime_stability=0.5,
            predicted_action_payoff=0.5,
            confidence=0.5,
            description="next",
        ),
        error=PredictionError(
            task_error=-0.6,
            relationship_error=-0.5,
            regime_error=-0.3,
            action_error=-0.5,
            magnitude=1.9,
            signed_reward=-0.475,
            description="large negative errors",
        ),
        turn_index=2,
        bootstrap=False,
        description="pe chain",
    )
    snapshot = asyncio.run(
        reflection.process_standalone(
            prediction_error_snapshot=prediction_error_snapshot,
            timestamp_ms=20,
        )
    ).value
    assert snapshot.primary_prediction_error_dimension == "task"
    assert snapshot.error_driven_lesson_count >= 1
    assert snapshot.repeated_prediction_failures
    assert any("prediction_error" in tension for tension in snapshot.tensions_identified)
    assert "prediction_error=task" in snapshot.interaction_trace_summary
    assert snapshot.consolidation_score.description


def test_reflection_ignores_runtime_fallback_caution_as_relationship_tension():
    reflection = ReflectionModule(wiring_level=WiringLevel.ACTIVE)
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s-fallback",
            wave_id="w-fallback",
            timestamp_ms=10,
        )
    ).value
    evaluation_snapshot = type(evaluation_snapshot)(
        turn_scores=evaluation_snapshot.turn_scores,
        session_scores=evaluation_snapshot.session_scores,
        alerts=("MEDIUM: substrate fallback is active", "HIGH: contract integrity below threshold"),
        description=evaluation_snapshot.description,
    )

    snapshot = asyncio.run(
        reflection.process_standalone(
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=11,
        )
    ).value

    assert "MEDIUM: substrate fallback is active" not in snapshot.tensions_identified
    assert "HIGH: contract integrity below threshold" not in snapshot.tensions_identified


def test_reflection_module_runs_in_shadow_chain():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="deliver useful task help",
            track=Track.WORLD,
            stratum=MemoryStratum.EPISODIC,
            strength=0.8,
        ),
        timestamp_ms=10,
    )
    store.write(
        MemoryWriteRequest(
            content="stay emotionally attuned",
            track=Track.SELF,
            stratum=MemoryStratum.EPISODIC,
            strength=0.75,
        ),
        timestamp_ms=11,
    )

    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="reflection-model",
            feature_surface=(FeatureSignal(name="reflection_context", values=(0.4,), source="adapter"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    memory = MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE)
    dual_track = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    evaluation = EvaluationModule(session_id="s1", wave_id="w1", wiring_level=WiringLevel.ACTIVE)
    credit = CreditModule(
        wiring_level=WiringLevel.ACTIVE,
        pending_proposals=(
            ModificationProposal(
                target="strategy_prior",
                desired_gate=ModificationGate.ONLINE,
                old_value_hash="old-prior",
                new_value_hash="new-prior",
                justification="reflection should see gate audit",
            ),
        )
    )
    reflection = ReflectionModule(wiring_level=WiringLevel.SHADOW)
    shadow_snapshots: dict[str, object] = {}

    result = asyncio.run(
        propagate(
            [substrate, memory, dual_track, evaluation, credit, reflection],
            session_id="s1",
            wave_id="w1",
            shadow_snapshots=shadow_snapshots,
        )
    )

    assert "credit" in result
    assert "reflection" not in result
    reflection_snapshot = shadow_snapshots["reflection"]
    assert reflection_snapshot.value.writeback_mode == WritebackMode.PROPOSAL_ONLY.value
    assert reflection_snapshot.value.review_required is True
    assert reflection_snapshot.value.consolidation_score.description


def test_reflection_apply_path_supports_checkpoint_and_rollback():
    store = MemoryStore()
    regime = RegimeModule(wiring_level=WiringLevel.ACTIVE)
    initial_snapshot = asyncio.run(
        MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE).process_standalone(
            user_text="remember this durable insight",
            timestamp_ms=40,
        )
    ).value
    regime_snapshot = asyncio.run(regime.process_standalone()).value
    reflection_snapshot = asyncio.run(
        ReflectionModule(
            engine=ReflectionEngine(writeback_mode=WritebackMode.APPLY),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone(
            memory_snapshot=initial_snapshot,
            regime_snapshot=regime_snapshot,
            timestamp_ms=41,
        )
    ).value
    engine = ReflectionEngine(writeback_mode=WritebackMode.APPLY)
    writeback = engine.apply(
        memory_store=store,
        reflection_snapshot=reflection_snapshot,
        credit_snapshot=None,
        regime_module=regime,
        checkpoint_id="rollback-checkpoint",
    )

    assert writeback.applied_operations
    assert writeback.checkpoint is not None
    assert any(operation.startswith("promotion-threshold:") for operation in writeback.applied_operations)

    engine.rollback(memory_store=store, checkpoint=writeback.checkpoint, regime_module=regime)
    restored_snapshot = store.snapshot(retrieved_entries=())
    assert restored_snapshot.description.startswith("Memory store")


def test_reflection_consumes_metacontroller_gate_audit_evidence():
    store = MemoryStore()
    memory_snapshot = asyncio.run(
        MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE).process_standalone(
            user_text="remember careful controller adjustments",
            timestamp_ms=70,
        )
    ).value
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(world_entries=(), self_entries=())
    ).value
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s2",
            wave_id="w2",
            timestamp_ms=71,
        )
    ).value
    base_credit_snapshot = asyncio.run(
        CreditModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=72,
        )
    ).value
    credit_snapshot = type(base_credit_snapshot)(
        recent_credits=base_credit_snapshot.recent_credits,
        recent_modifications=base_credit_snapshot.recent_modifications
        + (
            SelfModificationRecord(
                target="metacontroller.runtime_adaptation",
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.BLOCK,
                old_value_hash="old",
                new_value_hash="new",
                justification="BLOCKED runtime metacontroller adaptation after drift",
                timestamp_ms=73,
                is_reversible=True,
            ),
        ),
        cumulative_credit_by_level=base_credit_snapshot.cumulative_credit_by_level,
        description=base_credit_snapshot.description,
    )

    reflection_snapshot = asyncio.run(
        ReflectionModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            credit_snapshot=credit_snapshot,
            timestamp_ms=74,
        )
    ).value

    assert "pause_metacontroller_writeback_after_runtime_guard" in reflection_snapshot.policy_consolidation.controller_updates
    assert "respect_metacontroller_runtime_guard" in reflection_snapshot.lessons_extracted
    assert reflection_snapshot.consolidation_score.threshold_delta >= -0.05


def test_reflection_generates_durable_identity_entries_from_delayed_regime_outcomes():
    store = MemoryStore()
    memory_snapshot = asyncio.run(
        MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE).process_standalone(
            user_text="remember that steady relational continuity matters",
            timestamp_ms=90,
        )
    ).value
    regime_snapshot = asyncio.run(RegimeModule(wiring_level=WiringLevel.ACTIVE).process_standalone()).value
    regime_snapshot = type(regime_snapshot)(
        active_regime=regime_snapshot.active_regime,
        previous_regime=regime_snapshot.previous_regime,
        switch_reason=regime_snapshot.switch_reason,
        candidate_regimes=regime_snapshot.candidate_regimes,
        turns_in_current_regime=regime_snapshot.turns_in_current_regime,
        description=regime_snapshot.description,
        delayed_outcomes=((regime_snapshot.active_regime.regime_id, 0.82),),
        identity_hints=(
            "identity:relationship:steady relational continuity matters",
            "identity:user:the user asked for calm planning support",
        ),
    )

    reflection_snapshot = asyncio.run(
        ReflectionModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            memory_snapshot=memory_snapshot,
            regime_snapshot=regime_snapshot,
            timestamp_ms=91,
        )
    ).value

    assert reflection_snapshot.memory_consolidation.new_durable_entries
    assert any("identity" in entry.tags for entry in reflection_snapshot.memory_consolidation.new_durable_entries)
    assert any(
        belief.startswith("delayed_regime:")
        for belief in reflection_snapshot.memory_consolidation.beliefs_updated
    )


def test_reflection_emits_structural_temporal_proposals_from_delayed_attribution():
    regime_snapshot = asyncio.run(RegimeModule(wiring_level=WiringLevel.ACTIVE).process_standalone()).value
    regime_snapshot = type(regime_snapshot)(
        active_regime=regime_snapshot.active_regime,
        previous_regime=regime_snapshot.previous_regime,
        switch_reason=regime_snapshot.switch_reason,
        candidate_regimes=regime_snapshot.candidate_regimes,
        turns_in_current_regime=regime_snapshot.turns_in_current_regime,
        description=regime_snapshot.description,
        delayed_outcomes=((regime_snapshot.active_regime.regime_id, 0.32),),
        delayed_attributions=(
            DelayedOutcomeAttribution(
                regime_id=regime_snapshot.active_regime.regime_id,
                outcome_score=0.32,
                source_turn_index=2,
                source_wave_id="wave-2",
                abstract_action="discovered_family_1",
                action_family_version=3,
            ),
        ),
        identity_hints=regime_snapshot.identity_hints,
    )

    reflection_snapshot = asyncio.run(
        ReflectionModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            regime_snapshot=regime_snapshot,
            timestamp_ms=92,
        )
    ).value

    temporal_update = reflection_snapshot.policy_consolidation.temporal_prior_update
    assert temporal_update is not None
    assert temporal_update.structure_proposals
    assert "action-family-structure" in temporal_update.target_groups


def test_reflection_consumes_action_family_competition_evidence():
    regime_snapshot = asyncio.run(RegimeModule(wiring_level=WiringLevel.ACTIVE).process_standalone()).value
    regime_snapshot = type(regime_snapshot)(
        active_regime=regime_snapshot.active_regime,
        previous_regime=regime_snapshot.previous_regime,
        switch_reason=regime_snapshot.switch_reason,
        candidate_regimes=regime_snapshot.candidate_regimes,
        turns_in_current_regime=regime_snapshot.turns_in_current_regime,
        description=regime_snapshot.description,
        delayed_outcomes=((regime_snapshot.active_regime.regime_id, 0.46),),
        delayed_attributions=(
            DelayedOutcomeAttribution(
                regime_id=regime_snapshot.active_regime.regime_id,
                outcome_score=0.46,
                source_turn_index=3,
                source_wave_id="wave-3",
                abstract_action="discovered_family_0",
                action_family_version=4,
            ),
        ),
        delayed_payoffs=(
            DelayedOutcomePayoff(
                regime_id=regime_snapshot.active_regime.regime_id,
                abstract_action="discovered_family_0",
                action_family_version=4,
                sample_count=3,
                rolling_payoff=0.44,
                latest_outcome=0.46,
                last_source_wave_id="wave-3",
            ),
        ),
        identity_hints=regime_snapshot.identity_hints,
    )
    evaluation_snapshot = type(
        asyncio.run(
            EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
                session_id="s-compete",
                wave_id="w-compete",
                timestamp_ms=10,
            )
        ).value
    )(
        turn_scores=(
            EvaluationScore("abstraction", "action_family_monopoly_pressure", 0.82, 0.6, "dominant family pressure"),
            EvaluationScore("abstraction", "action_family_turnover_health", 0.22, 0.6, "low turnover"),
            EvaluationScore("abstraction", "action_family_collapse_risk", 0.78, 0.6, "collapse risk high"),
        ),
        session_scores=(),
        alerts=("HIGH: action-family collapse risk is elevated",),
        description="competition pressure snapshot",
    )

    reflection_snapshot = asyncio.run(
        ReflectionModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            evaluation_snapshot=evaluation_snapshot,
            regime_snapshot=regime_snapshot,
            timestamp_ms=93,
        )
    ).value

    temporal_update = reflection_snapshot.policy_consolidation.temporal_prior_update
    assert temporal_update is not None
    assert "reduce_action_family_monopoly" in reflection_snapshot.policy_consolidation.controller_updates
    assert "prevent_action_family_collapse" in reflection_snapshot.policy_consolidation.controller_updates
    assert "action-families" in temporal_update.target_groups


def test_reflection_builds_structural_proposal_bundle_with_evidence():
    from volvence_zero.regime.identity import (
        DelayedOutcomeAttribution,
        DelayedOutcomePayoff,
        RegimeIdentity,
        RegimeSnapshot,
    )

    regime_snapshot = RegimeSnapshot(
        active_regime=RegimeIdentity(
            regime_id="problem_solving",
            name="problem solving",
            embedding=(0.8, 0.2, 0.35),
            entry_conditions="world-track urgency",
            exit_conditions="repair needed",
            historical_effectiveness=0.6,
        ),
        previous_regime=None,
        switch_reason="hold",
        candidate_regimes=(("problem_solving", 0.7),),
        turns_in_current_regime=3,
        description="",
        delayed_attributions=(
            DelayedOutcomeAttribution(
                regime_id="problem_solving",
                outcome_score=0.3,
                source_turn_index=1,
                source_wave_id="wave-1",
                abstract_action="fam_0",
                action_family_version=2,
                resolved_turn_index=3,
            ),
        ),
        delayed_payoffs=(
            DelayedOutcomePayoff(
                regime_id="problem_solving",
                abstract_action="fam_0",
                action_family_version=2,
                sample_count=3,
                rolling_payoff=0.35,
                latest_outcome=0.3,
                last_source_wave_id="wave-1",
            ),
        ),
    )
    eval_snap = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore("abstraction", "action_family_turnover_health", 0.3, 0.7, "low"),
            EvaluationScore("relationship", "cross_track_stability", 0.6, 0.7, "ok"),
        ),
        session_scores=(
            EvaluationScore("learning", "learning_quality", 0.55, 0.6, "session"),
        ),
        alerts=(),
        description="test",
    )
    engine = ReflectionEngine(writeback_mode=WritebackMode.PROPOSAL_ONLY)
    snap = engine.reflect(
        timestamp_ms=100,
        memory_snapshot=None,
        dual_track_snapshot=None,
        evaluation_snapshot=eval_snap,
        credit_snapshot=None,
        regime_snapshot=regime_snapshot,
    )
    temporal_update = snap.policy_consolidation.temporal_prior_update
    assert temporal_update is not None
    if temporal_update.structure_proposals:
        assert temporal_update.structure_bundle is not None
        bundle = temporal_update.structure_bundle
        assert bundle.scope in ("single-family", "family-cluster", "regime-sequence")
        assert bundle.evidence_pack is not None
        assert bundle.evidence_pack.delayed_credit_summary


def test_reflection_proposal_outcome_ledger_tracks_post_application_delta():
    engine = ReflectionEngine(writeback_mode=WritebackMode.PROPOSAL_ONLY)
    from volvence_zero.regime.identity import (
        DelayedOutcomeAttribution,
        DelayedOutcomePayoff,
        RegimeIdentity,
        RegimeSnapshot,
    )

    regime_snapshot = RegimeSnapshot(
        active_regime=RegimeIdentity(
            regime_id="problem_solving", name="ps", embedding=(0.8, 0.2, 0.35),
            entry_conditions="", exit_conditions="", historical_effectiveness=0.6,
        ),
        previous_regime=None, switch_reason="hold",
        candidate_regimes=(("problem_solving", 0.7),),
        turns_in_current_regime=3, description="",
        delayed_attributions=(
            DelayedOutcomeAttribution(
                regime_id="problem_solving", outcome_score=0.3,
                source_turn_index=1, source_wave_id="wave-1",
                abstract_action="fam_0", action_family_version=2,
                resolved_turn_index=3,
            ),
        ),
        delayed_payoffs=(
            DelayedOutcomePayoff(
                regime_id="problem_solving", abstract_action="fam_0",
                action_family_version=2, sample_count=3,
                rolling_payoff=0.35, latest_outcome=0.3,
                last_source_wave_id="wave-1",
            ),
        ),
    )
    eval1 = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore("abstraction", "action_family_turnover_health", 0.3, 0.7, "low"),
            EvaluationScore("relationship", "cross_track_stability", 0.5, 0.7, "ok"),
        ),
        session_scores=(), alerts=(), description="turn1",
    )
    # First reflection generates proposals and captures pre-metrics
    engine.reflect(
        timestamp_ms=100, memory_snapshot=None, dual_track_snapshot=None,
        evaluation_snapshot=eval1, credit_snapshot=None,
        regime_snapshot=regime_snapshot,
    )
    eval2 = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore("abstraction", "action_family_turnover_health", 0.5, 0.7, "improved"),
            EvaluationScore("relationship", "cross_track_stability", 0.7, 0.7, "better"),
        ),
        session_scores=(), alerts=(), description="turn2",
    )
    # Second reflection computes outcome delta
    snap2 = engine.reflect(
        timestamp_ms=200, memory_snapshot=None, dual_track_snapshot=None,
        evaluation_snapshot=eval2, credit_snapshot=None,
        regime_snapshot=regime_snapshot,
    )
    assert snap2.proposal_success_rate >= 0.0
    if engine.proposal_outcome_ledger:
        entry = engine.proposal_outcome_ledger[-1]
        assert entry.metric_delta >= 0.0
        assert entry.success is True


def test_reflection_default_wiring_is_shadow():
    from volvence_zero.reflection import ReflectionModule
    from volvence_zero.runtime import WiringLevel

    assert ReflectionModule.default_wiring_level == WiringLevel.SHADOW


def test_structural_proposal_has_scope_and_evidence():
    from volvence_zero.reflection import TemporalStructureProposal, EvidencePack

    proposal = TemporalStructureProposal(
        proposal_type="merge",
        family_id="fam_0",
        related_family_id="fam_1",
        confidence=0.85,
        justification="high similarity",
        scope="family_cluster",
        evidence=EvidencePack(
            source_benchmark_ids=("bench_1",),
            delayed_credit_summary=(("fam_0", 0.7), ("fam_1", 0.3)),
            session_trend=(("learning_quality", 0.05),),
            confidence=0.82,
            supporting_cycles=5,
        ),
    )
    assert proposal.scope == "family_cluster"
    assert proposal.evidence is not None
    assert proposal.evidence.supporting_cycles == 5
    assert len(proposal.evidence.delayed_credit_summary) == 2

    default_proposal = TemporalStructureProposal(
        proposal_type="prune",
        family_id="fam_2",
        related_family_id=None,
        confidence=0.6,
        justification="low support",
    )
    assert default_proposal.scope == "single_family"
    assert default_proposal.evidence is None


def test_reflection_pe_driven_strategy_direction():
    from volvence_zero.prediction import PredictionError, PredictionErrorSnapshot, PredictedOutcome, ActualOutcome
    from volvence_zero.dual_track import DualTrackSnapshot, TrackState

    pe = PredictionErrorSnapshot(
        evaluated_prediction=PredictedOutcome(0, 1, 0.7, 0.7, 0.7, 0.7, 0.7, "pred"),
        actual_outcome=ActualOutcome(1, 0.2, 0.8, 0.6, 0.5, "actual"),
        next_prediction=PredictedOutcome(1, 2, 0.5, 0.5, 0.5, 0.5, 0.5, "next"),
        error=PredictionError(-0.5, 0.1, -0.1, -0.2, 0.9, -0.175, "task much worse"),
        turn_index=1,
        bootstrap=False,
        description="task worse",
    )
    reflection = ReflectionModule(wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(
        reflection.process_standalone(
            prediction_error_snapshot=pe,
            timestamp_ms=30,
        )
    ).value
    assert any(
        "world_track" in update or "task" in update.lower()
        for update in snapshot.policy_consolidation.strategy_priors_updated
    ) or any(
        "task" in lesson.lower()
        for lesson in snapshot.lessons_extracted
    )


def test_reflection_pe_driven_memory_consolidation():
    from volvence_zero.prediction import PredictionError, PredictionErrorSnapshot, PredictedOutcome, ActualOutcome

    pe = PredictionErrorSnapshot(
        evaluated_prediction=PredictedOutcome(0, 1, 0.8, 0.8, 0.7, 0.7, 0.8, "pred"),
        actual_outcome=ActualOutcome(1, 0.2, 0.3, 0.5, 0.3, "actual"),
        next_prediction=PredictedOutcome(1, 2, 0.5, 0.5, 0.5, 0.5, 0.5, "next"),
        error=PredictionError(-0.6, -0.5, -0.2, -0.4, 1.7, -0.425, "large negative errors"),
        turn_index=1,
        bootstrap=False,
        description="large negative PE",
    )
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(content="task plan A", track=Track.WORLD, stratum=MemoryStratum.EPISODIC, strength=0.8),
        timestamp_ms=1,
    )
    store.write(
        MemoryWriteRequest(content="relationship note B", track=Track.SELF, stratum=MemoryStratum.EPISODIC, strength=0.3),
        timestamp_ms=2,
    )
    memory_snapshot = asyncio.run(
        MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE).process_standalone(
            user_text="check memory consolidation with PE",
            timestamp_ms=10,
        )
    ).value
    engine = ReflectionEngine(writeback_mode=WritebackMode.PROPOSAL_ONLY)
    snapshot = engine.reflect(
        timestamp_ms=20,
        memory_snapshot=memory_snapshot,
        dual_track_snapshot=None,
        evaluation_snapshot=None,
        credit_snapshot=None,
        prediction_error_snapshot=pe,
    )
    assert snapshot.error_driven_lesson_count >= 1
    assert snapshot.primary_prediction_error_dimension is not None


def test_evaluation_backbone_docstring_labels_readout():
    from volvence_zero.evaluation import EvaluationBackbone

    assert "readout" in EvaluationBackbone.__doc__.lower() or "observability" in EvaluationBackbone.__doc__.lower()
    assert "NOT" in EvaluationBackbone.__doc__ or "not the primary" in EvaluationBackbone.__doc__.lower()
