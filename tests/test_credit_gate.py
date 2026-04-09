from __future__ import annotations

import asyncio

from volvence_zero.credit import (
    CreditModule,
    GateDecision,
    ModificationGate,
    ModificationProposal,
    derive_abstract_action_credit_records,
    derive_delayed_attribution_credit_records,
    derive_learning_evidence_credit_records,
    derive_metacontroller_credit_records,
    derive_runtime_adaptation_audit_records,
    extend_credit_snapshot,
    evaluate_gate,
    has_blocking_writeback,
)
from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import EvaluationModule, EvaluationScore, EvaluationSnapshot
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.regime import DelayedOutcomeAttribution, DelayedOutcomePayoff, RegimeIdentity, RegimeSnapshot
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
)
from volvence_zero.temporal import LearnedLiteTemporalPolicy, MetacontrollerRuntimeState, TemporalModule


def test_gate_blocks_online_modification_when_high_alert_present():
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=10,
        )
    ).value
    evaluation_snapshot = type(evaluation_snapshot)(
        turn_scores=evaluation_snapshot.turn_scores,
        session_scores=evaluation_snapshot.session_scores,
        alerts=("HIGH: cross-track stability is degraded",),
        description=evaluation_snapshot.description,
    )

    decision = evaluate_gate(
        proposal=ModificationProposal(
            target="retrieval_weight",
            desired_gate=ModificationGate.ONLINE,
            old_value_hash="old",
            new_value_hash="new",
            justification="Tune retrieval after poor turn.",
        ),
        evaluation_snapshot=evaluation_snapshot,
    )

    assert decision is GateDecision.BLOCK


def test_credit_module_records_credits_and_modification_audit():
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(world_entries=(), self_entries=())
    ).value
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=10,
        )
    ).value
    evaluation_snapshot = EvaluationSnapshot(
        turn_scores=evaluation_snapshot.turn_scores,
        session_scores=evaluation_snapshot.session_scores,
        alerts=(),
        description=evaluation_snapshot.description,
    )

    module = CreditModule(
        wiring_level=WiringLevel.ACTIVE,
        pending_proposals=(
            ModificationProposal(
                target="strategy_prior",
                desired_gate=ModificationGate.ONLINE,
                old_value_hash="old-prior",
                new_value_hash="new-prior",
                justification="Small turn-level policy adjustment.",
            ),
        ),
    )
    snapshot = asyncio.run(
        module.process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=20,
        )
    )

    assert snapshot.value.recent_credits
    assert snapshot.value.recent_modifications
    assert snapshot.value.cumulative_credit_by_level
    assert snapshot.value.recent_modifications[-1].decision is GateDecision.ALLOW


def test_credit_module_consumes_chain_in_shadow_mode():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="answer carefully and completely",
            track=Track.WORLD,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
        ),
        timestamp_ms=10,
    )
    store.write(
        MemoryWriteRequest(
            content="keep the user feeling understood",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.7,
        ),
        timestamp_ms=11,
    )
    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="credit-model",
            feature_surface=(FeatureSignal(name="credit_context", values=(0.6,), source="adapter"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    memory = MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE)
    temporal = TemporalModule(policy=LearnedLiteTemporalPolicy(), wiring_level=WiringLevel.ACTIVE)
    dual_track = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    evaluation = EvaluationModule(session_id="s1", wave_id="w1", wiring_level=WiringLevel.ACTIVE)
    credit = CreditModule(
        pending_proposals=(
            ModificationProposal(
                target="retrieval_weight",
                desired_gate=ModificationGate.ONLINE,
                old_value_hash="old-weight",
                new_value_hash="new-weight",
                justification="Adjust retrieval after evaluation feedback.",
            ),
        ),
    )
    shadow_snapshots: dict[str, object] = {}

    result = asyncio.run(
        propagate(
            [substrate, memory, temporal, dual_track, evaluation, credit],
            session_id="s1",
            wave_id="w1",
            shadow_snapshots=shadow_snapshots,
        )
    )

    assert "evaluation" in result
    assert "credit" not in result
    credit_snapshot = shadow_snapshots["credit"]
    assert credit_snapshot.value.recent_credits
    assert credit_snapshot.value.recent_modifications


def test_has_blocking_writeback_detects_blocked_targets():
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=30,
        )
    ).value
    evaluation_snapshot = type(evaluation_snapshot)(
        turn_scores=evaluation_snapshot.turn_scores,
        session_scores=evaluation_snapshot.session_scores,
        alerts=("HIGH: writeback should pause",),
        description=evaluation_snapshot.description,
    )
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(world_entries=(), self_entries=())
    ).value
    credit_snapshot = asyncio.run(
        CreditModule(
            wiring_level=WiringLevel.ACTIVE,
            pending_proposals=(
                ModificationProposal(
                    target="memory.writeback.threshold",
                    desired_gate=ModificationGate.ONLINE,
                    old_value_hash="old",
                    new_value_hash="new",
                    justification="guard writeback",
                ),
            ),
        ).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=31,
        )
    ).value

    assert has_blocking_writeback(credit_snapshot, target_prefix="memory")


def test_credit_snapshot_can_be_extended_with_abstract_action_credit():
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(world_entries=(), self_entries=())
    ).value
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s2",
            wave_id="w2",
            timestamp_ms=50,
        )
    ).value
    temporal_snapshot = asyncio.run(
        TemporalModule(policy=LearnedLiteTemporalPolicy(), wiring_level=WiringLevel.ACTIVE).process_standalone(
            substrate_snapshot=asyncio.run(
                SubstrateModule(
                    adapter=FeatureSurfaceSubstrateAdapter(
                        model_id="temporal-credit-model",
                        feature_surface=(FeatureSignal(name="credit_signal", values=(0.5,), source="adapter"),),
                    ),
                    wiring_level=WiringLevel.ACTIVE,
                ).process_standalone()
            ).value
        )
    ).value
    base_snapshot = asyncio.run(
        CreditModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=51,
        )
    ).value

    extended = extend_credit_snapshot(
        credit_snapshot=base_snapshot,
        extra_records=derive_abstract_action_credit_records(
            temporal_snapshot=temporal_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=52,
        ),
    )

    assert any(record.level == "abstract_action" for record in extended.recent_credits)


def test_runtime_adaptation_audit_records_capture_rollback_evidence():
    records = derive_runtime_adaptation_audit_records(
        rollback_reasons=("reward-regression", "metacontroller-drift"),
        metacontroller_state_description="Metacontroller runtime state mode=learned-lite.",
        timestamp_ms=60,
        rollback_applied=True,
    )

    assert len(records) == 1
    assert records[0].decision is GateDecision.BLOCK
    assert records[0].target == "metacontroller.runtime_adaptation"
    assert "reward-regression" in records[0].justification


def test_metacontroller_credit_records_capture_kernel_evidence():
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
        latent_mean=(0.4, 0.6, 0.5),
        latent_scale=(0.1, 0.2, 0.1),
        decoder_control=(0.45, 0.55, 0.50),
        latest_switch_gate=0.35,
        sequence_length=4,
        latest_ssl_loss=0.12,
        latest_ssl_kl_loss=0.04,
        active_label="repair_controller",
        posterior_mean=(0.40, 0.60, 0.50),
        posterior_std=(0.10, 0.20, 0.10),
        z_tilde=(0.42, 0.68, 0.55),
        posterior_hidden_state=(0.36, 0.52, 0.49),
        posterior_drift=0.18,
        beta_binary=1,
        switch_sparsity=0.65,
        binary_switch_rate=0.55,
        mean_persistence_window=1.0,
        decoder_applied_control=(0.47, 0.58, 0.52),
        policy_replacement_score=0.50,
        description="Metacontroller runtime state mode=full-learned.",
    )

    records = derive_metacontroller_credit_records(
        metacontroller_state=runtime_state,
        policy_objective=0.25,
        rollback_reasons=(),
        timestamp_ms=70,
    )

    assert len(records) == 1
    assert records[0].source_event == "repair_controller"
    assert records[0].level == "abstract_action"
    assert "posterior_drift" in records[0].context
    assert "family_version=" in records[0].context


def test_learning_evidence_credit_records_include_published_metacontroller_metrics():
    evaluation_snapshot = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                family="learning",
                metric_name="joint_learning_progress",
                value=0.68,
                confidence=0.55,
                evidence="joint progress",
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="abstract_action_usefulness",
                value=0.72,
                confidence=0.6,
                evidence="published metacontroller evidence",
            ),
        ),
        session_scores=(),
        alerts=(),
        description="evaluation snapshot with kernel metrics",
    )

    records = derive_learning_evidence_credit_records(
        evaluation_snapshot=evaluation_snapshot,
        timestamp_ms=90,
    )

    assert {record.source_event for record in records} == {
        "joint_learning_progress",
        "evaluation:abstract_action_usefulness",
    }
    abstract_record = next(record for record in records if record.source_event == "evaluation:abstract_action_usefulness")
    assert abstract_record.level == "abstract_action"


def test_delayed_attribution_credit_records_capture_regime_and_action_history():
    regime_snapshot = RegimeSnapshot(
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
        delayed_outcomes=(("guided_exploration", 0.78),),
        delayed_attributions=(
            DelayedOutcomeAttribution(
                regime_id="guided_exploration",
                outcome_score=0.78,
                source_turn_index=3,
                source_wave_id="wave-3",
                abstract_action="discovered_family_2",
                action_family_version=4,
            ),
        ),
        delayed_payoffs=(
            DelayedOutcomePayoff(
                regime_id="guided_exploration",
                abstract_action="discovered_family_2",
                action_family_version=4,
                sample_count=3,
                rolling_payoff=0.74,
                latest_outcome=0.78,
                last_source_wave_id="wave-3",
            ),
        ),
    )

    records = derive_delayed_attribution_credit_records(
        regime_snapshot=regime_snapshot,
        timestamp_ms=120,
    )

    assert {record.source_event for record in records} == {
        "delayed_regime:guided_exploration",
        "delayed_action:discovered_family_2",
        "delayed_payoff:guided_exploration",
        "delayed_payoff_action:discovered_family_2",
    }
    assert all("source_wave_id=wave-3" in record.context for record in records)
