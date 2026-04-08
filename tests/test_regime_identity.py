from __future__ import annotations

import asyncio

from volvence_zero.dual_track import DualTrackModule
from volvence_zero.dual_track.core import DualTrackSnapshot, TrackState
from volvence_zero.evaluation import EvaluationModule, EvaluationScore, EvaluationSnapshot
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.regime import RegimeModule
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
)
from volvence_zero.temporal import LearnedLiteTemporalPolicy, MetacontrollerRuntimeState


def test_regime_module_produces_structured_identity():
    snapshot = asyncio.run(
        RegimeModule(wiring_level=WiringLevel.ACTIVE).process_standalone()
    )

    assert snapshot.value.active_regime.regime_id
    assert snapshot.value.active_regime.embedding
    assert isinstance(snapshot.value.candidate_regimes, tuple)
    assert snapshot.value.turns_in_current_regime >= 1


def test_regime_module_prefers_repair_when_cross_track_tension_is_high():
    memory_store = MemoryStore()
    world_entry = memory_store.write(
        MemoryWriteRequest(
            content="urgent task pressure",
            track=Track.WORLD,
            stratum=MemoryStratum.TRANSIENT,
            strength=0.95,
        ),
        timestamp_ms=10,
    )
    self_entry = memory_store.write(
        MemoryWriteRequest(
            content="repair trust before pushing harder",
            track=Track.SELF,
            stratum=MemoryStratum.TRANSIENT,
            strength=0.2,
        ),
        timestamp_ms=11,
    )
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            world_entries=(world_entry,),
            self_entries=(self_entry,),
        )
    ).value
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=12,
        )
    ).value
    evaluation_snapshot = type(evaluation_snapshot)(
        turn_scores=evaluation_snapshot.turn_scores,
        session_scores=evaluation_snapshot.session_scores,
        alerts=("HIGH: cross-track stability is degraded",),
        description=evaluation_snapshot.description,
    )

    snapshot = asyncio.run(
        RegimeModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
        )
    )

    assert snapshot.value.active_regime.regime_id == "repair_and_deescalation"


def test_regime_module_prefers_problem_solving_when_world_drive_and_task_signal_dominate():
    dual_track_snapshot = DualTrackSnapshot(
        world_track=TrackState(
            track=Track.WORLD,
            active_goals=("finish the task plan",),
            recent_credits=(),
            controller_code=(0.92, 0.45, 0.72),
            tension_level=0.82,
            controller_source="temporal-track-projected",
        ),
        self_track=TrackState(
            track=Track.SELF,
            active_goals=("stay steady",),
            recent_credits=(),
            controller_code=(0.18, 0.28, 0.72),
            tension_level=0.22,
            controller_source="temporal-track-projected",
        ),
        cross_track_tension=0.18,
        description="world-drive dominates while relationship remains stable",
    )
    evaluation_snapshot = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                family="task",
                metric_name="info_integration",
                value=0.86,
                confidence=0.8,
                evidence="high task coherence",
            ),
            EvaluationScore(
                family="interaction",
                metric_name="warmth",
                value=0.36,
                confidence=0.6,
                evidence="support present but secondary",
            ),
            EvaluationScore(
                family="relationship",
                metric_name="cross_track_stability",
                value=0.82,
                confidence=0.7,
                evidence="stable relationship frame",
            ),
        ),
        session_scores=(),
        alerts=(),
        description="task-dominant evaluation",
    )

    snapshot = asyncio.run(
        RegimeModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
        )
    )

    assert snapshot.value.active_regime.regime_id == "problem_solving"


def test_regime_module_prefers_emotional_support_when_support_signal_dominates():
    dual_track_snapshot = DualTrackSnapshot(
        world_track=TrackState(
            track=Track.WORLD,
            active_goals=("finish one small task",),
            recent_credits=(),
            controller_code=(0.18, 0.35, 0.42),
            tension_level=0.18,
            controller_source="temporal-track-projected",
        ),
        self_track=TrackState(
            track=Track.SELF,
            active_goals=("stay emotionally supported",),
            recent_credits=(),
            controller_code=(0.82, 0.62, 0.42),
            tension_level=0.88,
            controller_source="temporal-track-projected",
        ),
        cross_track_tension=0.16,
        description="support-dominant evaluation",
    )
    evaluation_snapshot = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                family="task",
                metric_name="info_integration",
                value=0.38,
                confidence=0.7,
                evidence="task is secondary",
            ),
            EvaluationScore(
                family="task",
                metric_name="task_pressure",
                value=0.24,
                confidence=0.7,
                evidence="task pressure is low",
            ),
            EvaluationScore(
                family="interaction",
                metric_name="warmth",
                value=0.82,
                confidence=0.7,
                evidence="warmth is required",
            ),
            EvaluationScore(
                family="interaction",
                metric_name="support_presence",
                value=0.91,
                confidence=0.7,
                evidence="support need dominates",
            ),
            EvaluationScore(
                family="relationship",
                metric_name="cross_track_stability",
                value=0.84,
                confidence=0.7,
                evidence="relationship frame remains stable",
            ),
        ),
        session_scores=(),
        alerts=(),
        description="support-dominant evaluation",
    )

    snapshot = asyncio.run(
        RegimeModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
        )
    )

    assert snapshot.value.active_regime.regime_id == "emotional_support"


def test_regime_module_runs_in_shadow_chain():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="provide concrete help for the user task",
            track=Track.WORLD,
            stratum=MemoryStratum.DURABLE,
            strength=0.75,
        ),
        timestamp_ms=20,
    )
    store.write(
        MemoryWriteRequest(
            content="maintain relational continuity and warmth",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.7,
        ),
        timestamp_ms=21,
    )
    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="regime-model",
            feature_surface=(FeatureSignal(name="regime_context", values=(0.5,), source="adapter"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    memory = MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE)
    dual_track = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    evaluation = EvaluationModule(session_id="s1", wave_id="w1", wiring_level=WiringLevel.ACTIVE)
    regime = RegimeModule()
    shadow_snapshots: dict[str, object] = {}

    result = asyncio.run(
        propagate(
            [substrate, memory, dual_track, evaluation, regime],
            session_id="s1",
            wave_id="w1",
            shadow_snapshots=shadow_snapshots,
        )
    )

    assert "evaluation" in result
    assert "regime" not in result
    regime_snapshot = shadow_snapshots["regime"]
    assert regime_snapshot.value.active_regime.regime_id
    assert regime_snapshot.value.candidate_regimes


def test_regime_module_applies_policy_consolidation_and_restores_checkpoint():
    module = RegimeModule(wiring_level=WiringLevel.ACTIVE)
    initial = asyncio.run(module.process_standalone()).value
    checkpoint = module.create_checkpoint(checkpoint_id="regime-1")

    applied = module.apply_policy_consolidation(
        strategy_updates=("increase_self_track_priority",),
        regime_effectiveness_updates=((initial.active_regime.regime_id, 0.9),),
    )
    updated = asyncio.run(module.process_standalone()).value

    assert applied
    assert updated.active_regime.historical_effectiveness >= initial.active_regime.historical_effectiveness

    module.restore_checkpoint(checkpoint)
    restored = asyncio.run(module.process_standalone()).value
    assert restored.active_regime.historical_effectiveness <= updated.active_regime.historical_effectiveness


def test_regime_module_emits_delayed_outcomes_one_turn_later():
    module = RegimeModule(wiring_level=WiringLevel.ACTIVE)
    first = asyncio.run(
        module.process_standalone(
            evaluation_snapshot=EvaluationSnapshot(
                turn_scores=(
                    EvaluationScore("task", "info_integration", 0.4, 0.7, "baseline"),
                    EvaluationScore("interaction", "warmth", 0.5, 0.7, "baseline"),
                    EvaluationScore("relationship", "cross_track_stability", 0.5, 0.7, "baseline"),
                ),
                session_scores=(),
                alerts=(),
                description="turn one",
            )
        )
    ).value
    second = asyncio.run(
        module.process_standalone(
            evaluation_snapshot=EvaluationSnapshot(
                turn_scores=(
                    EvaluationScore("task", "info_integration", 0.9, 0.7, "delayed improvement"),
                    EvaluationScore("interaction", "warmth", 0.8, 0.7, "delayed improvement"),
                    EvaluationScore("relationship", "cross_track_stability", 0.85, 0.7, "delayed improvement"),
                ),
                session_scores=(),
                alerts=(),
                description="turn two",
            )
        )
    ).value

    assert second.delayed_outcomes
    assert second.delayed_outcomes[0][0] == first.active_regime.regime_id
    assert second.delayed_outcomes[0][1] > 0.7


def test_regime_module_projects_identity_hints_from_memory_snapshot():
    memory_store = MemoryStore()
    relationship_entry = memory_store.write(
        MemoryWriteRequest(
            content="remember the relationship needs gentle continuity",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
        ),
        timestamp_ms=30,
    )
    shared_entry = memory_store.write(
        MemoryWriteRequest(
            content="the user asked for calm planning support",
            track=Track.SHARED,
            stratum=MemoryStratum.TRANSIENT,
            strength=0.7,
            tags=("user_input",),
        ),
        timestamp_ms=31,
    )
    memory_snapshot = memory_store.snapshot(retrieved_entries=(relationship_entry, shared_entry))

    snapshot = asyncio.run(
        RegimeModule(wiring_level=WiringLevel.ACTIVE).process_standalone(memory_snapshot=memory_snapshot)
    ).value

    assert snapshot.identity_hints
    assert any(hint.startswith("identity:relationship:") for hint in snapshot.identity_hints)


def test_regime_module_consumes_metacontroller_evidence():
    module = RegimeModule(wiring_level=WiringLevel.ACTIVE)
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
        latent_mean=(0.25, 0.70, 0.60),
        latent_scale=(0.1, 0.2, 0.1),
        decoder_control=(0.35, 0.72, 0.58),
        latest_switch_gate=0.60,
        sequence_length=4,
        latest_ssl_loss=0.15,
        latest_ssl_kl_loss=0.05,
        active_label="repair_controller",
        posterior_mean=(0.25, 0.70, 0.60),
        posterior_std=(0.10, 0.20, 0.10),
        z_tilde=(0.28, 0.75, 0.62),
        posterior_hidden_state=(0.24, 0.61, 0.58),
        posterior_drift=0.52,
        beta_binary=1,
        switch_sparsity=0.40,
        binary_switch_rate=0.62,
        mean_persistence_window=0.0,
        decoder_applied_control=(0.39, 0.70, 0.60),
        policy_replacement_score=0.58,
        description="Metacontroller runtime state mode=full-learned.",
    )

    applied = module.apply_metacontroller_evidence(
        metacontroller_state=runtime_state,
        rollback_reasons=("metacontroller-drift",),
    )

    assert "metacontroller:repair" in applied
    assert "metacontroller:sparse-switch" in applied
    assert "metacontroller:posterior-guard" in applied
    assert "metacontroller:replacement" in applied
    assert "metacontroller:guard" in applied
