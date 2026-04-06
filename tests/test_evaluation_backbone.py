from __future__ import annotations

import asyncio

from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import EvaluationBackbone, EvaluationModule
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
)
from volvence_zero.temporal import LearnedLiteTemporalPolicy, MetacontrollerRuntimeState


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
    dual_track = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    evaluation = EvaluationModule(session_id="s1", wave_id="w1", wiring_level=WiringLevel.ACTIVE)

    result = asyncio.run(
        propagate(
            [substrate, memory, dual_track, evaluation],
            session_id="s1",
            wave_id="w1",
        )
    )

    assert "evaluation" in result
    evaluation_snapshot = result["evaluation"]
    assert evaluation_snapshot.value.turn_scores
    assert evaluation_snapshot.value.session_scores


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

    assert len(scores) == 7
    assert any(record.metric_name == "adaptive_stability" for record in backbone.records)
    assert any(record.metric_name == "posterior_stability" for record in backbone.records)
    assert any(record.metric_name == "policy_replacement_quality" for record in backbone.records)
    assert any(family == "abstraction" for family, _ in report.scores_by_family)
