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
