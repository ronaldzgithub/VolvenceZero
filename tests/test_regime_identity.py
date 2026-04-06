from __future__ import annotations

import asyncio

from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import EvaluationModule
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.regime import RegimeModule
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
)


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
