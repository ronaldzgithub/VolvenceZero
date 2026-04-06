from __future__ import annotations

import asyncio

from volvence_zero.dual_track import DualTrackModule, derive_cross_track_tension
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
)
from volvence_zero.temporal import ControllerState, TemporalAbstractionSnapshot


def test_dual_track_standalone_builds_separated_track_states():
    memory = MemoryStore()
    world_entry = memory.write(
        MemoryWriteRequest(
            content="finish the planning task",
            track=Track.WORLD,
            stratum=MemoryStratum.EPISODIC,
            strength=0.8,
            tags=("task",),
        ),
        timestamp_ms=10,
    )
    self_entry = memory.write(
        MemoryWriteRequest(
            content="maintain a calm supportive tone",
            track=Track.SELF,
            stratum=MemoryStratum.EPISODIC,
            strength=0.6,
            tags=("relationship",),
        ),
        timestamp_ms=12,
    )

    module = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(
        module.process_standalone(
            world_entries=(world_entry,),
            self_entries=(self_entry,),
        )
    )

    assert snapshot.value.world_track.track is Track.WORLD
    assert snapshot.value.self_track.track is Track.SELF
    assert "finish the planning task" in snapshot.value.world_track.active_goals
    assert "maintain a calm supportive tone" in snapshot.value.self_track.active_goals
    assert snapshot.value.cross_track_tension >= 0.0


def test_cross_track_tension_increases_when_tracks_diverge():
    module = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    low_tension = asyncio.run(
        module.process_standalone(world_entries=(), self_entries=())
    ).value.cross_track_tension

    memory = MemoryStore()
    world_entry = memory.write(
        MemoryWriteRequest(
            content="urgent deadline planning",
            track=Track.WORLD,
            stratum=MemoryStratum.TRANSIENT,
            strength=0.95,
        ),
        timestamp_ms=20,
    )
    self_entry = memory.write(
        MemoryWriteRequest(
            content="slow down and repair trust",
            track=Track.SELF,
            stratum=MemoryStratum.TRANSIENT,
            strength=0.2,
        ),
        timestamp_ms=21,
    )
    high_tension = asyncio.run(
        module.process_standalone(
            world_entries=(world_entry,),
            self_entries=(self_entry,),
        )
    ).value.cross_track_tension

    assert high_tension >= low_tension
    assert derive_cross_track_tension(
        asyncio.run(module.process_standalone(world_entries=(world_entry,), self_entries=())).value.world_track,
        asyncio.run(module.process_standalone(world_entries=(), self_entries=(self_entry,))).value.self_track,
    ) >= 0.0


def test_dual_track_module_consumes_memory_snapshot_in_shadow_mode():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="prepare a concrete answer for the user",
            track=Track.WORLD,
            stratum=MemoryStratum.DURABLE,
            strength=0.7,
        ),
        timestamp_ms=30,
    )
    store.write(
        MemoryWriteRequest(
            content="keep the interaction warm and non-intrusive",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.65,
        ),
        timestamp_ms=31,
    )
    memory_module = MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE)
    substrate_module = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="dual-track-test-model",
            feature_surface=(
                FeatureSignal(name="planning_context", values=(0.7,), source="adapter"),
            ),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    dual_track_module = DualTrackModule()
    shadow_snapshots: dict[str, object] = {}

    result = asyncio.run(
        propagate(
            [substrate_module, memory_module, dual_track_module],
            upstream={},
            shadow_snapshots=shadow_snapshots,
            session_id="s1",
            wave_id="w1",
        )
    )

    assert "substrate" in result
    assert "memory" in result
    assert "dual_track" not in result
    dual_snapshot = shadow_snapshots["dual_track"]
    assert dual_snapshot.value.world_track.track is Track.WORLD
    assert dual_snapshot.value.self_track.track is Track.SELF


def test_dual_track_module_consumes_temporal_snapshot_as_controller_evidence():
    module = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    temporal_snapshot = TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(0.9, 0.2, 0.4),
            code_dim=3,
            switch_gate=0.7,
            is_switching=True,
            steps_since_switch=1,
        ),
        active_abstract_action="task_controller",
        controller_params_hash="hash",
        description="temporal control evidence",
    )

    snapshot = asyncio.run(
        module.process_standalone(
            world_entries=(),
            self_entries=(),
            temporal_snapshot=temporal_snapshot,
        )
    )

    assert snapshot.value.world_track.controller_source == "temporal+memory"
    assert snapshot.value.self_track.controller_source == "temporal+memory"
    assert snapshot.value.world_track.abstract_action_hint == "task_controller"
    assert snapshot.value.world_track.controller_code[-1] == 0.7
