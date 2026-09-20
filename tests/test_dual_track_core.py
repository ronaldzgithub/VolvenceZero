from __future__ import annotations

import asyncio
from dataclasses import dataclass

from volvence_zero.dual_track import (
    DualTrackLearnedGateShadow,
    DualTrackModule,
    derive_cross_track_tension,
    derive_learned_gate_shadow,
)
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.runtime import Snapshot, WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
    SubstrateSnapshot,
    SurfaceKind,
)
from volvence_zero.temporal import ControllerState, TemporalAbstractionSnapshot


@dataclass(frozen=True)
class _SemanticOwnerValue:
    description: str


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
    assert isinstance(snapshot.value.learned_gate_shadow, DualTrackLearnedGateShadow)
    gate = snapshot.value.learned_gate_shadow
    assert abs(gate.world_weight + gate.self_weight - 1.0) <= 1e-6


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


def test_dual_track_learned_gate_shadow_stays_report_only_and_bounded():
    module = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
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
    snapshot = asyncio.run(
        module.process_standalone(
            world_entries=(world_entry,),
            self_entries=(self_entry,),
        )
    ).value

    gate = snapshot.learned_gate_shadow
    assert isinstance(gate, DualTrackLearnedGateShadow)
    assert 0.0 <= gate.world_weight <= 1.0
    assert 0.0 <= gate.self_weight <= 1.0
    assert abs(gate.world_weight + gate.self_weight - 1.0) <= 1e-6
    assert "report-only" in gate.description
    manual = derive_learned_gate_shadow(
        world_track=snapshot.world_track,
        self_track=snapshot.self_track,
        cross_track_tension=snapshot.cross_track_tension,
    )
    assert manual == gate


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


def test_dual_track_module_can_project_shared_entries_into_both_tracks():
    memory = MemoryStore()
    shared_entry = memory.write(
        MemoryWriteRequest(
            content="stabilize the task while keeping the tone supportive",
            track=Track.SHARED,
            stratum=MemoryStratum.TRANSIENT,
            strength=0.75,
        ),
        timestamp_ms=40,
    )
    module = DualTrackModule(wiring_level=WiringLevel.ACTIVE)

    snapshot = asyncio.run(
        module.process_standalone(
            world_entries=(),
            self_entries=(),
            shared_entries=(shared_entry,),
        )
    )

    assert snapshot.value.world_track.active_goals
    assert snapshot.value.self_track.active_goals
    assert snapshot.value.world_track.tension_level > 0.0
    assert snapshot.value.self_track.tension_level > 0.0


def test_dual_track_module_uses_substrate_semantic_signals_for_track_separation():
    module = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    substrate_snapshot = SubstrateSnapshot(
        model_id="semantic-substrate",
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(),
        feature_surface=(
            FeatureSignal(name="semantic_task_pull", values=(0.88,), source="test"),
            FeatureSignal(name="semantic_support_pull", values=(0.24,), source="test"),
            FeatureSignal(name="semantic_repair_pull", values=(0.18,), source="test"),
            FeatureSignal(name="semantic_exploration_pull", values=(0.33,), source="test"),
        ),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="semantic substrate snapshot",
    )

    snapshot = asyncio.run(
        module.process_standalone(
            world_entries=(),
            self_entries=(),
            shared_entries=(),
            substrate_snapshot=substrate_snapshot,
        )
    )

    assert snapshot.value.world_track.controller_code[0] > snapshot.value.self_track.controller_code[0]
    assert "substrate:task-focused" in snapshot.value.world_track.active_goals


def test_dual_track_prefers_semantic_owner_descriptions_over_shared_projection():
    memory = MemoryStore()
    shared_entry = memory.write(
        MemoryWriteRequest(
            content="keep things steady while choosing the next concrete step",
            track=Track.SHARED,
            stratum=MemoryStratum.TRANSIENT,
            strength=0.75,
        ),
        timestamp_ms=50,
    )
    semantic_snapshots = {
        "plan_intent": Snapshot(
            slot_name="plan_intent",
            owner="PlanIntentModule",
            version=1,
            timestamp_ms=1,
            value=_SemanticOwnerValue(description="owner-published plan state"),
        ),
        "relationship_state": Snapshot(
            slot_name="relationship_state",
            owner="RelationshipStateModule",
            version=1,
            timestamp_ms=1,
            value=_SemanticOwnerValue(description="owner-published relationship state"),
        ),
    }
    module = DualTrackModule(wiring_level=WiringLevel.ACTIVE)

    snapshot = asyncio.run(
        module.process_standalone(
            world_entries=(),
            self_entries=(),
            shared_entries=(shared_entry,),
            semantic_snapshots=semantic_snapshots,
        )
    )

    assert snapshot.value.world_track.active_goals[0] == "plan_intent:owner-published plan state"
    assert snapshot.value.self_track.active_goals[0] == "relationship_state:owner-published relationship state"
    assert snapshot.value.world_track.controller_source == "semantic-owner"
    assert snapshot.value.self_track.controller_source == "semantic-owner"


def test_live_cognition_semantics_are_async_and_yield_to_heartbeat():
    from volvence_zero.apprenticeship import (
        ApprenticeshipAlignmentModule,
        build_intent_constraint,
    )
    from volvence_zero.evaluation import EvaluationBackbone, EvaluationModule
    from volvence_zero.semantic_embedding import (
        reset_semantic_embedding_backend,
        set_semantic_embedding_backend,
    )

    class _AsyncOnlyBackend:
        def __init__(self) -> None:
            self.async_calls = 0

        def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
            raise AssertionError("live cognition called synchronous embed")

        async def embed_async(
            self,
            text: str,
            *,
            dim: int,
        ) -> tuple[float, ...]:
            self.async_calls += 1
            await asyncio.sleep(0)
            return tuple(
                1.0 if index == 0 else 0.0 for index in range(dim)
            )

    async def exercise_live_callers() -> tuple[int, int]:
        backend = _AsyncOnlyBackend()
        set_semantic_embedding_backend(backend, owner="cognition-live-test")
        heartbeat_ticks = 0
        stop_heartbeat = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal heartbeat_ticks
            while not stop_heartbeat.is_set():
                heartbeat_ticks += 1
                await asyncio.sleep(0)

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            memory = MemoryStore()
            shared_entry = memory.write(
                MemoryWriteRequest(
                    content="stabilize the task while preserving trust",
                    track=Track.SHARED,
                    stratum=MemoryStratum.TRANSIENT,
                    strength=0.8,
                ),
                timestamp_ms=1,
            )
            await DualTrackModule(
                wiring_level=WiringLevel.ACTIVE
            ).process_standalone(
                world_entries=(),
                self_entries=(),
                shared_entries=(shared_entry,),
            )
            backbone = EvaluationBackbone()
            await EvaluationModule(
                backbone=backbone,
                wiring_level=WiringLevel.ACTIVE,
            ).process_standalone(
                session_id="async-live",
                wave_id="turn-1",
                timestamp_ms=2,
            )
            await backbone.run_default_evolution_benchmark_async(timestamp_ms=3)
            await ApprenticeshipAlignmentModule(
                wiring_level=WiringLevel.ACTIVE,
                apprenticeship=True,
            ).process_standalone(
                apprenticeship=True,
                constraints=(
                    build_intent_constraint(
                        constraint_id="guidance-1",
                        statement="keep the response grounded",
                        target_key="grounded response",
                        confidence=0.8,
                        source_turn=1,
                    ),
                ),
                turn_index=1,
            )
        finally:
            stop_heartbeat.set()
            await heartbeat_task
            reset_semantic_embedding_backend()
        return backend.async_calls, heartbeat_ticks

    async_calls, heartbeat_ticks = asyncio.run(exercise_live_callers())
    assert async_calls > 0
    assert heartbeat_ticks > 1


def test_non_apprenticeship_returns_before_cognition_embedding():
    from companion_standard.semantic_state import (
        BeliefAssumptionSnapshot,
        SemanticRecord,
    )

    from volvence_zero.apprenticeship import (
        ApprenticeshipAlignmentModule,
        VersionSpaceStatus,
    )
    from volvence_zero.semantic_embedding import (
        reset_semantic_embedding_backend,
        set_semantic_embedding_backend,
    )

    class _NoEmbeddingBackend:
        def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
            raise AssertionError("non-apprenticeship turn called sync embed")

        async def embed_async(
            self,
            text: str,
            *,
            dim: int,
        ) -> tuple[float, ...]:
            raise AssertionError("non-apprenticeship turn called async embed")

    belief = BeliefAssumptionSnapshot(
        beliefs=(
            SemanticRecord(
                record_id="belief-1",
                summary="known fact",
                detail="this would require embedding if collected",
                confidence=0.8,
                status="active",
                source_turn=1,
                evidence="test",
            ),
        ),
        assumptions=(),
        verification_needs=(),
        contradiction_refs=(),
        mean_confidence=0.8,
        control_signal=0.0,
        description="test belief state",
    )
    set_semantic_embedding_backend(_NoEmbeddingBackend(), owner="idle-test")
    try:
        snapshot = asyncio.run(
            ApprenticeshipAlignmentModule(
                wiring_level=WiringLevel.ACTIVE,
                apprenticeship=False,
            ).process_standalone(
                apprenticeship=False,
                belief_assumption=belief,
                guidance_text="ignored on a normal turn",
                turn_index=1,
            )
        )
    finally:
        reset_semantic_embedding_backend()

    assert snapshot.value.version_space_status == VersionSpaceStatus.IDLE.value
    assert snapshot.value.active_constraint_count == 0
