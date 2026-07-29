from __future__ import annotations

import asyncio

import pytest

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.memory import Track
from volvence_zero.runtime import Snapshot
from volvence_zero.substrate import PlaceholderSubstrateAdapter
from volvence_zero.temporal import (
    ControllerState,
    TemporalAbstractionSnapshot,
    TemporalImplementationMode,
    TemporalPolicy,
    TemporalStep,
    TrackTemporalModule,
)


class _AlwaysSwitchingPolicy(TemporalPolicy):
    mode = TemporalImplementationMode.PLACEHOLDER

    def __init__(self, *, track: str) -> None:
        self._track = track

    def step(
        self,
        *,
        substrate_snapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        memory_snapshot=None,
        reflection_snapshot=None,
    ) -> TemporalStep:
        del substrate_snapshot, memory_snapshot, reflection_snapshot
        generation = 1 if previous_snapshot is None else 2
        return TemporalStep(
            controller_state=ControllerState(
                code=(float(generation),),
                code_dim=1,
                switch_gate=1.0,
                is_switching=True,
                steps_since_switch=0,
            ),
            active_abstract_action=f"{self._track}-action-{generation}",
            controller_params_hash=f"{self._track}-params",
            description="Deterministic switch fixture.",
            action_family_version=generation,
        )


def _snapshots(result) -> dict[str, Snapshot[object]]:
    return {
        **result.active_snapshots,
        **result.shadow_snapshots,
    }


def test_final_wiring_restores_public_track_snapshots_for_segment_closure() -> None:
    world_policy = _AlwaysSwitchingPolicy(track="world")
    self_policy = _AlwaysSwitchingPolicy(track="self")
    adapter = PlaceholderSubstrateAdapter(model_id="segment-carryover")

    first = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=adapter,
            world_temporal_policy=world_policy,
            self_temporal_policy=self_policy,
            session_id="segment-carryover",
            wave_id="wave-1",
        )
    )
    first_snapshots = _snapshots(first)
    assert first_snapshots["world_temporal"].value.closed_segments == ()
    assert first_snapshots["self_temporal"].value.closed_segments == ()

    second = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=adapter,
            world_temporal_policy=world_policy,
            self_temporal_policy=self_policy,
            upstream_snapshots=first_snapshots,
            session_id="segment-carryover",
            wave_id="wave-2",
        )
    )
    second_snapshots = _snapshots(second)

    for track_name in ("world", "self"):
        previous = first_snapshots[f"{track_name}_temporal"].value
        current = second_snapshots[f"{track_name}_temporal"].value
        assert len(current.closed_segments) == 1
        closure = current.closed_segments[0]
        assert closure.abstract_action_id == previous.active_abstract_action
        assert closure.z_t_digest == previous.controller_state.code
        assert closure.open_turn_index == 1
        assert closure.close_turn_index == 2

    aggregate = second_snapshots["temporal_abstraction"].value
    assert aggregate.closed_segments


def test_final_wiring_rejects_invalid_track_temporal_carryover() -> None:
    with pytest.raises(
        TypeError,
        match="world_temporal must publish TemporalAbstractionSnapshot",
    ):
        asyncio.run(
            run_final_wiring_turn(
                config=FinalRolloutConfig(),
                substrate_adapter=PlaceholderSubstrateAdapter(
                    model_id="invalid-segment-carryover"
                ),
                upstream_snapshots={
                    "world_temporal": Snapshot(
                        slot_name="world_temporal",
                        version=1,
                        value="invalid",
                        owner="test",
                        timestamp_ms=1,
                    )
                },
            )
        )


def test_track_temporal_standalone_counts_opening_turn_in_segment_interval() -> None:
    policy = _AlwaysSwitchingPolicy(track="world")
    module = TrackTemporalModule(
        track=Track.WORLD,
        policy=policy,
    )
    substrate_snapshot = asyncio.run(
        PlaceholderSubstrateAdapter(
            model_id="standalone-segment"
        ).capture()
    )
    first = asyncio.run(
        module.process_standalone(
            substrate_snapshot=substrate_snapshot
        )
    )
    second = asyncio.run(
        module.process_standalone(
            substrate_snapshot=substrate_snapshot
        )
    )

    closure = second.value.closed_segments[0]
    assert closure.open_turn_index == first.version
    assert closure.close_turn_index == second.version
