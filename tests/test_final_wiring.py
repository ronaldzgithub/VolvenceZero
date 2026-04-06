from __future__ import annotations

import asyncio

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.memory import MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.reflection import WritebackMode
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
)
from volvence_zero.temporal import ControllerState, TemporalAbstractionSnapshot
from volvence_zero.dual_track import DualTrackSnapshot, TrackState


def test_final_wiring_turn_builds_expected_active_and_shadow_chain():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="final-model",
                feature_surface=(FeatureSignal(name="final_context", values=(0.5,), source="adapter"),),
            ),
            session_id="s1",
            wave_id="w1",
        )
    )

    assert result.acceptance_report.passed is True
    assert "substrate" in result.active_snapshots
    assert "memory" in result.active_snapshots
    assert "dual_track" in result.active_snapshots
    assert "evaluation" in result.active_snapshots
    assert "regime" in result.active_snapshots
    assert "credit" in result.active_snapshots
    assert "reflection" in result.shadow_snapshots
    assert "temporal_abstraction" in result.shadow_snapshots
    assert result.temporal_runtime_state is not None
    assert result.temporal_runtime_state.mode == "full-learned"
    metric_names = {score.metric_name for score in result.active_snapshots["evaluation"].value.turn_scores}
    assert "retrieval_quality" in metric_names
    assert "reflection_usefulness" in metric_names


def test_final_wiring_honors_kill_switches():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(kill_switches=frozenset({"reflection", "temporal"})),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="kill-switch-model",
                feature_surface=(FeatureSignal(name="kill_switch_context", values=(0.4,), source="adapter"),),
            ),
            session_id="s1",
            wave_id="w1",
        )
    )

    assert "reflection" not in result.shadow_snapshots
    assert "temporal_abstraction" not in result.shadow_snapshots
    assert "reflection" in result.acceptance_report.disabled_slots
    assert "temporal" in result.acceptance_report.disabled_slots


def test_final_wiring_allows_active_widening_but_reports_caution():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(
                reflection=WiringLevel.ACTIVE,
                temporal=WiringLevel.ACTIVE,
            ),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="wide-model",
                feature_surface=(FeatureSignal(name="wide_context", values=(0.8,), source="adapter"),),
            ),
            session_id="s1",
            wave_id="w1",
        )
    )

    assert result.acceptance_report.passed is True
    assert "reflection" in result.active_snapshots
    assert "temporal_abstraction" in result.active_snapshots
    assert result.acceptance_report.recommendations


def test_final_wiring_can_apply_bounded_writeback_when_enabled():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(reflection=WiringLevel.ACTIVE, temporal=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="apply-model",
                feature_surface=(FeatureSignal(name="apply_context", values=(0.9,), source="adapter"),),
            ),
            memory_store=MemoryStore(),
            reflection_mode=WritebackMode.APPLY,
            session_id="s1",
            wave_id="w2",
        )
    )

    assert result.writeback_result is not None
    assert result.writeback_result.description


def test_final_wiring_can_apply_bounded_writeback_from_shadow_reflection():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="shadow-apply-model",
                feature_surface=(FeatureSignal(name="shadow_apply_context", values=(0.7,), source="adapter"),),
            ),
            memory_store=MemoryStore(),
            reflection_mode=WritebackMode.APPLY,
            session_id="s1",
            wave_id="w-shadow-apply",
        )
    )

    assert "reflection" in result.shadow_snapshots
    assert result.writeback_result is not None
    assert result.writeback_source == "shadow"


def test_final_wiring_can_seed_memory_query_from_previous_wave_snapshots():
    memory_store = MemoryStore()
    memory_store.write(
        request=MemoryWriteRequest(
            content="repair_controller maintain a calm supportive tone while planning next steps",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
            tags=("repair", "support"),
        ),
        timestamp_ms=1,
    )
    prior_temporal = Snapshot(
        slot_name="temporal_abstraction",
        owner="TemporalModule",
        version=1,
        timestamp_ms=1,
        value=TemporalAbstractionSnapshot(
            controller_state=ControllerState(
                code=(0.2, 0.8, 0.4),
                code_dim=3,
                switch_gate=0.7,
                is_switching=True,
                steps_since_switch=1,
            ),
            active_abstract_action="repair_controller",
            controller_params_hash="hash",
            description="prior temporal",
        ),
    )
    prior_dual_track = Snapshot(
        slot_name="dual_track",
        owner="DualTrackModule",
        version=1,
        timestamp_ms=1,
        value=DualTrackSnapshot(
            world_track=TrackState(
                track=Track.WORLD,
                active_goals=("planning next steps",),
                recent_credits=(),
                controller_code=(0.6, 0.4),
                tension_level=0.4,
            ),
            self_track=TrackState(
                track=Track.SELF,
                active_goals=("maintain a calm supportive tone",),
                recent_credits=(),
                controller_code=(0.3, 0.7),
                tension_level=0.6,
            ),
            cross_track_tension=0.2,
            description="prior dual track",
        ),
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="seeded-model",
                feature_surface=(FeatureSignal(name="neutral_context", values=(0.1,), source="adapter"),),
            ),
            memory_store=memory_store,
            upstream_snapshots={
                "temporal_abstraction": prior_temporal,
                "dual_track": prior_dual_track,
            },
            session_id="s1",
            wave_id="w-seeded",
        )
    )

    retrieved_contents = tuple(entry.content for entry in result.active_snapshots["memory"].value.retrieved_entries)
    assert "repair_controller maintain a calm supportive tone while planning next steps" in retrieved_contents
