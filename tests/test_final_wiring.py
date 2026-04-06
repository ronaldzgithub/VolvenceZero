from __future__ import annotations

import asyncio

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.memory import MemoryStore
from volvence_zero.reflection import WritebackMode
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
)


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
    assert result.temporal_runtime_state.mode == "learned-lite"


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
