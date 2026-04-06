from __future__ import annotations

import asyncio

from volvence_zero.agent import AgentSessionRunner, default_active_runner
from volvence_zero.reflection import WritebackMode
from volvence_zero.substrate import (
    OpenWeightResidualStreamSubstrateAdapter,
    SurfaceKind,
    SyntheticOpenWeightResidualRuntime,
)


def test_agent_session_runner_executes_single_turn():
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("I need help organizing my plan and I also feel overwhelmed."))

    assert result.acceptance_passed is True
    assert result.wave_id == "wave-1"
    assert "evaluation" in result.active_snapshots
    assert result.active_regime is not None
    assert result.response.text
    assert result.event_count > 0
    assert result.joint_schedule_action in {"ssl-only", "full-cycle", "evidence-only"}
    assert result.active_snapshots["substrate"].value.surface_kind is SurfaceKind.RESIDUAL_STREAM


def test_agent_session_runner_reuses_session_memory_across_turns():
    runner = AgentSessionRunner(session_id="s1")
    first = asyncio.run(runner.run_turn("Remember that I prefer calm, reflective collaboration."))
    second = asyncio.run(runner.run_turn("Can you help me continue that plan from before?"))

    assert first.wave_id == "wave-1"
    assert second.wave_id == "wave-2"
    assert len(second.active_snapshots["memory"].value.retrieved_entries) >= 1
    assert second.active_snapshots["dual_track"].value.world_track.controller_source == "temporal+memory"


def test_agent_session_runner_exposes_temporal_and_regime_views():
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("Please guide me carefully through a difficult decision."))

    assert result.active_regime is not None
    assert result.active_abstract_action is not None
    assert result.metacontroller_state is not None
    assert result.metacontroller_state.mode == "full-learned"
    assert isinstance(result.evaluation_alerts, tuple)
    assert result.response.regime_id == result.active_regime
    assert result.response.abstract_action == result.active_abstract_action
    assert result.joint_learning_summary
    evaluation_metric_names = {score.metric_name for score in result.active_snapshots["evaluation"].value.turn_scores}
    assert "joint_learning_progress" in evaluation_metric_names
    credit_events = {record.source_event for record in result.active_snapshots["credit"].value.recent_credits}
    assert "joint_learning_progress" in credit_events


def test_agent_session_runner_returns_user_visible_response():
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("I feel tense and I need a careful response."))

    assert result.response.text
    assert result.response.rationale


def test_agent_session_runner_exposes_bounded_writeback_state():
    runner = AgentSessionRunner(session_id="writeback-session", reflection_mode=WritebackMode.APPLY)

    result = asyncio.run(runner.run_turn("Remember a stable preference and keep the interaction supportive."))

    assert result.writeback_source == "shadow"
    assert isinstance(result.writeback_operations, tuple)
    assert isinstance(result.writeback_blocks, tuple)


def test_agent_session_runner_accepts_hook_ready_substrate_factory():
    runtime = SyntheticOpenWeightResidualRuntime(model_id="hook-ready-runtime")
    runner = AgentSessionRunner(
        session_id="hook-session",
        substrate_adapter_factory=lambda user_input, turn_index: OpenWeightResidualStreamSubstrateAdapter(
            runtime=runtime,
            default_source_text=user_input,
        ),
    )

    result = asyncio.run(runner.run_turn("Use the hook-ready substrate path."))

    assert result.acceptance_passed is True
    assert result.active_snapshots["substrate"].value.model_id == "hook-ready-runtime"
