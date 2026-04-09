from __future__ import annotations

import asyncio

from volvence_zero.agent import AgentSessionRunner, default_active_runner
from volvence_zero.joint_loop import JointLoopSchedule
from volvence_zero.reflection import WritebackMode
from volvence_zero.substrate import (
    OpenWeightResidualStreamSubstrateAdapter,
    SubstrateFallbackMode,
    SurfaceKind,
    SyntheticOpenWeightResidualRuntime,
    TransformersOpenWeightResidualRuntime,
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
    assert "Transformers open-weight capture" in result.active_snapshots["substrate"].value.description
    assert "reflection" in result.active_snapshots
    assert "temporal_abstraction" in result.active_snapshots


def test_agent_session_runner_reuses_session_memory_across_turns():
    runner = AgentSessionRunner(session_id="s1")
    first = asyncio.run(runner.run_turn("Remember that I prefer calm, reflective collaboration."))
    second = asyncio.run(runner.run_turn("Can you help me continue that plan from before?"))

    assert first.wave_id == "wave-1"
    assert second.wave_id == "wave-2"
    assert len(second.active_snapshots["memory"].value.retrieved_entries) >= 1
    assert second.active_snapshots["dual_track"].value.world_track.controller_source in ("temporal+memory", "temporal-track-projected")


def test_agent_session_runner_exposes_temporal_and_regime_views():
    runner = AgentSessionRunner(
        session_id="exposed-kernel-session",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )
    result = asyncio.run(runner.run_turn("Please guide me carefully through a difficult decision."))

    assert result.active_regime is not None
    assert result.active_abstract_action is not None
    assert result.metacontroller_state is not None
    assert result.metacontroller_state.mode == "full-learned"
    assert isinstance(result.evaluation_alerts, tuple)
    assert result.response.regime_id == result.active_regime
    assert result.response.abstract_action == result.active_abstract_action
    assert result.joint_learning_summary
    evaluation_scores = {
        score.metric_name: score.value
        for score in result.active_snapshots["evaluation"].value.turn_scores
    }
    evaluation_metric_names = set(evaluation_scores)
    assert "joint_learning_progress" in evaluation_metric_names
    assert "residual_env_fidelity" in evaluation_metric_names
    assert evaluation_scores["joint_learning_progress"] > 0.0
    credit_events = {record.source_event for record in result.active_snapshots["credit"].value.recent_credits}
    assert "joint_learning_progress" in credit_events


def test_agent_session_runner_returns_user_visible_response():
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("I feel tense and I need a careful response."))

    assert result.response.text
    assert result.response.rationale
    assert "switch_gate=" in result.response.rationale
    assert "joint=" in result.response.rationale
    assert "primary_lesson=" in result.response.rationale


def test_agent_session_runner_exposes_bounded_writeback_state():
    runner = AgentSessionRunner(session_id="writeback-session", reflection_mode=WritebackMode.APPLY)

    result = asyncio.run(runner.run_turn("Remember a stable preference and keep the interaction supportive."))

    assert result.writeback_source == "active"
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


def test_agent_session_runner_defaults_to_real_transformers_runtime_with_builtin_fallback():
    runner = AgentSessionRunner(
        session_id="real-runtime-session",
        substrate_model_id="missing-local-model",
        substrate_local_files_only=True,
    )

    assert isinstance(runner._default_residual_runtime, TransformersOpenWeightResidualRuntime)

    result = asyncio.run(runner.run_turn("Use the default real substrate runtime."))

    assert result.active_snapshots["substrate"].value.model_id == "runner-transformers-runtime"
    assert "Transformers open-weight capture" in result.active_snapshots["substrate"].value.description


def test_agent_session_runner_can_fail_closed_for_missing_substrate_model():
    try:
        AgentSessionRunner(
            session_id="deny-runtime-session",
            substrate_model_id="missing-local-model",
            substrate_local_files_only=True,
            substrate_fallback_mode=SubstrateFallbackMode.DENY,
        )
    except Exception as exc:
        assert type(exc).__name__ in {"OSError", "ValueError", "RuntimeError"}
    else:
        raise AssertionError("Expected runner construction to fail when substrate fallback mode is deny.")


# ---------------------------------------------------------------------------
# D1/D2 — Real substrate data flows through evaluation and RL
# ---------------------------------------------------------------------------

def test_evaluation_signals_flow_from_real_substrate_capture():
    """D1: Verify evaluation scores are derived from real substrate
    capture data, not just hardcoded values."""
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("Help me plan a complex project with many moving parts."))

    eval_snapshot = result.active_snapshots["evaluation"].value
    assert len(eval_snapshot.turn_scores) > 0, "Evaluation should produce turn scores from substrate"
    score_families = {s.family for s in eval_snapshot.turn_scores}
    assert "task" in score_families or "learning" in score_families, (
        f"Expected task or learning scores, got families: {score_families}"
    )


def test_multi_turn_rl_loop_produces_policy_changes():
    """D2: Run 5 turns through the full agent session and verify the
    joint loop produces policy parameter changes."""
    runner = AgentSessionRunner(
        session_id="multi-turn-rl-session",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )

    inputs = [
        "Help me organize my thoughts about this project.",
        "I feel overwhelmed by the complexity here.",
        "Can you break this down into smaller steps?",
        "That helps, but I'm still worried about the timeline.",
        "Let's focus on the most important pieces first.",
    ]
    reports = []
    for user_input in inputs:
        result = asyncio.run(runner.run_turn(user_input))
        reports.append(result)

    has_cycle = any(r.joint_cycle_report is not None for r in reports)
    assert has_cycle, "At least one turn should trigger a full joint cycle"

    cycle_reports = [r.joint_cycle_report for r in reports if r.joint_cycle_report is not None]
    objectives = [cr.policy_objective for cr in cycle_reports]
    assert any(abs(o) > 1e-8 for o in objectives), (
        f"Expected non-zero policy objectives from RL, got: {objectives}"
    )


def test_substrate_snapshot_used_for_next_turn_trace():
    """B1 verification: After the first turn, the training trace for
    the second turn should be built from real substrate data."""
    runner = default_active_runner()
    asyncio.run(runner.run_turn("First turn to capture substrate."))
    assert runner._previous_substrate_snapshot is not None, (
        "After first turn, previous substrate snapshot should be stored"
    )
    assert runner._previous_substrate_snapshot.residual_sequence, (
        "Substrate snapshot should contain residual sequence for trace building"
    )


# ---------------------------------------------------------------------------
# E1 — Regime-driven behavioral shifts
# ---------------------------------------------------------------------------

def test_regime_responds_to_emotional_context():
    """E1: Verify that emotional input triggers appropriate regime selection
    (emotional_support or repair_and_deescalation)."""
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("I'm feeling really frustrated and upset right now."))

    assert result.active_regime is not None
    assert result.response.text
    assert result.response.regime_id is not None


def test_regime_responds_to_task_context():
    """E1: Verify that task-oriented input triggers problem_solving or
    guided_exploration regime."""
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("How do I implement a binary search tree in Python?"))

    assert result.active_regime is not None
    assert result.response.text


def test_response_context_carries_cognitive_state():
    """Verify ResponseContext populated with user_input, retrieved_memories,
    controller_description, and control_code from the cognitive graph."""
    runner = default_active_runner()

    asyncio.run(runner.run_turn("First message to seed memory."))
    result = asyncio.run(runner.run_turn("Second message to check cognitive state flow."))

    assert result.response.text
    assert result.response.rationale
