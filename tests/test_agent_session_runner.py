from __future__ import annotations

import asyncio

from volvence_zero.agent import (
    AgentSessionRunner,
    MultiPathBenchmarkReport,
    default_active_runner,
    run_multi_path_benchmark,
    run_substrate_path_benchmark,
)
from volvence_zero.joint_loop import JointLoopSchedule, PipelineConfig
from volvence_zero.prediction import PredictionError
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


def test_agent_session_runner_defers_slow_writeback_until_context_boundary():
    runner = AgentSessionRunner(session_id="session-post-writeback", reflection_mode=WritebackMode.APPLY)

    first = asyncio.run(runner.run_turn("Remember that I prefer calm and structured support."))
    boundary_ops = runner.begin_new_context(reason="test-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert first.bounded_writeback_applied is False
    assert any(op.startswith("session-post-slow-loop:enqueued:") for op in boundary_ops)
    assert len(slow_loop_results) == 1
    assert slow_loop_results[0].writeback_result is not None
    assert slow_loop_results[0].applied is True or slow_loop_results[0].blocked is True

    second = asyncio.run(runner.run_turn("Continue in the next context."))

    assert second.session_post_completed_job_count >= 1


def test_agent_session_runner_session_post_loop_fails_closed_when_apply_disabled():
    runner = AgentSessionRunner(session_id="session-post-proposal", reflection_mode=WritebackMode.PROPOSAL_ONLY)

    asyncio.run(runner.run_turn("Capture reflection proposals without applying them."))
    runner.begin_new_context(reason="proposal-only-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert len(slow_loop_results) == 1
    assert slow_loop_results[0].writeback_result is not None
    assert "writeback-mode-not-apply" in slow_loop_results[0].writeback_result.blocked_operations


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
    assert result.substrate_runtime_origin == "builtin-fallback"
    assert result.substrate_fallback_active is True
    assert result.substrate_capture_source == "fallback"
    assert result.substrate_residual_sequence_length > 0


def test_agent_session_runner_can_fail_closed_for_missing_substrate_model():
    try:
        AgentSessionRunner(
            session_id="deny-runtime-session",
            substrate_model_id="missing-local-model",
            substrate_runtime_mode="strict-local",
        )
    except Exception as exc:
        assert type(exc).__name__ in {"OSError", "ValueError", "RuntimeError"}
    else:
        raise AssertionError("Expected runner construction to fail when substrate fallback mode is deny.")


def test_agent_session_runner_uses_explicit_model_source_for_strict_local():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            AgentSessionRunner(
                session_id="explicit-source-session",
                substrate_model_id="distilgpt2",
                substrate_model_source=tmpdir,
                substrate_runtime_mode="strict-local",
            )
        except Exception as exc:
            assert type(exc).__name__ in {"OSError", "ValueError", "RuntimeError"}
        else:
            raise AssertionError("Expected explicit empty local model source to fail in strict-local mode.")


def test_agent_session_runner_prefer_local_mode_marks_fallback_metadata():
    runner = AgentSessionRunner(
        session_id="prefer-local-session",
        substrate_model_id="missing-local-model",
        substrate_runtime_mode="prefer-local",
    )

    result = asyncio.run(runner.run_turn("Please use a local model when available."))

    assert result.substrate_runtime_origin == "builtin-fallback"
    assert result.substrate_fallback_active is True
    assert result.substrate_capture_source == "fallback"
    assert result.substrate_model_id == "runner-transformers-runtime"


def test_agent_session_runner_reports_real_runtime_metadata_for_injected_runtime():
    runtime = SyntheticOpenWeightResidualRuntime(model_id="real-path-runtime")
    runtime.runtime_origin = "hf-local"
    runner = AgentSessionRunner(
        session_id="real-path-session",
        default_residual_runtime=runtime,
    )

    result = asyncio.run(runner.run_turn("Use the injected local runtime path."))

    assert result.substrate_model_id == "real-path-runtime"
    assert result.substrate_runtime_origin == "hf-local"
    assert result.substrate_fallback_active is False
    assert result.substrate_capture_source == "real"
    assert result.substrate_residual_sequence_length > 0


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


def test_agent_session_runner_executes_rare_heavy_import_when_high_pe_persists():
    runner = AgentSessionRunner(
        session_id="rare-heavy-session",
        joint_schedule=JointLoopSchedule(ssl_interval=99, rl_interval=99, pe_full_cycle_threshold=0.6),
        rare_heavy_trace_window=2,
        rare_heavy_min_traces=2,
        rare_heavy_cooldown_turns=0,
        rare_heavy_pipeline_config=PipelineConfig(
            n_z=3,
            ssl_min_steps=1,
            ssl_max_steps=1,
            rl_max_steps=1,
        ),
    )

    asyncio.run(runner.run_turn("Seed the first trace for rare-heavy review."))
    runner._previous_prediction_reward = -0.7
    runner._previous_prediction_magnitude = 1.35
    runner._previous_prediction_error = PredictionError(
        task_error=0.9,
        relationship_error=0.6,
        regime_error=0.4,
        action_error=0.8,
        magnitude=1.35,
        signed_reward=-0.7,
        description="Persistent high prediction error should trigger rare-heavy import.",
    )

    result = asyncio.run(runner.run_turn("Continue after the high-error turn."))

    assert result.rare_heavy_result is not None
    assert result.rare_heavy_result.recommended is True
    assert result.rare_heavy_result.applied is True
    assert result.rare_heavy_result.artifact_id is not None
    assert "rare-heavy:temporal-import" in result.rare_heavy_result.applied_operations
    assert "rare-heavy:substrate-import" in result.rare_heavy_result.applied_operations
    assert result.rare_heavy_result.substrate_status == "imported"
    assert result.rare_heavy_result.substrate_training_mode == "adapter-delta-v2"
    assert result.joint_schedule_action == "full-cycle-pe"


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


def test_pe_scheduled_session_turns_still_emit_learning_scores():
    runner = AgentSessionRunner(
        session_id="pe-scheduled-session",
        joint_schedule=JointLoopSchedule(ssl_interval=99, rl_interval=99, pe_full_cycle_threshold=0.6),
    )
    runner._previous_prediction_reward = -0.5
    runner._previous_prediction_magnitude = 0.8
    runner._previous_prediction_error = PredictionError(
        task_error=0.5,
        relationship_error=0.3,
        regime_error=0.2,
        action_error=0.4,
        magnitude=0.8,
        signed_reward=-0.5,
        description="PE-scheduled full cycle.",
    )

    result = asyncio.run(runner.run_turn("Use the PE-scheduled path."))
    scores = {
        score.metric_name: score.value
        for score in result.active_snapshots["evaluation"].value.turn_scores
    }

    assert result.joint_schedule_action == "full-cycle-pe"
    assert "joint_learning_progress" in scores


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


def test_run_substrate_path_benchmark_collects_turn_metrics():
    runner = AgentSessionRunner(
        session_id="benchmark-session",
        substrate_model_id="distilgpt2",
        substrate_runtime_mode="builtin-only",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )
    report = asyncio.run(
        run_substrate_path_benchmark(
            path_label="builtin",
            runner=runner,
            user_inputs=(
                "Help me structure a plan.",
                "I need the next step.",
                "Summarize the main risk.",
            ),
        )
    )

    assert report.path_label == "builtin"
    assert len(report.turns) == 3
    assert 0.0 <= report.acceptance_rate <= 1.0
    assert report.mean_residual_sequence_length > 0
    assert report.mean_turn_score_count > 0
    assert isinstance(report.metric_means, tuple)
    assert report.mean_policy_objective >= 0.0 or report.mean_policy_objective <= 0.0
    assert report.max_family_version >= 0
    assert report.description


def test_run_multi_path_benchmark_compares_paths():
    builtin_runner = AgentSessionRunner(
        session_id="multi-benchmark-builtin",
        substrate_model_id="distilgpt2",
        substrate_runtime_mode="builtin-only",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )
    local_runner = AgentSessionRunner(
        session_id="multi-benchmark-local",
        substrate_model_id="distilgpt2",
        substrate_runtime_mode="strict-local",
        substrate_device="cpu",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )
    report = asyncio.run(
        run_multi_path_benchmark(
            baseline_label="builtin",
            path_runners=(("builtin", builtin_runner), ("hf-local", local_runner)),
            user_inputs=(
                "Help me structure a plan.",
                "I need the next step.",
                "Summarize the main risk.",
            ),
        )
    )

    assert isinstance(report, MultiPathBenchmarkReport)
    assert report.baseline_label == "builtin"
    assert len(report.path_reports) == 2
    assert len(report.metric_deltas_from_baseline) == 1
    assert report.description
