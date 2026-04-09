from __future__ import annotations

import asyncio

from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.internal_rl import (
    DualTrackOptimizationReport,
    InternalRLEnvironment,
    InternalRLSandbox,
    OptimizationReport,
    derive_abstract_action_credit,
)
from volvence_zero.joint_loop import ETANLJointLoop, JointLoopSchedule, PipelineConfig, SSLRLTrainingPipeline
from volvence_zero.memory import Track
from volvence_zero.substrate import (
    NoOpResidualInterventionBackend,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    build_training_trace,
)


def _snapshot_from_step(trace_id: str, step: object) -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id=trace_id,
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.1, 0.2),
        feature_surface=step.feature_surface,
        residual_activations=step.residual_activations,
        residual_sequence=(
            ResidualSequenceStep(
                step=step.step,
                token=step.token,
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                description=f"trace token {step.token}",
            ),
        ),
        unavailable_fields=(),
        description=f"trace step {step.step}",
    )


def test_internal_rl_sandbox_emits_abstract_action_credit():
    trace = build_training_trace(trace_id="rl-trace", source_text="steady warm planning")
    sandbox = InternalRLSandbox()
    rollout = sandbox.rollout(
        rollout_id="rollout-1",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
    )
    credits = derive_abstract_action_credit(rollout=rollout, timestamp_ms=100)

    assert rollout.transitions
    assert credits
    assert credits[0].level == "abstract_action"


def test_internal_rl_sandbox_runs_dual_track_rollout():
    trace = build_training_trace(trace_id="dual-track", source_text="plan carefully and stay warm")
    sandbox = InternalRLSandbox()
    dual_rollout = sandbox.rollout_dual_track(
        rollout_id="rollout-2",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
    )

    assert dual_rollout.task_rollout.track.value == "world"
    assert dual_rollout.relationship_rollout.track.value == "self"
    assert dual_rollout.task_rollout.transitions
    assert dual_rollout.relationship_rollout.transitions
    assert dual_rollout.task_rollout.replacement_mode == "causal-binary"
    assert dual_rollout.task_rollout.transitions[0].policy_action
    assert dual_rollout.task_rollout.transitions[0].applied_control
    assert dual_rollout.task_rollout.transitions[0].policy_replacement_quality >= -1.0
    assert dual_rollout.task_rollout.transitions[0].backend_name == "trace-residual-backend"
    assert dual_rollout.task_rollout.transitions[0].controller_state.switch_gate in {0.0, 1.0}


def test_internal_rl_sandbox_supports_checkpoint_and_optimization_report():
    trace = build_training_trace(trace_id="opt-trace", source_text="balance task and relationship")
    sandbox = InternalRLSandbox()
    initial_temporal_params = sandbox.policy.export_parameters()
    checkpoint = sandbox.create_checkpoint(checkpoint_id="policy-1")
    checkpoint_temporal_params = sandbox.policy.export_parameters()
    dual_rollout = sandbox.rollout_dual_track(
        rollout_id="rollout-3",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
    )
    report = sandbox.optimize(dual_rollout)

    assert report.description
    assert report.task_report.parameter_summary
    assert report.relationship_report.parameter_summary
    assert report.task_report.clip_fraction >= 0.0
    assert report.relationship_report.kl_penalty >= 0.0
    assert sandbox.policy.export_parameters() != initial_temporal_params

    sandbox.restore_checkpoint(checkpoint)
    restored = sandbox.causal_policy.export_parameters()
    assert restored[0].update_step == checkpoint.parameters_by_track[0].update_step
    assert sandbox.policy.export_parameters() == checkpoint_temporal_params


def test_internal_rl_sandbox_can_run_noop_backend_baseline_path():
    trace = build_training_trace(trace_id="baseline-trace", source_text="steady guided planning")
    sandbox = InternalRLSandbox(env=InternalRLEnvironment(control_backend=NoOpResidualInterventionBackend()))

    rollout = sandbox.rollout(
        rollout_id="rollout-baseline",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
    )

    assert rollout.transitions
    assert rollout.transitions[0].backend_name == "noop-residual-backend"


def test_internal_rl_sandbox_can_run_continuous_causal_path():
    trace = build_training_trace(trace_id="continuous-trace", source_text="steady guided planning")
    sandbox = InternalRLSandbox()

    rollout = sandbox.rollout(
        rollout_id="rollout-continuous",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
        replacement_mode="causal",
    )

    assert rollout.replacement_mode == "causal"
    assert 0.0 <= rollout.transitions[0].controller_state.switch_gate <= 1.0


def test_eta_nl_joint_loop_runs_minimal_cycle():
    loop = ETANLJointLoop()
    trace = build_training_trace(trace_id="joint-trace", source_text="repair tension then continue helpfully")
    before = loop.temporal_policy.export_parameters()
    report = asyncio.run(loop.run_cycle(cycle_index=1, trace=trace))
    after = loop.temporal_policy.export_parameters()

    assert report.acceptance_passed is True
    assert report.policy_objective != 0.0
    assert report.task_reward != 0.0
    assert report.relationship_reward != 0.0
    assert report.applied_operations
    assert report.metacontroller_state is not None
    assert report.metacontroller_state.mode == "full-learned"
    assert report.ssl_prediction_loss >= 0.0
    assert report.ssl_kl_loss >= 0.0
    assert report.ssl_posterior_drift >= 0.0
    assert report.kernel_score_count >= 1
    assert report.rollback_reasons == ()
    assert report.ssl_rollback_applied is False
    assert report.owner_path == "online-joint-loop"
    assert report.cms_description
    assert after != before
    assert report.metacontroller_state.active_label.startswith("discovered_family_")
    assert report.metacontroller_state.structure_frozen is True
    assert report.metacontroller_state.learning_phase == "rl-online"
    assert any(operation.startswith("temporal-prior:") for operation in report.applied_operations)


def test_eta_nl_joint_loop_can_rollback_policy_when_reward_regresses():
    loop = ETANLJointLoop()
    loop._previous_total_reward = 999.0
    trace = build_training_trace(trace_id="rollback-trace", source_text="brief weak signal")
    report = asyncio.run(loop.run_cycle(cycle_index=2, trace=trace))

    assert report.policy_rollback_applied is True
    assert report.ssl_rollback_applied is True
    assert report.policy_objective != 0.0
    assert "reward-regression" in report.rollback_reasons
    assert "ssl-rollback" in report.applied_operations
    assert "policy-rollback" in report.applied_operations


def test_eta_nl_joint_loop_detects_surrogate_outcome_decoupling():
    loop = ETANLJointLoop()
    reasons = loop._rollback_reasons(
        total_reward=0.05,
        evaluation_snapshot=EvaluationSnapshot(
            turn_scores=(
                EvaluationScore(family="task", metric_name="task_pressure", value=0.20, confidence=0.6, evidence="low task"),
                EvaluationScore(
                    family="relationship",
                    metric_name="cross_track_stability",
                    value=0.25,
                    confidence=0.6,
                    evidence="low relationship",
                ),
                EvaluationScore(
                    family="learning",
                    metric_name="joint_learning_progress",
                    value=0.35,
                    confidence=0.6,
                    evidence="weak learning",
                ),
                EvaluationScore(
                    family="abstraction",
                    metric_name="abstract_action_usefulness",
                    value=0.30,
                    confidence=0.6,
                    evidence="weak abstraction",
                ),
                EvaluationScore(
                    family="safety",
                    metric_name="contract_integrity",
                    value=1.0,
                    confidence=0.9,
                    evidence="safe",
                ),
            ),
            session_scores=(),
            alerts=(),
            description="low outcome alignment snapshot",
        ),
        optimization_report=DualTrackOptimizationReport(
            task_report=OptimizationReport(
                track=Track.WORLD,
                average_reward=0.4,
                baseline_reward=0.1,
                mean_advantage=0.3,
                surrogate_objective=0.22,
                clip_fraction=0.0,
                kl_penalty=0.05,
                parameter_summary="task positive surrogate",
            ),
            relationship_report=OptimizationReport(
                track=Track.SELF,
                average_reward=0.35,
                baseline_reward=0.1,
                mean_advantage=0.25,
                surrogate_objective=0.18,
                clip_fraction=0.0,
                kl_penalty=0.05,
                parameter_summary="relationship positive surrogate",
            ),
            description="positive surrogate but weak outcomes",
        ),
        metacontroller_state=None,
    )

    assert "surrogate-outcome-decoupling" in reasons


def test_eta_nl_joint_loop_supports_scheduled_ssl_only_and_full_cycle_paths():
    loop = ETANLJointLoop()
    trace = build_training_trace(trace_id="scheduled-trace", source_text="repair then continue carefully")

    ssl_only = asyncio.run(
        loop.run_scheduled_step(
            turn_index=1,
            trace=trace,
            schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        )
    )
    full_cycle = asyncio.run(
        loop.run_scheduled_step(
            turn_index=3,
            trace=trace,
            schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        )
    )

    assert ssl_only.schedule_action == "ssl-only"
    assert ssl_only.cycle_report is None
    assert ssl_only.metacontroller_state is not None
    assert ssl_only.owner_path == "online-joint-loop"
    assert ssl_only.metacontroller_state.learning_phase == "ssl"
    assert ssl_only.metacontroller_state.structure_frozen is False
    assert dict(ssl_only.schedule_telemetry)["ssl_due"] == 1
    assert dict(ssl_only.schedule_telemetry)["rl_due"] == 0
    assert full_cycle.schedule_action == "full-cycle"
    assert full_cycle.cycle_report is not None
    assert full_cycle.owner_path == "online-joint-loop"
    assert dict(full_cycle.schedule_telemetry)["rl_due"] == 1


def test_eta_nl_joint_loop_can_apply_and_rollback_rare_heavy_artifact():
    pipeline = SSLRLTrainingPipeline(
        config=PipelineConfig(n_z=8, ssl_min_steps=2, ssl_max_steps=3, rl_max_steps=1)
    )
    traces = tuple(
        build_training_trace(trace_id=f"rare-heavy-{index}", source_text="repair tension then plan steadily")
        for index in range(5)
    )
    pipeline.run_pipeline(traces=traces)
    artifact = pipeline.export_rare_heavy_artifact(artifact_id="offline-artifact")

    loop = ETANLJointLoop()
    before = loop.temporal_policy.export_parameters()
    result = loop.apply_rare_heavy_artifact(artifact)
    after = loop.temporal_policy.export_parameters()

    assert result.artifact_id == "offline-artifact"
    assert "rare-heavy:temporal-import" in result.applied_operations
    assert after != before

    rollback_operations = loop.rollback_rare_heavy_import(result.checkpoint)
    restored = loop.temporal_policy.export_parameters()

    assert "rare-heavy:temporal-rollback" in rollback_operations
    assert restored == before
