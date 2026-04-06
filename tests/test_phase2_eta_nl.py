from __future__ import annotations

import asyncio

from volvence_zero.internal_rl import InternalRLSandbox, derive_abstract_action_credit
from volvence_zero.joint_loop import ETANLJointLoop
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind, build_training_trace


def _snapshot_from_step(trace_id: str, step: object) -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id=trace_id,
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.1, 0.2),
        feature_surface=step.feature_surface,
        residual_activations=step.residual_activations,
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


def test_eta_nl_joint_loop_runs_minimal_cycle():
    loop = ETANLJointLoop()
    trace = build_training_trace(trace_id="joint-trace", source_text="repair tension then continue helpfully")
    report = asyncio.run(loop.run_cycle(cycle_index=1, trace=trace))

    assert report.acceptance_passed is True
    assert report.policy_objective != 0.0
    assert report.task_reward != 0.0
    assert report.relationship_reward != 0.0
    assert report.applied_operations
    assert report.metacontroller_state is not None
    assert report.metacontroller_state.mode == "learned-lite"
    assert report.rollback_reasons == ()
    assert report.cms_description


def test_eta_nl_joint_loop_can_rollback_policy_when_reward_regresses():
    loop = ETANLJointLoop()
    loop._previous_total_reward = 999.0
    trace = build_training_trace(trace_id="rollback-trace", source_text="brief weak signal")
    report = asyncio.run(loop.run_cycle(cycle_index=2, trace=trace))

    assert report.policy_rollback_applied is True
    assert report.policy_objective != 0.0
    assert "reward-regression" in report.rollback_reasons
    assert "policy-rollback" in report.applied_operations
