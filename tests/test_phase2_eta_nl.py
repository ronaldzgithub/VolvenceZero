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


def test_eta_nl_joint_loop_runs_minimal_cycle():
    loop = ETANLJointLoop()
    trace = build_training_trace(trace_id="joint-trace", source_text="repair tension then continue helpfully")
    report = asyncio.run(loop.run_cycle(cycle_index=1, trace=trace))

    assert report.acceptance_passed is True
    assert report.cms_description
