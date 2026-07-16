"""T3 (#88): learned SSL->RL schedule gate (report-only SHADOW) tests."""

from __future__ import annotations

import asyncio

import pytest

from volvence_zero.joint_loop import ETANLJointLoop, JointLoopSchedule
from volvence_zero.joint_loop.gate_learner import (
    ScheduleGateLearner,
    build_schedule_gate_features,
)
from volvence_zero.substrate import build_training_trace


def _features(**overrides: float) -> tuple[float, ...]:
    values = dict(
        pe_pressure=0.8,
        family_stability=0.5,
        rollback_risk=0.2,
        transition_pressure=0.4,
        substrate_pressure=0.3,
        rare_heavy_pressure=0.1,
        experience_credit=0.5,
        control_prior_strength=0.5,
        pe_full_cycle_due=True,
        rl_due=True,
    )
    values.update(overrides)
    return build_schedule_gate_features(**values)


def test_gate_learner_starts_neutral_and_learns_from_settlement() -> None:
    learner = ScheduleGateLearner()
    first = learner.derive_shadow(_features())
    assert first.predicted_learning_gain == pytest.approx(0.5)
    for _ in range(30):
        learner.derive_shadow(_features())
        learner.observe_realized_outcome(realized_gain=0.4)
    trained = learner.derive_shadow(_features())
    assert trained.predicted_learning_gain > 0.6
    assert trained.recommends_learning
    assert learner.settled_count == 30


def test_gate_learner_weights_stay_bounded() -> None:
    learner = ScheduleGateLearner()
    for _ in range(2000):
        learner.derive_shadow(_features())
        learner.observe_realized_outcome(realized_gain=1.0)
    assert all(abs(weight) <= 3.0 + 1e-9 for weight in learner.weights())


def test_gate_learner_settlement_requires_pending_shadow() -> None:
    learner = ScheduleGateLearner()
    assert learner.observe_realized_outcome(realized_gain=0.5) is False


def test_gate_learner_rejects_wrong_feature_dim() -> None:
    learner = ScheduleGateLearner()
    with pytest.raises(ValueError, match="features"):
        learner.derive_shadow((1.0, 2.0))


def test_scheduled_step_publishes_learned_gate_telemetry_report_only() -> None:
    loop = ETANLJointLoop()
    trace = build_training_trace(
        trace_id="gate-shadow-trace", source_text="repair then continue carefully"
    )
    result = asyncio.run(
        loop.run_scheduled_step(
            turn_index=3,
            trace=trace,
            schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        )
    )
    telemetry = dict(result.schedule_telemetry)
    assert "learned_gate_pressure_x1000" in telemetry
    assert "learned_gate_recommends_learning" in telemetry
    assert "learned_gate_settled_count" in telemetry
    # Report-only: the live decision is still the rule cascade.
    assert result.schedule_action in {
        "full-cycle",
        "full-cycle-pe",
        "ssl-only",
        "ssl-only-pe",
        "evidence-only",
        "full-cycle-collect",
    }


def test_scheduled_step_settles_gate_on_learning_turns() -> None:
    loop = ETANLJointLoop()
    trace = build_training_trace(
        trace_id="gate-settle-trace", source_text="steady support then deepen planning"
    )
    schedule = JointLoopSchedule(ssl_interval=1, rl_interval=99)
    asyncio.run(loop.run_scheduled_step(turn_index=1, trace=trace, schedule=schedule))
    asyncio.run(loop.run_scheduled_step(turn_index=2, trace=trace, schedule=schedule))
    # First learning turn seeds the baseline; the second settles against it.
    assert loop.schedule_gate_learner.observation_count >= 2


def test_torch_internal_rl_loader_is_first_class() -> None:
    from volvence_zero.internal_rl import load_torch_internal_rl

    module = load_torch_internal_rl()
    assert hasattr(module, "TorchInternalRLConfig")
