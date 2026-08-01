from __future__ import annotations

import asyncio

from volvence_zero.joint_loop import ETANLJointLoop
from volvence_zero.substrate import build_training_trace


def _trace():
    return build_training_trace(
        trace_id="gate-suite-control",
        source_text=("steady waters carry the harbor plan through changing tides"),
    )


def test_no_ssl_control_runs_readout_then_restores_ssl_owner_state() -> None:
    trace = _trace()
    loop = ETANLJointLoop(apply_ssl_optimization=False)
    before = loop.create_learning_checkpoint(checkpoint_id="before")

    report = asyncio.run(
        loop.run_cycle(
            cycle_index=1,
            trace=trace,
            apply_writeback=False,
            apply_policy_optimization=False,
        )
    )
    after = loop.create_learning_checkpoint(checkpoint_id="after")

    assert report.ssl_prediction_loss > 0.0
    assert report.ssl_rollback_applied is True
    assert "ssl-rollback" in report.applied_operations
    assert after.world_temporal_snapshot.encoder_weights == before.world_temporal_snapshot.encoder_weights
    assert after.self_temporal_snapshot.encoder_weights == before.self_temporal_snapshot.encoder_weights


def test_m3_slow_gain_reaches_live_ssl_optimizer_report() -> None:
    loop = ETANLJointLoop(ssl_m3_slow_gain=1.0)
    trace = _trace()
    for cycle in range(1, 4):
        asyncio.run(
            loop.run_cycle(
                cycle_index=cycle,
                trace=trace,
                apply_writeback=False,
                apply_policy_optimization=False,
            )
        )

    report = loop.latest_ssl_report
    assert report is not None
    assert report.encoder_optimizer_state is not None
    assert report.encoder_optimizer_state.slow_gain == 1.0
    assert any(abs(value) > 0.0 for value in report.m3_slow_momentum_signal)
