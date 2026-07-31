from __future__ import annotations

import pytest

from volvence_zero.temporal import M3Optimizer


def test_slow_gain_changes_trajectory_and_zero_gain_rolls_back() -> None:
    params = ((0.5,),)
    gradients = (((0.3,),), ((0.2,),), ((-0.1,),), ((-0.2,),))
    m3 = M3Optimizer(
        num_groups=1,
        group_dim=1,
        slow_interval=1,
        slow_gain=1.0,
    )
    rollback = M3Optimizer(
        num_groups=1,
        group_dim=1,
        slow_interval=1,
        slow_gain=0.0,
    )
    plain_param = params
    plain_momentum = 0.0
    m3_param = params
    rollback_param = params
    for step_gradient in gradients:
        m3_param = m3.update(
            gradients=step_gradient,
            learning_rate=0.1,
            parameters=m3_param,
        )
        rollback_param = rollback.update(
            gradients=step_gradient,
            learning_rate=0.1,
            parameters=rollback_param,
        )
        plain_momentum = 0.9 * plain_momentum + 0.1 * step_gradient[0][0]
        plain_param = ((plain_param[0][0] + 0.1 * plain_momentum,),)

    assert m3_param != plain_param
    assert rollback_param == plain_param


def test_slow_gain_round_trips_and_mismatched_restore_fails_loudly() -> None:
    optimizer = M3Optimizer(
        num_groups=1,
        group_dim=2,
        slow_interval=1,
        slow_gain=0.75,
    )
    optimizer.update(
        gradients=((0.2, -0.1),),
        learning_rate=0.1,
        parameters=((0.5, 0.5),),
    )
    state = optimizer.export_state()
    restored = M3Optimizer(
        num_groups=1,
        group_dim=2,
        slow_interval=1,
        slow_gain=0.75,
    )
    restored.restore_state(state)

    assert state.slow_gain == 0.75
    assert restored.export_state() == state
    with pytest.raises(ValueError, match="configuration mismatch"):
        M3Optimizer(
            num_groups=1,
            group_dim=2,
            slow_interval=1,
            slow_gain=0.0,
        ).restore_state(state)
