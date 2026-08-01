from __future__ import annotations

import pytest

from volvence_zero.internal_rl.sandbox import CausalZPolicy
from volvence_zero.memory import Track
from volvence_zero.temporal import MetacontrollerParameterStore


def _noise(policy: CausalZPolicy) -> tuple[float, ...]:
    return policy._policy_noise(
        hidden_state=(0.1, 0.2, 0.3, 0.4),
        surface=(0.4, 0.3, 0.2, 0.1),
        step_index=2,
        track=Track.WORLD,
    )


def _policy(seed: int) -> CausalZPolicy:
    return CausalZPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=4),
        exploration_seed=seed,
    )


def test_policy_noise_is_sampled_and_seed_reproducible() -> None:
    left = _policy(17)
    right = _policy(17)
    other = _policy(18)

    first = _noise(left)
    second = _noise(left)

    assert first == _noise(right)
    assert second == _noise(right)
    assert first != second
    assert first != _noise(other)
    assert all(value != 0.0 for value in first)


def test_policy_checkpoint_restores_exploration_rng_state() -> None:
    policy = _policy(23)
    _noise(policy)
    checkpoint = policy.create_checkpoint(checkpoint_id="before-second-draw")
    expected = _noise(policy)
    _noise(policy)

    policy.restore_checkpoint(checkpoint)

    assert _noise(policy) == expected
    assert checkpoint.exploration_rng_state is not None


@pytest.mark.parametrize("seed", (-1, True, 1.5))
def test_policy_rejects_invalid_exploration_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="exploration_seed"):
        CausalZPolicy(
            parameter_store=MetacontrollerParameterStore(n_z=4),
            exploration_seed=seed,  # type: ignore[arg-type]
        )
