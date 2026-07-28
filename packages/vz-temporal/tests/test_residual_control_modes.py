from __future__ import annotations

import pytest

from volvence_zero.internal_rl import InternalRLEnvironment


_CONTROL = (0.1, -0.2, 0.3, -0.4)


def _transformed(mode: str, *, seed: int = 17, step_index: int = 2) -> tuple[float, ...]:
    environment = InternalRLEnvironment(
        residual_control_mode=mode,
        residual_control_seed=seed,
    )
    return environment._transform_residual_control(
        _CONTROL,
        step_index=step_index,
    )


def test_residual_control_identity_and_zero_are_exact() -> None:
    assert _transformed("identity") == _CONTROL
    assert _transformed("zero") == (0.0, 0.0, 0.0, 0.0)


def test_residual_control_reverse_is_exact() -> None:
    assert _transformed("reversed") == tuple(reversed(_CONTROL))


def test_residual_control_shuffle_is_deterministic_non_identity_permutation() -> None:
    first = _transformed("shuffled")
    second = _transformed("shuffled")

    assert first == second
    assert first != _CONTROL
    assert sorted(first) == sorted(_CONTROL)


def test_residual_control_mode_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="Unsupported residual control mode"):
        InternalRLEnvironment(residual_control_mode="semantic-label-hack")

    with pytest.raises(TypeError, match="seed must be an integer"):
        InternalRLEnvironment(residual_control_seed=True)


def test_shuffled_control_requires_multiple_dimensions() -> None:
    environment = InternalRLEnvironment(residual_control_mode="shuffled")

    with pytest.raises(ValueError, match="at least two control dimensions"):
        environment._transform_residual_control((0.2,), step_index=0)
