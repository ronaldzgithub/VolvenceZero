from __future__ import annotations

import pytest

from volvence_zero.internal_rl import InternalRLEnvironment
from volvence_zero.internal_rl.sandbox import (
    CausalZPolicy,
    _sequence_action_head_state,
)
from volvence_zero.substrate import (
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
)
from volvence_zero.temporal import MetacontrollerParameterStore


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


def test_rich_action_head_state_preserves_signed_residual_direction() -> None:
    def snapshot(values: tuple[float, ...]) -> SubstrateSnapshot:
        residual = ResidualActivation(
            layer_index=3,
            activation=values,
            step=0,
        )
        step = ResidualSequenceStep(
            step=0,
            token="prefix",
            feature_surface=(),
            residual_activations=(residual,),
            description="Signed residual direction fixture.",
        )
        return SubstrateSnapshot(
            model_id="signed-state-fixture",
            is_frozen=True,
            surface_kind=SurfaceKind.RESIDUAL_STREAM,
            token_logits=(),
            feature_surface=(),
            residual_activations=(residual,),
            residual_sequence=(step,),
            unavailable_fields=(),
            description="Signed residual direction fixture.",
        )

    positive = snapshot((1.0, -1.0, 1.0, -1.0))
    negative = snapshot((-1.0, 1.0, -1.0, 1.0))

    assert _sequence_action_head_state(
        positive,
        12,
    ) == _sequence_action_head_state(negative, 12)
    positive_rich = _sequence_action_head_state(positive, 48)
    negative_rich = _sequence_action_head_state(negative, 48)
    assert len(positive_rich) == 48
    assert positive_rich != negative_rich


def test_unit_latent_contract_applies_to_sampled_action() -> None:
    store = MetacontrollerParameterStore(n_z=3)
    signed_policy = CausalZPolicy(
        parameter_store=store,
        latent_unit_clamp=False,
    )
    unit_policy = CausalZPolicy(
        parameter_store=store,
        latent_unit_clamp=True,
    )
    sample = {
        "policy_mean": (0.0, 0.5, 1.0),
        "policy_std": (0.2, 0.2, 0.2),
        "policy_noise": (-1.0, 0.0, 1.0),
    }

    assert signed_policy._sample_action(**sample) == (
        pytest.approx(-0.1),
        pytest.approx(0.5),
        pytest.approx(1.0),
    )
    assert unit_policy._sample_action(**sample) == (
        pytest.approx(0.0),
        pytest.approx(0.5),
        pytest.approx(1.0),
    )
