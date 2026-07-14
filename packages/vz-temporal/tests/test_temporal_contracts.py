"""Minimal contract baseline for the vz-temporal wheel (GAP-02).

vz-temporal previously had zero wheel-local tests; every invariant relied on
downstream vz-runtime coverage. This file pins the four owner-local contracts
the AGI-uplift plan depends on:

1. ``MetacontrollerParameterStore`` n_z=3 (legacy) vs n_z>3 (ndim) dual state:
   ndim encoder/switch/decoder parameters are really instantiated (CP-02).
2. ``train_store_ssl`` write-back semantics: SHADOW never touches the store,
   ACTIVE does (CP-05).
3. ``runtime_ndim_shadow_compare`` pure/torch forward parity report (CP-06).
4. ``torch_causal_ppo_update`` write-back gate on the live policy params (CP-07).

Torch-dependent tests SKIP when torch is missing (never silently pass).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import build_training_trace
from volvence_zero.temporal.interface import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
)
from volvence_zero.tensor_backend import is_torch_available

torch_only = pytest.mark.skipif(not is_torch_available(), reason="torch not installed")

_NDIM = 16


def _trace(trace_id: str = "vz-temporal-contract") -> object:
    return build_training_trace(
        trace_id=trace_id,
        source_text="steady waters carry the harbor plan through changing tides",
    )


# ---------------------------------------------------------------------------
# 1. Parameter store legacy vs ndim dual state (CP-02)
# ---------------------------------------------------------------------------


def test_parameter_store_legacy_nz3_has_no_ndim_parameters() -> None:
    store = MetacontrollerParameterStore(n_z=3)
    assert store.n_z == 3
    assert store.ndim_encoder_parameters is None
    assert store.ndim_switch_parameters is None
    assert store.ndim_decoder_parameters is None


def test_parameter_store_ndim_instantiates_encoder_switch_decoder() -> None:
    store = MetacontrollerParameterStore(n_z=_NDIM)
    assert store.n_z == _NDIM
    assert store.ndim_encoder_parameters is not None
    assert store.ndim_switch_parameters is not None
    assert store.ndim_decoder_parameters is not None
    # Track weights and switch weights must match the unlocked latent dim.
    assert len(store.switch_weights) == _NDIM
    for track in (Track.WORLD, Track.SELF, Track.SHARED):
        assert len(store.track_weights[track]) == _NDIM


def test_full_learned_policy_ndim_components_follow_store() -> None:
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    assert policy.parameter_store.n_z == _NDIM
    # Default backend is the pure rollback baseline.
    assert policy.runtime_backend is WiringLevel.DISABLED


# ---------------------------------------------------------------------------
# 2. train_store_ssl write-back semantics (CP-05)
# ---------------------------------------------------------------------------


@torch_only
def test_store_ssl_shadow_never_writes_back() -> None:
    from volvence_zero.temporal.torch_store_ssl import train_store_ssl

    store = MetacontrollerParameterStore(n_z=_NDIM)
    before = (
        store.ndim_encoder_parameters,
        store.ndim_switch_parameters,
        store.ndim_decoder_parameters,
    )
    report = train_store_ssl(
        store=store, trace=_trace(), n_z=_NDIM, write_back=False
    )
    assert report.wrote_back is False
    assert report.trained_steps >= 1
    assert report.parameters_changed > 0  # the COPY trained for real
    after = (
        store.ndim_encoder_parameters,
        store.ndim_switch_parameters,
        store.ndim_decoder_parameters,
    )
    assert after == before, "SHADOW SSL mutated the live parameter store"


@torch_only
def test_store_ssl_active_writes_back() -> None:
    from volvence_zero.temporal.torch_store_ssl import train_store_ssl

    store = MetacontrollerParameterStore(n_z=_NDIM)
    before_encoder = store.ndim_encoder_parameters
    report = train_store_ssl(
        store=store, trace=_trace(), n_z=_NDIM, write_back=True
    )
    assert report.wrote_back is True
    assert store.ndim_encoder_parameters != before_encoder


@torch_only
def test_store_ssl_requires_ndim_parameters() -> None:
    from volvence_zero.temporal.torch_store_ssl import train_store_ssl

    with pytest.raises(RuntimeError, match="ndim parameters"):
        train_store_ssl(
            store=MetacontrollerParameterStore(n_z=3),
            trace=_trace(),
            n_z=3,
            write_back=False,
        )


# ---------------------------------------------------------------------------
# 3. runtime_ndim_shadow_compare parity report (CP-06)
# ---------------------------------------------------------------------------


@torch_only
def test_runtime_ndim_shadow_compare_reports_parity() -> None:
    from volvence_zero.temporal.backend_ndim_runtime import (
        runtime_ndim_shadow_compare,
    )

    store = MetacontrollerParameterStore(n_z=_NDIM)
    trace = _trace()
    report = runtime_ndim_shadow_compare(
        store=store, substrate_snapshot=_trace_step_snapshot(trace)
    )
    assert report.steps_compared == 1
    assert report.torch_available is True
    assert report.within_tolerance is True, report.description
    for value in (
        report.max_abs_diff_posterior_mean,
        report.max_abs_diff_z_tilde,
        report.max_abs_diff_beta,
        report.max_abs_diff_applied,
    ):
        assert 0.0 <= value <= report.tolerance


def _trace_step_snapshot(trace: object) -> object:
    """Substrate-like view over one trace step (what the runtime encoder reads)."""

    from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind

    step = trace.steps[0]
    return SubstrateSnapshot(
        model_id="vz-temporal-contract",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.5,),
        feature_surface=step.feature_surface,
        residual_activations=step.residual_activations,
        residual_sequence=(),
        unavailable_fields=(),
        description="single trace step as substrate view",
    )


# ---------------------------------------------------------------------------
# 4. torch_causal_ppo_update write-back gate (CP-07)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MiniTransition:
    observation_signature: tuple[float, ...]
    policy_action: tuple[float, ...]
    advantage_estimate: float
    return_estimate: float
    reward: float


def _transitions(n_z: int) -> tuple[_MiniTransition, ...]:
    return tuple(
        _MiniTransition(
            observation_signature=tuple(
                min(1.0, 0.1 + 0.05 * ((i + j) % 7)) for j in range(n_z)
            ),
            policy_action=tuple(
                min(1.0, 0.2 + 0.04 * ((i * 3 + j) % 5)) for j in range(n_z)
            ),
            advantage_estimate=0.1 * (i % 3 - 1),
            return_estimate=0.5 + 0.1 * (i % 2),
            reward=0.4 + 0.05 * i,
        )
        for i in range(6)
    )


@torch_only
@pytest.mark.parametrize("write_back", (False, True))
def test_torch_causal_ppo_write_back_gate(write_back: bool) -> None:
    from volvence_zero.internal_rl.torch_causal_ppo import torch_causal_ppo_update

    store = MetacontrollerParameterStore(n_z=_NDIM)
    value_weights = {Track.WORLD: tuple(0.1 for _ in range(_NDIM))}
    value_bias = {Track.WORLD: 0.05}
    before_weights = store.track_weights[Track.WORLD]
    before_critic = value_weights[Track.WORLD]

    report = torch_causal_ppo_update(
        parameter_store=store,
        value_weights=value_weights,
        value_bias=value_bias,
        track=Track.WORLD,
        transitions=_transitions(_NDIM),
        n_z=_NDIM,
        write_back=write_back,
    )
    assert report.transition_count == 6
    assert report.wrote_back is write_back
    if write_back:
        assert (
            store.track_weights[Track.WORLD] != before_weights
            or value_weights[Track.WORLD] != before_critic
        ), "ACTIVE PPO step changed nothing on the live params"
    else:
        assert store.track_weights[Track.WORLD] == before_weights
        assert value_weights[Track.WORLD] == before_critic
