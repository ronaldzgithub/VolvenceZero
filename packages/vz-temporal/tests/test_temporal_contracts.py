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

import asyncio
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


# ---------------------------------------------------------------------------
# 3b. Internal-RL -> runtime code bridge (autograd-owner-integration)
#
# Regression guard for the diagnosed structural break: reward-driven learning
# writes ``track_weights``, but the ndim runtime forward produced ``code`` from
# the encoder/switch params only, so RL never reached z_t. The gated runtime
# track modulation opens that bridge while keeping strength 0 an exact no-op.
# ---------------------------------------------------------------------------


def _skewed_track_weights(n_z: int) -> dict:
    """A non-uniform, normalized per-track mixture (what RL would settle to)."""

    skew = tuple((1.0 if i == 0 else 0.02) for i in range(n_z))
    total = sum(skew)
    normalized = tuple(v / total for v in skew)
    return {track: normalized for track in (Track.WORLD, Track.SELF, Track.SHARED)}


def _first_code(policy: FullLearnedTemporalPolicy, snapshot: object) -> tuple[float, ...]:
    return policy.step(
        substrate_snapshot=snapshot, previous_snapshot=None
    ).controller_state.code


def test_runtime_track_modulation_zero_is_exact_noop() -> None:
    """strength 0: mutating track_weights must NOT change code (rollback)."""

    snapshot = _trace_step_snapshot(_trace())

    baseline = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    code_default = _first_code(baseline, snapshot)

    mutated = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    mutated.parameter_store.track_weights = _skewed_track_weights(_NDIM)
    code_mutated = _first_code(mutated, snapshot)

    # With modulation off (default), track_weights are irrelevant to code.
    assert code_default == code_mutated


def test_runtime_track_modulation_lets_track_weights_reach_code() -> None:
    """strength > 0: different track_weights -> different code (bridge open)."""

    snapshot = _trace_step_snapshot(_trace())

    uniform = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    uniform.set_runtime_track_modulation(0.5)
    uniform.parameter_store.track_weights = {
        track: tuple(1.0 / _NDIM for _ in range(_NDIM))
        for track in (Track.WORLD, Track.SELF, Track.SHARED)
    }
    code_uniform = _first_code(uniform, snapshot)

    skewed = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    skewed.set_runtime_track_modulation(0.5)
    skewed.parameter_store.track_weights = _skewed_track_weights(_NDIM)
    code_skewed = _first_code(skewed, snapshot)

    # A uniform mixture is a no-op even at strength>0; a skewed (RL-learned)
    # mixture must move code. This is the reward->code bridge.
    assert code_uniform != code_skewed
    # And the uniform-mixture code must equal the strength-0 baseline (uniform
    # deviates from itself by nothing, so the gain is identity).
    baseline = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    baseline.parameter_store.track_weights = {
        track: tuple(1.0 / _NDIM for _ in range(_NDIM))
        for track in (Track.WORLD, Track.SELF, Track.SHARED)
    }
    assert code_uniform == _first_code(baseline, snapshot)


def test_runtime_track_modulation_rejects_negative_strength() -> None:
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    with pytest.raises(ValueError, match="must be >= 0"):
        policy.set_runtime_track_modulation(-0.1)


def test_runtime_posterior_exploration_is_opt_in_and_reproducible() -> None:
    snapshot = _trace_step_snapshot(_trace())
    baseline = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    baseline_code = _first_code(baseline, snapshot)

    explored_a = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    explored_a.set_runtime_exploration(1.0)
    explored_a_code = _first_code(explored_a, snapshot)
    explored_b = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    explored_b.set_runtime_exploration(1.0)
    explored_b_code = _first_code(explored_b, snapshot)

    assert explored_a_code != baseline_code
    assert explored_a_code == explored_b_code
    assert (
        explored_a.export_runtime_state().posterior_sample_noise
        == explored_b.export_runtime_state().posterior_sample_noise
    )


def test_runtime_posterior_exploration_rejects_out_of_range_strength() -> None:
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        policy.set_runtime_exploration(1.01)


def test_sandbox_policy_mean_uses_live_runtime_track_modulation() -> None:
    """Sandbox policy distribution and live code share one modulation rule."""

    from volvence_zero.internal_rl.sandbox import InternalRLSandbox

    store = MetacontrollerParameterStore(n_z=_NDIM)
    strength = 0.5
    policy = FullLearnedTemporalPolicy(parameter_store=store)
    policy.set_runtime_track_modulation(strength)
    causal = InternalRLSandbox(policy=policy).causal_policy
    assert causal.runtime_track_modulation_strength == strength
    hidden = tuple(0.1 + i * 0.01 for i in range(_NDIM))
    surface = tuple(0.2 + i * 0.005 for i in range(_NDIM))
    previous = tuple(0.05 for _ in range(_NDIM))
    weights = _skewed_track_weights(_NDIM)[Track.WORLD]
    base_candidate = tuple(
        min(
            1.0,
            hidden[i] * 0.50 + surface[i] * 0.30 + previous[i] * 0.20,
        )
        for i in range(_NDIM)
    )
    expected = store.runtime_track_modulated_code(
        base_candidate,
        strength=strength,
        track_override=(Track.WORLD, weights),
    )

    actual = causal._policy_mean(
        track=Track.WORLD,
        hidden_state=hidden,
        surface=surface,
        previous_action=previous,
        weights=weights,
    )
    assert actual == expected


def test_causal_override_is_not_runtime_modulated_twice() -> None:
    """Sandbox emits a final modulated candidate; live override consumes it once."""

    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    policy.set_runtime_track_modulation(0.5)
    override = tuple(0.8 for _ in range(_NDIM))
    step = policy.step_with_causal_override(
        substrate_snapshot=_trace_step_snapshot(_trace()),
        previous_snapshot=None,
        latent_override=override,
        policy_replacement_score=1.0,
        binary_gate_override=True,
    )
    assert step.controller_state.code == override


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
    hidden_state: tuple[float, ...]
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
            hidden_state=tuple(
                min(1.0, 0.15 + 0.03 * ((i + j * 2) % 7))
                for j in range(n_z)
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


@torch_only
def test_torch_causal_ppo_update_changes_live_runtime_code_when_bridge_open() -> None:
    """End-to-end regression: PPO writeback must move the real ndim ``code``."""

    from volvence_zero.internal_rl.torch_causal_ppo import torch_causal_ppo_update

    snapshot = _trace_step_snapshot(_trace())
    strength = 0.5

    baseline_store = MetacontrollerParameterStore(n_z=_NDIM)
    baseline_policy = FullLearnedTemporalPolicy(parameter_store=baseline_store)
    baseline_policy.set_runtime_track_modulation(strength)
    before_code = _first_code(baseline_policy, snapshot)

    learned_store = MetacontrollerParameterStore(n_z=_NDIM)
    value_weights = {Track.WORLD: tuple(0.1 for _ in range(_NDIM))}
    value_bias = {Track.WORLD: 0.05}
    report = torch_causal_ppo_update(
        parameter_store=learned_store,
        value_weights=value_weights,
        value_bias=value_bias,
        track=Track.WORLD,
        transitions=_transitions(_NDIM),
        n_z=_NDIM,
        write_back=True,
        runtime_track_modulation_strength=strength,
    )
    learned_policy = FullLearnedTemporalPolicy(parameter_store=learned_store)
    learned_policy.set_runtime_track_modulation(strength)
    after_code = _first_code(learned_policy, snapshot)

    assert report.wrote_back is True
    assert report.parameters_changed > 0
    assert learned_store.track_weights[Track.WORLD] != baseline_store.track_weights[
        Track.WORLD
    ]
    assert after_code != before_code


def test_joint_loop_no_optimize_reports_but_does_not_persist_rl_update() -> None:
    """Matched control runs the optimizer but restores its policy/critic write."""

    from volvence_zero.joint_loop import ETANLJointLoop

    trace = _trace("joint-no-optimize")
    enabled = ETANLJointLoop()
    enabled_report = asyncio.run(
        enabled.run_cycle(
            cycle_index=1,
            trace=trace,
            apply_policy_optimization=True,
        )
    )
    disabled = ETANLJointLoop()
    disabled_report = asyncio.run(
        disabled.run_cycle(
            cycle_index=1,
            trace=trace,
            apply_policy_optimization=False,
        )
    )

    assert enabled_report.policy_objective != 0.0
    assert disabled_report.policy_objective != 0.0
    assert enabled_report.policy_update_applied is True
    assert disabled_report.policy_update_applied is False
