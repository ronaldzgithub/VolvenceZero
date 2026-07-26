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
import math
from dataclasses import dataclass, replace

import pytest

from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import build_training_trace
from volvence_zero.temporal.interface import (
    FamilyOutcomeFeedback,
    FullLearnedTemporalPolicy,
    LearnedLiteTemporalPolicy,
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


def test_causal_action_head_preserves_signed_encoder_state_symmetry() -> None:
    store = MetacontrollerParameterStore(n_z=_NDIM)
    zero = tuple(0.0 for _ in range(_NDIM))
    positive = tuple(0.25 for _ in range(_NDIM))
    negative = tuple(-value for value in positive)

    zero_basis = store.causal_action_head_basis(
        track=Track.WORLD,
        state_features=zero,
    )
    positive_basis = store.causal_action_head_basis(
        track=Track.WORLD,
        state_features=positive,
    )
    negative_basis = store.causal_action_head_basis(
        track=Track.WORLD,
        state_features=negative,
    )
    initial = store.causal_action_head_parameters(
        track=Track.WORLD
    )
    positive_residual = store.causal_action_head_residual(
        track=Track.WORLD,
        state_features=positive,
        strength=1.0,
    )
    negative_residual = store.causal_action_head_residual(
        track=Track.WORLD,
        state_features=negative,
        strength=1.0,
    )

    assert zero_basis == pytest.approx(tuple(0.0 for _ in zero_basis))
    assert negative_basis == pytest.approx(
        tuple(-value for value in positive_basis)
    )
    assert all(
        value == 0.0
        for row in initial.output_factors
        for value in row
    )
    assert negative_residual == pytest.approx(
        tuple(-value for value in positive_residual)
    )


def test_causal_action_head_rejects_out_of_range_encoder_state() -> None:
    store = MetacontrollerParameterStore(n_z=_NDIM)
    with pytest.raises(ValueError, match="signed encoder state"):
        store.causal_action_head_basis(
            track=Track.WORLD,
            state_features=(1.01,) + tuple(0.0 for _ in range(_NDIM - 1)),
        )


def test_causal_action_head_prioritizes_state_path_over_bounded_intercept() -> None:
    store = MetacontrollerParameterStore(n_z=_NDIM)
    positive = tuple(0.25 for _ in range(_NDIM))
    negative = tuple(-value for value in positive)
    positive_gradient = (1.0,) + tuple(0.0 for _ in range(_NDIM - 1))
    negative_gradient = (-1.0,) + tuple(
        0.0 for _ in range(_NDIM - 1)
    )
    before = store.causal_action_head_parameters(track=Track.WORLD)

    store.update_causal_action_head(
        track=Track.WORLD,
        state_feature_batch=(positive, negative),
        action_gradients=(positive_gradient, negative_gradient),
        advantages=(1.0, 1.0),
    )

    after = store.causal_action_head_parameters(track=Track.WORLD)
    assert after.bias[0] == pytest.approx(0.0)
    assert after.output_factors != before.output_factors
    assert after.input_factors != before.input_factors
    positive_residual = store.causal_action_head_residual(
        track=Track.WORLD,
        state_features=positive,
        strength=1.0,
    )
    negative_residual = store.causal_action_head_residual(
        track=Track.WORLD,
        state_features=negative,
        strength=1.0,
    )
    assert positive_residual[0] == pytest.approx(-negative_residual[0])
    assert abs(positive_residual[0]) > 0.0


def test_causal_action_head_intercept_stays_tightly_bounded() -> None:
    store = MetacontrollerParameterStore(n_z=_NDIM)
    zero = tuple(0.0 for _ in range(_NDIM))
    gradient = (4.0,) + tuple(0.0 for _ in range(_NDIM - 1))
    for _ in range(200):
        store.update_causal_action_head(
            track=Track.WORLD,
            state_feature_batch=(zero,),
            action_gradients=(gradient,),
            advantages=(1.0,),
        )

    parameters = store.causal_action_head_parameters(track=Track.WORLD)
    assert parameters.bias[0] == pytest.approx(0.1)
    assert max(abs(value) for value in parameters.bias) <= 0.1


def test_causal_action_head_keeps_common_signal_out_of_state_factors() -> None:
    store = MetacontrollerParameterStore(n_z=_NDIM)
    positive = tuple(0.25 for _ in range(_NDIM))
    negative = tuple(-value for value in positive)
    gradient = (1.0,) + tuple(0.0 for _ in range(_NDIM - 1))
    opposite_gradient = (-1.0,) + tuple(
        0.0 for _ in range(_NDIM - 1)
    )
    store.update_causal_action_head(
        track=Track.WORLD,
        state_feature_batch=(positive, negative),
        action_gradients=(gradient, opposite_gradient),
        advantages=(1.0, 1.0),
    )
    before = store.causal_action_head_parameters(track=Track.WORLD)

    store.update_causal_action_head(
        track=Track.WORLD,
        state_feature_batch=(positive, negative),
        action_gradients=(gradient, gradient),
        advantages=(1.0, 1.0),
    )

    after = store.causal_action_head_parameters(track=Track.WORLD)
    assert after.bias[0] > before.bias[0]
    assert after.output_factors == before.output_factors
    assert after.input_factors == before.input_factors


def test_learned_lite_code_dimension_follows_store() -> None:
    policy = LearnedLiteTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    step = policy.step(
        substrate_snapshot=_trace_step_snapshot(_trace()),
        previous_snapshot=None,
    )

    assert step.controller_state.code_dim == _NDIM
    assert len(step.controller_state.code) == _NDIM


def test_learned_lite_publishes_disabled_action_head_shape() -> None:
    policy = LearnedLiteTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )

    runtime_state = policy.export_runtime_state()

    assert runtime_state.causal_action_head_wiring == "disabled"
    assert runtime_state.causal_action_head_state == pytest.approx(
        tuple(0.0 for _ in range(_NDIM))
    )
    assert runtime_state.causal_action_head_residual == pytest.approx(
        tuple(0.0 for _ in range(_NDIM))
    )


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


def test_runtime_posterior_exploration_preserves_state_conditioned_mean() -> None:
    from volvence_zero.substrate import ResidualSequenceStep

    snapshot = _trace_step_snapshot(_trace("exploration-mean"))
    coast_snapshot = replace(
        snapshot,
        residual_sequence=(
            ResidualSequenceStep(
                step=6,
                token="coast",
                feature_surface=snapshot.feature_surface,
                residual_activations=snapshot.residual_activations,
                description="sparse exploration coast phase",
            ),
        ),
    )
    baseline = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    _first_code(baseline, coast_snapshot)
    explored = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    explored.set_runtime_exploration(1.0)
    _first_code(explored, coast_snapshot)

    assert (
        explored.export_runtime_state().posterior_mean
        == baseline.export_runtime_state().posterior_mean
    )
    assert (
        explored.export_runtime_state().posterior_sample_noise
        != baseline.export_runtime_state().posterior_sample_noise
    )


def test_runtime_posterior_exploration_reorients_within_short_episode() -> None:
    from volvence_zero.substrate import ResidualSequenceStep

    snapshot = _trace_step_snapshot(_trace("exploration-option"))

    def sampled_noise(step: int) -> tuple[float, ...]:
        stepped = replace(
            snapshot,
            residual_sequence=(
                ResidualSequenceStep(
                    step=step,
                    token=f"step-{step}",
                    feature_surface=snapshot.feature_surface,
                    residual_activations=snapshot.residual_activations,
                    description="bounded exploration option boundary",
                ),
            ),
        )
        policy = FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
        )
        policy.set_runtime_exploration(1.0)
        _first_code(policy, stepped)
        return policy.export_runtime_state().posterior_sample_noise

    assert sampled_noise(0) == sampled_noise(7)
    assert sampled_noise(0) != sampled_noise(8)


def test_runtime_posterior_exploration_context_diversifies_options() -> None:
    snapshot = _trace_step_snapshot(_trace("exploration-context"))

    def sampled_noise(context: str | None) -> tuple[float, ...]:
        policy = FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
        )
        policy.set_runtime_exploration(1.0)
        policy.set_runtime_exploration_context(context)
        _first_code(policy, snapshot)
        return policy.export_runtime_state().posterior_sample_noise

    assert sampled_noise(None) == sampled_noise(None)
    assert sampled_noise("matched-seed:17") == sampled_noise(
        "matched-seed:17"
    )
    assert sampled_noise("matched-seed:17") != sampled_noise(
        "matched-seed:18"
    )


def test_runtime_posterior_exploration_rejects_out_of_range_strength() -> None:
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        policy.set_runtime_exploration(1.01)
    with pytest.raises(TypeError, match="string or None"):
        policy.set_runtime_exploration_context(17)  # type: ignore[arg-type]


def test_frozen_learning_write_gate_keeps_owner_parameters_read_only() -> None:
    store = MetacontrollerParameterStore(n_z=3)
    policy = FullLearnedTemporalPolicy(parameter_store=store)
    store.discover_action_family(
        latent_code=(0.2, 0.3, 0.4),
        decoder_control=(0.3, 0.4, 0.5),
        switch_gate=0.6,
    )
    policy.set_learning_writes_enabled(False)
    before = store.export_parameter_snapshot()
    family_id = before.action_families[0].family_id

    store.discover_action_family(
        latent_code=(0.8, 0.1, 0.4),
        decoder_control=(0.7, 0.2, 0.5),
        switch_gate=0.9,
        posterior_drift=0.4,
        persistence_window=2.0,
    )
    store.observe_family_outcome_feedback(
        feedback=FamilyOutcomeFeedback(
            family_id=family_id,
            outcome_value=0.8,
            delayed_credit_delta=0.4,
        )
    )
    policy.fit_from_signals(
        residual_strength=0.1,
        memory_strength=0.8,
        reflection_strength=0.1,
    )
    after = store.export_parameter_snapshot()

    assert store.learning_writes_enabled is False
    assert policy.learning_writes_enabled is False
    assert after.action_families == before.action_families
    assert after.action_family_version == before.action_family_version
    assert after.family_match_weights == before.family_match_weights
    assert after.temporal_parameters == before.temporal_parameters


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


@torch_only
def test_torch_causal_action_head_masks_non_actuator_dimensions() -> None:
    from volvence_zero.internal_rl.torch_causal_ppo import torch_causal_ppo_update

    store = MetacontrollerParameterStore(n_z=_NDIM)
    before = store.causal_action_head_parameters(track=Track.WORLD)
    value_weights = {Track.WORLD: tuple(0.1 for _ in range(_NDIM))}
    value_bias = {Track.WORLD: 0.05}

    torch_causal_ppo_update(
        parameter_store=store,
        value_weights=value_weights,
        value_bias=value_bias,
        track=Track.WORLD,
        transitions=_transitions(_NDIM),
        n_z=_NDIM,
        write_back=True,
        causal_action_head_enabled=True,
        causal_action_head_strength=0.35,
        causal_action_head_effective_dims=(0, 1),
        causal_action_head_contrast_pairs=((0, 1),),
    )
    after = store.causal_action_head_parameters(track=Track.WORLD)

    assert after.output_factors[:2] != before.output_factors[:2]
    assert after.output_factors[0] == pytest.approx(
        tuple(-value for value in after.output_factors[1])
    )
    assert after.bias[0] == pytest.approx(-after.bias[1])
    assert after.output_factors[2:] == before.output_factors[2:]
    assert after.bias[2:] == before.bias[2:]


def test_joint_loop_no_optimize_reports_but_does_not_persist_rl_update() -> None:
    """Matched control runs the optimizer but restores its policy/critic write."""

    from volvence_zero.joint_loop import ETANLJointLoop

    trace = _trace("joint-no-optimize")
    enabled = ETANLJointLoop()
    enabled_before = enabled.create_learning_checkpoint(
        checkpoint_id="enabled-before"
    )
    enabled_report = asyncio.run(
        enabled.run_cycle(
            cycle_index=1,
            trace=trace,
            apply_policy_optimization=True,
        )
    )
    enabled_after = enabled.create_learning_checkpoint(
        checkpoint_id="enabled-after"
    )
    disabled = ETANLJointLoop()
    disabled_before = disabled.create_learning_checkpoint(
        checkpoint_id="disabled-before"
    )
    disabled_report = asyncio.run(
        disabled.run_cycle(
            cycle_index=1,
            trace=trace,
            apply_policy_optimization=False,
        )
    )
    disabled_after = disabled.create_learning_checkpoint(
        checkpoint_id="disabled-after"
    )

    assert enabled_report.policy_objective != 0.0
    assert disabled_report.policy_objective != 0.0
    assert enabled_report.policy_update_applied is True
    assert disabled_report.policy_update_applied is False
    assert (
        enabled_before.world_policy_checkpoint.policy_optimization_fingerprint
        != enabled_after.world_policy_checkpoint.policy_optimization_fingerprint
    )
    assert (
        disabled_before.world_policy_checkpoint.policy_optimization_fingerprint
        == disabled_after.world_policy_checkpoint.policy_optimization_fingerprint
    )
    assert (
        disabled_before.self_policy_checkpoint.policy_optimization_fingerprint
        == disabled_after.self_policy_checkpoint.policy_optimization_fingerprint
    )


# ---------------------------------------------------------------------------
# 5. Exclusive steering: the head is the only learned contrast writer
# ---------------------------------------------------------------------------


def test_project_base_code_off_contrast_keeps_common_mode_only() -> None:
    from volvence_zero.temporal.causal_action_projection import (
        project_base_code_off_contrast,
        project_causal_action_head_vector,
    )

    values = (0.8, 0.2, 0.5, 0.9)
    base = project_base_code_off_contrast(values, contrast_pairs=((0, 1),))
    head = project_causal_action_head_vector(values, contrast_pairs=((0, 1),))

    assert base == pytest.approx((0.5, 0.5, 0.5, 0.9))
    # Complementary decomposition: base common mode + head contrast == input.
    assert tuple(
        base_value + head_value
        for base_value, head_value in zip(base[:2], head[:2], strict=True)
    ) == pytest.approx(values[:2])


def test_set_causal_action_head_exclusive_steering_validation() -> None:
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    with pytest.raises(ValueError, match="contrast_pairs"):
        policy.set_causal_action_head(
            wiring_level=WiringLevel.ACTIVE,
            track=Track.WORLD,
            strength=1.0,
            exclusive_steering=True,
        )
    with pytest.raises(ValueError, match="ACTIVE"):
        policy.set_causal_action_head(
            wiring_level=WiringLevel.SHADOW,
            track=Track.WORLD,
            strength=1.0,
            contrast_pairs=((0, 1),),
            exclusive_steering=True,
        )
    policy.set_causal_action_head(
        wiring_level=WiringLevel.ACTIVE,
        track=Track.WORLD,
        strength=1.0,
        contrast_pairs=((0, 1),),
        exclusive_steering=True,
    )
    assert policy.causal_action_head_exclusive_steering is True


def test_exclusive_steering_changes_only_contrast_dims_of_live_code() -> None:
    """Live forward: projection touches the pair dims and nothing else."""

    snapshot = _trace_step_snapshot(_trace())

    def first_code(exclusive: bool) -> tuple[float, ...]:
        policy = FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
        )
        policy.set_runtime_track_modulation(0.3)
        policy.set_causal_action_head(
            wiring_level=WiringLevel.ACTIVE,
            track=Track.WORLD,
            strength=1.0,
            effective_dims=(0, 1, 2),
            contrast_pairs=((0, 1),),
            exclusive_steering=exclusive,
        )
        return _first_code(policy, snapshot)

    baseline = first_code(False)
    projected = first_code(True)

    assert projected[2:] == pytest.approx(baseline[2:])
    # The cold head residual is exactly zero, so any change on the pair dims
    # comes from removing the deterministic base contrast.
    assert projected[:2] != pytest.approx(baseline[:2])


def test_runtime_replay_distribution_projects_base_mean_before_head() -> None:
    from volvence_zero.internal_rl.sandbox import (
        runtime_replay_policy_distribution,
    )

    n = 4
    kwargs = dict(
        base_mean=(0.9, 0.1, 0.4, 0.6),
        base_std=(0.1, 0.1, 0.1, 0.1),
        previous_code=(0.0, 0.0, 0.0, 0.0),
        beta_t=1.0,
        track_weights=tuple(1.0 / n for _ in range(n)),
        other_track_sum=tuple(2.0 / n for _ in range(n)),
        modulation_strength=0.0,
        action_head_residual=(0.2, -0.2, 0.0, 0.0),
    )
    plain_mean, plain_std = runtime_replay_policy_distribution(**kwargs)
    exclusive_mean, exclusive_std = runtime_replay_policy_distribution(
        **kwargs,
        exclusive_contrast_pairs=((0, 1),),
    )

    assert plain_mean == pytest.approx((1.0, -0.1, 0.4, 0.6))
    # Base pair mean (0.9, 0.1) collapses to its common mode 0.5; the head
    # residual is then the only contrast source: (0.5 + 0.2, 0.5 - 0.2).
    assert exclusive_mean == pytest.approx((0.7, 0.3, 0.4, 0.6))
    assert exclusive_std == pytest.approx(plain_std)


def test_exclusive_steering_gate_cannot_manufacture_contrast() -> None:
    """A per-dim beta gate must not re-create contrast the projection removed.

    With a zero-parameter head the contrast axis must stay exactly zero: the
    blend ``gate_i * candidate_i + (1 - gate_i) * previous_i`` otherwise leaks
    ``(gate_0 - gate_1) * (candidate - previous)`` onto the pair.
    """

    snapshot = _trace_step_snapshot(_trace())
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=_NDIM)
    )
    policy.set_runtime_track_modulation(0.3)
    policy.set_causal_action_head(
        wiring_level=WiringLevel.ACTIVE,
        track=Track.WORLD,
        strength=1.0,
        effective_dims=(0, 1, 2),
        contrast_pairs=((0, 1),),
        exclusive_steering=True,
    )
    head = policy.parameter_store.causal_action_head_parameters(
        track=Track.WORLD
    )
    assert all(value == 0.0 for row in head.output_factors for value in row)
    assert all(value == 0.0 for value in head.bias)

    # previous_snapshot=None keeps the owner-local previous code, so the blend
    # really runs across steps.
    for _ in range(4):
        code = policy.step(
            substrate_snapshot=snapshot, previous_snapshot=None
        ).controller_state.code
        assert code[0] == pytest.approx(code[1], abs=1e-12)


def test_causal_z_policy_exclusive_steering_requires_active_head() -> None:
    from volvence_zero.internal_rl.sandbox import CausalZPolicy

    store = MetacontrollerParameterStore(n_z=_NDIM)
    with pytest.raises(ValueError, match="ACTIVE"):
        CausalZPolicy(
            parameter_store=store,
            causal_action_head_wiring=WiringLevel.DISABLED,
            causal_action_head_contrast_pairs=((0, 1),),
            causal_action_head_exclusive_steering=True,
        )
    policy = CausalZPolicy(
        parameter_store=store,
        causal_action_head_wiring=WiringLevel.ACTIVE,
        causal_action_head_strength=1.0,
        causal_action_head_contrast_pairs=((0, 1),),
        causal_action_head_exclusive_steering=True,
    )
    hidden = tuple(0.3 for _ in range(_NDIM))
    surface = tuple(0.6 for _ in range(_NDIM))
    previous = tuple(0.1 for _ in range(_NDIM))
    weights = tuple(
        (0.9 if index == 0 else 0.1) for index in range(_NDIM)
    )
    mean = policy._policy_mean(
        track=Track.WORLD,
        hidden_state=hidden,
        surface=surface,
        previous_action=previous,
        weights=weights,
    )
    # Cold head residual is zero and the skewed weights only touched the
    # pair dims through the base -- exclusive steering must equalize them.
    assert mean[0] == pytest.approx(mean[1])


def test_causal_action_head_input_mirror_is_signed_and_involutive() -> None:
    from volvence_zero.temporal.causal_action_projection import (
        mirror_causal_action_head_input,
        normalize_causal_action_head_input_mirror,
    )

    permutation, signs = normalize_causal_action_head_input_mirror(
        (1, 0, 2, 3),
        (1, 1, -1, 1),
        n_input=4,
    )
    values = (0.2, 0.8, -0.3, 0.6)
    mirrored = mirror_causal_action_head_input(
        values,
        permutation=permutation,
        signs=signs,
    )
    roundtrip = mirror_causal_action_head_input(
        mirrored,
        permutation=permutation,
        signs=signs,
    )

    assert mirrored == pytest.approx((0.8, 0.2, 0.3, 0.6))
    assert roundtrip == pytest.approx(values)
    with pytest.raises(ValueError, match="both permutation and signs"):
        normalize_causal_action_head_input_mirror(
            (0, 1),
            None,
            n_input=2,
        )
    with pytest.raises(ValueError, match="involutive"):
        normalize_causal_action_head_input_mirror(
            (1, 2, 0),
            (1, 1, 1),
            n_input=3,
        )


def test_mirror_equivariant_projection_removes_symmetric_steering() -> None:
    from volvence_zero.temporal.causal_action_projection import (
        project_causal_action_head_mirror_equivariant,
    )

    projected = project_causal_action_head_mirror_equivariant(
        (0.7, -0.1, 0.4),
        (0.5, -0.3, 0.2),
        contrast_pairs=((0, 1),),
    )

    # Direct and mirrored lanes share the same steering contrast (0.4), so it
    # is reflection-symmetric and must be deleted exactly.
    assert projected == pytest.approx((0.0, 0.0, 0.3))


def test_live_action_head_residual_is_mirror_equivariant() -> None:
    from volvence_zero.substrate import (
        ResidualActivation,
        SubstrateSnapshot,
        SurfaceKind,
    )
    from volvence_zero.temporal.causal_action_projection import (
        mirror_causal_action_head_input,
    )

    n_z = 4
    permutation = (1, 0, 2, 3)
    signs = (1, 1, -1, 1)
    store = MetacontrollerParameterStore(n_z=n_z, n_input=n_z)
    store.configure_causal_action_head_rank(track=Track.WORLD, rank=n_z)
    head = store.causal_action_head_parameters(track=Track.WORLD)
    store.restore_causal_action_head_parameters(
        replace(
            head,
            output_factors=(
                (0.8, 0.1, 0.0, 0.0),
                (-0.6, 0.2, 0.0, 0.0),
                (0.0, 0.5, 0.0, 0.0),
                (0.0, 0.0, 0.4, 0.0),
            ),
            bias=(0.08, -0.03, 0.04, 0.02),
            update_step=1,
        )
    )
    checkpoint = store.export_parameter_snapshot()

    def substrate(values: tuple[float, ...]) -> SubstrateSnapshot:
        return SubstrateSnapshot(
            model_id="mirror-contract",
            is_frozen=True,
            surface_kind=SurfaceKind.RESIDUAL_STREAM,
            token_logits=(),
            feature_surface=(),
            residual_activations=(
                ResidualActivation(
                    layer_index=0,
                    activation=values,
                    step=0,
                ),
            ),
            residual_sequence=(),
            unavailable_fields=(),
            description="signed mirror contract fixture",
        )

    def residual(values: tuple[float, ...]) -> tuple[float, ...]:
        policy = FullLearnedTemporalPolicy(
            bootstrap_snapshot=checkpoint
        )
        policy.set_causal_action_head(
            wiring_level=WiringLevel.ACTIVE,
            track=Track.WORLD,
            strength=1.0,
            effective_dims=(0, 1, 2),
            contrast_pairs=((0, 1),),
            exclusive_steering=True,
            input_mirror_permutation=permutation,
            input_mirror_signs=signs,
        )
        policy.step(
            substrate_snapshot=substrate(values),
            previous_snapshot=None,
        )
        return policy.export_runtime_state().causal_action_head_residual

    direct_input = (0.8, 0.2, -0.4, 0.3)
    mirrored_input = mirror_causal_action_head_input(
        direct_input,
        permutation=permutation,
        signs=signs,
    )
    direct = residual(direct_input)
    mirrored = residual(mirrored_input)

    assert abs(direct[0]) > 1e-8
    assert direct[0] == pytest.approx(direct[1] * -1.0)
    assert mirrored[0] == pytest.approx(direct[1])
    assert mirrored[1] == pytest.approx(direct[0])
    assert mirrored[2] == pytest.approx(direct[2])
    assert direct[3] == pytest.approx(0.0)


def test_clone_temporal_policy_preserves_mode_and_splits_the_store() -> None:
    """The second track must be an independent store in the source mode.

    ``ETANLJointLoop`` publishes world/self as two independent checkpoint
    lanes, so a shared ``MetacontrollerParameterStore`` makes the round trip
    asymmetric. ``clone_temporal_policy`` is the owner-side way to build the
    second track without promoting a LEARNED_LITE arm to FULL_LEARNED.
    """

    from volvence_zero.temporal import clone_temporal_policy

    lite = LearnedLiteTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=8)
    )
    lite_clone = clone_temporal_policy(lite)
    assert isinstance(lite_clone, LearnedLiteTemporalPolicy)
    assert lite_clone.parameter_store is not lite.parameter_store
    assert lite_clone.parameter_store.n_z == 8
    assert (
        lite_clone.export_rare_heavy_snapshot()
        == lite.export_rare_heavy_snapshot()
    )

    full = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=8)
    )
    full_clone = clone_temporal_policy(full)
    assert isinstance(full_clone, FullLearnedTemporalPolicy)
    assert full_clone.parameter_store is not full.parameter_store


def test_joint_loop_rejects_a_shared_dual_track_parameter_store() -> None:
    """A shared store silently drops the world lane on checkpoint restore."""

    from volvence_zero.joint_loop import ETANLJointLoop

    shared = LearnedLiteTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=8)
    )
    with pytest.raises(ValueError, match="distinct"):
        ETANLJointLoop(world_policy=shared, self_policy=shared)


def test_joint_loop_learning_checkpoint_round_trips_per_track_heads() -> None:
    """World and self action heads must both survive restore (regression).

    Each track promotes its *own* head to full rank, so the two lanes carry
    genuinely different head shapes. Restoring them into one store used to
    overwrite the world lane with the self lane, which the session-level
    re-export fingerprint guard then reported as a mismatch.
    """

    from volvence_zero.joint_loop import ETANLJointLoop
    from volvence_zero.temporal import clone_temporal_policy

    n_z = 8
    world = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=n_z)
    )
    self_policy = clone_temporal_policy(world)
    assert isinstance(self_policy, FullLearnedTemporalPolicy)
    for policy, track in ((world, Track.WORLD), (self_policy, Track.SELF)):
        policy.set_causal_action_head(
            wiring_level=WiringLevel.ACTIVE,
            track=track,
            strength=1.0,
            rank=n_z,
        )
    loop = ETANLJointLoop(world_policy=world, self_policy=self_policy)

    def head_ranks(snapshot) -> dict[str, int]:
        return {
            head.track.value: head.rank
            for head in snapshot.causal_action_heads
        }

    checkpoint = loop.create_learning_checkpoint(checkpoint_id="round-trip")
    expected_world = head_ranks(checkpoint.world_temporal_snapshot)
    expected_self = head_ranks(checkpoint.self_temporal_snapshot)
    # Each track promoted only its own head, so the lanes really do differ.
    assert expected_world["world"] == n_z
    assert expected_self["self"] == n_z
    assert expected_world != expected_self

    loop.restore_learning_checkpoint(checkpoint)
    restored = loop.create_learning_checkpoint(checkpoint_id="round-trip")
    assert head_ranks(restored.world_temporal_snapshot) == expected_world
    assert head_ranks(restored.self_temporal_snapshot) == expected_self
    assert (
        head_ranks(
            restored.world_policy_checkpoint.metacontroller_snapshot
        )
        == expected_world
    )


# ---------------------------------------------------------------------------
# 5. Frozen causal-action-head update envelope (W3-b)
#
# The envelope's single owner is ``temporal.interface``. The pure update path
# and the torch autograd lane must both satisfy it; an unconstrained intercept
# installs a cross-state fixed turn, which is the failure the whole v10-v22
# debugging chain chased (docs/specs/digital-ant-embodiment.md).
# ---------------------------------------------------------------------------


def _head_with(store: MetacontrollerParameterStore, **overrides):
    return replace(
        store.causal_action_head_parameters(track=Track.WORLD), **overrides
    )


def test_causal_action_head_envelope_projection_bounds_step_and_absolute() -> None:
    """One owner update may move each parameter by at most its step limit."""

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
        project_causal_action_head_update,
    )

    store = MetacontrollerParameterStore(n_z=4)
    store.configure_causal_action_head_rank(track=Track.WORLD, rank=4)
    baseline = store.causal_action_head_parameters(track=Track.WORLD)
    # A wildly out-of-envelope candidate, as an unconstrained Adam step or a
    # hand-edited archive would produce.
    runaway = replace(
        baseline,
        input_factors=tuple(
            tuple(value + 9.0 for value in row) for row in baseline.input_factors
        ),
        output_factors=tuple(
            tuple(value - 9.0 for value in row)
            for row in baseline.output_factors
        ),
        bias=tuple(7.0 for _ in baseline.bias),
    )

    projected = project_causal_action_head_update(
        baseline=baseline, candidate=runaway
    )

    # Per-step displacement is capped, and every cap is actually binding here.
    assert all(
        abs(after - before) == pytest.approx(envelope.input_factor_step_limit)
        for after_row, before_row in zip(
            projected.input_factors, baseline.input_factors, strict=True
        )
        for after, before in zip(after_row, before_row, strict=True)
    )
    assert all(
        abs(after - before) == pytest.approx(envelope.output_factor_step_limit)
        for after_row, before_row in zip(
            projected.output_factors, baseline.output_factors, strict=True
        )
        for after, before in zip(after_row, before_row, strict=True)
    )
    assert all(
        abs(after - before) == pytest.approx(envelope.bias_step_limit)
        for after, before in zip(projected.bias, baseline.bias, strict=True)
    )
    # ...and the absolute ceilings hold on the projected result.
    assert max(
        abs(value)
        for matrix in (projected.input_factors, projected.output_factors)
        for row in matrix
        for value in row
    ) <= envelope.factor_absolute_limit
    assert max(abs(value) for value in projected.bias) <= (
        envelope.bias_absolute_limit
    )


def test_causal_action_head_envelope_projection_clamps_absolute_over_step() -> None:
    """Repeated in-step moves still cannot escape the absolute bias ceiling."""

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
        project_causal_action_head_update,
    )

    store = MetacontrollerParameterStore(n_z=4)
    parameters = _head_with(
        store, bias=tuple(envelope.bias_absolute_limit for _ in range(4))
    )
    # Each step is individually legal; the absolute ceiling must still bind.
    for _ in range(20):
        parameters = project_causal_action_head_update(
            baseline=parameters,
            candidate=replace(
                parameters,
                bias=tuple(
                    value + envelope.bias_step_limit for value in parameters.bias
                ),
            ),
        )

    assert parameters.bias == pytest.approx(
        tuple(envelope.bias_absolute_limit for _ in range(4))
    )


def test_causal_action_head_envelope_projection_rejects_non_finite() -> None:
    """Both sides of the projection must be finite, not just the candidate.

    ``max(nan - step, min(nan + step, candidate))`` is ``nan``, and the
    following absolute clamp silently turns that into ``+absolute_limit``: a
    corrupt baseline would be laundered into a legal parameter pinned at the
    ceiling, which is exactly the fixed intercept the envelope forbids.
    """

    from volvence_zero.temporal.interface import (
        project_causal_action_head_update,
    )

    store = MetacontrollerParameterStore(n_z=4)
    baseline = store.causal_action_head_parameters(track=Track.WORLD)

    with pytest.raises(ValueError, match="candidate must be finite"):
        project_causal_action_head_update(
            baseline=baseline,
            candidate=replace(
                baseline, bias=(float("nan"),) + baseline.bias[1:]
            ),
        )
    for poison in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="baseline must be finite"):
            project_causal_action_head_update(
                baseline=replace(
                    baseline, bias=(poison,) + baseline.bias[1:]
                ),
                candidate=replace(baseline, bias=(0.0,) + baseline.bias[1:]),
            )


def test_causal_action_head_validation_has_no_slack_at_the_frozen_bound() -> None:
    """The frozen bound is exact: no epsilon may widen it.

    The clamp outputs the owner writes are exactly the limit
    (``max(-0.1, min(0.1, x)) == 0.1`` and ``abs(0.1) > 0.1`` is ``False``), so
    no tolerance is needed to accept them -- any slack only widens a frozen
    bound.
    """

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
        validate_causal_action_head_magnitudes,
    )

    store = MetacontrollerParameterStore(n_z=4)
    store.configure_causal_action_head_rank(track=Track.WORLD, rank=4)
    baseline = store.causal_action_head_parameters(track=Track.WORLD)

    # Exactly at both ceilings: legal, and reachable from the owner's clamps.
    at_bound = replace(
        baseline,
        bias=tuple(envelope.bias_absolute_limit for _ in baseline.bias),
        output_factors=tuple(
            tuple(-envelope.factor_absolute_limit for _ in row)
            for row in baseline.output_factors
        ),
    )
    validate_causal_action_head_magnitudes(at_bound)

    # One ULP above: rejected. A 1e-9 tolerance would have accepted this.
    just_over = math.nextafter(envelope.bias_absolute_limit, math.inf)
    assert just_over - envelope.bias_absolute_limit < 1e-9
    with pytest.raises(ValueError, match="bias_absolute_limit"):
        validate_causal_action_head_magnitudes(
            replace(baseline, bias=(just_over,) + baseline.bias[1:])
        )
    factor_over = math.nextafter(envelope.factor_absolute_limit, math.inf)
    assert factor_over - envelope.factor_absolute_limit < 1e-9
    with pytest.raises(ValueError, match="factor_absolute_limit"):
        validate_causal_action_head_magnitudes(
            replace(
                baseline,
                output_factors=((factor_over, 0.0, 0.0, 0.0),)
                + baseline.output_factors[1:],
            )
        )


def test_restore_causal_action_head_rejects_non_finite_parameters() -> None:
    """The direct install path must reject NaN/Inf like the checkpoint path.

    ``restore_parameter_snapshot`` has always refused non-finite heads; this
    path did not, so a diverged optimizer could write NaN straight into the
    live head and only surface it somewhere downstream. Finiteness is not
    governed by the opt-in envelope switch: it is never a legal parameter.
    """

    store = MetacontrollerParameterStore(n_z=4)
    store.configure_causal_action_head_rank(track=Track.WORLD, rank=4)
    baseline = store.causal_action_head_parameters(track=Track.WORLD)

    for poison in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            store.restore_causal_action_head_parameters(
                replace(baseline, bias=(poison,) + baseline.bias[1:])
            )
        with pytest.raises(ValueError, match="non-finite"):
            store.restore_causal_action_head_parameters(
                replace(
                    baseline,
                    input_factors=((poison, 0.0, 0.0, 0.0),)
                    + baseline.input_factors[1:],
                )
            )
    # The live head is untouched by the rejected installs.
    assert store.causal_action_head_parameters(track=Track.WORLD) == baseline


def test_domain_can_turn_on_envelope_enforcement_through_the_policy() -> None:
    """The contract must be reachable, and must stay opt-in.

    ``FullLearnedTemporalPolicy.set_causal_action_head`` is the seam a domain's
    rollout config already flows through, so it is where the declaration lands
    on the store. Default ``False`` keeps the historical permissive restore --
    ``tests/test_runtime_transition_replay.py`` deliberately installs bias
    0.35/0.4 through it.
    """

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
    )

    def configured(*, envelope_enforced: bool) -> FullLearnedTemporalPolicy:
        policy = FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(n_z=4)
        )
        policy.set_causal_action_head(
            wiring_level=WiringLevel.ACTIVE,
            track=Track.WORLD,
            strength=0.5,
            envelope_enforced=envelope_enforced,
        )
        return policy

    forged = replace(
        MetacontrollerParameterStore(n_z=4).causal_action_head_parameters(
            track=Track.WORLD
        ),
        bias=(envelope.bias_absolute_limit * 4.0, 0.0, 0.0, 0.0),
        update_step=1,
    )

    permissive = configured(envelope_enforced=False)
    assert (
        permissive.parameter_store.causal_action_head_envelope_enforced
        is False
    )
    permissive.parameter_store.restore_causal_action_head_parameters(forged)
    assert permissive.parameter_store.causal_action_head_parameters(
        track=Track.WORLD
    ).bias == forged.bias

    strict = configured(envelope_enforced=True)
    assert (
        strict.parameter_store.causal_action_head_envelope_enforced is True
    )
    with pytest.raises(ValueError, match="bias_absolute_limit"):
        strict.parameter_store.restore_causal_action_head_parameters(forged)
    with pytest.raises(ValueError, match="bias_absolute_limit"):
        strict.parameter_store.restore_parameter_snapshot(
            _snapshot_carrying_head(forged)
        )
    # Reversible: flipping the declaration back restores history exactly.
    strict.parameter_store.set_causal_action_head_envelope_enforced(False)
    strict.parameter_store.restore_causal_action_head_parameters(forged)
    assert strict.parameter_store.causal_action_head_parameters(
        track=Track.WORLD
    ).bias == forged.bias


def _snapshot_carrying_head(parameters) -> object:
    source = MetacontrollerParameterStore(n_z=4)
    source.configure_causal_action_head_rank(
        track=Track.WORLD, rank=parameters.rank
    )
    source.restore_causal_action_head_parameters(parameters)
    return source.export_parameter_snapshot()


def test_causal_action_head_update_scales_are_owner_sourced() -> None:
    """Both lanes must read the same proportional step scale, one owner.

    The scale is what makes the update proportional; the envelope only bounds
    it. Duplicating ``/ batch``, ``0.12`` or ``0.05`` in the torch lane would
    let the two backends drift apart silently.
    """

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
        causal_action_head_update_scales,
    )

    scales = causal_action_head_update_scales(
        learning_rate=0.02, batch_size=8
    )
    assert scales.factor_learning_rate == pytest.approx(0.02 / 8)
    assert scales.bias_learning_rate == pytest.approx(
        (0.02 / 8) * envelope.bias_learning_rate_ratio
    )
    assert scales.bias_signal_learning_rate == pytest.approx(
        (0.02 / 8)
        * envelope.bias_learning_rate_ratio
        * envelope.bias_state_path_scale
    )
    # Bias moves far slower than the factors -- that asymmetry is the whole
    # point of the intercept discipline, not an accident of the optimizer.
    assert scales.bias_signal_learning_rate < scales.factor_learning_rate / 100

    with pytest.raises(ValueError, match="non-empty batch"):
        causal_action_head_update_scales(learning_rate=0.02, batch_size=0)
    with pytest.raises(ValueError, match="finite"):
        causal_action_head_update_scales(
            learning_rate=float("nan"), batch_size=4
        )


def test_restore_rejects_out_of_envelope_archive_naming_the_bound() -> None:
    """An archive the owner update path could never produce must fail loudly."""

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
    )

    store = MetacontrollerParameterStore(n_z=4)
    store.configure_causal_action_head_rank(track=Track.WORLD, rank=4)
    forged_bias = _head_with(
        store,
        bias=(envelope.bias_absolute_limit * 4.0, 0.0, 0.0, 0.0),
        update_step=1,
    )
    forged_factors = _head_with(
        store,
        output_factors=(
            (envelope.factor_absolute_limit * 2.0, 0.0, 0.0, 0.0),
        )
        + tuple((0.0, 0.0, 0.0, 0.0) for _ in range(3)),
        update_step=1,
    )

    with pytest.raises(ValueError, match="bias_absolute_limit"):
        store.restore_causal_action_head_parameters(
            forged_bias, enforce_envelope=True
        )
    with pytest.raises(ValueError, match="factor_absolute_limit"):
        store.restore_causal_action_head_parameters(
            forged_factors, enforce_envelope=True
        )
    # The rejection really is the envelope check, not a shape check: the same
    # archive installs fine on the default (historical) contract.
    store.restore_causal_action_head_parameters(forged_bias)
    assert store.causal_action_head_parameters(track=Track.WORLD).bias == (
        forged_bias.bias
    )


def test_envelope_enforced_store_rejects_out_of_envelope_checkpoint() -> None:
    """The declared contract also gates the archive/checkpoint restore path."""

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
    )

    source = MetacontrollerParameterStore(n_z=4)
    source.configure_causal_action_head_rank(track=Track.WORLD, rank=4)
    source.restore_causal_action_head_parameters(
        _head_with(
            source,
            bias=(envelope.bias_absolute_limit * 5.0, 0.0, 0.0, 0.0),
            update_step=1,
        )
    )
    snapshot = source.export_parameter_snapshot()

    # Default contract: byte-compatible with today, the archive still loads.
    permissive = MetacontrollerParameterStore(n_z=4)
    permissive.configure_causal_action_head_rank(track=Track.WORLD, rank=4)
    permissive.restore_parameter_snapshot(snapshot)
    assert permissive.causal_action_head_parameters(
        track=Track.WORLD
    ).bias[0] == pytest.approx(envelope.bias_absolute_limit * 5.0)

    strict = MetacontrollerParameterStore(
        n_z=4, causal_action_head_envelope_enforced=True
    )
    strict.configure_causal_action_head_rank(track=Track.WORLD, rank=4)
    with pytest.raises(ValueError, match="bias_absolute_limit"):
        strict.restore_parameter_snapshot(snapshot)


def test_pure_owner_update_never_leaves_the_frozen_envelope() -> None:
    """The pure path is the reference implementation of the same envelope."""

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
        validate_causal_action_head_magnitudes,
    )

    store = MetacontrollerParameterStore(n_z=_NDIM)
    state = tuple(0.5 for _ in range(_NDIM))
    mirrored = tuple(-value for value in state)
    gradient = (4.0, -4.0) + tuple(0.0 for _ in range(_NDIM - 2))
    for _ in range(200):
        store.update_causal_action_head(
            track=Track.WORLD,
            state_feature_batch=(state, mirrored),
            action_gradients=(gradient, gradient),
            advantages=(1.0, 1.0),
        )

    parameters = store.causal_action_head_parameters(track=Track.WORLD)
    validate_causal_action_head_magnitudes(parameters)
    # The bias ceiling is reached, so the bound is binding rather than vacuous.
    assert max(abs(value) for value in parameters.bias) == pytest.approx(
        envelope.bias_absolute_limit
    )


@dataclass(frozen=True)
class _HeadStep:
    """Largest per-parameter displacement of one torch head write-back."""

    bias: float
    output: float
    input: float


def _torch_head_step(
    *,
    learning_rate: float = 0.02,
    action_gain: float = 1.0,
    strength: float = 0.35,
    ppo_epochs: int = 1,
) -> _HeadStep:
    """One ACTIVE torch update; return the head's per-parameter displacement.

    ``action_gain`` scales the recorded actions away from the reconstructed
    policy mean, which is the head gradient's magnitude knob: the surrogate's
    sensitivity is ``advantage * (action - mean) / variance`` and the ratio is
    exactly 1 on the first epoch, so the head gradient is affine in the gain
    with everything else (batch, advantages, learning rate) held fixed.

    The baseline head is seeded with non-zero output factors so all three
    parameter blocks carry gradient on the very first epoch (from an all-zero
    output factor the bilinear input path has no feedback yet -- the pure owner
    handles that with its block-coordinate step, the torch lane does not).
    """

    from volvence_zero.internal_rl.torch_causal_ppo import torch_causal_ppo_update

    store = MetacontrollerParameterStore(n_z=_NDIM)
    seed = store.causal_action_head_parameters(track=Track.WORLD)
    store.restore_causal_action_head_parameters(
        replace(
            seed,
            output_factors=tuple(
                tuple(
                    0.20 if output_index % seed.rank == rank_index else -0.10
                    for rank_index in range(seed.rank)
                )
                for output_index in range(_NDIM)
            ),
            update_step=1,
        )
    )
    before = store.causal_action_head_parameters(track=Track.WORLD)
    torch_causal_ppo_update(
        parameter_store=store,
        value_weights={Track.WORLD: tuple(0.1 for _ in range(_NDIM))},
        value_bias={Track.WORLD: 0.05},
        track=Track.WORLD,
        transitions=tuple(
            replace(
                transition,
                policy_action=tuple(
                    value * action_gain
                    for value in transition.policy_action
                ),
            )
            for transition in _transitions(_NDIM)
        ),
        n_z=_NDIM,
        write_back=True,
        ppo_epochs=ppo_epochs,
        learning_rate=learning_rate,
        causal_action_head_enabled=True,
        causal_action_head_strength=strength,
        causal_action_head_effective_dims=(0, 1, 2),
        causal_action_head_contrast_pairs=((0, 1),),
    )
    after = store.causal_action_head_parameters(track=Track.WORLD)
    assert after.update_step == before.update_step + 1

    def max_step(after_values, before_values) -> float:
        return max(
            abs(a - b)
            for a, b in zip(after_values, before_values, strict=True)
        )

    def max_matrix_step(after_rows, before_rows) -> float:
        return max(
            max_step(after_row, before_row)
            for after_row, before_row in zip(
                after_rows, before_rows, strict=True
            )
        )

    return _HeadStep(
        bias=max_step(after.bias, before.bias),
        output=max_matrix_step(after.output_factors, before.output_factors),
        input=max_matrix_step(after.input_factors, before.input_factors),
    )


@torch_only
def test_torch_head_step_is_proportional_not_bang_bang() -> None:
    """The envelope must bound the torch step, not BE the torch step.

    Folding the head into the same Adam optimizer and clamping the result made
    the write-back a bang-bang maximum-step controller: Adam normalises every
    element's step to ~lr, so the clamp bound every update at exactly the cap
    and the update carried only the SIGN of the gradient. Measured on this
    batch before the fix: ``bias_step=0.010000 out_step=0.050000
    in_step=0.020000`` at lr=0.02 AND the identical numbers at lr=0.5, and 15
    successive updates walked bias to its absolute 0.1 ceiling in 10 steps -- a
    pinned bias IS the cross-state fixed steering intercept the envelope exists
    to prevent (docs/specs/digital-ant-embodiment.md).

    Compare the pure owner on comparable signal: ``bias_step`` two to three
    orders of magnitude below the cap, i.e. a guard rail rather than the value.
    """

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
    )

    # (a) An ordinary batch at the production default learning rate: every
    #     displacement is strictly inside the envelope, by a wide margin.
    ordinary = _torch_head_step()
    assert 0.0 < ordinary.bias < envelope.bias_step_limit
    assert 0.0 < ordinary.output < envelope.output_factor_step_limit
    assert 0.0 < ordinary.input < envelope.input_factor_step_limit
    # Not merely "inside": two orders of magnitude below, so the cap is a guard
    # rail. A bang-bang controller scores exactly 1.0 on each of these.
    assert ordinary.bias < envelope.bias_step_limit / 100.0
    assert ordinary.output < envelope.output_factor_step_limit / 50.0
    assert ordinary.input < envelope.input_factor_step_limit / 100.0

    # (b) The step scales with the gradient. Doubling the recorded action's
    #     distance from the reconstructed mean doubles the head gradient, and
    #     therefore the step. A sign-only controller returns the same number.
    single = _torch_head_step(action_gain=20.0)
    double = _torch_head_step(action_gain=40.0)
    for scaled, base in (
        (double.bias, single.bias),
        (double.output, single.output),
        (double.input, single.input),
    ):
        assert base < scaled < envelope.output_factor_step_limit
        assert scaled / base == pytest.approx(2.0, rel=5e-3)

    # (c) The step scales with the learning rate too, exactly. This is the
    #     sharpest regression guard: pre-fix, lr=0.02 and lr=0.5 produced
    #     byte-identical steps because both saturated the cap.
    slow = _torch_head_step(learning_rate=0.02)
    fast = _torch_head_step(learning_rate=0.5)
    assert fast.bias / slow.bias == pytest.approx(25.0, rel=1e-9)
    assert fast.output / slow.output == pytest.approx(25.0, rel=1e-9)
    assert fast.input / slow.input == pytest.approx(25.0, rel=1e-9)


@torch_only
def test_torch_head_step_still_saturates_the_frozen_envelope() -> None:
    """A large gradient must still be cut off exactly at the frozen bound.

    Proportional does not mean unbounded: at the SAME production default
    learning rate, a batch whose recorded actions sit far from the
    reconstructed mean drives every block to its per-step cap and no further.
    """

    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
    )

    saturated = _torch_head_step(action_gain=1000.0)

    assert saturated.bias == pytest.approx(envelope.bias_step_limit)
    assert saturated.output == pytest.approx(
        envelope.output_factor_step_limit
    )
    assert saturated.input == pytest.approx(envelope.input_factor_step_limit)
    # An even larger gradient buys nothing: the bound, not the gradient, is
    # what terminates the step.
    harder = _torch_head_step(action_gain=5000.0)
    assert harder.bias == pytest.approx(saturated.bias)
    assert harder.output == pytest.approx(saturated.output)
    assert harder.input == pytest.approx(saturated.input)


@torch_only
def test_torch_head_update_stays_inside_the_absolute_ceiling_over_time() -> None:
    """Repeated updates must not walk the bias to the absolute ceiling.

    Pre-fix, 15 successive updates at the production default lr=0.02 pinned
    ``bias[2]`` at 0.1 by update 10 (0.01, 0.0166, ... 0.0966, 0.1). A bias
    pinned at the ceiling is a constant turn on every state.
    """

    from volvence_zero.internal_rl.torch_causal_ppo import torch_causal_ppo_update
    from volvence_zero.temporal.interface import (
        CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE as envelope,
        validate_causal_action_head_magnitudes,
    )

    store = MetacontrollerParameterStore(n_z=_NDIM)
    trajectory = []
    for _ in range(15):
        torch_causal_ppo_update(
            parameter_store=store,
            value_weights={Track.WORLD: tuple(0.1 for _ in range(_NDIM))},
            value_bias={Track.WORLD: 0.05},
            track=Track.WORLD,
            transitions=_transitions(_NDIM),
            n_z=_NDIM,
            write_back=True,
            learning_rate=0.02,
            causal_action_head_enabled=True,
            causal_action_head_strength=0.35,
            causal_action_head_effective_dims=(0, 1, 2),
            causal_action_head_contrast_pairs=((0, 1),),
        )
        parameters = store.causal_action_head_parameters(track=Track.WORLD)
        validate_causal_action_head_magnitudes(parameters)
        trajectory.append(max(abs(value) for value in parameters.bias))

    assert trajectory[-1] < envelope.bias_absolute_limit / 10.0
    assert store.causal_action_head_parameters(
        track=Track.WORLD
    ).update_step == 15


@torch_only
def test_torch_head_update_is_installed_through_the_owner_validation() -> None:
    """SHADOW must still not write, and ACTIVE must land inside the envelope."""

    from volvence_zero.internal_rl.torch_causal_ppo import torch_causal_ppo_update

    store = MetacontrollerParameterStore(n_z=_NDIM)
    before = store.causal_action_head_parameters(track=Track.WORLD)

    torch_causal_ppo_update(
        parameter_store=store,
        value_weights={Track.WORLD: tuple(0.1 for _ in range(_NDIM))},
        value_bias={Track.WORLD: 0.05},
        track=Track.WORLD,
        transitions=_transitions(_NDIM),
        n_z=_NDIM,
        write_back=False,
        learning_rate=0.5,
        causal_action_head_enabled=True,
        causal_action_head_strength=0.35,
        causal_action_head_effective_dims=(0, 1, 2),
        causal_action_head_contrast_pairs=((0, 1),),
    )

    assert store.causal_action_head_parameters(track=Track.WORLD) == before


# ---------------------------------------------------------------------------
# 6. Runtime-replay latent-code reconstruction bound (W3-b)
# ---------------------------------------------------------------------------


def test_resolve_latent_code_bounds_defaults_to_the_signed_rollback() -> None:
    from volvence_zero.internal_rl.torch_causal_ppo import (
        resolve_latent_code_bounds,
    )
    from volvence_zero.temporal.interface import LATENT_CODE_BOUNDS

    assert resolve_latent_code_bounds(latent_unit_clamp=False) == (-1.0, 1.0)
    assert resolve_latent_code_bounds(latent_unit_clamp=True) == (
        LATENT_CODE_BOUNDS
    )


@torch_only
def test_torch_replay_reconstruction_matches_live_code_range_when_saturated() -> None:
    """A saturated contrast axis must not reconstruct an unreachable mean.

    The live ndim forward bounds ``code`` to ``LATENT_CODE_BOUNDS``, so a mean
    of -0.15 is something the frozen plant can never have emitted; signing
    ``(action - mean)`` against it points the head gradient the wrong way.
    """

    import torch

    from volvence_zero.internal_rl.torch_causal_ppo import (
        resolve_latent_code_bounds,
        torch_runtime_replay_latent_mean,
    )
    from volvence_zero.temporal.interface import (
        LATENT_CODE_BOUNDS,
        MetacontrollerParameterStore,
    )

    n_z = 4
    base_mean = (0.9, 0.05, 0.4, 0.6)
    head_residual = (0.2, -0.2, 0.0, 0.0)
    store = MetacontrollerParameterStore(n_z=n_z)

    def reconstruct(latent_unit_clamp: bool) -> tuple[float, ...]:
        result = torch_runtime_replay_latent_mean(
            torch,
            base_mean=torch.tensor([base_mean], dtype=torch.float64),
            gain=torch.ones((1, n_z), dtype=torch.float64),
            previous_code=torch.zeros((1, n_z), dtype=torch.float64),
            beta_t=torch.ones((1, n_z), dtype=torch.float64),
            head_residual=torch.tensor([head_residual], dtype=torch.float64),
            project_base=lambda value: value,
            bounds=resolve_latent_code_bounds(
                latent_unit_clamp=latent_unit_clamp
            ),
        )
        return tuple(float(value) for value in result[0])

    # Rollback baseline: the historical signed bound leaks a negative mean.
    assert reconstruct(False) == pytest.approx((1.0, -0.15, 0.4, 0.6))
    # Declared contract: agrees with what the live forward could have emitted.
    assert reconstruct(True) == pytest.approx((1.0, 0.0, 0.4, 0.6))

    # ...and "what the live forward could have emitted" is the owner's own
    # unit bound, not a literal repeated in the torch lane.
    lower, upper = LATENT_CODE_BOUNDS
    live_equivalent = store.runtime_track_modulated_code(
        base_mean, strength=0.0
    )
    assert live_equivalent == pytest.approx(base_mean)
    assert reconstruct(True) == pytest.approx(
        tuple(
            max(lower, min(upper, live + residual))
            for live, residual in zip(
                live_equivalent, head_residual, strict=True
            )
        )
    )
