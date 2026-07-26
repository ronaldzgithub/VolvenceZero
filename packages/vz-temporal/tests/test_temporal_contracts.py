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
