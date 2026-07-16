"""T1 (learned-code-completeness): learnable beta_threshold + capacity profiles.

The binarisation threshold for beta_t is owned by
``MetacontrollerParameterStore`` (updated online via the temporal-prior
``beta-threshold`` group); the switch units must honour the store value
instead of hardcoding 0.55. The ndim learned controller is a first-class
path selectable through ``BrainConfig.temporal_profile``.
"""

from __future__ import annotations

import pytest

from volvence_zero.brain import TEMPORAL_PROFILE_LATENT_DIMS, BrainConfig
from volvence_zero.temporal import FullLearnedTemporalPolicy, MetacontrollerParameterStore
from volvence_zero.temporal.metacontroller_components import (
    NdimSwitchUnit,
    SwitchUnit,
)


def test_switch_unit_honours_injected_beta_threshold() -> None:
    unit = SwitchUnit()
    kwargs = dict(
        previous_code=(0.0, 0.0, 0.0),
        z_tilde=(0.9, 0.9, 0.9),
        posterior_std=(0.1, 0.1, 0.1),
        switch_weights=(0.3, 0.3, 0.3),
        switch_bias=0.1,
        memory_signal=0.2,
        reflection_signal=0.2,
    )
    decision = unit.compute_decision(**kwargs)
    assert 0.0 < decision.beta_continuous <= 1.0
    low = unit.compute_decision(**kwargs, beta_threshold=decision.beta_continuous - 0.01)
    high = unit.compute_decision(**kwargs, beta_threshold=decision.beta_continuous + 0.01)
    assert low.beta_binary == 1
    assert high.beta_binary == 0


def test_ndim_switch_unit_honours_injected_beta_threshold() -> None:
    unit = NdimSwitchUnit(n_z=4, seed=7)
    kwargs = dict(
        z_tilde=(0.8, 0.2, 0.9, 0.4),
        previous_code=(0.1, 0.1, 0.1, 0.1),
        memory_signal=0.3,
        reflection_signal=0.3,
    )
    beta_cont, binary_default, _ = unit.compute(**kwargs)
    # Threshold below the minimum gate value -> everything switches;
    # above the maximum -> nothing switches.
    _, binary_all, _ = unit.compute(**kwargs, beta_threshold=min(beta_cont) - 0.01)
    _, binary_none, _ = unit.compute(**kwargs, beta_threshold=max(beta_cont) + 0.01)
    assert all(value == 1.0 for value in binary_all)
    assert all(value == 0.0 for value in binary_none)
    assert len(binary_default) == 4


def test_parameter_store_beta_threshold_reaches_full_learned_dispatch() -> None:
    store = MetacontrollerParameterStore(n_z=16)
    store.beta_threshold = 0.10
    policy = FullLearnedTemporalPolicy(parameter_store=store)
    assert policy.parameter_store.beta_threshold == 0.10


def test_temporal_profile_resolves_learned_ndim_capacity() -> None:
    assert BrainConfig().resolved_temporal_latent_dim() == 3
    assert (
        BrainConfig(temporal_profile="legacy-3dim").resolved_temporal_latent_dim() == 3
    )
    learned = BrainConfig(temporal_profile="learned-ndim")
    assert learned.resolved_temporal_latent_dim() == TEMPORAL_PROFILE_LATENT_DIMS["learned-ndim"]
    assert learned.resolved_temporal_latent_dim() == 16


def test_temporal_profile_fails_loudly_on_conflict_or_unknown() -> None:
    with pytest.raises(ValueError, match="unknown temporal_profile"):
        BrainConfig(temporal_profile="not-a-profile").resolved_temporal_latent_dim()
    with pytest.raises(ValueError, match="not both"):
        BrainConfig(
            temporal_profile="learned-ndim", temporal_latent_dim=64
        ).resolved_temporal_latent_dim()


def test_explicit_latent_dim_stays_authoritative_without_profile() -> None:
    assert BrainConfig(temporal_latent_dim=64).resolved_temporal_latent_dim() == 64


# --- T2: learned action-family match head -------------------------------


def test_family_match_weights_default_matches_historical_coefficients() -> None:
    from volvence_zero.temporal.metacontroller_components import (
        DEFAULT_FAMILY_MATCH_WEIGHTS,
        FamilyMatchWeights,
    )

    assert DEFAULT_FAMILY_MATCH_WEIGHTS == FamilyMatchWeights()
    assert DEFAULT_FAMILY_MATCH_WEIGHTS.latent_similarity == 0.48
    assert DEFAULT_FAMILY_MATCH_WEIGHTS.monopoly_pressure == -0.05


def test_family_match_weight_update_is_bounded_by_envelope() -> None:
    from volvence_zero.temporal.metacontroller_components import (
        DEFAULT_FAMILY_MATCH_WEIGHTS,
        FAMILY_MATCH_WEIGHT_ENVELOPE,
        update_family_match_weights,
    )

    weights = DEFAULT_FAMILY_MATCH_WEIGHTS
    features = tuple(1.0 for _ in weights.as_tuple())
    for _ in range(500):
        weights = update_family_match_weights(
            weights, features=features, outcome_signal=1.0
        )
    for updated, initial in zip(
        weights.as_tuple(), DEFAULT_FAMILY_MATCH_WEIGHTS.as_tuple(), strict=True
    ):
        assert updated <= initial + FAMILY_MATCH_WEIGHT_ENVELOPE + 1e-9
        assert updated >= initial - FAMILY_MATCH_WEIGHT_ENVELOPE - 1e-9
    assert weights.latent_similarity > DEFAULT_FAMILY_MATCH_WEIGHTS.latent_similarity


def test_family_outcome_feedback_updates_match_head() -> None:
    from volvence_zero.temporal.interface import FamilyOutcomeFeedback
    from volvence_zero.temporal.metacontroller_components import (
        DEFAULT_FAMILY_MATCH_WEIGHTS,
    )

    store = MetacontrollerParameterStore()
    store.discover_action_family(
        latent_code=(0.6, 0.4, 0.5),
        decoder_control=(0.5, 0.5, 0.5),
        switch_gate=0.6,
    )
    active_label = store.latest_active_label
    assert active_label != "unassigned_action"
    changed = store.observe_family_outcome_feedback(
        feedback=FamilyOutcomeFeedback(
            family_id=active_label,
            outcome_value=1.0,
            delayed_credit_delta=0.5,
        )
    )
    assert changed
    assert store.family_match_weights != DEFAULT_FAMILY_MATCH_WEIGHTS


def test_family_match_score_at_default_weights_is_byte_equivalent() -> None:
    """The learned head initialised at defaults reproduces the historical
    fixed-coefficient score (rollback baseline)."""

    from volvence_zero.temporal.metacontroller_components import (
        ActionFamilyObservation,
        _family_from_observation,
        _family_match_score,
        clamp_signed,
        clamp_unit,
        _cosine_similarity,
    )

    observation = ActionFamilyObservation(
        latent_code=(0.7, 0.3, 0.5),
        decoder_control=(0.4, 0.6, 0.5),
        switch_gate=0.5,
        posterior_drift=0.2,
        persistence_window=1.0,
    )
    family = _family_from_observation(
        family_id="fam", observation=observation, support=3, stability=0.8
    )
    score = _family_match_score(family, observation)
    expected = (
        _cosine_similarity(observation.latent_code, family.latent_centroid) * 0.48
        + _cosine_similarity(observation.decoder_control, family.decoder_centroid) * 0.24
        + family.stability * 0.08
        + clamp_unit(1.0 - abs(observation.switch_gate - family.switch_bias)) * 0.05
        + clamp_unit(1.0 - abs(observation.posterior_drift - family.mean_posterior_drift)) * 0.03
        + family.competition_score * 0.08
        + family.long_term_payoff * 0.08
        + clamp_signed(family.delayed_credit_sum / 3.0) * 0.14
        - family.monopoly_pressure * 0.05
        - family.stagnation_pressure * 0.03
    )
    assert score == pytest.approx(expected, abs=1e-12)
