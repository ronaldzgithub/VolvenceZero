"""Phase 2 tests: offline caste reprogramming (rare-heavy analogue)."""

from __future__ import annotations

import dataclasses

import pytest

from volvence_ant.caste import (
    ColonyRareHeavyBundle,
    EnvironmentPressure,
    IndividualRareHeavyRef,
    RoleProbe,
    cluster_behavioral_roles,
    reprogram_castes,
)
from volvence_ant.controllers.fixed_rule_ant import FixedRuleConfig


_ABUNDANT = EnvironmentPressure(
    label="abundant", food_distance=4.0, food_decay=2.5, food_strength=2.5, food_radius=2.0
)
_SCARCE = EnvironmentPressure(
    label="scarce", food_distance=7.0, food_decay=0.9, food_strength=2.5, food_radius=1.5
)


def test_runtime_cannot_trigger_reprogramming() -> None:
    with pytest.raises(RuntimeError):
        reprogram_castes(pressures=(_ABUNDANT,), allow_offline=False)


def test_role_distribution_shifts_with_pressure() -> None:
    result = reprogram_castes(
        pressures=(_ABUNDANT, _SCARCE),
        n_individuals=10,
        fraction_grid=(0.0, 0.5, 1.0),
        rounds=250,
        seed=0,
        n_seeds=1,
        allow_offline=True,
    )
    by_label = {p.pressure_label: p for p in result.profiles}
    # emergent, not hardcoded: scarce/steep food should not favour FEWER
    # explorers than abundant food
    assert by_label["scarce"].explorer_fraction >= by_label["abundant"].explorer_fraction
    assert result.role_shift_monotone


def test_caste_profile_is_immutable_and_configures_individuals() -> None:
    result = reprogram_castes(
        pressures=(_SCARCE,),
        n_individuals=8,
        fraction_grid=(0.0, 1.0),
        rounds=120,
        seed=0,
        n_seeds=1,
        allow_offline=True,
    )
    profile = result.profiles[0]
    assert len(profile.exploration_bias_by_individual) == 8
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.explorer_fraction = 0.5  # type: ignore[misc]
    cfg = profile.config_for(0, base=FixedRuleConfig(seed=0))
    assert cfg.exploration_bias == profile.exploration_bias_by_individual[0]


def test_rare_heavy_bundle_is_metadata_only_and_roles_are_readout() -> None:
    refs = tuple(
        IndividualRareHeavyRef(
            individual_id=index,
            artifact_id=f"artifact-{index}",
            artifact_digest=f"digest-{index}",
            provenance="offline-ant-trace",
            gate_verdict="reviewed",
        )
        for index in range(4)
    )
    bundle = ColonyRareHeavyBundle(
        schema_version="digital-ant-colony-rare-heavy.v1",
        pressure_label="scarce",
        individuals=refs,
        rollback_verified=True,
    )
    assert bundle.rollback_verified
    assert not hasattr(bundle, "temporal_state")

    roles = cluster_behavioral_roles(
        (
            RoleProbe(0, 8.0, 0.1, 0.8, 0.1),
            RoleProbe(1, 7.0, 0.2, 0.7, 0.2),
            RoleProbe(2, 2.0, 0.8, 0.1, 0.8),
            RoleProbe(3, 2.5, 0.7, 0.2, 0.7),
        ),
        seed=0,
    )
    assert {role.role_label for role in roles} == {"explorer", "patroller"}
