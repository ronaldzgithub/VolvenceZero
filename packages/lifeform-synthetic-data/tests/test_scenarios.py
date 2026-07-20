from __future__ import annotations

from collections import Counter

import pytest

from lifeform_synthetic_data.scenario import (
    UNIFIED_V1_FAMILIES,
    UnsupportedScenarioAdapterError,
    load_unified_v1_blueprints,
    to_companion_scenario_payload,
    to_vz_scenario_pack_payload,
    validate_unified_v1_package,
)


def test_unified_v1_package_passes_full_self_check() -> None:
    report = validate_unified_v1_package()

    assert report.scene_count == 96
    assert report.family_count == 16
    assert report.split_counts == (("test", 16), ("train", 64), ("val", 16))
    assert report.routing_test_count >= 16
    assert report.negative_routing_test_count >= 2
    assert report.semantic_coherence_count >= 8
    assert len(report.package_hash) == 64


def test_each_family_has_four_train_one_val_one_test() -> None:
    blueprints = load_unified_v1_blueprints()

    assert {item.family for item in blueprints} == set(UNIFIED_V1_FAMILIES)
    for family in UNIFIED_V1_FAMILIES:
        split_counts = Counter(item.split.value for item in blueprints if item.family == family)
        assert split_counts == {"train": 4, "val": 1, "test": 1}


def test_persona_and_latent_arc_never_cross_splits() -> None:
    blueprints = load_unified_v1_blueprints()

    for attribute in ("persona_id", "latent_arc_id"):
        seen: dict[str, str] = {}
        for blueprint in blueprints:
            value = object.__getattribute__(blueprint, attribute)
            prior = seen.setdefault(value, blueprint.split.value)
            assert prior == blueprint.split.value


def test_vz_scenario_adapter_is_explicit_and_complete() -> None:
    blueprint = load_unified_v1_blueprints()[0]

    payload = to_vz_scenario_pack_payload(blueprint)

    assert set(payload) == {"scenario_id", "description", "turns"}
    assert payload["scenario_id"] == blueprint.scenario_id
    turns = payload["turns"]
    assert isinstance(turns, list)
    assert len(turns) == sum((count + 1) // 2 for count in blueprint.turns_per_session)
    assert all(
        set(turn)
        == {
            "user_input",
            "expected_regime_in",
            "expected_min_pe_magnitude",
        }
        for turn in turns
    )


def test_companion_adapter_only_accepts_representable_families() -> None:
    blueprints = load_unified_v1_blueprints()
    relationship = next(item for item in blueprints if item.family == "relationship_continuity")
    payload = to_companion_scenario_payload(relationship, perturbation_seed=7)

    assert payload["family"] == "F1"
    assert payload["held_out"] is False
    assert payload["public_test"] is True

    task = next(item for item in blueprints if item.family == "task_tool_execution")
    with pytest.raises(UnsupportedScenarioAdapterError):
        to_companion_scenario_payload(task, perturbation_seed=7)
