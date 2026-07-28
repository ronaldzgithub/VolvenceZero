from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


_PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lifeform_domain_character"
    / "scenario_packages"
    / "zhang_wuji_character_migration_v1"
)


def _read_yaml(name: str) -> dict:
    return yaml.safe_load((_PACKAGE_DIR / name).read_text(encoding="utf-8"))


def _read_json(name: str) -> dict:
    return json.loads((_PACKAGE_DIR / name).read_text(encoding="utf-8"))


def test_zhang_wuji_migration_package_shape() -> None:
    manifest = _read_yaml("manifest.yaml")
    ssot = _read_json("ssot_fragment.json")
    scenes = _read_yaml("scenes.yaml")
    suite = _read_yaml("test_suite.yaml")

    assert re.fullmatch(r"[a-z][a-z0-9_]*", manifest["name"])
    assert manifest["name"] == "zhang_wuji_character_migration_v1"
    assert manifest["components"] == {
        "ssot_fragment": "ssot_fragment.json",
        "scenes": "scenes.yaml",
        "test_suite": "test_suite.yaml",
    }
    assert len(manifest["explanation"]) >= 200
    assert manifest["migration_contract"]["runtime_owner_policy"] == (
        "no_new_brain_owner"
    )

    assert ssot["package_name"] == manifest["name"]
    assert ssot["character_ssot"]["character_id"] == "zhang-wuji"
    assert ssot["design_contract"]["routing_method"] == (
        "semantic_embedding_plus_schema_bound_llm_structured_output"
    )
    assert "keyword_contains" in ssot["design_contract"]["forbidden_routing"]

    paths = {path["path_id"]: path for path in ssot["paths"]}
    assert {
        "path_character_profile_ssot_v1",
        "path_subjective_live_through_v1",
        "path_relationship_semantic_spine_v1",
        "path_action_abstraction_transfer_v1",
        "path_behavior_fidelity_holdout_v1",
    } == set(paths)

    sub_goal_to_path = {
        sub_goal["sub_goal_id"]: path_id
        for path_id, path in paths.items()
        for sub_goal in path["sub_goals"]
    }
    referenced_sub_goals: set[str] = set()
    for arc in ssot["arc_specs"]:
        phase_orders = [phase["phase_order"] for phase in arc["phases"]]
        assert phase_orders == list(range(len(phase_orders)))
        for phase in arc["phases"]:
            referenced_sub_goals.update(phase["sub_goal_refs"])

    assert referenced_sub_goals <= set(sub_goal_to_path)
    referenced_paths = {
        sub_goal_to_path[sub_goal_id] for sub_goal_id in referenced_sub_goals
    }
    assert referenced_paths == set(paths)

    assert scenes["package_name"] == manifest["name"]
    scene_path_ids = {scene["path_id"] for scene in scenes["scenes"]}
    assert scene_path_ids <= set(paths)
    assert any(
        scene["scenario_id"] == "zhang_wuji_behavior_negative_01"
        for scene in scenes["scenes"]
    )
    assert all(
        scene["semantic_routing"]["method"] != "keyword_matching"
        for scene in scenes["scenes"]
    )

    assert suite["package_name"] == manifest["name"]
    assert len(suite["routing_tests"]) >= 6
    assert any(
        test["case_type"] == "negative" for test in suite["routing_tests"]
    )
    assert len(suite["llm_evaluation"]["semantic_coherence"]) >= 3
    assert "keyword_to_route_dictionary" in suite["routing_policy"][
        "forbidden_methods"
    ]
