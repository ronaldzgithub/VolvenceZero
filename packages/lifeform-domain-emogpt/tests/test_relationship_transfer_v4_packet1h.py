from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_CONSUMER_SPLIT_NEXT_ACTION,
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME,
    RelationshipAction,
    RelationshipDatasetSplit,
    load_relationship_consumer_split_bundle,
    load_relationship_consumer_training_view,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
)


_V3_FINGERPRINT = "35b8c46e6fd5810779aff38ed935d8c4f0741bf7d496d2e3eec85f93fbf2134f"
_V4_FINGERPRINT = "9bfe6ae0b480ff9c549c4cde6756e47f4b7e25258b44a81e5c49955c8c495796"
_P1G_REPORT_ID = "9d7f05b574bafb21641d22c766fe31c4656c09bf6f5e04493474eee6c694e3c8"
_P1H_CONTRACT_ID = "2ce75cb44515b4c727ad065995501d063a8f3727923e8a322b4378b53e394af8"


def _v4_root() -> Path:
    return relationship_transfer_package_dir(RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME)


def _surface_families(package_name: str) -> set[str]:
    dataset = load_relationship_transfer_dataset(package_name=package_name)
    return {
        family
        for observation in dataset.observations
        for family in (
            observation.probe_surface_family,
            *(history.surface_family for history in observation.histories),
        )
    }


def test_p1h_freezes_seen_v3_as_training_and_v4_as_unseen_qualification() -> None:
    bundle = load_relationship_consumer_split_bundle()
    contract = bundle.contract
    assert bundle.artifact_id == _P1H_CONTRACT_ID
    assert contract.source_p1g_report_artifact_id == _P1G_REPORT_ID
    assert contract.source_required_verdict == "consumer_still_underqualified"
    assert contract.training_package_name == RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME
    assert contract.training_dataset_fingerprint == _V3_FINGERPRINT
    assert contract.training_role == "consumer_training_only"
    assert contract.training_contains_seen_qwen_outputs
    assert contract.qualification_package_name == RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME
    assert contract.qualification_dataset_fingerprint == _V4_FINGERPRINT
    assert contract.qualification_role == "unseen_qualification_only"
    assert contract.qualification_qwen_outputs_observed_before_freeze == 0
    assert contract.next_action == RELATIONSHIP_CONSUMER_SPLIT_NEXT_ACTION
    assert not contract.formal_hidden_test_opened
    assert not contract.p2_enabled


def test_v4_qualification_shape_balance_and_truth_isolation() -> None:
    dataset = load_relationship_transfer_dataset(
        package_name=RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME
    )
    assert dataset.dataset_fingerprint == _V4_FINGERPRINT
    assert len(dataset.observations) == 24
    assert len(dataset.mirrored_pairs()) == 12
    assert len(dataset.history_condition_bindings) == 96
    assert {
        dynamic.split for dynamic in dataset.dynamics
    } == {RelationshipDatasetSplit.HELDOUT}
    positive_outcomes = set(dataset.positive_outcomes)
    sealed_tokens = {
        condition.condition_id for condition in dataset.abstract_conditions
    } | {policy.policy_id for policy in dataset.policy_profiles}
    for _pair_id, members in dataset.mirrored_pairs():
        assert len({observation.current_input for observation, _dynamic in members}) == 1
        assert len({dynamic.probe_condition_id for _observation, dynamic in members}) == 1
        assert {dynamic.preferred_action for _observation, dynamic in members} == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        }
    for observation in dataset.observations:
        assert len(observation.histories) == 4
        assert observation.probe_surface_family not in {
            history.surface_family for history in observation.histories
        }
        assert Counter(history.assistant_action for history in observation.histories) == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: 2,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: 2,
        }
        for action in (
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        ):
            assert {
                history.typed_outcome in positive_outcomes
                for history in observation.histories
                if history.assistant_action is action
            } == {False, True}
        sut_payload = json.dumps(
            observation.to_sut_payload(),
            ensure_ascii=False,
            sort_keys=True,
        )
        assert not any(token in sut_payload for token in sealed_tokens)
        assert "preferred_action" not in sut_payload
        assert "probe_condition_id" not in sut_payload


def test_p1h_training_and_qualification_public_surfaces_are_disjoint() -> None:
    bundle = load_relationship_consumer_split_bundle()
    assert not (
        _surface_families(RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME)
        & _surface_families(RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME)
    )
    training_scene_ids = {
        item.scene_id for item in bundle.training_dataset.observations
    }
    qualification_scene_ids = {
        item.scene_id for item in bundle.qualification_dataset.observations
    }
    assert training_scene_ids.isdisjoint(qualification_scene_ids)
    assert {
        item.user_scope_hash for item in bundle.training_dataset.observations
    }.isdisjoint(
        {
            item.user_scope_hash
            for item in bundle.qualification_dataset.observations
        }
    )


def test_p1i_training_view_does_not_materialize_qualification_data() -> None:
    view = load_relationship_consumer_training_view()
    assert view.training_dataset.package_name == RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME
    assert "qualification_dataset" not in {field.name for field in fields(view)}


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        (
            "qualification_split",
            "qwen_outputs_observed_before_freeze",
            1,
            "first v4 Qwen output",
        ),
        (
            "consumer_search_budget",
            "maximum_revision_rounds",
            4,
            "three rounds",
        ),
        (
            "experiment_guards",
            "qualification_feedback_to_consumer",
            True,
            "cannot open qualification feedback",
        ),
        (
            "frozen_qualification_gate",
            "minimum_pair_flip_rate",
            0.25,
            "diverges from P1g",
        ),
    ),
)
def test_p1h_rejects_post_freeze_weakening(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    root = tmp_path / RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME
    shutil.copytree(_v4_root(), root)
    contract_path = root / "consumer_split_contract.json"
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    raw[section][field] = value
    contract_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_relationship_consumer_split_bundle(contract_path)


def test_p1h_rejects_training_qualification_surface_collision(tmp_path: Path) -> None:
    root = tmp_path / RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME
    shutil.copytree(_v4_root(), root)
    rendered_path = root / "rendered_observations.json"
    rendered = json.loads(rendered_path.read_text(encoding="utf-8"))
    rendered["scenes"][0]["probe_surface_family"] = "work"
    rendered_path.write_text(
        json.dumps(rendered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mutated = load_relationship_transfer_dataset(
        root,
        package_name=RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME,
    )
    contract_path = root / "consumer_split_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["qualification_split"]["dataset_fingerprint"] = (
        mutated.dataset_fingerprint
    )
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="surface families overlap"):
        load_relationship_consumer_split_bundle(contract_path)


def test_relationship_transfer_v4_scenario_package_contract() -> None:
    root = _v4_root()
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    ssot = json.loads((root / "ssot_fragment.json").read_text(encoding="utf-8"))
    scenes = yaml.safe_load((root / "scenes.yaml").read_text(encoding="utf-8"))
    suite = yaml.safe_load((root / "test_suite.yaml").read_text(encoding="utf-8"))

    assert re.fullmatch(r"[a-z][a-z0-9_]*", manifest["name"])
    assert manifest["name"] == RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME
    assert len(manifest["explanation"]) >= 200
    assert manifest["owner_contract"]["owner"] == "lifeform-domain-emogpt.lab"
    for relative in (
        *manifest["components"].values(),
        *manifest["lab_artifacts"].values(),
    ):
        assert (root / relative).is_file()

    paths = {item["path_id"]: item for item in ssot["paths"]}
    referenced_paths: set[str] = set()
    referenced_sub_goals: set[str] = set()
    for arc in ssot["arc_specs"]:
        referenced_paths.update(arc["path_ids"])
        assert [phase["phase_order"] for phase in arc["phases"]] == list(
            range(len(arc["phases"]))
        )
        for phase in arc["phases"]:
            referenced_sub_goals.update(phase["sub_goal_refs"])
    all_sub_goals = {
        sub_goal["sub_goal_id"]
        for path in paths.values()
        for sub_goal in path["sub_goals"]
    }
    assert referenced_paths == set(paths)
    assert referenced_sub_goals == all_sub_goals
    assert len(scenes["scenes"]) == 24
    assert len({item["mirror_group"] for item in scenes["scenes"]}) == 12
    assert {item["split"] for item in scenes["scenes"]} == {"heldout"}
    assert "embedding" in scenes["semantic_routing"]["method"]
    assert "keyword_dictionary" in scenes["semantic_routing"]["forbidden"]
    assert len(suite["routing_tests"]) >= 6
    assert any(item["case_type"] == "negative" for item in suite["routing_tests"])
    assert len(suite["llm_evaluation"]["semantic_coherence"]) >= 3
    assert "keyword_to_route_dictionary" in (
        suite["routing_policy"]["forbidden_methods"]
    )


def test_v4_package_path_stays_inside_relationship_vertical() -> None:
    expected_parent = Path(__file__).resolve().parents[1] / "src/lifeform_domain_emogpt"
    assert _v4_root().is_relative_to(expected_parent)
