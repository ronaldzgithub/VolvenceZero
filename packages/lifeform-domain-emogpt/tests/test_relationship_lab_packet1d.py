from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest
import yaml

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME,
    RelationshipAction,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
)


_V1_FINGERPRINT = "953b0ee3483846e4aac876b0b1e93d58a4c8fb705e1a79db1df093be463e866a"
_V2_FINGERPRINT = "d8e002d6d529476bf29622d4872afb0b1d7fec9d9c2e5942ecb830c8428b660b"


def _v2_root() -> Path:
    return relationship_transfer_package_dir(RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME)


def test_default_v1_lineage_remains_unchanged_and_v2_is_explicit() -> None:
    v1 = load_relationship_transfer_dataset()
    v2 = load_relationship_transfer_dataset(
        package_name=RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    )
    inferred_v2 = load_relationship_transfer_dataset(_v2_root())
    assert v1.package_name == RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME
    assert v1.dataset_fingerprint == _V1_FINGERPRINT
    assert v2.package_name == RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    assert v2.dataset_fingerprint == _V2_FINGERPRINT
    assert inferred_v2.dataset_fingerprint == v2.dataset_fingerprint
    assert v1.dataset_fingerprint != v2.dataset_fingerprint


def test_v2_has_no_global_action_outcome_shortcut() -> None:
    dataset = load_relationship_transfer_dataset(
        package_name=RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    )
    assert dataset.dataset_schema_version == "relationship-transfer-dataset.v2"
    assert dataset.truth_schema_version == "relationship-transfer-truth.v2"
    assert len(dataset.observations) == 12
    assert len(dataset.mirrored_pairs()) == 6
    assert len({item.probe_surface_family for item in dataset.observations}) == 6
    assert len(dataset.abstract_conditions) == 2
    assert len(dataset.policy_profiles) == 2
    assert len(dataset.history_condition_bindings) == 48

    positive_outcomes = set(dataset.positive_outcomes)
    for observation in dataset.observations:
        assert len(observation.histories) == 4
        assert len({item.surface_family for item in observation.histories}) == 4
        assert observation.probe_surface_family not in {
            item.surface_family for item in observation.histories
        }
        action_counts = Counter(item.assistant_action for item in observation.histories)
        assert action_counts == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: 2,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: 2,
        }
        for action in (
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        ):
            polarities = {
                item.typed_outcome in positive_outcomes
                for item in observation.histories
                if item.assistant_action is action
            }
            assert polarities == {False, True}


def test_v2_requires_conditioned_policy_transfer_and_mirrored_flip() -> None:
    dataset = load_relationship_transfer_dataset(
        package_name=RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    )
    policies = {item.policy_id: item for item in dataset.policy_profiles}
    history_conditions = dict(dataset.history_condition_bindings)
    positive_outcomes = set(dataset.positive_outcomes)

    for _pair_id, members in dataset.mirrored_pairs():
        assert len({item[0].current_input.encode("utf-8") for item in members}) == 1
        assert len({item[1].probe_condition_id for item in members}) == 1
        assert len({item[1].policy_id for item in members}) == 2
        assert {item[1].preferred_action for item in members} == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        }
        for observation, dynamic in members:
            assert dynamic.policy_id is not None
            assert dynamic.probe_condition_id is not None
            policy = policies[dynamic.policy_id]
            assert policy.action_for(dynamic.probe_condition_id) is dynamic.preferred_action
            grouped: dict[str, list[object]] = {}
            for history in observation.histories:
                condition_id = history_conditions[history.event_id]
                grouped.setdefault(condition_id, []).append(history)
                expected_action = policy.action_for(condition_id)
                assert (
                    history.assistant_action is expected_action
                ) == (history.typed_outcome in positive_outcomes)
            assert {len(items) for items in grouped.values()} == {2}


def test_v2_sut_payload_excludes_condition_policy_and_truth() -> None:
    dataset = load_relationship_transfer_dataset(
        package_name=RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    )
    sealed_tokens = {
        item.condition_id for item in dataset.abstract_conditions
    } | {
        item.policy_id for item in dataset.policy_profiles
    }
    forbidden_keys = {
        "condition_id",
        "history_condition_bindings",
        "policy_id",
        "preferred_action",
        "probe_condition_id",
    }
    for observation in dataset.observations:
        payload = observation.to_sut_payload()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        assert payload["schema_version"] == "relationship-transfer-dataset.v2"
        assert not any(key in encoded for key in forbidden_keys)
        assert not any(token in encoded for token in sealed_tokens)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("outcome_polarity", "outcome polarity"),
        ("probe_policy", "probe condition"),
        ("surface_copy", "probe family"),
    ),
)
def test_v2_loader_rejects_shortcut_or_lineage_mutation(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    root = tmp_path / "relationship_transfer_v2"
    shutil.copytree(_v2_root(), root)
    rendered_path = root / "rendered_observations.json"
    truth_path = root / "generator_truth.json"
    rendered = json.loads(rendered_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if mutation == "outcome_polarity":
        rendered["scenes"][0]["histories"][0]["typed_outcome"] = "helped"
        rendered_path.write_text(
            json.dumps(rendered, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif mutation == "probe_policy":
        truth["dynamics"][0]["probe_condition_id"] = (
            "latent_condition_agency_pressure_v2"
        )
        truth_path.write_text(
            json.dumps(truth, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif mutation == "surface_copy":
        rendered["scenes"][0]["histories"][0]["surface_family"] = "work"
        rendered_path.write_text(
            json.dumps(rendered, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        raise AssertionError(f"unknown mutation {mutation}")
    with pytest.raises(ValueError, match=message):
        load_relationship_transfer_dataset(root)


def test_relationship_transfer_v2_scenario_package_contract() -> None:
    root = _v2_root()
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    ssot = json.loads((root / "ssot_fragment.json").read_text(encoding="utf-8"))
    scenes = yaml.safe_load((root / "scenes.yaml").read_text(encoding="utf-8"))
    suite = yaml.safe_load((root / "test_suite.yaml").read_text(encoding="utf-8"))
    prereg = json.loads((root / "prereg_template.json").read_text(encoding="utf-8"))

    assert re.fullmatch(r"[a-z][a-z0-9_]*", manifest["name"])
    assert manifest["name"] == RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    assert len(manifest["explanation"]) >= 200
    assert manifest["owner_contract"]["owner"] == "lifeform-domain-emogpt.lab"
    assert manifest["owner_contract"]["runtime_owner_policy"] == (
        "no_new_runtime_owner_or_slot"
    )
    for relative in (*manifest["components"].values(), *manifest["lab_artifacts"].values()):
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

    assert len(scenes["scenes"]) == 12
    assert len({item["mirror_group"] for item in scenes["scenes"]}) == 6
    assert len({item["probe_surface_family"] for item in scenes["scenes"]}) == 6
    assert "embedding" in scenes["semantic_routing"]["method"]
    assert "keyword_dictionary" in scenes["semantic_routing"]["forbidden"]
    assert "global_action_outcome_majority" in scenes["semantic_routing"]["forbidden"]

    assert len(suite["routing_tests"]) >= 6
    assert any(item["case_type"] == "negative" for item in suite["routing_tests"])
    assert len(suite["llm_evaluation"]["semantic_coherence"]) >= 3
    assert "keyword_to_route_dictionary" in suite["routing_policy"]["forbidden_methods"]
    assert prereg["schema_version"] == "relationship-lab-prereg.v6"
    assert prereg["status"] == "development_scenario_candidate_not_formal"
    assert prereg["development_lineage"]["dataset_fingerprint"] == _V2_FINGERPRINT
    assert not prereg["development_lineage"]["default_v1_consumer_switched"]
    assert prereg["scenario_admission_contract"]["global_action_tally_expected_result"] == "tie"
    assert prereg["next_consumer_packet_requirements"]["rag_top_k"] == 4
    assert prereg["next_consumer_packet_requirements"]["all_four_histories_must_be_available"]
    assert not prereg["next_consumer_packet_requirements"]["formal_hidden_test_opened"]


def test_v2_package_path_stays_inside_relationship_vertical() -> None:
    root = _v2_root()
    expected_parent = Path(__file__).resolve().parents[1] / "src/lifeform_domain_emogpt"
    assert root.is_relative_to(expected_parent)
