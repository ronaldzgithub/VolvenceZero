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
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RelationshipAction,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
)


_V1_FINGERPRINT = "953b0ee3483846e4aac876b0b1e93d58a4c8fb705e1a79db1df093be463e866a"
_V2_FINGERPRINT = "d8e002d6d529476bf29622d4872afb0b1d7fec9d9c2e5942ecb830c8428b660b"
_V3_FINGERPRINT = "35b8c46e6fd5810779aff38ed935d8c4f0741bf7d496d2e3eec85f93fbf2134f"
_P1E_REPORT_ID = "232afebb56afb5e457af3d7ca4ccfc560cc417447defcb6d265263085fad8693"
_PUBLIC_CONTRACT_ID = "8ba8a6788d35e959c4a6fa42d31f54baa7d5e1ba48f52603e4bec510232d3cbb"
_P1F_REPORT_ID = "a231e2096b2c4b5fcf3e8b36fd099d0955ce2e355e793d38f5ed8e87a047ecbd"


def _v3_root() -> Path:
    return relationship_transfer_package_dir(RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME)


def test_v3_is_explicit_and_preserves_v1_v2_lineage() -> None:
    v1 = load_relationship_transfer_dataset(package_name=RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME)
    v2 = load_relationship_transfer_dataset(package_name=RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME)
    v3 = load_relationship_transfer_dataset(package_name=RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME)
    inferred_v3 = load_relationship_transfer_dataset(_v3_root())
    assert v1.dataset_fingerprint == _V1_FINGERPRINT
    assert v2.dataset_fingerprint == _V2_FINGERPRINT
    assert v3.dataset_fingerprint == _V3_FINGERPRINT
    assert inferred_v3.dataset_fingerprint == v3.dataset_fingerprint
    assert len({v1.dataset_fingerprint, v2.dataset_fingerprint, v3.dataset_fingerprint}) == 3
    assert load_relationship_transfer_dataset().package_name == (RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME)


def test_v3_public_evidence_contract_is_frozen_before_qwen_output() -> None:
    dataset = load_relationship_transfer_dataset(package_name=RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME)
    contract = dataset.public_evidence_contract
    assert contract is not None
    assert contract.contract_sha256 == _PUBLIC_CONTRACT_ID
    assert contract.source_p1e_report_artifact_id == _P1E_REPORT_ID
    assert contract.source_required_verdict == "rewrite_public_evidence_contract"
    assert contract.history_text_fields == ("user_utterance", "user_reaction")
    assert contract.history_text_joiner == "\n"
    assert contract.probe_text_fields == ("current_input",)
    assert contract.semantic_auditor_version == ("relationship-public-evidence-auditor.v1")
    assert contract.semantic_similarity == "cosine"
    assert contract.semantic_audit_embedder == "bge-m3"
    assert contract.semantic_audit_model_source == "BAAI/bge-m3"
    assert contract.score_precision_decimals == 12
    assert contract.top1_tie_policy == "fail_expected_anchor"
    assert contract.required_evidence_units == 60
    assert contract.required_top1_accuracy == 1.0
    assert contract.minimum_correct_anchor_margin == 0.02
    assert contract.minimum_mean_correct_anchor_margin == 0.07
    assert contract.human_anchor_status == "pending_before_formal"


def test_v3_keeps_compositional_balance_and_truth_isolation() -> None:
    dataset = load_relationship_transfer_dataset(package_name=RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME)
    assert len(dataset.observations) == 12
    assert len(dataset.mirrored_pairs()) == 6
    assert len(dataset.history_condition_bindings) == 48
    positive_outcomes = set(dataset.positive_outcomes)
    sealed_tokens = {condition.condition_id for condition in dataset.abstract_conditions} | {
        policy.policy_id for policy in dataset.policy_profiles
    }
    for _pair_id, members in dataset.mirrored_pairs():
        assert len({observation.current_input for observation, _dynamic in members}) == 1
        assert len({dynamic.probe_condition_id for _observation, dynamic in members}) == 1
        assert {dynamic.preferred_action for _observation, dynamic in members} == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        }
    for observation in dataset.observations:
        assert len(observation.histories) == 4
        assert observation.probe_surface_family not in {history.surface_family for history in observation.histories}
        assert Counter(history.assistant_action for history in observation.histories) == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: 2,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: 2,
        }
        for action in (
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        ):
            polarities = {
                history.typed_outcome in positive_outcomes
                for history in observation.histories
                if history.assistant_action is action
            }
            assert polarities == {False, True}
        sut_payload = json.dumps(
            observation.to_sut_payload(),
            ensure_ascii=False,
            sort_keys=True,
        )
        assert not any(token in sut_payload for token in sealed_tokens)
        assert "preferred_action" not in sut_payload
        assert "probe_condition_id" not in sut_payload


@pytest.mark.parametrize(
    ("mutation", "expected_exception", "message"),
    (
        ("missing_contract", FileNotFoundError, "public_evidence_contract"),
        ("visible_label", ValueError, "condition_name_visible"),
        ("wrong_units", ValueError, "exactly 60"),
        ("claim_broadening", ValueError, "claim_boundary"),
    ),
)
def test_v3_loader_rejects_missing_or_weakened_public_evidence_contract(
    tmp_path: Path,
    mutation: str,
    expected_exception: type[Exception],
    message: str,
) -> None:
    root = tmp_path / "relationship_transfer_v3"
    shutil.copytree(_v3_root(), root)
    contract_path = root / "public_evidence_contract.json"
    if mutation == "missing_contract":
        contract_path.unlink()
    else:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
        if mutation == "visible_label":
            raw["public_language_contract"]["condition_name_visible_to_sut"] = True
        elif mutation == "wrong_units":
            raw["semantic_legibility_audit"]["required_evidence_units"] = 59
        elif mutation == "claim_broadening":
            raw["claim_boundary"] = "This proves the complete system."
        else:
            raise AssertionError(f"unknown mutation {mutation}")
        contract_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(expected_exception, match=message):
        load_relationship_transfer_dataset(root)


def test_v3_fingerprint_binds_public_language_contract(tmp_path: Path) -> None:
    root = tmp_path / "relationship_transfer_v3"
    shutil.copytree(_v3_root(), root)
    contract_path = root / "public_evidence_contract.json"
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    raw["trigger"]["p1e_report_artifact_id"] = "0" * 64
    contract_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mutated = load_relationship_transfer_dataset(root)
    assert mutated.dataset_fingerprint != _V3_FINGERPRINT
    assert mutated.public_evidence_contract is not None
    assert mutated.public_evidence_contract.contract_sha256 != _PUBLIC_CONTRACT_ID


def test_relationship_transfer_v3_scenario_package_contract() -> None:
    root = _v3_root()
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    ssot = json.loads((root / "ssot_fragment.json").read_text(encoding="utf-8"))
    scenes = yaml.safe_load((root / "scenes.yaml").read_text(encoding="utf-8"))
    suite = yaml.safe_load((root / "test_suite.yaml").read_text(encoding="utf-8"))
    prereg = json.loads((root / "prereg_template.json").read_text(encoding="utf-8"))

    assert re.fullmatch(r"[a-z][a-z0-9_]*", manifest["name"])
    assert manifest["name"] == RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME
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
        assert [phase["phase_order"] for phase in arc["phases"]] == list(range(len(arc["phases"])))
        for phase in arc["phases"]:
            referenced_sub_goals.update(phase["sub_goal_refs"])
    all_sub_goals = {sub_goal["sub_goal_id"] for path in paths.values() for sub_goal in path["sub_goals"]}
    assert referenced_paths == set(paths)
    assert referenced_sub_goals == all_sub_goals

    assert len(scenes["scenes"]) == 12
    assert len({item["mirror_group"] for item in scenes["scenes"]}) == 6
    assert "embedding" in scenes["semantic_routing"]["method"]
    assert "keyword_dictionary" in scenes["semantic_routing"]["forbidden"]
    assert len(suite["routing_tests"]) >= 6
    assert any(item["case_type"] == "negative" for item in suite["routing_tests"])
    assert len(suite["llm_evaluation"]["semantic_coherence"]) >= 3
    assert "keyword_to_route_dictionary" in (suite["routing_policy"]["forbidden_methods"])
    assert prereg["schema_version"] == "relationship-lab-prereg.v8"
    assert prereg["development_lineage"]["dataset_fingerprint"] == _V3_FINGERPRINT
    assert prereg["development_lineage"]["p1f_public_evidence_audit_report_artifact_id"] == _P1F_REPORT_ID
    assert prereg["triggered_by_frozen_p1e_verdict"]["p1e_report_artifact_id"] == (_P1E_REPORT_ID)
    assert prereg["public_evidence_admission_contract"]["contract_sha256"] == (_PUBLIC_CONTRACT_ID)
    assert prereg["next_consumer_packet_requirements"]["packet_id"] == (
        "relationship_p1g_v3_consumer_freeze_and_qualification"
    )
    assert prereg["next_consumer_packet_requirements"]["conditioned_readout_prompt_must_be_frozen_before_model_output"]
    assert not prereg["next_consumer_packet_requirements"]["formal_hidden_test_opened"]


def test_v3_package_path_stays_inside_relationship_vertical() -> None:
    expected_parent = Path(__file__).resolve().parents[1] / "src/lifeform_domain_emogpt"
    assert _v3_root().is_relative_to(expected_parent)
