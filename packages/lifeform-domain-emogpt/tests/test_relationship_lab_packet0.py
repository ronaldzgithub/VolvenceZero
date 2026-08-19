from __future__ import annotations

import ast
import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from lifeform_domain_emogpt.lab import (
    CandidateOutcomePrediction,
    OutcomeProbability,
    PreActionRelationshipDecision,
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    ReactiveRelationshipEnvironment,
    RelationshipAction,
    RelationshipDecisionTrace,
    RelationshipModelLineage,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
    sha256_json,
)


def _uniform_predictions() -> tuple[CandidateOutcomePrediction, ...]:
    return tuple(
        CandidateOutcomePrediction(
            action_id=action,
            outcomes=tuple(OutcomeProbability(kind, 0.25) for kind in RELATIONSHIP_OUTCOMES),
        )
        for action in RELATIONSHIP_ACTIONS
    )


def _trace() -> RelationshipDecisionTrace:
    dataset = load_relationship_transfer_dataset()
    observation = dataset.observations[0]
    dynamic = dataset.dynamic_for_scene(observation.scene_id)
    environment = ReactiveRelationshipEnvironment(dataset)
    pre_action = PreActionRelationshipDecision(
        decision_id="packet0-test-decision",
        pre_action_timestamp="2026-08-19T00:00:00+00:00",
        candidate_predictions=_uniform_predictions(),
        chosen_action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
        source_snapshot_hashes=(dataset.dataset_fingerprint,),
        lineage=RelationshipModelLineage(
            model_id="packet0-fixture",
            weights_sha256=sha256_json("weights"),
            prompt_sha256=sha256_json("prompt"),
            generation_config_sha256=sha256_json("generation"),
            seed=7,
        ),
    )
    outcome = environment.settle(
        scene_id=observation.scene_id,
        decision_id=pre_action.decision_id,
        action=pre_action.chosen_action_id,
        seed=7,
    )
    return RelationshipDecisionTrace(
        trajectory_sha256=observation.trajectory_sha256,
        user_scope_hash=observation.user_scope_hash,
        scenario_family=observation.probe_surface_family,
        surface_scene_id=observation.scene_id,
        split=dynamic.split,
        sealed_latent_dynamic_id=dynamic.dynamic_id,
        pre_action=pre_action,
        observed_typed_outcome=outcome.typed_outcome,
        outcome_observed_at="2026-08-19T00:00:01+00:00",
        environment_evidence_ref=outcome.environment_evidence_ref,
    )


def test_dataset_has_six_mirrored_pairs_four_families_and_clean_sut_payloads() -> None:
    dataset = load_relationship_transfer_dataset()
    pairs = dataset.mirrored_pairs()
    assert len(pairs) == 6
    assert len(dataset.observations) == 12
    assert {item.probe_surface_family for item in dataset.observations} == {
        "family",
        "friends",
        "intimacy",
        "work",
    }
    for _pair_id, members in pairs:
        assert len(members) == 2
        assert len({item[0].current_input.encode("utf-8") for item in members}) == 1
        assert {item[1].preferred_action for item in members} == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        }
        assert len({item[1].split for item in members}) == 1
        for observation, dynamic in members:
            payload = observation.to_sut_payload()
            encoded = json.dumps(payload, ensure_ascii=False)
            assert "scene_id" not in payload
            assert dynamic.dynamic_id not in encoded
            assert dynamic.outcome_profile_id not in encoded
            assert dynamic.mirror_pair_id not in encoded
            assert "preferred_action" not in encoded
    dataset.assert_no_sut_truth_leakage()


def test_reactive_environment_is_deterministic_and_action_conditional() -> None:
    dataset = load_relationship_transfer_dataset()
    environment = ReactiveRelationshipEnvironment(dataset)
    observation = dataset.observations[0]
    stay = environment.settle(
        scene_id=observation.scene_id,
        decision_id="deterministic-decision",
        action=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
        seed=13,
    )
    repeated = environment.settle(
        scene_id=observation.scene_id,
        decision_id="deterministic-decision",
        action=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
        seed=13,
    )
    space = environment.settle(
        scene_id=observation.scene_id,
        decision_id="deterministic-decision",
        action=RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        seed=13,
    )
    assert stay == repeated
    assert stay.selected_action is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
    assert space.selected_action is RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION
    assert stay.outcome_distribution != space.outcome_distribution
    assert stay.environment_evidence_ref != space.environment_evidence_ref


def test_every_latent_dynamic_has_a_large_preferred_action_effect() -> None:
    dataset = load_relationship_transfer_dataset()
    for observation in dataset.observations:
        dynamic = dataset.dynamic_for_scene(observation.scene_id)
        opposite = (
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION
            if dynamic.preferred_action is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
            else RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
        )
        preferred_distribution = dataset.distribution(
            observation.scene_id,
            dynamic.preferred_action,
        )
        opposite_distribution = dataset.distribution(
            observation.scene_id,
            opposite,
        )
        preferred_positive = sum(preferred_distribution.probability_of(kind) for kind in dataset.positive_outcomes)
        opposite_positive = sum(opposite_distribution.probability_of(kind) for kind in dataset.positive_outcomes)
        assert preferred_positive - opposite_positive >= 0.7


def test_decision_trace_round_trips_and_rejects_tampering() -> None:
    trace = _trace()
    restored = RelationshipDecisionTrace.from_json(trace.to_json())
    assert restored == trace
    assert restored.artifact_id == trace.artifact_id
    with pytest.raises(FrozenInstanceError):
        trace.surface_scene_id = "mutated"  # type: ignore[misc]

    tampered = json.loads(trace.to_json())
    tampered["observed_typed_outcome"] = "over_directive"
    with pytest.raises(ValueError, match="artifact_id does not match"):
        RelationshipDecisionTrace.from_json(json.dumps(tampered))


def test_decision_trace_enforces_bet_then_settle_ordering() -> None:
    trace = _trace()
    with pytest.raises(ValueError, match="strictly after"):
        RelationshipDecisionTrace(
            trajectory_sha256=trace.trajectory_sha256,
            user_scope_hash=trace.user_scope_hash,
            scenario_family=trace.scenario_family,
            surface_scene_id=trace.surface_scene_id,
            split=trace.split,
            sealed_latent_dynamic_id=trace.sealed_latent_dynamic_id,
            pre_action=trace.pre_action,
            observed_typed_outcome=trace.observed_typed_outcome,
            outcome_observed_at=trace.pre_action.pre_action_timestamp,
            environment_evidence_ref=trace.environment_evidence_ref,
        )


def test_prediction_distribution_must_cover_closed_outcome_surface() -> None:
    with pytest.raises(ValueError, match="every relationship outcome"):
        CandidateOutcomePrediction(
            action_id=RelationshipAction.NEUTRAL_NOOP,
            outcomes=(OutcomeProbability(RELATIONSHIP_OUTCOMES[0], 1.0),),
        )


def test_relationship_transfer_scenario_package_contract() -> None:
    root = relationship_transfer_package_dir()
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    ssot = json.loads((root / "ssot_fragment.json").read_text(encoding="utf-8"))
    scenes = yaml.safe_load((root / "scenes.yaml").read_text(encoding="utf-8"))
    suite = yaml.safe_load((root / "test_suite.yaml").read_text(encoding="utf-8"))
    prereg = json.loads((root / "prereg_template.json").read_text(encoding="utf-8"))

    assert re.fullmatch(r"[a-z][a-z0-9_]*", manifest["name"])
    assert manifest["name"] == "relationship_transfer_v1"
    assert len(manifest["explanation"]) >= 200
    assert manifest["components"] == {
        "ssot_fragment": "ssot_fragment.json",
        "scenes": "scenes.yaml",
        "test_suite": "test_suite.yaml",
    }
    assert manifest["owner_contract"]["owner"] == "lifeform-domain-emogpt.lab"
    assert manifest["owner_contract"]["runtime_owner_policy"] == ("no_new_runtime_owner_or_slot")
    for relative in (*manifest["components"].values(), *manifest["lab_artifacts"].values()):
        assert (root / relative).is_file()

    paths = {item["path_id"]: item for item in ssot["paths"]}
    referenced_paths: set[str] = set()
    referenced_sub_goals: set[str] = set()
    for arc in ssot["arc_specs"]:
        referenced_paths.update(arc["path_ids"])
        assert [phase["phase_order"] for phase in arc["phases"]] == list(range(len(arc["phases"])))
        for phase in arc["phases"]:
            referenced_sub_goals.update(phase["sub_goal_refs"])
    assert referenced_paths == set(paths)
    all_sub_goals = {sub_goal["sub_goal_id"] for path in paths.values() for sub_goal in path["sub_goals"]}
    assert referenced_sub_goals == all_sub_goals

    assert len(scenes["scenes"]) == 12
    assert len({item["mirror_group"] for item in scenes["scenes"]}) == 6
    assert {item["probe_surface_family"] for item in scenes["scenes"]} == {
        "family",
        "friends",
        "intimacy",
        "work",
    }
    assert "embedding" in scenes["semantic_routing"]["method"]
    assert "keyword_dictionary" in scenes["semantic_routing"]["forbidden"]

    assert len(suite["routing_tests"]) >= 6
    assert any(item["case_type"] == "negative" for item in suite["routing_tests"])
    assert len(suite["llm_evaluation"]["semantic_coherence"]) >= 3
    assert "keyword_to_route_dictionary" in suite["routing_policy"]["forbidden_methods"]
    assert prereg["status"] == "template_not_frozen"
    assert prereg["schema_version"] == "relationship-lab-prereg.v5"
    assert not prereg["p1c_development_qualification"]["formal_hidden_test_opened"]
    assert set(prereg["p1c_development_qualification"]["allowed_verdicts"]) == {
        "formal_prereg_freeze_candidate",
        "rewrite_public_evidence_contract",
        "version_scenario_dataset_saturated",
    }
    assert prereg["p1b_development_protocol"] == {
        "split": ["train", "validation"],
        "shared_readout_across_contextual_arms": True,
        "readout_fields": [
            "stay_present_without_probe_score",
            "respect_space_with_return_option_score",
        ],
        "readout_value_domain": [-1, 0, 1],
        "typed_action_compiler_version": "relationship-evidence-argmax.v1",
        "rag_top_k": 2,
        "expected_action_visible_before_readout": False,
        "saturation_verdict_above_accuracy": 0.875,
    }
    assert {item["arm_id"] for item in prereg["arms"]} == {
        "stateless",
        "prompt-steelman",
        "rag-steelman",
        "structured-state",
        "volvence-cold",
        "volvence",
        "oracle-concept",
    }
    assert not prereg["reporting_contract"]["finite_prompt_impossibility_claim_allowed"]


def test_package_path_is_inside_relationship_vertical() -> None:
    root = relationship_transfer_package_dir()
    expected_parent = Path(__file__).resolve().parents[1] / "src/lifeform_domain_emogpt"
    assert root.is_relative_to(expected_parent)


def test_companion_product_modules_do_not_import_offline_lab() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src/lifeform_domain_emogpt"
    lab_root = package_root / "lab"
    product_modules = tuple(path for path in package_root.rglob("*.py") if not path.is_relative_to(lab_root))
    for path in product_modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
        assert not any(name.startswith("lifeform_domain_emogpt.lab") for name in imported), path
