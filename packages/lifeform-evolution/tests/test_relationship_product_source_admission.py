from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from collections import defaultdict

import pytest

from lifeform_domain_emogpt.lab.contracts import canonical_json
from lifeform_domain_emogpt.lab.relationship_product_pilot_source_v2 import (
    build_relationship_product_pilot_environment,
    build_relationship_product_pilot_evaluator_bundle,
    load_relationship_product_pilot_source_protocol,
)
from lifeform_domain_emogpt.relationship_action_contracts import RELATIONSHIP_ACTIONS
from lifeform_evolution.relationship_product_source_admission import (
    build_relationship_product_source_action_commitments,
    build_relationship_product_source_admission_materialization,
    finalize_relationship_product_source_admission,
    load_relationship_product_source_admission_protocol,
    materialize_relationship_product_source_admission,
    relationship_product_source_admission_protocol_path,
    validate_relationship_product_source_admission,
    validate_relationship_product_source_admission_materialization,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "run_relationship_product_source_admission.py"
_EXPECTED_PROTOCOL_ID = "98d51d845fd0c5753e401b2a63ad71f4acfef0eba4db45982863f6eb67526338"
_FORBIDDEN_PUBLIC_KEYS = {
    "condition_id",
    "policy_id",
    "preferred_action_id",
    "environment_seed",
    "subject_seed",
    "typed_outcome_id",
    "action_counterfactual_commitments",
}


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for item in value.values()
            for key in _all_keys(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _run_child(*args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(_SCRIPT), *args],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait(process: subprocess.Popen[str]) -> str:
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 0, stderr
    return stdout


def test_protocol_pins_source_inventory_randomness_and_claim_ceiling() -> None:
    protocol, protocol_id = load_relationship_product_source_admission_protocol()

    assert relationship_product_source_admission_protocol_path().name == (
        "relationship_product_source_v3_campaign_admission_v1.json"
    )
    assert protocol_id == _EXPECTED_PROTOCOL_ID
    assert protocol["evidence_tier"] == "development"
    assert protocol["source"] == {
        "schema_version": "relationship-product-pilot-source.v3",
        "protocol_raw_sha256": "09b0fe4adad95a23dda06570e6720381ebac46f1ebacfb13b57924671f45b22f",
        "protocol_raw_bytes": 1410,
        "protocol_sha256": "d17a49edc5dc549a325648f9c430340d0d3e7cabe634ba339506bd7e56b24be8",
        "public_plan_sha256": "4f5e8c1508b533d9d64434c5e3ba6a8e9b95814b070fec21ff2d03841746bb05",
        "sealed_bundle_sha256": "7ddec3e160967381a51e6ffdf2362924619249e8b43ffc0a531aaf2015cdc18d",
    }
    assert protocol["inventory"] == {
        "subject_count": 8,
        "onboarding_session_count": 32,
        "decision_count": 192,
        "action_order": [action.value for action in RELATIONSHIP_ACTIONS],
        "action_counterfactual_commitment_count": 576,
    }
    assert protocol["randomness"]["selected_action_is_in_draw_hash"] is True
    assert protocol["randomness"]["common_random_number_design"] is False
    assert protocol["claims"]["campaign_input_admission_may_be_derived"] is True
    assert protocol["claims"]["campaign_execution_authorized"] is False
    assert protocol["claims"]["formal_evidence_authorized"] is False
    assert protocol["claims"]["four_able_complete"] is False
    assert protocol["claims"]["fresh_process_independence_proven"] is False
    assert protocol["claims"]["model_output_count"] == 0


def test_materialization_has_exact_public_sealed_inventory_without_truth_leakage() -> None:
    files = build_relationship_product_source_admission_materialization()
    assert tuple(sorted(files)) == (
        "manifest.json",
        "protocol.json",
        "public/source_plan.json",
        "sealed/action_counterfactual_commitments.json",
        "sealed/evaluator_bundle.json",
    )
    public = json.loads(files["public/source_plan.json"])
    evaluator = json.loads(files["sealed/evaluator_bundle.json"])
    commitments = json.loads(files["sealed/action_counterfactual_commitments.json"])
    manifest = json.loads(files["manifest.json"])

    assert len(public["subjects"]) == 8
    assert len(evaluator["onboarding_sessions"]) == 32
    assert len(evaluator["decision_sessions"]) == 192
    assert commitments["decision_count"] == 192
    assert commitments["commitment_count"] == 576
    assert len(commitments["commitments"]) == 576
    assert not (_all_keys(public) & _FORBIDDEN_PUBLIC_KEYS)
    assert manifest["claims"]["materialization_complete"] is True
    assert manifest["claims"]["campaign_input_admitted"] is False
    assert manifest["claims"]["campaign_execution_authorized"] is False


def test_every_action_commitment_replays_through_existing_environment_owner() -> None:
    source = load_relationship_product_pilot_source_protocol()
    evaluator = build_relationship_product_pilot_evaluator_bundle(source)
    payload = build_relationship_product_source_action_commitments(evaluator)
    decisions = {item.decision_id: item for item in evaluator.decision_sessions}
    environments = {
        subject_id: build_relationship_product_pilot_environment(
            evaluator,
            subject_id=subject_id,
        )
        for subject_id in {item.subject_id for item in evaluator.decision_sessions}
    }
    actions_by_decision: defaultdict[str, set[str]] = defaultdict(set)
    commitment_ids: set[str] = set()
    for row in payload["commitments"]:
        decision = decisions[row["decision_id"]]
        action = next(
            candidate
            for candidate in RELATIONSHIP_ACTIONS
            if candidate.value == row["selected_action_id"]
        )
        settled = environments[decision.subject_id].settle(
            scene_id=decision.scene_id,
            decision_id=decision.decision_id,
            action=action,
            seed=decision.environment_seed,
        )
        actions_by_decision[decision.decision_id].add(action.value)
        commitment_ids.add(row["commitment_id"])
        assert row["outcome_distribution"] == settled.outcome_distribution.to_payload()
        assert row["deterministic_draw"] == settled.deterministic_draw
        assert row["typed_outcome_id"] == settled.typed_outcome.value
        assert row["rendered_user_reaction"] == settled.rendered_user_reaction
        assert row["environment_evidence_ref"] == settled.environment_evidence_ref

    assert len(actions_by_decision) == 192
    assert all(actions == {action.value for action in RELATIONSHIP_ACTIONS} for actions in actions_by_decision.values())
    assert len(commitment_ids) == 576


def test_two_fresh_workers_and_third_comparator_derive_input_only_admission(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "admission"
    root.mkdir()
    replay_a = root / "replay_a"
    replay_b = root / "replay_b"
    worker_a = _run_child(
        "worker",
        "--output-dir",
        str(replay_a),
        "--expected-protocol-id",
        _EXPECTED_PROTOCOL_ID,
    )
    worker_b = _run_child(
        "worker",
        "--output-dir",
        str(replay_b),
        "--expected-protocol-id",
        _EXPECTED_PROTOCOL_ID,
    )
    _wait(worker_a)
    _wait(worker_b)
    comparator = _run_child(
        "compare",
        "--output-dir",
        str(root),
        "--expected-protocol-id",
        _EXPECTED_PROTOCOL_ID,
        "--worker-a-pid",
        str(worker_a.pid),
        "--worker-b-pid",
        str(worker_b.pid),
    )
    _wait(comparator)
    manifest = finalize_relationship_product_source_admission(
        root,
        implementation_git_commit="a" * 40,
    )
    validated = validate_relationship_product_source_admission(
        root,
        expected_protocol_id=_EXPECTED_PROTOCOL_ID,
    )
    comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))

    assert manifest == validated
    assert len({worker_a.pid, worker_b.pid, comparison["comparator_pid"]}) == 3
    assert comparison["byte_exact"] is True
    assert comparison["process_ids_self_reported"] is True
    assert comparison["process_independence_proven"] is False
    assert comparison["claims"]["campaign_input_admitted"] is True
    assert comparison["claims"]["campaign_execution_authorized"] is False
    assert comparison["claims"]["campaign_runtime_order_verified"] is False
    assert comparison["claims"]["formal_evidence_authorized"] is False
    assert comparison["claims"]["four_able_complete"] is False
    assert comparison["claims"]["fresh_process_independence_proven"] is False
    assert manifest["action_counterfactual_commitment_count"] == 576


def test_create_only_and_commitment_tamper_fail_loudly(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "one-root"
    materialize_relationship_product_source_admission(root)
    with pytest.raises(FileExistsError, match="create-only"):
        materialize_relationship_product_source_admission(root)

    commitments_path = root / "sealed" / "action_counterfactual_commitments.json"
    payload = json.loads(commitments_path.read_text(encoding="utf-8"))
    payload["commitments"][0]["typed_outcome_id"] = "missed"
    commitments_path.write_text(
        canonical_json(payload) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="byte drifted"):
        validate_relationship_product_source_admission_materialization(
            root,
            expected_protocol_id=_EXPECTED_PROTOCOL_ID,
        )


def test_protocol_claim_or_source_closure_drift_fails_before_materialization(
    tmp_path: pathlib.Path,
) -> None:
    payload = json.loads(
        relationship_product_source_admission_protocol_path().read_text(encoding="utf-8")
    )
    payload["claims"]["model_output_count"] = 1
    bad_claim = tmp_path / "bad-claim.json"
    bad_claim.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="claim ceiling drifted"):
        load_relationship_product_source_admission_protocol(bad_claim)

    payload = json.loads(
        relationship_product_source_admission_protocol_path().read_text(encoding="utf-8")
    )
    payload["direct_execution_closure"][0]["raw_sha256"] = "0" * 64
    bad_closure = tmp_path / "bad-closure.json"
    bad_closure.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="closure SHA-256 drifted"):
        load_relationship_product_source_admission_protocol(bad_closure)


def test_comparator_pid_must_be_distinct_from_workers(tmp_path: pathlib.Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    materialize_relationship_product_source_admission(root_a)
    materialize_relationship_product_source_admission(root_b)
    from lifeform_evolution.relationship_product_source_admission import (
        build_relationship_product_source_admission_comparison,
    )

    with pytest.raises(ValueError, match="distinct comparator"):
        build_relationship_product_source_admission_comparison(
            root_a,
            root_b,
            expected_protocol_id=_EXPECTED_PROTOCOL_ID,
            worker_a_pid=os.getpid(),
            worker_b_pid=os.getpid() + 1,
            comparator_pid=os.getpid(),
        )
