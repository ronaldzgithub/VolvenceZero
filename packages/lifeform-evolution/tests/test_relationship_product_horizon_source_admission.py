from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    build_relationship_product_horizon_evaluator_bundle,
)
from lifeform_domain_emogpt.relationship_action_contracts import RELATIONSHIP_ACTIONS
from lifeform_evolution.relationship_product_horizon_source_admission import (
    HORIZON_SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION,
    HORIZON_SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION,
    HORIZON_SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION,
    build_relationship_product_horizon_source_action_commitment,
    load_relationship_product_horizon_source_admission_protocol,
    materialize_relationship_product_horizon_source_admission,
    relationship_product_horizon_source_admission_protocol_path,
    validate_relationship_product_horizon_source_admission,
)
from lifeform_evolution.relationship_product_source_admission import (
    load_relationship_product_source_admission_protocol,
)


_TEST_IMPLEMENTATION_COMMIT = "0" * 40


@pytest.fixture(scope="module")
def admitted_root(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    root = tmp_path_factory.mktemp("source-v4-admission") / "artifact"
    manifest = materialize_relationship_product_horizon_source_admission(
        root,
        implementation_git_commit=_TEST_IMPLEMENTATION_COMMIT,
    )
    return root, manifest


def test_source_v4_admission_protocol_freezes_input_only_boundary() -> None:
    protocol_path = relationship_product_horizon_source_admission_protocol_path()
    protocol, protocol_id = load_relationship_product_horizon_source_admission_protocol()

    assert len(protocol_path.read_bytes()) == 5_666
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == (
        "81822b796ae0acdd990ba5139e7157e5f6871e43336acbd96920c361a28c7188"
    )
    assert protocol["schema_version"] == HORIZON_SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION
    assert protocol_id == "b3988b21200bf0b9239de08ac683d74b589b3cb6f75a62592373bf8a358f2102"
    assert protocol["source"] == {
        "schema_version": "relationship-product-horizon-source.v4",
        "protocol_raw_sha256": "29162022c011b311369816f74087b16c9b262dc50a7a7434227d150b3b2e8bd3",
        "protocol_raw_bytes": 4_977,
        "protocol_id": "dbf0526299558842b52f293875520e4524afff0e5a1636ab1fd10da9f74d1d91",
        "public_plan_sha256": "f46336a95aeac2c7be60616388a31333fb45e46cd18320dae2f9bd25179a86d6",
        "sealed_bundle_sha256": "51900ec798a8afdbcdab7547b1ad2c7c22fc098122900ec96050a5931c308936",
    }
    assert protocol["inventory"] == {
        "root_count": 112,
        "onboarding_session_count": 448,
        "collection_decision_count": 896,
        "evaluation_decision_count": 4_480,
        "decision_count": 5_376,
        "action_order": [action.value for action in RELATIONSHIP_ACTIONS],
        "action_counterfactual_commitment_count": 16_128,
    }
    assert protocol["replay"] == {
        "materialization_count": 1,
        "pre_manifest_full_rebuild_required": True,
        "consumer_validate_existing_required": True,
        "byte_exact_required": True,
        "process_independence_security_claim": False,
    }
    assert protocol["claims"]["campaign_input_admission_may_be_derived"] is True
    assert protocol["claims"]["campaign_execution_authorized"] is False
    assert protocol["claims"]["model_output_count"] == 0
    assert protocol["claims"]["cuda_invocation_count"] == 0


def test_source_v4_admission_persists_compact_complete_branch_index(
    admitted_root: tuple[Path, dict[str, object]],
) -> None:
    root, manifest = admitted_root
    files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    assert files == [
        "manifest.json",
        "protocol.json",
        "public/source_plan.json",
        "sealed/action_counterfactual_commitment_index.json",
        "sealed/evaluator_bundle.json",
        "source/source_protocol.json",
    ]
    commitments = json.loads(
        (root / "sealed/action_counterfactual_commitment_index.json").read_text(encoding="utf-8")
    )
    assert commitments["schema_version"] == HORIZON_SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION
    assert commitments["decision_count"] == 5_376
    assert commitments["commitment_count"] == 16_128
    assert commitments["action_order"] == [action.value for action in RELATIONSHIP_ACTIONS]
    assert len(commitments["decision_branch_commitments"]) == 5_376
    commitment_ids: set[str] = set()
    decision_ids: set[str] = set()
    for row in commitments["decision_branch_commitments"]:
        assert set(row) == {"decision_id", "branches"}
        assert row["decision_id"] not in decision_ids
        decision_ids.add(row["decision_id"])
        assert [branch["selected_action_id"] for branch in row["branches"]] == [
            action.value for action in RELATIONSHIP_ACTIONS
        ]
        for branch in row["branches"]:
            assert set(branch) == {"selected_action_id", "commitment_id"}
            assert branch["commitment_id"] not in commitment_ids
            commitment_ids.add(branch["commitment_id"])
    assert len(decision_ids) == 5_376
    assert len(commitment_ids) == 16_128
    evaluator = build_relationship_product_horizon_evaluator_bundle()
    index_by_decision = {
        row["decision_id"]: {
            branch["selected_action_id"]: branch["commitment_id"]
            for branch in row["branches"]
        }
        for row in commitments["decision_branch_commitments"]
    }
    for decision in (
        evaluator.decision_sessions[0],
        evaluator.decision_sessions[16],
        evaluator.decision_sessions[32],
        evaluator.decision_sessions[-1],
    ):
        for action in RELATIONSHIP_ACTIONS:
            commitment = build_relationship_product_horizon_source_action_commitment(
                evaluator,
                subject_id=decision.subject_id,
                decision_id=decision.decision_id,
                action=action,
            )
            assert list(commitment["preimage"]) == commitments["commitment_preimage_fields"]
            assert sha256_json(commitment["preimage"]) == commitment["commitment_id"]
            assert index_by_decision[decision.decision_id][action.value] == commitment["commitment_id"]
    assert manifest["schema_version"] == HORIZON_SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION
    assert manifest["segment_decision_counts"] == {
        "correction": 896,
        "matched_collection": 896,
        "mixed_stress": 896,
        "post_correction": 896,
        "post_reversal": 896,
        "return_after_gap": 896,
    }
    assert manifest["claims"]["campaign_input_admitted"] is True
    assert manifest["claims"]["pre_manifest_byte_rebuild_verified"] is True
    assert manifest["claims"]["action_branch_coverage_complete"] is True
    assert manifest["claims"]["campaign_execution_authorized"] is False
    assert manifest["claims"]["reader_input_materialized"] is False
    assert manifest["claims"]["theta0_materialized"] is False


def test_source_v4_admission_validate_existing_is_exact_and_read_only(
    admitted_root: tuple[Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    root, manifest = admitted_root
    _, protocol_id = load_relationship_product_horizon_source_admission_protocol()
    before = {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    validated = validate_relationship_product_horizon_source_admission(
        root,
        expected_protocol_id=protocol_id,
        expected_artifact_id=str(manifest["artifact_id"]),
    )
    assert validated == manifest
    after = {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    with pytest.raises(FileExistsError, match="create-only"):
        materialize_relationship_product_horizon_source_admission(
            root,
            implementation_git_commit=_TEST_IMPLEMENTATION_COMMIT,
        )
    with pytest.raises(ValueError, match="external expected artifact"):
        validate_relationship_product_horizon_source_admission(
            root,
            expected_protocol_id=protocol_id,
            expected_artifact_id="f" * 64,
        )
    with pytest.raises(ValueError, match="external expected protocol"):
        validate_relationship_product_horizon_source_admission(
            root,
            expected_protocol_id="f" * 64,
            expected_artifact_id=str(manifest["artifact_id"]),
        )

    extra = root / "unexpected.json"
    extra.write_bytes(b"{}\n")
    try:
        with pytest.raises(ValueError, match="file inventory drifted"):
            validate_relationship_product_horizon_source_admission(
                root,
                expected_protocol_id=protocol_id,
                expected_artifact_id=str(manifest["artifact_id"]),
            )
    finally:
        extra.unlink()

    tampered = tmp_path / "tampered"
    shutil.copytree(root, tampered)
    target = tampered / "sealed/action_counterfactual_commitment_index.json"
    raw = target.read_bytes()
    target.write_bytes(raw.replace(b'"commitment_count":16128', b'"commitment_count":16127', 1))
    with pytest.raises(ValueError, match="byte drifted"):
        validate_relationship_product_horizon_source_admission(
            tampered,
            expected_protocol_id=protocol_id,
            expected_artifact_id=str(manifest["artifact_id"]),
        )


def test_source_v4_admission_protocol_fails_loudly_on_claim_or_byte_drift(
    tmp_path: Path,
) -> None:
    canonical_path = relationship_product_horizon_source_admission_protocol_path()
    raw = canonical_path.read_bytes()
    payload = json.loads(raw)
    payload["claims"]["campaign_execution_authorized"] = True
    drifted = tmp_path / "drifted.json"
    drifted.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="claim ceiling drifted"):
        load_relationship_product_horizon_source_admission_protocol(drifted)

    drifted.write_bytes(raw.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="LF-only"):
        load_relationship_product_horizon_source_admission_protocol(drifted)

    drifted.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="exactly one LF"):
        load_relationship_product_horizon_source_admission_protocol(drifted)

    duplicate = raw.replace(
        b'{\n  "schema_version":',
        b'{\n  "schema_version": "duplicate",\n  "schema_version":',
        1,
    )
    drifted.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_relationship_product_horizon_source_admission_protocol(drifted)


def test_source_v4_lane_does_not_change_frozen_source_v3_admission() -> None:
    protocol, protocol_id = load_relationship_product_source_admission_protocol()

    assert protocol_id == "98d51d845fd0c5753e401b2a63ad71f4acfef0eba4db45982863f6eb67526338"
    assert protocol["source"]["schema_version"] == "relationship-product-pilot-source.v3"
    assert protocol["inventory"]["action_counterfactual_commitment_count"] == 576
