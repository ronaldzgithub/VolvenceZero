from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from lifeform_evolution.relationship_product_horizon_source_v5_admission import (
    load_relationship_product_horizon_source_v5_admission_protocol,
    materialize_relationship_product_horizon_source_v5_admission,
    validate_relationship_product_horizon_source_v5_admission,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_ROOT = _REPO_ROOT / "artifacts" / "relationship_lab"
_SOURCE_V3_ADMISSION_ROOT = (
    _ARTIFACT_ROOT
    / "relationship_product_source_v3_campaign_admission_20260826_p98d51d845fd0"
)
_SOURCE_V4_ADMISSION_ROOT = (
    _ARTIFACT_ROOT
    / "relationship_product_horizon_source_v4_admission_20260826_b3988b21"
)
_DEVELOPMENT_READER_ROOT = (
    _ARTIFACT_ROOT
    / "relationship_product_horizon_development_reader_20260826_pa1ea1e30fd7b_ce8272ce7d3da"
)
_ATTEMPT03_TABLE = (
    _ARTIFACT_ROOT
    / "relationship_product_public_bge_m3_v4_weight_runtime_pinned_20260823.json"
)
_ATTEMPT03_REOBSERVATION = (
    _ARTIFACT_ROOT
    / "relationship_product_public_bge_m3_v4_weight_runtime_pinned_20260823.reobservation.json"
)
_QUALIFICATION_V5_TABLE = (
    _ARTIFACT_ROOT
    / "relationship_condition_reader_qualification_v5_windows_cuda_20260825_p723796027a64_c50b54fc1"
    / "prediction_runs"
    / "run-1"
    / "embedding_table.json"
)
_PROTOCOL_ID = "d07bdb21cadc809b605d36e76ebaba45da8334acbdc8d2b6dc68417bb13efcd4"
_LINEAGE_INVENTORY_ID = (
    "3c829dafa3604c33fd03b137a3dd00ce4aa95609adb9564fbf0d8cd0637192c8"
)
_IMPLEMENTATION_COMMIT = "a" * 40
_ACTION_ORDER = (
    "stay_present_without_probe",
    "respect_space_with_return_option",
    "neutral_noop",
)


def _input_kwargs() -> dict[str, Path]:
    return {
        "source_v3_admission_root": _SOURCE_V3_ADMISSION_ROOT,
        "source_v4_admission_root": _SOURCE_V4_ADMISSION_ROOT,
        "development_reader_root": _DEVELOPMENT_READER_ROOT,
        "attempt03_embedding_table_path": _ATTEMPT03_TABLE,
        "attempt03_reobservation_path": _ATTEMPT03_REOBSERVATION,
        "qualification_v5_embedding_table_path": _QUALIFICATION_V5_TABLE,
    }


def _metadata(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module")
def materialized_admission(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, object]]:
    output = tmp_path_factory.mktemp("source-v5-admission") / "artifact"
    manifest = materialize_relationship_product_horizon_source_v5_admission(
        output,
        implementation_git_commit=_IMPLEMENTATION_COMMIT,
        **_input_kwargs(),
    )
    return output, manifest


def test_protocol_freezes_closed_registry_and_claim_ceiling() -> None:
    protocol, protocol_id = (
        load_relationship_product_horizon_source_v5_admission_protocol()
    )

    assert protocol_id == _PROTOCOL_ID
    assert protocol["source"] == {
        "schema_version": "relationship-product-horizon-source.v5",
        "protocol_raw_sha256": (
            "33623a4409d3e5419207340e08bd90462b6b1675afb089433b89fbdb2d859134"
        ),
        "protocol_raw_bytes": 17_908,
        "protocol_id": (
            "71dc200630bf09ee66ce47b9f45460f30ec14cd3ff4e08366c7946497babad9b"
        ),
        "public_view_schema_version": "relationship-product-horizon-public-view.v4",
        "evaluator_schema_version": (
            "relationship-product-horizon-evaluator-bundle.v4"
        ),
        "public_plan_sha256": (
            "bab2ff2291b95d4eef6107a58ebf4575b08490775bee71b2ad99a5b029e09f6c"
        ),
        "sealed_bundle_sha256": (
            "01026dd6ec00c58762e8d62263d3a114a954c04c8e9ff6cbd7e09ed7054ab0ab"
        ),
        "reader_text_occurrence_count": 5_824,
        "reader_text_unique_count": 3_946,
        "reader_text_inventory_sha256": (
            "13794eb5b73c9d9f6d69553278c03b0f3121b5eb450efdff41f85a1987dbb082"
        ),
    }
    registry = protocol["exclusion_inputs"]["adaptive_reader_input_registry"]
    assert registry["union_unique_count"] == 2_135
    assert registry["qualification_v5_table_lineage_only"][
        "exact_subset_of_development_reader_table"
    ] is True
    assert protocol["claims"]["campaign_input_admission_may_be_derived"] is True
    assert protocol["claims"]["exact_disjoint_admission_may_be_derived"] is True
    false_claims = {
        key: value
        for key, value in protocol["claims"].items()
        if key
        not in {
            "campaign_input_admission_may_be_derived",
            "exact_disjoint_admission_may_be_derived",
        }
    }
    assert not any(value for value in false_claims.values())


def test_materialization_closes_full_string_and_branch_inventories(
    materialized_admission: tuple[Path, dict[str, object]],
) -> None:
    root, manifest = materialized_admission
    expected_files = {
        "lineage/exact_disjoint_inventory.json",
        "manifest.json",
        "protocol.json",
        "public/source_plan.json",
        "sealed/action_counterfactual_commitment_index.json",
        "sealed/evaluator_bundle.json",
        "source/source_protocol.json",
    }
    assert {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    } == expected_files
    assert manifest["reader_text_inventory_artifact_id"] == _LINEAGE_INVENTORY_ID
    assert manifest["action_counterfactual_commitment_count"] == 16_128
    assert manifest["claims"]["campaign_input_admitted"] is True
    assert manifest["claims"]["campaign_execution_authorized"] is False
    assert manifest["claims"]["semantic_novelty_established"] is False
    assert manifest["claims"]["learnable_effect"] is False
    assert manifest["claims"]["steerable_effect"] is False

    lineage = json.loads(
        (root / "lineage" / "exact_disjoint_inventory.json").read_bytes()
    )
    assert lineage["artifact_id"] == _LINEAGE_INVENTORY_ID
    assert {
        name: len(lineage[name]["rows"])
        for name in (
            "source_v5",
            "source_v3",
            "source_v4",
            "development_reader_table",
            "attempt03_table",
            "qualification_v5_table_lineage_only",
            "adaptive_reader_union",
        )
    } == {
        "source_v5": 3_946,
        "source_v3": 224,
        "source_v4": 1_881,
        "development_reader_table": 2_109,
        "attempt03_table": 30,
        "qualification_v5_table_lineage_only": 228,
        "adaptive_reader_union": 2_135,
    }
    assert set(lineage["full_string_overlap_counts"].values()) == {0}
    assert set(lineage["subset_relations"].values()) == {True}

    commitments = json.loads(
        (
            root
            / "sealed"
            / "action_counterfactual_commitment_index.json"
        ).read_bytes()
    )
    assert commitments["decision_count"] == 5_376
    assert commitments["commitment_count"] == 16_128
    assert tuple(commitments["action_order"]) == _ACTION_ORDER
    assert all(
        tuple(branch["selected_action_id"] for branch in item["branches"])
        == _ACTION_ORDER
        for item in commitments["decision_branch_commitments"]
    )
    all_commitment_ids = {
        branch["commitment_id"]
        for item in commitments["decision_branch_commitments"]
        for branch in item["branches"]
    }
    assert len(all_commitment_ids) == 16_128


def test_validate_existing_is_read_only_and_requires_external_ids(
    materialized_admission: tuple[Path, dict[str, object]],
) -> None:
    root, manifest = materialized_admission
    before = {
        "output": _metadata(root),
        "source_v3": _metadata(_SOURCE_V3_ADMISSION_ROOT),
        "source_v4": _metadata(_SOURCE_V4_ADMISSION_ROOT),
        "development": _metadata(_DEVELOPMENT_READER_ROOT),
    }
    validated = validate_relationship_product_horizon_source_v5_admission(
        root,
        expected_protocol_id=_PROTOCOL_ID,
        expected_artifact_id=str(manifest["artifact_id"]),
        **_input_kwargs(),
    )
    after = {
        "output": _metadata(root),
        "source_v3": _metadata(_SOURCE_V3_ADMISSION_ROOT),
        "source_v4": _metadata(_SOURCE_V4_ADMISSION_ROOT),
        "development": _metadata(_DEVELOPMENT_READER_ROOT),
    }
    assert validated == manifest
    assert after == before

    with pytest.raises(ValueError, match="external protocol identity"):
        validate_relationship_product_horizon_source_v5_admission(
            root,
            expected_protocol_id="0" * 64,
            expected_artifact_id=str(manifest["artifact_id"]),
            **_input_kwargs(),
        )
    with pytest.raises(ValueError, match="external artifact identity"):
        validate_relationship_product_horizon_source_v5_admission(
            root,
            expected_protocol_id=_PROTOCOL_ID,
            expected_artifact_id="0" * 64,
            **_input_kwargs(),
        )


def test_create_only_and_extra_file_fail_before_semantic_rebuild(
    tmp_path: Path,
    materialized_admission: tuple[Path, dict[str, object]],
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        materialize_relationship_product_horizon_source_v5_admission(
            existing,
            implementation_git_commit=_IMPLEMENTATION_COMMIT,
            **_input_kwargs(),
        )

    source, manifest = materialized_admission
    tampered = tmp_path / "tampered"
    shutil.copytree(source, tampered)
    (tampered / "EXTRA.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory drifted"):
        validate_relationship_product_horizon_source_v5_admission(
            tampered,
            expected_protocol_id=_PROTOCOL_ID,
            expected_artifact_id=str(manifest["artifact_id"]),
            **_input_kwargs(),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows hard-link behavior is host-specific")
def test_validate_existing_rejects_hard_linked_output(
    tmp_path: Path,
    materialized_admission: tuple[Path, dict[str, object]],
) -> None:
    source, manifest = materialized_admission
    tampered = tmp_path / "hardlinked"
    shutil.copytree(source, tampered)
    original = tampered / "protocol.json"
    replacement = tampered / "protocol-hardlink.json"
    os.link(original, replacement)
    with pytest.raises(ValueError, match="hard-linked file"):
        validate_relationship_product_horizon_source_v5_admission(
            tampered,
            expected_protocol_id=_PROTOCOL_ID,
            expected_artifact_id=str(manifest["artifact_id"]),
            **_input_kwargs(),
        )
