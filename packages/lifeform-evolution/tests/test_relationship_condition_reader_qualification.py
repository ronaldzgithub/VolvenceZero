from __future__ import annotations

from collections import Counter
import hashlib
import json
import pathlib

import pytest

from lifeform_evolution.relationship_condition_reader_qualification import (
    RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V1,
    RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V2,
    load_relationship_condition_reader_qualification_protocol,
    prepare_relationship_condition_reader_qualification_preflight,
    relationship_condition_reader_qualification_protocol_path,
    validate_relationship_condition_reader_qualification_preflight,
)


_PROTOCOL_ID = "723796027a64a627f8f858e4499d5956ad43d7c45bbc20e20f7b04fd197c8e6b"
_PROTOCOL_RAW_SHA256 = "fe4ef1efad7b03121ee1ee4f956c2dfc50cbf9dc66449f335221620f8c120dce"


def _read_json(path: pathlib.Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _prepare(tmp_path: pathlib.Path, suffix: str) -> tuple[pathlib.Path, pathlib.Path, dict[str, object]]:
    preflight = tmp_path / f"preflight-{suffix}"
    execution = tmp_path / f"execution-{suffix}"
    result = prepare_relationship_condition_reader_qualification_preflight(
        preflight_root=preflight,
        proposed_execution_root=execution,
    )
    return preflight, execution, dict(result)


def test_qualification_protocol_freezes_honest_model_free_boundary() -> None:
    protocol = load_relationship_condition_reader_qualification_protocol()
    payload = protocol.to_payload()

    assert protocol.schema_version == RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V2
    assert protocol.protocol_id == _PROTOCOL_ID
    assert protocol.raw_sha256 == _PROTOCOL_RAW_SHA256
    assert protocol.raw_bytes == 5658
    assert protocol.training_source.schema_version == "relationship-product-pilot-source.v1"
    assert protocol.challenge_source.schema_version == "relationship-product-pilot-source.v3"
    assert payload["label_crosswalk"] == {
        "agency_under_override": "agency_displacement",
        "connection_under_exclusion": "belonging_erasure",
    }
    assert payload["qualification_gates"] == {
        "training_unique_count": 4,
        "training_count_per_class": 2,
        "challenge_row_count": 224,
        "challenge_count_per_class": 112,
        "challenge_group_count": 28,
        "rows_per_challenge_group": 8,
        "required_correct_rows": 224,
        "required_correct_groups": 28,
        "group_correctness_rule": "all_eight_rows_correct_and_margin_gate",
        "challenge_group_assignment": (
            "qualification_owned_surface_kind_plus_source_position.v1"
        ),
        "top1_expected_required": True,
        "minimum_normalized_margin": 0.01,
        "minimum_normalized_margin_comparator": ">=",
        "tie_policy": "fail",
        "fresh_bge_process_count": 2,
        "fresh_bge_exact_vector_reobservation_required": True,
        "reader_artifact_rederived_from_reobservation_required": True,
        "prediction_ledger_fsync_before_label_release": True,
        "statistical_independence_claim": False,
    }
    assert set(payload["claims"].values()) == {False}
    execution = payload["execution"]
    assert execution["qualification_execution_authorized"] is False
    assert execution["model_output_count"] == 0
    assert execution["process_firewall_security_claim"] is False


def test_preflight_projection_is_opaque_and_split_counts_are_exact(
    tmp_path: pathlib.Path,
) -> None:
    preflight, execution, result = _prepare(tmp_path, "opaque")

    assert result["protocol_id"] == _PROTOCOL_ID
    assert result["training_input_count"] == 4
    assert result["challenge_input_count"] == 224
    assert result["challenge_group_count"] == 28
    assert result["proposed_execution_root"] == str(execution.resolve())
    assert result["model_or_cuda_used"] is False
    assert result["qualification_execution_authorized"] is False
    assert result["condition_reader_qualified"] is False
    assert result["campaign_execution_admitted"] is False
    assert result["readable_product_effect"] is False
    assert result["four_able_complete"] is False

    predictor = _read_json(preflight / "public" / "predictor_request.json")
    challenge_rows = predictor["challenge_inputs"]
    assert isinstance(challenge_rows, list)
    assert len(challenge_rows) == 224
    assert all(set(row) == {"item_id", "text", "text_sha256"} for row in challenge_rows)
    assert len({row["item_id"] for row in challenge_rows}) == 224
    assert len({row["text_sha256"] for row in challenge_rows}) == 224
    assert all(
        hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        == row["text_sha256"]
        for row in challenge_rows
    )
    serialized_predictor = json.dumps(predictor, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "condition_id",
        "condition_label",
        "group_id",
        "session_id",
        "source_position",
        "subject_index",
        "surface_kind",
        "phase_id",
        "decision_index",
    ):
        assert forbidden not in serialized_predictor

    training = _read_json(preflight / "sealed" / "condition_training_labels.json")
    challenge = _read_json(preflight / "sealed" / "challenge_labels.json")
    groups = _read_json(preflight / "sealed" / "group_split.json")
    assert Counter(row["condition_label"] for row in training["rows"]) == Counter(
        {"agency_displacement": 2, "belonging_erasure": 2}
    )
    assert Counter(row["condition_label"] for row in challenge["rows"]) == Counter(
        {"agency_displacement": 112, "belonging_erasure": 112}
    )
    assert len(groups["challenge_groups"]) == 28
    assert all(row["row_count"] == 8 for row in groups["challenge_groups"])
    assert Counter(row["condition_label"] for row in groups["challenge_groups"]) == Counter(
        {"agency_displacement": 14, "belonging_erasure": 14}
    )
    assert groups["group_level_evaluation_unit_count"] == 28
    assert groups["statistical_independence_claim"] is False


def test_preflight_content_ids_are_stable_across_local_roots(
    tmp_path: pathlib.Path,
) -> None:
    _, _, first = _prepare(tmp_path, "first")
    _, _, second = _prepare(tmp_path, "second")

    stable_fields = {
        "public_corpus_artifact_id",
        "predictor_request_artifact_id",
        "training_labels_artifact_id",
        "challenge_labels_artifact_id",
        "group_split_artifact_id",
    }
    assert {field: first[field] for field in stable_fields} == {
        field: second[field] for field in stable_fields
    }
    assert first["publication_request_artifact_id"] != second[
        "publication_request_artifact_id"
    ]


def test_preflight_requires_absent_disjoint_roots(tmp_path: pathlib.Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="preflight root already exists"):
        prepare_relationship_condition_reader_qualification_preflight(
            preflight_root=existing,
            proposed_execution_root=tmp_path / "future-a",
        )
    with pytest.raises(FileExistsError, match="execution root must not exist"):
        prepare_relationship_condition_reader_qualification_preflight(
            preflight_root=tmp_path / "future-b",
            proposed_execution_root=existing,
        )
    with pytest.raises(ValueError, match="must be disjoint"):
        prepare_relationship_condition_reader_qualification_preflight(
            preflight_root=tmp_path / "future-c",
            proposed_execution_root=tmp_path / "future-c" / "execution",
        )


def test_validation_requires_external_publication_identity_and_root(
    tmp_path: pathlib.Path,
) -> None:
    preflight, execution, result = _prepare(tmp_path, "external")
    common = {
        "preflight_root": preflight,
        "expected_protocol_id": _PROTOCOL_ID,
        "expected_publication_request_artifact_id": result[
            "publication_request_artifact_id"
        ],
        "expected_proposed_execution_root": execution,
    }
    assert validate_relationship_condition_reader_qualification_preflight(
        **common
    )["protocol_id"] == _PROTOCOL_ID
    with pytest.raises(ValueError, match="publication request artifact id mismatch"):
        validate_relationship_condition_reader_qualification_preflight(
            **{**common, "expected_publication_request_artifact_id": "f" * 64}
        )
    with pytest.raises(ValueError, match="proposed execution root mismatch"):
        validate_relationship_condition_reader_qualification_preflight(
            **{
                **common,
                "expected_proposed_execution_root": tmp_path / "other-execution",
            }
        )
    with pytest.raises(ValueError, match="protocol id mismatch"):
        validate_relationship_condition_reader_qualification_preflight(
            **{**common, "expected_protocol_id": "f" * 64}
        )


def test_validation_rejects_extra_file_and_artifact_tamper(
    tmp_path: pathlib.Path,
) -> None:
    extra_root, extra_execution, extra_result = _prepare(tmp_path, "extra")
    (extra_root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or extra files"):
        validate_relationship_condition_reader_qualification_preflight(
            preflight_root=extra_root,
            expected_protocol_id=_PROTOCOL_ID,
            expected_publication_request_artifact_id=extra_result[
                "publication_request_artifact_id"
            ],
            expected_proposed_execution_root=extra_execution,
        )

    tamper_root, tamper_execution, tamper_result = _prepare(tmp_path, "tamper")
    predictor = tamper_root / "public" / "predictor_request.json"
    predictor.write_bytes(predictor.read_bytes() + b" ")
    with pytest.raises(ValueError, match="not canonical JSON"):
        validate_relationship_condition_reader_qualification_preflight(
            preflight_root=tamper_root,
            expected_protocol_id=_PROTOCOL_ID,
            expected_publication_request_artifact_id=tamper_result[
                "publication_request_artifact_id"
            ],
            expected_proposed_execution_root=tamper_execution,
        )


def test_protocol_loader_rejects_duplicate_nonfinite_and_crlf(
    tmp_path: pathlib.Path,
) -> None:
    source = relationship_condition_reader_qualification_protocol_path()
    original = source.read_text(encoding="utf-8")
    schema_line = (
        '"schema_version": "relationship-condition-reader-qualification-protocol.v2",'
    )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        original.replace(
            schema_line,
            f'{schema_line}\n  {schema_line}',
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_relationship_condition_reader_qualification_protocol(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        original.replace('"minimum_normalized_margin": 0.01', '"minimum_normalized_margin": NaN'),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_relationship_condition_reader_qualification_protocol(nonfinite)

    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="LF-only"):
        load_relationship_condition_reader_qualification_protocol(crlf)


def test_qualification_protocol_registry_preserves_historical_v1_bytes() -> None:
    v1_path = relationship_condition_reader_qualification_protocol_path(
        RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V1
    )
    v2_path = relationship_condition_reader_qualification_protocol_path(
        RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V2
    )
    historical = load_relationship_condition_reader_qualification_protocol(v1_path)

    assert v1_path.name == "relationship_condition_reader_qualification_v1.json"
    assert v2_path.name == "relationship_condition_reader_qualification_v2.json"
    assert historical.schema_version == RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V1
    assert historical.protocol_id == "6a78e2bb3f2921d02ef72b14a25419badd8d814e7d2a47b050e7ee195b39479f"
    assert historical.raw_sha256 == "fbd7bb1be677a9359cf45f9ce58bebad49e9f23dbff6146f54bd110ba10b1130"
    assert historical.challenge_source.schema_version == "relationship-product-pilot-source.v2"
