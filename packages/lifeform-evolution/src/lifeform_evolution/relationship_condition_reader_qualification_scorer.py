"""Model-free scorer for the source-separated condition-reader qualification.

The scorer deliberately opens evaluator labels only after it has validated a
closed, fsynced prediction-ledger commit receipt.  It never imports the BGE
adapter or loads model weights.  The boundary is a reviewed process-order
firewall, not an operating-system security sandbox.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import os
import pathlib
import stat
import sys
from typing import Mapping

from volvence_zero.canonical_json import canonical_json_bytes, strict_json_loads
from volvence_zero.social_cognition import (
    relationship_condition_readout_from_payload,
)

from .relationship_condition_reader_qualification_runtime_binding import (
    snapshot_file_backed_module_origins,
)


RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION = "relationship-condition-reader-prediction-ledger.v1"
RELATIONSHIP_READER_PREDICTION_LEDGER_COMMIT_SCHEMA_VERSION = (
    "relationship-condition-reader-prediction-ledger-commit.v1"
)
RELATIONSHIP_READER_SCORING_REQUEST_SCHEMA_VERSION = "relationship-condition-reader-qualification-scoring-request.v1"
RELATIONSHIP_READER_QUALIFICATION_REPORT_SCHEMA_VERSION = "relationship-condition-reader-qualification-report.v1"
RELATIONSHIP_READER_QUALIFICATION_SCORER_ATTESTATION_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-scorer-attestation.v3"
)
RELATIONSHIP_READER_QUALIFICATION_SCORER_MANIFEST_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-scorer-manifest.v1"
)

_CHALLENGE_LABELS_SCHEMA_VERSION = "relationship-condition-reader-qualification-challenge-labels.v1"
_GROUP_SPLIT_SCHEMA_VERSION = "relationship-condition-reader-qualification-group-split.v1"
_LABELS = ("agency_displacement", "belonging_erasure")
_EXPECTED_ROW_COUNT = 224
_EXPECTED_GROUP_COUNT = 28
_EXPECTED_ROWS_PER_GROUP = 8
_MAX_SMALL_ARTIFACT_BYTES = 2_000_000
_MAX_LEDGER_BYTES = 4_000_000


def score_relationship_condition_reader_qualification(
    *,
    scoring_request_path: pathlib.Path,
    output_root: pathlib.Path,
) -> Mapping[str, object]:
    """Score one committed ledger and publish an immutable scoped verdict."""

    _assert_model_free_process()
    root = pathlib.Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"qualification scorer output root exists: {root}")

    sequence: list[str] = []
    request, _request_raw = _load_artifact(
        pathlib.Path(scoring_request_path),
        schema_version=RELATIONSHIP_READER_SCORING_REQUEST_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    _validate_scoring_request(request)
    sequence.append("scoring_request_validated")

    commit_path = pathlib.Path(_text(request["commit_receipt_path"], "commit_receipt_path"))
    commit, _commit_raw = _load_artifact(
        commit_path,
        schema_version=RELATIONSHIP_READER_PREDICTION_LEDGER_COMMIT_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    _validate_commit_receipt(commit, request=request)
    sequence.append("commit_receipt_validated")

    ledger_path = pathlib.Path(_text(request["prediction_ledger_path"], "prediction_ledger_path"))
    ledger, ledger_raw = _load_artifact(
        ledger_path,
        schema_version=RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION,
        max_bytes=_MAX_LEDGER_BYTES,
    )
    _validate_prediction_ledger(ledger, request=request, commit=commit)
    _validate_file_identity(
        ledger_raw,
        expected_sha256=_digest(
            request["prediction_ledger_raw_sha256"],
            "prediction_ledger_raw_sha256",
        ),
        expected_bytes=_positive_integer(
            request["prediction_ledger_raw_bytes"],
            "prediction_ledger_raw_bytes",
        ),
        field_name="prediction ledger",
    )
    # Parse every owner-published readout before evaluator files are opened.
    # This rejects a structurally valid but semantically inconsistent ledger
    # (for example, a non-top condition label or a forged margin) without
    # burning the sealed qualification labels.
    ledger_rows = _prediction_rows(ledger)
    sequence.append("prediction_ledger_semantically_revalidated")

    # This is the first label-file open in this scorer process.  All commit and
    # ledger checks above must complete before control can reach this statement.
    challenge_path = pathlib.Path(_text(request["challenge_labels_path"], "challenge_labels_path"))
    challenge, challenge_raw = _load_artifact(
        challenge_path,
        schema_version=_CHALLENGE_LABELS_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    _validate_file_identity(
        challenge_raw,
        expected_sha256=_digest(
            request["challenge_labels_raw_sha256"],
            "challenge_labels_raw_sha256",
        ),
        expected_bytes=_positive_integer(
            request["challenge_labels_raw_bytes"],
            "challenge_labels_raw_bytes",
        ),
        field_name="challenge labels",
    )
    if challenge["artifact_id"] != request["challenge_labels_artifact_id"]:
        raise ValueError("challenge labels external artifact identity mismatch")
    sequence.append("challenge_labels_opened")

    group_path = pathlib.Path(_text(request["group_split_path"], "group_split_path"))
    groups, groups_raw = _load_artifact(
        group_path,
        schema_version=_GROUP_SPLIT_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    _validate_file_identity(
        groups_raw,
        expected_sha256=_digest(
            request["group_split_raw_sha256"],
            "group_split_raw_sha256",
        ),
        expected_bytes=_positive_integer(
            request["group_split_raw_bytes"],
            "group_split_raw_bytes",
        ),
        field_name="group split",
    )
    if groups["artifact_id"] != request["group_split_artifact_id"]:
        raise ValueError("group split external artifact identity mismatch")
    sequence.append("group_split_opened")

    report = _score(
        request=request,
        ledger=ledger,
        ledger_rows=ledger_rows,
        challenge=challenge,
        groups=groups,
    )
    _assert_model_free_process()
    root.mkdir(parents=True)
    report_identity = _write_artifact_create_only(root / "report.json", report)
    sequence.append("report_fsynced_and_reopened")
    attestation = _with_artifact_id(
        {
            "schema_version": (RELATIONSHIP_READER_QUALIFICATION_SCORER_ATTESTATION_SCHEMA_VERSION),
            "qualification_protocol_id": request["qualification_protocol_id"],
            "execution_protocol_id": request["execution_protocol_id"],
            "scoring_request_artifact_id": request["artifact_id"],
            "prediction_ledger_commit_artifact_id": commit["artifact_id"],
            "process_pid": os.getpid(),
            "parent_pid": os.getppid(),
            "run_nonce": request["run_nonce"],
            "process_executable": str(pathlib.Path(sys.executable).resolve()),
            "process_argv": list(sys.argv),
            "process_cwd": str(pathlib.Path.cwd().resolve()),
            "process_sys_path": list(sys.path),
            "process_runtime_flags": _process_runtime_flags(),
            "environment_key_names": sorted(os.environ),
            "environment_value_sha256s": {
                key: hashlib.sha256(value.encode("utf-8")).hexdigest() for key, value in sorted(os.environ.items())
            },
            "unlisted_environment_variables_recorded": True,
            "loaded_file_backed_module_origins": snapshot_file_backed_module_origins(sys.modules),
            "volvence_zero_namespace_search_locations": (_volvence_zero_namespace_search_locations()),
            "event_sequence": list(sequence),
            "challenge_labels_first_open_after_commit_validation": True,
            "model_or_cuda_used": False,
            "torch_imported": False,
            "sentence_transformers_imported": False,
            "os_security_boundary": False,
            "windows_directory_entry_durability_attested": False,
        }
    )

    attestation_identity = _write_artifact_create_only(
        root / "scorer_attestation.json",
        attestation,
    )
    manifest = _with_artifact_id(
        {
            "schema_version": (RELATIONSHIP_READER_QUALIFICATION_SCORER_MANIFEST_SCHEMA_VERSION),
            "qualification_protocol_id": request["qualification_protocol_id"],
            "execution_protocol_id": request["execution_protocol_id"],
            "scoring_request_artifact_id": request["artifact_id"],
            "files": [
                {"path": "report.json", **report_identity},
                {"path": "scorer_attestation.json", **attestation_identity},
            ],
            "file_count": 2,
            "model_or_cuda_used": False,
        }
    )
    _write_artifact_create_only(root / "manifest.json", manifest)
    return report


def _score(
    *,
    request: Mapping[str, object],
    ledger: Mapping[str, object],
    ledger_rows: Mapping[str, Mapping[str, object]],
    challenge: Mapping[str, object],
    groups: Mapping[str, object],
) -> Mapping[str, object]:
    if challenge["protocol_id"] != request["qualification_protocol_id"]:
        raise ValueError("challenge labels qualification protocol mismatch")
    if challenge["label_release_condition"] != "prediction_ledger_create_only_fsynced":
        raise ValueError("challenge label release condition drifted")
    if groups["protocol_id"] != request["qualification_protocol_id"]:
        raise ValueError("group split qualification protocol mismatch")
    challenge_rows = _challenge_rows(challenge)
    group_rows = _group_rows(groups)
    if set(ledger_rows) != set(challenge_rows):
        raise ValueError("prediction ledger and challenge label item sets differ")
    group_item_ids = {item_id for group in group_rows.values() for item_id in group["item_ids"]}
    if group_item_ids != set(challenge_rows):
        raise ValueError("group split and challenge label item sets differ")

    minimum_margin = _canonical_hex_float(
        request["minimum_normalized_margin_hex"],
        "minimum_normalized_margin_hex",
    )
    row_results: dict[str, dict[str, object]] = {}
    for item_id in sorted(ledger_rows):
        prediction = ledger_rows[item_id]
        expected = challenge_rows[item_id]
        if prediction["text_sha256"] != expected["text_sha256"]:
            raise ValueError("prediction/challenge text identity mismatch")
        correct = prediction["condition_label"] == expected["condition_label"]
        margin = prediction["normalized_margin"]
        margin_pass = margin >= minimum_margin
        row_results[item_id] = {
            "correct": correct,
            "margin_pass": margin_pass,
            "row_pass": correct and margin_pass,
        }

    group_passes: dict[str, bool] = {}
    for group_id, group in group_rows.items():
        expected_group_label = group["condition_label"]
        if any(
            challenge_rows[item_id]["condition_label"] != expected_group_label
            or challenge_rows[item_id]["group_id"] != group_id
            for item_id in group["item_ids"]
        ):
            raise ValueError("group split assignment disagrees with challenge labels")
        group_passes[group_id] = all(row_results[item_id]["row_pass"] is True for item_id in group["item_ids"])

    correct_rows = sum(result["correct"] is True for result in row_results.values())
    margin_rows = sum(result["margin_pass"] is True for result in row_results.values())
    passing_rows = sum(result["row_pass"] is True for result in row_results.values())
    passing_groups = sum(group_passes.values())
    admitted = (
        correct_rows == _EXPECTED_ROW_COUNT
        and margin_rows == _EXPECTED_ROW_COUNT
        and passing_rows == _EXPECTED_ROW_COUNT
        and passing_groups == _EXPECTED_GROUP_COUNT
    )
    return _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_QUALIFICATION_REPORT_SCHEMA_VERSION,
            "qualification_protocol_id": request["qualification_protocol_id"],
            "execution_protocol_id": request["execution_protocol_id"],
            "scoring_request_artifact_id": request["artifact_id"],
            "prediction_ledger_artifact_id": ledger["artifact_id"],
            "challenge_labels_artifact_id": challenge["artifact_id"],
            "group_split_artifact_id": groups["artifact_id"],
            "row_count": _EXPECTED_ROW_COUNT,
            "effective_group_count": _EXPECTED_GROUP_COUNT,
            "rows_per_group": _EXPECTED_ROWS_PER_GROUP,
            "correct_row_count": correct_rows,
            "margin_passing_row_count": margin_rows,
            "passing_row_count": passing_rows,
            "passing_group_count": passing_groups,
            "minimum_normalized_margin_hex": request["minimum_normalized_margin_hex"],
            "exact_source_reader_development_admitted": admitted,
            "verdict": (
                "exact_source_reader_development_admitted"
                if admitted
                else "exact_source_reader_development_not_admitted"
            ),
            "statistical_independence_claim": False,
            "campaign_execution_admitted": False,
            "readable_product_effect": False,
            "appendable_product_effect": False,
            "learnable_product_effect": False,
            "steerable_product_effect": False,
            "four_able_complete": False,
            "formal_evidence_authorized": False,
            "human_product_validation": False,
            "production_active": False,
        }
    )


def _prediction_rows(
    ledger: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    rows = _list(ledger["rows"], "prediction ledger rows")
    if len(rows) != _EXPECTED_ROW_COUNT:
        raise ValueError("prediction ledger must contain 224 rows")
    parsed: dict[str, dict[str, object]] = {}
    observed_item_ids: list[str] = []
    for index, value in enumerate(rows):
        row = _mapping(value, f"prediction row {index}")
        _exact_keys(
            row,
            {
                "item_id",
                "text_sha256",
                "condition_label",
                "confidence_hex",
                "normalized_margin_hex",
                "candidate_scores",
                "reader_artifact_id",
                "source_observation_sha256",
            },
            f"prediction row {index}",
        )
        item_id = _digest(row["item_id"], "prediction item_id")
        text_sha256 = _digest(row["text_sha256"], "prediction text_sha256")
        source_sha256 = _digest(
            row["source_observation_sha256"],
            "prediction source_observation_sha256",
        )
        if source_sha256 != text_sha256:
            raise ValueError("prediction source observation hash mismatch")
        condition_label = _text(row["condition_label"], "prediction condition_label")
        confidence = _canonical_hex_float(row["confidence_hex"], "confidence_hex")
        margin = _canonical_hex_float(
            row["normalized_margin_hex"],
            "normalized_margin_hex",
        )
        raw_scores = _list(row["candidate_scores"], "candidate_scores")
        scores: list[tuple[str, float]] = []
        for score_index, value in enumerate(raw_scores):
            score = _mapping(value, f"candidate score {score_index}")
            _exact_keys(
                score,
                {"label", "score_hex"},
                f"candidate score {score_index}",
            )
            scores.append(
                (
                    _text(score["label"], "candidate label"),
                    _canonical_hex_float(score["score_hex"], "candidate score_hex"),
                )
            )
        if tuple(label for label, _ in scores) != _LABELS:
            raise ValueError("prediction candidate label order drifted")
        normalized_margin_hex = _text(
            row["normalized_margin_hex"],
            "normalized_margin_hex",
        )
        ordered_scores = sorted(scores, key=lambda item: item[1], reverse=True)
        exact_margin = min(
            1.0,
            max(0.0, (ordered_scores[0][1] - ordered_scores[1][1]) / 2.0),
        )
        if _canonical_float_hex(exact_margin) != normalized_margin_hex:
            raise ValueError("prediction normalized margin is not exact")
        maximum = max(score for _, score in scores)
        exponentials = tuple(math.exp(score - maximum) for _, score in scores)
        top_index = max(range(len(scores)), key=lambda item: scores[item][1])
        exact_confidence = exponentials[top_index] / math.fsum(exponentials)
        if _canonical_float_hex(exact_confidence) != row["confidence_hex"]:
            raise ValueError("prediction confidence is not exact")
        readout = relationship_condition_readout_from_payload(
            {
                "condition_label": condition_label,
                "confidence": confidence,
                "normalized_margin": margin,
                "candidate_scores": [{"label": label, "score": score} for label, score in scores],
                "reader_artifact_id": row["reader_artifact_id"],
                "source_observation_sha256": source_sha256,
            }
        )
        if item_id in parsed:
            raise ValueError("prediction ledger item ids must be unique")
        if readout.reader_artifact_id != ledger["reader_artifact_id"]:
            raise ValueError("prediction row reader artifact lineage mismatch")
        observed_item_ids.append(item_id)
        parsed[item_id] = {
            "text_sha256": text_sha256,
            "condition_label": readout.condition_label,
            "normalized_margin": readout.normalized_margin,
        }
    if observed_item_ids != sorted(observed_item_ids):
        raise ValueError("prediction ledger rows must use canonical item_id order")
    return parsed


def _challenge_rows(
    challenge: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    _exact_keys(
        challenge,
        {
            "schema_version",
            "protocol_id",
            "public_corpus_artifact_id",
            "rows",
            "row_count",
            "label_release_condition",
            "artifact_id",
        },
        "challenge labels",
    )
    if _positive_integer(challenge["row_count"], "challenge row_count") != _EXPECTED_ROW_COUNT:
        raise ValueError("challenge labels row_count drifted")
    rows = _list(challenge["rows"], "challenge rows")
    if len(rows) != _EXPECTED_ROW_COUNT:
        raise ValueError("challenge labels must contain 224 rows")
    parsed: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    observed_item_ids: list[str] = []
    for index, value in enumerate(rows):
        row = _mapping(value, f"challenge row {index}")
        _exact_keys(
            row,
            {
                "item_id",
                "text_sha256",
                "condition_label",
                "group_id",
                "subject_index",
                "surface_kind",
                "source_position",
                "source_session_id",
            },
            f"challenge row {index}",
        )
        item_id = _digest(row["item_id"], "challenge item_id")
        label = _text(row["condition_label"], "challenge condition_label")
        if label not in _LABELS:
            raise ValueError("challenge condition label drifted")
        if item_id in parsed:
            raise ValueError("challenge item ids must be unique")
        parsed[item_id] = {
            "text_sha256": _digest(row["text_sha256"], "challenge text_sha256"),
            "condition_label": label,
            "group_id": _digest(row["group_id"], "challenge group_id"),
        }
        observed_item_ids.append(item_id)
        counts[label] += 1
    if counts != Counter({label: 112 for label in _LABELS}):
        raise ValueError("challenge labels must remain balanced 112/112")
    if observed_item_ids != sorted(observed_item_ids):
        raise ValueError("challenge labels must use canonical item_id order")
    return parsed


def _group_rows(groups: Mapping[str, object]) -> dict[str, dict[str, object]]:
    _exact_keys(
        groups,
        {
            "schema_version",
            "protocol_id",
            "training_item_ids",
            "challenge_item_ids",
            "challenge_groups",
            "challenge_group_count",
            "rows_per_challenge_group",
            "training_challenge_text_overlap_count",
            "statistical_independence_claim",
            "grouping_owner",
            "grouping_contract",
            "group_level_evaluation_unit_count",
            "artifact_id",
        },
        "group split",
    )
    if (
        _positive_integer(groups["challenge_group_count"], "challenge_group_count") != _EXPECTED_GROUP_COUNT
        or _positive_integer(groups["rows_per_challenge_group"], "rows_per_group") != _EXPECTED_ROWS_PER_GROUP
        or groups["statistical_independence_claim"] is not False
        or _positive_integer(
            groups["group_level_evaluation_unit_count"],
            "group_level_evaluation_unit_count",
        )
        != _EXPECTED_GROUP_COUNT
        or _nonnegative_integer(
            groups["training_challenge_text_overlap_count"],
            "training_challenge_text_overlap_count",
        )
        != 0
        or groups["grouping_owner"] != "qualification_preflight"
        or groups["grouping_contract"] != "surface_kind_and_source_position_across_voice_variants.v1"
    ):
        raise ValueError("group split count or honesty boundary drifted")
    training_item_ids = tuple(
        _digest(value, f"training_item_ids[{index}]")
        for index, value in enumerate(_list(groups["training_item_ids"], "training_item_ids"))
    )
    if (
        len(training_item_ids) != 4
        or len(set(training_item_ids)) != 4
        or training_item_ids != tuple(sorted(training_item_ids))
    ):
        raise ValueError("group split training ids must be four unique canonical ids")
    challenge_item_ids = tuple(
        _digest(value, f"challenge_item_ids[{index}]")
        for index, value in enumerate(_list(groups["challenge_item_ids"], "challenge_item_ids"))
    )
    if (
        len(challenge_item_ids) != _EXPECTED_ROW_COUNT
        or len(set(challenge_item_ids)) != _EXPECTED_ROW_COUNT
        or challenge_item_ids != tuple(sorted(challenge_item_ids))
    ):
        raise ValueError("group split challenge ids must be canonical and unique")
    raw_groups = _list(groups["challenge_groups"], "challenge_groups")
    if len(raw_groups) != _EXPECTED_GROUP_COUNT:
        raise ValueError("group split must contain 28 groups")
    parsed: dict[str, dict[str, object]] = {}
    observed_group_ids: list[str] = []
    condition_counts: Counter[str] = Counter()
    for index, value in enumerate(raw_groups):
        group = _mapping(value, f"challenge group {index}")
        _exact_keys(
            group,
            {"group_id", "item_ids", "row_count", "condition_label"},
            f"challenge group {index}",
        )
        group_id = _digest(group["group_id"], "group_id")
        item_ids = tuple(
            _digest(item, f"group item {item_index}")
            for item_index, item in enumerate(_list(group["item_ids"], "group item_ids"))
        )
        if (
            _positive_integer(group["row_count"], "group row_count") != _EXPECTED_ROWS_PER_GROUP
            or len(item_ids) != _EXPECTED_ROWS_PER_GROUP
            or len(set(item_ids)) != len(item_ids)
            or item_ids != tuple(sorted(item_ids))
        ):
            raise ValueError("each challenge group must contain eight unique rows")
        if group_id in parsed:
            raise ValueError("challenge group ids must be unique")
        condition_label = _text(
            group["condition_label"],
            "group condition_label",
        )
        if condition_label not in _LABELS:
            raise ValueError("challenge group condition label drifted")
        parsed[group_id] = {
            "item_ids": item_ids,
            "condition_label": condition_label,
        }
        observed_group_ids.append(group_id)
        condition_counts[condition_label] += 1
    if observed_group_ids != sorted(observed_group_ids):
        raise ValueError("challenge groups must use canonical group_id order")
    if condition_counts != Counter({label: 14 for label in _LABELS}):
        raise ValueError("challenge groups must remain balanced 14/14")
    if set(challenge_item_ids) != {item_id for group in parsed.values() for item_id in group["item_ids"]}:
        raise ValueError("challenge_item_ids disagree with grouped rows")
    return parsed


def _validate_scoring_request(request: Mapping[str, object]) -> None:
    _exact_keys(
        request,
        {
            "schema_version",
            "qualification_protocol_id",
            "execution_protocol_id",
            "run_nonce",
            "prediction_ledger_path",
            "prediction_ledger_artifact_id",
            "prediction_ledger_raw_sha256",
            "prediction_ledger_raw_bytes",
            "commit_receipt_path",
            "commit_receipt_artifact_id",
            "challenge_labels_path",
            "challenge_labels_artifact_id",
            "challenge_labels_raw_sha256",
            "challenge_labels_raw_bytes",
            "group_split_path",
            "group_split_artifact_id",
            "group_split_raw_sha256",
            "group_split_raw_bytes",
            "minimum_normalized_margin_hex",
            "artifact_id",
        },
        "scoring request",
    )
    for field_name in (
        "qualification_protocol_id",
        "execution_protocol_id",
        "prediction_ledger_artifact_id",
        "prediction_ledger_raw_sha256",
        "commit_receipt_artifact_id",
        "challenge_labels_artifact_id",
        "challenge_labels_raw_sha256",
        "group_split_artifact_id",
        "group_split_raw_sha256",
    ):
        _digest(request[field_name], field_name)
    _text(request["run_nonce"], "run_nonce")
    _positive_integer(request["prediction_ledger_raw_bytes"], "ledger raw bytes")
    _positive_integer(request["challenge_labels_raw_bytes"], "labels raw bytes")
    _positive_integer(request["group_split_raw_bytes"], "groups raw bytes")
    if (
        _canonical_hex_float(
            request["minimum_normalized_margin_hex"],
            "minimum_normalized_margin_hex",
        )
        != 0.01
    ):
        raise ValueError("qualification minimum margin must remain exactly 0.01")


def _validate_commit_receipt(
    commit: Mapping[str, object],
    *,
    request: Mapping[str, object],
) -> None:
    _exact_keys(
        commit,
        {
            "schema_version",
            "qualification_protocol_id",
            "execution_protocol_id",
            "child_request_artifact_id",
            "predictor_request_artifact_id",
            "prediction_ledger_artifact_id",
            "prediction_ledger_raw_sha256",
            "prediction_ledger_raw_bytes",
            "prediction_run_manifest_artifact_ids",
            "prediction_run_attestation_artifact_ids",
            "fresh_process_count",
            "predictor_processes_exited",
            "predictor_job_objects_empty",
            "embedding_tables_byte_exact",
            "reader_artifacts_byte_exact",
            "prediction_ledgers_byte_exact",
            "ledger_file_fsync_completed",
            "ledger_same_descriptor_readback",
            "ledger_closed_reopen_readback",
            "windows_directory_entry_durability_attested",
            "artifact_id",
        },
        "prediction ledger commit receipt",
    )
    if commit["artifact_id"] != request["commit_receipt_artifact_id"]:
        raise ValueError("commit receipt external artifact identity mismatch")
    if (
        commit["qualification_protocol_id"] != request["qualification_protocol_id"]
        or commit["execution_protocol_id"] != request["execution_protocol_id"]
        or commit["prediction_ledger_artifact_id"] != request["prediction_ledger_artifact_id"]
        or commit["prediction_ledger_raw_sha256"] != request["prediction_ledger_raw_sha256"]
        or commit["prediction_ledger_raw_bytes"] != request["prediction_ledger_raw_bytes"]
    ):
        raise ValueError("commit receipt ledger lineage mismatch")
    for field_name in (
        "child_request_artifact_id",
        "predictor_request_artifact_id",
    ):
        _digest(commit[field_name], field_name)
    for field_name in (
        "prediction_run_manifest_artifact_ids",
        "prediction_run_attestation_artifact_ids",
    ):
        values = _list(commit[field_name], field_name)
        if len(values) != 2:
            raise ValueError(f"commit receipt {field_name} must contain two entries")
        digests = tuple(_digest(value, field_name) for value in values)
        if len(set(digests)) != 2:
            raise ValueError(f"commit receipt {field_name} entries must be distinct")
    if _positive_integer(commit["fresh_process_count"], "fresh_process_count") != 2:
        raise ValueError("commit receipt requires exactly two fresh processes")
    for field_name in (
        "predictor_processes_exited",
        "predictor_job_objects_empty",
        "embedding_tables_byte_exact",
        "reader_artifacts_byte_exact",
        "prediction_ledgers_byte_exact",
        "ledger_file_fsync_completed",
        "ledger_same_descriptor_readback",
        "ledger_closed_reopen_readback",
    ):
        if commit[field_name] is not True:
            raise ValueError(f"commit receipt requires {field_name}=true")
    if commit["windows_directory_entry_durability_attested"] is not False:
        raise ValueError("commit receipt must not claim directory-entry durability")


def _validate_prediction_ledger(
    ledger: Mapping[str, object],
    *,
    request: Mapping[str, object],
    commit: Mapping[str, object],
) -> None:
    _exact_keys(
        ledger,
        {
            "schema_version",
            "protocol_id",
            "execution_protocol_id",
            "child_request_artifact_id",
            "predictor_request_artifact_id",
            "embedding_table_artifact_id",
            "reader_artifact_id",
            "rows",
            "row_count",
            "challenge_labels_present",
            "qualification_scored",
            "artifact_id",
        },
        "prediction ledger",
    )
    if (
        ledger["artifact_id"] != request["prediction_ledger_artifact_id"]
        or ledger["artifact_id"] != commit["prediction_ledger_artifact_id"]
        or ledger["protocol_id"] != request["qualification_protocol_id"]
        or ledger["execution_protocol_id"] != request["execution_protocol_id"]
        or ledger["child_request_artifact_id"] != commit["child_request_artifact_id"]
        or ledger["predictor_request_artifact_id"] != commit["predictor_request_artifact_id"]
        or _positive_integer(ledger["row_count"], "prediction row_count") != _EXPECTED_ROW_COUNT
        or ledger["challenge_labels_present"] is not False
        or ledger["qualification_scored"] is not False
    ):
        raise ValueError("prediction ledger identity or count drifted")
    for field_name in (
        "protocol_id",
        "execution_protocol_id",
        "child_request_artifact_id",
        "predictor_request_artifact_id",
        "embedding_table_artifact_id",
        "reader_artifact_id",
    ):
        _digest(ledger[field_name], field_name)


def _load_artifact(
    path: pathlib.Path,
    *,
    schema_version: str,
    max_bytes: int,
) -> tuple[Mapping[str, object], bytes]:
    source = pathlib.Path(path)
    if source.is_symlink():
        raise ValueError(f"qualification artifact must not be a symlink: {source}")
    before = source.stat(follow_symlinks=False)
    if before.st_nlink != 1:
        raise ValueError(f"qualification artifact must not be hard linked: {source}")
    if os.name == "nt" and getattr(before, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"qualification artifact must not be a reparse point: {source}")
    with source.open("rb") as handle:
        during = os.fstat(handle.fileno())
        raw = handle.read(max_bytes + 1)
    after = source.stat(follow_symlinks=False)
    if len(raw) > max_bytes:
        raise ValueError(f"qualification artifact exceeds byte bound: {source}")
    if _file_identity(before) != _file_identity(during) or _file_identity(during) != _file_identity(after):
        raise ValueError(f"qualification artifact identity changed while reading: {source}")
    parsed = strict_json_loads(raw, max_bytes=max_bytes)
    payload = _mapping(parsed, str(source))
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"qualification artifact schema mismatch: {source}")
    if raw != canonical_json_bytes(payload) + b"\n":
        raise ValueError(f"qualification artifact is not canonical JSON: {source}")
    artifact_id = _digest(payload.get("artifact_id"), f"{source} artifact_id")
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    if artifact_id != _sha256_json(core):
        raise ValueError(f"qualification artifact content address mismatch: {source}")
    return payload, raw


def _write_artifact_create_only(
    path: pathlib.Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    raw = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x+b") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        if handle.read() != raw:
            raise RuntimeError(f"qualification same-descriptor readback failed: {path}")
    with path.open("rb") as handle:
        if handle.read() != raw:
            raise RuntimeError(f"qualification closed-reopen readback failed: {path}")
    return {
        "artifact_id": payload["artifact_id"],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
    }


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    if "artifact_id" in core:
        raise ValueError("artifact core must not already contain artifact_id")
    return {**core, "artifact_id": _sha256_json(core)}


def _sha256_json(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_file_identity(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_bytes: int,
    field_name: str,
) -> None:
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{field_name} raw identity mismatch")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_nlink,
    )


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{field_name} keys mismatch; missing={missing}, extra={extra}")


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _digest(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _canonical_hex_float(value: object, field_name: str) -> float:
    text = _text(value, field_name)
    try:
        parsed = float.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a hexadecimal float") from exc
    if not math.isfinite(parsed) or parsed.hex() != text or (parsed == 0.0 and text != (0.0).hex()):
        raise ValueError(f"{field_name} must be a finite canonical hexadecimal float")
    return parsed


def _canonical_float_hex(value: float) -> str:
    normalized = 0.0 if value == 0.0 else value
    return normalized.hex()


def _assert_model_free_process() -> None:
    forbidden = tuple(
        module_name
        for module_name in sys.modules
        if module_name == "torch"
        or module_name.startswith("torch.")
        or module_name == "sentence_transformers"
        or module_name.startswith("sentence_transformers.")
    )
    if forbidden:
        raise RuntimeError(
            "qualification scorer must run in a fresh model-free process; "
            f"forbidden modules are loaded: {sorted(forbidden)[:8]}"
        )


def _process_runtime_flags() -> dict[str, object]:
    return {
        "dont_write_bytecode": sys.dont_write_bytecode,
        "no_site": sys.flags.no_site,
        "pycache_prefix": sys.pycache_prefix,
        "safe_path": sys.flags.safe_path,
        "utf8_mode": sys.flags.utf8_mode,
    }


def _volvence_zero_namespace_search_locations() -> list[str]:
    namespace = sys.modules.get("volvence_zero")
    if namespace is None:
        raise RuntimeError("volvence_zero namespace is not loaded")
    namespace_path = getattr(namespace, "__path__", None)
    if namespace_path is None:
        raise RuntimeError("volvence_zero namespace has no search locations")
    locations = [str(pathlib.Path(value).resolve()) for value in namespace_path]
    if locations != sorted(locations, key=lambda value: value.encode("utf-8")):
        raise RuntimeError("volvence_zero namespace search locations are not canonical")
    if len(locations) != len(set(locations)):
        raise RuntimeError("volvence_zero namespace search locations are not unique")
    return locations


__all__ = [
    "RELATIONSHIP_READER_PREDICTION_LEDGER_COMMIT_SCHEMA_VERSION",
    "RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION",
    "RELATIONSHIP_READER_SCORING_REQUEST_SCHEMA_VERSION",
    "score_relationship_condition_reader_qualification",
]
