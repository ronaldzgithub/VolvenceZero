from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import pathlib
import types
from typing import cast

import pytest

import lifeform_evolution.relationship_condition_reader_qualification_scorer as scorer
from volvence_zero.canonical_json import canonical_json_bytes


_LABELS = ("agency_displacement", "belonging_erasure")
_PREFLIGHT_PROTOCOL_ID = hashlib.sha256(b"qualification-preflight").hexdigest()
_EXECUTION_PROTOCOL_ID = hashlib.sha256(b"qualification-execution").hexdigest()
_PUBLIC_CORPUS_ARTIFACT_ID = hashlib.sha256(b"public-corpus").hexdigest()
_CHILD_REQUEST_ARTIFACT_ID = hashlib.sha256(b"child-request").hexdigest()
_PREDICTOR_REQUEST_ARTIFACT_ID = hashlib.sha256(b"predictor-request").hexdigest()
_EMBEDDING_TABLE_ARTIFACT_ID = hashlib.sha256(b"embedding-table").hexdigest()
_READER_ARTIFACT_ID = hashlib.sha256(b"reader-artifact").hexdigest()
_PREDICTION_RUN_MANIFEST_ARTIFACT_IDS = (
    hashlib.sha256(b"prediction-run-manifest-0").hexdigest(),
    hashlib.sha256(b"prediction-run-manifest-1").hexdigest(),
)
_PREDICTION_RUN_ATTESTATION_ARTIFACT_IDS = (
    hashlib.sha256(b"prediction-run-attestation-0").hexdigest(),
    hashlib.sha256(b"prediction-run-attestation-1").hexdigest(),
)

Payload = dict[str, object]
PayloadMutator = Callable[[Payload], None]


@dataclass(frozen=True)
class _FixturePaths:
    request: pathlib.Path
    commit: pathlib.Path
    ledger: pathlib.Path
    challenge: pathlib.Path
    groups: pathlib.Path


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _with_artifact_id(core: Payload) -> Payload:
    assert "artifact_id" not in core
    return {
        **core,
        "artifact_id": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
    }


def _readdress(payload: Payload) -> Payload:
    return _with_artifact_id({key: value for key, value in payload.items() if key != "artifact_id"})


def _artifact_bytes(payload: Payload) -> bytes:
    return canonical_json_bytes(payload) + b"\n"


def _write_artifact(path: pathlib.Path, payload: Payload) -> bytes:
    raw = _artifact_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _prediction_values(
    condition_label: str,
    *,
    top_score: float = 0.8,
    second_score: float = 0.2,
) -> tuple[str, str, list[Payload]]:
    if condition_label == _LABELS[0]:
        scores = (top_score, second_score)
    else:
        scores = (second_score, top_score)
    margin = (top_score - second_score) / 2.0
    maximum = max(scores)
    exponentials = tuple(math.exp(score - maximum) for score in scores)
    confidence = max(exponentials) / math.fsum(exponentials)
    return (
        confidence.hex(),
        margin.hex(),
        [{"label": label, "score_hex": score.hex()} for label, score in zip(_LABELS, scores, strict=True)],
    )


def _base_payloads() -> tuple[Payload, Payload, Payload]:
    challenge_rows: list[Payload] = []
    ledger_rows: list[Payload] = []
    group_rows: list[Payload] = []

    for group_index in range(28):
        group_id = _digest(f"group-{group_index:02d}")
        condition_label = _LABELS[0] if group_index < 14 else _LABELS[1]
        item_ids: list[str] = []
        for variant_index in range(8):
            item_id = _digest(f"item-{group_index:02d}-{variant_index:02d}")
            text_sha256 = _digest(f"text-{group_index:02d}-{variant_index:02d}")
            item_ids.append(item_id)
            challenge_rows.append(
                {
                    "item_id": item_id,
                    "text_sha256": text_sha256,
                    "condition_label": condition_label,
                    "group_id": group_id,
                    "subject_index": variant_index,
                    "surface_kind": ("onboarding" if group_index < 4 else "decision"),
                    "source_position": group_index,
                    "source_session_id": (f"source-session-{group_index:02d}-{variant_index:02d}"),
                }
            )
            confidence_hex, margin_hex, candidate_scores = _prediction_values(condition_label)
            ledger_rows.append(
                {
                    "item_id": item_id,
                    "text_sha256": text_sha256,
                    "condition_label": condition_label,
                    "confidence_hex": confidence_hex,
                    "normalized_margin_hex": margin_hex,
                    "candidate_scores": candidate_scores,
                    "reader_artifact_id": _READER_ARTIFACT_ID,
                    "source_observation_sha256": text_sha256,
                }
            )
        group_rows.append(
            {
                "group_id": group_id,
                "item_ids": sorted(item_ids),
                "row_count": 8,
                "condition_label": condition_label,
            }
        )

    challenge_rows.sort(key=lambda row: cast(str, row["item_id"]))
    ledger_rows.sort(key=lambda row: cast(str, row["item_id"]))
    group_rows.sort(key=lambda row: cast(str, row["group_id"]))
    challenge_item_ids = [cast(str, row["item_id"]) for row in challenge_rows]

    ledger = _with_artifact_id(
        {
            "schema_version": scorer.RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION,
            "protocol_id": _PREFLIGHT_PROTOCOL_ID,
            "execution_protocol_id": _EXECUTION_PROTOCOL_ID,
            "child_request_artifact_id": _CHILD_REQUEST_ARTIFACT_ID,
            "predictor_request_artifact_id": _PREDICTOR_REQUEST_ARTIFACT_ID,
            "embedding_table_artifact_id": _EMBEDDING_TABLE_ARTIFACT_ID,
            "reader_artifact_id": _READER_ARTIFACT_ID,
            "rows": ledger_rows,
            "row_count": 224,
            "challenge_labels_present": False,
            "qualification_scored": False,
        }
    )
    challenge = _with_artifact_id(
        {
            "schema_version": ("relationship-condition-reader-qualification-challenge-labels.v1"),
            "protocol_id": _PREFLIGHT_PROTOCOL_ID,
            "public_corpus_artifact_id": _PUBLIC_CORPUS_ARTIFACT_ID,
            "rows": challenge_rows,
            "row_count": 224,
            "label_release_condition": "prediction_ledger_create_only_fsynced",
        }
    )
    groups = _with_artifact_id(
        {
            "schema_version": ("relationship-condition-reader-qualification-group-split.v1"),
            "protocol_id": _PREFLIGHT_PROTOCOL_ID,
            "training_item_ids": sorted(_digest(f"training-item-{index}") for index in range(4)),
            "challenge_item_ids": challenge_item_ids,
            "challenge_groups": group_rows,
            "challenge_group_count": 28,
            "rows_per_challenge_group": 8,
            "training_challenge_text_overlap_count": 0,
            "statistical_independence_claim": False,
            "grouping_owner": "qualification_preflight",
            "grouping_contract": ("surface_kind_and_source_position_across_voice_variants.v1"),
            "group_level_evaluation_unit_count": 28,
        }
    )
    return ledger, challenge, groups


def _persist_fixture(
    root: pathlib.Path,
    *,
    ledger_mutator: PayloadMutator | None = None,
    challenge_mutator: PayloadMutator | None = None,
    group_mutator: PayloadMutator | None = None,
    commit_mutator: PayloadMutator | None = None,
) -> _FixturePaths:
    ledger, challenge, groups = deepcopy(_base_payloads())
    if ledger_mutator is not None:
        ledger_mutator(ledger)
    if challenge_mutator is not None:
        challenge_mutator(challenge)
    if group_mutator is not None:
        group_mutator(groups)
    ledger = _readdress(ledger)
    challenge = _readdress(challenge)
    groups = _readdress(groups)

    ledger_path = root / "prediction" / "prediction_ledger.json"
    challenge_path = root / "sealed" / "challenge_labels.json"
    group_path = root / "sealed" / "group_split.json"
    ledger_raw = _write_artifact(ledger_path, ledger)
    challenge_raw = _write_artifact(challenge_path, challenge)
    group_raw = _write_artifact(group_path, groups)

    commit = _with_artifact_id(
        {
            "schema_version": (scorer.RELATIONSHIP_READER_PREDICTION_LEDGER_COMMIT_SCHEMA_VERSION),
            "qualification_protocol_id": _PREFLIGHT_PROTOCOL_ID,
            "execution_protocol_id": _EXECUTION_PROTOCOL_ID,
            "child_request_artifact_id": _CHILD_REQUEST_ARTIFACT_ID,
            "predictor_request_artifact_id": _PREDICTOR_REQUEST_ARTIFACT_ID,
            "prediction_ledger_artifact_id": ledger["artifact_id"],
            "prediction_ledger_raw_sha256": hashlib.sha256(ledger_raw).hexdigest(),
            "prediction_ledger_raw_bytes": len(ledger_raw),
            "prediction_run_manifest_artifact_ids": list(_PREDICTION_RUN_MANIFEST_ARTIFACT_IDS),
            "prediction_run_attestation_artifact_ids": list(_PREDICTION_RUN_ATTESTATION_ARTIFACT_IDS),
            "fresh_process_count": 2,
            "predictor_processes_exited": True,
            "predictor_job_objects_empty": True,
            "embedding_tables_byte_exact": True,
            "reader_artifacts_byte_exact": True,
            "prediction_ledgers_byte_exact": True,
            "ledger_file_fsync_completed": True,
            "ledger_same_descriptor_readback": True,
            "ledger_closed_reopen_readback": True,
            "windows_directory_entry_durability_attested": False,
        }
    )
    if commit_mutator is not None:
        commit_mutator(commit)
        commit = _readdress(commit)
    commit_path = root / "prediction" / "prediction_ledger_commit.json"
    _write_artifact(commit_path, commit)

    request = _with_artifact_id(
        {
            "schema_version": scorer.RELATIONSHIP_READER_SCORING_REQUEST_SCHEMA_VERSION,
            "qualification_protocol_id": _PREFLIGHT_PROTOCOL_ID,
            "execution_protocol_id": _EXECUTION_PROTOCOL_ID,
            "run_nonce": "scorer-test-run",
            "prediction_ledger_path": str(ledger_path.resolve()),
            "prediction_ledger_artifact_id": ledger["artifact_id"],
            "prediction_ledger_raw_sha256": hashlib.sha256(ledger_raw).hexdigest(),
            "prediction_ledger_raw_bytes": len(ledger_raw),
            "commit_receipt_path": str(commit_path.resolve()),
            "commit_receipt_artifact_id": commit["artifact_id"],
            "challenge_labels_path": str(challenge_path.resolve()),
            "challenge_labels_artifact_id": challenge["artifact_id"],
            "challenge_labels_raw_sha256": hashlib.sha256(challenge_raw).hexdigest(),
            "challenge_labels_raw_bytes": len(challenge_raw),
            "group_split_path": str(group_path.resolve()),
            "group_split_artifact_id": groups["artifact_id"],
            "group_split_raw_sha256": hashlib.sha256(group_raw).hexdigest(),
            "group_split_raw_bytes": len(group_raw),
            "minimum_normalized_margin_hex": (0.01).hex(),
        }
    )
    request_path = root / "scoring_request.json"
    _write_artifact(request_path, request)
    return _FixturePaths(
        request=request_path,
        commit=commit_path,
        ledger=ledger_path,
        challenge=challenge_path,
        groups=group_path,
    )


def _read_json(path: pathlib.Path) -> Payload:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _loaded_paths(monkeypatch: pytest.MonkeyPatch) -> list[pathlib.Path]:
    opened: list[pathlib.Path] = []
    original = scorer._load_artifact

    def tracking_load(
        path: pathlib.Path,
        *,
        schema_version: str,
        max_bytes: int,
    ) -> tuple[object, bytes]:
        opened.append(pathlib.Path(path).resolve())
        return original(path, schema_version=schema_version, max_bytes=max_bytes)

    monkeypatch.setattr(scorer, "_load_artifact", tracking_load)
    return opened


def test_valid_224_rows_and_28_groups_are_admitted_with_order_attested(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _persist_fixture(tmp_path / "inputs")
    opened = _loaded_paths(monkeypatch)
    output = tmp_path / "score"

    report = scorer.score_relationship_condition_reader_qualification(
        scoring_request_path=fixture.request,
        output_root=output,
    )

    assert report["correct_row_count"] == 224
    assert report["margin_passing_row_count"] == 224
    assert report["passing_row_count"] == 224
    assert report["passing_group_count"] == 28
    assert report["exact_source_reader_development_admitted"] is True
    assert report["verdict"] == "exact_source_reader_development_admitted"
    assert report["execution_protocol_id"] == _EXECUTION_PROTOCOL_ID
    assert report["row_count"] == 224
    assert report["effective_group_count"] == 28
    assert report["rows_per_group"] == 8
    assert report["minimum_normalized_margin_hex"] == (0.01).hex()
    for claim in (
        "statistical_independence_claim",
        "campaign_execution_admitted",
        "readable_product_effect",
        "appendable_product_effect",
        "learnable_product_effect",
        "steerable_product_effect",
        "four_able_complete",
        "formal_evidence_authorized",
        "human_product_validation",
        "production_active",
    ):
        assert report[claim] is False
    assert opened == [
        fixture.request.resolve(),
        fixture.commit.resolve(),
        fixture.ledger.resolve(),
        fixture.challenge.resolve(),
        fixture.groups.resolve(),
    ]

    commit = _read_json(fixture.commit)
    assert commit["qualification_protocol_id"] == _PREFLIGHT_PROTOCOL_ID
    assert commit["execution_protocol_id"] == _EXECUTION_PROTOCOL_ID
    assert commit["child_request_artifact_id"] == _CHILD_REQUEST_ARTIFACT_ID
    assert commit["predictor_request_artifact_id"] == _PREDICTOR_REQUEST_ARTIFACT_ID
    assert commit["prediction_run_manifest_artifact_ids"] == list(_PREDICTION_RUN_MANIFEST_ARTIFACT_IDS)
    assert commit["prediction_run_attestation_artifact_ids"] == list(_PREDICTION_RUN_ATTESTATION_ARTIFACT_IDS)
    assert commit["predictor_processes_exited"] is True
    assert commit["predictor_job_objects_empty"] is True
    assert commit["ledger_file_fsync_completed"] is True

    attestation = _read_json(output / "scorer_attestation.json")
    assert attestation["schema_version"] == ("relationship-condition-reader-qualification-scorer-attestation.v3")
    origins = cast(list[Payload], attestation["loaded_file_backed_module_origins"])
    assert [row["module_name"] for row in origins] == sorted(row["module_name"] for row in origins)
    assert len({cast(str, row["module_name"]) for row in origins}) == len(origins)
    assert all(set(row) == {"module_name", "origin"} for row in origins)
    assert attestation["event_sequence"] == [
        "scoring_request_validated",
        "commit_receipt_validated",
        "prediction_ledger_semantically_revalidated",
        "challenge_labels_opened",
        "group_split_opened",
        "report_fsynced_and_reopened",
    ]
    assert attestation["challenge_labels_first_open_after_commit_validation"] is True
    assert attestation["model_or_cuda_used"] is False
    assert attestation["torch_imported"] is False
    assert attestation["sentence_transformers_imported"] is False
    assert attestation["os_security_boundary"] is False
    assert attestation["windows_directory_entry_durability_attested"] is False
    assert attestation["execution_protocol_id"] == _EXECUTION_PROTOCOL_ID
    assert attestation["prediction_ledger_commit_artifact_id"] == commit["artifact_id"]
    manifest = _read_json(output / "manifest.json")
    assert manifest["model_or_cuda_used"] is False
    assert manifest["file_count"] == 2
    assert [entry["path"] for entry in cast(list[Payload], manifest["files"])] == [
        "report.json",
        "scorer_attestation.json",
    ]


def _mutate_wrong_prediction(payload: Payload) -> None:
    rows = cast(list[Payload], payload["rows"])
    row = rows[0]
    expected = cast(str, row["condition_label"])
    wrong = _LABELS[1] if expected == _LABELS[0] else _LABELS[0]
    confidence_hex, margin_hex, candidate_scores = _prediction_values(wrong)
    row["condition_label"] = wrong
    row["confidence_hex"] = confidence_hex
    row["normalized_margin_hex"] = margin_hex
    row["candidate_scores"] = candidate_scores


def _mutate_low_margin(payload: Payload) -> None:
    row = cast(list[Payload], payload["rows"])[0]
    label = cast(str, row["condition_label"])
    confidence_hex, margin_hex, candidate_scores = _prediction_values(
        label,
        top_score=0.01,
        second_score=0.0,
    )
    row["confidence_hex"] = confidence_hex
    row["normalized_margin_hex"] = margin_hex
    row["candidate_scores"] = candidate_scores


def _mutate_tie(payload: Payload) -> None:
    rows = cast(list[Payload], payload["rows"])
    row = next(value for value in rows if value["condition_label"] == _LABELS[0])
    confidence_hex, margin_hex, candidate_scores = _prediction_values(
        _LABELS[0],
        top_score=0.0,
        second_score=0.0,
    )
    row["condition_label"] = _LABELS[0]
    row["confidence_hex"] = confidence_hex
    row["normalized_margin_hex"] = margin_hex
    row["candidate_scores"] = candidate_scores


@pytest.mark.parametrize(
    ("mutator", "expected_correct", "expected_margin"),
    [
        (_mutate_wrong_prediction, 223, 224),
        (_mutate_low_margin, 224, 223),
        (_mutate_tie, 224, 223),
    ],
    ids=("one-wrong-row", "one-low-margin", "one-tie"),
)
def test_row_failure_publishes_scoped_negative_report(
    tmp_path: pathlib.Path,
    mutator: PayloadMutator,
    expected_correct: int,
    expected_margin: int,
) -> None:
    fixture = _persist_fixture(tmp_path / "inputs", ledger_mutator=mutator)
    output = tmp_path / "score"

    report = scorer.score_relationship_condition_reader_qualification(
        scoring_request_path=fixture.request,
        output_root=output,
    )

    assert report["correct_row_count"] == expected_correct
    assert report["margin_passing_row_count"] == expected_margin
    assert report["passing_row_count"] == 223
    assert report["passing_group_count"] == 27
    assert report["exact_source_reader_development_admitted"] is False
    assert report["verdict"] == "exact_source_reader_development_not_admitted"
    assert (output / "report.json").is_file()


@pytest.mark.parametrize(
    "damage",
    ("missing-commit", "tampered-commit", "missing-ledger", "tampered-ledger"),
)
def test_invalid_commit_or_ledger_never_opens_labels_or_creates_output(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    fixture = _persist_fixture(tmp_path / "inputs")
    if damage == "missing-commit":
        fixture.commit.unlink()
    elif damage == "tampered-commit":
        fixture.commit.write_bytes(fixture.commit.read_bytes() + b" ")
    elif damage == "missing-ledger":
        fixture.ledger.unlink()
    else:
        fixture.ledger.write_bytes(fixture.ledger.read_bytes() + b" ")
    opened = _loaded_paths(monkeypatch)
    output = tmp_path / "score"

    with pytest.raises((FileNotFoundError, ValueError)):
        scorer.score_relationship_condition_reader_qualification(
            scoring_request_path=fixture.request,
            output_root=output,
        )

    assert fixture.challenge.resolve() not in opened
    assert fixture.groups.resolve() not in opened
    assert not output.exists()


def _mutate_nonempty_predictor_job(payload: Payload) -> None:
    payload["predictor_job_objects_empty"] = False


def _mutate_duplicate_run_manifests(payload: Payload) -> None:
    value = cast(list[str], payload["prediction_run_manifest_artifact_ids"])[0]
    payload["prediction_run_manifest_artifact_ids"] = [value, value]


def _mutate_duplicate_run_attestations(payload: Payload) -> None:
    value = cast(list[str], payload["prediction_run_attestation_artifact_ids"])[0]
    payload["prediction_run_attestation_artifact_ids"] = [value, value]


def _mutate_commit_child_lineage(payload: Payload) -> None:
    payload["child_request_artifact_id"] = _digest("different-child-request")


@pytest.mark.parametrize(
    "mutator",
    (
        _mutate_nonempty_predictor_job,
        _mutate_duplicate_run_manifests,
        _mutate_duplicate_run_attestations,
        _mutate_commit_child_lineage,
    ),
    ids=(
        "predictor-job-not-empty",
        "duplicate-run-manifests",
        "duplicate-run-attestations",
        "child-lineage-drift",
    ),
)
def test_commit_process_closure_drift_fails_before_label_open(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: PayloadMutator,
) -> None:
    fixture = _persist_fixture(tmp_path / "inputs", commit_mutator=mutator)
    opened = _loaded_paths(monkeypatch)
    output = tmp_path / "score"

    with pytest.raises(ValueError):
        scorer.score_relationship_condition_reader_qualification(
            scoring_request_path=fixture.request,
            output_root=output,
        )

    assert fixture.challenge.resolve() not in opened
    assert fixture.groups.resolve() not in opened
    assert not output.exists()


def _mutate_duplicate_item(payload: Payload) -> None:
    rows = cast(list[Payload], payload["rows"])
    rows[1]["item_id"] = rows[0]["item_id"]


def _mutate_reordered_items(payload: Payload) -> None:
    rows = cast(list[Payload], payload["rows"])
    payload["rows"] = list(reversed(rows))


def _mutate_unknown_item(payload: Payload) -> None:
    rows = cast(list[Payload], payload["rows"])
    rows[0]["item_id"] = _digest("unknown-item")
    rows.sort(key=lambda row: cast(str, row["item_id"]))


def _mutate_non_top_condition_label(payload: Payload) -> None:
    row = cast(list[Payload], payload["rows"])[0]
    row["condition_label"] = _LABELS[1] if row["condition_label"] == _LABELS[0] else _LABELS[0]


def _mutate_inexact_confidence(payload: Payload) -> None:
    row = cast(list[Payload], payload["rows"])[0]
    row["confidence_hex"] = (0.75).hex()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (_mutate_non_top_condition_label, "top-scoring candidate"),
        (_mutate_inexact_confidence, "confidence is not exact"),
    ],
    ids=("non-top-label", "inexact-confidence"),
)
def test_semantically_invalid_ledger_never_opens_labels(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: PayloadMutator,
    message: str,
) -> None:
    fixture = _persist_fixture(
        tmp_path / "inputs",
        ledger_mutator=mutator,
    )
    opened = _loaded_paths(monkeypatch)
    output = tmp_path / "score"

    with pytest.raises(ValueError, match=message):
        scorer.score_relationship_condition_reader_qualification(
            scoring_request_path=fixture.request,
            output_root=output,
        )

    assert fixture.challenge.resolve() not in opened
    assert fixture.groups.resolve() not in opened
    assert not output.exists()


def _mutate_duplicate_group(payload: Payload) -> None:
    groups = cast(list[Payload], payload["challenge_groups"])
    groups[1]["group_id"] = groups[0]["group_id"]


def _mutate_reordered_groups(payload: Payload) -> None:
    groups = cast(list[Payload], payload["challenge_groups"])
    payload["challenge_groups"] = list(reversed(groups))


def _mutate_reordered_group_items(payload: Payload) -> None:
    group = cast(list[Payload], payload["challenge_groups"])[0]
    group["item_ids"] = list(reversed(cast(list[str], group["item_ids"])))


def _mutate_unknown_group(payload: Payload) -> None:
    groups = cast(list[Payload], payload["challenge_groups"])
    groups[0]["group_id"] = _digest("unknown-group")
    groups.sort(key=lambda group: cast(str, group["group_id"]))


@pytest.mark.parametrize(
    ("target", "mutator"),
    [
        ("ledger", _mutate_duplicate_item),
        ("ledger", _mutate_reordered_items),
        ("ledger", _mutate_unknown_item),
        ("groups", _mutate_duplicate_group),
        ("groups", _mutate_reordered_groups),
        ("groups", _mutate_reordered_group_items),
        ("groups", _mutate_unknown_group),
    ],
    ids=(
        "duplicate-item",
        "reordered-items",
        "unknown-item",
        "duplicate-group",
        "reordered-groups",
        "reordered-group-items",
        "unknown-group",
    ),
)
def test_item_and_group_identity_drift_fails_closed(
    tmp_path: pathlib.Path,
    target: str,
    mutator: PayloadMutator,
) -> None:
    fixture = _persist_fixture(
        tmp_path / "inputs",
        ledger_mutator=mutator if target == "ledger" else None,
        group_mutator=mutator if target == "groups" else None,
    )
    output = tmp_path / "score"

    with pytest.raises(ValueError):
        scorer.score_relationship_condition_reader_qualification(
            scoring_request_path=fixture.request,
            output_root=output,
        )

    assert not output.exists()


def test_preexisting_output_root_is_rejected_before_any_input_open(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _persist_fixture(tmp_path / "inputs")
    output = tmp_path / "score"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("existing output\n", encoding="utf-8")
    opened = _loaded_paths(monkeypatch)

    with pytest.raises(FileExistsError, match="output root exists"):
        scorer.score_relationship_condition_reader_qualification(
            scoring_request_path=fixture.request,
            output_root=output,
        )

    assert opened == []
    assert marker.read_text(encoding="utf-8") == "existing output\n"
    assert set(output.iterdir()) == {marker}


def test_loaded_model_runtime_is_rejected_before_any_input_open(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _persist_fixture(tmp_path / "inputs")
    output = tmp_path / "score"
    opened = _loaded_paths(monkeypatch)
    monkeypatch.setitem(scorer.sys.modules, "torch", types.ModuleType("torch"))

    with pytest.raises(RuntimeError, match="fresh model-free process"):
        scorer.score_relationship_condition_reader_qualification(
            scoring_request_path=fixture.request,
            output_root=output,
        )

    assert opened == []
    assert not output.exists()
