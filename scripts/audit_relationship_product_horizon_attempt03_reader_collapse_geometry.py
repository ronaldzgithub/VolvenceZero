#!/usr/bin/env python3
"""Model-free geometry diagnosis of the attempt03 named-reader collapse.

The frozen prototype-cosine reader emitted ``agency_displacement`` for all
192 decisions.  This audit replays the reader geometry from the attempt's
pinned public BGE-M3 embedding table only: prototype-prototype cosine, the
per-text scores against both prototypes, the signed margin along the
prototype difference direction, and the exact rank separation of that margin
across the sealed conditions.

It distinguishes three candidate mechanical patterns without assigning a
causal or capability claim:

1. constant-winner offset: every input scores higher against one prototype,
   so argmax collapses regardless of condition;
2. residual condition signal: whether the margin still rank-separates the
   sealed conditions (exact pairwise AUC), i.e. whether the failure is an
   uncalibrated offset rather than an unresolvable embedding;
3. input-distribution distance: absolute cosine levels of inputs against
   both prototypes.

This is a post-hoc development-tier diagnostic.  It does not modify or
re-judge attempt03, does not run a model, and does not authorize a Readable
claim or a reader redesign verdict.  Claim ceiling:
``post_hoc_reader_collapse_geometry_diagnosis_only``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Sequence


sys.dont_write_bytecode = True


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_MODULE_PATH = _REPO_ROOT / "scripts" / "audit_relationship_product_horizon_attempt03_owner_history.py"
_BASE_MODULE_NAME = "attempt03_owner_history_audit_base_for_reader_geometry"
_BASE_MODULE_RAW_SHA256 = "5a74df39e7b6625244e0f0e8d2fc98236757e90aa8851ad695419ef2608d92a4"
_BASE_MODULE_RAW_BYTES = 101204
_AUDIT_SCOPE = "post_hoc_reader_collapse_geometry_diagnosis_only"
_OUTPUT_FILES = ("rows.json", "report.json", "manifest.json")
_DIAGNOSIS_ARM = "volvence_full"


class ReaderGeometryContractError(ValueError):
    """Raised when immutable evidence or a diagnosis invariant is violated."""


def _fail(message: str) -> None:
    raise ReaderGeometryContractError(message)


def _load_base_module():
    raw = _BASE_MODULE_PATH.read_bytes()
    if len(raw) != _BASE_MODULE_RAW_BYTES:
        _fail(
            "pinned owner-history audit module byte count drifted: "
            f"observed={len(raw)}, expected={_BASE_MODULE_RAW_BYTES}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _BASE_MODULE_RAW_SHA256:
        _fail(
            "pinned owner-history audit module raw SHA-256 drifted: "
            f"observed={digest}, expected={_BASE_MODULE_RAW_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_MODULE_PATH)
    if spec is None or spec.loader is None:
        _fail("cannot construct import spec for the pinned owner-history audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_module()


def _collect_condition_joined_texts(reader) -> dict[str, dict[str, object]]:
    """Join every decision current_input with its sealed condition, fail-loud."""

    subject_scopes = sorted(path.name for path in (reader.root / "chains").iterdir() if path.is_dir())
    _BASE._require_equal(len(subject_scopes), 8, "attempt subject scope count")
    joined: dict[str, dict[str, object]] = {}
    for subject_scope in subject_scopes:
        prefix = f"chains/{subject_scope}/{_DIAGNOSIS_ARM}"
        for decision_index in range(24):
            stem = f"decision-{decision_index:02d}"
            request = reader.load(f"{prefix}/requests/{stem}.json")
            sealed = reader.load(f"{prefix}/sealed/{stem}.json")
            session = _BASE._object(request.get("session"), "request session")
            _BASE._require_equal(
                session.get("decision_index"), decision_index, "session decision index"
            )
            text = _BASE._text(session.get("current_input"), "session current_input")
            condition = _BASE._text(sealed.get("condition_id"), "sealed condition_id")
            if condition not in _BASE._CONDITION_TO_READER_LABEL:
                _fail(f"unknown sealed condition: {condition}")
            digest = _BASE._sha256_text(text)
            entry = joined.get(digest)
            if entry is None:
                joined[digest] = {
                    "text_sha256": digest,
                    "text": text,
                    "condition_id": condition,
                    "decision_indices": [decision_index],
                    "occurrence_count": 1,
                }
            else:
                _BASE._require_equal(entry["condition_id"], condition, f"text {digest} condition stability")
                entry["occurrence_count"] = _BASE._integer(entry["occurrence_count"], "occurrence") + 1
                indices = entry["decision_indices"]
                assert isinstance(indices, list)
                if decision_index not in indices:
                    indices.append(decision_index)
    _BASE._require_equal(len(joined), 24, "distinct decision input text count")
    total = sum(_BASE._integer(entry["occurrence_count"], "occurrence") for entry in joined.values())
    _BASE._require_equal(total, 192, "decision input occurrence total")
    condition_counts = Counter(str(entry["condition_id"]) for entry in joined.values())
    _BASE._require_equal(
        dict(condition_counts),
        {"agency_under_override": 12, "connection_under_exclusion": 12},
        "distinct text condition balance",
    )
    return joined


def _exact_rank_separation(
    agency_margins: Sequence[float],
    belonging_margins: Sequence[float],
) -> dict[str, object]:
    """Exact pairwise AUC of margin(agency texts) > margin(belonging texts)."""

    wins = 0
    ties = 0
    for agency_value in agency_margins:
        for belonging_value in belonging_margins:
            if agency_value > belonging_value:
                wins += 1
            elif agency_value == belonging_value:
                ties += 1
    pairs = len(agency_margins) * len(belonging_margins)
    if pairs == 0:
        _fail("rank separation requires both condition groups")
    auc = (wins + 0.5 * ties) / pairs
    return {
        "pair_count": pairs,
        "agency_margin_greater_pair_count": wins,
        "tie_pair_count": ties,
        "exact_pairwise_auc_agency_condition_has_larger_margin": auc,
        "exact_pairwise_auc_hex": auc.hex(),
    }


def _margin_statistics(values: Sequence[float]) -> dict[str, object]:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        _fail("margin statistics require at least one value")
    mean = math.fsum(ordered) / count
    if count % 2 == 1:
        median = ordered[count // 2]
    else:
        median = 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])
    return {
        "count": count,
        "min": ordered[0],
        "min_hex": ordered[0].hex(),
        "max": ordered[-1],
        "max_hex": ordered[-1].hex(),
        "mean": mean,
        "mean_hex": mean.hex(),
        "median": median,
        "median_hex": median.hex(),
        "all_positive": ordered[0] > 0.0,
    }


def build_audit_documents(attempt_root: Path) -> dict[str, bytes]:
    """Build every output in memory, failing before any write on drift."""

    reader = _BASE._AttemptReader(attempt_root)
    _protocol, frozen_report, _windows, source_provenance = _BASE._validate_attempt_authority(reader)
    vectors, embedding_table, reader_artifact = _BASE._load_embedding_table(reader)

    prototype_vectors = {
        label: _BASE._vector_for_text(vectors, summary) for label, summary in _BASE._PROTOTYPES
    }
    labels = [label for label, _ in _BASE._PROTOTYPES]
    _BASE._require_equal(labels, ["agency_displacement", "belonging_erasure"], "prototype label order")
    agency_vector = prototype_vectors["agency_displacement"]
    belonging_vector = prototype_vectors["belonging_erasure"]
    prototype_cosine = math.fsum(a * b for a, b in zip(agency_vector, belonging_vector, strict=True))

    joined = _collect_condition_joined_texts(reader)
    rows: list[dict[str, object]] = []
    margins_by_condition: dict[str, list[float]] = {
        "agency_under_override": [],
        "connection_under_exclusion": [],
    }
    scores_by_condition: dict[str, dict[str, list[float]]] = {
        "agency_under_override": {"agency": [], "belonging": []},
        "connection_under_exclusion": {"agency": [], "belonging": []},
    }
    observed_labels: Counter[str] = Counter()
    for digest in sorted(joined):
        entry = joined[digest]
        text = _BASE._text(entry["text"], "joined text")
        condition = _BASE._text(entry["condition_id"], "joined condition")
        readout = _BASE._condition_readout(text, vectors)
        observed_label = _BASE._text(readout["condition_label"], "readout label")
        observed_labels[observed_label] += 1
        candidate_scores = {
            _BASE._text(item["label"], "candidate label"): _BASE._number(item["score"], "candidate score")
            for item in _BASE._array(readout["candidate_scores"], "candidate scores")
        }
        agency_score = candidate_scores["agency_displacement"]
        belonging_score = candidate_scores["belonging_erasure"]
        margin = agency_score - belonging_score
        margins_by_condition[condition].append(margin)
        scores_by_condition[condition]["agency"].append(agency_score)
        scores_by_condition[condition]["belonging"].append(belonging_score)
        rows.append(
            _BASE._with_row_id(
                {
                    "text_sha256": digest,
                    "sealed_condition_id": condition,
                    "expected_reader_label": _BASE._CONDITION_TO_READER_LABEL[condition],
                    "observed_reader_label": observed_label,
                    "reader_truth_match": observed_label == _BASE._CONDITION_TO_READER_LABEL[condition],
                    "occurrence_count": entry["occurrence_count"],
                    "decision_indices": sorted(entry["decision_indices"]),
                    "agency_prototype_cosine": agency_score,
                    "agency_prototype_cosine_hex": agency_score.hex(),
                    "belonging_prototype_cosine": belonging_score,
                    "belonging_prototype_cosine_hex": belonging_score.hex(),
                    "agency_minus_belonging_margin": margin,
                    "agency_minus_belonging_margin_hex": margin.hex(),
                }
            )
        )
    _BASE._require_equal(len(rows), 24, "diagnosis row count")
    _BASE._require_equal(dict(observed_labels), {"agency_displacement": 24}, "collapsed reader output")

    separation = _exact_rank_separation(
        margins_by_condition["agency_under_override"],
        margins_by_condition["connection_under_exclusion"],
    )
    all_margins = (
        margins_by_condition["agency_under_override"] + margins_by_condition["connection_under_exclusion"]
    )
    constant_winner_offset = min(all_margins) > 0.0
    auc = _BASE._number(
        separation["exact_pairwise_auc_agency_condition_has_larger_margin"], "exact AUC"
    )
    if constant_winner_offset and auc > 0.5:
        pattern = "constant_winner_offset_with_residual_condition_rank_signal"
    elif constant_winner_offset:
        pattern = "constant_winner_offset_without_condition_rank_signal"
    else:
        pattern = "no_constant_winner_offset_observed"

    audit_script_path = Path(__file__).resolve()
    audit_script_raw = audit_script_path.read_bytes()
    rows_document, rows_raw = _BASE._artifact_document(
        "relationship-product-horizon-attempt03-reader-collapse-geometry-rows.v1",
        audit_scope=_AUDIT_SCOPE,
        source_protocol_id=_BASE._EXPECTED_PROTOCOL_ID,
        row_count=len(rows),
        rows=rows,
    )
    report_payload: dict[str, object] = {
        "audit_scope": _AUDIT_SCOPE,
        "source_attempt": {
            "attempt_root": reader.root_relative,
            "manifest_artifact_id": _BASE._EXPECTED_MANIFEST_ARTIFACT_ID,
            "manifest_raw_sha256": _BASE._EXPECTED_MANIFEST_RAW_SHA256,
            "protocol_id": _BASE._EXPECTED_PROTOCOL_ID,
            "protocol_raw_sha256": _BASE._EXPECTED_PROTOCOL_RAW_SHA256,
            "report_artifact_id": _BASE._EXPECTED_REPORT_ARTIFACT_ID,
            "report_raw_sha256": _BASE._EXPECTED_REPORT_RAW_SHA256,
            "frozen_verdict": frozen_report.get("verdict"),
            "source_public_and_sealed": source_provenance,
        },
        "reader_geometry": {
            "reader_artifact": {**reader_artifact, "artifact_id": _BASE._EXPECTED_READER_ARTIFACT_ID},
            "public_embedding_table": {
                **reader.reference(_BASE._EMBEDDING_TABLE_PATH, embedding_table),
                "artifact_id": _BASE._EXPECTED_EMBEDDING_TABLE_ID,
            },
            "score_definition": "cosine(unit_input, unit_prototype), argmax over two prototypes, no bias term",
            "prototype_labels": labels,
            "prototype_prototype_cosine": prototype_cosine,
            "prototype_prototype_cosine_hex": prototype_cosine.hex(),
        },
        "join": {
            "arm_id": _DIAGNOSIS_ARM,
            "distinct_decision_input_text_count": 24,
            "decision_occurrence_total": 192,
            "condition_balance": {"agency_under_override": 12, "connection_under_exclusion": 12},
            "condition_stable_per_text": True,
            "sealed_truth_used_for_labeling_only": True,
        },
        "collapse_confirmation": {
            "observed_reader_label_counts": dict(sorted(observed_labels.items())),
            "constant_winner_offset_all_margins_positive": constant_winner_offset,
        },
        "margin_statistics": {
            "all_texts": _margin_statistics(all_margins),
            "agency_condition_texts": _margin_statistics(margins_by_condition["agency_under_override"]),
            "belonging_condition_texts": _margin_statistics(
                margins_by_condition["connection_under_exclusion"]
            ),
        },
        "absolute_cosine_statistics": {
            condition: {
                "against_agency_prototype": _margin_statistics(scores["agency"]),
                "against_belonging_prototype": _margin_statistics(scores["belonging"]),
            }
            for condition, scores in scores_by_condition.items()
        },
        "condition_rank_separation": separation,
        "descriptive_root_cause_pattern": pattern,
        "candidate_pattern_definitions": {
            "constant_winner_offset_with_residual_condition_rank_signal": (
                "Every input scores strictly higher against the agency prototype (argmax collapses), "
                "yet the signed margin still rank-orders the sealed conditions above chance; the "
                "operative mechanical failure is an uncalibrated constant offset between prototypes, "
                "not an unresolvable embedding."
            ),
            "constant_winner_offset_without_condition_rank_signal": (
                "Argmax collapses and the margin carries no rank information about the sealed "
                "conditions; the embedding does not resolve the conditions at prototype granularity."
            ),
            "no_constant_winner_offset_observed": "Collapse is not reproduced by geometry replay.",
        },
        "audit_implementation": {
            "path": _BASE._repo_relative(audit_script_path, label="audit script"),
            "bytes": len(audit_script_raw),
            "raw_sha256": hashlib.sha256(audit_script_raw).hexdigest(),
            "standard_library_only": True,
        },
        "base_module": {
            "path": _BASE._repo_relative(_BASE_MODULE_PATH, label="base audit module"),
            "bytes": _BASE_MODULE_RAW_BYTES,
            "raw_sha256": _BASE_MODULE_RAW_SHA256,
            "reused_for": "manifest-bound reads, embedding table load, condition readout replay",
        },
        "frozen_judgment_preserved": {
            "verdict": frozen_report.get("verdict"),
            "four_able_complete": False,
            "formal_evidence_authorized": False,
            "single_axis_contrast_claim_authorized": False,
            "attempt03_files_modified": False,
        },
        "honest_boundaries": {
            "post_hoc_diagnosis": True,
            "pre_registered_confirmatory_analysis": False,
            "model_output_count": 0,
            "cuda_used": False,
            "network_used": False,
            "readable_capability_established": False,
            "reader_redesign_verdict_authorized": False,
            "reader_error_attribution_to_outcomes_authorized": False,
            "evidence_tier": "development",
            "claim_ceiling": (
                "Geometry replay of the frozen prototype-cosine reader over the 24 distinct "
                "decision inputs. The descriptive pattern classification is a mechanical "
                "observation about embedding geometry; it does not attribute outcome differences "
                "to the reader and does not qualify or disqualify any successor reader."
            ),
        },
        "outputs": {
            "rows": {
                "path": "rows.json",
                "artifact_id": rows_document["artifact_id"],
                "raw_sha256": _BASE._sha256_bytes(rows_raw),
                "bytes": len(rows_raw),
                "row_count": len(rows),
            }
        },
    }
    report_document, report_raw = _BASE._artifact_document(
        "relationship-product-horizon-attempt03-reader-collapse-geometry-report.v1",
        **report_payload,
    )
    manifest_document, manifest_raw = _BASE._artifact_document(
        "relationship-product-horizon-attempt03-reader-collapse-geometry-manifest.v1",
        manifest_written_last=True,
        source_attempt_manifest_artifact_id=_BASE._EXPECTED_MANIFEST_ARTIFACT_ID,
        source_attempt_manifest_raw_sha256=_BASE._EXPECTED_MANIFEST_RAW_SHA256,
        report_artifact_id=report_document["artifact_id"],
        files=[
            {
                "path": "rows.json",
                "bytes": len(rows_raw),
                "raw_sha256": _BASE._sha256_bytes(rows_raw),
                "artifact_id": rows_document["artifact_id"],
            },
            {
                "path": "report.json",
                "bytes": len(report_raw),
                "raw_sha256": _BASE._sha256_bytes(report_raw),
                "artifact_id": report_document["artifact_id"],
            },
        ],
    )
    return {
        "rows.json": rows_raw,
        "report.json": report_raw,
        "manifest.json": manifest_raw,
    }


def materialize(attempt_root: Path, output_dir: Path) -> dict[str, object]:
    documents = build_audit_documents(attempt_root)
    _BASE._write_documents_create_only(output_dir, documents)
    report = _BASE._strict_json_object(documents["report.json"], label="generated report")
    manifest = _BASE._strict_json_object(documents["manifest.json"], label="generated manifest")
    return {
        "output_dir": _BASE._repo_relative(output_dir, label="audit output directory"),
        "report_artifact_id": report["artifact_id"],
        "manifest_artifact_id": manifest["artifact_id"],
        "descriptive_root_cause_pattern": report["descriptive_root_cause_pattern"],
        "prototype_prototype_cosine_hex": _BASE._object(
            report["reader_geometry"], "reader geometry"
        )["prototype_prototype_cosine_hex"],
        "exact_pairwise_auc_hex": _BASE._object(
            report["condition_rank_separation"], "rank separation"
        )["exact_pairwise_auc_hex"],
    }


def validate_existing(attempt_root: Path, output_dir: Path) -> dict[str, object]:
    expected = build_audit_documents(attempt_root)
    output = output_dir.resolve()
    _BASE._repo_relative(output, label="audit output directory")
    if not output.is_dir():
        _fail(f"audit output directory does not exist: {output}")
    observed_names = sorted(path.name for path in output.iterdir())
    _BASE._require_equal(observed_names, sorted(_OUTPUT_FILES), "audit output exact file set")
    for name in _OUTPUT_FILES:
        observed = (output / name).read_bytes()
        _BASE._require_equal(observed, expected[name], f"audit byte-exact replay {name}")
    report = _BASE._strict_json_object(expected["report.json"], label="expected report")
    manifest = _BASE._strict_json_object(expected["manifest.json"], label="expected manifest")
    return {
        "output_dir": _BASE._repo_relative(output, label="audit output directory"),
        "report_artifact_id": report["artifact_id"],
        "manifest_artifact_id": manifest["artifact_id"],
        "byte_exact_replay": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-free attempt03 reader-collapse geometry diagnosis",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "validate-existing"):
        command = commands.add_parser(name)
        command.add_argument("--attempt-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize(args.attempt_root, args.output_dir)
    elif args.command == "validate-existing":
        result = validate_existing(args.attempt_root, args.output_dir)
    else:  # pragma: no cover - argparse enforces the command set.
        _fail(f"unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
