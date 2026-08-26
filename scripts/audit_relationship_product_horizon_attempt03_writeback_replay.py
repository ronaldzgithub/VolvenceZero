#!/usr/bin/env python3
"""Full-coverage mechanical writeback replay for attempt03 (development tier).

Extends the frozen 36-row owner-history audit to every full/frozen
arm-decision: 384 settlement->PE->credit->gate->owner->next-tick transitions,
184 full-arm owner continuity edges, 192 frozen-arm resets, and 368 gate
continuity edges, then reconciles the replayed counts against the frozen
attempt03 mechanism-evidence numbers.

This is a model-free, post-hoc diagnostic.  It reuses the byte-pinned
owner-history audit module for all per-decision contract validation, reads
only manifest-bound attempt03 artifacts plus the attempt's frozen public
embedding table, does not import the current runtime, does not run a model,
does not revise the frozen verdict, and does not authorize a Learnable or
product-causal claim.  Claim ceiling:
``post_hoc_mechanical_writeback_replay_only``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


sys.dont_write_bytecode = True


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_MODULE_PATH = _REPO_ROOT / "scripts" / "audit_relationship_product_horizon_attempt03_owner_history.py"
_BASE_MODULE_NAME = "attempt03_owner_history_audit_base"
_BASE_MODULE_RAW_SHA256 = "5a74df39e7b6625244e0f0e8d2fc98236757e90aa8851ad695419ef2608d92a4"
_BASE_MODULE_RAW_BYTES = 101204
_AUDIT_SCOPE = "post_hoc_mechanical_writeback_replay_only"
_OUTPUT_FILES = ("rows.json", "report.json", "manifest.json")
_EXPECTED_ONBOARDING_BOUNDARY_BASIS = {
    "volvence_full": "hydrate_exact_prior_decision_boundary",
    "appendable_frozen_onboarding": "same_frozen_post_onboarding_boundary_each_decision",
}


class WritebackAuditContractError(ValueError):
    """Raised when immutable evidence or a replay invariant is violated."""


def _fail(message: str) -> None:
    raise WritebackAuditContractError(message)


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


def _compact_row(detail: Mapping[str, object], *, segment: str | None) -> dict[str, object]:
    """Project one fully validated arm-decision onto compact replay evidence."""

    unit = _BASE._object(detail["unit"], "detail unit")
    identity = _BASE._object(detail["record_identity"], "detail record identity")
    pre_owner = _BASE._object(detail["pre_owner"], "detail pre owner")
    post_owner = _BASE._object(detail["post_owner"], "detail post owner")
    pre_hydration = _BASE._object(pre_owner["hydration"], "pre hydration")
    post_hydration = _BASE._object(post_owner["hydration"], "post hydration")
    transition = _BASE._object(detail["owner_transition"], "detail owner transition")
    gate = _BASE._object(detail["gate"], "detail gate")
    forecast = _BASE._object(detail["forecast"], "detail forecast")
    readout = _BASE._object(detail["named_readout"], "detail named readout")
    mechanism = _BASE._object(detail["postaction_mechanism_refs"], "detail mechanism refs")
    decision_index = _BASE._integer(unit["decision_index"], "unit decision index")
    row = {
        "arm_id": _BASE._text(unit["arm_id"], "unit arm"),
        "subject_scope": _BASE._text(identity["subject_scope"], "identity subject"),
        "decision_id": _BASE._text(identity["decision_id"], "identity decision"),
        "decision_index": decision_index,
        "world_clone_id": _BASE._sha256(unit["world_clone_id"], "unit world"),
        "primary_window": 12 <= decision_index <= 23,
        "horizon_segment": segment,
        "writeback_chain": {
            "pre_owner_payload_sha256": _BASE._sha256(pre_hydration["payload_sha256"], "pre payload sha"),
            "post_owner_payload_sha256": _BASE._sha256(post_hydration["payload_sha256"], "post payload sha"),
            "appended_evidence_id": _BASE._text(transition["appended_record_id"], "appended evidence"),
            "evicted_record_count": len(_BASE._array(transition["evicted_record_ids"], "evicted records")),
            "evicted_outcome_count": len(
                _BASE._array(transition["evicted_outcome_evidence_ids"], "evicted outcomes")
            ),
            "observed_post_matches_hard_window": _BASE._boolean(
                transition["observed_post_matches_hard_window"], "hard window law"
            ),
            "pre_record_count": _BASE._integer(pre_owner["record_count"], "pre record count"),
            "post_record_count": _BASE._integer(post_owner["record_count"], "post record count"),
        },
        "forecast": {
            "forecast_sha256": _BASE._sha256(forecast["forecast_sha256"], "forecast sha"),
            "canonical_payload_byte_exact_with_receipt": _BASE._boolean(
                forecast["canonical_payload_byte_exact_with_receipt"], "forecast byte exact"
            ),
            "numeric_values_bit_exact_with_receipt": _BASE._boolean(
                forecast["numeric_values_bit_exact_with_receipt"], "forecast bit exact"
            ),
        },
        "named_readout": {
            "condition_label": _BASE._text(readout["condition_label"], "readout label"),
            "normalized_margin_hex": _BASE._number(
                readout["normalized_margin"], "readout margin"
            ).hex(),
        },
        "gate": {
            "gate_action": gate["gate_action"],
            "steer_probability_hex": _BASE._text(gate["steer_probability_hex"], "gate probability hex"),
            "update_count_before": _BASE._integer(gate["update_count_before"], "gate before"),
            "update_count_after": _BASE._integer(gate["update_count_after"], "gate after"),
        },
        "settlement": {
            "selected_action_id": _BASE._text(detail["selected_action_id"], "selected action"),
            "typed_outcome_id": _BASE._text(detail["typed_outcome_id"], "typed outcome"),
            "positive_outcome": _BASE._boolean(detail["positive_outcome"], "positive outcome"),
            "settlement_id": _BASE._text(mechanism["settlement_id"], "settlement ID"),
            "settlement_payload_sha256": _BASE._sha256(
                mechanism["settlement_payload_sha256"], "settlement payload sha"
            ),
        },
        "prediction_error": {
            "social_prediction_error_snapshot_sha256": _BASE._sha256(
                mechanism["social_prediction_error_snapshot_sha256"], "PE snapshot sha"
            ),
        },
        "credit": {
            "credit_record_id": _BASE._text(mechanism["credit_record_id"], "credit record ID"),
            "credit_applied_to_gate": _BASE._boolean(mechanism["credit_applied_to_gate"], "credit applied"),
        },
    }
    return _BASE._with_row_id(row)


def _replay_all_transitions(
    reader,
    vectors: Mapping[str, tuple[float, ...]],
    windows: Mapping[str, tuple[int, int]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    subject_scopes = sorted(path.name for path in (reader.root / "chains").iterdir() if path.is_dir())
    _BASE._require_equal(len(subject_scopes), 8, "attempt subject scope count")
    rows: list[dict[str, object]] = []
    counts = {
        arm_id: Counter()
        for arm_id in _BASE._ARMS
    }
    owner_continuity_edges = {arm_id: 0 for arm_id in _BASE._ARMS}
    gate_continuity_edges = {arm_id: 0 for arm_id in _BASE._ARMS}
    frozen_reset_count = 0
    credit_record_ids: dict[str, list[str]] = {arm_id: [] for arm_id in _BASE._ARMS}
    cross_arm_credit_pairs_equal = 0
    action_divergent_keys: list[tuple[str, str]] = []
    per_key_credit: dict[tuple[str, str], dict[str, str]] = {}
    per_key_action: dict[tuple[str, str], dict[str, str]] = {}

    for subject_scope in subject_scopes:
        boundary_shas: dict[str, str] = {}
        for arm_id in _BASE._ARMS:
            chain, records = _BASE._load_chain_summaries(
                reader,
                subject_scope=subject_scope,
                arm_id=arm_id,
            )
            _BASE._require_equal(
                chain.get("appendable_reset_basis"),
                _EXPECTED_ONBOARDING_BOUNDARY_BASIS[arm_id],
                f"{subject_scope}/{arm_id} appendable reset basis",
            )
            boundary_shas[arm_id] = _BASE._sha256(
                chain.get("onboarding_boundary_sha256"),
                f"{subject_scope}/{arm_id} onboarding boundary",
            )
            ordered = sorted(
                records.values(),
                key=lambda item: _BASE._integer(item["decision_index"], "summary decision index"),
            )
            _BASE._require_equal(len(ordered), 24, f"{subject_scope}/{arm_id} decision count")
            previous_post_sha: str | None = None
            previous_gate_after: int | None = None
            first_pre_sha: str | None = None
            for summary in ordered:
                detail = _BASE._detailed_arm_decision(reader=reader, summary=summary, vectors=vectors)
                decision_index = _BASE._integer(summary["decision_index"], "summary decision index")
                segment = _BASE._segment_for_index(decision_index, windows)
                row = _compact_row(detail, segment=segment)
                chain_record = _BASE._object(summary["chain_record"], "chain decision record")
                writeback = row["writeback_chain"]
                _BASE._require_equal(
                    chain_record.get("pre_owner_snapshot_sha256"),
                    writeback["pre_owner_payload_sha256"],
                    f"{subject_scope}/{arm_id}[{decision_index}] chain pre owner sha",
                )
                _BASE._require_equal(
                    chain_record.get("post_owner_snapshot_sha256"),
                    writeback["post_owner_payload_sha256"],
                    f"{subject_scope}/{arm_id}[{decision_index}] chain post owner sha",
                )
                gate_before = row["gate"]["update_count_before"]
                gate_after = row["gate"]["update_count_after"]
                _BASE._require_equal(
                    gate_before,
                    decision_index,
                    f"{subject_scope}/{arm_id}[{decision_index}] gate baseline law",
                )
                credit_applied = row["credit"]["credit_applied_to_gate"]
                _BASE._require_equal(
                    gate_after,
                    gate_before + (1 if credit_applied else 0),
                    f"{subject_scope}/{arm_id}[{decision_index}] gate increment law",
                )
                if previous_gate_after is not None:
                    _BASE._require_equal(
                        gate_before,
                        previous_gate_after,
                        f"{subject_scope}/{arm_id}[{decision_index}] gate continuity edge",
                    )
                    gate_continuity_edges[arm_id] += 1
                previous_gate_after = gate_after
                pre_sha = writeback["pre_owner_payload_sha256"]
                if first_pre_sha is None:
                    first_pre_sha = pre_sha
                if arm_id == _BASE._FULL_ARM:
                    if previous_post_sha is not None:
                        _BASE._require_equal(
                            pre_sha,
                            previous_post_sha,
                            f"{subject_scope}/{arm_id}[{decision_index}] owner continuity edge",
                        )
                        owner_continuity_edges[arm_id] += 1
                    previous_post_sha = writeback["post_owner_payload_sha256"]
                else:
                    _BASE._require_equal(
                        pre_sha,
                        first_pre_sha,
                        f"{subject_scope}/{arm_id}[{decision_index}] frozen reset law",
                    )
                    frozen_reset_count += 1
                arm_counts = counts[arm_id]
                arm_counts["decision_count"] += 1
                arm_counts["owner_loaded_count"] += 1
                arm_counts["pe_receipt_count"] += 1
                arm_counts["named_readout_count"] += 1
                arm_counts["credit_applied_count"] += int(credit_applied)
                arm_counts["credit_withheld_count"] += int(not credit_applied)
                arm_counts["gate_update_increment_count"] += int(gate_after == gate_before + 1)
                arm_counts["forecast_byte_exact_count"] += int(
                    row["forecast"]["canonical_payload_byte_exact_with_receipt"]
                )
                arm_counts["hard_window_law_count"] += int(
                    writeback["observed_post_matches_hard_window"]
                )
                arm_counts["positive_outcome_count"] += int(row["settlement"]["positive_outcome"])
                credit_record_ids[arm_id].append(row["credit"]["credit_record_id"])
                key = (subject_scope, row["decision_id"])
                per_key_credit.setdefault(key, {})[arm_id] = row["credit"]["credit_record_id"]
                per_key_action.setdefault(key, {})[arm_id] = row["settlement"]["selected_action_id"]
                rows.append(row)
        _BASE._require_equal(
            boundary_shas[_BASE._FULL_ARM],
            boundary_shas[_BASE._FROZEN_ARM],
            f"{subject_scope} cross-arm onboarding boundary",
        )

    for key, arm_credit in sorted(per_key_credit.items()):
        _BASE._require_equal(sorted(arm_credit), sorted(_BASE._ARMS), f"{key} credit arm coverage")
        if arm_credit[_BASE._FULL_ARM] == arm_credit[_BASE._FROZEN_ARM]:
            cross_arm_credit_pairs_equal += 1
        actions = per_key_action[key]
        if actions[_BASE._FULL_ARM] != actions[_BASE._FROZEN_ARM]:
            action_divergent_keys.append(key)

    for arm_id in _BASE._ARMS:
        ids = credit_record_ids[arm_id]
        _BASE._require_equal(len(ids), 192, f"{arm_id} credit record cardinality")
        _BASE._require_equal(len(set(ids)), 192, f"{arm_id} credit record uniqueness")

    summary = {
        "per_arm_counts": {arm_id: dict(sorted(counts[arm_id].items())) for arm_id in _BASE._ARMS},
        "owner_continuity_edges": dict(owner_continuity_edges),
        "gate_continuity_edges": dict(gate_continuity_edges),
        "frozen_reset_count": frozen_reset_count,
        "cross_arm_credit_record_id_equal_pair_count": cross_arm_credit_pairs_equal,
        "full_vs_frozen_action_divergent_decision_count": len(action_divergent_keys),
    }
    return rows, summary


def _reconcile_with_frozen_mechanism_evidence(
    frozen_report: Mapping[str, object],
    summary: Mapping[str, object],
) -> dict[str, object]:
    mechanism = _BASE._object(frozen_report.get("mechanism_evidence"), "frozen mechanism evidence")
    per_arm = {
        _BASE._text(entry.get("arm_id"), "mechanism arm ID"): _BASE._object(entry, "mechanism arm entry")
        for entry in _BASE._array(mechanism.get("per_arm"), "mechanism per-arm entries")
    }
    replayed = _BASE._object(summary["per_arm_counts"], "replayed per-arm counts")
    expectations = {
        _BASE._FULL_ARM: {
            "owner_continuity_transition_count": 184,
            "frozen_owner_reset_count": 0,
        },
        _BASE._FROZEN_ARM: {
            "owner_continuity_transition_count": 0,
            "frozen_owner_reset_count": 192,
        },
    }
    checks: dict[str, object] = {}
    for arm_id in _BASE._ARMS:
        frozen_arm = per_arm[arm_id]
        replay_arm = _BASE._object(replayed[arm_id], f"replayed counts {arm_id}")
        for field in (
            "decision_count",
            "owner_loaded_count",
            "pe_receipt_count",
            "named_readout_count",
            "credit_applied_count",
            "credit_withheld_count",
            "gate_update_increment_count",
        ):
            _BASE._require_equal(
                replay_arm.get(field),
                frozen_arm.get(field),
                f"{arm_id} replay/frozen {field}",
            )
        _BASE._require_equal(frozen_arm.get("unnamed_readout_count"), 0, f"{arm_id} unnamed readout count")
        continuity = _BASE._object(summary["owner_continuity_edges"], "continuity edges")
        _BASE._require_equal(
            continuity.get(arm_id) if arm_id == _BASE._FULL_ARM else 0,
            expectations[arm_id]["owner_continuity_transition_count"],
            f"{arm_id} replayed owner continuity edges",
        )
        _BASE._require_equal(
            frozen_arm.get("owner_continuity_transition_count"),
            expectations[arm_id]["owner_continuity_transition_count"],
            f"{arm_id} frozen owner continuity count",
        )
        _BASE._require_equal(
            frozen_arm.get("frozen_owner_reset_count"),
            expectations[arm_id]["frozen_owner_reset_count"],
            f"{arm_id} frozen reset count",
        )
        checks[arm_id] = {
            "replayed_counts": dict(replay_arm),
            "frozen_mechanism_counts": {
                field: frozen_arm.get(field)
                for field in (
                    "decision_count",
                    "owner_loaded_count",
                    "pe_receipt_count",
                    "named_readout_count",
                    "unnamed_readout_count",
                    "credit_applied_count",
                    "credit_withheld_count",
                    "gate_update_increment_count",
                    "owner_continuity_transition_count",
                    "frozen_owner_reset_count",
                )
            },
            "exact_match": True,
        }
    _BASE._require_equal(summary.get("frozen_reset_count"), 192, "replayed frozen reset total")
    _BASE._require_equal(
        summary.get("full_vs_frozen_action_divergent_decision_count"),
        36,
        "replayed action divergence total",
    )
    divergence = {
        entry.get("comparator"): entry
        for entry in _BASE._array(
            mechanism.get("action_divergence_vs_full"), "mechanism action divergence"
        )
    }
    frozen_divergence = _BASE._object(
        divergence.get(_BASE._FROZEN_ARM), "frozen action divergence entry"
    )
    _BASE._require_equal(
        frozen_divergence.get("action_divergence_count"),
        36,
        "frozen mechanism action divergence count",
    )
    return {
        "per_arm": checks,
        "action_divergence_count_matches_frozen_report": True,
        "gate_interior_states": "rederived_endpoint_bound",
    }


def build_audit_documents(attempt_root: Path) -> dict[str, bytes]:
    """Build every output in memory, failing before any write on drift."""

    reader = _BASE._AttemptReader(attempt_root)
    _protocol, frozen_report, windows, source_provenance = _BASE._validate_attempt_authority(reader)
    vectors, embedding_table, reader_artifact = _BASE._load_embedding_table(reader)
    rows, summary = _replay_all_transitions(reader, vectors, windows)
    _BASE._require_equal(len(rows), 384, "replayed arm-decision row count")
    reconciliation = _reconcile_with_frozen_mechanism_evidence(frozen_report, summary)

    base_module_reference = {
        "path": _BASE._repo_relative(_BASE_MODULE_PATH, label="base audit module"),
        "bytes": _BASE_MODULE_RAW_BYTES,
        "raw_sha256": _BASE_MODULE_RAW_SHA256,
        "reused_for": "per-decision contract validation, forecast recomputation, owner transitions",
    }
    audit_script_path = Path(__file__).resolve()
    audit_script_raw = audit_script_path.read_bytes()
    audit_script_reference = {
        "path": _BASE._repo_relative(audit_script_path, label="audit script"),
        "bytes": len(audit_script_raw),
        "raw_sha256": hashlib.sha256(audit_script_raw).hexdigest(),
        "standard_library_only": True,
    }

    rows_document, rows_raw = _BASE._artifact_document(
        "relationship-product-horizon-attempt03-writeback-replay-rows.v1",
        audit_scope=_AUDIT_SCOPE,
        source_protocol_id=_BASE._EXPECTED_PROTOCOL_ID,
        record_identity_fields=["arm_id", "subject_scope", "decision_id"],
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
        "coverage": {
            "arm_ids": list(_BASE._ARMS),
            "matched_world_count": 8,
            "decisions_per_world": 24,
            "matched_decision_count": 192,
            "replayed_arm_decision_count": 384,
            "exact_forecast_recomputation_count": 384,
            "full_owner_continuity_edge_count": summary["owner_continuity_edges"][_BASE._FULL_ARM],
            "frozen_owner_reset_count": summary["frozen_reset_count"],
            "gate_continuity_edge_count_per_arm": summary["gate_continuity_edges"],
            "gate_continuity_edge_count_total": sum(summary["gate_continuity_edges"].values()),
            "prior_owner_history_audit_covered_arm_decisions": 72,
            "newly_covered_arm_decisions": 384 - 72,
        },
        "writeback_laws_verified": {
            "settlement_to_pe": "postaction receipt binds settlement payload and PE snapshot per decision",
            "pe_to_credit": "each decision has exactly one credit record; applied flag replayed",
            "credit_to_gate": "gate update count increments by one exactly when credit is applied",
            "gate_to_owner": "post owner equals source-order hard window of pre owner plus current evidence",
            "owner_to_next_tick_full": "full-arm pre owner payload equals prior decision post owner payload",
            "owner_to_next_tick_frozen": "frozen-arm pre owner payload equals the frozen onboarding boundary",
            "gate_baseline": "gate update count before decision t equals t in both arms",
        },
        "replay_summary": summary,
        "frozen_mechanism_reconciliation": reconciliation,
        "mechanical_replay": {
            "public_embedding_table": {
                **reader.reference(_BASE._EMBEDDING_TABLE_PATH, embedding_table),
                "artifact_id": _BASE._EXPECTED_EMBEDDING_TABLE_ID,
            },
            "condition_reader_artifact": {
                **reader_artifact,
                "artifact_id": _BASE._EXPECTED_READER_ARTIFACT_ID,
            },
            "receipt_canonical_payload_byte_exact_count": 384,
            "model_output_count": 0,
            "cuda_used": False,
            "network_used": False,
        },
        "gate_interior_state_note": (
            "Only gate update counters and the endpoint checkpoint are materialized in attempt03. "
            "Interior gate parameter states are rederived_endpoint_bound and are not claimed as "
            "directly observed history."
        ),
        "audit_implementation": audit_script_reference,
        "base_module": base_module_reference,
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
            "mechanical_writeback_replay_only": True,
            "product_causal_effect_established": False,
            "learnable_capability_established": False,
            "appendable_capability_established": False,
            "reader_error_attribution_authorized": False,
            "human_product_validation": False,
            "production_active": False,
            "evidence_tier": "development",
            "claim_ceiling": (
                "Every full/frozen settlement->PE->credit->gate->owner->next-tick transition in "
                "attempt03 replays mechanically and reconciles with the frozen mechanism-evidence "
                "counts. This establishes writeback bookkeeping integrity only; it does not "
                "identify a causal owner of the outcome differences and does not establish any "
                "of the four capability axes."
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
        "relationship-product-horizon-attempt03-writeback-replay-report.v1",
        **report_payload,
    )
    manifest_document, manifest_raw = _BASE._artifact_document(
        "relationship-product-horizon-attempt03-writeback-replay-manifest.v1",
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
        "replayed_arm_decision_count": 384,
        "full_owner_continuity_edge_count": 184,
        "frozen_owner_reset_count": 192,
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
        "replayed_arm_decision_count": 384,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-free attempt03 full-coverage writeback replay audit",
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
