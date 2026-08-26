"""Sequential source-v4 collection-prefix gate for Product Horizon.

This development-only workflow keeps the theta0-v2 policy immutable while it
runs the eight collection decisions of every source-v4 root in chronological
order.  It proves the outcome-free first preaction is identical to the frozen
public scanner, then carries each actual settlement owner state into the next
decision.  It estimates no arm or product effect.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, replace
import pathlib
from typing import Mapping

from lifeform_domain_emogpt.lab.contracts import sha256_json
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    HorizonPublicDecisionSession,
    HorizonPublicRoot,
    RelationshipProductHorizonPublicView,
    build_relationship_product_horizon_evaluator_bundle,
    load_relationship_product_horizon_source_protocol,
)
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductExecutorDisposition,
    RelationshipProductFrozenPreActionSnapshot,
    settle_relationship_product_frozen_pulse,
)
from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from lifeform_evolution.relationship_product_horizon_source_admission import (
    build_relationship_product_horizon_source_action_commitment,
)
from lifeform_evolution import relationship_product_horizon_theta0_calibration as cal
from lifeform_evolution import (
    relationship_product_horizon_transductive_public_opportunity as scan,
)
from volvence_zero.social import social_record_store_persistence_sha256
from volvence_zero.social_cognition import preference_action_forecast_to_payload


DYNAMIC_COLLECTION_PREFIX_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-dynamic-collection-prefix-protocol.v1"
)
DYNAMIC_COLLECTION_PREFIX_TRACE_SCHEMA_VERSION = (
    "relationship-product-horizon-dynamic-collection-prefix-trace.v1"
)
DYNAMIC_COLLECTION_PREFIX_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-dynamic-collection-prefix-manifest.v1"
)

_PROTOCOL_FILENAME = "relationship_product_horizon_dynamic_collection_prefix_v1.json"
_TRACE_FILENAME = "dynamic_collection_prefix.jsonl"
_OUTPUT_FILES = frozenset({"protocol.json", _TRACE_FILENAME, "manifest.json"})
_SUCCESS_STATUS = (
    "development_dynamic_collection_prefix_closed_"
    "forced_batch_protocol_freeze_authorized_effect_not_tested"
)
_FAIL_STATUS = (
    "development_dynamic_collection_prefix_gate_failed_"
    "campaign_blocked_effect_not_tested"
)
_CREDIT_TIMESTAMP_FORMULA = (
    "root_sequence_index_times_20_plus_5_plus_2_times_decision_index"
)


@dataclass(frozen=True)
class RelationshipProductHorizonDynamicCollectionPrefixProtocol:
    payload: Mapping[str, object]
    raw_bytes: bytes
    protocol_id: str
    raw_sha256: str


@dataclass(frozen=True)
class _Dependencies:
    protocol: RelationshipProductHorizonDynamicCollectionPrefixProtocol
    scanner_dependencies: scan._Dependencies
    scanner_manifest: Mapping[str, object]
    expected_first_projections: tuple[Mapping[str, object], ...]
    source_v4_admission_root: pathlib.Path

    @property
    def public_view(self) -> RelationshipProductHorizonPublicView:
        return self.scanner_dependencies.public_view


@dataclass(frozen=True)
class _SafeSelectedBranchOutcome:
    environment_subject_id: str
    selected_action_id: str
    typed_outcome_id: str
    rendered_user_reaction: str
    environment_evidence_ref: str
    environment_version: str
    commitment_id: str


@dataclass(frozen=True)
class _DynamicReplay:
    completed_root_count: int
    onboarding_count: int
    preaction_count: int
    postaction_count: int
    first_preaction_exact_match_count: int
    later_owner_handoff_count: int
    owner_writeback_change_count: int
    selected_branch_resolution_count: int
    selected_branch_commitment_match_count: int
    unique_selected_branch_commitment_count: int
    prediction_error_count: int
    credit_count: int
    unique_credit_count: int
    unique_settlement_count: int
    unique_environment_evidence_ref_count: int
    cold_checkpoint_unchanged_count: int
    temporal_delivered_action_counts: Mapping[str, int]
    temporal_delivered_nonnoop_count: int
    temporal_delivered_nonnoop_root_count: int
    first_preaction_projection_sha256: str
    terminal_failure_reasons: tuple[str, ...]
    terminal_status: str


def relationship_product_horizon_dynamic_collection_prefix_protocol_path(
) -> pathlib.Path:
    return pathlib.Path(__file__).with_name("protocols") / _PROTOCOL_FILENAME


def load_relationship_product_horizon_dynamic_collection_prefix_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonDynamicCollectionPrefixProtocol:
    source = pathlib.Path(
        path or relationship_product_horizon_dynamic_collection_prefix_protocol_path()
    )
    raw = source.read_bytes()
    payload = cal._parse_json_bytes(raw, source="dynamic collection-prefix protocol")
    cal._exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "purpose",
            "adaptive_lineage",
            "upstream_scanner",
            "source_v4_environment",
            "runtime_inputs",
            "collection_prefix",
            "runtime_order",
            "terminal_gates",
            "causal_firewall",
            "claims",
            "claim_boundary",
        },
        "dynamic collection-prefix protocol",
    )
    if (
        payload["schema_version"]
        != DYNAMIC_COLLECTION_PREFIX_PROTOCOL_SCHEMA_VERSION
        or payload["evidence_tier"] != "development"
        or payload["owner"]
        != (
            "lifeform_evolution."
            "relationship_product_horizon_dynamic_collection_prefix"
        )
        or payload["purpose"]
        != "natural_apply_sequential_dynamic_collection_prefix_gate"
    ):
        raise ValueError("dynamic collection-prefix protocol identity drifted")
    _validate_protocol(payload)
    return RelationshipProductHorizonDynamicCollectionPrefixProtocol(
        payload=payload,
        raw_bytes=raw,
        protocol_id=sha256_json(payload),
        raw_sha256=cal._sha256_bytes(raw),
    )


def _validate_protocol(payload: Mapping[str, object]) -> None:
    adaptive = cal._mapping(payload["adaptive_lineage"], "adaptive_lineage")
    scanner_pin = cal._mapping(payload["upstream_scanner"], "upstream_scanner")
    source = cal._mapping(payload["source_v4_environment"], "source_v4_environment")
    inputs = cal._mapping(payload["runtime_inputs"], "runtime_inputs")
    collection = cal._mapping(payload["collection_prefix"], "collection_prefix")
    order = cal._mapping(payload["runtime_order"], "runtime_order")
    terminal = cal._mapping(payload["terminal_gates"], "terminal_gates")
    firewall = cal._mapping(payload["causal_firewall"], "causal_firewall")
    claims = cal._mapping(payload["claims"], "claims")
    expected_keys = {
        "adaptive_lineage": {
            "source_v4_public_previously_observed",
            "development_reader_unqualified",
            "theta0_v2_trained_on_already_used_source_v3",
            "upstream_opportunity_scan_transductive",
            "transductive",
            "unseen",
            "confirmatory",
            "formal",
        },
        "upstream_scanner": {
            "protocol_id",
            "protocol_raw_sha256",
            "artifact_id",
            "manifest_raw_sha256",
            "trace_relative_path",
            "trace_raw_sha256",
            "terminal_status",
            "first_preaction_projection_schema_version",
            "first_preaction_projection_fields",
            "first_preaction_projection_sha256",
            "first_preaction_projection_count",
            "collection_prefix_protocol_freeze_authorized",
            "collection_prefix_execution_authorized",
        },
        "source_v4_environment": {
            "admission_protocol_id",
            "admission_artifact_id",
            "admission_manifest_raw_sha256",
            "source_protocol_relative_path",
            "source_protocol_raw_sha256",
            "source_protocol_id",
            "sealed_evaluator_relative_path",
            "sealed_evaluator_raw_sha256",
            "sealed_evaluator_bundle_sha256",
            "commitment_index_relative_path",
            "commitment_index_raw_sha256",
            "commitment_index_schema_version",
            "commitment_count",
            "root_count",
            "onboarding_session_count",
            "collection_decision_count",
            "environment_seed_owner",
            "selected_action_is_in_draw_hash",
            "common_random_number_design",
        },
        "runtime_inputs": {
            "development_reader_artifact_id",
            "embedding_table_artifact_id",
            "reader_artifact_id",
            "condition_reader_qualified",
            "theta0_v2_artifact_id",
            "cold_checkpoint_content_sha256",
            "cold_frozen_policy_id",
            "cold_update_count",
            "cold_processed_credit_id_count",
            "cold_pending_decision_count",
        },
        "collection_prefix": {
            "root_order",
            "root_count",
            "owner_reset_each_root",
            "onboarding_once_per_root",
            "onboarding_session_count_per_root",
            "decision_order",
            "decision_count_per_root",
            "decision_count",
            "first_preaction_state",
            "later_preaction_state",
            "later_handoff_count",
            "executor_disposition",
            "environment_settles_only_temporal_delivered_action",
            "frozen_policy_immutable_across_collection",
            "prediction_error_derived_from_actual_outcome",
            "credit_derived_from_prediction_error",
            "credit_timestamp_formula",
            "credit_timestamps_strictly_increasing",
            "credit_applied_online",
            "gate_update_count",
            "evaluation_decision_count",
            "arm_count",
            "effect_estimand_count",
        },
        "runtime_order": {
            "append_only_trace_create_only",
            "preaction_record_append_only",
            "preaction_record_fsync_before_environment_scope_creation",
            "environment_scope_created_after_first_preaction_fsync",
            "current_preaction_fsync_before_selected_branch_open",
            "selected_branch_opened_by_source_owner_after_actual_action",
            "commitment_checked_before_owner_settlement",
            "postaction_record_fsync_before_next_preaction",
            "sealed_truth_passed_to_forecast_reader_gate_or_executor",
            "incomplete_root_without_manifest_is_evidence",
        },
        "terminal_gates": {
            "first_preaction_exact_match_count",
            "preaction_count",
            "postaction_count",
            "later_owner_handoff_count",
            "owner_writeback_change_count",
            "selected_branch_resolution_count",
            "selected_branch_commitment_match_count",
            "unique_selected_branch_commitment_count",
            "prediction_error_count",
            "credit_count",
            "unique_credit_count",
            "unique_settlement_count",
            "unique_environment_evidence_ref_count",
            "credit_applied_to_gate_count",
            "gate_update_count",
            "cold_checkpoint_unchanged_count",
            "forced_executor_count",
            "forced_schedule_count",
            "batch_apply_count",
            "branch_resolution_before_current_preaction_fsync_count",
            "excluded_root_count",
            "duplicate_root_count",
            "temporal_delivered_nonnoop_min_count",
            "successful_terminal",
            "failed_terminal",
            "forced_common_batch_protocol_freeze_authorized_on_success",
            "forced_common_batch_execution_authorized_on_success",
            "evaluation_protocol_freeze_authorized_on_success",
            "campaign_execution_authorized_on_success",
            "scientific_retry_with_modified_root_theta_reader_source_order_or_gate_forbidden",
        },
        "causal_firewall": {
            "source_v4_public_plan_file_read_count",
            "source_v4_sealed_file_read_count",
            "source_v4_source_protocol_file_read_count",
            "upstream_scanner_trace_file_read_count",
            "challenge_label_file_read_count",
            "group_split_file_read_count",
            "evaluation_decision_count",
            "evaluation_or_judge_feedback_count",
            "gate_update_count",
            "model_output_count",
            "cuda_execution_count",
            "human_sample_count",
        },
        "claims": {
            "dynamic_collection_gate_execution_authorized",
            "dynamic_collection_gate_completed",
            "first_preaction_exact_match_verified",
            "sequential_owner_writeback_verified",
            "selected_branch_commitment_verified",
            "pe_credit_chain_verified",
            "forced_common_batch_protocol_freeze_authorized",
            "forced_common_batch_execution_authorized",
            "evaluation_protocol_freeze_authorized",
            "campaign_execution_authorized",
            "reader_qualified",
            "formal_evidence_authorized",
            "unseen_evidence_authorized",
            "integrated_horizon_authorized",
            "appendable_effect",
            "readable_effect",
            "learnable_effect",
            "steerable_effect",
            "four_able_complete",
            "human_validation_complete",
            "production_active",
        },
    }
    for name, value in (
        ("adaptive_lineage", adaptive),
        ("upstream_scanner", scanner_pin),
        ("source_v4_environment", source),
        ("runtime_inputs", inputs),
        ("collection_prefix", collection),
        ("runtime_order", order),
        ("terminal_gates", terminal),
        ("causal_firewall", firewall),
        ("claims", claims),
    ):
        cal._exact_keys(value, expected_keys[name], name)

    for field in (
        "source_v4_public_previously_observed",
        "development_reader_unqualified",
        "theta0_v2_trained_on_already_used_source_v3",
        "upstream_opportunity_scan_transductive",
        "transductive",
    ):
        if not cal._boolean(adaptive[field], f"adaptive_lineage.{field}"):
            raise ValueError(f"adaptive_lineage.{field} must remain true")
    for field in ("unseen", "confirmatory", "formal"):
        if cal._boolean(adaptive[field], f"adaptive_lineage.{field}"):
            raise ValueError(f"adaptive_lineage.{field} must remain false")

    digest_fields = (
        (scanner_pin, "protocol_id"),
        (scanner_pin, "protocol_raw_sha256"),
        (scanner_pin, "artifact_id"),
        (scanner_pin, "manifest_raw_sha256"),
        (scanner_pin, "trace_raw_sha256"),
        (scanner_pin, "first_preaction_projection_sha256"),
        (source, "admission_protocol_id"),
        (source, "admission_artifact_id"),
        (source, "admission_manifest_raw_sha256"),
        (source, "source_protocol_raw_sha256"),
        (source, "source_protocol_id"),
        (source, "sealed_evaluator_raw_sha256"),
        (source, "sealed_evaluator_bundle_sha256"),
        (source, "commitment_index_raw_sha256"),
        (inputs, "development_reader_artifact_id"),
        (inputs, "embedding_table_artifact_id"),
        (inputs, "reader_artifact_id"),
        (inputs, "cold_checkpoint_content_sha256"),
    )
    for container, field in digest_fields:
        cal._digest(container[field], field)
    for container, field in (
        (inputs, "theta0_v2_artifact_id"),
        (inputs, "cold_frozen_policy_id"),
    ):
        cal._text(container[field], field)
    expected_scanner = {
        "trace_relative_path": "public_opportunity_scan.jsonl",
        "terminal_status": (
            "development_transductive_public_opportunity_present_"
            "collection_prefix_protocol_freeze_authorized_effect_not_tested"
        ),
        "first_preaction_projection_schema_version": (
            scan._REACHABLE_FIRST_PROJECTION_SCHEMA_VERSION
        ),
        "first_preaction_projection_count": 112,
    }
    for field, expected in expected_scanner.items():
        if scanner_pin[field] != expected:
            raise ValueError(f"upstream_scanner.{field} drifted")
    if scanner_pin["first_preaction_projection_fields"] != list(
        scan._REACHABLE_FIRST_PROJECTION_FIELDS
    ):
        raise ValueError("upstream scanner first-preaction fields drifted")
    if not cal._boolean(
        scanner_pin["collection_prefix_protocol_freeze_authorized"],
        "upstream_scanner.collection_prefix_protocol_freeze_authorized",
    ) or cal._boolean(
        scanner_pin["collection_prefix_execution_authorized"],
        "upstream_scanner.collection_prefix_execution_authorized",
    ):
        raise ValueError("upstream scanner authority boundary drifted")

    expected_source_strings = {
        "source_protocol_relative_path": "source/source_protocol.json",
        "sealed_evaluator_relative_path": "sealed/evaluator_bundle.json",
        "commitment_index_relative_path": (
            "sealed/action_counterfactual_commitment_index.json"
        ),
        "commitment_index_schema_version": (
            "relationship-product-horizon-action-counterfactual-commitments.v1"
        ),
        "environment_seed_owner": "sealed_evaluator_decision_environment_seed",
    }
    for field, expected in expected_source_strings.items():
        if source[field] != expected:
            raise ValueError(f"source_v4_environment.{field} drifted")
    expected_counts = {
        ("source", "commitment_count"): 16128,
        ("source", "root_count"): 112,
        ("source", "onboarding_session_count"): 448,
        ("source", "collection_decision_count"): 896,
        ("inputs", "cold_update_count"): 0,
        ("inputs", "cold_processed_credit_id_count"): 0,
        ("inputs", "cold_pending_decision_count"): 0,
        ("collection", "root_count"): 112,
        ("collection", "onboarding_session_count_per_root"): 4,
        ("collection", "decision_count_per_root"): 8,
        ("collection", "decision_count"): 896,
        ("collection", "later_handoff_count"): 784,
        ("collection", "gate_update_count"): 0,
        ("collection", "evaluation_decision_count"): 0,
        ("collection", "arm_count"): 0,
        ("collection", "effect_estimand_count"): 0,
    }
    mappings = {"source": source, "inputs": inputs, "collection": collection}
    for (name, field), expected in expected_counts.items():
        if cal._integer(mappings[name][field], f"{name}.{field}") != expected:
            raise ValueError(f"{name}.{field} drifted")
    if cal._boolean(inputs["condition_reader_qualified"], "condition_reader_qualified"):
        raise ValueError("development reader must remain unqualified")
    if not cal._boolean(source["selected_action_is_in_draw_hash"], "selected_action_is_in_draw_hash"):
        raise ValueError("selected action must remain in the environment draw hash")
    if cal._boolean(source["common_random_number_design"], "common_random_number_design"):
        raise ValueError("source-v4 must not be called a common-random-number design")

    expected_collection_strings = {
        "root_order": "source_v4_public_root_array_order",
        "decision_order": "decision_index_0_through_7",
        "first_preaction_state": "post_four_public_onboarding_sessions",
        "later_preaction_state": "exact_prior_postaction_owner_persistence",
        "executor_disposition": "apply_candidate",
        "credit_timestamp_formula": _CREDIT_TIMESTAMP_FORMULA,
    }
    for field, expected in expected_collection_strings.items():
        if collection[field] != expected:
            raise ValueError(f"collection_prefix.{field} drifted")
    for field in (
        "owner_reset_each_root",
        "onboarding_once_per_root",
        "environment_settles_only_temporal_delivered_action",
        "frozen_policy_immutable_across_collection",
        "prediction_error_derived_from_actual_outcome",
        "credit_derived_from_prediction_error",
        "credit_timestamps_strictly_increasing",
    ):
        if not cal._boolean(collection[field], f"collection_prefix.{field}"):
            raise ValueError(f"collection_prefix.{field} must remain true")
    if cal._boolean(collection["credit_applied_online"], "credit_applied_online"):
        raise ValueError("collection credit must remain withheld from the gate")

    for field in (
        "append_only_trace_create_only",
        "preaction_record_append_only",
        "preaction_record_fsync_before_environment_scope_creation",
        "environment_scope_created_after_first_preaction_fsync",
        "current_preaction_fsync_before_selected_branch_open",
        "selected_branch_opened_by_source_owner_after_actual_action",
        "commitment_checked_before_owner_settlement",
        "postaction_record_fsync_before_next_preaction",
    ):
        if not cal._boolean(order[field], f"runtime_order.{field}"):
            raise ValueError(f"runtime_order.{field} must remain true")
    for field in (
        "sealed_truth_passed_to_forecast_reader_gate_or_executor",
        "incomplete_root_without_manifest_is_evidence",
    ):
        if cal._boolean(order[field], f"runtime_order.{field}"):
            raise ValueError(f"runtime_order.{field} must remain false")

    expected_terminal_counts = {
        "first_preaction_exact_match_count": 112,
        "preaction_count": 896,
        "postaction_count": 896,
        "later_owner_handoff_count": 784,
        "owner_writeback_change_count": 896,
        "selected_branch_resolution_count": 896,
        "selected_branch_commitment_match_count": 896,
        "unique_selected_branch_commitment_count": 896,
        "prediction_error_count": 896,
        "credit_count": 896,
        "unique_credit_count": 896,
        "unique_settlement_count": 896,
        "unique_environment_evidence_ref_count": 896,
        "credit_applied_to_gate_count": 0,
        "gate_update_count": 0,
        "cold_checkpoint_unchanged_count": 896,
        "forced_executor_count": 0,
        "forced_schedule_count": 0,
        "batch_apply_count": 0,
        "branch_resolution_before_current_preaction_fsync_count": 0,
        "excluded_root_count": 0,
        "duplicate_root_count": 0,
        "temporal_delivered_nonnoop_min_count": 1,
    }
    for field, expected in expected_terminal_counts.items():
        if cal._integer(terminal[field], f"terminal_gates.{field}") != expected:
            raise ValueError(f"terminal_gates.{field} drifted")
    if (
        terminal["successful_terminal"] != _SUCCESS_STATUS
        or terminal["failed_terminal"] != _FAIL_STATUS
    ):
        raise ValueError("dynamic collection-prefix terminal statuses drifted")
    for field in (
        "forced_common_batch_protocol_freeze_authorized_on_success",
        "scientific_retry_with_modified_root_theta_reader_source_order_or_gate_forbidden",
    ):
        if not cal._boolean(terminal[field], f"terminal_gates.{field}"):
            raise ValueError(f"terminal_gates.{field} must remain true")
    for field in (
        "forced_common_batch_execution_authorized_on_success",
        "evaluation_protocol_freeze_authorized_on_success",
        "campaign_execution_authorized_on_success",
    ):
        if cal._boolean(terminal[field], f"terminal_gates.{field}"):
            raise ValueError(f"terminal_gates.{field} must remain false")

    expected_firewall_counts = {
        "source_v4_public_plan_file_read_count": 1,
        "source_v4_sealed_file_read_count": 2,
        "source_v4_source_protocol_file_read_count": 2,
        "upstream_scanner_trace_file_read_count": 1,
        "challenge_label_file_read_count": 0,
        "group_split_file_read_count": 0,
        "evaluation_decision_count": 0,
        "evaluation_or_judge_feedback_count": 0,
        "gate_update_count": 0,
        "model_output_count": 0,
        "cuda_execution_count": 0,
        "human_sample_count": 0,
    }
    for field, expected in expected_firewall_counts.items():
        if cal._integer(firewall[field], f"causal_firewall.{field}") != expected:
            raise ValueError(f"causal_firewall.{field} drifted")
    true_claims = {field for field, value in claims.items() if value is True}
    if true_claims != {"dynamic_collection_gate_execution_authorized"}:
        raise ValueError("dynamic collection-prefix protocol claim ceiling drifted")
    if any(type(value) is not bool for value in claims.values()):
        raise ValueError("dynamic collection-prefix claims must be booleans")
    cal._text(payload["claim_boundary"], "claim_boundary")


def _projection_from_scanner_record(
    record: Mapping[str, object],
    *,
    cold_frozen_policy_id: str,
) -> Mapping[str, object]:
    return {
        "schema_version": scan._REACHABLE_FIRST_PROJECTION_SCHEMA_VERSION,
        "root_sequence_index": record["root_sequence_index"],
        "subject_id": record["subject_id"],
        "decision_index": record["decision_index"],
        "session_id": record["session_id"],
        "decision_id": record["decision_id"],
        "owner_input_persistence_sha256": record[
            "owner_input_persistence_sha256"
        ],
        "owner_output_persistence_sha256": record[
            "owner_output_persistence_sha256"
        ],
        "forecast_sha256": record["forecast_sha256"],
        "frozen_decision_sha256": record["frozen_decision_sha256"],
        "gate_action": record["gate_action"],
        "steer_probability_hex": record["steer_probability_hex"],
        "frozen_selected_action_id": record["frozen_selected_action_id"],
        "executor_disposition": record["executor_disposition"],
        "executor_status": record["executor_status"],
        "candidate_advisory_id": record["candidate_advisory_id"],
        "delivered_advisory_id": record["delivered_advisory_id"],
        "temporal_delivered_action_id": record[
            "temporal_delivered_action_id"
        ],
        "temporal_controller_params_hash": record[
            "temporal_controller_params_hash"
        ],
        "temporal_action_family_version": record[
            "temporal_action_family_version"
        ],
        "temporal_action_advisory_status": record[
            "temporal_action_advisory_status"
        ],
        "cold_checkpoint_content_sha256": record[
            "cold_checkpoint_content_sha256"
        ],
        "cold_frozen_policy_id": cold_frozen_policy_id,
    }


def _load_scanner_first_projections(
    *,
    scanner_root: pathlib.Path,
    scanner_pin: Mapping[str, object],
    cold_frozen_policy_id: str,
) -> tuple[Mapping[str, object], ...]:
    trace_path = scanner_root / cal._text(
        scanner_pin["trace_relative_path"],
        "upstream_scanner.trace_relative_path",
    )
    raw = cal._require_raw_sha(
        trace_path,
        scanner_pin["trace_raw_sha256"],
        "upstream scanner trace",
    )
    if not raw.endswith(b"\n"):
        raise ValueError("upstream scanner trace must end with a newline")
    projections: list[Mapping[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        record = cal._parse_json_bytes(
            line,
            source=f"upstream scanner trace line {line_number}",
        )
        if line != cal._canonical_bytes(record):
            raise ValueError("upstream scanner trace lines must use canonical bytes")
        if record.get("record_type") == "probe" and record.get("decision_index") == 0:
            projection = _projection_from_scanner_record(
                record,
                cold_frozen_policy_id=cold_frozen_policy_id,
            )
            if tuple(projection) != scan._REACHABLE_FIRST_PROJECTION_FIELDS:
                raise ValueError("upstream first-preaction projection schema drifted")
            projections.append(projection)
    if len(projections) != 112:
        raise ValueError("upstream scanner must publish 112 first-preaction rows")
    if tuple(item["root_sequence_index"] for item in projections) != tuple(range(112)):
        raise ValueError("upstream first-preaction root order drifted")
    if sha256_json(projections) != scanner_pin["first_preaction_projection_sha256"]:
        raise ValueError("upstream first-preaction projection digest drifted")
    return tuple(projections)


def _load_dependencies(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
) -> _Dependencies:
    protocol = load_relationship_product_horizon_dynamic_collection_prefix_protocol()
    scanner_pin = cal._mapping(protocol.payload["upstream_scanner"], "upstream_scanner")
    scanner_dependencies = scan._load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
    )
    if (
        scanner_dependencies.protocol.protocol_id != scanner_pin["protocol_id"]
        or scanner_dependencies.protocol.raw_sha256
        != scanner_pin["protocol_raw_sha256"]
    ):
        raise ValueError("registered upstream scanner protocol drifted")
    inputs = cal._mapping(protocol.payload["runtime_inputs"], "runtime_inputs")
    if (
        scanner_dependencies.reader_artifact.artifact_id
        != inputs["reader_artifact_id"]
        or scanner_dependencies.embedding_table.artifact_id
        != inputs["embedding_table_artifact_id"]
        or scanner_dependencies.theta0.artifact_id != inputs["theta0_v2_artifact_id"]
        or scanner_dependencies.frozen_policy.policy_id
        != inputs["cold_frozen_policy_id"]
        or scanner_dependencies.frozen_policy.checkpoint.content_sha256
        != inputs["cold_checkpoint_content_sha256"]
    ):
        raise ValueError("dynamic collection-prefix runtime inputs drifted")
    root = pathlib.Path(scanner_root)
    manifest_raw = cal._require_raw_sha(
        root / "manifest.json",
        scanner_pin["manifest_raw_sha256"],
        "upstream scanner manifest",
    )
    scanner_manifest = cal._parse_json_bytes(
        manifest_raw,
        source="upstream scanner manifest",
    )
    if manifest_raw != cal._canonical_bytes(scanner_manifest):
        raise ValueError("upstream scanner manifest must use canonical bytes")
    if (
        scanner_manifest["protocol_id"] != scanner_pin["protocol_id"]
        or scanner_manifest["artifact_id"] != scanner_pin["artifact_id"]
        or scanner_manifest["status"] != scanner_pin["terminal_status"]
        or scanner_manifest["reachable_first_projection_sha256"]
        != scanner_pin["first_preaction_projection_sha256"]
        or scanner_manifest["cold_frozen_policy_id"]
        != inputs["cold_frozen_policy_id"]
        or scanner_manifest["claims"][
            "collection_prefix_protocol_freeze_authorized"
        ]
        is not True
        or scanner_manifest["claims"]["collection_prefix_execution_authorized"]
        is not False
    ):
        raise ValueError("upstream scanner authority envelope drifted")
    if scanner_manifest["artifact_id"] != sha256_json(
        {key: value for key, value in scanner_manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("upstream scanner manifest content identity drifted")
    scanner_files = scan._file_entries(scanner_manifest, source="upstream scanner")
    trace_relative = cal._text(
        scanner_pin["trace_relative_path"],
        "upstream_scanner.trace_relative_path",
    )
    if scanner_files[trace_relative]["raw_sha256"] != scanner_pin["trace_raw_sha256"]:
        raise ValueError("upstream scanner trace manifest pin drifted")
    projections = _load_scanner_first_projections(
        scanner_root=root,
        scanner_pin=scanner_pin,
        cold_frozen_policy_id=cal._text(
            scanner_manifest["cold_frozen_policy_id"],
            "scanner_manifest.cold_frozen_policy_id",
        ),
    )

    source_root = pathlib.Path(source_v4_admission_root)
    source_pin = cal._mapping(
        protocol.payload["source_v4_environment"],
        "source_v4_environment",
    )
    source_manifest_raw = cal._require_raw_sha(
        source_root / "manifest.json",
        source_pin["admission_manifest_raw_sha256"],
        "source-v4 admission manifest",
    )
    source_manifest = cal._parse_json_bytes(
        source_manifest_raw,
        source="source-v4 admission manifest",
    )
    if source_manifest_raw != cal._canonical_bytes(source_manifest):
        raise ValueError("source-v4 admission manifest must use canonical bytes")
    if (
        source_manifest["protocol_id"] != source_pin["admission_protocol_id"]
        or source_manifest["artifact_id"] != source_pin["admission_artifact_id"]
        or source_manifest["source_protocol_id"] != source_pin["source_protocol_id"]
        or source_manifest["sealed_bundle_sha256"]
        != source_pin["sealed_evaluator_bundle_sha256"]
        or source_manifest["status"]
        != "campaign_input_admitted_execution_not_authorized"
    ):
        raise ValueError("source-v4 admission environment envelope drifted")
    if source_manifest["artifact_id"] != sha256_json(
        {key: value for key, value in source_manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("source-v4 admission manifest content identity drifted")
    source_files = scan._file_entries(source_manifest, source="source-v4 admission")
    for path_field, hash_field in (
        ("source_protocol_relative_path", "source_protocol_raw_sha256"),
        ("sealed_evaluator_relative_path", "sealed_evaluator_raw_sha256"),
        ("commitment_index_relative_path", "commitment_index_raw_sha256"),
    ):
        relative = cal._text(source_pin[path_field], path_field)
        if source_files[relative]["raw_sha256"] != source_pin[hash_field]:
            raise ValueError(f"source-v4 admission file pin drifted: {relative}")
    return _Dependencies(
        protocol=protocol,
        scanner_dependencies=scanner_dependencies,
        scanner_manifest=scanner_manifest,
        expected_first_projections=projections,
        source_v4_admission_root=source_root,
    )


class _SelectedBranchEnvironmentScope:
    """Load sealed source only after the first preaction has been fsynced."""

    def __init__(self, *, dependencies: _Dependencies) -> None:
        source_pin = cal._mapping(
            dependencies.protocol.payload["source_v4_environment"],
            "source_v4_environment",
        )
        root = dependencies.source_v4_admission_root
        source_path = root / cal._text(
            source_pin["source_protocol_relative_path"],
            "source_protocol_relative_path",
        )
        cal._require_raw_sha(
            source_path,
            source_pin["source_protocol_raw_sha256"],
            "source-v4 source protocol",
        )
        source_protocol = load_relationship_product_horizon_source_protocol(source_path)
        if source_protocol.protocol_id != source_pin["source_protocol_id"]:
            raise ValueError("source-v4 source protocol identity drifted")

        evaluator_path = root / cal._text(
            source_pin["sealed_evaluator_relative_path"],
            "sealed_evaluator_relative_path",
        )
        evaluator_raw = cal._require_raw_sha(
            evaluator_path,
            source_pin["sealed_evaluator_raw_sha256"],
            "source-v4 sealed evaluator",
        )
        evaluator_payload = cal._parse_json_bytes(
            evaluator_raw,
            source="source-v4 sealed evaluator",
        )
        if evaluator_raw != cal._canonical_bytes(evaluator_payload):
            raise ValueError("source-v4 sealed evaluator must use canonical bytes")
        evaluator = build_relationship_product_horizon_evaluator_bundle(source_protocol)
        if (
            evaluator.sealed_bundle_sha256
            != source_pin["sealed_evaluator_bundle_sha256"]
            or cal._canonical_bytes(evaluator.to_payload()) != evaluator_raw
        ):
            raise ValueError("source-v4 sealed evaluator owner rebuild drifted")

        commitment_path = root / cal._text(
            source_pin["commitment_index_relative_path"],
            "commitment_index_relative_path",
        )
        commitment_raw = cal._require_raw_sha(
            commitment_path,
            source_pin["commitment_index_raw_sha256"],
            "source-v4 action commitment index",
        )
        commitment_payload = cal._parse_json_bytes(
            commitment_raw,
            source="source-v4 action commitment index",
        )
        if commitment_raw != cal._canonical_bytes(commitment_payload):
            raise ValueError("source-v4 commitment index must use canonical bytes")
        cal._exact_keys(
            commitment_payload,
            {
                "schema_version",
                "source_protocol_id",
                "sealed_evaluator_bundle_sha256",
                "environment_version",
                "randomness_contract",
                "commitment_hash_algorithm",
                "commitment_preimage_fields",
                "action_order",
                "decision_count",
                "commitment_count",
                "decision_branch_commitments",
            },
            "source-v4 commitment index",
        )
        if (
            commitment_payload["schema_version"]
            != source_pin["commitment_index_schema_version"]
            or commitment_payload["source_protocol_id"]
            != source_pin["source_protocol_id"]
            or commitment_payload["sealed_evaluator_bundle_sha256"]
            != source_pin["sealed_evaluator_bundle_sha256"]
            or commitment_payload["commitment_count"]
            != source_pin["commitment_count"]
            or commitment_payload["decision_count"] != 5376
        ):
            raise ValueError("source-v4 commitment index identity drifted")
        branch_ids: dict[tuple[str, str], str] = {}
        for decision_index, raw_decision in enumerate(
            cal._list(
                commitment_payload["decision_branch_commitments"],
                "decision_branch_commitments",
            )
        ):
            decision = cal._mapping(
                raw_decision,
                f"decision_branch_commitments[{decision_index}]",
            )
            cal._exact_keys(
                decision,
                {"decision_id", "branches"},
                f"decision_branch_commitments[{decision_index}]",
            )
            decision_id = cal._text(decision["decision_id"], "decision_id")
            branches = cal._list(decision["branches"], "branches")
            if len(branches) != 3:
                raise ValueError("source-v4 commitment decision must contain three branches")
            for branch_index, raw_branch in enumerate(branches):
                branch = cal._mapping(raw_branch, f"branches[{branch_index}]")
                cal._exact_keys(
                    branch,
                    {"selected_action_id", "commitment_id"},
                    f"branches[{branch_index}]",
                )
                key = (
                    decision_id,
                    cal._text(branch["selected_action_id"], "selected_action_id"),
                )
                if key in branch_ids:
                    raise ValueError("source-v4 commitment branch identity reused")
                branch_ids[key] = cal._digest(
                    branch["commitment_id"],
                    "commitment_id",
                )
        if len(branch_ids) != source_pin["commitment_count"]:
            raise ValueError("source-v4 commitment branch inventory drifted")

        public_roots = dependencies.public_view.roots
        if tuple(
            (item.root_index, item.subject_id, item.public_trajectory_sha256)
            for item in evaluator.root_manifests
        ) != tuple(
            (index, root.subject_id, root.public_trajectory_sha256)
            for index, root in enumerate(public_roots)
        ):
            raise ValueError("source-v4 public/evaluator root join drifted")
        self._evaluator = evaluator
        self._branch_ids = branch_ids
        self._decisions = {
            (item.subject_id, item.decision_id): item
            for item in evaluator.decision_sessions
        }

    def settle(
        self,
        *,
        public_root: HorizonPublicRoot,
        public_decision: HorizonPublicDecisionSession,
        delivered_action_id: str,
    ) -> _SafeSelectedBranchOutcome:
        decision = self._decisions.get(
            (public_root.subject_id, public_decision.decision_id)
        )
        if decision is None:
            raise ValueError("public decision has no exact sealed evaluator row")
        if (
            decision.subject_id != public_root.subject_id
            or decision.session_id != public_decision.session_id
            or decision.decision_id != public_decision.decision_id
            or decision.decision_index != public_decision.decision_index
            or decision.virtual_day != public_decision.virtual_day
        ):
            raise ValueError("source-v4 public/sealed decision join drifted")
        action = RelationshipAction(delivered_action_id)
        commitment = build_relationship_product_horizon_source_action_commitment(
            self._evaluator,
            subject_id=public_root.subject_id,
            decision_id=public_decision.decision_id,
            action=action,
        )
        commitment_id = cal._digest(commitment["commitment_id"], "commitment_id")
        expected_commitment_id = self._branch_ids.get(
            (public_decision.decision_id, delivered_action_id)
        )
        if commitment_id != expected_commitment_id:
            raise ValueError("selected source-v4 action branch commitment drifted")
        preimage = cal._mapping(commitment["preimage"], "commitment.preimage")
        if (
            preimage["subject_id"] != public_root.subject_id
            or preimage["decision_id"] != public_decision.decision_id
            or preimage["scene_id"] != decision.scene_id
            or preimage["environment_seed"] != decision.environment_seed
            or preimage["selected_action_id"] != delivered_action_id
            or preimage["sealed_evaluator_bundle_sha256"]
            != self._evaluator.sealed_bundle_sha256
        ):
            raise ValueError("selected source-v4 action branch preimage drifted")
        return _SafeSelectedBranchOutcome(
            environment_subject_id=cal._text(preimage["subject_id"], "subject_id"),
            selected_action_id=cal._text(
                preimage["selected_action_id"],
                "selected_action_id",
            ),
            typed_outcome_id=cal._text(
                preimage["typed_outcome_id"],
                "typed_outcome_id",
            ),
            rendered_user_reaction=cal._text(
                preimage["rendered_user_reaction"],
                "rendered_user_reaction",
            ),
            environment_evidence_ref=cal._text(
                preimage["environment_evidence_ref"],
                "environment_evidence_ref",
            ),
            environment_version=cal._text(
                preimage["environment_version"],
                "environment_version",
            ),
            commitment_id=commitment_id,
        )


def _stable_first_preaction_projection(
    *,
    root_sequence_index: int,
    root: HorizonPublicRoot,
    decision: HorizonPublicDecisionSession,
    owner_input_sha256: str,
    preaction: RelationshipProductFrozenPreActionSnapshot,
) -> Mapping[str, object]:
    receipt = preaction.execution_receipt
    temporal = receipt.temporal_delivery
    gate_decision = preaction.frozen_decision.decision
    projection = {
        "schema_version": scan._REACHABLE_FIRST_PROJECTION_SCHEMA_VERSION,
        "root_sequence_index": root_sequence_index,
        "subject_id": root.subject_id,
        "decision_index": decision.decision_index,
        "session_id": decision.session_id,
        "decision_id": decision.decision_id,
        "owner_input_persistence_sha256": owner_input_sha256,
        "owner_output_persistence_sha256": (
            social_record_store_persistence_sha256(
                preaction.owner_persistence_snapshot
            )
        ),
        "forecast_sha256": sha256_json(
            preference_action_forecast_to_payload(preaction.forecast)
        ),
        "frozen_decision_sha256": sha256_json(
            preaction.frozen_decision.to_payload()
        ),
        "gate_action": gate_decision.gate_action.value,
        "steer_probability_hex": gate_decision.steer_probability.hex(),
        "frozen_selected_action_id": gate_decision.selected_action_id,
        "executor_disposition": (
            RelationshipProductExecutorDisposition.APPLY_CANDIDATE.value
        ),
        "executor_status": receipt.executor_status.value,
        "candidate_advisory_id": receipt.candidate_advisory.advisory_id,
        "delivered_advisory_id": receipt.delivered_advisory.advisory_id,
        "temporal_delivered_action_id": preaction.delivered_action_id,
        "temporal_controller_params_hash": temporal.controller_params_hash,
        "temporal_action_family_version": temporal.action_family_version,
        "temporal_action_advisory_status": temporal.action_advisory_status.value,
        "cold_checkpoint_content_sha256": (
            preaction.frozen_policy.checkpoint.content_sha256
        ),
        "cold_frozen_policy_id": preaction.frozen_policy.policy_id,
    }
    if tuple(projection) != scan._REACHABLE_FIRST_PROJECTION_FIELDS:
        raise RuntimeError("dynamic first-preaction projection schema drifted")
    return projection


def _credit_timestamp(root_sequence_index: int, decision_index: int) -> int:
    return root_sequence_index * 20 + 5 + 2 * decision_index


def _terminal_failure_reasons(
    *,
    completed_root_count: int,
    onboarding_count: int,
    preaction_count: int,
    postaction_count: int,
    first_preaction_exact_match_count: int,
    later_owner_handoff_count: int,
    owner_writeback_change_count: int,
    selected_branch_resolution_count: int,
    selected_branch_commitment_match_count: int,
    unique_selected_branch_commitment_count: int,
    prediction_error_count: int,
    credit_count: int,
    unique_credit_count: int,
    unique_settlement_count: int,
    unique_environment_evidence_ref_count: int,
    cold_checkpoint_unchanged_count: int,
    temporal_delivered_nonnoop_count: int,
) -> tuple[str, ...]:
    checks = (
        (completed_root_count != 112, "completed_root_count_not_112"),
        (onboarding_count != 448, "onboarding_count_not_448"),
        (preaction_count != 896, "preaction_count_not_896"),
        (postaction_count != 896, "postaction_count_not_896"),
        (
            first_preaction_exact_match_count != 112,
            "first_preaction_exact_match_count_not_112",
        ),
        (later_owner_handoff_count != 784, "later_owner_handoff_count_not_784"),
        (
            owner_writeback_change_count != 896,
            "owner_writeback_change_count_not_896",
        ),
        (
            selected_branch_resolution_count != 896,
            "selected_branch_resolution_count_not_896",
        ),
        (
            selected_branch_commitment_match_count != 896,
            "selected_branch_commitment_match_count_not_896",
        ),
        (
            unique_selected_branch_commitment_count != 896,
            "unique_selected_branch_commitment_count_not_896",
        ),
        (prediction_error_count != 896, "prediction_error_count_not_896"),
        (credit_count != 896, "credit_count_not_896"),
        (unique_credit_count != 896, "unique_credit_count_not_896"),
        (unique_settlement_count != 896, "unique_settlement_count_not_896"),
        (
            unique_environment_evidence_ref_count != 896,
            "unique_environment_evidence_ref_count_not_896",
        ),
        (
            cold_checkpoint_unchanged_count != 896,
            "cold_checkpoint_unchanged_count_not_896",
        ),
        (
            temporal_delivered_nonnoop_count < 1,
            "temporal_delivered_nonnoop_count_below_one",
        ),
    )
    return tuple(reason for failed, reason in checks if failed)


async def _run_dynamic_collection_prefix(
    *,
    dependencies: _Dependencies,
    sink: cal._TraceSink,
) -> _DynamicReplay:
    frozen_policy = dependencies.scanner_dependencies.frozen_policy
    checkpoint = frozen_policy.checkpoint
    if checkpoint.update_count != 0 or checkpoint.processed_credit_ids or checkpoint.pending_decisions:
        raise RuntimeError("dynamic collection-prefix requires cold 0/0/0 theta0")
    authorization = scan._authorization(
        protocol_id=dependencies.protocol.protocol_id,
        frozen_policy=frozen_policy,
    )
    sink.append(
        {
            "schema_version": DYNAMIC_COLLECTION_PREFIX_TRACE_SCHEMA_VERSION,
            "record_type": "header",
            "protocol_id": dependencies.protocol.protocol_id,
            "upstream_scanner_artifact_id": dependencies.scanner_manifest[
                "artifact_id"
            ],
            "source_v4_public_plan_sha256": dependencies.public_view.public_plan_sha256,
            "development_reader_artifact_id": (
                dependencies.scanner_dependencies.reader_artifact.artifact_id
            ),
            "theta0_v2_artifact_id": dependencies.scanner_dependencies.theta0.artifact_id,
            "cold_checkpoint_content_sha256": checkpoint.content_sha256,
            "cold_frozen_policy_id": frozen_policy.policy_id,
            "root_count": 112,
            "decision_indices": list(range(8)),
            "credit_timestamp_formula": _CREDIT_TIMESTAMP_FORMULA,
            "environment_scope_created": False,
            "source_v4_sealed_file_read_count": 0,
            "gate_update_count": 0,
            "model_output_count": 0,
            "cuda_execution_count": 0,
        }
    )
    environment_scope: _SelectedBranchEnvironmentScope | None = None
    first_projections: list[Mapping[str, object]] = []
    action_counts: Counter[str] = Counter()
    nonnoop_roots: set[int] = set()
    credit_ids: set[str] = set()
    settlement_ids: set[str] = set()
    environment_evidence_refs: set[str] = set()
    commitment_ids: set[str] = set()
    onboarding_count = 0
    preaction_count = 0
    postaction_count = 0
    first_exact_count = 0
    handoff_count = 0
    writeback_change_count = 0
    branch_resolution_count = 0
    commitment_match_count = 0
    pe_count = 0
    credit_count = 0
    checkpoint_unchanged_count = 0
    completed_root_count = 0
    previous_credit_timestamp = -1

    for root_index, root in enumerate(dependencies.public_view.roots):
        owner_persistence = await scan._post_onboarding_state(root)
        onboarding_count += len(root.onboarding_sessions)
        post_onboarding_sha = social_record_store_persistence_sha256(
            owner_persistence
        )
        sink.append(
            {
                "schema_version": DYNAMIC_COLLECTION_PREFIX_TRACE_SCHEMA_VERSION,
                "record_type": "root_start",
                "root_sequence_index": root_index,
                "subject_id": root.subject_id,
                "owner_reset": True,
                "onboarding_appended_once": True,
                "onboarding_session_count": 4,
                "post_onboarding_persistence_sha256": post_onboarding_sha,
            }
        )
        prior_postaction_owner_sha: str | None = None
        for decision in root.decision_sessions[:8]:
            sequence_index = root_index * 8 + decision.decision_index
            owner_input_sha = social_record_store_persistence_sha256(
                owner_persistence
            )
            if decision.decision_index == 0:
                if owner_input_sha != post_onboarding_sha:
                    raise RuntimeError("first preaction did not consume onboarding state")
            else:
                if owner_input_sha != prior_postaction_owner_sha:
                    raise RuntimeError("later preaction did not consume prior postaction state")
                handoff_count += 1
            preaction = await scan.prepare_relationship_product_frozen_preaction(
                request=scan._request(subject_id=root.subject_id, decision=decision),
                owner_persistence_snapshot=owner_persistence,
                forecast_runtime=dependencies.scanner_dependencies.forecast_runtime,
                frozen_policy=frozen_policy,
                executor_disposition=(
                    RelationshipProductExecutorDisposition.APPLY_CANDIDATE
                ),
                authorization=authorization,
                substrate_snapshot=cal._placeholder_substrate(),
            )
            if preaction.frozen_policy != frozen_policy:
                raise RuntimeError("dynamic collection-prefix frozen policy drifted")
            if preaction.execution_receipt.command.owner_prestate_sha256 != social_record_store_persistence_sha256(
                preaction.owner_persistence_snapshot
            ):
                raise RuntimeError("dynamic executor owner prestate lineage drifted")
            projection = _stable_first_preaction_projection(
                root_sequence_index=root_index,
                root=root,
                decision=decision,
                owner_input_sha256=owner_input_sha,
                preaction=preaction,
            )
            if decision.decision_index == 0:
                expected = dependencies.expected_first_projections[root_index]
                if projection != expected:
                    raise RuntimeError(
                        "dynamic first preaction differs from frozen scanner seam"
                    )
                first_projections.append(projection)
                first_exact_count += 1
            delivered_action = preaction.delivered_action_id
            if delivered_action != preaction.frozen_decision.decision.selected_action_id:
                raise RuntimeError("natural APPLY did not deliver frozen selected action")
            action_counts[delivered_action] += 1
            if delivered_action != RelationshipAction.NEUTRAL_NOOP.value:
                nonnoop_roots.add(root_index)
            preaction_count += 1
            sink.append(
                {
                    "schema_version": DYNAMIC_COLLECTION_PREFIX_TRACE_SCHEMA_VERSION,
                    "record_type": "preaction",
                    "global_sequence_index": sequence_index,
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "decision_index": decision.decision_index,
                    "session_id": decision.session_id,
                    "decision_id": decision.decision_id,
                    "owner_input_persistence_sha256": owner_input_sha,
                    "consumed_exact_prior_postaction_owner": (
                        decision.decision_index > 0
                    ),
                    "stable_projection": projection,
                    "preaction_append_fsynced_before_selected_branch_open": True,
                    "environment_scope_already_created_before_current_preaction": (
                        environment_scope is not None
                    ),
                    "sealed_truth_passed_to_preaction": False,
                    "selected_branch_opened": False,
                }
            )
            if environment_scope is None:
                environment_scope = _SelectedBranchEnvironmentScope(
                    dependencies=dependencies
                )
            branch = environment_scope.settle(
                public_root=root,
                public_decision=decision,
                delivered_action_id=delivered_action,
            )
            branch_resolution_count += 1
            commitment_match_count += 1
            commitment_ids.add(branch.commitment_id)
            if branch.selected_action_id != delivered_action:
                raise RuntimeError("environment selected a different actual action")
            credit_timestamp = _credit_timestamp(root_index, decision.decision_index)
            if credit_timestamp <= previous_credit_timestamp:
                raise RuntimeError("dynamic credit timestamps are not strictly increasing")
            previous_credit_timestamp = credit_timestamp
            action_turn = 4 + 2 * decision.decision_index
            settlement_input = replace(
                cal._settlement_input(
                    subject_scope=root.subject_id,
                    decision=decision,
                    forecast_id=preaction.forecast.forecast_id,
                    selected_action_id=delivered_action,
                    environment_outcome=branch,
                    action_turn=action_turn,
                    credit_timestamp=credit_timestamp,
                ),
                apply_credit_to_gate=False,
            )
            settled = await settle_relationship_product_frozen_pulse(
                preaction=preaction,
                settlement_input=settlement_input,
            )
            if (
                settled.credit_applied_to_gate
                or settled.evaluation_gate_update_delta != 0
                or settled.gate_checkpoint != checkpoint
                or settled.credit.prediction_id != preaction.forecast.forecast_id
                or settled.credit.abstract_action_id != delivered_action
                or settled.credit.timestamp_ms != credit_timestamp
            ):
                raise RuntimeError("dynamic frozen PE-credit/gate lineage drifted")
            checkpoint_unchanged_count += 1
            pe_count += 1
            credit_count += 1
            if settled.credit.record_id in credit_ids:
                raise RuntimeError("dynamic collection-prefix credit ID was reused")
            if settled.settlement.settlement_id in settlement_ids:
                raise RuntimeError("dynamic collection-prefix settlement ID was reused")
            if branch.environment_evidence_ref in environment_evidence_refs:
                raise RuntimeError("dynamic environment evidence ref was reused")
            credit_ids.add(settled.credit.record_id)
            settlement_ids.add(settled.settlement.settlement_id)
            environment_evidence_refs.add(branch.environment_evidence_ref)
            owner_preaction_sha = social_record_store_persistence_sha256(
                preaction.owner_persistence_snapshot
            )
            owner_postaction_sha = social_record_store_persistence_sha256(
                settled.owner_persistence_snapshot
            )
            if owner_postaction_sha != owner_preaction_sha:
                writeback_change_count += 1
            owner_persistence = settled.owner_persistence_snapshot
            prior_postaction_owner_sha = owner_postaction_sha
            postaction_count += 1
            sink.append(
                {
                    "schema_version": DYNAMIC_COLLECTION_PREFIX_TRACE_SCHEMA_VERSION,
                    "record_type": "postaction",
                    "global_sequence_index": sequence_index,
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "decision_index": decision.decision_index,
                    "session_id": decision.session_id,
                    "decision_id": decision.decision_id,
                    "preaction_forecast_id": preaction.forecast.forecast_id,
                    "temporal_delivered_action_id": delivered_action,
                    "selected_branch_commitment_id": branch.commitment_id,
                    "environment_subject_id": branch.environment_subject_id,
                    "environment_selected_action_id": branch.selected_action_id,
                    "typed_outcome_id": branch.typed_outcome_id,
                    "rendered_user_reaction_sha256": cal._sha256_text(
                        branch.rendered_user_reaction
                    ),
                    "environment_evidence_ref": branch.environment_evidence_ref,
                    "environment_version": branch.environment_version,
                    "selected_branch_opened_after_current_preaction_fsync": True,
                    "environment_scope_created_after_global_first_preaction_fsync": (
                        sequence_index == 0
                    ),
                    "settlement_id": settled.settlement.settlement_id,
                    "social_prediction_error": cal._social_pe_payload(
                        settled.social_prediction_error_snapshot.value
                    ),
                    "credit": cal._credit_payload(settled.credit),
                    "credit_applied_to_gate": False,
                    "gate_update_delta": 0,
                    "cold_checkpoint_content_sha256": checkpoint.content_sha256,
                    "cold_update_count": checkpoint.update_count,
                    "cold_processed_credit_id_count": len(
                        checkpoint.processed_credit_ids
                    ),
                    "cold_pending_decision_count": len(
                        checkpoint.pending_decisions
                    ),
                    "owner_preaction_persistence_sha256": owner_preaction_sha,
                    "owner_postaction_persistence_sha256": owner_postaction_sha,
                    "owner_writeback_changed_persistence": (
                        owner_postaction_sha != owner_preaction_sha
                    ),
                    "postaction_append_fsynced_before_next_preaction": True,
                    "evaluation_or_judge_feedback_received": False,
                }
            )
        completed_root_count += 1

    if environment_scope is None:
        raise RuntimeError("dynamic collection-prefix opened no environment scope")
    first_projection_sha = sha256_json(first_projections)
    if first_projection_sha != dependencies.protocol.payload["upstream_scanner"][
        "first_preaction_projection_sha256"
    ]:
        raise RuntimeError("dynamic first-preaction aggregate digest drifted")
    nonnoop_count = sum(
        count
        for action_id, count in action_counts.items()
        if action_id != RelationshipAction.NEUTRAL_NOOP.value
    )
    failure_reasons = _terminal_failure_reasons(
        completed_root_count=completed_root_count,
        onboarding_count=onboarding_count,
        preaction_count=preaction_count,
        postaction_count=postaction_count,
        first_preaction_exact_match_count=first_exact_count,
        later_owner_handoff_count=handoff_count,
        owner_writeback_change_count=writeback_change_count,
        selected_branch_resolution_count=branch_resolution_count,
        selected_branch_commitment_match_count=commitment_match_count,
        unique_selected_branch_commitment_count=len(commitment_ids),
        prediction_error_count=pe_count,
        credit_count=credit_count,
        unique_credit_count=len(credit_ids),
        unique_settlement_count=len(settlement_ids),
        unique_environment_evidence_ref_count=len(environment_evidence_refs),
        cold_checkpoint_unchanged_count=checkpoint_unchanged_count,
        temporal_delivered_nonnoop_count=nonnoop_count,
    )
    status = _SUCCESS_STATUS if not failure_reasons else _FAIL_STATUS
    sink.append(
        {
            "schema_version": DYNAMIC_COLLECTION_PREFIX_TRACE_SCHEMA_VERSION,
            "record_type": "terminal",
            "completed_root_count": completed_root_count,
            "onboarding_count": onboarding_count,
            "preaction_count": preaction_count,
            "postaction_count": postaction_count,
            "first_preaction_exact_match_count": first_exact_count,
            "first_preaction_projection_sha256": first_projection_sha,
            "later_owner_handoff_count": handoff_count,
            "owner_writeback_change_count": writeback_change_count,
            "selected_branch_resolution_count": branch_resolution_count,
            "selected_branch_commitment_match_count": commitment_match_count,
            "unique_selected_branch_commitment_count": len(commitment_ids),
            "prediction_error_count": pe_count,
            "credit_count": credit_count,
            "unique_credit_count": len(credit_ids),
            "unique_settlement_count": len(settlement_ids),
            "unique_environment_evidence_ref_count": len(
                environment_evidence_refs
            ),
            "cold_checkpoint_unchanged_count": checkpoint_unchanged_count,
            "temporal_delivered_action_counts": dict(sorted(action_counts.items())),
            "temporal_delivered_nonnoop_count": nonnoop_count,
            "temporal_delivered_nonnoop_root_count": len(nonnoop_roots),
            "credit_applied_to_gate_count": 0,
            "gate_update_count": 0,
            "forced_executor_count": 0,
            "forced_schedule_count": 0,
            "batch_apply_count": 0,
            "branch_resolution_before_current_preaction_fsync_count": 0,
            "excluded_root_count": 0,
            "duplicate_root_count": 0,
            "evaluation_decision_count": 0,
            "model_output_count": 0,
            "cuda_execution_count": 0,
            "terminal_failure_reasons": list(failure_reasons),
            "terminal_status": status,
            "forced_common_batch_protocol_freeze_authorized": not failure_reasons,
            "forced_common_batch_execution_authorized": False,
            "evaluation_protocol_freeze_authorized": False,
            "campaign_execution_authorized": False,
        }
    )
    return _DynamicReplay(
        completed_root_count=completed_root_count,
        onboarding_count=onboarding_count,
        preaction_count=preaction_count,
        postaction_count=postaction_count,
        first_preaction_exact_match_count=first_exact_count,
        later_owner_handoff_count=handoff_count,
        owner_writeback_change_count=writeback_change_count,
        selected_branch_resolution_count=branch_resolution_count,
        selected_branch_commitment_match_count=commitment_match_count,
        unique_selected_branch_commitment_count=len(commitment_ids),
        prediction_error_count=pe_count,
        credit_count=credit_count,
        unique_credit_count=len(credit_ids),
        unique_settlement_count=len(settlement_ids),
        unique_environment_evidence_ref_count=len(environment_evidence_refs),
        cold_checkpoint_unchanged_count=checkpoint_unchanged_count,
        temporal_delivered_action_counts=dict(sorted(action_counts.items())),
        temporal_delivered_nonnoop_count=nonnoop_count,
        temporal_delivered_nonnoop_root_count=len(nonnoop_roots),
        first_preaction_projection_sha256=first_projection_sha,
        terminal_failure_reasons=failure_reasons,
        terminal_status=status,
    )


def materialize_relationship_product_horizon_dynamic_collection_prefix(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    commit = cal._git_commit(implementation_git_commit)
    root = pathlib.Path(output_dir)
    if root.exists():
        raise FileExistsError(f"dynamic collection-prefix root is create-only: {root}")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
        scanner_root=pathlib.Path(scanner_root),
    )
    root.mkdir(parents=True, exist_ok=False)
    cal._write_create_only(root / "protocol.json", dependencies.protocol.raw_bytes)
    sink = cal._FsyncTraceSink(root / _TRACE_FILENAME)
    try:
        replay = asyncio.run(
            _run_dynamic_collection_prefix(dependencies=dependencies, sink=sink)
        )
    finally:
        sink.close()
    manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        replay=replay,
        implementation_git_commit=commit,
    )
    cal._write_create_only(root / "manifest.json", cal._canonical_bytes(manifest))
    return manifest


def validate_relationship_product_horizon_dynamic_collection_prefix(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    external_protocol = cal._digest(expected_protocol_id, "expected_protocol_id")
    external_artifact = cal._digest(expected_artifact_id, "expected_artifact_id")
    root = pathlib.Path(output_dir)
    manifest_raw = cal._read_regular(root / "manifest.json")
    manifest = cal._parse_json_bytes(
        manifest_raw,
        source="dynamic collection-prefix manifest",
    )
    if manifest_raw != cal._canonical_bytes(manifest):
        raise ValueError("dynamic collection-prefix manifest must use canonical bytes")
    if manifest["protocol_id"] != external_protocol:
        raise ValueError("external dynamic collection-prefix protocol ID drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("external dynamic collection-prefix artifact ID drifted")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
        scanner_root=pathlib.Path(scanner_root),
    )
    if dependencies.protocol.protocol_id != external_protocol:
        raise ValueError("packaged dynamic collection-prefix protocol ID drifted")
    if cal._read_regular(root / "protocol.json") != dependencies.protocol.raw_bytes:
        raise ValueError("persisted dynamic collection-prefix protocol bytes drifted")
    sink = cal._MemoryTraceSink()
    replay = asyncio.run(
        _run_dynamic_collection_prefix(dependencies=dependencies, sink=sink)
    )
    if cal._read_regular(root / _TRACE_FILENAME) != sink.raw_bytes:
        raise ValueError("dynamic collection-prefix stable trace bytes drifted")
    if cal._regular_file_inventory(root) != _OUTPUT_FILES:
        raise ValueError("dynamic collection-prefix output inventory drifted")
    expected_manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        replay=replay,
        implementation_git_commit=cal._git_commit(
            manifest["implementation_git_commit"]
        ),
    )
    if manifest != expected_manifest or manifest_raw != cal._canonical_bytes(
        expected_manifest
    ):
        raise ValueError("dynamic collection-prefix manifest content drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("dynamic collection-prefix artifact identity drifted")
    return manifest


def _build_manifest(
    *,
    root: pathlib.Path,
    dependencies: _Dependencies,
    replay: _DynamicReplay,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    files = []
    for relative in ("protocol.json", _TRACE_FILENAME):
        raw = cal._read_regular(root / relative)
        files.append(
            {
                "path": relative,
                "raw_bytes": len(raw),
                "raw_sha256": cal._sha256_bytes(raw),
            }
        )
    success = replay.terminal_status == _SUCCESS_STATUS
    claims = {
        "dynamic_collection_gate_completed": True,
        "first_preaction_exact_match_verified": success,
        "sequential_owner_writeback_verified": success,
        "selected_branch_commitment_verified": success,
        "pe_credit_chain_verified": success,
        "forced_common_batch_protocol_freeze_authorized": success,
        "forced_common_batch_execution_authorized": False,
        "evaluation_protocol_freeze_authorized": False,
        "campaign_execution_authorized": False,
        "reader_qualified": False,
        "formal_evidence_authorized": False,
        "unseen_evidence_authorized": False,
        "integrated_horizon_authorized": False,
        "appendable_effect": False,
        "readable_effect": False,
        "learnable_effect": False,
        "steerable_effect": False,
        "four_able_complete": False,
        "human_validation_complete": False,
        "production_active": False,
    }
    source_pin = cal._mapping(
        dependencies.protocol.payload["source_v4_environment"],
        "source_v4_environment",
    )
    scanner_pin = cal._mapping(
        dependencies.protocol.payload["upstream_scanner"],
        "upstream_scanner",
    )
    core = {
        "schema_version": DYNAMIC_COLLECTION_PREFIX_MANIFEST_SCHEMA_VERSION,
        "protocol_id": dependencies.protocol.protocol_id,
        "protocol_raw_sha256": dependencies.protocol.raw_sha256,
        "implementation_git_commit": implementation_git_commit,
        "source_v4_admission_artifact_id": source_pin["admission_artifact_id"],
        "source_v4_public_plan_sha256": dependencies.public_view.public_plan_sha256,
        "source_v4_sealed_evaluator_bundle_sha256": source_pin[
            "sealed_evaluator_bundle_sha256"
        ],
        "source_v4_commitment_index_raw_sha256": source_pin[
            "commitment_index_raw_sha256"
        ],
        "upstream_scanner_protocol_id": scanner_pin["protocol_id"],
        "upstream_scanner_artifact_id": scanner_pin["artifact_id"],
        "development_reader_artifact_id": (
            dependencies.scanner_dependencies.reader_artifact.artifact_id
        ),
        "embedding_table_artifact_id": (
            dependencies.scanner_dependencies.embedding_table.artifact_id
        ),
        "theta0_v2_artifact_id": dependencies.scanner_dependencies.theta0.artifact_id,
        "cold_checkpoint_content_sha256": (
            dependencies.scanner_dependencies.frozen_policy.checkpoint.content_sha256
        ),
        "cold_frozen_policy_id": (
            dependencies.scanner_dependencies.frozen_policy.policy_id
        ),
        "completed_root_count": replay.completed_root_count,
        "onboarding_count": replay.onboarding_count,
        "preaction_count": replay.preaction_count,
        "postaction_count": replay.postaction_count,
        "first_preaction_exact_match_count": (
            replay.first_preaction_exact_match_count
        ),
        "first_preaction_projection_sha256": (
            replay.first_preaction_projection_sha256
        ),
        "first_preaction_projection_schema_version": (
            scan._REACHABLE_FIRST_PROJECTION_SCHEMA_VERSION
        ),
        "first_preaction_projection_fields": list(
            scan._REACHABLE_FIRST_PROJECTION_FIELDS
        ),
        "later_owner_handoff_count": replay.later_owner_handoff_count,
        "owner_writeback_change_count": replay.owner_writeback_change_count,
        "selected_branch_resolution_count": replay.selected_branch_resolution_count,
        "selected_branch_commitment_match_count": (
            replay.selected_branch_commitment_match_count
        ),
        "unique_selected_branch_commitment_count": (
            replay.unique_selected_branch_commitment_count
        ),
        "prediction_error_count": replay.prediction_error_count,
        "credit_count": replay.credit_count,
        "unique_credit_count": replay.unique_credit_count,
        "unique_settlement_count": replay.unique_settlement_count,
        "unique_environment_evidence_ref_count": (
            replay.unique_environment_evidence_ref_count
        ),
        "credit_applied_to_gate_count": 0,
        "gate_update_count": 0,
        "cold_checkpoint_unchanged_count": replay.cold_checkpoint_unchanged_count,
        "temporal_delivered_action_counts": replay.temporal_delivered_action_counts,
        "temporal_delivered_nonnoop_count": replay.temporal_delivered_nonnoop_count,
        "temporal_delivered_nonnoop_root_count": (
            replay.temporal_delivered_nonnoop_root_count
        ),
        "forced_executor_count": 0,
        "forced_schedule_count": 0,
        "batch_apply_count": 0,
        "branch_resolution_before_current_preaction_fsync_count": 0,
        "excluded_root_count": 0,
        "duplicate_root_count": 0,
        "evaluation_decision_count": 0,
        "evaluation_or_judge_feedback_count": 0,
        "source_v4_public_plan_file_read_count": 1,
        "source_v4_sealed_file_read_count": 2,
        "source_v4_source_protocol_file_read_count": 2,
        "upstream_scanner_trace_file_read_count": 1,
        "challenge_label_file_read_count": 0,
        "group_split_file_read_count": 0,
        "model_output_count": 0,
        "cuda_execution_count": 0,
        "human_sample_count": 0,
        "terminal_failure_reasons": list(replay.terminal_failure_reasons),
        "files": files,
        "status": replay.terminal_status,
        "claims": claims,
        "claim_boundary": dependencies.protocol.payload["claim_boundary"],
    }
    return {"artifact_id": sha256_json(core), **core}


__all__ = [
    "DYNAMIC_COLLECTION_PREFIX_MANIFEST_SCHEMA_VERSION",
    "DYNAMIC_COLLECTION_PREFIX_PROTOCOL_SCHEMA_VERSION",
    "DYNAMIC_COLLECTION_PREFIX_TRACE_SCHEMA_VERSION",
    "RelationshipProductHorizonDynamicCollectionPrefixProtocol",
    "load_relationship_product_horizon_dynamic_collection_prefix_protocol",
    "materialize_relationship_product_horizon_dynamic_collection_prefix",
    "relationship_product_horizon_dynamic_collection_prefix_protocol_path",
    "validate_relationship_product_horizon_dynamic_collection_prefix",
]
