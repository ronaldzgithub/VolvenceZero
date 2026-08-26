"""Transductive public-only opportunity scan for Product Horizon theta0 v2.

The source-v4 public plan, development reader, and theta0 v2 have all already
been observed.  This workflow therefore makes no unseen or effect claim.  It
restores every root only to its public post-onboarding state, invokes the actual
typed executor on 5,376 reset-state probes, and publishes two canonical
candidate-vs-strict-noop witnesses without opening source-v4 sealed truth or
settling an environment.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import pathlib
from typing import Mapping

from lifeform_domain_emogpt.lab.contracts import sha256_json
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductExecutorDisposition,
    RelationshipProductFrozenPreActionSnapshot,
    RelationshipProductFrozenPulseAuthorization,
    RelationshipProductOnboardingInput,
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    append_relationship_product_onboarding,
    prepare_relationship_product_frozen_preaction,
)
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    HorizonPublicDecisionSession,
    HorizonPublicRoot,
    RelationshipProductHorizonPublicView,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateFrozenPolicy,
    RelationshipActionGateTheta0Artifact,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    FrozenLinearRelationshipConditionReaderArtifact,
    FrozenLinearRelationshipConditionReaderRuntime,
    FrozenLinearRelationshipPreferenceForecastRuntime,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (
    PrecomputedPublicEmbeddingTable,
    PrecomputedPublicSemanticEmbedder,
    load_precomputed_public_embedding_table,
)
from lifeform_evolution import relationship_product_horizon_theta0_calibration as cal
from volvence_zero.social import (
    PreferenceActionForecastRequest,
    social_record_store_persistence_sha256,
)
from volvence_zero.social_cognition import preference_action_forecast_to_payload


TRANSDUCTIVE_PUBLIC_OPPORTUNITY_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-transductive-public-opportunity-protocol.v1"
)
TRANSDUCTIVE_PUBLIC_OPPORTUNITY_TRACE_SCHEMA_VERSION = (
    "relationship-product-horizon-transductive-public-opportunity-trace.v1"
)
TRANSDUCTIVE_PUBLIC_OPPORTUNITY_WITNESS_SCHEMA_VERSION = (
    "relationship-product-horizon-transductive-public-opportunity-witness.v1"
)
TRANSDUCTIVE_PUBLIC_OPPORTUNITY_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-transductive-public-opportunity-manifest.v1"
)

_PROTOCOL_FILENAME = (
    "relationship_product_horizon_transductive_public_opportunity_v1.json"
)
_TRACE_FILENAME = "public_opportunity_scan.jsonl"
_WITNESS_FILENAME = "paired_witnesses.json"
_OUTPUT_FILES = frozenset(
    {"protocol.json", _TRACE_FILENAME, _WITNESS_FILENAME, "manifest.json"}
)
_INTERLOCUTOR_ID = "primary"
_REACHABLE = "reachable_first_preaction"
_COLLECTION_STRESS = "collection_stress"
_EVALUATION_STRESS = "evaluation_stress"
_SUCCESS_STATUS = (
    "development_transductive_public_opportunity_present_"
    "collection_prefix_protocol_freeze_authorized_effect_not_tested"
)
_FAIL_STATUS = (
    "development_transductive_public_opportunity_absent_"
    "campaign_blocked_effect_not_tested"
)
_REACHABLE_FIRST_PROJECTION_SCHEMA_VERSION = (
    "relationship-product-horizon-first-preaction-projection.v1"
)
_REACHABLE_FIRST_PROJECTION_FIELDS = (
    "schema_version",
    "root_sequence_index",
    "subject_id",
    "decision_index",
    "session_id",
    "decision_id",
    "owner_input_persistence_sha256",
    "owner_output_persistence_sha256",
    "forecast_sha256",
    "frozen_decision_sha256",
    "gate_action",
    "steer_probability_hex",
    "frozen_selected_action_id",
    "executor_disposition",
    "executor_status",
    "candidate_advisory_id",
    "delivered_advisory_id",
    "temporal_delivered_action_id",
    "temporal_controller_params_hash",
    "temporal_action_family_version",
    "temporal_action_advisory_status",
    "cold_checkpoint_content_sha256",
    "cold_frozen_policy_id",
)


@dataclass(frozen=True)
class RelationshipProductHorizonTransductivePublicOpportunityProtocol:
    payload: Mapping[str, object]
    raw_bytes: bytes
    protocol_id: str
    raw_sha256: str


@dataclass(frozen=True)
class _Dependencies:
    protocol: RelationshipProductHorizonTransductivePublicOpportunityProtocol
    public_view: RelationshipProductHorizonPublicView
    embedding_table: PrecomputedPublicEmbeddingTable
    reader_artifact: FrozenLinearRelationshipConditionReaderArtifact
    forecast_runtime: FrozenLinearRelationshipPreferenceForecastRuntime
    theta0: RelationshipActionGateTheta0Artifact
    frozen_policy: RelationshipActionGateFrozenPolicy


@dataclass(frozen=True)
class _ScanReplay:
    category_probe_counts: Mapping[str, int]
    public_index_bucket_counts: Mapping[str, int]
    selected_action_counts: Mapping[str, int]
    delivered_action_counts: Mapping[str, int]
    selected_action_counts_by_category: Mapping[str, Mapping[str, int]]
    delivered_action_counts_by_category: Mapping[str, Mapping[str, int]]
    nonnoop_counts_by_category: Mapping[str, int]
    nonnoop_root_counts_by_category: Mapping[str, int]
    reachable_first_projection_sha256: str
    witness_artifact: Mapping[str, object]
    witness_pass_count: int
    strict_noop_witness_executor_count: int
    terminal_failure_reasons: tuple[str, ...]
    terminal_status: str


def relationship_product_horizon_transductive_public_opportunity_protocol_path(
) -> pathlib.Path:
    return pathlib.Path(__file__).with_name("protocols") / _PROTOCOL_FILENAME


def load_relationship_product_horizon_transductive_public_opportunity_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonTransductivePublicOpportunityProtocol:
    source = pathlib.Path(
        path
        or relationship_product_horizon_transductive_public_opportunity_protocol_path()
    )
    raw = source.read_bytes()
    payload = cal._parse_json_bytes(raw, source="transductive opportunity protocol")
    cal._exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "purpose",
            "adaptive_lineage",
            "source_v4_public",
            "development_reader",
            "theta0_v2",
            "scan",
            "paired_witness",
            "terminal_gates",
            "causal_firewall",
            "claims",
            "claim_boundary",
        },
        "transductive opportunity protocol",
    )
    if (
        payload["schema_version"]
        != TRANSDUCTIVE_PUBLIC_OPPORTUNITY_PROTOCOL_SCHEMA_VERSION
        or payload["evidence_tier"] != "development"
        or payload["owner"]
        != (
            "lifeform_evolution."
            "relationship_product_horizon_transductive_public_opportunity"
        )
        or payload["purpose"]
        != "transductive_public_only_gate_and_executor_opportunity_scan"
    ):
        raise ValueError("transductive opportunity protocol identity drifted")
    _validate_protocol(payload)
    return RelationshipProductHorizonTransductivePublicOpportunityProtocol(
        payload=payload,
        raw_bytes=raw,
        protocol_id=sha256_json(payload),
        raw_sha256=cal._sha256_bytes(raw),
    )


def _validate_protocol(payload: Mapping[str, object]) -> None:
    adaptive = cal._mapping(payload["adaptive_lineage"], "adaptive_lineage")
    source = cal._mapping(payload["source_v4_public"], "source_v4_public")
    reader = cal._mapping(payload["development_reader"], "development_reader")
    theta = cal._mapping(payload["theta0_v2"], "theta0_v2")
    scan = cal._mapping(payload["scan"], "scan")
    witness = cal._mapping(payload["paired_witness"], "paired_witness")
    terminal = cal._mapping(payload["terminal_gates"], "terminal_gates")
    firewall = cal._mapping(payload["causal_firewall"], "causal_firewall")
    claims = cal._mapping(payload["claims"], "claims")

    expected_keys = {
        "adaptive_lineage": {
            "source_v4_public_previously_materialized_for_development_reader",
            "source_v4_public_previously_scanned_with_theta0_v1",
            "theta0_v2_trained_on_already_used_source_v3",
            "transductive",
            "unseen",
            "confirmatory",
            "formal",
        },
        "source_v4_public": {
            "admission_protocol_id",
            "admission_artifact_id",
            "admission_manifest_raw_sha256",
            "source_protocol_id",
            "public_plan_relative_path",
            "public_plan_schema_version",
            "public_plan_sha256",
            "public_plan_raw_sha256",
            "root_count",
            "onboarding_session_count",
            "decision_count",
            "sealed_file_read_count",
        },
        "development_reader": {
            "protocol_id",
            "artifact_id",
            "manifest_raw_sha256",
            "embedding_table_artifact_id",
            "embedding_table_raw_sha256",
            "reader_artifact_id",
            "reader_artifact_raw_sha256",
            "condition_reader_qualified",
            "runtime_fit_allowed",
            "model_execution_count",
            "cuda_execution_count",
        },
        "theta0_v2": {
            "bootstrap_protocol_id",
            "bootstrap_artifact_id",
            "bootstrap_manifest_raw_sha256",
            "artifact_id",
            "artifact_raw_sha256",
            "source_checkpoint_content_sha256",
            "source_batch_artifact_id",
            "cold_checkpoint_content_sha256",
            "cold_frozen_policy_id",
            "cold_update_count",
            "cold_processed_credit_id_count",
            "cold_pending_decision_count",
        },
        "scan": {
            "root_order",
            "decision_order",
            "owner_reset_each_root",
            "owner_state",
            "decision_writeback_during_scan",
            "reset_state_probe",
            "natural_executor_disposition",
            "reachable_first_preaction_definition",
            "collection_stress_definition",
            "evaluation_stress_definition",
            "reachable_first_preaction_count",
            "collection_stress_count",
            "evaluation_stress_count",
            "reset_state_counterfactual_probe_count",
            "total_probe_count",
            "public_index_bucket_count",
            "public_index_bucket_width",
            "public_index_bucket_probe_count",
            "public_onboarding_outcome_read_count",
            "natural_apply_candidate_executor_count",
            "environment_settlement_count",
            "prediction_error_count",
            "credit_count",
            "gate_update_count",
        },
        "paired_witness": {
            "categories",
            "selection_rule",
            "apply_disposition",
            "control_disposition",
            "same_forecast_required",
            "same_owner_prestate_required",
            "same_frozen_decision_required",
            "same_candidate_advisory_required",
            "same_frozen_policy_required",
            "apply_delivers_frozen_selected_action",
            "control_delivers_neutral_noop",
            "actual_action_divergence_required",
            "cold_checkpoint_unchanged_required",
            "strict_noop_executor_count_on_pass",
        },
        "terminal_gates": {
            "reachable_first_temporal_delivered_nonnoop_min_count",
            "evaluation_stress_temporal_delivered_nonnoop_min_count",
            "evaluation_stress_temporal_delivered_nonnoop_root_min_count",
            "paired_witness_pass_count",
            "successful_terminal",
            "failed_terminal",
            "collection_prefix_decision_count_per_root_on_success",
            "collection_prefix_protocol_freeze_authorized_on_success",
            "collection_prefix_execution_authorized_on_success",
            "full_evaluation_authorized_on_success",
            "campaign_authorized_on_success",
            "future_collection_first_preaction_exact_match_required",
            "future_collection_first_preaction_projection_schema_version",
            "future_collection_first_preaction_projection_fields",
            "scientific_retry_with_modified_theta_reader_source_threshold_or_selection_forbidden",
        },
        "causal_firewall": {
            "source_v4_public_plan_file_read_count",
            "source_v4_sealed_file_read_count",
            "challenge_label_file_read_count",
            "group_split_file_read_count",
            "environment_settlement_count",
            "prediction_error_count",
            "credit_count",
            "gate_update_count",
            "model_output_count",
            "cuda_execution_count",
            "temporal_timestamp_excluded_from_stable_projection",
            "executor_receipt_byte_exact_claim",
        },
        "claims": {
            "scanner_execution_authorized",
            "source_v4_opportunity_established",
            "collection_prefix_protocol_freeze_authorized",
            "collection_prefix_execution_authorized",
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
    for name, mapping in (
        ("adaptive_lineage", adaptive),
        ("source_v4_public", source),
        ("development_reader", reader),
        ("theta0_v2", theta),
        ("scan", scan),
        ("paired_witness", witness),
        ("terminal_gates", terminal),
        ("causal_firewall", firewall),
        ("claims", claims),
    ):
        cal._exact_keys(mapping, expected_keys[name], name)

    for field in (
        "source_v4_public_previously_materialized_for_development_reader",
        "source_v4_public_previously_scanned_with_theta0_v1",
        "theta0_v2_trained_on_already_used_source_v3",
        "transductive",
    ):
        if not cal._boolean(adaptive[field], f"adaptive_lineage.{field}"):
            raise ValueError(f"adaptive_lineage.{field} must remain true")
    for field in ("unseen", "confirmatory", "formal"):
        if cal._boolean(adaptive[field], f"adaptive_lineage.{field}"):
            raise ValueError(f"adaptive_lineage.{field} must remain false")

    for mapping_name, mapping, digest_fields in (
        (
            "source_v4_public",
            source,
            (
                "admission_protocol_id",
                "admission_artifact_id",
                "admission_manifest_raw_sha256",
                "source_protocol_id",
                "public_plan_sha256",
                "public_plan_raw_sha256",
            ),
        ),
        (
            "development_reader",
            reader,
            (
                "protocol_id",
                "artifact_id",
                "manifest_raw_sha256",
                "embedding_table_artifact_id",
                "embedding_table_raw_sha256",
                "reader_artifact_id",
                "reader_artifact_raw_sha256",
            ),
        ),
        (
            "theta0_v2",
            theta,
            (
                "bootstrap_protocol_id",
                "bootstrap_artifact_id",
                "bootstrap_manifest_raw_sha256",
                "artifact_raw_sha256",
                "source_checkpoint_content_sha256",
            ),
        ),
    ):
        for field in digest_fields:
            cal._digest(mapping[field], f"{mapping_name}.{field}")
    cal._text(theta["artifact_id"], "theta0_v2.artifact_id")
    cal._text(theta["source_batch_artifact_id"], "theta0_v2.source_batch_artifact_id")
    cal._digest(
        theta["cold_checkpoint_content_sha256"],
        "theta0_v2.cold_checkpoint_content_sha256",
    )
    cal._text(theta["cold_frozen_policy_id"], "theta0_v2.cold_frozen_policy_id")
    if (
        source["public_plan_relative_path"] != "public/source_plan.json"
        or source["public_plan_schema_version"]
        != "relationship-product-horizon-public-view.v4"
    ):
        raise ValueError("source-v4 public path/schema drifted")

    expected_counts = {
        ("source_v4_public", "root_count"): 112,
        ("source_v4_public", "onboarding_session_count"): 448,
        ("source_v4_public", "decision_count"): 5376,
        ("source_v4_public", "sealed_file_read_count"): 0,
        ("scan", "reachable_first_preaction_count"): 112,
        ("scan", "collection_stress_count"): 784,
        ("scan", "evaluation_stress_count"): 4480,
        ("scan", "reset_state_counterfactual_probe_count"): 5264,
        ("scan", "total_probe_count"): 5376,
        ("scan", "public_index_bucket_count"): 6,
        ("scan", "public_index_bucket_width"): 8,
        ("scan", "public_index_bucket_probe_count"): 896,
        ("scan", "public_onboarding_outcome_read_count"): 448,
        ("scan", "natural_apply_candidate_executor_count"): 5376,
        ("scan", "environment_settlement_count"): 0,
        ("scan", "prediction_error_count"): 0,
        ("scan", "credit_count"): 0,
        ("scan", "gate_update_count"): 0,
        ("paired_witness", "strict_noop_executor_count_on_pass"): 2,
        (
            "terminal_gates",
            "reachable_first_temporal_delivered_nonnoop_min_count",
        ): 1,
        (
            "terminal_gates",
            "evaluation_stress_temporal_delivered_nonnoop_min_count",
        ): 1,
        (
            "terminal_gates",
            "evaluation_stress_temporal_delivered_nonnoop_root_min_count",
        ): 1,
        ("terminal_gates", "paired_witness_pass_count"): 2,
        ("terminal_gates", "collection_prefix_decision_count_per_root_on_success"): 8,
    }
    mappings = {
        "source_v4_public": source,
        "scan": scan,
        "paired_witness": witness,
        "terminal_gates": terminal,
    }
    for (name, field), expected in expected_counts.items():
        if cal._integer(mappings[name][field], f"{name}.{field}") != expected:
            raise ValueError(f"{name}.{field} drifted")
    for field in (
        "cold_update_count",
        "cold_processed_credit_id_count",
        "cold_pending_decision_count",
    ):
        if cal._integer(theta[field], f"theta0_v2.{field}") != 0:
            raise ValueError(f"theta0_v2.{field} must remain zero")
    for field in ("model_execution_count", "cuda_execution_count"):
        if cal._integer(reader[field], f"development_reader.{field}") != 0:
            raise ValueError(f"development_reader.{field} must remain zero")
    for field in ("condition_reader_qualified", "runtime_fit_allowed"):
        if cal._boolean(reader[field], f"development_reader.{field}"):
            raise ValueError(f"development_reader.{field} must remain false")

    expected_scan_strings = {
        "root_order": "public_root_array_order",
        "decision_order": "decision_index_0_to_47",
        "owner_state": "post_four_public_onboarding_sessions",
        "natural_executor_disposition": "apply_candidate",
        "reachable_first_preaction_definition": "decision_index_equals_0",
        "collection_stress_definition": (
            "decision_index_1_through_7_from_post_onboarding_reset_state"
        ),
        "evaluation_stress_definition": (
            "decision_index_8_through_47_from_post_onboarding_reset_state"
        ),
    }
    for field, expected in expected_scan_strings.items():
        if scan[field] != expected:
            raise ValueError(f"scan.{field} drifted")
    for field in ("owner_reset_each_root", "reset_state_probe"):
        if not cal._boolean(scan[field], f"scan.{field}"):
            raise ValueError(f"scan.{field} must remain true")
    if cal._boolean(
        scan["decision_writeback_during_scan"],
        "scan.decision_writeback_during_scan",
    ):
        raise ValueError("scan decision writeback must remain false")

    if witness["categories"] != [_REACHABLE, _EVALUATION_STRESS]:
        raise ValueError("paired witness categories drifted")
    expected_witness_strings = {
        "selection_rule": (
            "first_temporal_delivered_nonnoop_in_canonical_root_then_decision_order"
        ),
        "apply_disposition": "apply_candidate",
        "control_disposition": "force_strict_noop",
    }
    for field, expected in expected_witness_strings.items():
        if witness[field] != expected:
            raise ValueError(f"paired_witness.{field} drifted")
    for field in (
        "same_forecast_required",
        "same_owner_prestate_required",
        "same_frozen_decision_required",
        "same_candidate_advisory_required",
        "same_frozen_policy_required",
        "apply_delivers_frozen_selected_action",
        "control_delivers_neutral_noop",
        "actual_action_divergence_required",
        "cold_checkpoint_unchanged_required",
    ):
        if not cal._boolean(witness[field], f"paired_witness.{field}"):
            raise ValueError(f"paired_witness.{field} must remain true")
    if (
        terminal["successful_terminal"] != _SUCCESS_STATUS
        or terminal["failed_terminal"] != _FAIL_STATUS
    ):
        raise ValueError("transductive opportunity terminal status drifted")
    if not cal._boolean(
        terminal["collection_prefix_protocol_freeze_authorized_on_success"],
        "terminal_gates.collection_prefix_protocol_freeze_authorized_on_success",
    ):
        raise ValueError("collection-prefix protocol freeze must be authorized on success")
    for field in (
        "collection_prefix_execution_authorized_on_success",
        "full_evaluation_authorized_on_success",
        "campaign_authorized_on_success",
    ):
        if cal._boolean(terminal[field], f"terminal_gates.{field}"):
            raise ValueError(f"terminal_gates.{field} must remain false")
    if not cal._boolean(
        terminal["future_collection_first_preaction_exact_match_required"],
        "terminal_gates.future_collection_first_preaction_exact_match_required",
    ):
        raise ValueError("future collection first-preaction match must remain required")
    if (
        terminal["future_collection_first_preaction_projection_schema_version"]
        != _REACHABLE_FIRST_PROJECTION_SCHEMA_VERSION
        or terminal["future_collection_first_preaction_projection_fields"]
        != list(_REACHABLE_FIRST_PROJECTION_FIELDS)
    ):
        raise ValueError("future collection first-preaction projection drifted")
    if not cal._boolean(
        terminal[
            "scientific_retry_with_modified_theta_reader_source_threshold_or_selection_forbidden"
        ],
        "terminal_gates.scientific_retry",
    ):
        raise ValueError("transductive opportunity retry boundary drifted")

    expected_firewall_counts = {
        "source_v4_public_plan_file_read_count": 1,
        "source_v4_sealed_file_read_count": 0,
        "challenge_label_file_read_count": 0,
        "group_split_file_read_count": 0,
        "environment_settlement_count": 0,
        "prediction_error_count": 0,
        "credit_count": 0,
        "gate_update_count": 0,
        "model_output_count": 0,
        "cuda_execution_count": 0,
    }
    for field, expected in expected_firewall_counts.items():
        if cal._integer(firewall[field], f"causal_firewall.{field}") != expected:
            raise ValueError(f"causal_firewall.{field} drifted")
    if not cal._boolean(
        firewall["temporal_timestamp_excluded_from_stable_projection"],
        "causal_firewall.temporal_timestamp_excluded_from_stable_projection",
    ):
        raise ValueError("stable temporal projection declaration drifted")
    if cal._boolean(
        firewall["executor_receipt_byte_exact_claim"],
        "causal_firewall.executor_receipt_byte_exact_claim",
    ):
        raise ValueError("scanner cannot claim byte-exact executor receipts")
    true_claims = {field for field, value in claims.items() if value is True}
    if true_claims != {"scanner_execution_authorized"}:
        raise ValueError("transductive opportunity protocol claim ceiling drifted")
    if any(type(value) is not bool for value in claims.values()):
        raise ValueError("transductive opportunity claims must be booleans")
    cal._text(payload["claim_boundary"], "claim_boundary")


def _file_entries(manifest: Mapping[str, object], *, source: str) -> Mapping[str, Mapping[str, object]]:
    entries: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(cal._list(manifest["files"], f"{source} files")):
        item = cal._mapping(value, f"{source} files[{index}]")
        path = cal._text(item["path"], f"{source} files[{index}].path")
        if path in entries:
            raise ValueError(f"{source} file paths must be unique")
        entries[path] = item
    return entries


def _load_dependencies(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
) -> _Dependencies:
    protocol = load_relationship_product_horizon_transductive_public_opportunity_protocol()
    source_pin = cal._mapping(protocol.payload["source_v4_public"], "source pin")
    source_root = pathlib.Path(source_v4_admission_root)
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
        or source_manifest["public_plan_sha256"] != source_pin["public_plan_sha256"]
        or source_manifest["status"]
        != "campaign_input_admitted_execution_not_authorized"
        or source_manifest["root_count"] != 112
        or source_manifest["onboarding_session_count"] != 448
        or source_manifest["decision_count"] != 5376
        or source_manifest["claims"]["campaign_execution_authorized"] is not False
        or source_manifest["claims"]["reader_input_materialized"] is not False
        or source_manifest["claims"]["theta0_materialized"] is not False
    ):
        raise ValueError("source-v4 admission public envelope drifted")
    if source_manifest["artifact_id"] != sha256_json(
        {key: value for key, value in source_manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("source-v4 admission manifest content identity drifted")
    source_files = _file_entries(source_manifest, source="source-v4 admission")
    public_relative = cal._text(
        source_pin["public_plan_relative_path"],
        "source_v4_public.public_plan_relative_path",
    )
    if source_files[public_relative]["raw_sha256"] != source_pin[
        "public_plan_raw_sha256"
    ]:
        raise ValueError("source-v4 public plan manifest pin drifted")
    public_raw = cal._require_raw_sha(
        source_root / public_relative,
        source_pin["public_plan_raw_sha256"],
        "source-v4 public plan",
    )
    public_plan = cal._parse_json_bytes(public_raw, source="source-v4 public plan")
    if public_raw != cal._canonical_bytes(public_plan):
        raise ValueError("source-v4 public plan must use canonical bytes")
    public_view = RelationshipProductHorizonPublicView.from_payload(public_plan)
    if (
        public_view.schema_version != source_pin["public_plan_schema_version"]
        or public_view.protocol_id != source_pin["source_protocol_id"]
        or public_view.public_plan_sha256 != source_pin["public_plan_sha256"]
    ):
        raise ValueError("source-v4 owner-restored public view identity drifted")

    reader_pin = cal._mapping(protocol.payload["development_reader"], "reader pin")
    development_root = pathlib.Path(reader_root)
    reader_manifest_raw = cal._require_raw_sha(
        development_root / "manifest.json",
        reader_pin["manifest_raw_sha256"],
        "development reader manifest",
    )
    reader_manifest = cal._parse_json_bytes(
        reader_manifest_raw,
        source="development reader manifest",
    )
    if reader_manifest_raw != cal._canonical_bytes(reader_manifest):
        raise ValueError("development reader manifest must use canonical bytes")
    if (
        reader_manifest["protocol_id"] != reader_pin["protocol_id"]
        or reader_manifest["artifact_id"] != reader_pin["artifact_id"]
        or reader_manifest["embedding_table_artifact_id"]
        != reader_pin["embedding_table_artifact_id"]
        or reader_manifest["reader_artifact_id"]
        != reader_pin["reader_artifact_id"]
        or reader_manifest["source_v4_sealed_file_read_count"] != 0
        or reader_manifest["challenge_label_file_read_count"] != 0
        or reader_manifest["status"] != "development_unqualified_reader_materialized"
        or reader_manifest["claims"]["condition_reader_qualified"] is not False
        or reader_manifest["claims"]["campaign_execution_authorized"] is not False
        or reader_manifest["claims"]["readable_effect"] is not False
    ):
        raise ValueError("development reader public envelope drifted")
    if reader_manifest["artifact_id"] != sha256_json(
        {key: value for key, value in reader_manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("development reader manifest content identity drifted")
    reader_files = _file_entries(reader_manifest, source="development reader")
    if (
        reader_files["embedding_table.json"]["raw_sha256"]
        != reader_pin["embedding_table_raw_sha256"]
        or reader_files["reader_artifact.json"]["raw_sha256"]
        != reader_pin["reader_artifact_raw_sha256"]
    ):
        raise ValueError("development reader file pins drifted")
    table_path = development_root / "embedding_table.json"
    cal._require_raw_sha(
        table_path,
        reader_pin["embedding_table_raw_sha256"],
        "development embedding table",
    )
    table = load_precomputed_public_embedding_table(table_path)
    if table.artifact_id != reader_pin["embedding_table_artifact_id"]:
        raise ValueError("development embedding table identity drifted")
    reader_raw = cal._require_raw_sha(
        development_root / "reader_artifact.json",
        reader_pin["reader_artifact_raw_sha256"],
        "development reader artifact",
    )
    reader_artifact = FrozenLinearRelationshipConditionReaderArtifact.from_json(
        reader_raw.decode("utf-8")
    )
    if reader_artifact.artifact_id != reader_pin["reader_artifact_id"]:
        raise ValueError("development reader artifact identity drifted")
    semantic = PrecomputedPublicSemanticEmbedder(table)
    reader_runtime = FrozenLinearRelationshipConditionReaderRuntime(
        artifact=reader_artifact,
        embedder=semantic,
    )
    forecast_runtime = FrozenLinearRelationshipPreferenceForecastRuntime(
        reader=reader_runtime
    )

    theta_pin = cal._mapping(protocol.payload["theta0_v2"], "theta0 v2 pin")
    theta_root = pathlib.Path(theta0_v2_root)
    theta_manifest_raw = cal._require_raw_sha(
        theta_root / "manifest.json",
        theta_pin["bootstrap_manifest_raw_sha256"],
        "theta0 v2 manifest",
    )
    theta_manifest = cal._parse_json_bytes(
        theta_manifest_raw,
        source="theta0 v2 manifest",
    )
    if theta_manifest_raw != cal._canonical_bytes(theta_manifest):
        raise ValueError("theta0 v2 manifest must use canonical bytes")
    if (
        theta_manifest["protocol_id"] != theta_pin["bootstrap_protocol_id"]
        or theta_manifest["artifact_id"] != theta_pin["bootstrap_artifact_id"]
        or theta_manifest["candidate_theta0_artifact_id"] != theta_pin["artifact_id"]
        or theta_manifest["published_theta0_artifact_id"] != theta_pin["artifact_id"]
        or theta_manifest["post_checkpoint_content_sha256"]
        != theta_pin["source_checkpoint_content_sha256"]
        or theta_manifest["credit_batch_id"] != theta_pin["source_batch_artifact_id"]
        or theta_manifest["status"]
        != (
            "development_theta0_v2_materialized_training_support_"
            "opportunity_only_effect_not_tested"
        )
        or theta_manifest["claims"]["source_v4_opportunity_established"] is not False
        or theta_manifest["claims"]["campaign_execution_authorized"] is not False
    ):
        raise ValueError("theta0 v2 public envelope drifted")
    if theta_manifest["artifact_id"] != sha256_json(
        {key: value for key, value in theta_manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("theta0 v2 manifest content identity drifted")
    theta_files = _file_entries(theta_manifest, source="theta0 v2")
    if theta_files["theta0_artifact.json"]["raw_sha256"] != theta_pin[
        "artifact_raw_sha256"
    ]:
        raise ValueError("theta0 v2 artifact manifest pin drifted")
    theta_raw = cal._require_raw_sha(
        theta_root / "theta0_artifact.json",
        theta_pin["artifact_raw_sha256"],
        "theta0 v2 artifact",
    )
    theta0 = RelationshipActionGateTheta0Artifact.from_payload(
        cal._parse_json_bytes(theta_raw, source="theta0 v2 artifact")
    )
    if (
        theta0.artifact_id != theta_pin["artifact_id"]
        or theta0.source_checkpoint_content_sha256
        != theta_pin["source_checkpoint_content_sha256"]
        or theta0.source_batch_artifact_id != theta_pin["source_batch_artifact_id"]
    ):
        raise ValueError("theta0 v2 artifact identity drifted")
    gate = RelationshipActionGate.from_theta0(theta0)
    cold = gate.validate_frozen_theta0()
    frozen_policy = gate.freeze_for_evaluation()
    if (
        cold.content_sha256 != theta_pin["cold_checkpoint_content_sha256"]
        or frozen_policy.policy_id != theta_pin["cold_frozen_policy_id"]
        or cold.update_count != 0
        or cold.processed_credit_ids
        or cold.pending_decisions
    ):
        raise ValueError("theta0 v2 cold checkpoint is not 0/0/0")
    return _Dependencies(
        protocol=protocol,
        public_view=public_view,
        embedding_table=table,
        reader_artifact=reader_artifact,
        forecast_runtime=forecast_runtime,
        theta0=theta0,
        frozen_policy=frozen_policy,
    )


def _probe_category(decision_index: int) -> str:
    if isinstance(decision_index, bool) or not isinstance(decision_index, int):
        raise TypeError("decision_index must be an integer")
    if decision_index == 0:
        return _REACHABLE
    if 1 <= decision_index <= 7:
        return _COLLECTION_STRESS
    if 8 <= decision_index <= 47:
        return _EVALUATION_STRESS
    raise ValueError("decision_index is outside source-v4 public range")


def _public_index_bucket(decision_index: int) -> str:
    _probe_category(decision_index)
    return f"public_index_bucket_{decision_index // 8}"


def _authorization(
    *,
    protocol_id: str,
    frozen_policy: RelationshipActionGateFrozenPolicy,
) -> RelationshipProductFrozenPulseAuthorization:
    pulse = RelationshipProductPulseAuthorization(
        authorization_id=f"transductive-public-opportunity:{protocol_id}",
        allowed_policy_artifact_id=frozen_policy.artifact.artifact_id,
        allowed_policy_artifact_version=frozen_policy.artifact.artifact_version,
    )
    return RelationshipProductFrozenPulseAuthorization(
        pulse_authorization=pulse,
        allowed_frozen_policy_id=frozen_policy.policy_id,
        allowed_checkpoint_content_sha256=frozen_policy.checkpoint.content_sha256,
    )


def _request(
    *,
    subject_id: str,
    decision: HorizonPublicDecisionSession,
) -> RelationshipProductPreActionRequest:
    decision_index = decision.decision_index
    action_turn = 4 + decision_index * 2
    return RelationshipProductPreActionRequest(
        session_id=decision.session_id,
        forecast_request=PreferenceActionForecastRequest(
            decision_id=decision.decision_id,
            interlocutor_id=_INTERLOCUTOR_ID,
            current_observation=decision.current_input,
            observation_ref=f"public-decision:{sha256_json(decision.to_payload())}",
            candidate_action_ids=tuple(action.value for action in RELATIONSHIP_ACTIONS),
            outcome_ids=tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES),
            turn_index=action_turn,
            session_scope=subject_id,
        ),
        outcome_turn_index=action_turn + 1,
    )


async def _post_onboarding_state(root: HorizonPublicRoot) -> object:
    owner_persistence = None
    for item in root.onboarding_sessions:
        appended = await append_relationship_product_onboarding(
            owner_persistence_snapshot=owner_persistence,
            onboarding=RelationshipProductOnboardingInput(
                session_id=item.session_id,
                session_index=item.session_index,
                turn_index=item.session_index,
                public_observation=item.user_utterance,
                action_id=item.exposed_action_id,
                observed_outcome_id=item.observed_outcome_id,
                reaction_summary=item.rendered_user_reaction,
                evidence_ref=f"public-onboarding:{sha256_json(item.to_payload())}",
            ),
        )
        owner_persistence = appended.owner_persistence_snapshot
    if owner_persistence is None:
        raise RuntimeError("source-v4 root produced no post-onboarding state")
    return owner_persistence


def _nonnoop(action_id: str) -> bool:
    return action_id != RelationshipAction.NEUTRAL_NOOP.value


def _terminal_failure_reasons(
    *,
    nonnoop_counts_by_category: Mapping[str, int],
    nonnoop_root_counts_by_category: Mapping[str, int],
    witness_pass_count: int,
) -> tuple[str, ...]:
    return tuple(
        reason
        for failed, reason in (
            (
                nonnoop_counts_by_category[_REACHABLE] < 1,
                "reachable_first_temporal_delivered_nonnoop_below_one",
            ),
            (
                nonnoop_counts_by_category[_EVALUATION_STRESS] < 1,
                "evaluation_stress_temporal_delivered_nonnoop_below_one",
            ),
            (
                nonnoop_root_counts_by_category[_EVALUATION_STRESS] < 1,
                "evaluation_stress_temporal_delivered_nonnoop_root_below_one",
            ),
            (witness_pass_count != 2, "paired_witness_pass_count_not_two"),
        )
        if failed
    )


async def _run_scan(
    *,
    dependencies: _Dependencies,
    sink: cal._TraceSink,
) -> _ScanReplay:
    protocol = dependencies.protocol
    frozen_policy = dependencies.frozen_policy
    checkpoint = frozen_policy.checkpoint
    authorization = _authorization(
        protocol_id=protocol.protocol_id,
        frozen_policy=frozen_policy,
    )
    if checkpoint.update_count != 0 or checkpoint.processed_credit_ids or checkpoint.pending_decisions:
        raise RuntimeError("transductive scanner requires cold 0/0/0 theta0")
    sink.append(
        {
            "schema_version": TRANSDUCTIVE_PUBLIC_OPPORTUNITY_TRACE_SCHEMA_VERSION,
            "record_type": "header",
            "protocol_id": protocol.protocol_id,
            "source_v4_public_plan_sha256": protocol.payload["source_v4_public"][
                "public_plan_sha256"
            ],
            "development_reader_artifact_id": dependencies.reader_artifact.artifact_id,
            "theta0_v2_artifact_id": dependencies.theta0.artifact_id,
            "cold_checkpoint_content_sha256": checkpoint.content_sha256,
            "cold_frozen_policy_id": frozen_policy.policy_id,
            "owner_state": "post_four_public_onboarding_sessions",
            "decision_writeback_during_scan": False,
            "source_v4_sealed_file_read_count": 0,
            "environment_settlement_count": 0,
            "prediction_error_count": 0,
            "credit_count": 0,
            "gate_update_count": 0,
            "model_output_count": 0,
            "cuda_execution_count": 0,
        }
    )
    category_probe_counts: Counter[str] = Counter()
    public_index_bucket_counts: Counter[str] = Counter()
    selected_action_counts: Counter[str] = Counter()
    delivered_action_counts: Counter[str] = Counter()
    selected_by_category: dict[str, Counter[str]] = {
        _REACHABLE: Counter(),
        _COLLECTION_STRESS: Counter(),
        _EVALUATION_STRESS: Counter(),
    }
    delivered_by_category: dict[str, Counter[str]] = {
        _REACHABLE: Counter(),
        _COLLECTION_STRESS: Counter(),
        _EVALUATION_STRESS: Counter(),
    }
    nonnoop_roots: dict[str, set[int]] = {
        _REACHABLE: set(),
        _COLLECTION_STRESS: set(),
        _EVALUATION_STRESS: set(),
    }
    witness_candidates: dict[
        str,
        tuple[
            int,
            HorizonPublicRoot,
            object,
            HorizonPublicDecisionSession,
            RelationshipProductFrozenPreActionSnapshot,
        ],
    ] = {}
    reachable_first_projection: list[Mapping[str, object]] = []
    for root_index, root in enumerate(dependencies.public_view.roots):
        subject_id = root.subject_id
        owner_persistence = await _post_onboarding_state(root)
        owner_sha = social_record_store_persistence_sha256(owner_persistence)
        sink.append(
            {
                "schema_version": TRANSDUCTIVE_PUBLIC_OPPORTUNITY_TRACE_SCHEMA_VERSION,
                "record_type": "root_start",
                "root_sequence_index": root_index,
                "subject_id": subject_id,
                "public_onboarding_outcome_read_count": 4,
                "post_onboarding_persistence_sha256": owner_sha,
                "owner_reset": True,
            }
        )
        for decision in root.decision_sessions:
            decision_index = decision.decision_index
            category = _probe_category(decision_index)
            public_index_bucket = _public_index_bucket(decision_index)
            preaction = await prepare_relationship_product_frozen_preaction(
                request=_request(subject_id=subject_id, decision=decision),
                owner_persistence_snapshot=owner_persistence,
                forecast_runtime=dependencies.forecast_runtime,
                frozen_policy=frozen_policy,
                executor_disposition=(
                    RelationshipProductExecutorDisposition.APPLY_CANDIDATE
                ),
                authorization=authorization,
                substrate_snapshot=cal._placeholder_substrate(),
            )
            if preaction.frozen_policy != frozen_policy:
                raise RuntimeError("transductive scan frozen policy drifted")
            gate_decision = preaction.frozen_decision.decision
            selected = gate_decision.selected_action_id
            delivered = preaction.delivered_action_id
            if delivered != selected:
                raise RuntimeError("APPLY_CANDIDATE did not deliver frozen selected action")
            if preaction.execution_receipt.temporal_delivery.active_abstract_action != delivered:
                raise RuntimeError("temporal actual action drifted from executor delivery")
            category_probe_counts[category] += 1
            public_index_bucket_counts[public_index_bucket] += 1
            selected_action_counts[selected] += 1
            delivered_action_counts[delivered] += 1
            selected_by_category[category][selected] += 1
            delivered_by_category[category][delivered] += 1
            if _nonnoop(delivered):
                nonnoop_roots[category].add(root_index)
                if category in (_REACHABLE, _EVALUATION_STRESS):
                    witness_candidates.setdefault(
                        category,
                        (root_index, root, owner_persistence, decision, preaction),
                    )
            readout = preaction.forecast.condition_readout
            if readout is None:
                raise RuntimeError("transductive scan forecast omitted named readout")
            receipt = preaction.execution_receipt
            temporal = receipt.temporal_delivery
            forecast_sha256 = sha256_json(
                preference_action_forecast_to_payload(preaction.forecast)
            )
            owner_output_sha = social_record_store_persistence_sha256(
                preaction.owner_persistence_snapshot
            )
            frozen_decision_sha256 = sha256_json(
                preaction.frozen_decision.to_payload()
            )
            if category == _REACHABLE:
                reachable_first_projection.append(
                    {
                        "schema_version": _REACHABLE_FIRST_PROJECTION_SCHEMA_VERSION,
                        "root_sequence_index": root_index,
                        "subject_id": subject_id,
                        "decision_index": decision_index,
                        "session_id": decision.session_id,
                        "decision_id": decision.decision_id,
                        "owner_input_persistence_sha256": owner_sha,
                        "owner_output_persistence_sha256": owner_output_sha,
                        "forecast_sha256": forecast_sha256,
                        "frozen_decision_sha256": frozen_decision_sha256,
                        "gate_action": gate_decision.gate_action.value,
                        "steer_probability_hex": gate_decision.steer_probability.hex(),
                        "frozen_selected_action_id": selected,
                        "executor_disposition": (
                            RelationshipProductExecutorDisposition.APPLY_CANDIDATE.value
                        ),
                        "executor_status": receipt.executor_status.value,
                        "candidate_advisory_id": receipt.candidate_advisory.advisory_id,
                        "delivered_advisory_id": receipt.delivered_advisory.advisory_id,
                        "temporal_delivered_action_id": delivered,
                        "temporal_controller_params_hash": (
                            temporal.controller_params_hash
                        ),
                        "temporal_action_family_version": temporal.action_family_version,
                        "temporal_action_advisory_status": (
                            temporal.action_advisory_status.value
                        ),
                        "cold_checkpoint_content_sha256": checkpoint.content_sha256,
                        "cold_frozen_policy_id": frozen_policy.policy_id,
                    }
                )
            sink.append(
                {
                    "schema_version": TRANSDUCTIVE_PUBLIC_OPPORTUNITY_TRACE_SCHEMA_VERSION,
                    "record_type": "probe",
                    "global_sequence_index": root_index * 48 + decision_index,
                    "root_sequence_index": root_index,
                    "subject_id": subject_id,
                    "decision_index": decision_index,
                    "session_id": decision.session_id,
                    "decision_id": decision.decision_id,
                    "probe_category": category,
                    "public_index_bucket": public_index_bucket,
                    "reset_state_probe": True,
                    "dynamically_reachable_preaction": category == _REACHABLE,
                    "owner_input_persistence_sha256": owner_sha,
                    "owner_output_persistence_sha256": (
                        owner_output_sha
                    ),
                    "forecast_sha256": forecast_sha256,
                    "frozen_decision_sha256": frozen_decision_sha256,
                    "condition_label": readout.condition_label,
                    "condition_confidence_hex": readout.confidence.hex(),
                    "recommended_action_id": preaction.forecast.recommended_action_id,
                    "gate_action": gate_decision.gate_action.value,
                    "steer_probability_hex": gate_decision.steer_probability.hex(),
                    "frozen_selected_action_id": selected,
                    "executor_disposition": (
                        RelationshipProductExecutorDisposition.APPLY_CANDIDATE.value
                    ),
                    "executor_status": preaction.execution_receipt.executor_status.value,
                    "candidate_advisory_id": receipt.candidate_advisory.advisory_id,
                    "delivered_advisory_id": receipt.delivered_advisory.advisory_id,
                    "temporal_delivered_action_id": delivered,
                    "temporal_controller_params_hash": temporal.controller_params_hash,
                    "temporal_action_family_version": temporal.action_family_version,
                    "temporal_action_advisory_status": (
                        temporal.action_advisory_status.value
                    ),
                    "cold_checkpoint_content_sha256": checkpoint.content_sha256,
                    "cold_update_count": checkpoint.update_count,
                    "cold_processed_credit_id_count": len(checkpoint.processed_credit_ids),
                    "cold_pending_decision_count": len(checkpoint.pending_decisions),
                    "temporal_timestamp_excluded_from_stable_projection": True,
                    "executor_receipt_id_excluded_from_stable_projection": True,
                    "environment_settle_called": False,
                }
            )

    expected_category_counts = {
        _REACHABLE: 112,
        _COLLECTION_STRESS: 784,
        _EVALUATION_STRESS: 4480,
    }
    if dict(category_probe_counts) != expected_category_counts:
        raise RuntimeError("transductive scan category inventory drifted")
    expected_bucket_counts = {
        f"public_index_bucket_{index}": 896 for index in range(6)
    }
    if dict(public_index_bucket_counts) != expected_bucket_counts:
        raise RuntimeError("transductive scan public index buckets drifted")
    if len(reachable_first_projection) != 112:
        raise RuntimeError("transductive reachable-first projection count drifted")
    if any(
        tuple(item.keys()) != _REACHABLE_FIRST_PROJECTION_FIELDS
        for item in reachable_first_projection
    ):
        raise RuntimeError("transductive reachable-first projection schema drifted")
    reachable_first_projection_sha256 = sha256_json(reachable_first_projection)
    witnesses = []
    witness_pass_count = 0
    strict_executor_count = 0
    for category in (_REACHABLE, _EVALUATION_STRESS):
        candidate = witness_candidates.get(category)
        if candidate is None:
            witnesses.append(
                {
                    "category": category,
                    "status": "no_temporal_delivered_nonnoop",
                    "passed": False,
                }
            )
            continue
        root_index, root, owner_persistence, decision, apply_preaction = candidate
        subject_id = root.subject_id
        strict_preaction = await prepare_relationship_product_frozen_preaction(
            request=_request(subject_id=subject_id, decision=decision),
            owner_persistence_snapshot=owner_persistence,
            forecast_runtime=dependencies.forecast_runtime,
            frozen_policy=frozen_policy,
            executor_disposition=RelationshipProductExecutorDisposition.FORCE_STRICT_NOOP,
            authorization=authorization,
            substrate_snapshot=cal._placeholder_substrate(),
        )
        strict_executor_count += 1
        same_forecast = strict_preaction.forecast == apply_preaction.forecast
        same_decision = (
            strict_preaction.frozen_decision == apply_preaction.frozen_decision
        )
        same_owner_prestate = (
            strict_preaction.execution_receipt.command.owner_prestate_sha256
            == apply_preaction.execution_receipt.command.owner_prestate_sha256
        )
        same_candidate_advisory = (
            strict_preaction.execution_receipt.candidate_advisory
            == apply_preaction.execution_receipt.candidate_advisory
        )
        same_frozen_policy = (
            strict_preaction.frozen_policy
            == apply_preaction.frozen_policy
            == frozen_policy
        )
        apply_receipt = apply_preaction.execution_receipt
        strict_receipt = strict_preaction.execution_receipt
        selected = apply_preaction.frozen_decision.decision.selected_action_id
        apply_delivered = apply_preaction.delivered_action_id
        strict_delivered = strict_preaction.delivered_action_id
        passed = (
            same_forecast
            and same_decision
            and same_owner_prestate
            and same_candidate_advisory
            and same_frozen_policy
            and _nonnoop(selected)
            and apply_delivered == selected
            and strict_delivered == RelationshipAction.NEUTRAL_NOOP.value
            and apply_delivered != strict_delivered
            and apply_receipt.executor_status.value == "applied_candidate"
            and strict_receipt.executor_status.value == "strict_noop"
            and apply_receipt.candidate_applied
            and not apply_receipt.strict_noop_substituted
            and not apply_receipt.action_diverged
            and not strict_receipt.candidate_applied
            and strict_receipt.strict_noop_substituted
            and strict_receipt.action_diverged
            and apply_preaction.frozen_policy.checkpoint == checkpoint
            and strict_preaction.frozen_policy.checkpoint == checkpoint
        )
        if passed:
            witness_pass_count += 1
        witnesses.append(
            {
                "category": category,
                "status": "paired" if passed else "paired_contract_failed",
                "passed": passed,
                "selection_rule": (
                    "first_temporal_delivered_nonnoop_in_canonical_root_then_decision_order"
                ),
                "root_sequence_index": root_index,
                "subject_id": subject_id,
                "decision_index": decision.decision_index,
                "session_id": decision.session_id,
                "decision_id": decision.decision_id,
                "forecast_sha256": sha256_json(
                    preference_action_forecast_to_payload(apply_preaction.forecast)
                ),
                "same_forecast": same_forecast,
                "same_frozen_decision": same_decision,
                "same_owner_prestate": same_owner_prestate,
                "same_candidate_advisory": same_candidate_advisory,
                "same_frozen_policy": same_frozen_policy,
                "candidate_advisory_id": (
                    apply_preaction.execution_receipt.candidate_advisory.advisory_id
                ),
                "owner_prestate_sha256": (
                    apply_preaction.execution_receipt.command.owner_prestate_sha256
                ),
                "frozen_selected_action_id": selected,
                "apply_executor_status": (
                    apply_receipt.executor_status.value
                ),
                "apply_candidate_applied": apply_receipt.candidate_applied,
                "apply_strict_noop_substituted": (
                    apply_receipt.strict_noop_substituted
                ),
                "apply_action_diverged": apply_receipt.action_diverged,
                "apply_temporal_delivered_action_id": apply_delivered,
                "strict_executor_status": (
                    strict_receipt.executor_status.value
                ),
                "strict_candidate_applied": strict_receipt.candidate_applied,
                "strict_noop_substituted": strict_receipt.strict_noop_substituted,
                "strict_action_diverged": strict_receipt.action_diverged,
                "strict_temporal_delivered_action_id": strict_delivered,
                "apply_delivered_advisory_id": (
                    apply_preaction.execution_receipt.delivered_advisory.advisory_id
                ),
                "strict_delivered_advisory_id": (
                    strict_preaction.execution_receipt.delivered_advisory.advisory_id
                ),
                "actual_action_divergence": apply_delivered != strict_delivered,
                "cold_checkpoint_content_sha256": checkpoint.content_sha256,
                "cold_update_count": checkpoint.update_count,
                "cold_processed_credit_id_count": len(checkpoint.processed_credit_ids),
                "cold_pending_decision_count": len(checkpoint.pending_decisions),
                "temporal_timestamps_excluded_from_stable_projection": True,
                "executor_receipt_ids_excluded_from_stable_projection": True,
            }
        )
    witness_core = {
        "schema_version": TRANSDUCTIVE_PUBLIC_OPPORTUNITY_WITNESS_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "theta0_v2_artifact_id": dependencies.theta0.artifact_id,
        "witnesses": witnesses,
    }
    witness_artifact = {"artifact_id": sha256_json(witness_core), **witness_core}
    nonnoop_counts = {
        category: sum(
            count
            for action_id, count in delivered_by_category[category].items()
            if _nonnoop(action_id)
        )
        for category in (_REACHABLE, _COLLECTION_STRESS, _EVALUATION_STRESS)
    }
    nonnoop_root_counts = {
        category: len(nonnoop_roots[category])
        for category in (_REACHABLE, _COLLECTION_STRESS, _EVALUATION_STRESS)
    }
    failure_reasons = _terminal_failure_reasons(
        nonnoop_counts_by_category=nonnoop_counts,
        nonnoop_root_counts_by_category=nonnoop_root_counts,
        witness_pass_count=witness_pass_count,
    )
    status = _SUCCESS_STATUS if not failure_reasons else _FAIL_STATUS
    sink.append(
        {
            "schema_version": TRANSDUCTIVE_PUBLIC_OPPORTUNITY_TRACE_SCHEMA_VERSION,
            "record_type": "terminal",
            "category_probe_counts": dict(sorted(category_probe_counts.items())),
            "public_index_bucket_counts": dict(
                sorted(public_index_bucket_counts.items())
            ),
            "frozen_selected_action_counts": dict(
                sorted(selected_action_counts.items())
            ),
            "temporal_delivered_action_counts": dict(
                sorted(delivered_action_counts.items())
            ),
            "frozen_selected_action_counts_by_category": {
                category: dict(sorted(selected_by_category[category].items()))
                for category in (_REACHABLE, _COLLECTION_STRESS, _EVALUATION_STRESS)
            },
            "temporal_delivered_action_counts_by_category": {
                category: dict(sorted(delivered_by_category[category].items()))
                for category in (_REACHABLE, _COLLECTION_STRESS, _EVALUATION_STRESS)
            },
            "temporal_delivered_nonnoop_counts_by_category": nonnoop_counts,
            "temporal_delivered_nonnoop_root_counts_by_category": nonnoop_root_counts,
            "reachable_first_projection_sha256": reachable_first_projection_sha256,
            "reachable_first_projection_schema_version": (
                _REACHABLE_FIRST_PROJECTION_SCHEMA_VERSION
            ),
            "reachable_first_projection_fields": list(
                _REACHABLE_FIRST_PROJECTION_FIELDS
            ),
            "witness_artifact_id": witness_artifact["artifact_id"],
            "witness_pass_count": witness_pass_count,
            "strict_noop_witness_executor_count": strict_executor_count,
            "terminal_failure_reasons": list(failure_reasons),
            "terminal_status": status,
            "collection_prefix_protocol_freeze_authorized": not failure_reasons,
            "collection_prefix_execution_authorized": False,
            "full_evaluation_authorized": False,
            "campaign_execution_authorized": False,
        }
    )
    return _ScanReplay(
        category_probe_counts=dict(sorted(category_probe_counts.items())),
        public_index_bucket_counts=dict(sorted(public_index_bucket_counts.items())),
        selected_action_counts=dict(sorted(selected_action_counts.items())),
        delivered_action_counts=dict(sorted(delivered_action_counts.items())),
        selected_action_counts_by_category={
            category: dict(sorted(selected_by_category[category].items()))
            for category in (_REACHABLE, _COLLECTION_STRESS, _EVALUATION_STRESS)
        },
        delivered_action_counts_by_category={
            category: dict(sorted(delivered_by_category[category].items()))
            for category in (_REACHABLE, _COLLECTION_STRESS, _EVALUATION_STRESS)
        },
        nonnoop_counts_by_category=nonnoop_counts,
        nonnoop_root_counts_by_category=nonnoop_root_counts,
        reachable_first_projection_sha256=reachable_first_projection_sha256,
        witness_artifact=witness_artifact,
        witness_pass_count=witness_pass_count,
        strict_noop_witness_executor_count=strict_executor_count,
        terminal_failure_reasons=failure_reasons,
        terminal_status=status,
    )


def materialize_relationship_product_horizon_transductive_public_opportunity(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    commit = cal._git_commit(implementation_git_commit)
    root = pathlib.Path(output_dir)
    if root.exists():
        raise FileExistsError(f"transductive opportunity root is create-only: {root}")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
    )
    root.mkdir(parents=True, exist_ok=False)
    cal._write_create_only(root / "protocol.json", dependencies.protocol.raw_bytes)
    sink = cal._FsyncTraceSink(root / _TRACE_FILENAME)
    try:
        replay = asyncio.run(_run_scan(dependencies=dependencies, sink=sink))
    finally:
        sink.close()
    cal._write_create_only(
        root / _WITNESS_FILENAME,
        cal._canonical_bytes(replay.witness_artifact),
    )
    manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        replay=replay,
        implementation_git_commit=commit,
    )
    cal._write_create_only(root / "manifest.json", cal._canonical_bytes(manifest))
    return manifest


def validate_relationship_product_horizon_transductive_public_opportunity(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
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
        source="transductive opportunity manifest",
    )
    if manifest_raw != cal._canonical_bytes(manifest):
        raise ValueError("transductive opportunity manifest must use canonical bytes")
    if manifest["protocol_id"] != external_protocol:
        raise ValueError("external transductive opportunity protocol ID drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("external transductive opportunity artifact ID drifted")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
    )
    if dependencies.protocol.protocol_id != external_protocol:
        raise ValueError("packaged transductive opportunity protocol ID drifted")
    if cal._read_regular(root / "protocol.json") != dependencies.protocol.raw_bytes:
        raise ValueError("persisted transductive opportunity protocol bytes drifted")
    sink = cal._MemoryTraceSink()
    replay = asyncio.run(_run_scan(dependencies=dependencies, sink=sink))
    if cal._read_regular(root / _TRACE_FILENAME) != sink.raw_bytes:
        raise ValueError("transductive opportunity stable trace bytes drifted")
    if cal._read_regular(root / _WITNESS_FILENAME) != cal._canonical_bytes(
        replay.witness_artifact
    ):
        raise ValueError("transductive opportunity witness bytes drifted")
    if cal._regular_file_inventory(root) != _OUTPUT_FILES:
        raise ValueError("transductive opportunity output file inventory drifted")
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
        raise ValueError("transductive opportunity manifest content drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("transductive opportunity artifact identity drifted")
    return manifest


def _build_manifest(
    *,
    root: pathlib.Path,
    dependencies: _Dependencies,
    replay: _ScanReplay,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    files = []
    for relative in ("protocol.json", _TRACE_FILENAME, _WITNESS_FILENAME):
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
        "scanner_completed": True,
        "source_v4_opportunity_established": success,
        "collection_prefix_protocol_freeze_authorized": success,
        "collection_prefix_execution_authorized": False,
        "full_evaluation_authorized": False,
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
    core = {
        "schema_version": TRANSDUCTIVE_PUBLIC_OPPORTUNITY_MANIFEST_SCHEMA_VERSION,
        "protocol_id": dependencies.protocol.protocol_id,
        "protocol_raw_sha256": dependencies.protocol.raw_sha256,
        "implementation_git_commit": implementation_git_commit,
        "source_v4_admission_artifact_id": dependencies.protocol.payload[
            "source_v4_public"
        ]["admission_artifact_id"],
        "source_v4_public_plan_sha256": dependencies.protocol.payload[
            "source_v4_public"
        ]["public_plan_sha256"],
        "development_reader_artifact_id": dependencies.protocol.payload[
            "development_reader"
        ]["artifact_id"],
        "embedding_table_artifact_id": dependencies.embedding_table.artifact_id,
        "reader_artifact_id": dependencies.reader_artifact.artifact_id,
        "theta0_v2_bootstrap_artifact_id": dependencies.protocol.payload[
            "theta0_v2"
        ]["bootstrap_artifact_id"],
        "theta0_v2_artifact_id": dependencies.theta0.artifact_id,
        "cold_checkpoint_content_sha256": dependencies.frozen_policy.checkpoint.content_sha256,
        "cold_frozen_policy_id": dependencies.frozen_policy.policy_id,
        "cold_update_count": dependencies.frozen_policy.checkpoint.update_count,
        "cold_processed_credit_id_count": len(
            dependencies.frozen_policy.checkpoint.processed_credit_ids
        ),
        "cold_pending_decision_count": len(
            dependencies.frozen_policy.checkpoint.pending_decisions
        ),
        "root_count": 112,
        "public_onboarding_outcome_read_count": 448,
        "category_probe_counts": replay.category_probe_counts,
        "reset_state_counterfactual_probe_count": 5264,
        "public_index_bucket_counts": replay.public_index_bucket_counts,
        "frozen_selected_action_counts": replay.selected_action_counts,
        "temporal_delivered_action_counts": replay.delivered_action_counts,
        "frozen_selected_action_counts_by_category": (
            replay.selected_action_counts_by_category
        ),
        "temporal_delivered_action_counts_by_category": (
            replay.delivered_action_counts_by_category
        ),
        "temporal_delivered_nonnoop_counts_by_category": (
            replay.nonnoop_counts_by_category
        ),
        "temporal_delivered_nonnoop_root_counts_by_category": (
            replay.nonnoop_root_counts_by_category
        ),
        "reachable_first_projection_sha256": (
            replay.reachable_first_projection_sha256
        ),
        "reachable_first_projection_schema_version": (
            _REACHABLE_FIRST_PROJECTION_SCHEMA_VERSION
        ),
        "reachable_first_projection_fields": list(
            _REACHABLE_FIRST_PROJECTION_FIELDS
        ),
        "future_collection_first_preaction_exact_match_required": True,
        "natural_apply_candidate_executor_count": 5376,
        "strict_noop_witness_executor_count": replay.strict_noop_witness_executor_count,
        "total_executor_attempt_count": (
            5376 + replay.strict_noop_witness_executor_count
        ),
        "witness_artifact_id": replay.witness_artifact["artifact_id"],
        "witness_pass_count": replay.witness_pass_count,
        "terminal_failure_reasons": list(replay.terminal_failure_reasons),
        "source_v4_public_plan_file_read_count": 1,
        "source_v4_sealed_file_read_count": 0,
        "challenge_label_file_read_count": 0,
        "group_split_file_read_count": 0,
        "environment_settlement_count": 0,
        "prediction_error_count": 0,
        "credit_count": 0,
        "gate_update_count": 0,
        "model_output_count": 0,
        "cuda_execution_count": 0,
        "temporal_timestamp_excluded_from_stable_projection": True,
        "files": files,
        "status": replay.terminal_status,
        "claims": claims,
        "claim_boundary": dependencies.protocol.payload["claim_boundary"],
    }
    return {"artifact_id": sha256_json(core), **core}


__all__ = [
    "TRANSDUCTIVE_PUBLIC_OPPORTUNITY_MANIFEST_SCHEMA_VERSION",
    "TRANSDUCTIVE_PUBLIC_OPPORTUNITY_PROTOCOL_SCHEMA_VERSION",
    "TRANSDUCTIVE_PUBLIC_OPPORTUNITY_TRACE_SCHEMA_VERSION",
    "TRANSDUCTIVE_PUBLIC_OPPORTUNITY_WITNESS_SCHEMA_VERSION",
    "RelationshipProductHorizonTransductivePublicOpportunityProtocol",
    "load_relationship_product_horizon_transductive_public_opportunity_protocol",
    "materialize_relationship_product_horizon_transductive_public_opportunity",
    "relationship_product_horizon_transductive_public_opportunity_protocol_path",
    "validate_relationship_product_horizon_transductive_public_opportunity",
]
