"""Development-only forced-batch stabilization of the Product Horizon theta0.

This workflow starts from the immutable failed-but-loadable theta0 v1.  It freezes
one public-position-only forced-action schedule, settles all 192 actual actions to
PE-derived credit without online gate updates, and applies the exact chronological
batch once.  The source has already been used, so every output is adaptive
development evidence only.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, replace
import math
import pathlib
from typing import Mapping

from lifeform_domain_emogpt.lab.contracts import sha256_json
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductForcedActionRole,
    RelationshipProductForcedCollectionAuthorization,
    RelationshipProductForcedCollectionScheduleArtifact,
    RelationshipProductForcedCollectionScheduleEntry,
    RelationshipProductFrozenPulseAuthorization,
    RelationshipProductOnboardingInput,
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    append_relationship_product_onboarding,
    prepare_relationship_product_forced_collection_preaction,
    settle_relationship_product_forced_collection,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateBatchDisposition,
    RelationshipActionGateBatchReceipt,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateCreditBatch,
    RelationshipActionGateTheta0Artifact,
)
from lifeform_evolution import relationship_product_horizon_theta0_calibration as v1
from volvence_zero.social import (
    PreferenceActionForecastRequest,
    social_record_store_persistence_sha256,
)
from volvence_zero.social_cognition import preference_action_forecast_to_payload


THETA0_V2_BOOTSTRAP_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-v2-bootstrap-protocol.v1"
)
THETA0_V2_BOOTSTRAP_TRACE_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-v2-bootstrap-trace.v1"
)
THETA0_V2_BOOTSTRAP_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-v2-bootstrap-manifest.v1"
)

_PROTOCOL_FILENAME = "relationship_product_horizon_theta0_v2_bootstrap_v1.json"
_TRACE_FILENAME = "forced_batch_trace.jsonl"
_BASE_OUTPUT_FILES = frozenset(
    {
        "protocol.json",
        "schedule.json",
        _TRACE_FILENAME,
        "credit_batch.json",
        "apply_receipt.json",
        "withhold_receipt.json",
        "manifest.json",
    }
)
_SUCCESS_OUTPUT_FILES = frozenset({*_BASE_OUTPUT_FILES, "theta0_artifact.json"})
_INTERLOCUTOR_ID = "primary"
_SUCCESS_STATUS = (
    "development_theta0_v2_materialized_training_support_opportunity_only_"
    "effect_not_tested"
)
_FAIL_STATUS = (
    "development_theta0_v2_batch_applied_training_support_all_noop_"
    "no_consumable_theta0"
)
_TERMINAL_GATE_FAIL_STATUS = (
    "development_theta0_v2_bootstrap_terminal_gate_failed_no_consumable_theta0"
)


@dataclass(frozen=True)
class RelationshipProductHorizonTheta0V2BootstrapProtocol:
    payload: Mapping[str, object]
    raw_bytes: bytes
    protocol_id: str
    raw_sha256: str


@dataclass(frozen=True)
class _Dependencies:
    protocol: RelationshipProductHorizonTheta0V2BootstrapProtocol
    inherited: v1._Dependencies
    base_theta0: RelationshipActionGateTheta0Artifact


@dataclass(frozen=True)
class _BootstrapReplay:
    batch: RelationshipActionGateCreditBatch
    apply_receipt: RelationshipActionGateBatchReceipt
    withhold_receipt: RelationshipActionGateBatchReceipt
    post_checkpoint: RelationshipActionGateCheckpoint
    candidate_theta0: RelationshipActionGateTheta0Artifact | None
    published_theta0: RelationshipActionGateTheta0Artifact | None
    root_mapping: tuple[Mapping[str, object], ...]
    scheduled_role_counts: Mapping[str, int]
    forced_action_counts: Mapping[str, int]
    training_support_selected_action_counts: Mapping[str, int]
    terminal_failure_reasons: tuple[str, ...]
    terminal_status: str


def relationship_product_horizon_theta0_v2_bootstrap_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).with_name("protocols") / _PROTOCOL_FILENAME


def load_relationship_product_horizon_theta0_v2_bootstrap_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonTheta0V2BootstrapProtocol:
    protocol_path = pathlib.Path(
        path or relationship_product_horizon_theta0_v2_bootstrap_protocol_path()
    )
    raw = protocol_path.read_bytes()
    payload = v1._parse_json_bytes(raw, source="theta0 v2 bootstrap protocol")
    v1._exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "purpose",
            "inherited_lineage",
            "base_theta0",
            "forced_schedule",
            "bootstrap",
            "terminal_gates",
            "causal_firewall",
            "claims",
            "claim_boundary",
        },
        "theta0 v2 bootstrap protocol",
    )
    if payload["schema_version"] != THETA0_V2_BOOTSTRAP_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("theta0 v2 bootstrap protocol schema drifted")
    if payload["evidence_tier"] != "development":
        raise ValueError("theta0 v2 bootstrap evidence tier drifted")
    if payload["owner"] != (
        "lifeform_evolution.relationship_product_horizon_theta0_v2_bootstrap"
    ):
        raise ValueError("theta0 v2 bootstrap owner drifted")
    if payload["purpose"] != (
        "adaptive_development_forced_batch_terminal_stabilization"
    ):
        raise ValueError("theta0 v2 bootstrap purpose drifted")
    _validate_protocol(payload)
    return RelationshipProductHorizonTheta0V2BootstrapProtocol(
        payload=payload,
        raw_bytes=raw,
        protocol_id=sha256_json(payload),
        raw_sha256=v1._sha256_bytes(raw),
    )


def _validate_protocol(payload: Mapping[str, object]) -> None:
    inherited = v1._mapping(payload["inherited_lineage"], "inherited_lineage")
    base = v1._mapping(payload["base_theta0"], "base_theta0")
    schedule = v1._mapping(payload["forced_schedule"], "forced_schedule")
    bootstrap = v1._mapping(payload["bootstrap"], "bootstrap")
    terminal = v1._mapping(payload["terminal_gates"], "terminal_gates")
    firewall = v1._mapping(payload["causal_firewall"], "causal_firewall")
    claims = v1._mapping(payload["claims"], "claims")

    v1._exact_keys(
        inherited,
        {
            "theta0_v1_protocol_id",
            "theta0_v1_protocol_raw_sha256",
            "theta0_v1_calibration_artifact_id",
            "theta0_v1_manifest_raw_sha256",
            "source_v3_admission_artifact_id",
            "development_reader_artifact_id",
            "source_v3_already_used_for_theta0_v1_training",
            "source_v3_already_used_for_terminal_failure_diagnosis",
            "source_v3_unseen_evidence",
            "adaptive_double_use_declared",
        },
        "inherited_lineage",
    )
    v1._exact_keys(
        base,
        {
            "artifact_id",
            "artifact_raw_sha256",
            "source_checkpoint_content_sha256",
            "cold_checkpoint_content_sha256",
            "cold_frozen_policy_id",
            "cold_update_count",
            "cold_processed_credit_id_count",
            "cold_pending_decision_count",
            "terminal_source_v3_nonnoop_count_before_v2",
            "terminal_source_v4_public_reset_state_nonnoop_count_before_v2",
        },
        "base_theta0",
    )
    v1._exact_keys(
        schedule,
        {
            "root_count",
            "decision_per_root",
            "entry_count",
            "global_order",
            "global_sequence_index",
            "role_formula",
            "expected_schedule_artifact_id",
            "owner_recommendation_role_count",
            "neutral_noop_role_count",
            "owner_recommendation_per_root",
            "neutral_noop_per_root",
            "role_frozen_before_any_environment_construction",
            "role_frozen_before_forecast_and_outcome",
            "role_depends_on_forecast",
            "role_depends_on_outcome",
            "role_depends_on_sealed_truth",
            "symbolic_role_balance_not_concrete_action_balance",
        },
        "forced_schedule",
    )
    v1._exact_keys(
        bootstrap,
        {
            "owner_reset_each_root",
            "owner_state_sequential_within_root",
            "single_base_cold_policy_for_all_exposures",
            "forced_preaction_owner",
            "actual_action_owner",
            "environment_owner_api",
            "settlement_owner",
            "learning_source",
            "online_gate_credit_apply",
            "credit_timestamp_formula",
            "credit_timestamp_strictly_increasing",
            "preaction_trace_fsync_before_environment_settle",
            "credit_batch_order_sensitive",
            "credit_batch_operator",
            "credit_batch_dispositions",
            "apply_atomic_commit_count",
            "withhold_atomic_commit_count",
            "apply_and_withhold_share_exact_batch_and_plan",
            "theta0_v2_source_batch_identity",
            "threshold_or_bias_tuning",
            "early_checkpoint_selection",
            "seed_or_order_selection",
        },
        "bootstrap",
    )
    v1._exact_keys(
        terminal,
        {
            "forced_exposure_count",
            "credit_count",
            "unique_credit_id_count",
            "batch_update_count_delta",
            "batch_atomic_commit_count",
            "withhold_update_count_delta",
            "withhold_parameter_delta_all_zero",
            "withhold_post_checkpoint_equals_base_cold_checkpoint",
            "post_batch_pending_decision_count",
            "theta0_v2_cold_update_count",
            "theta0_v2_cold_processed_credit_id_count",
            "theta0_v2_cold_pending_decision_count",
            "training_support_physical_nonnoop_min_count",
            "forced_actual_physical_nonnoop_min_count",
            "forced_actual_neutral_noop_min_count",
            "parameter_delta_nonzero_required",
            "terminal_parameter_finite_required",
            "terminal_parameter_nonzero_required",
            "terminal_parameter_cap_hit_forbidden",
            "forced_actual_physical_nonnoop_definition",
            "training_support_physical_nonnoop_definition",
            "all_noop_terminal",
            "other_terminal_gate_failure",
            "successful_terminal",
            "scientific_retry_with_different_schedule_order_base_threshold_or_seed_forbidden",
            "source_v4_public_opportunity_scan_in_this_protocol",
            "campaign_authorized_on_success",
        },
        "terminal_gates",
    )
    v1._exact_keys(
        firewall,
        {
            "environment_safe_projection_fields",
            "hidden_fields_forbidden_from_reader_gate_credit",
            "evaluation_or_judge_feedback_to_learning",
            "oracle_input_to_learning",
            "model_output_count",
            "cuda_execution_count",
            "os_process_secrecy_claim",
            "disk_transaction_atomicity_claim",
        },
        "causal_firewall",
    )
    v1._exact_keys(
        claims,
        {
            "bootstrap_execution_authorized",
            "development_theta0_v2_may_be_materialized",
            "source_v4_opportunity_established",
            "reader_qualified",
            "campaign_execution_authorized",
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
        "claims",
    )

    if not all(
        v1._boolean(inherited[field], f"inherited_lineage.{field}")
        for field in (
            "source_v3_already_used_for_theta0_v1_training",
            "source_v3_already_used_for_terminal_failure_diagnosis",
            "adaptive_double_use_declared",
        )
    ):
        raise ValueError("theta0 v2 adaptive reuse declarations must remain true")
    if v1._boolean(
        inherited["source_v3_unseen_evidence"],
        "inherited_lineage.source_v3_unseen_evidence",
    ):
        raise ValueError("theta0 v2 cannot claim source-v3 is unseen")
    for field in (
        "theta0_v1_protocol_id",
        "theta0_v1_protocol_raw_sha256",
        "theta0_v1_calibration_artifact_id",
        "theta0_v1_manifest_raw_sha256",
        "source_v3_admission_artifact_id",
        "development_reader_artifact_id",
    ):
        v1._digest(inherited[field], f"inherited_lineage.{field}")

    v1._text(base["artifact_id"], "base_theta0.artifact_id")
    for field in (
        "artifact_raw_sha256",
        "source_checkpoint_content_sha256",
        "cold_checkpoint_content_sha256",
    ):
        v1._digest(base[field], f"base_theta0.{field}")
    v1._text(base["cold_frozen_policy_id"], "base_theta0.cold_frozen_policy_id")
    for field in (
        "cold_update_count",
        "cold_processed_credit_id_count",
        "cold_pending_decision_count",
        "terminal_source_v3_nonnoop_count_before_v2",
        "terminal_source_v4_public_reset_state_nonnoop_count_before_v2",
    ):
        if v1._integer(base[field], f"base_theta0.{field}") != 0:
            raise ValueError(f"base_theta0.{field} must remain zero")

    expected_schedule_values = {
        "root_count": 8,
        "decision_per_root": 24,
        "entry_count": 192,
        "owner_recommendation_role_count": 96,
        "neutral_noop_role_count": 96,
        "owner_recommendation_per_root": 12,
        "neutral_noop_per_root": 12,
    }
    for field, expected in expected_schedule_values.items():
        if v1._integer(schedule[field], f"forced_schedule.{field}") != expected:
            raise ValueError(f"forced_schedule.{field} drifted")
    if schedule["global_order"] != "public_subject_array_then_decision_0_to_23":
        raise ValueError("theta0 v2 schedule order drifted")
    if schedule["global_sequence_index"] != (
        "root_index_times_24_plus_decision_index"
    ):
        raise ValueError("theta0 v2 schedule sequence rule drifted")
    if schedule["role_formula"] != (
        "owner_recommendation_iff_((root_index_floor_div_2)+"
        "(decision_index_floor_div_2))_mod_2_equals_0_else_neutral_noop"
    ):
        raise ValueError("theta0 v2 schedule role formula drifted")
    v1._text(
        schedule["expected_schedule_artifact_id"],
        "forced_schedule.expected_schedule_artifact_id",
    )
    for field in (
        "role_frozen_before_any_environment_construction",
        "role_frozen_before_forecast_and_outcome",
        "symbolic_role_balance_not_concrete_action_balance",
    ):
        if not v1._boolean(schedule[field], f"forced_schedule.{field}"):
            raise ValueError(f"forced_schedule.{field} must remain true")
    for field in (
        "role_depends_on_forecast",
        "role_depends_on_outcome",
        "role_depends_on_sealed_truth",
    ):
        if v1._boolean(schedule[field], f"forced_schedule.{field}"):
            raise ValueError(f"forced_schedule.{field} must remain false")

    expected_bootstrap_strings = {
        "forced_preaction_owner": (
            "prepare_relationship_product_forced_collection_preaction"
        ),
        "actual_action_owner": "temporal_delivery.active_abstract_action",
        "environment_owner_api": "ReactiveRelationshipEnvironment.settle",
        "settlement_owner": "settle_relationship_product_forced_collection",
        "learning_source": "social_prediction_error_to_credit_only",
        "credit_timestamp_formula": (
            "root_index_times_52_plus_5_plus_2_times_decision_index"
        ),
        "credit_batch_operator": (
            "frozen_probability_and_features_sequential_accumulation_with_clipping"
        ),
        "theta0_v2_source_batch_identity": (
            "relationship_action_gate_credit_batch_id"
        ),
    }
    for field, expected in expected_bootstrap_strings.items():
        if bootstrap[field] != expected:
            raise ValueError(f"bootstrap.{field} drifted")
    for field in (
        "owner_reset_each_root",
        "owner_state_sequential_within_root",
        "single_base_cold_policy_for_all_exposures",
        "credit_timestamp_strictly_increasing",
        "preaction_trace_fsync_before_environment_settle",
        "credit_batch_order_sensitive",
    ):
        if not v1._boolean(bootstrap[field], f"bootstrap.{field}"):
            raise ValueError(f"bootstrap.{field} must remain true")
    if bootstrap["credit_batch_dispositions"] != ["apply", "withhold"]:
        raise ValueError("theta0 v2 batch dispositions drifted")
    for field in (
        "role_frozen_before_any_environment_construction",
        "role_frozen_before_forecast_and_outcome",
    ):
        if not v1._boolean(schedule[field], f"forced_schedule.{field}"):
            raise ValueError(f"forced_schedule.{field} must remain true")
    if v1._boolean(bootstrap["online_gate_credit_apply"], "online apply"):
        raise ValueError("theta0 v2 cannot apply credit online")
    for field in (
        "threshold_or_bias_tuning",
        "early_checkpoint_selection",
        "seed_or_order_selection",
    ):
        if v1._boolean(bootstrap[field], f"bootstrap.{field}"):
            raise ValueError(f"bootstrap.{field} must remain false")
    if v1._integer(bootstrap["apply_atomic_commit_count"], "apply commit") != 1:
        raise ValueError("theta0 v2 APPLY must commit exactly once")
    if v1._integer(bootstrap["withhold_atomic_commit_count"], "withhold commit") != 0:
        raise ValueError("theta0 v2 WITHHOLD cannot commit")
    if not v1._boolean(
        bootstrap["apply_and_withhold_share_exact_batch_and_plan"],
        "shared batch and plan",
    ):
        raise ValueError("theta0 v2 APPLY/WITHHOLD must share one plan")

    expected_terminal_values = {
        "forced_exposure_count": 192,
        "credit_count": 192,
        "unique_credit_id_count": 192,
        "batch_update_count_delta": 192,
        "batch_atomic_commit_count": 1,
        "withhold_update_count_delta": 0,
        "post_batch_pending_decision_count": 0,
        "theta0_v2_cold_update_count": 0,
        "theta0_v2_cold_processed_credit_id_count": 0,
        "theta0_v2_cold_pending_decision_count": 0,
        "training_support_physical_nonnoop_min_count": 1,
        "forced_actual_physical_nonnoop_min_count": 1,
        "forced_actual_neutral_noop_min_count": 1,
    }
    for field, expected in expected_terminal_values.items():
        if v1._integer(terminal[field], f"terminal_gates.{field}") != expected:
            raise ValueError(f"terminal_gates.{field} drifted")
    for field in (
        "withhold_parameter_delta_all_zero",
        "withhold_post_checkpoint_equals_base_cold_checkpoint",
        "parameter_delta_nonzero_required",
        "terminal_parameter_finite_required",
        "terminal_parameter_nonzero_required",
        "terminal_parameter_cap_hit_forbidden",
        "scientific_retry_with_different_schedule_order_base_threshold_or_seed_forbidden",
    ):
        if not v1._boolean(terminal[field], f"terminal_gates.{field}"):
            raise ValueError(f"terminal_gates.{field} must remain true")
    for field in (
        "source_v4_public_opportunity_scan_in_this_protocol",
        "campaign_authorized_on_success",
    ):
        if v1._boolean(terminal[field], f"terminal_gates.{field}"):
            raise ValueError(f"terminal_gates.{field} must remain false")
    if terminal["all_noop_terminal"] != _FAIL_STATUS:
        raise ValueError("theta0 v2 failure terminal drifted")
    if terminal["other_terminal_gate_failure"] != _TERMINAL_GATE_FAIL_STATUS:
        raise ValueError("theta0 v2 terminal-gate failure status drifted")
    if terminal["successful_terminal"] != _SUCCESS_STATUS:
        raise ValueError("theta0 v2 success terminal drifted")
    if terminal["forced_actual_physical_nonnoop_definition"] != (
        "temporal_delivery_active_abstract_action_not_equal_neutral_noop"
    ):
        raise ValueError("theta0 v2 forced actual nonnoop definition drifted")
    if terminal["training_support_physical_nonnoop_definition"] != (
        "frozen_selected_action_id_not_equal_neutral_noop"
    ):
        raise ValueError("theta0 v2 training-support nonnoop definition drifted")

    for field in ("model_output_count", "cuda_execution_count"):
        if v1._integer(firewall[field], f"causal_firewall.{field}") != 0:
            raise ValueError(f"causal_firewall.{field} must remain zero")
    if firewall["environment_safe_projection_fields"] != [
        "environment_subject_id",
        "selected_action_id",
        "typed_outcome_id",
        "rendered_user_reaction_sha256",
        "environment_evidence_ref",
        "environment_version",
    ]:
        raise ValueError("theta0 v2 safe environment projection drifted")
    if firewall["hidden_fields_forbidden_from_reader_gate_credit"] != [
        "condition_id",
        "policy_id",
        "preferred_action_id",
        "environment_seed",
        "outcome_distribution",
        "deterministic_draw",
        "evaluation_score",
        "judge_feedback",
    ]:
        raise ValueError("theta0 v2 forbidden hidden fields drifted")
    for field in (
        "evaluation_or_judge_feedback_to_learning",
        "oracle_input_to_learning",
        "os_process_secrecy_claim",
        "disk_transaction_atomicity_claim",
    ):
        if v1._boolean(firewall[field], f"causal_firewall.{field}"):
            raise ValueError(f"causal_firewall.{field} must remain false")
    true_claims = {field for field, value in claims.items() if value is True}
    if true_claims != {
        "bootstrap_execution_authorized",
        "development_theta0_v2_may_be_materialized",
    }:
        raise ValueError("theta0 v2 claim ceiling drifted")
    if any(isinstance(value, bool) is False for value in claims.values()):
        raise ValueError("theta0 v2 claims must be booleans")
    v1._text(payload["claim_boundary"], "claim_boundary")


def _load_dependencies(
    *,
    source_v3_admission_root: pathlib.Path,
    preflight_root: pathlib.Path,
    reader_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    base_theta0_root: pathlib.Path,
) -> _Dependencies:
    protocol = load_relationship_product_horizon_theta0_v2_bootstrap_protocol()
    inherited = v1._load_dependencies(
        source_v3_admission_root=source_v3_admission_root,
        preflight_root=preflight_root,
        reader_root=reader_root,
        source_v4_admission_root=source_v4_admission_root,
    )
    lineage = v1._mapping(protocol.payload["inherited_lineage"], "inherited lineage")
    if (
        inherited.protocol.protocol_id != lineage["theta0_v1_protocol_id"]
        or inherited.protocol.raw_sha256 != lineage["theta0_v1_protocol_raw_sha256"]
        or inherited.protocol.payload["source_v3_admission"]["artifact_id"]
        != lineage["source_v3_admission_artifact_id"]
        or inherited.protocol.payload["development_reader"]["artifact_id"]
        != lineage["development_reader_artifact_id"]
    ):
        raise ValueError("theta0 v2 inherited public lineage drifted")

    base_pin = v1._mapping(protocol.payload["base_theta0"], "base theta0 pin")
    manifest_raw = v1._require_raw_sha(
        base_theta0_root / "manifest.json",
        lineage["theta0_v1_manifest_raw_sha256"],
        "theta0 v1 manifest",
    )
    manifest = v1._parse_json_bytes(manifest_raw, source="theta0 v1 manifest")
    if (
        manifest["artifact_id"] != lineage["theta0_v1_calibration_artifact_id"]
        or manifest["protocol_id"] != lineage["theta0_v1_protocol_id"]
        or manifest["theta0_artifact_id"] != base_pin["artifact_id"]
        or manifest["final_checkpoint_content_sha256"]
        != base_pin["source_checkpoint_content_sha256"]
        or manifest["status"] != "development_theta0_materialized_effect_not_tested"
    ):
        raise ValueError("theta0 v1 manifest lineage drifted")
    theta_raw = v1._require_raw_sha(
        base_theta0_root / "theta0_artifact.json",
        base_pin["artifact_raw_sha256"],
        "theta0 v1 artifact",
    )
    theta = RelationshipActionGateTheta0Artifact.from_payload(
        v1._parse_json_bytes(theta_raw, source="theta0 v1 artifact")
    )
    if (
        theta.artifact_id != base_pin["artifact_id"]
        or theta.source_checkpoint_content_sha256
        != base_pin["source_checkpoint_content_sha256"]
    ):
        raise ValueError("theta0 v1 artifact pin drifted")
    gate = RelationshipActionGate.from_theta0(theta)
    cold = gate.validate_frozen_theta0()
    policy = gate.freeze_for_evaluation()
    if (
        cold.content_sha256 != base_pin["cold_checkpoint_content_sha256"]
        or policy.policy_id != base_pin["cold_frozen_policy_id"]
        or cold.update_count != 0
        or cold.processed_credit_ids
        or cold.pending_decisions
    ):
        raise ValueError("theta0 v1 cold replay drifted")
    return _Dependencies(protocol=protocol, inherited=inherited, base_theta0=theta)


def _forced_role(
    *,
    root_index: int,
    decision_index: int,
) -> RelationshipProductForcedActionRole:
    return (
        RelationshipProductForcedActionRole.OWNER_RECOMMENDATION
        if ((root_index // 2) + (decision_index // 2)) % 2 == 0
        else RelationshipProductForcedActionRole.NEUTRAL_NOOP
    )


def _build_forced_schedule(
    dependencies: _Dependencies,
) -> RelationshipProductForcedCollectionScheduleArtifact:
    entries = tuple(
        RelationshipProductForcedCollectionScheduleEntry(
            decision_id=decision.decision_id,
            sequence_index=root_index * 24 + decision.decision_index,
            forced_action_role=_forced_role(
                root_index=root_index,
                decision_index=decision.decision_index,
            ),
        )
        for root_index, subject in enumerate(dependencies.inherited.public_view.subjects)
        for decision in subject.decision_sessions
    )
    schedule = RelationshipProductForcedCollectionScheduleArtifact(entries=entries)
    schedule_pin = v1._mapping(
        dependencies.protocol.payload["forced_schedule"],
        "forced schedule pin",
    )
    counts = Counter(item.forced_action_role.value for item in entries)
    if (
        schedule.artifact_id != schedule_pin["expected_schedule_artifact_id"]
        or counts[RelationshipProductForcedActionRole.OWNER_RECOMMENDATION.value] != 96
        or counts[RelationshipProductForcedActionRole.NEUTRAL_NOOP.value] != 96
    ):
        raise ValueError("theta0 v2 forced schedule identity or balance drifted")
    for root_index in range(8):
        root_entries = entries[root_index * 24 : (root_index + 1) * 24]
        root_counts = Counter(item.forced_action_role.value for item in root_entries)
        if set(root_counts.values()) != {12}:
            raise ValueError("theta0 v2 per-root schedule balance drifted")
    for decision_index in range(24):
        column = entries[decision_index::24]
        column_counts = Counter(item.forced_action_role.value for item in column)
        if set(column_counts.values()) != {4}:
            raise ValueError("theta0 v2 decision-position schedule balance drifted")
    return schedule


def _pulse_authorization(
    *,
    dependencies: _Dependencies,
    frozen_policy: object,
) -> RelationshipProductFrozenPulseAuthorization:
    pulse = RelationshipProductPulseAuthorization(
        authorization_id=(
            "relationship-product-horizon-theta0-v2-bootstrap:"
            f"{dependencies.protocol.protocol_id}"
        ),
        allowed_policy_artifact_id=frozen_policy.artifact.artifact_id,
        allowed_policy_artifact_version=frozen_policy.artifact.artifact_version,
    )
    return RelationshipProductFrozenPulseAuthorization(
        pulse_authorization=pulse,
        allowed_frozen_policy_id=frozen_policy.policy_id,
        allowed_checkpoint_content_sha256=frozen_policy.checkpoint.content_sha256,
    )


def _request(*, subject: object, decision: object) -> RelationshipProductPreActionRequest:
    action_turn = 4 + decision.decision_index * 2
    return RelationshipProductPreActionRequest(
        session_id=decision.session_id,
        forecast_request=PreferenceActionForecastRequest(
            decision_id=decision.decision_id,
            interlocutor_id=_INTERLOCUTOR_ID,
            current_observation=decision.current_input,
            observation_ref=(
                f"public-decision:{sha256_json(decision.to_sut_payload())}"
            ),
            candidate_action_ids=tuple(action.value for action in RELATIONSHIP_ACTIONS),
            outcome_ids=tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES),
            turn_index=action_turn,
            session_scope=subject.subject_scope,
        ),
        outcome_turn_index=action_turn + 1,
    )


def _index_public_join(
    public_join: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    rows = v1._list(public_join["rows"], "public join rows")
    join_by_session: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(rows):
        row = v1._mapping(item, f"public join rows[{index}]")
        session_id = v1._text(
            row["session_id"],
            f"public join rows[{index}].session_id",
        )
        if session_id in join_by_session:
            raise ValueError("theta0 v2 public join session identity is not unique")
        join_by_session[session_id] = row
    if len(join_by_session) != 224:
        raise ValueError("theta0 v2 public join must contain 224 unique sessions")
    return join_by_session


def _create_candidate_theta0(
    *,
    post_checkpoint: RelationshipActionGateCheckpoint,
    base_theta0: RelationshipActionGateTheta0Artifact,
    batch_id: str,
) -> RelationshipActionGateTheta0Artifact | None:
    parameters = (*post_checkpoint.weights, post_checkpoint.bias)
    if not all(math.isfinite(value) for value in parameters):
        return None
    if not any(value != 0.0 for value in parameters):
        return None
    candidate = RelationshipActionGateTheta0Artifact.create(
        source_checkpoint=post_checkpoint,
        learning_rate=base_theta0.learning_rate,
        max_abs_parameter=base_theta0.max_abs_parameter,
        source_batch_artifact_id=batch_id,
    )
    candidate.validate_source_checkpoint(post_checkpoint)
    return candidate


async def _run_bootstrap(
    *,
    dependencies: _Dependencies,
    public_join: Mapping[str, object],
    schedule: RelationshipProductForcedCollectionScheduleArtifact,
    sink: v1._TraceSink,
) -> _BootstrapReplay:
    join_by_session = _index_public_join(public_join)
    base_gate = RelationshipActionGate.from_theta0(dependencies.base_theta0)
    base_checkpoint = base_gate.validate_frozen_theta0()
    frozen_policy = base_gate.freeze_for_evaluation()
    frozen_authorization = _pulse_authorization(
        dependencies=dependencies,
        frozen_policy=frozen_policy,
    )
    sink.append(
        {
            "schema_version": THETA0_V2_BOOTSTRAP_TRACE_SCHEMA_VERSION,
            "record_type": "header",
            "protocol_id": dependencies.protocol.protocol_id,
            "theta0_v1_protocol_id": dependencies.inherited.protocol.protocol_id,
            "public_join_artifact_id": public_join["artifact_id"],
            "base_theta0_artifact_id": dependencies.base_theta0.artifact_id,
            "base_cold_checkpoint_content_sha256": base_checkpoint.content_sha256,
            "base_frozen_policy_id": frozen_policy.policy_id,
            "forced_schedule_artifact_id": schedule.artifact_id,
            "schedule_persisted_before_environment": True,
            "online_gate_credit_apply": False,
            "environment_scope_created": False,
            "model_output_count": 0,
            "cuda_execution_count": 0,
        }
    )
    environment_scope: v1._EnvironmentScope | None = None
    exposures = []
    credits = []
    forced_action_counts: Counter[str] = Counter()
    previous_credit_timestamp = -1

    for root_index, subject in enumerate(dependencies.inherited.public_view.subjects):
        owner_persistence = None
        sink.append(
            {
                "schema_version": THETA0_V2_BOOTSTRAP_TRACE_SCHEMA_VERSION,
                "record_type": "root_start",
                "root_sequence_index": root_index,
                "subject_scope": subject.subject_scope,
                "world_clone_id": subject.world_clone_id,
                "owner_reset": True,
                "base_checkpoint_content_sha256": base_checkpoint.content_sha256,
                "base_update_count": 0,
                "base_pending_count": 0,
            }
        )
        for onboarding in subject.onboarding_sessions:
            join_row = join_by_session[onboarding.session_id]
            appended = await append_relationship_product_onboarding(
                owner_persistence_snapshot=owner_persistence,
                onboarding=RelationshipProductOnboardingInput(
                    session_id=onboarding.session_id,
                    session_index=onboarding.session_index,
                    turn_index=onboarding.session_index,
                    public_observation=onboarding.user_utterance,
                    action_id=onboarding.assistant_action_id,
                    observed_outcome_id=onboarding.observed_outcome_id,
                    reaction_summary=onboarding.rendered_user_reaction,
                    evidence_ref=(
                        "public-onboarding:"
                        f"{sha256_json(onboarding.to_sut_payload())}"
                    ),
                ),
            )
            owner_persistence = appended.owner_persistence_snapshot
            sink.append(
                {
                    "schema_version": THETA0_V2_BOOTSTRAP_TRACE_SCHEMA_VERSION,
                    "record_type": "onboarding",
                    "root_sequence_index": root_index,
                    "session_index": onboarding.session_index,
                    "session_id": onboarding.session_id,
                    "join_row_id": join_row["join_row_id"],
                    "owner_persistence_sha256": (
                        social_record_store_persistence_sha256(owner_persistence)
                    ),
                }
            )
        if owner_persistence is None:
            raise RuntimeError("theta0 v2 root has no onboarding state")

        for decision in subject.decision_sessions:
            sequence = root_index * 24 + decision.decision_index
            action_turn = 4 + decision.decision_index * 2
            credit_timestamp = root_index * 52 + 5 + 2 * decision.decision_index
            if credit_timestamp <= previous_credit_timestamp:
                raise RuntimeError("theta0 v2 credit timestamps are not chronological")
            previous_credit_timestamp = credit_timestamp
            join_row = join_by_session[decision.session_id]
            owner_before_sha = social_record_store_persistence_sha256(
                owner_persistence
            )
            authorization = RelationshipProductForcedCollectionAuthorization(
                frozen_pulse_authorization=frozen_authorization,
                schedule_artifact=schedule,
                decision_id=decision.decision_id,
            )
            preaction = await prepare_relationship_product_forced_collection_preaction(
                request=_request(subject=subject, decision=decision),
                owner_persistence_snapshot=owner_persistence,
                forecast_runtime=dependencies.inherited.forecast_runtime,
                frozen_policy=frozen_policy,
                authorization=authorization,
                substrate_snapshot=v1._placeholder_substrate(),
            )
            if preaction.frozen_policy.checkpoint != base_checkpoint:
                raise RuntimeError("theta0 v2 forced preaction changed cold checkpoint")
            if preaction.forced_exposure.sequence_index != sequence:
                raise RuntimeError("theta0 v2 forced exposure sequence drifted")
            delivered = preaction.delivered_action_id
            forced_action_counts[delivered] += 1
            temporal = preaction.execution_receipt.temporal_delivery
            sink.append(
                {
                    "schema_version": THETA0_V2_BOOTSTRAP_TRACE_SCHEMA_VERSION,
                    "record_type": "preaction",
                    "global_sequence_index": sequence,
                    "root_sequence_index": root_index,
                    "decision_index": decision.decision_index,
                    "session_id": decision.session_id,
                    "decision_id": decision.decision_id,
                    "join_row_id": join_row["join_row_id"],
                    "schedule_entry_id": authorization.schedule_entry_id,
                    "forced_action_role": authorization.forced_action_role.value,
                    "owner_prestate_sha256": owner_before_sha,
                    "owner_preaction_persistence_sha256": (
                        social_record_store_persistence_sha256(
                            preaction.owner_persistence_snapshot
                        )
                    ),
                    "forecast_sha256": sha256_json(
                        preference_action_forecast_to_payload(preaction.forecast)
                    ),
                    "condition_label": (
                        preaction.forecast.condition_readout.condition_label
                    ),
                    "recommended_action_id": (
                        preaction.forecast.recommended_action_id
                    ),
                    "base_gate_action": (
                        preaction.frozen_decision.decision.gate_action.value
                    ),
                    "base_gate_selected_action_id": (
                        preaction.frozen_decision.decision.selected_action_id
                    ),
                    "base_steer_probability_hex": (
                        preaction.frozen_decision.decision.steer_probability.hex()
                    ),
                    "forced_action_id": preaction.forced_exposure.forced_action_id,
                    "delivered_action_id": delivered,
                    "gate_would_noop": preaction.execution_receipt.gate_would_noop,
                    "forced_override": preaction.execution_receipt.forced_override,
                    "temporal_action_advisory_status": (
                        temporal.action_advisory_status.value
                    ),
                    "temporal_timestamp_excluded_from_stable_projection": True,
                    "environment_settle_called": False,
                }
            )

            if environment_scope is None:
                environment_scope = v1._EnvironmentScope(
                    dependencies=dependencies.inherited
                )
            environment_outcome = environment_scope.settle(
                public_subject=subject,
                public_session=decision,
                selected_action_id=delivered,
            )
            settlement_input = replace(
                v1._settlement_input(
                    subject_scope=subject.subject_scope,
                    decision=decision,
                    forecast_id=preaction.forecast.forecast_id,
                    selected_action_id=delivered,
                    environment_outcome=environment_outcome,
                    action_turn=action_turn,
                    credit_timestamp=credit_timestamp,
                ),
                apply_credit_to_gate=False,
            )
            settled = await settle_relationship_product_forced_collection(
                preaction=preaction,
                settlement_input=settlement_input,
            )
            if settled.credit.prediction_id != preaction.forecast.forecast_id:
                raise RuntimeError("theta0 v2 credit forecast lineage drifted")
            if settled.credit.abstract_action_id != delivered:
                raise RuntimeError("theta0 v2 credit forced-action lineage drifted")
            if settled.credit.timestamp_ms != credit_timestamp:
                raise RuntimeError("theta0 v2 credit timestamp drifted")
            if settled.preaction.frozen_policy.checkpoint != base_checkpoint:
                raise RuntimeError("theta0 v2 settlement changed cold checkpoint")
            exposures.append(preaction.forced_exposure)
            credits.append(settled.credit)
            owner_persistence = settled.owner_persistence_snapshot
            sink.append(
                {
                    "schema_version": THETA0_V2_BOOTSTRAP_TRACE_SCHEMA_VERSION,
                    "record_type": "postaction",
                    "global_sequence_index": sequence,
                    "root_sequence_index": root_index,
                    "decision_index": decision.decision_index,
                    "session_id": decision.session_id,
                    "decision_id": decision.decision_id,
                    "environment": {
                        "environment_subject_id": (
                            environment_outcome.environment_subject_id
                        ),
                        "selected_action_id": delivered,
                        "typed_outcome_id": environment_outcome.typed_outcome_id,
                        "rendered_user_reaction_sha256": v1._sha256_text(
                            environment_outcome.rendered_user_reaction
                        ),
                        "environment_evidence_ref": (
                            environment_outcome.environment_evidence_ref
                        ),
                        "environment_version": environment_outcome.environment_version,
                    },
                    "settlement_id": settled.settlement.settlement_id,
                    "social_prediction_error": v1._social_pe_payload(
                        settled.social_prediction_error_snapshot.value
                    ),
                    "credit": v1._credit_payload(settled.credit),
                    "credit_applied_online": False,
                    "base_checkpoint_content_sha256": base_checkpoint.content_sha256,
                    "base_update_count": 0,
                    "base_pending_count": 0,
                    "owner_poststate_sha256": (
                        social_record_store_persistence_sha256(owner_persistence)
                    ),
                    "evaluation_or_judge_feedback_received": False,
                }
            )

    if environment_scope is None:
        raise RuntimeError("theta0 v2 bootstrap produced no environment settlements")
    batch = RelationshipActionGateCreditBatch(
        exposures=tuple(exposures),
        credits=tuple(credits),
    )
    apply_gate = RelationshipActionGate.from_theta0(dependencies.base_theta0)
    apply_pre = apply_gate.export_checkpoint()
    apply_plan = apply_gate.plan_credit_batch(batch)
    if apply_gate.export_checkpoint() != apply_pre:
        raise RuntimeError("theta0 v2 pure batch plan mutated APPLY gate")
    apply_receipt = apply_gate.commit_credit_batch(
        apply_plan,
        disposition=RelationshipActionGateBatchDisposition.APPLY,
    )
    post_checkpoint = apply_gate.export_checkpoint()

    withhold_gate = RelationshipActionGate.from_theta0(dependencies.base_theta0)
    withhold_plan = withhold_gate.plan_credit_batch(batch)
    if withhold_plan != apply_plan:
        raise RuntimeError("theta0 v2 APPLY/WITHHOLD did not share one exact plan")
    withhold_receipt = withhold_gate.commit_credit_batch(
        withhold_plan,
        disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
    )
    if withhold_gate.export_checkpoint() != base_checkpoint:
        raise RuntimeError("theta0 v2 WITHHOLD changed the base checkpoint")
    replayed = RelationshipActionGate.from_applied_credit_batch(
        dependencies.base_theta0,
        batch=batch,
        receipt=apply_receipt,
    )
    if replayed.export_checkpoint() != post_checkpoint:
        raise RuntimeError("theta0 v2 batch owner replay drifted")
    if (
        len(batch.exposures) != 192
        or len(batch.credits) != 192
        or len({item.record_id for item in batch.credits}) != 192
        or apply_receipt.update_count_delta != 192
        or apply_receipt.atomic_commit_count != 1
        or withhold_receipt.update_count_delta != 0
        or withhold_receipt.atomic_commit_count != 0
        or post_checkpoint.pending_decisions
    ):
        raise RuntimeError("theta0 v2 terminal batch counts did not close")
    parameter_delta_nonzero = any(
        value != 0.0
        for value in (*apply_receipt.weight_delta, apply_receipt.bias_delta)
    )
    cap = dependencies.base_theta0.max_abs_parameter
    terminal_parameters = (*post_checkpoint.weights, post_checkpoint.bias)
    terminal_parameter_finite = all(
        math.isfinite(value) for value in terminal_parameters
    )
    terminal_parameter_nonzero = any(
        value != 0.0 for value in terminal_parameters
    )
    parameter_cap_hit = any(
        abs(value) >= cap for value in terminal_parameters
    )

    candidate_theta0 = _create_candidate_theta0(
        post_checkpoint=post_checkpoint,
        base_theta0=dependencies.base_theta0,
        batch_id=batch.batch_id,
    )
    if candidate_theta0 is None:
        support_counts: Counter[str] = Counter()
        if terminal_parameter_finite and not terminal_parameter_nonzero:
            support_counts[RelationshipAction.NEUTRAL_NOOP.value] = len(
                batch.exposures
            )
    else:
        candidate_gate = RelationshipActionGate.from_theta0(candidate_theta0)
        candidate_cold = candidate_gate.validate_frozen_theta0()
        if (
            candidate_cold.weights != post_checkpoint.weights
            or candidate_cold.bias != post_checkpoint.bias
            or candidate_cold.update_count != 0
            or candidate_cold.processed_credit_ids
            or candidate_cold.pending_decisions
        ):
            raise RuntimeError("theta0 v2 cold replay drifted")
        support_counts = Counter(
            candidate_gate.freeze_for_evaluation()
            .decide(exposure.forecast)
            .decision.selected_action_id
            for exposure in batch.exposures
        )
    support_nonnoop = sum(
        count
        for action_id, count in support_counts.items()
        if action_id != RelationshipAction.NEUTRAL_NOOP.value
    )
    forced_nonnoop = sum(
        count
        for action_id, count in forced_action_counts.items()
        if action_id != RelationshipAction.NEUTRAL_NOOP.value
    )
    forced_neutral = forced_action_counts[RelationshipAction.NEUTRAL_NOOP.value]
    failure_reasons = tuple(
        reason
        for failed, reason in (
            (forced_nonnoop < 1, "forced_actual_physical_nonnoop_below_one"),
            (forced_neutral < 1, "forced_actual_neutral_noop_below_one"),
            (not parameter_delta_nonzero, "parameter_delta_all_zero"),
            (not terminal_parameter_finite, "terminal_parameter_nonfinite"),
            (not terminal_parameter_nonzero, "terminal_parameter_all_zero"),
            (parameter_cap_hit, "terminal_parameter_touched_cap"),
            (support_nonnoop < 1, "training_support_physical_nonnoop_below_one"),
        )
        if failed
    )
    success = not failure_reasons
    if success and candidate_theta0 is None:
        raise RuntimeError("theta0 v2 success did not produce a candidate artifact")
    published = candidate_theta0 if success else None
    if success:
        terminal_status = _SUCCESS_STATUS
    elif failure_reasons == ("training_support_physical_nonnoop_below_one",):
        terminal_status = _FAIL_STATUS
    else:
        terminal_status = _TERMINAL_GATE_FAIL_STATUS
    scheduled_counts = Counter(item.forced_action_role.value for item in schedule.entries)
    root_mapping = environment_scope.root_mapping
    sink.append(
        {
            "schema_version": THETA0_V2_BOOTSTRAP_TRACE_SCHEMA_VERSION,
            "record_type": "terminal",
            "forced_schedule_artifact_id": schedule.artifact_id,
            "scheduled_role_counts": dict(sorted(scheduled_counts.items())),
            "forced_action_counts": dict(sorted(forced_action_counts.items())),
            "forced_physical_nonnoop_count": forced_nonnoop,
            "credit_batch_id": batch.batch_id,
            "credit_count": len(batch.credits),
            "unique_credit_count": len({item.record_id for item in batch.credits}),
            "apply_plan_id": apply_plan.plan_id,
            "apply_receipt": apply_receipt.to_payload(),
            "withhold_receipt": withhold_receipt.to_payload(),
            "post_checkpoint": post_checkpoint.to_payload(),
            "candidate_theta0_artifact_id": (
                None if candidate_theta0 is None else candidate_theta0.artifact_id
            ),
            "published_theta0_artifact_id": (
                None if published is None else published.artifact_id
            ),
            "training_support_selected_action_counts": dict(
                sorted(support_counts.items())
            ),
            "training_support_physical_nonnoop_count": support_nonnoop,
            "parameter_delta_nonzero": parameter_delta_nonzero,
            "terminal_parameter_finite": terminal_parameter_finite,
            "terminal_parameter_nonzero": terminal_parameter_nonzero,
            "terminal_parameter_cap_hit": parameter_cap_hit,
            "terminal_failure_reasons": list(failure_reasons),
            "root_mapping": list(root_mapping),
            "root_mapping_sha256": sha256_json(list(root_mapping)),
            "terminal_status": terminal_status,
            "model_output_count": 0,
            "cuda_execution_count": 0,
        }
    )
    return _BootstrapReplay(
        batch=batch,
        apply_receipt=apply_receipt,
        withhold_receipt=withhold_receipt,
        post_checkpoint=post_checkpoint,
        candidate_theta0=candidate_theta0,
        published_theta0=published,
        root_mapping=root_mapping,
        scheduled_role_counts=dict(sorted(scheduled_counts.items())),
        forced_action_counts=dict(sorted(forced_action_counts.items())),
        training_support_selected_action_counts=dict(sorted(support_counts.items())),
        terminal_failure_reasons=failure_reasons,
        terminal_status=terminal_status,
    )


def materialize_relationship_product_horizon_theta0_v2_bootstrap(
    *,
    source_v3_admission_root: pathlib.Path,
    preflight_root: pathlib.Path,
    reader_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    base_theta0_root: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    commit = v1._git_commit(implementation_git_commit)
    root = pathlib.Path(output_dir)
    if root.exists():
        raise FileExistsError(f"theta0 v2 bootstrap root is create-only: {root}")
    dependencies = _load_dependencies(
        source_v3_admission_root=pathlib.Path(source_v3_admission_root),
        preflight_root=pathlib.Path(preflight_root),
        reader_root=pathlib.Path(reader_root),
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        base_theta0_root=pathlib.Path(base_theta0_root),
    )
    public_join = v1._build_public_join(dependencies.inherited)
    schedule = _build_forced_schedule(dependencies)
    root.mkdir(parents=True, exist_ok=False)
    v1._write_create_only(root / "protocol.json", dependencies.protocol.raw_bytes)
    v1._write_create_only(root / "schedule.json", v1._canonical_bytes(schedule.to_payload()))
    sink = v1._FsyncTraceSink(root / _TRACE_FILENAME)
    try:
        replay = asyncio.run(
            _run_bootstrap(
                dependencies=dependencies,
                public_join=public_join,
                schedule=schedule,
                sink=sink,
            )
        )
    finally:
        sink.close()
    _write_replay_files(root=root, replay=replay)
    manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        public_join=public_join,
        schedule=schedule,
        replay=replay,
        implementation_git_commit=commit,
    )
    v1._write_create_only(root / "manifest.json", v1._canonical_bytes(manifest))
    return manifest


def _write_replay_files(*, root: pathlib.Path, replay: _BootstrapReplay) -> None:
    payloads = {
        "credit_batch.json": replay.batch.to_payload(),
        "apply_receipt.json": replay.apply_receipt.to_payload(),
        "withhold_receipt.json": replay.withhold_receipt.to_payload(),
    }
    if replay.published_theta0 is not None:
        payloads["theta0_artifact.json"] = replay.published_theta0.to_payload()
    for relative, payload in payloads.items():
        v1._write_create_only(root / relative, v1._canonical_bytes(payload))


def validate_relationship_product_horizon_theta0_v2_bootstrap(
    *,
    source_v3_admission_root: pathlib.Path,
    preflight_root: pathlib.Path,
    reader_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    base_theta0_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    external_protocol = v1._digest(expected_protocol_id, "expected_protocol_id")
    external_artifact = v1._digest(expected_artifact_id, "expected_artifact_id")
    root = pathlib.Path(output_dir)
    manifest_raw = v1._read_regular(root / "manifest.json")
    manifest = v1._parse_json_bytes(manifest_raw, source="theta0 v2 manifest")
    if manifest_raw != v1._canonical_bytes(manifest):
        raise ValueError("theta0 v2 manifest must use canonical bytes")
    if manifest["protocol_id"] != external_protocol:
        raise ValueError("theta0 v2 external expected protocol identity drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("theta0 v2 external expected artifact identity drifted")
    dependencies = _load_dependencies(
        source_v3_admission_root=pathlib.Path(source_v3_admission_root),
        preflight_root=pathlib.Path(preflight_root),
        reader_root=pathlib.Path(reader_root),
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        base_theta0_root=pathlib.Path(base_theta0_root),
    )
    if dependencies.protocol.protocol_id != external_protocol:
        raise ValueError("theta0 v2 packaged protocol identity drifted")
    if v1._read_regular(root / "protocol.json") != dependencies.protocol.raw_bytes:
        raise ValueError("theta0 v2 persisted protocol bytes drifted")
    public_join = v1._build_public_join(dependencies.inherited)
    schedule = _build_forced_schedule(dependencies)
    if v1._read_regular(root / "schedule.json") != v1._canonical_bytes(
        schedule.to_payload()
    ):
        raise ValueError("theta0 v2 schedule bytes drifted")
    sink = v1._MemoryTraceSink()
    replay = asyncio.run(
        _run_bootstrap(
            dependencies=dependencies,
            public_join=public_join,
            schedule=schedule,
            sink=sink,
        )
    )
    if v1._read_regular(root / _TRACE_FILENAME) != sink.raw_bytes:
        raise ValueError("theta0 v2 stable trace bytes drifted")
    expected_payloads = {
        "credit_batch.json": replay.batch.to_payload(),
        "apply_receipt.json": replay.apply_receipt.to_payload(),
        "withhold_receipt.json": replay.withhold_receipt.to_payload(),
    }
    if replay.published_theta0 is not None:
        expected_payloads["theta0_artifact.json"] = replay.published_theta0.to_payload()
    for relative, payload in expected_payloads.items():
        raw = v1._read_regular(root / relative)
        if raw != v1._canonical_bytes(payload):
            raise ValueError(f"theta0 v2 {relative} bytes drifted")
    loaded_batch = RelationshipActionGateCreditBatch.from_payload(
        v1._parse_json_bytes(
            v1._read_regular(root / "credit_batch.json"),
            source="theta0 v2 credit batch",
        )
    )
    loaded_apply = RelationshipActionGateBatchReceipt.from_payload(
        v1._parse_json_bytes(
            v1._read_regular(root / "apply_receipt.json"),
            source="theta0 v2 apply receipt",
        )
    )
    loaded_withhold = RelationshipActionGateBatchReceipt.from_payload(
        v1._parse_json_bytes(
            v1._read_regular(root / "withhold_receipt.json"),
            source="theta0 v2 withhold receipt",
        )
    )
    if (
        loaded_batch != replay.batch
        or loaded_apply != replay.apply_receipt
        or loaded_withhold != replay.withhold_receipt
    ):
        raise ValueError("theta0 v2 owner strict replay payload drifted")
    if replay.published_theta0 is not None:
        loaded_theta0 = RelationshipActionGateTheta0Artifact.from_payload(
            v1._parse_json_bytes(
                v1._read_regular(root / "theta0_artifact.json"),
                source="theta0 v2 artifact",
            )
        )
        if loaded_theta0 != replay.published_theta0:
            raise ValueError("theta0 v2 strict theta artifact replay drifted")
    expected_files = (
        _SUCCESS_OUTPUT_FILES
        if replay.published_theta0 is not None
        else _BASE_OUTPUT_FILES
    )
    if v1._regular_file_inventory(root) != expected_files:
        raise ValueError("theta0 v2 output file inventory drifted")
    expected_manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        public_join=public_join,
        schedule=schedule,
        replay=replay,
        implementation_git_commit=v1._git_commit(
            manifest["implementation_git_commit"]
        ),
    )
    if manifest != expected_manifest or manifest_raw != v1._canonical_bytes(
        expected_manifest
    ):
        raise ValueError("theta0 v2 manifest content drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("theta0 v2 artifact identity drifted")
    return manifest


def _build_manifest(
    *,
    root: pathlib.Path,
    dependencies: _Dependencies,
    public_join: Mapping[str, object],
    schedule: RelationshipProductForcedCollectionScheduleArtifact,
    replay: _BootstrapReplay,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    paths = [
        "protocol.json",
        "schedule.json",
        _TRACE_FILENAME,
        "credit_batch.json",
        "apply_receipt.json",
        "withhold_receipt.json",
    ]
    if replay.published_theta0 is not None:
        paths.append("theta0_artifact.json")
    files = []
    for relative in paths:
        raw = v1._read_regular(root / relative)
        files.append(
            {
                "path": relative,
                "raw_bytes": len(raw),
                "raw_sha256": v1._sha256_bytes(raw),
            }
        )
    support_nonnoop = sum(
        count
        for action_id, count in replay.training_support_selected_action_counts.items()
        if action_id != RelationshipAction.NEUTRAL_NOOP.value
    )
    forced_nonnoop = sum(
        count
        for action_id, count in replay.forced_action_counts.items()
        if action_id != RelationshipAction.NEUTRAL_NOOP.value
    )
    claims = {
        "bootstrap_completed": True,
        "development_theta0_v2_materialized": replay.published_theta0 is not None,
        "source_v4_opportunity_established": False,
        "reader_qualified": False,
        "campaign_execution_authorized": False,
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
        "model_output_count": 0,
        "cuda_execution_count": 0,
    }
    core = {
        "schema_version": THETA0_V2_BOOTSTRAP_MANIFEST_SCHEMA_VERSION,
        "protocol_id": dependencies.protocol.protocol_id,
        "protocol_raw_sha256": dependencies.protocol.raw_sha256,
        "implementation_git_commit": implementation_git_commit,
        "theta0_v1_protocol_id": dependencies.inherited.protocol.protocol_id,
        "source_v3_admission_artifact_id": dependencies.protocol.payload[
            "inherited_lineage"
        ]["source_v3_admission_artifact_id"],
        "development_reader_artifact_id": dependencies.protocol.payload[
            "inherited_lineage"
        ]["development_reader_artifact_id"],
        "base_theta0_artifact_id": dependencies.base_theta0.artifact_id,
        "base_cold_checkpoint_content_sha256": (
            RelationshipActionGate.from_theta0(dependencies.base_theta0)
            .validate_frozen_theta0()
            .content_sha256
        ),
        "public_join_artifact_id": public_join["artifact_id"],
        "forced_schedule_artifact_id": schedule.artifact_id,
        "credit_batch_id": replay.batch.batch_id,
        "apply_receipt_id": replay.apply_receipt.receipt_id,
        "withhold_receipt_id": replay.withhold_receipt.receipt_id,
        "post_checkpoint_content_sha256": replay.post_checkpoint.content_sha256,
        "candidate_theta0_artifact_id": (
            None
            if replay.candidate_theta0 is None
            else replay.candidate_theta0.artifact_id
        ),
        "published_theta0_artifact_id": (
            None
            if replay.published_theta0 is None
            else replay.published_theta0.artifact_id
        ),
        "root_count": 8,
        "onboarding_count": 32,
        "forced_exposure_count": len(replay.batch.exposures),
        "credit_count": len(replay.batch.credits),
        "unique_credit_id_count": len(
            {item.record_id for item in replay.batch.credits}
        ),
        "scheduled_role_counts": replay.scheduled_role_counts,
        "forced_action_counts": replay.forced_action_counts,
        "forced_physical_nonnoop_count": forced_nonnoop,
        "training_support_selected_action_counts": (
            replay.training_support_selected_action_counts
        ),
        "training_support_physical_nonnoop_count": support_nonnoop,
        "terminal_failure_reasons": list(replay.terminal_failure_reasons),
        "terminal_parameter_finite": all(
            math.isfinite(value)
            for value in (*replay.post_checkpoint.weights, replay.post_checkpoint.bias)
        ),
        "terminal_parameter_nonzero": any(
            value != 0.0
            for value in (*replay.post_checkpoint.weights, replay.post_checkpoint.bias)
        ),
        "apply_update_count_delta": replay.apply_receipt.update_count_delta,
        "apply_atomic_commit_count": replay.apply_receipt.atomic_commit_count,
        "withhold_update_count_delta": replay.withhold_receipt.update_count_delta,
        "withhold_atomic_commit_count": replay.withhold_receipt.atomic_commit_count,
        "root_mapping_sha256": sha256_json(list(replay.root_mapping)),
        "preaction_trace_fsync_before_environment_settle": True,
        "temporal_timestamp_excluded_from_stable_projection": True,
        "challenge_label_file_read_count": 0,
        "group_split_file_read_count": 0,
        "files": files,
        "status": replay.terminal_status,
        "claims": claims,
        "claim_boundary": dependencies.protocol.payload["claim_boundary"],
    }
    return {"artifact_id": sha256_json(core), **core}


__all__ = [
    "THETA0_V2_BOOTSTRAP_MANIFEST_SCHEMA_VERSION",
    "THETA0_V2_BOOTSTRAP_PROTOCOL_SCHEMA_VERSION",
    "THETA0_V2_BOOTSTRAP_TRACE_SCHEMA_VERSION",
    "RelationshipProductHorizonTheta0V2BootstrapProtocol",
    "load_relationship_product_horizon_theta0_v2_bootstrap_protocol",
    "materialize_relationship_product_horizon_theta0_v2_bootstrap",
    "relationship_product_horizon_theta0_v2_bootstrap_protocol_path",
    "validate_relationship_product_horizon_theta0_v2_bootstrap",
]
