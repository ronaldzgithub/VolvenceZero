"""Root-local forced common-collection batches for Product Horizon.

This development-only workflow executes the eight matched-collection decisions
of every source-v4 root exactly once under a public, position-only forced-action
schedule.  It preserves the cold theta0 policy during collection, then builds
one eight-credit batch per root and proves the APPLY/WITHHOLD arm-initialization
transition.  It does not run the forty-decision evaluation or estimate effects.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field as dataclass_field, replace
from enum import Enum
import math
import pathlib
from typing import Mapping

from lifeform_domain_emogpt.lab.contracts import sha256_json
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    HorizonPublicDecisionSession,
    HorizonPublicRoot,
    RelationshipProductHorizonPublicView,
)
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductExecutorDisposition,
    RelationshipProductForcedActionRole,
    RelationshipProductForcedCollectionAuthorization,
    RelationshipProductForcedCollectionPreActionSnapshot,
    RelationshipProductForcedCollectionScheduleArtifact,
    RelationshipProductForcedCollectionScheduleEntry,
    RelationshipProductFrozenPulseAuthorization,
    RelationshipProductPulseAuthorization,
    prepare_relationship_product_forced_collection_preaction,
    settle_relationship_product_forced_collection,
)
from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateBatchDisposition,
    RelationshipActionGateBatchReceipt,
    RelationshipActionGateCreditBatch,
    RelationshipActionGateFrozenPolicy,
    RelationshipActionGateTheta0Artifact,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    FrozenLinearRelationshipConditionReaderArtifact,
    FrozenLinearRelationshipConditionReaderRuntime,
    FrozenLinearRelationshipPreferenceForecastRuntime,
)
from lifeform_evolution import (
    relationship_product_horizon_dynamic_collection_prefix as dynamic,
)
from lifeform_evolution import relationship_product_horizon_theta0_calibration as cal
from lifeform_evolution.relationship_lab_product_model_adapters import (
    PrecomputedPublicEmbeddingTable,
    PrecomputedPublicSemanticEmbedder,
)
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot
from volvence_zero.social import (
    SocialRecordStore,
    social_record_store_persistence_sha256,
)
from volvence_zero.social_cognition import preference_action_forecast_to_payload


FORCED_COMMON_BATCH_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-forced-common-batch-protocol.v1"
)
FORCED_SCHEDULE_INDEX_SCHEMA_VERSION = (
    "relationship-product-horizon-forced-schedule-index.v1"
)
FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION = (
    "relationship-product-horizon-forced-common-batch-trace.v1"
)
FORCED_COMMON_BATCH_ROOT_STATE_SCHEMA_VERSION = (
    "relationship-product-horizon-forced-common-batch-root-state.v1"
)
FORCED_COMMON_BATCH_TRANSITION_SCHEMA_VERSION = (
    "relationship-product-horizon-forced-common-batch-transition.v1"
)
FORCED_COMMON_BATCH_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-forced-common-batch-manifest.v1"
)
FORCED_CAMPAIGN_INPUT_LINEAGE_SCHEMA_VERSION = (
    "relationship-product-horizon-forced-campaign-input-lineage.v1"
)
FORCED_SAFE_FIRST_PROJECTION_SCHEMA_VERSION = (
    "relationship-product-horizon-forced-safe-first-preaction-projection.v1"
)
FORCED_SAFE_FIRST_PROJECTION_FIELDS = (
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
    "candidate_advisory_id",
    "cold_checkpoint_content_sha256",
    "cold_frozen_policy_id",
)

_PROTOCOL_FILENAME = "relationship_product_horizon_forced_common_batch_v1.json"
_SCHEDULE_FILENAME = "schedule_index.json"
_TRACE_FILENAME = "forced_common_collection_trace.jsonl"
_STATE_FILENAME = "root_owner_states.jsonl"
_TRANSITION_FILENAME = "root_batch_transitions.jsonl"
_OUTPUT_FILES = frozenset(
    {
        "protocol.json",
        _SCHEDULE_FILENAME,
        _TRACE_FILENAME,
        _STATE_FILENAME,
        _TRANSITION_FILENAME,
        "manifest.json",
    }
)
_ROLE_FORMULA = (
    "owner_recommendation_iff_((root_index_div_2)+"
    "(decision_index_div_2))_mod_2_eq_0_else_neutral_noop"
)
_CREDIT_TIMESTAMP_FORMULA = (
    "root_sequence_index_times_20_plus_5_plus_2_times_decision_index"
)
_SUCCESS_STATUS = (
    "development_forced_common_collection_batches_materialized_"
    "campaign_protocol_freeze_authorized_effect_not_tested"
)
_FAIL_STATUS = (
    "development_forced_common_collection_batch_gate_failed_"
    "campaign_blocked_effect_not_tested"
)
_DEGENERATE_STATUS = "arm_degeneracy_invalid_contrast_no_claim"


@dataclass(frozen=True)
class RelationshipProductHorizonForcedCommonBatchProtocol:
    payload: Mapping[str, object]
    raw_bytes: bytes
    protocol_id: str
    raw_sha256: str


class RelationshipProductHorizonCampaignArm(str, Enum):
    """The three frozen development-campaign treatment identities."""

    FULL = "full"
    FROZEN_THETA0 = "frozen_theta0"
    STRICT_NOOP = "strict_noop"


@dataclass(frozen=True)
class RelationshipProductHorizonCampaignLineageEntry:
    """One immutable identity required by the downstream preregistration."""

    name: str
    value: str


@dataclass(frozen=True)
class RelationshipProductHorizonCampaignArmInitialization:
    """One complete fresh arm start; no mutable state is shared across arms."""

    arm_id: RelationshipProductHorizonCampaignArm
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    starting_owner_persistence_sha256: str
    batch: RelationshipActionGateCreditBatch
    batch_receipt: RelationshipActionGateBatchReceipt
    frozen_policy: RelationshipActionGateFrozenPolicy
    forecast_runtime: FrozenLinearRelationshipPreferenceForecastRuntime
    executor_disposition: RelationshipProductExecutorDisposition


@dataclass(frozen=True)
class RelationshipProductHorizonForcedCampaignRootInput:
    """One verified root-local arm initialization without live shared state."""

    root_sequence_index: int
    public_root: HorizonPublicRoot
    schedule_artifact_id: str
    transition_raw_sha256: str
    common_terminal_owner_persistence_sha256: str
    batch: RelationshipActionGateCreditBatch
    apply_receipt: RelationshipActionGateBatchReceipt
    withhold_receipt: RelationshipActionGateBatchReceipt
    full_policy_id: str
    full_checkpoint_content_sha256: str
    cold_frozen_policy_id: str
    _owner_persistence_bytes: bytes = dataclass_field(repr=False, compare=False)
    _theta0: RelationshipActionGateTheta0Artifact = dataclass_field(
        repr=False,
        compare=False,
    )
    _cold_random_seed: str = dataclass_field(repr=False, compare=False)
    _embedding_table: PrecomputedPublicEmbeddingTable = dataclass_field(
        repr=False,
        compare=False,
    )
    _reader_artifact: FrozenLinearRelationshipConditionReaderArtifact = dataclass_field(
        repr=False,
        compare=False,
    )

    def _fresh_owner_persistence(self) -> OwnerPersistenceSnapshot:
        """Hydrate a fresh opaque owner envelope from canonical persisted bytes."""

        snapshot = _owner_snapshot_from_payload(
            cal._parse_json_bytes(
                self._owner_persistence_bytes,
                source=(
                    "forced campaign root "
                    f"{self.root_sequence_index} terminal owner"
                ),
            )
        )
        if (
            social_record_store_persistence_sha256(snapshot)
            != self.common_terminal_owner_persistence_sha256
        ):
            raise ValueError("forced campaign terminal owner identity drifted")
        return snapshot

    def _fresh_full_policy(self) -> RelationshipActionGateFrozenPolicy:
        """Replay the exact APPLY receipt into a fresh gate and freeze it."""

        policy = RelationshipActionGate.from_applied_credit_batch(
            self._theta0,
            batch=self.batch,
            receipt=self.apply_receipt,
            random_seed=self._cold_random_seed,
        ).freeze_for_evaluation()
        if (
            policy.policy_id != self.full_policy_id
            or policy.checkpoint.content_sha256
            != self.full_checkpoint_content_sha256
        ):
            raise ValueError("forced campaign full policy replay drifted")
        return policy

    def _fresh_cold_policy(self) -> RelationshipActionGateFrozenPolicy:
        """Build one fresh WITHHOLD/cold policy for frozen or strict arms."""

        gate = RelationshipActionGate.from_theta0(
            self._theta0,
            random_seed=self._cold_random_seed,
        )
        plan = gate.plan_credit_batch(self.batch)
        receipt = gate.commit_credit_batch(
            plan,
            disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
        )
        if receipt != self.withhold_receipt:
            raise ValueError("forced campaign WITHHOLD receipt replay drifted")
        policy = gate.freeze_for_evaluation()
        if policy.policy_id != self.cold_frozen_policy_id:
            raise ValueError("forced campaign cold policy replay drifted")
        return policy

    def _fresh_forecast_runtime(
        self,
    ) -> FrozenLinearRelationshipPreferenceForecastRuntime:
        reader = FrozenLinearRelationshipConditionReaderRuntime(
            artifact=self._reader_artifact,
            embedder=PrecomputedPublicSemanticEmbedder(self._embedding_table),
        )
        return FrozenLinearRelationshipPreferenceForecastRuntime(reader=reader)

    def fresh_arm_initializations(
        self,
    ) -> tuple[RelationshipProductHorizonCampaignArmInitialization, ...]:
        """Publish the exact arm mapping with three fresh runtime states."""

        full = RelationshipProductHorizonCampaignArmInitialization(
            arm_id=RelationshipProductHorizonCampaignArm.FULL,
            owner_persistence_snapshot=self._fresh_owner_persistence(),
            starting_owner_persistence_sha256=(
                self.common_terminal_owner_persistence_sha256
            ),
            batch=self.batch,
            batch_receipt=self.apply_receipt,
            frozen_policy=self._fresh_full_policy(),
            forecast_runtime=self._fresh_forecast_runtime(),
            executor_disposition=(
                RelationshipProductExecutorDisposition.APPLY_CANDIDATE
            ),
        )
        frozen = RelationshipProductHorizonCampaignArmInitialization(
            arm_id=RelationshipProductHorizonCampaignArm.FROZEN_THETA0,
            owner_persistence_snapshot=self._fresh_owner_persistence(),
            starting_owner_persistence_sha256=(
                self.common_terminal_owner_persistence_sha256
            ),
            batch=self.batch,
            batch_receipt=self.withhold_receipt,
            frozen_policy=self._fresh_cold_policy(),
            forecast_runtime=self._fresh_forecast_runtime(),
            executor_disposition=(
                RelationshipProductExecutorDisposition.APPLY_CANDIDATE
            ),
        )
        strict = RelationshipProductHorizonCampaignArmInitialization(
            arm_id=RelationshipProductHorizonCampaignArm.STRICT_NOOP,
            owner_persistence_snapshot=self._fresh_owner_persistence(),
            starting_owner_persistence_sha256=(
                self.common_terminal_owner_persistence_sha256
            ),
            batch=self.batch,
            batch_receipt=self.withhold_receipt,
            frozen_policy=self._fresh_cold_policy(),
            forecast_runtime=self._fresh_forecast_runtime(),
            executor_disposition=(
                RelationshipProductExecutorDisposition.FORCE_STRICT_NOOP
            ),
        )
        return full, frozen, strict


@dataclass(frozen=True)
class RelationshipProductHorizonForcedCampaignInputs:
    """Validated immutable inputs for a separate Product Horizon campaign."""

    forced_protocol_id: str
    forced_protocol_raw_sha256: str
    forced_artifact_id: str
    forced_manifest_raw_sha256: str
    public_plan_sha256: str
    lineage_schema_version: str
    lineage_id: str
    lineage: tuple[RelationshipProductHorizonCampaignLineageEntry, ...]
    public_view: RelationshipProductHorizonPublicView
    roots: tuple[RelationshipProductHorizonForcedCampaignRootInput, ...]


@dataclass(frozen=True)
class _Dependencies:
    protocol: RelationshipProductHorizonForcedCommonBatchProtocol
    dynamic_dependencies: dynamic._Dependencies
    dynamic_manifest: Mapping[str, object]
    dynamic_root: pathlib.Path
    schedule_index: Mapping[str, object]
    schedules: tuple[RelationshipProductForcedCollectionScheduleArtifact, ...]

    @property
    def public_view(self) -> RelationshipProductHorizonPublicView:
        return self.dynamic_dependencies.public_view


@dataclass(frozen=True)
class _ForcedBatchReplay:
    completed_root_count: int
    onboarding_count: int
    preaction_count: int
    postaction_count: int
    first_preaction_exact_match_count: int
    first_preaction_projection_sha256: str
    later_owner_handoff_count: int
    owner_writeback_change_count: int
    selected_branch_resolution_count: int
    selected_branch_commitment_match_count: int
    unique_command_count: int
    unique_receipt_count: int
    unique_exposure_count: int
    unique_forecast_count: int
    unique_commitment_count: int
    unique_settlement_count: int
    unique_environment_evidence_ref_count: int
    unique_credit_count: int
    cold_checkpoint_unchanged_count: int
    root_batch_count: int
    unique_batch_count: int
    apply_receipt_count: int
    withhold_receipt_count: int
    full_owner_replay_count: int
    owner_roundtrip_count: int
    parameter_delta_nonzero_root_count: int
    parameter_cap_hit_root_count: int
    scheduled_role_counts: Mapping[str, int]
    delivered_action_counts: Mapping[str, int]
    terminal_failure_reasons: tuple[str, ...]
    terminal_status: str


def relationship_product_horizon_forced_common_batch_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).with_name("protocols") / _PROTOCOL_FILENAME


def load_relationship_product_horizon_forced_common_batch_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonForcedCommonBatchProtocol:
    source = pathlib.Path(
        path or relationship_product_horizon_forced_common_batch_protocol_path()
    )
    raw = source.read_bytes()
    payload = cal._parse_json_bytes(raw, source="forced common-batch protocol")
    cal._exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "purpose",
            "adaptive_lineage",
            "upstream_dynamic_gate",
            "runtime_inputs",
            "forced_schedule_index",
            "collection",
            "root_local_batch_transition",
            "runtime_order",
            "terminal_gates",
            "causal_firewall",
            "claims",
            "claim_boundary",
        },
        "forced common-batch protocol",
    )
    if (
        payload["schema_version"] != FORCED_COMMON_BATCH_PROTOCOL_SCHEMA_VERSION
        or payload["evidence_tier"] != "development"
        or payload["owner"]
        != "lifeform_evolution.relationship_product_horizon_forced_common_batch"
        or payload["purpose"]
        != "root_local_forced_common_collection_and_arm_initialization"
    ):
        raise ValueError("forced common-batch protocol identity drifted")
    _validate_protocol(payload)
    return RelationshipProductHorizonForcedCommonBatchProtocol(
        payload=payload,
        raw_bytes=raw,
        protocol_id=sha256_json(payload),
        raw_sha256=cal._sha256_bytes(raw),
    )


def _expect_fields(
    payload: Mapping[str, object],
    expected: set[str],
    source: str,
) -> None:
    cal._exact_keys(payload, expected, source)


def _contract_equal(actual: object, expected: object) -> bool:
    """Compare protocol literals without Python's bool/int aliasing."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            return False
        return all(
            _contract_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(
            _contract_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return type(actual) is type(expected) and actual == expected


def _validate_protocol(payload: Mapping[str, object]) -> None:
    lineage = cal._mapping(payload["adaptive_lineage"], "adaptive_lineage")
    upstream = cal._mapping(
        payload["upstream_dynamic_gate"], "upstream_dynamic_gate"
    )
    runtime = cal._mapping(payload["runtime_inputs"], "runtime_inputs")
    schedule = cal._mapping(
        payload["forced_schedule_index"], "forced_schedule_index"
    )
    collection = cal._mapping(payload["collection"], "collection")
    transition = cal._mapping(
        payload["root_local_batch_transition"], "root_local_batch_transition"
    )
    order = cal._mapping(payload["runtime_order"], "runtime_order")
    terminal = cal._mapping(payload["terminal_gates"], "terminal_gates")
    firewall = cal._mapping(payload["causal_firewall"], "causal_firewall")
    claims = cal._mapping(payload["claims"], "claims")
    _expect_fields(
        lineage,
        {
            "source_v4_public_previously_observed",
            "development_reader_unqualified",
            "theta0_v2_trained_on_already_used_source_v3",
            "upstream_dynamic_gate_transductive",
            "transductive",
            "unseen",
            "confirmatory",
            "formal",
        },
        "adaptive_lineage",
    )
    if not _contract_equal(lineage, {
        "source_v4_public_previously_observed": True,
        "development_reader_unqualified": True,
        "theta0_v2_trained_on_already_used_source_v3": True,
        "upstream_dynamic_gate_transductive": True,
        "transductive": True,
        "unseen": False,
        "confirmatory": False,
        "formal": False,
    }):
        raise ValueError("forced common-batch adaptive lineage drifted")
    _expect_fields(
        upstream,
        {
            "protocol_id",
            "protocol_raw_sha256",
            "artifact_id",
            "manifest_raw_sha256",
            "trace_relative_path",
            "trace_raw_sha256",
            "terminal_status",
            "forced_common_batch_protocol_freeze_authorized",
            "forced_common_batch_execution_authorized",
        },
        "upstream_dynamic_gate",
    )
    for field in (
        "protocol_id",
        "protocol_raw_sha256",
        "artifact_id",
        "manifest_raw_sha256",
        "trace_raw_sha256",
    ):
        cal._digest(upstream[field], f"upstream_dynamic_gate.{field}")
    if (
        upstream["trace_relative_path"] != "dynamic_collection_prefix.jsonl"
        or upstream["terminal_status"]
        != (
            "development_dynamic_collection_prefix_closed_"
            "forced_batch_protocol_freeze_authorized_effect_not_tested"
        )
        or upstream["forced_common_batch_protocol_freeze_authorized"] is not True
        or upstream["forced_common_batch_execution_authorized"] is not False
    ):
        raise ValueError("upstream dynamic-gate authority drifted")
    _expect_fields(
        runtime,
        {
            "source_v4_admission_protocol_id",
            "source_v4_admission_artifact_id",
            "source_v4_public_plan_sha256",
            "source_v4_public_plan_raw_sha256",
            "source_v4_sealed_bundle_sha256",
            "source_v4_commitment_index_raw_sha256",
            "development_reader_package_artifact_id",
            "development_reader_manifest_raw_sha256",
            "embedding_table_artifact_id",
            "embedding_table_raw_sha256",
            "reader_artifact_id",
            "reader_artifact_raw_sha256",
            "condition_reader_qualified",
            "theta0_v2_bootstrap_protocol_id",
            "theta0_v2_bootstrap_artifact_id",
            "theta0_v2_manifest_raw_sha256",
            "theta0_v2_artifact_id",
            "theta0_v2_artifact_raw_sha256",
            "cold_checkpoint_content_sha256",
            "cold_frozen_policy_id",
            "cold_random_seed",
            "cold_update_count",
            "cold_processed_credit_id_count",
            "cold_pending_decision_count",
        },
        "runtime_inputs",
    )
    for field, value in runtime.items():
        if field.endswith("_sha256") and field != "cold_frozen_policy_id":
            cal._digest(value, f"runtime_inputs.{field}")
    if (
        runtime["condition_reader_qualified"] is not False
        or not _contract_equal(runtime["cold_update_count"], 0)
        or not _contract_equal(runtime["cold_processed_credit_id_count"], 0)
        or not _contract_equal(runtime["cold_pending_decision_count"], 0)
    ):
        raise ValueError("forced common-batch runtime cold boundary drifted")
    for field in (
        "source_v4_admission_protocol_id",
        "source_v4_admission_artifact_id",
        "development_reader_package_artifact_id",
        "embedding_table_artifact_id",
        "reader_artifact_id",
        "theta0_v2_bootstrap_protocol_id",
        "theta0_v2_bootstrap_artifact_id",
        "theta0_v2_artifact_id",
        "cold_checkpoint_content_sha256",
        "cold_frozen_policy_id",
        "cold_random_seed",
    ):
        cal._text(runtime[field], f"runtime_inputs.{field}")
    _expect_fields(
        schedule,
        {
            "schema_version",
            "expected_schedule_index_id",
            "expected_schedule_index_raw_sha256",
            "root_order",
            "role_formula",
            "root_count",
            "decision_count_per_root",
            "schedule_artifact_count",
            "schedule_entry_count",
            "ordered_schedule_artifact_ids",
            "global_role_counts",
            "per_root_role_counts",
            "per_decision_position_role_counts",
            "schedule_uses_only_public_root_and_decision_positions",
        },
        "forced_schedule_index",
    )
    if (
        schedule["schema_version"] != FORCED_SCHEDULE_INDEX_SCHEMA_VERSION
        or schedule["root_order"] != "source_v4_public_root_array_order"
        or schedule["role_formula"] != _ROLE_FORMULA
        or schedule["root_count"] != 112
        or schedule["decision_count_per_root"] != 8
        or schedule["schedule_artifact_count"] != 112
        or schedule["schedule_entry_count"] != 896
        or not _contract_equal(
            schedule["global_role_counts"],
            {"neutral_noop": 448, "owner_recommendation": 448},
        )
        or not _contract_equal(
            schedule["per_root_role_counts"],
            {"neutral_noop": 4, "owner_recommendation": 4},
        )
        or not _contract_equal(
            schedule["per_decision_position_role_counts"],
            {"neutral_noop": 56, "owner_recommendation": 56},
        )
        or schedule["schedule_uses_only_public_root_and_decision_positions"]
        is not True
    ):
        raise ValueError("forced schedule-index design drifted")
    cal._digest(
        schedule["expected_schedule_index_id"],
        "forced_schedule_index.expected_schedule_index_id",
    )
    cal._digest(
        schedule["expected_schedule_index_raw_sha256"],
        "forced_schedule_index.expected_schedule_index_raw_sha256",
    )
    schedule_ids = cal._list(
        schedule["ordered_schedule_artifact_ids"],
        "forced_schedule_index.ordered_schedule_artifact_ids",
    )
    if (
        len(schedule_ids) != 112
        or len(set(schedule_ids)) != 112
        or any(
            not isinstance(item, str)
            or not item.startswith(
                "relationship-product-forced-collection-schedule-sha256:"
            )
            or len(item.rsplit(":", maxsplit=1)[-1]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in item.rsplit(":", maxsplit=1)[-1]
            )
            for item in schedule_ids
        )
    ):
        raise ValueError("ordered schedule artifact IDs drifted")
    _validate_collection_contract(collection)
    _validate_transition_contract(transition)
    _validate_order_and_claims(order, terminal, firewall, claims)
    cal._text(payload["claim_boundary"], "claim_boundary")


def _validate_collection_contract(collection: Mapping[str, object]) -> None:
    _expect_fields(
        collection,
        {
            "root_count",
            "onboarding_session_count_per_root",
            "decision_count_per_root",
            "decision_indices",
            "owner_reset_each_root",
            "owner_handoff_within_root",
            "collection_runs_once_per_root",
            "arm_field_forbidden_during_collection",
            "forced_exposure_sequence_scope",
            "actual_action_source",
            "credit_timestamp_formula",
            "credit_applied_online",
            "gate_update_count",
            "environment_common_random_number_design",
        },
        "collection",
    )
    if not _contract_equal(collection, {
        "root_count": 112,
        "onboarding_session_count_per_root": 4,
        "decision_count_per_root": 8,
        "decision_indices": list(range(8)),
        "owner_reset_each_root": True,
        "owner_handoff_within_root": "exact_prior_forced_postaction_persistence",
        "collection_runs_once_per_root": True,
        "arm_field_forbidden_during_collection": True,
        "forced_exposure_sequence_scope": "root_local_zero_through_seven",
        "actual_action_source": "forced_temporal_receipt_delivered_action_id",
        "credit_timestamp_formula": _CREDIT_TIMESTAMP_FORMULA,
        "credit_applied_online": False,
        "gate_update_count": 0,
        "environment_common_random_number_design": False,
    }):
        raise ValueError("forced collection contract drifted")


def _validate_transition_contract(transition: Mapping[str, object]) -> None:
    _expect_fields(
        transition,
        {
            "batch_scope",
            "batch_count",
            "credit_count_per_batch",
            "global_batch_forbidden",
            "full_disposition",
            "frozen_theta0_disposition",
            "strict_noop_disposition",
            "frozen_and_strict_share_exact_withhold_receipt",
            "full_recovery",
            "common_terminal_owner_state",
            "owner_payload_consumer_interpretation_forbidden",
            "evaluation_executor_dispositions",
            "evaluation_online_gate_update_count",
            "learnable_estimand_future_only",
            "steerable_estimand_future_only",
        },
        "root_local_batch_transition",
    )
    expected = {
        "batch_scope": "one_eight_credit_batch_per_root",
        "batch_count": 112,
        "credit_count_per_batch": 8,
        "global_batch_forbidden": True,
        "full_disposition": "apply",
        "frozen_theta0_disposition": "withhold",
        "strict_noop_disposition": "withhold",
        "frozen_and_strict_share_exact_withhold_receipt": True,
        "full_recovery": "theta0_plus_exact_batch_plus_apply_receipt_only",
        "common_terminal_owner_state": True,
        "owner_payload_consumer_interpretation_forbidden": True,
        "evaluation_executor_dispositions": {
            "full": "apply_candidate",
            "frozen_theta0": "apply_candidate",
            "strict_noop": "force_strict_noop",
        },
        "evaluation_online_gate_update_count": 0,
        "learnable_estimand_future_only": "full_minus_frozen_theta0",
        "steerable_estimand_future_only": "frozen_theta0_minus_strict_noop",
    }
    if not _contract_equal(transition, expected):
        raise ValueError("root-local batch transition contract drifted")


def _validate_order_and_claims(
    order: Mapping[str, object],
    terminal: Mapping[str, object],
    firewall: Mapping[str, object],
    claims: Mapping[str, object],
) -> None:
    required_true_order = {
        "schedule_index_create_only_and_fsynced_before_first_forecast",
        "preaction_fsynced_before_selected_branch_open",
        "selected_branch_commitment_checked_before_settlement",
        "postaction_fsynced_before_next_preaction",
        "manifest_written_last",
    }
    required_false_order = {
        "dynamic_trace_read",
        "dynamic_natural_outcome_read",
        "dynamic_natural_credit_read",
        "sealed_truth_passed_to_reader_forecast_gate_or_executor",
        "incomplete_root_without_manifest_is_evidence",
    }
    _expect_fields(
        order,
        required_true_order | required_false_order,
        "runtime_order",
    )
    if any(order[field] is not True for field in required_true_order) or any(
        order[field] is not False for field in required_false_order
    ):
        raise ValueError("forced common-batch runtime order drifted")
    terminal_expected = {
        "root_count": 112,
        "onboarding_count": 448,
        "preaction_count": 896,
        "postaction_count": 896,
        "later_owner_handoff_count": 784,
        "owner_writeback_change_count": 896,
        "first_preaction_exact_match_count": 112,
        "root_batch_count": 112,
        "credit_count_per_batch": 8,
        "unique_batch_count": 112,
        "apply_receipt_count": 112,
        "withhold_receipt_count": 112,
        "full_owner_replay_count": 112,
        "owner_roundtrip_count": 112,
        "parameter_delta_nonzero_root_min_count": 1,
        "parameter_cap_hit_root_count": 0,
        "successful_terminal": _SUCCESS_STATUS,
        "failed_terminal": _FAIL_STATUS,
        "degenerate_terminal": _DEGENERATE_STATUS,
        "campaign_protocol_freeze_authorized_on_success": True,
        "campaign_execution_authorized_on_success": False,
        "scientific_retry_with_modified_root_theta_reader_source_schedule_or_order_forbidden": True,
    }
    _expect_fields(terminal, set(terminal_expected), "terminal_gates")
    if not _contract_equal(terminal, terminal_expected):
        raise ValueError("forced common-batch terminal gates drifted")
    firewall_expected = {
        "dynamic_manifest_file_read_count": 1,
        "dynamic_trace_file_read_count": 0,
        "dynamic_natural_outcome_read_count": 0,
        "dynamic_natural_credit_read_count": 0,
        "source_v4_public_plan_file_read_count": 1,
        "source_v4_sealed_file_read_count": 2,
        "upstream_scanner_trace_file_read_count": 1,
        "challenge_label_file_read_count": 0,
        "group_split_file_read_count": 0,
        "collection_credit_applied_count": 0,
        "collection_gate_update_count": 0,
        "evaluation_decision_count": 0,
        "evaluation_or_judge_feedback_count": 0,
        "model_output_count": 0,
        "cuda_execution_count": 0,
        "human_sample_count": 0,
    }
    _expect_fields(firewall, set(firewall_expected), "causal_firewall")
    if not _contract_equal(firewall, firewall_expected):
        raise ValueError("forced common-batch causal firewall drifted")
    allowed_true = {"forced_common_batch_execution_authorized"}
    _expect_fields(
        claims,
        {
            "forced_common_batch_execution_authorized",
            "forced_common_collection_batches_materialized",
            "arm_initialization_transition_verified",
            "campaign_protocol_freeze_authorized",
            "evaluation_execution_authorized",
            "campaign_execution_authorized",
            "effect_tested",
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
        "claims",
    )
    if any(type(value) is not bool for value in claims.values()):
        raise ValueError("forced common-batch claims must be booleans")
    if {key for key, value in claims.items() if value} != allowed_true:
        raise ValueError("forced common-batch protocol claim ceiling drifted")


def _forced_role(
    *, root_index: int, decision_index: int
) -> RelationshipProductForcedActionRole:
    return (
        RelationshipProductForcedActionRole.OWNER_RECOMMENDATION
        if ((root_index // 2) + (decision_index // 2)) % 2 == 0
        else RelationshipProductForcedActionRole.NEUTRAL_NOOP
    )


def _build_schedule_index(
    public_view: RelationshipProductHorizonPublicView,
) -> tuple[
    Mapping[str, object],
    tuple[RelationshipProductForcedCollectionScheduleArtifact, ...],
]:
    if len(public_view.roots) != 112:
        raise ValueError("forced schedule index requires exactly 112 public roots")
    schedules: list[RelationshipProductForcedCollectionScheduleArtifact] = []
    entries: list[Mapping[str, object]] = []
    global_counts: Counter[str] = Counter()
    column_counts: dict[int, Counter[str]] = {
        index: Counter() for index in range(8)
    }
    decision_ids: set[str] = set()
    schedule_entry_ids: set[str] = set()
    for root_index, root in enumerate(public_view.roots):
        decisions = root.decision_sessions[:8]
        if tuple(item.decision_index for item in decisions) != tuple(range(8)):
            raise ValueError("forced schedule root decisions must be ordered zero to seven")
        schedule_entries = tuple(
            RelationshipProductForcedCollectionScheduleEntry(
                decision_id=decision.decision_id,
                sequence_index=decision.decision_index,
                forced_action_role=_forced_role(
                    root_index=root_index,
                    decision_index=decision.decision_index,
                ),
            )
            for decision in decisions
        )
        schedule = RelationshipProductForcedCollectionScheduleArtifact(
            entries=schedule_entries
        )
        root_counts = Counter(
            item.forced_action_role.value for item in schedule_entries
        )
        if root_counts != Counter(
            {"neutral_noop": 4, "owner_recommendation": 4}
        ):
            raise ValueError("forced schedule per-root role balance drifted")
        for item in schedule_entries:
            if item.decision_id in decision_ids or item.entry_id in schedule_entry_ids:
                raise ValueError("forced schedule decision or entry identity was reused")
            decision_ids.add(item.decision_id)
            schedule_entry_ids.add(item.entry_id)
            global_counts[item.forced_action_role.value] += 1
            column_counts[item.sequence_index][item.forced_action_role.value] += 1
        schedules.append(schedule)
        entries.append(
            {
                "root_sequence_index": root_index,
                "subject_id": root.subject_id,
                "schedule_artifact_id": schedule.artifact_id,
                "schedule_artifact": schedule.to_payload(),
            }
        )
    if global_counts != Counter(
        {"neutral_noop": 448, "owner_recommendation": 448}
    ):
        raise ValueError("forced schedule global role balance drifted")
    if any(
        counts != Counter({"neutral_noop": 56, "owner_recommendation": 56})
        for counts in column_counts.values()
    ):
        raise ValueError("forced schedule decision-position balance drifted")
    if len(decision_ids) != 896 or len(schedule_entry_ids) != 896:
        raise ValueError("forced schedule unique entry count drifted")
    core = {
        "schema_version": FORCED_SCHEDULE_INDEX_SCHEMA_VERSION,
        "source_v4_public_plan_sha256": public_view.public_plan_sha256,
        "root_order": "source_v4_public_root_array_order",
        "role_formula": _ROLE_FORMULA,
        "root_count": 112,
        "decision_count_per_root": 8,
        "entries": entries,
    }
    return {"schedule_index_id": sha256_json(core), **core}, tuple(schedules)


def _file_entries(
    manifest: Mapping[str, object], *, source: str
) -> Mapping[str, Mapping[str, object]]:
    return cal._file_entry_map(manifest["files"], f"{source}.files")


def _load_dependencies(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    dynamic_root: pathlib.Path,
) -> _Dependencies:
    protocol = load_relationship_product_horizon_forced_common_batch_protocol()
    upstream = cal._mapping(
        protocol.payload["upstream_dynamic_gate"], "upstream_dynamic_gate"
    )
    dynamic_path = pathlib.Path(dynamic_root)
    manifest_raw = cal._require_raw_sha(
        dynamic_path / "manifest.json",
        upstream["manifest_raw_sha256"],
        "upstream dynamic-gate manifest",
    )
    manifest = cal._parse_json_bytes(
        manifest_raw, source="upstream dynamic-gate manifest"
    )
    if manifest_raw != cal._canonical_bytes(manifest):
        raise ValueError("upstream dynamic-gate manifest must use canonical bytes")
    if (
        manifest["protocol_id"] != upstream["protocol_id"]
        or manifest["protocol_raw_sha256"] != upstream["protocol_raw_sha256"]
        or manifest["artifact_id"] != upstream["artifact_id"]
        or manifest["status"] != upstream["terminal_status"]
        or manifest["claims"]["forced_common_batch_protocol_freeze_authorized"]
        is not True
        or manifest["claims"]["forced_common_batch_execution_authorized"]
        is not False
    ):
        raise ValueError("upstream dynamic-gate manifest authority drifted")
    if manifest["artifact_id"] != sha256_json(
        {key: value for key, value in manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("upstream dynamic-gate artifact identity drifted")
    dynamic_files = _file_entries(manifest, source="upstream dynamic-gate")
    trace_relative = cal._text(
        upstream["trace_relative_path"],
        "upstream_dynamic_gate.trace_relative_path",
    )
    if dynamic_files[trace_relative]["raw_sha256"] != upstream["trace_raw_sha256"]:
        raise ValueError("upstream dynamic trace manifest pin drifted")

    inherited = dynamic._load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
        scanner_root=pathlib.Path(scanner_root),
    )
    runtime = cal._mapping(protocol.payload["runtime_inputs"], "runtime_inputs")
    scanner_dependencies = inherited.scanner_dependencies
    source_manifest = cal._parse_json_bytes(
        cal._read_regular(pathlib.Path(source_v4_admission_root) / "manifest.json"),
        source="source-v4 admission manifest",
    )
    source_files = _file_entries(source_manifest, source="source-v4 admission")
    reader_manifest_raw = cal._read_regular(pathlib.Path(reader_root) / "manifest.json")
    theta_manifest_raw = cal._read_regular(pathlib.Path(theta0_v2_root) / "manifest.json")
    if (
        source_manifest["protocol_id"]
        != runtime["source_v4_admission_protocol_id"]
        or source_manifest["artifact_id"]
        != runtime["source_v4_admission_artifact_id"]
        or source_manifest["public_plan_sha256"]
        != runtime["source_v4_public_plan_sha256"]
        or source_manifest["sealed_bundle_sha256"]
        != runtime["source_v4_sealed_bundle_sha256"]
        or source_files["public/source_plan.json"]["raw_sha256"]
        != runtime["source_v4_public_plan_raw_sha256"]
        or source_files[
            "sealed/action_counterfactual_commitment_index.json"
        ]["raw_sha256"]
        != runtime["source_v4_commitment_index_raw_sha256"]
        or inherited.public_view.public_plan_sha256
        != runtime["source_v4_public_plan_sha256"]
    ):
        raise ValueError("forced common-batch source-v4 inputs drifted")
    reader_manifest = cal._parse_json_bytes(
        reader_manifest_raw, source="development reader manifest"
    )
    reader_files = _file_entries(reader_manifest, source="development reader")
    if (
        cal._sha256_bytes(reader_manifest_raw)
        != runtime["development_reader_manifest_raw_sha256"]
        or reader_manifest["artifact_id"]
        != runtime["development_reader_package_artifact_id"]
        or reader_manifest["embedding_table_artifact_id"]
        != runtime["embedding_table_artifact_id"]
        or reader_manifest["reader_artifact_id"] != runtime["reader_artifact_id"]
        or reader_manifest["claims"]["condition_reader_qualified"] is not False
        or reader_files["embedding_table.json"]["raw_sha256"]
        != runtime["embedding_table_raw_sha256"]
        or reader_files["reader_artifact.json"]["raw_sha256"]
        != runtime["reader_artifact_raw_sha256"]
    ):
        raise ValueError("forced common-batch reader inputs drifted")
    theta_manifest = cal._parse_json_bytes(
        theta_manifest_raw, source="theta0-v2 manifest"
    )
    theta_files = _file_entries(theta_manifest, source="theta0-v2")
    policy = scanner_dependencies.frozen_policy
    theta0 = scanner_dependencies.theta0
    if (
        cal._sha256_bytes(theta_manifest_raw)
        != runtime["theta0_v2_manifest_raw_sha256"]
        or theta_manifest["protocol_id"]
        != runtime["theta0_v2_bootstrap_protocol_id"]
        or theta_manifest["artifact_id"]
        != runtime["theta0_v2_bootstrap_artifact_id"]
        or theta_manifest["published_theta0_artifact_id"]
        != runtime["theta0_v2_artifact_id"]
        or theta_files["theta0_artifact.json"]["raw_sha256"]
        != runtime["theta0_v2_artifact_raw_sha256"]
        or theta0.artifact_id != runtime["theta0_v2_artifact_id"]
        or policy.checkpoint.content_sha256
        != runtime["cold_checkpoint_content_sha256"]
        or policy.policy_id != runtime["cold_frozen_policy_id"]
        or policy.random_seed != runtime["cold_random_seed"]
        or policy.checkpoint.update_count != 0
        or policy.checkpoint.processed_credit_ids
        or policy.checkpoint.pending_decisions
    ):
        raise ValueError("forced common-batch theta0-v2 inputs drifted")
    schedule_index, schedules = _build_schedule_index(inherited.public_view)
    schedule_pin = cal._mapping(
        protocol.payload["forced_schedule_index"], "forced_schedule_index"
    )
    if (
        schedule_index["schedule_index_id"]
        != schedule_pin["expected_schedule_index_id"]
        or cal._sha256_bytes(cal._canonical_bytes(schedule_index))
        != schedule_pin["expected_schedule_index_raw_sha256"]
        or [item.artifact_id for item in schedules]
        != schedule_pin["ordered_schedule_artifact_ids"]
    ):
        raise ValueError("forced schedule index identity drifted")
    return _Dependencies(
        protocol=protocol,
        dynamic_dependencies=inherited,
        dynamic_manifest=manifest,
        dynamic_root=dynamic_path,
        schedule_index=schedule_index,
        schedules=schedules,
    )


def _authorization(
    *,
    protocol_id: str,
    frozen_policy: object,
) -> RelationshipProductFrozenPulseAuthorization:
    pulse = RelationshipProductPulseAuthorization(
        authorization_id=f"forced-common-batch:{protocol_id}",
        allowed_policy_artifact_id=frozen_policy.artifact.artifact_id,
        allowed_policy_artifact_version=frozen_policy.artifact.artifact_version,
    )
    return RelationshipProductFrozenPulseAuthorization(
        pulse_authorization=pulse,
        allowed_frozen_policy_id=frozen_policy.policy_id,
        allowed_checkpoint_content_sha256=(
            frozen_policy.checkpoint.content_sha256
        ),
    )


def _forced_safe_projection_from_scanner(
    record: Mapping[str, object],
) -> Mapping[str, object]:
    projection = {
        "schema_version": FORCED_SAFE_FIRST_PROJECTION_SCHEMA_VERSION,
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
        "candidate_advisory_id": record["candidate_advisory_id"],
        "cold_checkpoint_content_sha256": record[
            "cold_checkpoint_content_sha256"
        ],
        "cold_frozen_policy_id": record["cold_frozen_policy_id"],
    }
    if tuple(projection) != FORCED_SAFE_FIRST_PROJECTION_FIELDS:
        raise RuntimeError("forced-safe scanner projection schema drifted")
    return projection


def _forced_safe_projection(
    *,
    root_sequence_index: int,
    root: HorizonPublicRoot,
    decision: HorizonPublicDecisionSession,
    owner_input_sha256: str,
    preaction: RelationshipProductForcedCollectionPreActionSnapshot,
) -> Mapping[str, object]:
    gate_decision = preaction.frozen_decision.decision
    projection = {
        "schema_version": FORCED_SAFE_FIRST_PROJECTION_SCHEMA_VERSION,
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
        "candidate_advisory_id": (
            preaction.execution_receipt.candidate_advisory.advisory_id
        ),
        "cold_checkpoint_content_sha256": (
            preaction.frozen_policy.checkpoint.content_sha256
        ),
        "cold_frozen_policy_id": preaction.frozen_policy.policy_id,
    }
    if tuple(projection) != FORCED_SAFE_FIRST_PROJECTION_FIELDS:
        raise RuntimeError("forced-safe first projection schema drifted")
    return projection


def _credit_timestamp(root_sequence_index: int, decision_index: int) -> int:
    return root_sequence_index * 20 + 5 + 2 * decision_index


def _owner_snapshot_payload(snapshot: OwnerPersistenceSnapshot) -> Mapping[str, object]:
    if not isinstance(snapshot, OwnerPersistenceSnapshot):
        raise TypeError("owner snapshot has unexpected type")
    return {
        "owner_name": snapshot.owner_name,
        "schema_version": snapshot.schema_version,
        "payload": snapshot.payload,
        "description": snapshot.description,
    }


def _owner_snapshot_from_payload(payload: object) -> OwnerPersistenceSnapshot:
    raw = cal._mapping(payload, "owner persistence envelope")
    cal._exact_keys(
        raw,
        {"owner_name", "schema_version", "payload", "description"},
        "owner persistence envelope",
    )
    snapshot = OwnerPersistenceSnapshot(
        owner_name=cal._text(raw["owner_name"], "owner_name"),
        schema_version=cal._integer(raw["schema_version"], "schema_version"),
        payload=cal._mapping(raw["payload"], "owner payload"),
        description=(
            raw["description"]
            if isinstance(raw["description"], str)
            else cal._text(raw["description"], "description")
        ),
    )
    store = SocialRecordStore()
    store.hydrate_from_persistence(snapshot)
    replayed = store.export_persistence_snapshot()
    if replayed != snapshot:
        raise ValueError("owner persistence envelope did not round-trip exactly")
    return snapshot


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
    unique_command_count: int,
    unique_receipt_count: int,
    unique_exposure_count: int,
    unique_forecast_count: int,
    unique_commitment_count: int,
    unique_settlement_count: int,
    unique_environment_evidence_ref_count: int,
    unique_credit_count: int,
    cold_checkpoint_unchanged_count: int,
    root_batch_count: int,
    unique_batch_count: int,
    apply_receipt_count: int,
    withhold_receipt_count: int,
    full_owner_replay_count: int,
    owner_roundtrip_count: int,
    parameter_cap_hit_root_count: int,
    delivered_noop_count: int,
    delivered_nonnoop_count: int,
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
        (unique_command_count != 896, "unique_command_count_not_896"),
        (unique_receipt_count != 896, "unique_receipt_count_not_896"),
        (unique_exposure_count != 896, "unique_exposure_count_not_896"),
        (unique_forecast_count != 896, "unique_forecast_count_not_896"),
        (unique_commitment_count != 896, "unique_commitment_count_not_896"),
        (unique_settlement_count != 896, "unique_settlement_count_not_896"),
        (
            unique_environment_evidence_ref_count != 896,
            "unique_environment_evidence_ref_count_not_896",
        ),
        (unique_credit_count != 896, "unique_credit_count_not_896"),
        (
            cold_checkpoint_unchanged_count != 896,
            "cold_checkpoint_unchanged_count_not_896",
        ),
        (root_batch_count != 112, "root_batch_count_not_112"),
        (unique_batch_count != 112, "unique_batch_count_not_112"),
        (apply_receipt_count != 112, "apply_receipt_count_not_112"),
        (withhold_receipt_count != 112, "withhold_receipt_count_not_112"),
        (full_owner_replay_count != 112, "full_owner_replay_count_not_112"),
        (owner_roundtrip_count != 112, "owner_roundtrip_count_not_112"),
        (
            parameter_cap_hit_root_count != 0,
            "parameter_cap_hit_root_count_not_zero",
        ),
        (delivered_noop_count < 1, "delivered_noop_count_below_one"),
        (delivered_nonnoop_count < 1, "delivered_nonnoop_count_below_one"),
    )
    return tuple(reason for failed, reason in checks if failed)


async def _run_forced_common_batch(
    *,
    dependencies: _Dependencies,
    trace_sink: cal._TraceSink,
    state_sink: cal._TraceSink,
    transition_sink: cal._TraceSink,
) -> _ForcedBatchReplay:
    scanner_dependencies = dependencies.dynamic_dependencies.scanner_dependencies
    theta0 = scanner_dependencies.theta0
    frozen_policy = scanner_dependencies.frozen_policy
    checkpoint = frozen_policy.checkpoint
    frozen_authorization = _authorization(
        protocol_id=dependencies.protocol.protocol_id,
        frozen_policy=frozen_policy,
    )
    expected_first = tuple(
        _forced_safe_projection_from_scanner(item)
        for item in dependencies.dynamic_dependencies.expected_first_projections
    )
    trace_sink.append(
        {
            "schema_version": FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION,
            "record_type": "header",
            "protocol_id": dependencies.protocol.protocol_id,
            "schedule_index_id": dependencies.schedule_index["schedule_index_id"],
            "schedule_index_fsynced_before_first_forecast": True,
            "dynamic_gate_artifact_id": dependencies.dynamic_manifest["artifact_id"],
            "dynamic_manifest_file_read_count": 1,
            "dynamic_trace_file_read_count": 0,
            "dynamic_natural_outcome_read_count": 0,
            "dynamic_natural_credit_read_count": 0,
            "root_count": 112,
            "decision_indices": list(range(8)),
            "cold_checkpoint_content_sha256": checkpoint.content_sha256,
            "cold_frozen_policy_id": frozen_policy.policy_id,
            "cold_random_seed": frozen_policy.random_seed,
            "condition_reader_qualified": False,
            "evaluation_decision_count": 0,
            "model_output_count": 0,
            "cuda_execution_count": 0,
        }
    )

    environment_scope: dynamic._SelectedBranchEnvironmentScope | None = None
    first_projections: list[Mapping[str, object]] = []
    scheduled_role_counts: Counter[str] = Counter()
    delivered_action_counts: Counter[str] = Counter()
    command_ids: set[str] = set()
    receipt_ids: set[str] = set()
    exposure_ids: set[str] = set()
    forecast_ids: set[str] = set()
    commitment_ids: set[str] = set()
    settlement_ids: set[str] = set()
    evidence_refs: set[str] = set()
    credit_ids: set[str] = set()
    batch_ids: set[str] = set()
    apply_receipt_ids: set[str] = set()
    withhold_receipt_ids: set[str] = set()
    completed_root_count = 0
    onboarding_count = 0
    preaction_count = 0
    postaction_count = 0
    first_exact_count = 0
    handoff_count = 0
    writeback_change_count = 0
    branch_resolution_count = 0
    commitment_match_count = 0
    checkpoint_unchanged_count = 0
    full_owner_replay_count = 0
    owner_roundtrip_count = 0
    parameter_delta_nonzero_root_count = 0
    parameter_cap_hit_root_count = 0
    previous_credit_timestamp = -1

    for root_index, root in enumerate(dependencies.public_view.roots):
        schedule = dependencies.schedules[root_index]
        owner_persistence = await dynamic.scan._post_onboarding_state(root)
        onboarding_count += len(root.onboarding_sessions)
        post_onboarding_sha = social_record_store_persistence_sha256(
            owner_persistence
        )
        trace_sink.append(
            {
                "schema_version": FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION,
                "record_type": "root_start",
                "root_sequence_index": root_index,
                "subject_id": root.subject_id,
                "schedule_artifact_id": schedule.artifact_id,
                "owner_reset": True,
                "onboarding_appended_once": True,
                "onboarding_session_count": 4,
                "post_onboarding_persistence_sha256": post_onboarding_sha,
            }
        )
        exposures = []
        credits = []
        prior_postaction_owner_sha: str | None = None
        root_credit_ids: set[str] = set()
        for decision in root.decision_sessions[:8]:
            global_sequence = root_index * 8 + decision.decision_index
            owner_input_sha = social_record_store_persistence_sha256(
                owner_persistence
            )
            if decision.decision_index == 0:
                if owner_input_sha != post_onboarding_sha:
                    raise RuntimeError("forced first preaction lost onboarding state")
            else:
                if owner_input_sha != prior_postaction_owner_sha:
                    raise RuntimeError(
                        "forced later preaction lost prior postaction state"
                    )
                handoff_count += 1
            authorization = RelationshipProductForcedCollectionAuthorization(
                frozen_pulse_authorization=frozen_authorization,
                schedule_artifact=schedule,
                decision_id=decision.decision_id,
            )
            scheduled_role_counts[authorization.forced_action_role.value] += 1
            preaction = await prepare_relationship_product_forced_collection_preaction(
                request=dynamic.scan._request(
                    subject_id=root.subject_id,
                    decision=decision,
                ),
                owner_persistence_snapshot=owner_persistence,
                forecast_runtime=scanner_dependencies.forecast_runtime,
                frozen_policy=frozen_policy,
                authorization=authorization,
                substrate_snapshot=cal._placeholder_substrate(),
            )
            if (
                preaction.frozen_policy != frozen_policy
                or preaction.forced_exposure.sequence_index
                != decision.decision_index
                or preaction.execution_receipt.command.authorization
                != authorization
            ):
                raise RuntimeError("forced preaction schedule or cold policy drifted")
            projection = _forced_safe_projection(
                root_sequence_index=root_index,
                root=root,
                decision=decision,
                owner_input_sha256=owner_input_sha,
                preaction=preaction,
            )
            if decision.decision_index == 0:
                if projection != expected_first[root_index]:
                    raise RuntimeError(
                        "forced first preaction differs from scanner-safe seam"
                    )
                first_projections.append(projection)
                first_exact_count += 1
            delivered_action = preaction.delivered_action_id
            expected_action = (
                preaction.forecast.recommended_action_id
                if authorization.forced_action_role
                is RelationshipProductForcedActionRole.OWNER_RECOMMENDATION
                else RelationshipAction.NEUTRAL_NOOP.value
            )
            if delivered_action != expected_action:
                raise RuntimeError("forced executor did not deliver scheduled role")
            delivered_action_counts[delivered_action] += 1
            command_id = preaction.execution_receipt.command.command_id
            receipt_id = preaction.execution_receipt.receipt_id
            exposure_id = preaction.forced_exposure.exposure_id
            forecast_id = preaction.forecast.forecast_id
            if (
                command_id in command_ids
                or receipt_id in receipt_ids
                or exposure_id in exposure_ids
                or forecast_id in forecast_ids
            ):
                raise RuntimeError("forced preaction identity was reused")
            command_ids.add(command_id)
            receipt_ids.add(receipt_id)
            exposure_ids.add(exposure_id)
            forecast_ids.add(forecast_id)
            preaction_count += 1
            trace_sink.append(
                {
                    "schema_version": FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION,
                    "record_type": "preaction",
                    "global_sequence_index": global_sequence,
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "decision_index": decision.decision_index,
                    "session_id": decision.session_id,
                    "decision_id": decision.decision_id,
                    "schedule_artifact_id": schedule.artifact_id,
                    "schedule_entry_id": authorization.schedule_entry_id,
                    "forced_action_role": authorization.forced_action_role.value,
                    "forced_action_id": preaction.forced_exposure.forced_action_id,
                    "delivered_action_id": delivered_action,
                    "command_id": command_id,
                    "receipt_id": receipt_id,
                    "exposure_id": exposure_id,
                    "owner_input_persistence_sha256": owner_input_sha,
                    "forced_safe_projection": projection,
                    "preaction_append_fsynced_before_selected_branch_open": True,
                    "environment_scope_already_created_before_current_preaction": (
                        environment_scope is not None
                    ),
                    "sealed_truth_passed_to_preaction": False,
                    "selected_branch_opened": False,
                    "arm_field_present": False,
                }
            )
            if environment_scope is None:
                environment_scope = dynamic._SelectedBranchEnvironmentScope(
                    dependencies=dependencies.dynamic_dependencies
                )
            branch = environment_scope.settle(
                public_root=root,
                public_decision=decision,
                delivered_action_id=delivered_action,
            )
            branch_resolution_count += 1
            commitment_match_count += 1
            if (
                branch.selected_action_id != delivered_action
                or branch.commitment_id in commitment_ids
            ):
                raise RuntimeError("forced environment branch lineage drifted")
            commitment_ids.add(branch.commitment_id)
            timestamp = _credit_timestamp(root_index, decision.decision_index)
            if timestamp <= previous_credit_timestamp:
                raise RuntimeError("forced credit timestamps are not chronological")
            previous_credit_timestamp = timestamp
            action_turn = 4 + 2 * decision.decision_index
            settlement_input = replace(
                cal._settlement_input(
                    subject_scope=root.subject_id,
                    decision=decision,
                    forecast_id=preaction.forecast.forecast_id,
                    selected_action_id=delivered_action,
                    environment_outcome=branch,
                    action_turn=action_turn,
                    credit_timestamp=timestamp,
                ),
                apply_credit_to_gate=False,
            )
            settled = await settle_relationship_product_forced_collection(
                preaction=preaction,
                settlement_input=settlement_input,
            )
            if (
                settled.credit_applied_to_gate
                or settled.collection_gate_update_delta != 0
                or settled.gate_checkpoint != checkpoint
                or settled.credit.prediction_id != forecast_id
                or settled.credit.abstract_action_id != delivered_action
                or settled.credit.timestamp_ms != timestamp
            ):
                raise RuntimeError("forced PE-credit cold-gate lineage drifted")
            if (
                settled.credit.record_id in credit_ids
                or settled.settlement.settlement_id in settlement_ids
                or branch.environment_evidence_ref in evidence_refs
            ):
                raise RuntimeError("forced postaction identity was reused")
            credit_ids.add(settled.credit.record_id)
            root_credit_ids.add(settled.credit.record_id)
            settlement_ids.add(settled.settlement.settlement_id)
            evidence_refs.add(branch.environment_evidence_ref)
            exposures.append(preaction.forced_exposure)
            credits.append(settled.credit)
            checkpoint_unchanged_count += 1
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
            trace_sink.append(
                {
                    "schema_version": FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION,
                    "record_type": "postaction",
                    "global_sequence_index": global_sequence,
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "decision_index": decision.decision_index,
                    "session_id": decision.session_id,
                    "decision_id": decision.decision_id,
                    "delivered_action_id": delivered_action,
                    "selected_branch_commitment_id": branch.commitment_id,
                    "environment_evidence_ref": branch.environment_evidence_ref,
                    "environment_selected_action_id": branch.selected_action_id,
                    "typed_outcome_id": branch.typed_outcome_id,
                    "settlement_id": settled.settlement.settlement_id,
                    "social_prediction_error": cal._social_pe_payload(
                        settled.social_prediction_error_snapshot.value
                    ),
                    "credit": cal._credit_payload(settled.credit),
                    "credit_applied_online": False,
                    "collection_gate_update_delta": 0,
                    "cold_checkpoint_content_sha256": checkpoint.content_sha256,
                    "owner_preaction_persistence_sha256": owner_preaction_sha,
                    "owner_postaction_persistence_sha256": owner_postaction_sha,
                    "owner_writeback_changed_persistence": (
                        owner_postaction_sha != owner_preaction_sha
                    ),
                    "selected_branch_opened_after_current_preaction_fsync": True,
                    "postaction_append_fsynced_before_next_preaction": True,
                    "evaluation_or_judge_feedback_received": False,
                    "arm_field_present": False,
                }
            )

        if len(exposures) != 8 or len(credits) != 8 or len(root_credit_ids) != 8:
            raise RuntimeError("forced root did not close one eight-credit batch")
        batch = RelationshipActionGateCreditBatch(
            exposures=tuple(exposures), credits=tuple(credits)
        )
        if batch.batch_id in batch_ids:
            raise RuntimeError("forced root batch identity was reused")
        batch_ids.add(batch.batch_id)
        full_gate = RelationshipActionGate.from_theta0(
            theta0, random_seed=frozen_policy.random_seed
        )
        frozen_gate = RelationshipActionGate.from_theta0(
            theta0, random_seed=frozen_policy.random_seed
        )
        strict_gate = RelationshipActionGate.from_theta0(
            theta0, random_seed=frozen_policy.random_seed
        )
        full_plan = full_gate.plan_credit_batch(batch)
        frozen_plan = frozen_gate.plan_credit_batch(batch)
        strict_plan = strict_gate.plan_credit_batch(batch)
        if full_plan != frozen_plan or full_plan != strict_plan:
            raise RuntimeError("root arms did not share one exact batch plan")
        apply_receipt = full_gate.commit_credit_batch(
            full_plan, disposition=RelationshipActionGateBatchDisposition.APPLY
        )
        frozen_receipt = frozen_gate.commit_credit_batch(
            frozen_plan,
            disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
        )
        strict_receipt = strict_gate.commit_credit_batch(
            strict_plan,
            disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
        )
        if frozen_receipt != strict_receipt:
            raise RuntimeError("frozen and strict arms did not share WITHHOLD receipt")
        if (
            apply_receipt.update_count_delta != 8
            or apply_receipt.atomic_commit_count != 1
            or tuple(apply_receipt.applied_credit_ids)
            != tuple(item.record_id for item in batch.credits)
            or frozen_receipt.update_count_delta != 0
            or frozen_receipt.atomic_commit_count != 0
            or frozen_gate.export_checkpoint() != checkpoint
            or strict_gate.export_checkpoint() != checkpoint
        ):
            raise RuntimeError("root batch APPLY/WITHHOLD contract drifted")
        full_checkpoint = full_gate.export_checkpoint()
        full_policy = full_gate.freeze_for_evaluation()
        frozen_eval_policy = frozen_gate.freeze_for_evaluation()
        strict_eval_policy = strict_gate.freeze_for_evaluation()
        if (
            frozen_eval_policy != strict_eval_policy
            or frozen_eval_policy != frozen_policy
        ):
            raise RuntimeError("frozen and strict policy initialization drifted")
        replayed_gate = RelationshipActionGate.from_applied_credit_batch(
            theta0,
            batch=batch,
            receipt=apply_receipt,
            random_seed=frozen_policy.random_seed,
        )
        if (
            replayed_gate.export_checkpoint() != full_checkpoint
            or replayed_gate.freeze_for_evaluation() != full_policy
        ):
            raise RuntimeError("full root batch owner replay drifted")
        full_owner_replay_count += 1
        parameter_delta_nonzero = any(
            value != 0.0
            for value in (*apply_receipt.weight_delta, apply_receipt.bias_delta)
        )
        parameter_delta_nonzero_root_count += int(parameter_delta_nonzero)
        terminal_parameters = (*full_checkpoint.weights, full_checkpoint.bias)
        if not all(math.isfinite(value) for value in terminal_parameters):
            raise RuntimeError("full root batch produced non-finite parameters")
        parameter_cap_hit = any(
            abs(value) >= theta0.max_abs_parameter for value in terminal_parameters
        )
        parameter_cap_hit_root_count += int(parameter_cap_hit)
        owner_payload = _owner_snapshot_payload(owner_persistence)
        owner_payload_raw = cal._canonical_bytes(owner_payload)
        roundtripped_arm_owners = tuple(
            _owner_snapshot_from_payload(
                cal._parse_json_bytes(
                    owner_payload_raw,
                    source=f"{arm_id} common terminal owner",
                )
            )
            for arm_id in ("full", "frozen_theta0", "strict_noop")
        )
        if any(item != owner_persistence for item in roundtripped_arm_owners):
            raise RuntimeError("root terminal owner round-trip drifted")
        owner_roundtrip_count += 1
        owner_sha = social_record_store_persistence_sha256(owner_persistence)
        state_sink.append(
            {
                "schema_version": FORCED_COMMON_BATCH_ROOT_STATE_SCHEMA_VERSION,
                "root_sequence_index": root_index,
                "subject_id": root.subject_id,
                "owner_persistence_sha256": owner_sha,
                "owner_persistence": owner_payload,
                "consumer_interpreted_owner_payload": False,
            }
        )
        transition_sink.append(
            {
                "schema_version": FORCED_COMMON_BATCH_TRANSITION_SCHEMA_VERSION,
                "root_sequence_index": root_index,
                "subject_id": root.subject_id,
                "schedule_artifact_id": schedule.artifact_id,
                "common_terminal_owner_persistence_sha256": owner_sha,
                "batch": batch.to_payload(),
                "plan_id": full_plan.plan_id,
                "candidate_checkpoint_content_sha256": (
                    full_plan.candidate_checkpoint.content_sha256
                ),
                "apply_receipt": apply_receipt.to_payload(),
                "withhold_receipt": frozen_receipt.to_payload(),
                "full_policy_id": full_policy.policy_id,
                "full_checkpoint_content_sha256": full_checkpoint.content_sha256,
                "cold_frozen_policy_id": frozen_policy.policy_id,
                "parameter_delta_nonzero": parameter_delta_nonzero,
                "parameter_cap_hit": parameter_cap_hit,
                "arm_bindings": {
                    "full": {
                        "batch_id": batch.batch_id,
                        "receipt_id": apply_receipt.receipt_id,
                        "policy_id": full_policy.policy_id,
                        "executor_disposition": "apply_candidate",
                        "online_gate_update_count": 0,
                    },
                    "frozen_theta0": {
                        "batch_id": batch.batch_id,
                        "receipt_id": frozen_receipt.receipt_id,
                        "policy_id": frozen_policy.policy_id,
                        "executor_disposition": "apply_candidate",
                        "online_gate_update_count": 0,
                    },
                    "strict_noop": {
                        "batch_id": batch.batch_id,
                        "receipt_id": frozen_receipt.receipt_id,
                        "policy_id": frozen_policy.policy_id,
                        "executor_disposition": "force_strict_noop",
                        "online_gate_update_count": 0,
                    },
                },
            }
        )
        apply_receipt_ids.add(apply_receipt.receipt_id)
        withhold_receipt_ids.add(frozen_receipt.receipt_id)
        trace_sink.append(
            {
                "schema_version": FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION,
                "record_type": "root_terminal",
                "root_sequence_index": root_index,
                "subject_id": root.subject_id,
                "schedule_artifact_id": schedule.artifact_id,
                "batch_id": batch.batch_id,
                "batch_credit_count": len(batch.credits),
                "apply_receipt_id": apply_receipt.receipt_id,
                "withhold_receipt_id": frozen_receipt.receipt_id,
                "full_policy_id": full_policy.policy_id,
                "cold_frozen_policy_id": frozen_policy.policy_id,
                "common_terminal_owner_persistence_sha256": owner_sha,
                "parameter_delta_nonzero": parameter_delta_nonzero,
                "parameter_cap_hit": parameter_cap_hit,
                "collection_executed_once": True,
                "arm_trajectory_count": 0,
            }
        )
        completed_root_count += 1

    if environment_scope is None:
        raise RuntimeError("forced common-batch opened no environment scope")
    first_projection_sha = sha256_json(first_projections)
    delivered_noop_count = delivered_action_counts[
        RelationshipAction.NEUTRAL_NOOP.value
    ]
    delivered_nonnoop_count = sum(delivered_action_counts.values()) - delivered_noop_count
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
        unique_command_count=len(command_ids),
        unique_receipt_count=len(receipt_ids),
        unique_exposure_count=len(exposure_ids),
        unique_forecast_count=len(forecast_ids),
        unique_commitment_count=len(commitment_ids),
        unique_settlement_count=len(settlement_ids),
        unique_environment_evidence_ref_count=len(evidence_refs),
        unique_credit_count=len(credit_ids),
        cold_checkpoint_unchanged_count=checkpoint_unchanged_count,
        root_batch_count=completed_root_count,
        unique_batch_count=len(batch_ids),
        apply_receipt_count=len(apply_receipt_ids),
        withhold_receipt_count=len(withhold_receipt_ids),
        full_owner_replay_count=full_owner_replay_count,
        owner_roundtrip_count=owner_roundtrip_count,
        parameter_cap_hit_root_count=parameter_cap_hit_root_count,
        delivered_noop_count=delivered_noop_count,
        delivered_nonnoop_count=delivered_nonnoop_count,
    )
    if failure_reasons:
        status = _FAIL_STATUS
    elif parameter_delta_nonzero_root_count < 1:
        status = _DEGENERATE_STATUS
    else:
        status = _SUCCESS_STATUS
    terminal_reasons = (
        failure_reasons
        if failure_reasons
        else (
            ("parameter_delta_nonzero_root_count_below_one",)
            if status == _DEGENERATE_STATUS
            else ()
        )
    )
    trace_sink.append(
        {
            "schema_version": FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION,
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
            "unique_command_count": len(command_ids),
            "unique_receipt_count": len(receipt_ids),
            "unique_exposure_count": len(exposure_ids),
            "unique_forecast_count": len(forecast_ids),
            "unique_commitment_count": len(commitment_ids),
            "unique_settlement_count": len(settlement_ids),
            "unique_environment_evidence_ref_count": len(evidence_refs),
            "unique_credit_count": len(credit_ids),
            "cold_checkpoint_unchanged_count": checkpoint_unchanged_count,
            "root_batch_count": completed_root_count,
            "unique_batch_count": len(batch_ids),
            "apply_receipt_count": len(apply_receipt_ids),
            "withhold_receipt_count": len(withhold_receipt_ids),
            "full_owner_replay_count": full_owner_replay_count,
            "owner_roundtrip_count": owner_roundtrip_count,
            "parameter_delta_nonzero_root_count": (
                parameter_delta_nonzero_root_count
            ),
            "parameter_cap_hit_root_count": parameter_cap_hit_root_count,
            "scheduled_role_counts": dict(sorted(scheduled_role_counts.items())),
            "delivered_action_counts": dict(sorted(delivered_action_counts.items())),
            "collection_credit_applied_count": 0,
            "collection_gate_update_count": 0,
            "evaluation_decision_count": 0,
            "model_output_count": 0,
            "cuda_execution_count": 0,
            "terminal_failure_reasons": list(terminal_reasons),
            "terminal_status": status,
            "forced_common_collection_batches_materialized": (
                status == _SUCCESS_STATUS
            ),
            "arm_initialization_transition_verified": status == _SUCCESS_STATUS,
            "campaign_protocol_freeze_authorized": status == _SUCCESS_STATUS,
            "campaign_execution_authorized": False,
            "effect_tested": False,
        }
    )
    return _ForcedBatchReplay(
        completed_root_count=completed_root_count,
        onboarding_count=onboarding_count,
        preaction_count=preaction_count,
        postaction_count=postaction_count,
        first_preaction_exact_match_count=first_exact_count,
        first_preaction_projection_sha256=first_projection_sha,
        later_owner_handoff_count=handoff_count,
        owner_writeback_change_count=writeback_change_count,
        selected_branch_resolution_count=branch_resolution_count,
        selected_branch_commitment_match_count=commitment_match_count,
        unique_command_count=len(command_ids),
        unique_receipt_count=len(receipt_ids),
        unique_exposure_count=len(exposure_ids),
        unique_forecast_count=len(forecast_ids),
        unique_commitment_count=len(commitment_ids),
        unique_settlement_count=len(settlement_ids),
        unique_environment_evidence_ref_count=len(evidence_refs),
        unique_credit_count=len(credit_ids),
        cold_checkpoint_unchanged_count=checkpoint_unchanged_count,
        root_batch_count=completed_root_count,
        unique_batch_count=len(batch_ids),
        apply_receipt_count=len(apply_receipt_ids),
        withhold_receipt_count=len(withhold_receipt_ids),
        full_owner_replay_count=full_owner_replay_count,
        owner_roundtrip_count=owner_roundtrip_count,
        parameter_delta_nonzero_root_count=parameter_delta_nonzero_root_count,
        parameter_cap_hit_root_count=parameter_cap_hit_root_count,
        scheduled_role_counts=dict(sorted(scheduled_role_counts.items())),
        delivered_action_counts=dict(sorted(delivered_action_counts.items())),
        terminal_failure_reasons=terminal_reasons,
        terminal_status=status,
    )


def materialize_relationship_product_horizon_forced_common_batch(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    dynamic_root: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    commit = cal._git_commit(implementation_git_commit)
    root = pathlib.Path(output_dir)
    if root.exists():
        raise FileExistsError(f"forced common-batch root is create-only: {root}")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
        scanner_root=pathlib.Path(scanner_root),
        dynamic_root=pathlib.Path(dynamic_root),
    )
    root.mkdir(parents=True, exist_ok=False)
    cal._write_create_only(root / "protocol.json", dependencies.protocol.raw_bytes)
    cal._write_create_only(
        root / _SCHEDULE_FILENAME,
        cal._canonical_bytes(dependencies.schedule_index),
    )
    if cal._read_regular(root / _SCHEDULE_FILENAME) != cal._canonical_bytes(
        dependencies.schedule_index
    ):
        raise RuntimeError("forced schedule index did not close byte-exactly")
    trace_sink = cal._FsyncTraceSink(root / _TRACE_FILENAME)
    state_sink = cal._FsyncTraceSink(root / _STATE_FILENAME)
    transition_sink = cal._FsyncTraceSink(root / _TRANSITION_FILENAME)
    try:
        replay = asyncio.run(
            _run_forced_common_batch(
                dependencies=dependencies,
                trace_sink=trace_sink,
                state_sink=state_sink,
                transition_sink=transition_sink,
            )
        )
    finally:
        trace_sink.close()
        state_sink.close()
        transition_sink.close()
    manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        replay=replay,
        implementation_git_commit=commit,
    )
    cal._write_create_only(root / "manifest.json", cal._canonical_bytes(manifest))
    return manifest


def _parse_jsonl_bytes(
    raw: bytes, *, source: str
) -> tuple[Mapping[str, object], ...]:
    if not raw.endswith(b"\n"):
        raise ValueError(f"{source} must end with a newline")
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        record = cal._parse_json_bytes(
            line, source=f"{source} line {line_number}"
        )
        if line != cal._canonical_bytes(record):
            raise ValueError(f"{source} lines must use canonical bytes")
        records.append(record)
    return tuple(records)


def _parse_jsonl(
    path: pathlib.Path, *, source: str
) -> tuple[Mapping[str, object], ...]:
    return _parse_jsonl_bytes(cal._read_regular(path), source=source)


def _replay_from_terminal(record: Mapping[str, object]) -> _ForcedBatchReplay:
    expected = {
        "schema_version",
        "record_type",
        "completed_root_count",
        "onboarding_count",
        "preaction_count",
        "postaction_count",
        "first_preaction_exact_match_count",
        "first_preaction_projection_sha256",
        "later_owner_handoff_count",
        "owner_writeback_change_count",
        "selected_branch_resolution_count",
        "selected_branch_commitment_match_count",
        "unique_command_count",
        "unique_receipt_count",
        "unique_exposure_count",
        "unique_forecast_count",
        "unique_commitment_count",
        "unique_settlement_count",
        "unique_environment_evidence_ref_count",
        "unique_credit_count",
        "cold_checkpoint_unchanged_count",
        "root_batch_count",
        "unique_batch_count",
        "apply_receipt_count",
        "withhold_receipt_count",
        "full_owner_replay_count",
        "owner_roundtrip_count",
        "parameter_delta_nonzero_root_count",
        "parameter_cap_hit_root_count",
        "scheduled_role_counts",
        "delivered_action_counts",
        "collection_credit_applied_count",
        "collection_gate_update_count",
        "evaluation_decision_count",
        "model_output_count",
        "cuda_execution_count",
        "terminal_failure_reasons",
        "terminal_status",
        "forced_common_collection_batches_materialized",
        "arm_initialization_transition_verified",
        "campaign_protocol_freeze_authorized",
        "campaign_execution_authorized",
        "effect_tested",
    }
    cal._exact_keys(record, expected, "forced common-batch terminal")
    if (
        record["schema_version"] != FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION
        or record["record_type"] != "terminal"
        or record["collection_credit_applied_count"] != 0
        or record["collection_gate_update_count"] != 0
        or record["evaluation_decision_count"] != 0
        or record["model_output_count"] != 0
        or record["cuda_execution_count"] != 0
        or record["campaign_execution_authorized"] is not False
        or record["effect_tested"] is not False
    ):
        raise ValueError("forced common-batch terminal boundary drifted")
    status = cal._text(record["terminal_status"], "terminal_status")
    if status not in {_SUCCESS_STATUS, _FAIL_STATUS, _DEGENERATE_STATUS}:
        raise ValueError("forced common-batch terminal status is unknown")
    success = status == _SUCCESS_STATUS
    if (
        record["forced_common_collection_batches_materialized"] is not success
        or record["arm_initialization_transition_verified"] is not success
        or record["campaign_protocol_freeze_authorized"] is not success
    ):
        raise ValueError("forced common-batch terminal claims drifted")
    failures = cal._list(
        record["terminal_failure_reasons"], "terminal_failure_reasons"
    )
    if any(not isinstance(item, str) or not item for item in failures):
        raise ValueError("terminal failure reasons must be non-empty strings")
    integer_fields = (
        "completed_root_count",
        "onboarding_count",
        "preaction_count",
        "postaction_count",
        "first_preaction_exact_match_count",
        "later_owner_handoff_count",
        "owner_writeback_change_count",
        "selected_branch_resolution_count",
        "selected_branch_commitment_match_count",
        "unique_command_count",
        "unique_receipt_count",
        "unique_exposure_count",
        "unique_forecast_count",
        "unique_commitment_count",
        "unique_settlement_count",
        "unique_environment_evidence_ref_count",
        "unique_credit_count",
        "cold_checkpoint_unchanged_count",
        "root_batch_count",
        "unique_batch_count",
        "apply_receipt_count",
        "withhold_receipt_count",
        "full_owner_replay_count",
        "owner_roundtrip_count",
        "parameter_delta_nonzero_root_count",
        "parameter_cap_hit_root_count",
    )
    values = {
        field: cal._integer(record[field], field) for field in integer_fields
    }
    return _ForcedBatchReplay(
        **values,
        first_preaction_projection_sha256=cal._digest(
            record["first_preaction_projection_sha256"],
            "first_preaction_projection_sha256",
        ),
        scheduled_role_counts=cal._mapping(
            record["scheduled_role_counts"], "scheduled_role_counts"
        ),
        delivered_action_counts=cal._mapping(
            record["delivered_action_counts"], "delivered_action_counts"
        ),
        terminal_failure_reasons=tuple(failures),
        terminal_status=status,
    )


def _validate_persisted_evidence(
    *, root: pathlib.Path, dependencies: _Dependencies
) -> _ForcedBatchReplay:
    if cal._read_regular(root / "protocol.json") != dependencies.protocol.raw_bytes:
        raise ValueError("persisted forced common-batch protocol bytes drifted")
    schedule_raw = cal._read_regular(root / _SCHEDULE_FILENAME)
    if schedule_raw != cal._canonical_bytes(dependencies.schedule_index):
        raise ValueError("persisted forced schedule-index bytes drifted")
    trace = _parse_jsonl(
        root / _TRACE_FILENAME, source="forced common-collection trace"
    )
    states = _parse_jsonl(root / _STATE_FILENAME, source="root owner states")
    transitions = _parse_jsonl(
        root / _TRANSITION_FILENAME, source="root batch transitions"
    )
    type_counts = Counter(item.get("record_type") for item in trace)
    if type_counts != Counter(
        {
            "header": 1,
            "root_start": 112,
            "preaction": 896,
            "postaction": 896,
            "root_terminal": 112,
            "terminal": 1,
        }
    ):
        raise ValueError("forced common-collection trace inventory drifted")
    expected_record_types = ["header"]
    for _ in range(112):
        expected_record_types.append("root_start")
        for _ in range(8):
            expected_record_types.extend(("preaction", "postaction"))
        expected_record_types.append("root_terminal")
    expected_record_types.append("terminal")
    if tuple(item.get("record_type") for item in trace) != tuple(
        expected_record_types
    ):
        raise ValueError("forced common-collection trace order drifted")
    if (
        trace[0].get("schedule_index_id")
        != dependencies.schedule_index["schedule_index_id"]
        or trace[0].get("schedule_index_fsynced_before_first_forecast") is not True
        or trace[0].get("dynamic_trace_file_read_count") != 0
        or trace[0].get("dynamic_natural_outcome_read_count") != 0
        or trace[0].get("dynamic_natural_credit_read_count") != 0
    ):
        raise ValueError("forced common-batch header firewall drifted")
    root_starts = tuple(
        item for item in trace if item.get("record_type") == "root_start"
    )
    preactions = tuple(item for item in trace if item.get("record_type") == "preaction")
    postactions = tuple(
        item for item in trace if item.get("record_type") == "postaction"
    )
    root_terminals = tuple(
        item for item in trace if item.get("record_type") == "root_terminal"
    )
    expected_first = tuple(
        _forced_safe_projection_from_scanner(item)
        for item in dependencies.dynamic_dependencies.expected_first_projections
    )
    observed_first = tuple(
        cal._mapping(item["forced_safe_projection"], "forced_safe_projection")
        for item in preactions
        if item.get("decision_index") == 0
    )
    if len(observed_first) != len(expected_first) or any(
        not _contract_equal(observed, expected)
        for observed, expected in zip(observed_first, expected_first, strict=True)
    ):
        raise ValueError("persisted forced-safe first-preaction seam drifted")
    if any(
        item.get("arm_field_present") is not False
        or item.get("preaction_append_fsynced_before_selected_branch_open")
        is not True
        for item in preactions
    ):
        raise ValueError("forced preaction ordering or arm firewall drifted")
    if any(
        item.get("arm_field_present") is not False
        or item.get("selected_branch_opened_after_current_preaction_fsync")
        is not True
        or item.get("postaction_append_fsynced_before_next_preaction") is not True
        or item.get("credit_applied_online") is not False
        or not _contract_equal(item.get("collection_gate_update_delta"), 0)
        for item in postactions
    ):
        raise ValueError("forced postaction ordering or cold-gate firewall drifted")
    for field in ("decision_id", "command_id", "receipt_id", "exposure_id"):
        if len({item[field] for item in preactions}) != 896:
            raise ValueError(f"forced preaction {field} uniqueness drifted")
    for field in (
        "selected_branch_commitment_id",
        "environment_evidence_ref",
        "settlement_id",
    ):
        if len({item[field] for item in postactions}) != 896:
            raise ValueError(f"forced postaction {field} uniqueness drifted")
    if (
        len(root_starts) != 112
        or len(states) != 112
        or len(transitions) != 112
        or len(root_terminals) != 112
    ):
        raise ValueError("forced common-batch root artifact count drifted")
    if any(
        not _contract_equal(item.get("root_sequence_index"), root_index)
        for root_index, item in enumerate(root_starts)
    ):
        raise ValueError("root start trace order drifted")
    if any(
        not _contract_equal(item.get("root_sequence_index"), root_index)
        for root_index, item in enumerate(states)
    ):
        raise ValueError("root owner-state order drifted")
    if any(
        not _contract_equal(item.get("root_sequence_index"), root_index)
        for root_index, item in enumerate(transitions)
    ):
        raise ValueError("root batch-transition order drifted")
    if any(
        not _contract_equal(item.get("root_sequence_index"), root_index)
        for root_index, item in enumerate(root_terminals)
    ):
        raise ValueError("root terminal trace order drifted")
    seen_batch_ids: set[str] = set()
    seen_credit_ids: set[str] = set()
    seen_apply_receipt_ids: set[str] = set()
    seen_withhold_receipt_ids: set[str] = set()
    seen_forecast_ids: set[str] = set()
    command_ids: set[str] = set()
    receipt_ids: set[str] = set()
    exposure_ids: set[str] = set()
    commitment_ids: set[str] = set()
    settlement_ids: set[str] = set()
    evidence_refs: set[str] = set()
    scheduled_role_counts: Counter[str] = Counter()
    delivered_action_counts: Counter[str] = Counter()
    onboarding_count = 0
    later_owner_handoff_count = 0
    owner_writeback_change_count = 0
    selected_branch_resolution_count = 0
    selected_branch_commitment_match_count = 0
    cold_checkpoint_unchanged_count = 0
    full_owner_replay_count = 0
    owner_roundtrip_count = 0
    parameter_delta_nonzero_root_count = 0
    parameter_cap_hit_root_count = 0
    theta0 = dependencies.dynamic_dependencies.scanner_dependencies.theta0
    cold_policy = (
        dependencies.dynamic_dependencies.scanner_dependencies.frozen_policy
    )
    for root_index, (root_start, state, transition, root_terminal) in enumerate(
        zip(root_starts, states, transitions, root_terminals, strict=True)
    ):
        expected_subject_id = dependencies.public_view.roots[root_index].subject_id
        schedule = dependencies.schedules[root_index]
        root_start_expected = {
            "schema_version",
            "record_type",
            "root_sequence_index",
            "subject_id",
            "schedule_artifact_id",
            "owner_reset",
            "onboarding_appended_once",
            "onboarding_session_count",
            "post_onboarding_persistence_sha256",
        }
        cal._exact_keys(root_start, root_start_expected, f"root start {root_index}")
        root_onboarding_count = cal._integer(
            root_start["onboarding_session_count"],
            f"root start {root_index} onboarding_session_count",
        )
        if (
            root_start["schema_version"]
            != FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION
            or root_start["record_type"] != "root_start"
            or not _contract_equal(root_start["root_sequence_index"], root_index)
            or root_start["subject_id"] != expected_subject_id
            or root_start["schedule_artifact_id"] != schedule.artifact_id
            or root_start["owner_reset"] is not True
            or root_start["onboarding_appended_once"] is not True
            or root_onboarding_count != 4
        ):
            raise ValueError("root start trace lineage drifted")
        onboarding_count += root_onboarding_count
        state_expected = {
            "schema_version",
            "root_sequence_index",
            "subject_id",
            "owner_persistence_sha256",
            "owner_persistence",
            "consumer_interpreted_owner_payload",
        }
        cal._exact_keys(state, state_expected, f"root owner state {root_index}")
        if (
            state["schema_version"]
            != FORCED_COMMON_BATCH_ROOT_STATE_SCHEMA_VERSION
            or not _contract_equal(state["root_sequence_index"], root_index)
            or state["subject_id"] != expected_subject_id
            or state["consumer_interpreted_owner_payload"] is not False
        ):
            raise ValueError("root owner-state envelope drifted")
        owner_raw = cal._canonical_bytes(
            cal._mapping(state["owner_persistence"], "owner_persistence")
        )
        arm_owners = tuple(
            _owner_snapshot_from_payload(
                cal._parse_json_bytes(
                    owner_raw,
                    source=f"persisted {arm_id} common terminal owner",
                )
            )
            for arm_id in ("full", "frozen_theta0", "strict_noop")
        )
        if len(set(map(social_record_store_persistence_sha256, arm_owners))) != 1:
            raise ValueError("root arms did not restore one common owner state")
        owner_roundtrip_count += 1
        owner = arm_owners[0]
        owner_sha = social_record_store_persistence_sha256(owner)
        if owner_sha != state["owner_persistence_sha256"]:
            raise ValueError("root owner-state identity drifted")
        transition_expected = {
            "schema_version",
            "root_sequence_index",
            "subject_id",
            "schedule_artifact_id",
            "common_terminal_owner_persistence_sha256",
            "batch",
            "plan_id",
            "candidate_checkpoint_content_sha256",
            "apply_receipt",
            "withhold_receipt",
            "full_policy_id",
            "full_checkpoint_content_sha256",
            "cold_frozen_policy_id",
            "parameter_delta_nonzero",
            "parameter_cap_hit",
            "arm_bindings",
        }
        cal._exact_keys(
            transition,
            transition_expected,
            f"root batch transition {root_index}",
        )
        if (
            transition["schema_version"]
            != FORCED_COMMON_BATCH_TRANSITION_SCHEMA_VERSION
            or not _contract_equal(
                transition["root_sequence_index"], root_index
            )
            or transition["subject_id"] != expected_subject_id
            or transition["schedule_artifact_id"]
            != dependencies.schedules[root_index].artifact_id
            or transition["common_terminal_owner_persistence_sha256"] != owner_sha
            or transition["cold_frozen_policy_id"] != cold_policy.policy_id
        ):
            raise ValueError("root batch-transition lineage drifted")
        batch = RelationshipActionGateCreditBatch.from_payload(transition["batch"])
        if (
            len(batch.exposures) != 8
            or tuple(item.sequence_index for item in batch.exposures)
            != tuple(range(8))
            or batch.batch_id in seen_batch_ids
        ):
            raise ValueError("root-local credit batch topology drifted")
        root_decision_ids = tuple(
            item.decision_id
            for item in dependencies.public_view.roots[
                root_index
            ].decision_sessions[:8]
        )
        if tuple(
            item.frozen_decision.decision.decision_id for item in batch.exposures
        ) != root_decision_ids:
            raise ValueError("root-local credit batch crossed root boundary")
        root_preactions = preactions[root_index * 8 : (root_index + 1) * 8]
        root_postactions = postactions[root_index * 8 : (root_index + 1) * 8]
        if (
            tuple(item["root_sequence_index"] for item in root_preactions)
            != (root_index,) * 8
            or tuple(item["root_sequence_index"] for item in root_postactions)
            != (root_index,) * 8
            or tuple(item["decision_index"] for item in root_preactions)
            != tuple(range(8))
            or tuple(item["decision_index"] for item in root_postactions)
            != tuple(range(8))
        ):
            raise ValueError("root trace decision order drifted")
        if root_start["post_onboarding_persistence_sha256"] != root_preactions[0][
            "owner_input_persistence_sha256"
        ]:
            raise ValueError("root onboarding state did not join first preaction")
        for decision_index, (exposure, credit, preaction, postaction) in enumerate(
            zip(
                batch.exposures,
                batch.credits,
                root_preactions,
                root_postactions,
                strict=True,
            )
        ):
            schedule_entry = schedule.entries[decision_index]
            safe_projection = cal._mapping(
                preaction["forced_safe_projection"],
                "forced_safe_projection",
            )
            social_pe = cal._mapping(
                postaction["social_prediction_error"],
                "social_prediction_error",
            )
            social_errors = tuple(
                cal._mapping(item, "social prediction error item")
                for item in cal._list(
                    social_pe["errors"], "social_prediction_error.errors"
                )
            )
            matching_credit_errors = tuple(
                item
                for item in social_errors
                if item.get("prediction_id") == credit.prediction_id
            )
            owner_writeback_changed = (
                postaction["owner_preaction_persistence_sha256"]
                != postaction["owner_postaction_persistence_sha256"]
            )
            if decision_index > 0:
                if (
                    root_postactions[decision_index - 1][
                        "owner_postaction_persistence_sha256"
                    ]
                    != preaction["owner_input_persistence_sha256"]
                ):
                    raise ValueError("root owner handoff trace join drifted")
                later_owner_handoff_count += 1
            if (
                preaction["subject_id"] != expected_subject_id
                or postaction["subject_id"] != expected_subject_id
                or not _contract_equal(
                    preaction["global_sequence_index"],
                    root_index * 8 + decision_index,
                )
                or not _contract_equal(
                    postaction["global_sequence_index"],
                    root_index * 8 + decision_index,
                )
                or not _contract_equal(
                    preaction["root_sequence_index"], root_index
                )
                or not _contract_equal(
                    postaction["root_sequence_index"], root_index
                )
                or not _contract_equal(
                    preaction["decision_index"], decision_index
                )
                or not _contract_equal(
                    postaction["decision_index"], decision_index
                )
                or preaction["session_id"]
                != dependencies.public_view.roots[root_index].decision_sessions[
                    decision_index
                ].session_id
                or postaction["session_id"] != preaction["session_id"]
                or preaction["decision_id"] != schedule_entry.decision_id
                or postaction["decision_id"] != schedule_entry.decision_id
                or preaction["schedule_artifact_id"] != schedule.artifact_id
                or preaction["schedule_entry_id"] != schedule_entry.entry_id
                or preaction["forced_action_role"]
                != schedule_entry.forced_action_role.value
                or preaction["exposure_id"] != exposure.exposure_id
                or preaction["forced_action_id"] != exposure.forced_action_id
                or preaction["delivered_action_id"] != exposure.forced_action_id
                or postaction["delivered_action_id"] != exposure.forced_action_id
                or postaction["environment_selected_action_id"]
                != exposure.forced_action_id
                or safe_projection["forecast_sha256"]
                != sha256_json(
                    preference_action_forecast_to_payload(exposure.forecast)
                )
                or safe_projection["frozen_decision_sha256"]
                != sha256_json(exposure.frozen_decision.to_payload())
                or safe_projection["owner_input_persistence_sha256"]
                != preaction["owner_input_persistence_sha256"]
                or safe_projection["owner_output_persistence_sha256"]
                != postaction["owner_preaction_persistence_sha256"]
                or postaction["credit"] != cal._credit_payload(credit)
                or len(matching_credit_errors) != 1
                or credit.source_event
                != f"social_pe:{matching_credit_errors[0]['error_id']}"
                or postaction["settlement_id"] == ""
                or postaction["owner_writeback_changed_persistence"]
                is not owner_writeback_changed
                or postaction["cold_checkpoint_content_sha256"]
                != cold_policy.checkpoint.content_sha256
            ):
                raise ValueError("root batch exposure/credit trace join drifted")
            scheduled_role_counts[preaction["forced_action_role"]] += 1
            delivered_action_counts[postaction["delivered_action_id"]] += 1
            command_ids.add(preaction["command_id"])
            receipt_ids.add(preaction["receipt_id"])
            exposure_ids.add(preaction["exposure_id"])
            seen_forecast_ids.add(exposure.forecast.forecast_id)
            commitment_ids.add(postaction["selected_branch_commitment_id"])
            settlement_ids.add(postaction["settlement_id"])
            evidence_refs.add(postaction["environment_evidence_ref"])
            owner_writeback_change_count += int(owner_writeback_changed)
            selected_branch_resolution_count += 1
            selected_branch_commitment_match_count += int(
                postaction["environment_selected_action_id"]
                == preaction["delivered_action_id"]
                and bool(postaction["selected_branch_commitment_id"])
            )
            cold_checkpoint_unchanged_count += int(
                postaction["cold_checkpoint_content_sha256"]
                == cold_policy.checkpoint.content_sha256
                and _contract_equal(postaction["collection_gate_update_delta"], 0)
            )
        if state["owner_persistence_sha256"] != root_postactions[-1][
            "owner_postaction_persistence_sha256"
        ]:
            raise ValueError("root terminal owner did not join final postaction")
        root_credit_ids = tuple(item.record_id for item in batch.credits)
        if any(item in seen_credit_ids for item in root_credit_ids):
            raise ValueError("credit was reused across root-local batches")
        seen_batch_ids.add(batch.batch_id)
        seen_credit_ids.update(root_credit_ids)
        apply_receipt = RelationshipActionGateBatchReceipt.from_payload(
            transition["apply_receipt"]
        )
        withhold_receipt = RelationshipActionGateBatchReceipt.from_payload(
            transition["withhold_receipt"]
        )
        full_gate = RelationshipActionGate.from_theta0(
            theta0, random_seed=cold_policy.random_seed
        )
        frozen_gate = RelationshipActionGate.from_theta0(
            theta0, random_seed=cold_policy.random_seed
        )
        strict_gate = RelationshipActionGate.from_theta0(
            theta0, random_seed=cold_policy.random_seed
        )
        full_plan = full_gate.plan_credit_batch(batch)
        frozen_plan = frozen_gate.plan_credit_batch(batch)
        strict_plan = strict_gate.plan_credit_batch(batch)
        if full_plan != frozen_plan or full_plan != strict_plan:
            raise ValueError("persisted root arms do not share one plan")
        replayed_apply = full_gate.commit_credit_batch(
            full_plan, disposition=RelationshipActionGateBatchDisposition.APPLY
        )
        replayed_frozen = frozen_gate.commit_credit_batch(
            frozen_plan,
            disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
        )
        replayed_strict = strict_gate.commit_credit_batch(
            strict_plan,
            disposition=RelationshipActionGateBatchDisposition.WITHHOLD,
        )
        full_policy = full_gate.freeze_for_evaluation()
        expected_parameter_delta_nonzero = any(
            value != 0.0
            for value in (*apply_receipt.weight_delta, apply_receipt.bias_delta)
        )
        full_checkpoint = full_gate.export_checkpoint()
        expected_parameter_cap_hit = any(
            abs(value) >= theta0.max_abs_parameter
            for value in (*full_checkpoint.weights, full_checkpoint.bias)
        )
        if (
            apply_receipt != replayed_apply
            or withhold_receipt != replayed_frozen
            or withhold_receipt != replayed_strict
            or transition["plan_id"] != full_plan.plan_id
            or transition["candidate_checkpoint_content_sha256"]
            != full_plan.candidate_checkpoint.content_sha256
            or transition["full_policy_id"] != full_policy.policy_id
            or transition["full_checkpoint_content_sha256"]
            != full_checkpoint.content_sha256
            or transition["parameter_delta_nonzero"]
            is not expected_parameter_delta_nonzero
            or transition["parameter_cap_hit"] is not expected_parameter_cap_hit
        ):
            raise ValueError("persisted root batch receipt replay drifted")
        parameter_delta_nonzero_root_count += int(
            expected_parameter_delta_nonzero
        )
        parameter_cap_hit_root_count += int(expected_parameter_cap_hit)
        seen_apply_receipt_ids.add(apply_receipt.receipt_id)
        seen_withhold_receipt_ids.add(withhold_receipt.receipt_id)
        owner_replay = RelationshipActionGate.from_applied_credit_batch(
            theta0,
            batch=batch,
            receipt=apply_receipt,
            random_seed=cold_policy.random_seed,
        )
        if owner_replay.freeze_for_evaluation() != full_policy:
            raise ValueError("persisted full policy owner replay drifted")
        full_owner_replay_count += 1
        bindings = cal._mapping(transition["arm_bindings"], "arm_bindings")
        cal._exact_keys(
            bindings, {"full", "frozen_theta0", "strict_noop"}, "arm_bindings"
        )
        binding_fields = {
            "batch_id",
            "receipt_id",
            "policy_id",
            "executor_disposition",
            "online_gate_update_count",
        }
        for arm_id in ("full", "frozen_theta0", "strict_noop"):
            cal._exact_keys(
                cal._mapping(bindings[arm_id], f"arm_bindings.{arm_id}"),
                binding_fields,
                f"arm_bindings.{arm_id}",
            )
        if (
            bindings["full"]["batch_id"] != batch.batch_id
            or bindings["full"]["receipt_id"] != apply_receipt.receipt_id
            or bindings["full"]["policy_id"] != full_policy.policy_id
            or bindings["full"]["executor_disposition"] != "apply_candidate"
            or bindings["frozen_theta0"]["receipt_id"]
            != withhold_receipt.receipt_id
            or bindings["strict_noop"]["receipt_id"]
            != withhold_receipt.receipt_id
            or bindings["frozen_theta0"]["batch_id"] != batch.batch_id
            or bindings["strict_noop"]["batch_id"] != batch.batch_id
            or bindings["frozen_theta0"]["policy_id"] != cold_policy.policy_id
            or bindings["strict_noop"]["policy_id"] != cold_policy.policy_id
            or bindings["frozen_theta0"]["executor_disposition"]
            != "apply_candidate"
            or bindings["strict_noop"]["executor_disposition"]
            != "force_strict_noop"
            or any(
                bindings[arm_id]["online_gate_update_count"] != 0
                or isinstance(
                    bindings[arm_id]["online_gate_update_count"], bool
                )
                for arm_id in ("full", "frozen_theta0", "strict_noop")
            )
        ):
            raise ValueError("persisted arm binding drifted")
        root_terminal_expected = {
            "schema_version",
            "record_type",
            "root_sequence_index",
            "subject_id",
            "schedule_artifact_id",
            "batch_id",
            "batch_credit_count",
            "apply_receipt_id",
            "withhold_receipt_id",
            "full_policy_id",
            "cold_frozen_policy_id",
            "common_terminal_owner_persistence_sha256",
            "parameter_delta_nonzero",
            "parameter_cap_hit",
            "collection_executed_once",
            "arm_trajectory_count",
        }
        cal._exact_keys(
            root_terminal,
            root_terminal_expected,
            f"root terminal {root_index}",
        )
        if (
            root_terminal["schema_version"]
            != FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION
            or root_terminal["record_type"] != "root_terminal"
            or not _contract_equal(
                root_terminal["root_sequence_index"], root_index
            )
            or root_terminal["subject_id"] != expected_subject_id
            or root_terminal["schedule_artifact_id"]
            != dependencies.schedules[root_index].artifact_id
            or root_terminal["batch_id"] != batch.batch_id
            or root_terminal["apply_receipt_id"] != apply_receipt.receipt_id
            or root_terminal["withhold_receipt_id"] != withhold_receipt.receipt_id
            or root_terminal["full_policy_id"] != full_policy.policy_id
            or root_terminal["cold_frozen_policy_id"] != cold_policy.policy_id
            or root_terminal["common_terminal_owner_persistence_sha256"]
            != owner_sha
            or not _contract_equal(root_terminal["batch_credit_count"], 8)
            or root_terminal["parameter_delta_nonzero"]
            is not expected_parameter_delta_nonzero
            or root_terminal["parameter_cap_hit"] is not expected_parameter_cap_hit
            or root_terminal["collection_executed_once"] is not True
            or not _contract_equal(root_terminal["arm_trajectory_count"], 0)
        ):
            raise ValueError("root terminal trace join drifted")
    if (
        len(seen_batch_ids) != 112
        or len(seen_credit_ids) != 896
        or len(seen_apply_receipt_ids) != 112
        or len(seen_withhold_receipt_ids) != 112
    ):
        raise ValueError("persisted root-local batch inventory drifted")
    delivered_noop_count = delivered_action_counts[
        RelationshipAction.NEUTRAL_NOOP.value
    ]
    delivered_nonnoop_count = (
        sum(delivered_action_counts.values()) - delivered_noop_count
    )
    terminal_failure_reasons = _terminal_failure_reasons(
        completed_root_count=len(root_terminals),
        onboarding_count=onboarding_count,
        preaction_count=len(preactions),
        postaction_count=len(postactions),
        first_preaction_exact_match_count=len(observed_first),
        later_owner_handoff_count=later_owner_handoff_count,
        owner_writeback_change_count=owner_writeback_change_count,
        selected_branch_resolution_count=selected_branch_resolution_count,
        selected_branch_commitment_match_count=(
            selected_branch_commitment_match_count
        ),
        unique_command_count=len(command_ids),
        unique_receipt_count=len(receipt_ids),
        unique_exposure_count=len(exposure_ids),
        unique_forecast_count=len(seen_forecast_ids),
        unique_commitment_count=len(commitment_ids),
        unique_settlement_count=len(settlement_ids),
        unique_environment_evidence_ref_count=len(evidence_refs),
        unique_credit_count=len(seen_credit_ids),
        cold_checkpoint_unchanged_count=cold_checkpoint_unchanged_count,
        root_batch_count=len(transitions),
        unique_batch_count=len(seen_batch_ids),
        apply_receipt_count=len(seen_apply_receipt_ids),
        withhold_receipt_count=len(seen_withhold_receipt_ids),
        full_owner_replay_count=full_owner_replay_count,
        owner_roundtrip_count=owner_roundtrip_count,
        parameter_cap_hit_root_count=parameter_cap_hit_root_count,
        delivered_noop_count=delivered_noop_count,
        delivered_nonnoop_count=delivered_nonnoop_count,
    )
    if terminal_failure_reasons:
        terminal_status = _FAIL_STATUS
    elif parameter_delta_nonzero_root_count < 1:
        terminal_status = _DEGENERATE_STATUS
    else:
        terminal_status = _SUCCESS_STATUS
    reported_reasons = (
        terminal_failure_reasons
        if terminal_failure_reasons
        else (
            ("parameter_delta_nonzero_root_count_below_one",)
            if terminal_status == _DEGENERATE_STATUS
            else ()
        )
    )
    derived_replay = _ForcedBatchReplay(
        completed_root_count=len(root_terminals),
        onboarding_count=onboarding_count,
        preaction_count=len(preactions),
        postaction_count=len(postactions),
        first_preaction_exact_match_count=len(observed_first),
        first_preaction_projection_sha256=sha256_json(observed_first),
        later_owner_handoff_count=later_owner_handoff_count,
        owner_writeback_change_count=owner_writeback_change_count,
        selected_branch_resolution_count=selected_branch_resolution_count,
        selected_branch_commitment_match_count=(
            selected_branch_commitment_match_count
        ),
        unique_command_count=len(command_ids),
        unique_receipt_count=len(receipt_ids),
        unique_exposure_count=len(exposure_ids),
        unique_forecast_count=len(seen_forecast_ids),
        unique_commitment_count=len(commitment_ids),
        unique_settlement_count=len(settlement_ids),
        unique_environment_evidence_ref_count=len(evidence_refs),
        unique_credit_count=len(seen_credit_ids),
        cold_checkpoint_unchanged_count=cold_checkpoint_unchanged_count,
        root_batch_count=len(transitions),
        unique_batch_count=len(seen_batch_ids),
        apply_receipt_count=len(seen_apply_receipt_ids),
        withhold_receipt_count=len(seen_withhold_receipt_ids),
        full_owner_replay_count=full_owner_replay_count,
        owner_roundtrip_count=owner_roundtrip_count,
        parameter_delta_nonzero_root_count=parameter_delta_nonzero_root_count,
        parameter_cap_hit_root_count=parameter_cap_hit_root_count,
        scheduled_role_counts=dict(sorted(scheduled_role_counts.items())),
        delivered_action_counts=dict(sorted(delivered_action_counts.items())),
        terminal_failure_reasons=reported_reasons,
        terminal_status=terminal_status,
    )
    terminal_replay = _replay_from_terminal(trace[-1])
    if terminal_replay != derived_replay:
        raise ValueError("forced common-batch terminal self-report drifted")
    return derived_replay


def _validate_forced_common_batch_artifact(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    dynamic_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> tuple[Mapping[str, object], _Dependencies, bytes]:
    external_protocol = cal._digest(expected_protocol_id, "expected_protocol_id")
    external_artifact = cal._digest(expected_artifact_id, "expected_artifact_id")
    root = pathlib.Path(output_dir)
    manifest_raw = cal._read_regular(root / "manifest.json")
    manifest = cal._parse_json_bytes(
        manifest_raw, source="forced common-batch manifest"
    )
    if manifest_raw != cal._canonical_bytes(manifest):
        raise ValueError("forced common-batch manifest must use canonical bytes")
    if manifest["protocol_id"] != external_protocol:
        raise ValueError("external forced common-batch protocol ID drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("external forced common-batch artifact ID drifted")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
        scanner_root=pathlib.Path(scanner_root),
        dynamic_root=pathlib.Path(dynamic_root),
    )
    if dependencies.protocol.protocol_id != external_protocol:
        raise ValueError("packaged forced common-batch protocol ID drifted")
    if cal._regular_file_inventory(root) != _OUTPUT_FILES:
        raise ValueError("forced common-batch output inventory drifted")
    replay = _validate_persisted_evidence(root=root, dependencies=dependencies)
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
        raise ValueError("forced common-batch manifest content drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("forced common-batch artifact identity drifted")
    return manifest, dependencies, manifest_raw


def validate_relationship_product_horizon_forced_common_batch(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    dynamic_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    """Validate one existing artifact without mutating any evidence bytes."""

    manifest, _, _ = _validate_forced_common_batch_artifact(
        source_v4_admission_root=source_v4_admission_root,
        reader_root=reader_root,
        theta0_v2_root=theta0_v2_root,
        scanner_root=scanner_root,
        dynamic_root=dynamic_root,
        output_dir=output_dir,
        expected_protocol_id=expected_protocol_id,
        expected_artifact_id=expected_artifact_id,
    )
    return manifest


def _campaign_lineage_projection(
    *,
    dependencies: _Dependencies,
    manifest: Mapping[str, object],
) -> tuple[RelationshipProductHorizonCampaignLineageEntry, ...]:
    runtime = cal._mapping(dependencies.protocol.payload["runtime_inputs"], "runtime")
    dynamic_pin = cal._mapping(
        dependencies.protocol.payload["upstream_dynamic_gate"],
        "upstream_dynamic_gate",
    )
    dynamic_protocol = dependencies.dynamic_dependencies.protocol.payload
    scanner_pin = cal._mapping(
        dynamic_protocol["upstream_scanner"],
        "upstream_scanner",
    )
    source_pin = cal._mapping(
        dynamic_protocol["source_v4_environment"],
        "source_v4_environment",
    )
    files = _file_entries(manifest, source="forced common-batch")
    values = {
        "forced_implementation_git_commit": manifest[
            "implementation_git_commit"
        ],
        "forced_schedule_index_id": manifest["schedule_index_id"],
        "forced_schedule_index_raw_sha256": files[_SCHEDULE_FILENAME][
            "raw_sha256"
        ],
        "forced_collection_trace_raw_sha256": files[_TRACE_FILENAME][
            "raw_sha256"
        ],
        "forced_root_owner_states_raw_sha256": files[_STATE_FILENAME][
            "raw_sha256"
        ],
        "forced_root_batch_transitions_raw_sha256": files[_TRANSITION_FILENAME][
            "raw_sha256"
        ],
        "dynamic_protocol_id": dynamic_pin["protocol_id"],
        "dynamic_protocol_raw_sha256": dynamic_pin["protocol_raw_sha256"],
        "dynamic_artifact_id": dynamic_pin["artifact_id"],
        "dynamic_manifest_raw_sha256": dynamic_pin["manifest_raw_sha256"],
        "scanner_protocol_id": scanner_pin["protocol_id"],
        "scanner_protocol_raw_sha256": scanner_pin["protocol_raw_sha256"],
        "scanner_artifact_id": scanner_pin["artifact_id"],
        "scanner_manifest_raw_sha256": scanner_pin["manifest_raw_sha256"],
        "source_v4_admission_protocol_id": runtime[
            "source_v4_admission_protocol_id"
        ],
        "source_v4_admission_artifact_id": runtime[
            "source_v4_admission_artifact_id"
        ],
        "source_v4_admission_manifest_raw_sha256": source_pin[
            "admission_manifest_raw_sha256"
        ],
        "source_v4_source_protocol_id": source_pin["source_protocol_id"],
        "source_v4_source_protocol_raw_sha256": source_pin[
            "source_protocol_raw_sha256"
        ],
        "source_v4_public_plan_sha256": runtime["source_v4_public_plan_sha256"],
        "source_v4_public_plan_raw_sha256": runtime[
            "source_v4_public_plan_raw_sha256"
        ],
        "source_v4_sealed_bundle_sha256": runtime[
            "source_v4_sealed_bundle_sha256"
        ],
        "source_v4_sealed_evaluator_raw_sha256": source_pin[
            "sealed_evaluator_raw_sha256"
        ],
        "source_v4_commitment_index_raw_sha256": runtime[
            "source_v4_commitment_index_raw_sha256"
        ],
        "development_reader_package_artifact_id": runtime[
            "development_reader_package_artifact_id"
        ],
        "development_reader_manifest_raw_sha256": runtime[
            "development_reader_manifest_raw_sha256"
        ],
        "embedding_table_artifact_id": runtime["embedding_table_artifact_id"],
        "embedding_table_raw_sha256": runtime["embedding_table_raw_sha256"],
        "reader_artifact_id": runtime["reader_artifact_id"],
        "reader_artifact_raw_sha256": runtime["reader_artifact_raw_sha256"],
        "theta0_v2_bootstrap_protocol_id": runtime[
            "theta0_v2_bootstrap_protocol_id"
        ],
        "theta0_v2_bootstrap_artifact_id": runtime[
            "theta0_v2_bootstrap_artifact_id"
        ],
        "theta0_v2_manifest_raw_sha256": runtime[
            "theta0_v2_manifest_raw_sha256"
        ],
        "theta0_v2_artifact_id": runtime["theta0_v2_artifact_id"],
        "theta0_v2_artifact_raw_sha256": runtime[
            "theta0_v2_artifact_raw_sha256"
        ],
        "cold_checkpoint_content_sha256": runtime[
            "cold_checkpoint_content_sha256"
        ],
        "cold_frozen_policy_id": runtime["cold_frozen_policy_id"],
        "cold_random_seed": runtime["cold_random_seed"],
    }
    if any(not isinstance(value, str) or value == "" for value in values.values()):
        raise ValueError("forced campaign lineage contains a non-text identity")
    return tuple(
        RelationshipProductHorizonCampaignLineageEntry(name=name, value=value)
        for name, value in sorted(values.items())
    )


def load_relationship_product_horizon_forced_campaign_inputs(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    dynamic_root: pathlib.Path,
    forced_common_batch_root: pathlib.Path,
    expected_forced_protocol_id: str,
    expected_forced_artifact_id: str,
) -> RelationshipProductHorizonForcedCampaignInputs:
    """Load typed campaign inputs only after external-ID evidence replay."""

    manifest, dependencies, manifest_raw = _validate_forced_common_batch_artifact(
        source_v4_admission_root=source_v4_admission_root,
        reader_root=reader_root,
        theta0_v2_root=theta0_v2_root,
        scanner_root=scanner_root,
        dynamic_root=dynamic_root,
        output_dir=forced_common_batch_root,
        expected_protocol_id=expected_forced_protocol_id,
        expected_artifact_id=expected_forced_artifact_id,
    )
    if (
        manifest["status"] != _SUCCESS_STATUS
        or manifest["claims"]["campaign_protocol_freeze_authorized"] is not True
        or manifest["claims"]["campaign_execution_authorized"] is not False
    ):
        raise ValueError("forced common-batch artifact does not authorize campaign freeze")

    root = pathlib.Path(forced_common_batch_root)
    files = _file_entries(manifest, source="forced common-batch")
    states_raw = cal._require_raw_sha(
        root / _STATE_FILENAME,
        files[_STATE_FILENAME]["raw_sha256"],
        "root owner states",
    )
    transitions_raw = cal._require_raw_sha(
        root / _TRANSITION_FILENAME,
        files[_TRANSITION_FILENAME]["raw_sha256"],
        source="root batch transitions",
    )
    states = _parse_jsonl_bytes(states_raw, source="root owner states")
    transitions = _parse_jsonl_bytes(
        transitions_raw,
        source="root batch transitions",
    )
    if len(states) != 112 or len(transitions) != 112:
        raise ValueError("forced campaign input root inventory drifted")

    scanner = dependencies.dynamic_dependencies.scanner_dependencies
    theta0 = scanner.theta0
    cold_policy = scanner.frozen_policy
    campaign_roots = []
    for root_index, (public_root, state, transition) in enumerate(
        zip(dependencies.public_view.roots, states, transitions, strict=True)
    ):
        if (
            state["root_sequence_index"] != root_index
            or transition["root_sequence_index"] != root_index
            or state["subject_id"] != public_root.subject_id
            or transition["subject_id"] != public_root.subject_id
        ):
            raise ValueError("forced campaign root order or subject join drifted")
        owner_payload = cal._mapping(
            state["owner_persistence"],
            f"root owner state {root_index}",
        )
        owner_bytes = cal._canonical_bytes(owner_payload)
        batch = RelationshipActionGateCreditBatch.from_payload(transition["batch"])
        apply_receipt = RelationshipActionGateBatchReceipt.from_payload(
            transition["apply_receipt"]
        )
        withhold_receipt = RelationshipActionGateBatchReceipt.from_payload(
            transition["withhold_receipt"]
        )
        bindings = cal._mapping(
            transition["arm_bindings"],
            f"root transition {root_index} arm bindings",
        )
        if (
            apply_receipt.batch_id != batch.batch_id
            or withhold_receipt.batch_id != batch.batch_id
            or apply_receipt.disposition
            is not RelationshipActionGateBatchDisposition.APPLY
            or withhold_receipt.disposition
            is not RelationshipActionGateBatchDisposition.WITHHOLD
            or bindings["full"]["receipt_id"] != apply_receipt.receipt_id
            or bindings["frozen_theta0"]["receipt_id"]
            != withhold_receipt.receipt_id
            or bindings["strict_noop"]["receipt_id"]
            != withhold_receipt.receipt_id
        ):
            raise ValueError("forced campaign root batch binding drifted")
        campaign_root = RelationshipProductHorizonForcedCampaignRootInput(
            root_sequence_index=root_index,
            public_root=public_root,
            schedule_artifact_id=cal._text(
                transition["schedule_artifact_id"],
                "schedule_artifact_id",
            ),
            transition_raw_sha256=cal._sha256_bytes(
                cal._canonical_bytes(transition)
            ),
            common_terminal_owner_persistence_sha256=cal._digest(
                state["owner_persistence_sha256"],
                "owner_persistence_sha256",
            ),
            batch=batch,
            apply_receipt=apply_receipt,
            withhold_receipt=withhold_receipt,
            full_policy_id=cal._text(
                transition["full_policy_id"],
                "full_policy_id",
            ),
            full_checkpoint_content_sha256=cal._digest(
                transition["full_checkpoint_content_sha256"],
                "full_checkpoint_content_sha256",
            ),
            cold_frozen_policy_id=cal._text(
                transition["cold_frozen_policy_id"],
                "cold_frozen_policy_id",
            ),
            _owner_persistence_bytes=owner_bytes,
            _theta0=theta0,
            _cold_random_seed=cold_policy.random_seed,
            _embedding_table=scanner.embedding_table,
            _reader_artifact=scanner.reader_artifact,
        )
        # Rebuild the complete typed arm mapping now; the consumer receives no
        # unchecked persisted policy payload and owns no arm-binding logic.
        full, frozen, strict = campaign_root.fresh_arm_initializations()
        owner_full = full.owner_persistence_snapshot
        owner_frozen = frozen.owner_persistence_snapshot
        owner_strict = strict.owner_persistence_snapshot
        if (
            owner_full is owner_frozen
            or owner_full is owner_strict
            or owner_frozen is owner_strict
            or owner_full != owner_frozen
            or owner_full != owner_strict
        ):
            raise ValueError("forced campaign root owners are not fresh exact copies")
        if (
            frozen.frozen_policy is strict.frozen_policy
            or frozen.frozen_policy != strict.frozen_policy
            or full.frozen_policy.checkpoint.update_count != 8
            or frozen.frozen_policy.checkpoint.update_count != 0
            or strict.frozen_policy.checkpoint.update_count != 0
            or full.forecast_runtime is frozen.forecast_runtime
            or full.forecast_runtime is strict.forecast_runtime
            or frozen.forecast_runtime is strict.forecast_runtime
        ):
            raise ValueError("forced campaign arm initialization drifted")
        campaign_roots.append(campaign_root)

    lineage = _campaign_lineage_projection(
        dependencies=dependencies,
        manifest=manifest,
    )
    lineage_id = sha256_json(
        {
            "schema_version": FORCED_CAMPAIGN_INPUT_LINEAGE_SCHEMA_VERSION,
            "entries": [
                {"name": item.name, "value": item.value} for item in lineage
            ],
        }
    )
    return RelationshipProductHorizonForcedCampaignInputs(
        forced_protocol_id=dependencies.protocol.protocol_id,
        forced_protocol_raw_sha256=dependencies.protocol.raw_sha256,
        forced_artifact_id=cal._digest(manifest["artifact_id"], "artifact_id"),
        forced_manifest_raw_sha256=cal._sha256_bytes(manifest_raw),
        public_plan_sha256=dependencies.public_view.public_plan_sha256,
        lineage_schema_version=FORCED_CAMPAIGN_INPUT_LINEAGE_SCHEMA_VERSION,
        lineage_id=lineage_id,
        lineage=lineage,
        public_view=dependencies.public_view,
        roots=tuple(campaign_roots),
    )


def _build_manifest(
    *,
    root: pathlib.Path,
    dependencies: _Dependencies,
    replay: _ForcedBatchReplay,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    files = []
    for relative in (
        "protocol.json",
        _SCHEDULE_FILENAME,
        _TRACE_FILENAME,
        _STATE_FILENAME,
        _TRANSITION_FILENAME,
    ):
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
        "forced_common_collection_batches_materialized": success,
        "arm_initialization_transition_verified": success,
        "campaign_protocol_freeze_authorized": success,
        "evaluation_execution_authorized": False,
        "campaign_execution_authorized": False,
        "effect_tested": False,
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
    runtime = cal._mapping(dependencies.protocol.payload["runtime_inputs"], "runtime")
    upstream = cal._mapping(
        dependencies.protocol.payload["upstream_dynamic_gate"], "upstream"
    )
    core = {
        "schema_version": FORCED_COMMON_BATCH_MANIFEST_SCHEMA_VERSION,
        "protocol_id": dependencies.protocol.protocol_id,
        "protocol_raw_sha256": dependencies.protocol.raw_sha256,
        "implementation_git_commit": implementation_git_commit,
        "upstream_dynamic_gate_protocol_id": upstream["protocol_id"],
        "upstream_dynamic_gate_artifact_id": upstream["artifact_id"],
        "upstream_dynamic_gate_manifest_raw_sha256": upstream[
            "manifest_raw_sha256"
        ],
        "dynamic_manifest_file_read_count": 1,
        "dynamic_trace_file_read_count": 0,
        "dynamic_natural_outcome_read_count": 0,
        "dynamic_natural_credit_read_count": 0,
        "source_v4_admission_artifact_id": runtime[
            "source_v4_admission_artifact_id"
        ],
        "source_v4_public_plan_sha256": runtime["source_v4_public_plan_sha256"],
        "development_reader_package_artifact_id": runtime[
            "development_reader_package_artifact_id"
        ],
        "reader_artifact_id": runtime["reader_artifact_id"],
        "condition_reader_qualified": False,
        "theta0_v2_artifact_id": runtime["theta0_v2_artifact_id"],
        "cold_checkpoint_content_sha256": runtime[
            "cold_checkpoint_content_sha256"
        ],
        "cold_frozen_policy_id": runtime["cold_frozen_policy_id"],
        "schedule_index_id": dependencies.schedule_index["schedule_index_id"],
        "schedule_artifact_count": 112,
        "schedule_entry_count": 896,
        "completed_root_count": replay.completed_root_count,
        "onboarding_count": replay.onboarding_count,
        "preaction_count": replay.preaction_count,
        "postaction_count": replay.postaction_count,
        "first_preaction_exact_match_count": (
            replay.first_preaction_exact_match_count
        ),
        "first_preaction_projection_schema_version": (
            FORCED_SAFE_FIRST_PROJECTION_SCHEMA_VERSION
        ),
        "first_preaction_projection_fields": list(
            FORCED_SAFE_FIRST_PROJECTION_FIELDS
        ),
        "first_preaction_projection_sha256": (
            replay.first_preaction_projection_sha256
        ),
        "later_owner_handoff_count": replay.later_owner_handoff_count,
        "owner_writeback_change_count": replay.owner_writeback_change_count,
        "selected_branch_resolution_count": replay.selected_branch_resolution_count,
        "selected_branch_commitment_match_count": (
            replay.selected_branch_commitment_match_count
        ),
        "unique_command_count": replay.unique_command_count,
        "unique_receipt_count": replay.unique_receipt_count,
        "unique_exposure_count": replay.unique_exposure_count,
        "unique_forecast_count": replay.unique_forecast_count,
        "unique_commitment_count": replay.unique_commitment_count,
        "unique_settlement_count": replay.unique_settlement_count,
        "unique_environment_evidence_ref_count": (
            replay.unique_environment_evidence_ref_count
        ),
        "unique_credit_count": replay.unique_credit_count,
        "cold_checkpoint_unchanged_count": replay.cold_checkpoint_unchanged_count,
        "root_batch_count": replay.root_batch_count,
        "unique_batch_count": replay.unique_batch_count,
        "credit_count_per_batch": 8,
        "apply_receipt_count": replay.apply_receipt_count,
        "withhold_receipt_count": replay.withhold_receipt_count,
        "full_owner_replay_count": replay.full_owner_replay_count,
        "owner_roundtrip_count": replay.owner_roundtrip_count,
        "parameter_delta_nonzero_root_count": (
            replay.parameter_delta_nonzero_root_count
        ),
        "parameter_cap_hit_root_count": replay.parameter_cap_hit_root_count,
        "scheduled_role_counts": replay.scheduled_role_counts,
        "delivered_action_counts": replay.delivered_action_counts,
        "collection_credit_applied_count": 0,
        "collection_gate_update_count": 0,
        "evaluation_decision_count": 0,
        "evaluation_or_judge_feedback_count": 0,
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
    "FORCED_COMMON_BATCH_MANIFEST_SCHEMA_VERSION",
    "FORCED_COMMON_BATCH_PROTOCOL_SCHEMA_VERSION",
    "FORCED_COMMON_BATCH_TRACE_SCHEMA_VERSION",
    "FORCED_CAMPAIGN_INPUT_LINEAGE_SCHEMA_VERSION",
    "FORCED_SAFE_FIRST_PROJECTION_FIELDS",
    "FORCED_SAFE_FIRST_PROJECTION_SCHEMA_VERSION",
    "RelationshipProductHorizonCampaignArm",
    "RelationshipProductHorizonCampaignArmInitialization",
    "RelationshipProductHorizonCampaignLineageEntry",
    "RelationshipProductHorizonForcedCampaignInputs",
    "RelationshipProductHorizonForcedCampaignRootInput",
    "RelationshipProductHorizonForcedCommonBatchProtocol",
    "load_relationship_product_horizon_forced_common_batch_protocol",
    "load_relationship_product_horizon_forced_campaign_inputs",
    "materialize_relationship_product_horizon_forced_common_batch",
    "relationship_product_horizon_forced_common_batch_protocol_path",
    "validate_relationship_product_horizon_forced_common_batch",
]
