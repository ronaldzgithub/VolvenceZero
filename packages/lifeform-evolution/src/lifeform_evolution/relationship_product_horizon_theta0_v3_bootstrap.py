"""Materialize one development-only gate-v2 federated Product Horizon theta0.

The owner freezes and durably publishes the complete 112-child assignment
parent before it permits onboarding, forecast publication, or selected-outcome
resolution.  Every child remains collection provenance only: the gate is
transitioned exactly once at the federated parent and then condensed into one
cold learned-theta0 artifact.  Source-v4 and the development reader are already
spent adaptive inputs, so this workflow cannot establish an ability effect.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
import os
import pathlib
import subprocess
from typing import Mapping, Protocol

from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    HorizonPublicDecisionSession,
    HorizonPublicRoot,
    RelationshipProductHorizonPublicView,
    build_relationship_product_horizon_evaluator_bundle,
    load_relationship_product_horizon_source_protocol,
)
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductOnboardingInput,
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    RelationshipProductV2CollectionSegment,
    RelationshipProductV2ForcedCollectionAuthorization,
    RelationshipProductV2FederatedCollectedCreditBatch,
    RelationshipProductV2FederatedMatchedGateTransitions,
    RelationshipProductV2SegmentedCollectedCreditBatch,
    append_relationship_product_onboarding,
    build_relationship_product_v2_federated_collected_credit_batch,
    build_relationship_product_v2_segmented_collected_credit_batch,
    commit_relationship_product_v2_federated_matched_gate_transitions,
    prepare_relationship_product_v2_forced_collection_preaction,
    settle_relationship_product_v2_forced_collection,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate_v2 import (
    RelationshipActionGateV2,
    RelationshipActionGateV2Artifact,
    RelationshipActionGateV2AssignmentReceipt,
    RelationshipActionGateV2AssignmentRole,
    RelationshipActionGateV2AssignmentScheduleArtifact,
    RelationshipActionGateV2AssignmentScheduleEntry,
    RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
    RelationshipActionGateV2FederatedScheduleSegment,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    FrozenLinearRelationshipConditionReaderArtifact,
    FrozenLinearRelationshipConditionReaderRuntime,
    FrozenLinearRelationshipPreferenceForecastRuntime,
)
from lifeform_evolution import relationship_product_horizon_theta0_calibration as cal
from lifeform_evolution.relationship_product_horizon_source_admission import (
    build_relationship_product_horizon_source_action_commitment,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (
    PrecomputedPublicSemanticEmbedder,
    load_precomputed_public_embedding_table,
)
from volvence_zero.social import (
    PreferenceActionForecastRequest,
    social_record_store_persistence_sha256,
)


THETA0_V3_BOOTSTRAP_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-v3-bootstrap-protocol.v2"
)
THETA0_V3_BOOTSTRAP_TRACE_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-v3-bootstrap-trace.v2"
)
THETA0_V3_BOOTSTRAP_TRANSITION_BUNDLE_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-v3-transition-bundle.v1"
)
THETA0_V3_BOOTSTRAP_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-v3-bootstrap-manifest.v2"
)
THETA0_V3_BOOTSTRAP_PROTOCOL_RAW_SHA256 = (
    "8306017942f014892f3e9652d63a51c7ea7240cdf1e88c2e437255395483e38f"
)
THETA0_V3_BOOTSTRAP_PROTOCOL_ID = (
    "f5c33f5c94f90bc701e8306193da790054a2af29196499bd91e1689d68841d26"
)

_PROTOCOL_FILENAME = "relationship_product_horizon_theta0_v3_bootstrap_v2.json"
_SCHEDULE_FILENAME = "parent_schedule.json"
_TRACE_FILENAME = "theta0_v3_trace.jsonl"
_TRANSITION_FILENAME = "federated_transition_bundle.json"
_THETA_FILENAME = "theta0_artifact.json"
_MANIFEST_FILENAME = "manifest.json"
_BASE_OUTPUT_FILES = frozenset(
    {
        "protocol.json",
        _SCHEDULE_FILENAME,
        _TRACE_FILENAME,
        _TRANSITION_FILENAME,
        _MANIFEST_FILENAME,
    }
)
_SUCCESS_OUTPUT_FILES = frozenset({*_BASE_OUTPUT_FILES, _THETA_FILENAME})
_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[4]
_OWNER_RELATIVE_PATH = (
    "packages/lifeform-evolution/src/lifeform_evolution/"
    "relationship_product_horizon_theta0_v3_bootstrap.py"
)
_IMPLEMENTATION_OWNED_PATHS = (
    "packages/vz-contracts/src/volvence_zero/runtime/kernel.py",
    "packages/vz-temporal/src/volvence_zero/temporal/interface.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/"
    "relationship_product_pulse.py",
    _OWNER_RELATIVE_PATH,
    "packages/lifeform-evolution/src/lifeform_evolution/protocols/"
    "relationship_product_horizon_theta0_v3_bootstrap_v2.json",
    "scripts/run_relationship_product_horizon_theta0_v3_bootstrap.py",
)
_MATERIALIZATION_CLEAN_SCOPE = (
    "packages",
    "scripts/run_relationship_product_horizon_theta0_v3_bootstrap.py",
)
_OUTPUT_FORBIDDEN_REPOSITORY_ROOTS = (
    _REPOSITORY_ROOT / "packages",
    _REPOSITORY_ROOT / "scripts",
)
_SOURCE_ARTIFACT_ID = (
    "relationship-product-source-sha256:"
    "4cc1ec455a40ffde2b994601f963527ea7d2300191b212b2e4988d2f6ccedd54"
)
_PARENT_SCOPE_ID = "relationship-product-horizon-medium-development-20260826"
_INTERLOCUTOR_ID = "primary"
_EMPTY_CHAIN_SHA256 = hashlib.sha256(b"").hexdigest()
_SUCCESS_STATUS = (
    "development_theta0_v3_federated_materialized_"
    "treatment_reachability_admission_next_effect_not_tested"
)
_FAIL_STATUS = (
    "development_theta0_v3_terminal_gate_failed_no_theta0_effect_not_tested"
)


@dataclass(frozen=True)
class RelationshipProductHorizonTheta0V3BootstrapProtocol:
    payload: Mapping[str, object]
    raw_bytes: bytes
    protocol_id: str
    raw_sha256: str


@dataclass(frozen=True)
class _Dependencies:
    protocol: RelationshipProductHorizonTheta0V3BootstrapProtocol
    source_root: pathlib.Path
    public_view: RelationshipProductHorizonPublicView
    forecast_runtime: FrozenLinearRelationshipPreferenceForecastRuntime


@dataclass(frozen=True)
class _ImplementationCheckoutReceipt:
    implementation_git_commit: str
    owned_blob_ids: tuple[tuple[str, str], ...]

    @property
    def owned_blob_ids_sha256(self) -> str:
        return sha256_json(
            [
                {"path": path, "git_blob_id": blob_id}
                for path, blob_id in self.owned_blob_ids
            ]
        )


@dataclass(frozen=True)
class _DurableParentScheduleReceipt:
    protocol_id: str
    implementation_git_commit: str
    parent_schedule_artifact_id: str
    parent_schedule_raw_sha256: str
    parent_schedule_raw_bytes: int
    ordered_child_schedule_ids_sha256: str
    implementation_owned_blob_ids_sha256: str

    def validate(
        self,
        *,
        protocol_id: str,
        implementation_git_commit: str,
        parent_schedule: RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
        implementation_checkout: _ImplementationCheckoutReceipt,
    ) -> None:
        raw = _canonical_bytes(parent_schedule.to_payload())
        expected = (
            protocol_id,
            implementation_git_commit,
            parent_schedule.artifact_id,
            _sha256_bytes(raw),
            len(raw),
            sha256_json(
                [
                    item.child_schedule_artifact.artifact_id
                    for item in parent_schedule.segments
                ]
            ),
            implementation_checkout.owned_blob_ids_sha256,
        )
        observed = (
            self.protocol_id,
            self.implementation_git_commit,
            self.parent_schedule_artifact_id,
            self.parent_schedule_raw_sha256,
            self.parent_schedule_raw_bytes,
            self.ordered_child_schedule_ids_sha256,
            self.implementation_owned_blob_ids_sha256,
        )
        if observed != expected:
            raise ValueError("theta0 v3 durable parent receipt drifted")


@dataclass(frozen=True)
class _ActiveParentScheduleReceipt:
    durable: _DurableParentScheduleReceipt
    implementation_checkout: _ImplementationCheckoutReceipt
    ledger_row_id: str
    physical_sequence_index: int

    def __post_init__(self) -> None:
        cal._digest(self.ledger_row_id, "ledger_row_id")
        if self.physical_sequence_index != 0:
            raise ValueError("parent durable receipt must be ledger row zero")


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
class _ChildTransitionObservation:
    apply_count: int
    withhold_count: int

    def __post_init__(self) -> None:
        for field, value in (
            ("apply_count", self.apply_count),
            ("withhold_count", self.withhold_count),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative exact integer")

    @property
    def total_count(self) -> int:
        return self.apply_count + self.withhold_count

    @property
    def accepted_lineage_has_no_child_transition(self) -> bool:
        return self.total_count == 0


@dataclass(frozen=True)
class _BootstrapReplay:
    implementation_checkout: _ImplementationCheckoutReceipt
    seed_artifact: RelationshipActionGateV2Artifact
    parent_schedule: RelationshipActionGateV2FederatedAssignmentScheduleArtifact
    child_collections: tuple[RelationshipProductV2SegmentedCollectedCreditBatch, ...]
    federated_collection: RelationshipProductV2FederatedCollectedCreditBatch
    matched_transitions: RelationshipProductV2FederatedMatchedGateTransitions
    learned_theta0: RelationshipActionGateV2Artifact | None
    parent_ledger_row_id: str
    completed_root_count: int
    onboarding_count: int
    preaction_count: int
    postaction_count: int
    onboarding_write_change_count: int
    owner_handoff_count: int
    owner_writeback_change_count: int
    child_transition_observation: _ChildTransitionObservation
    delivered_action_counts: Mapping[str, int]
    terminal_failure_reasons: tuple[str, ...]
    terminal_status: str


@dataclass(frozen=True)
class RelationshipProductHorizonTheta0V3Bundle:
    """Strictly replayed development bundle for the next admission owner."""

    manifest: Mapping[str, object]
    theta0_artifact: RelationshipActionGateV2Artifact
    federated_collection: RelationshipProductV2FederatedCollectedCreditBatch
    matched_transitions: RelationshipProductV2FederatedMatchedGateTransitions


class _RawSink(Protocol):
    def append(self, payload: Mapping[str, object]) -> None: ...

    @property
    def raw_bytes(self) -> bytes: ...


class _Ledger:
    def __init__(self, sink: _RawSink) -> None:
        self._sink = sink
        self._physical_sequence_index = 0
        self._chain_sha256 = _EMPTY_CHAIN_SHA256

    @property
    def raw_bytes(self) -> bytes:
        return self._sink.raw_bytes

    @property
    def next_physical_sequence_index(self) -> int:
        return self._physical_sequence_index

    def append(
        self,
        *,
        record_type: str,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        cal._text(record_type, "record_type")
        core = {
            "schema_version": THETA0_V3_BOOTSTRAP_TRACE_SCHEMA_VERSION,
            "physical_sequence_index": self._physical_sequence_index,
            "prior_chain_sha256": self._chain_sha256,
            "record_type": record_type,
            **payload,
        }
        row = {"row_id": sha256_json(core), **core}
        self._sink.append(row)
        self._chain_sha256 = hashlib.sha256(
            self._chain_sha256.encode("ascii") + _canonical_bytes(row)
        ).hexdigest()
        self._physical_sequence_index += 1
        return row


def relationship_product_horizon_theta0_v3_bootstrap_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).with_name("protocols") / _PROTOCOL_FILENAME


def load_relationship_product_horizon_theta0_v3_bootstrap_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonTheta0V3BootstrapProtocol:
    source = pathlib.Path(
        path or relationship_product_horizon_theta0_v3_bootstrap_protocol_path()
    )
    raw = source.read_bytes()
    payload = cal._parse_json_bytes(raw, source="theta0 v3 bootstrap protocol")
    cal._exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "purpose",
            "adaptive_lineage",
            "retired_replay_lineage",
            "implementation_lineage",
            "source_v4_admission",
            "development_reader",
            "gate_v2",
            "federated_schedule",
            "durability",
            "root_owner_replay",
            "parent_transition",
            "terminal_gates",
            "causal_firewall",
            "claims",
            "claim_boundary",
        },
        "theta0 v3 bootstrap protocol",
    )
    if _sha256_bytes(raw) != THETA0_V3_BOOTSTRAP_PROTOCOL_RAW_SHA256:
        raise ValueError("theta0 v3 bootstrap protocol raw bytes drifted")
    if (
        payload["schema_version"] != THETA0_V3_BOOTSTRAP_PROTOCOL_SCHEMA_VERSION
        or payload["evidence_tier"] != "development"
        or payload["owner"]
        != "lifeform_evolution.relationship_product_horizon_theta0_v3_bootstrap"
        or payload["purpose"]
        != "source_v4_spent_adaptive_federated_gate_v2_theta0_materialization"
    ):
        raise ValueError("theta0 v3 bootstrap protocol identity drifted")
    _validate_protocol(payload)
    protocol_id = sha256_json(payload)
    if protocol_id != THETA0_V3_BOOTSTRAP_PROTOCOL_ID:
        raise ValueError("theta0 v3 bootstrap protocol canonical identity drifted")
    return RelationshipProductHorizonTheta0V3BootstrapProtocol(
        payload=payload,
        raw_bytes=raw,
        protocol_id=protocol_id,
        raw_sha256=_sha256_bytes(raw),
    )


def _require_exact_contract(
    value: object,
    *,
    expected: Mapping[str, object],
    source: str,
) -> Mapping[str, object]:
    actual = cal._mapping(value, source)
    cal._exact_keys(actual, set(expected), source)
    for key, expected_value in expected.items():
        observed = actual[key]
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise ValueError(f"{source}.{key} drifted")
    return actual


def _validate_protocol(payload: Mapping[str, object]) -> None:
    _require_exact_contract(
        payload["adaptive_lineage"],
        source="adaptive_lineage",
        expected={
            "source_v4_already_used_for_development": True,
            "development_reader_qualified": False,
            "source_v4_unseen_evidence": False,
            "effect_estimand_present": False,
            "legacy_theta0_v2_inherited": False,
            "legacy_forced_common_batch_inherited": False,
            "legacy_outcome_or_credit_file_read_count": 0,
            "rehearsal_execution_authorized": False,
        },
    )
    _require_exact_contract(
        payload["retired_replay_lineage"],
        source="retired_replay_lineage",
        expected={
            "protocol_id": (
                "9c48a8e3d17f59b8bf62a7868c390a8b81af9eb1aef8dfb3096e08ee58dc12b5"
            ),
            "protocol_raw_sha256": (
                "c7e2d75fdeb3d825d3074351d369870c4578375aabff6ba94dcea0db8ed5883f"
            ),
            "implementation_git_commit": (
                "79891e3b6cab59e29de037570d1d4605bd1346ff"
            ),
            "materialized_manifest_artifact_id": (
                "0c596cd5f0a13d26dca5aeee5a3f83f2e32dc20b7d31991708cf1797e326076c"
            ),
            "validate_existing_passed": False,
            "scientific_terminal": False,
            "downstream_authority": False,
            "partial_resume_allowed": False,
            "output_root_reuse_allowed": False,
            "failure_reason": (
                "temporal_delivery_wall_clock_timestamp_made_forced_receipt_"
                "and_downstream_lineage_non_replayable"
            ),
        },
    )
    implementation = cal._mapping(
        payload["implementation_lineage"], "implementation_lineage"
    )
    cal._exact_keys(
        implementation,
        {
            "materialize_head_must_equal_implementation_commit",
            "materialize_clean_scope",
            "owned_paths",
            "validate_commit_must_exist",
            "validate_head_must_equal_implementation_commit",
            "validate_clean_scope_must_match_implementation_commit",
            "validate_owned_paths_must_match_commit",
        },
        "implementation_lineage",
    )
    if (
        implementation["materialize_head_must_equal_implementation_commit"]
        is not True
        or implementation["validate_commit_must_exist"] is not True
        or implementation["validate_head_must_equal_implementation_commit"]
        is not True
        or implementation[
            "validate_clean_scope_must_match_implementation_commit"
        ]
        is not True
        or implementation["validate_owned_paths_must_match_commit"] is not True
        or tuple(cal._list(implementation["owned_paths"], "owned_paths"))
        != _IMPLEMENTATION_OWNED_PATHS
        or tuple(
            cal._list(implementation["materialize_clean_scope"], "clean_scope")
        )
        != _MATERIALIZATION_CLEAN_SCOPE
    ):
        raise ValueError("theta0 v3 implementation-lineage contract drifted")
    gate = cal._mapping(payload["gate_v2"], "gate_v2")
    cal._exact_keys(
        gate,
        {
            "bootstrap_source_artifact_id",
            "bootstrap_learning_rate_hex",
            "online_learning_rate_hex",
            "max_abs_parameter_hex",
            "expected_seed_artifact_id",
            "expected_seed_raw_sha256",
            "expected_seed_checkpoint_content_sha256",
            "feature_order",
            "threshold_rule",
            "free_bias_present",
            "online_collection_apply",
        },
        "gate_v2",
    )
    if (
        gate["bootstrap_source_artifact_id"] != _SOURCE_ARTIFACT_ID
        or gate["bootstrap_learning_rate_hex"] != (1.0 / 512.0).hex()
        or gate["online_learning_rate_hex"] != (1.0 / 4.0).hex()
        or gate["max_abs_parameter_hex"] != (4.0).hex()
        or gate["free_bias_present"] is not False
        or gate["online_collection_apply"] is not False
    ):
        raise ValueError("theta0 v3 gate-v2 contract drifted")
    schedule = cal._mapping(payload["federated_schedule"], "federated_schedule")
    if (
        cal._integer(schedule["root_count"], "root_count") != 112
        or cal._integer(schedule["decision_count_per_root"], "decision_count_per_root")
        != 8
        or cal._integer(schedule["entry_count"], "entry_count") != 896
        or schedule["expected_parent_schedule_artifact_id"]
        != (
            "relationship-action-gate-v2-federated-assignment-schedule-sha256:"
            "c700009259a688dfb6cec5ea954a26da159a32c9ea870602fbdef8631c43a273"
        )
        or schedule["expected_parent_schedule_raw_sha256"]
        != "c1f8680644072ac666f44eafe44e1ca8979e03525550d0c110e84eb51df6fdba"
        or cal._integer(
            schedule["expected_parent_schedule_raw_bytes"],
            "expected_parent_schedule_raw_bytes",
        )
        != 382326
    ):
        raise ValueError("theta0 v3 federated schedule contract drifted")
    durability = cal._mapping(payload["durability"], "durability")
    if any(
        durability[field] is not True
        for field in (
            "output_root_create_only",
            "parent_schedule_create_only",
            "parent_schedule_write_flush_fsync_reopen_exact",
            "parent_durable_ledger_row_must_be_first",
            "parent_receipt_required_before_onboarding",
            "parent_receipt_required_before_forecast",
            "parent_receipt_required_before_environment_open",
            "preaction_row_fsync_before_selected_branch_open",
            "postaction_row_fsync_before_next_preaction",
            "trace_close_reopen_byte_exact_before_transition_and_manifest",
            "output_root_disjoint_from_frozen_input_roots",
            "manifest_written_last",
        )
    ):
        raise ValueError("theta0 v3 durability contract opened")
    if any(
        durability[field] is not False
        for field in (
            "partial_resume_allowed",
            "overwrite_allowed",
            "windows_directory_entry_durability_attested",
            "os_process_secrecy_claim",
        )
    ):
        raise ValueError("theta0 v3 durability claim ceiling drifted")
    _require_exact_contract(
        payload["root_owner_replay"],
        source="root_owner_replay",
        expected={
            "owner_reset_input": "literal_none",
            "onboarding_count_per_root": 4,
            "onboarding_order": "source_v4_public_onboarding_array_order",
            "segment_count_per_root": 1,
            "segment_scope": "source_v4_subject_id",
            "segment_start_equals_post_onboarding_snapshot": True,
            "settlement_handoff_required": True,
            "temporal_delivery_timestamp_owner": "SelfTemporalModule",
            "temporal_delivery_timestamp_clock": (
                "protocol_frozen_offline_logical_milliseconds"
            ),
            "temporal_delivery_timestamp_formula": (
                "root_index_times_20_plus_4_plus_2_times_decision_index"
            ),
            "temporal_delivery_timestamp_globally_strictly_increasing": True,
            "temporal_delivery_precedes_credit_timestamp_by_ms": 1,
            "wall_clock_temporal_delivery_timestamp_forbidden": True,
            "credit_timestamp_formula": (
                "root_index_times_20_plus_5_plus_2_times_decision_index"
            ),
            "credit_timestamp_globally_strictly_increasing": True,
        },
    )
    claims = cal._mapping(payload["claims"], "claims")
    if any(type(value) is not bool for value in claims.values()):
        raise ValueError("theta0 v3 claims must be exact booleans")
    if {key for key, value in claims.items() if value} != {
        "bootstrap_execution_authorized",
        "development_theta0_v3_may_be_materialized",
    }:
        raise ValueError("theta0 v3 protocol claim ceiling drifted")
    cal._text(payload["claim_boundary"], "claim_boundary")


def _forced_role(
    *, root_index: int, decision_index: int
) -> RelationshipActionGateV2AssignmentRole:
    return (
        RelationshipActionGateV2AssignmentRole.CANDIDATE
        if ((root_index // 2) + (decision_index // 2)) % 2 == 0
        else RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP
    )


def _build_federated_schedule(
    public_view: RelationshipProductHorizonPublicView,
) -> tuple[
    RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
    tuple[RelationshipActionGateV2AssignmentScheduleArtifact, ...],
]:
    if len(public_view.roots) != 112:
        raise ValueError("theta0 v3 schedule requires exactly 112 public roots")
    children: list[RelationshipActionGateV2AssignmentScheduleArtifact] = []
    parent_segments: list[RelationshipActionGateV2FederatedScheduleSegment] = []
    global_roles: Counter[RelationshipActionGateV2AssignmentRole] = Counter()
    position_roles = {index: Counter() for index in range(8)}
    for root_index, root in enumerate(public_view.roots):
        decisions = root.decision_sessions[:8]
        if tuple(item.decision_index for item in decisions) != tuple(range(8)):
            raise ValueError("theta0 v3 collection decisions must be zero through seven")
        entries = tuple(
            RelationshipActionGateV2AssignmentScheduleEntry(
                decision_id=decision.decision_id,
                sequence_index=decision.decision_index,
                assignment_role=_forced_role(
                    root_index=root_index,
                    decision_index=decision.decision_index,
                ),
            )
            for decision in decisions
        )
        child = RelationshipActionGateV2AssignmentScheduleArtifact(
            source_artifact_id=_SOURCE_ARTIFACT_ID,
            schedule_scope_id=root.subject_id,
            entries=entries,
        )
        if Counter(item.assignment_role for item in entries) != Counter(
            {
                RelationshipActionGateV2AssignmentRole.CANDIDATE: 4,
                RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP: 4,
            }
        ):
            raise ValueError("theta0 v3 child schedule is not exactly balanced")
        for item in entries:
            global_roles[item.assignment_role] += 1
            position_roles[item.sequence_index][item.assignment_role] += 1
        children.append(child)
        parent_segments.append(
            RelationshipActionGateV2FederatedScheduleSegment(
                global_start_index=root_index * 8,
                child_schedule_artifact=child,
            )
        )
    if global_roles != Counter(
        {
            RelationshipActionGateV2AssignmentRole.CANDIDATE: 448,
            RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP: 448,
        }
    ) or any(
        counts
        != Counter(
            {
                RelationshipActionGateV2AssignmentRole.CANDIDATE: 56,
                RelationshipActionGateV2AssignmentRole.NEUTRAL_NOOP: 56,
            }
        )
        for counts in position_roles.values()
    ):
        raise ValueError("theta0 v3 global or position balance drifted")
    parent = RelationshipActionGateV2FederatedAssignmentScheduleArtifact(
        source_artifact_id=_SOURCE_ARTIFACT_ID,
        schedule_scope_id=_PARENT_SCOPE_ID,
        segments=tuple(parent_segments),
    )
    return parent, tuple(children)


def _build_seed(
    protocol: RelationshipProductHorizonTheta0V3BootstrapProtocol,
) -> RelationshipActionGateV2Artifact:
    gate = cal._mapping(protocol.payload["gate_v2"], "gate_v2")
    seed = RelationshipActionGateV2Artifact.create_bootstrap_seed(
        bootstrap_learning_rate=float.fromhex(
            cal._text(gate["bootstrap_learning_rate_hex"], "bootstrap_learning_rate_hex")
        ),
        online_learning_rate=float.fromhex(
            cal._text(gate["online_learning_rate_hex"], "online_learning_rate_hex")
        ),
        max_abs_parameter=float.fromhex(
            cal._text(gate["max_abs_parameter_hex"], "max_abs_parameter_hex")
        ),
        bootstrap_source_artifact_id=cal._text(
            gate["bootstrap_source_artifact_id"], "bootstrap_source_artifact_id"
        ),
    )
    checkpoint = RelationshipActionGateV2(artifact=seed).export_checkpoint()
    if (
        seed.artifact_id != gate["expected_seed_artifact_id"]
        or _sha256_bytes(_canonical_bytes(seed.to_payload()))
        != gate["expected_seed_raw_sha256"]
        or checkpoint.content_sha256
        != gate["expected_seed_checkpoint_content_sha256"]
    ):
        raise ValueError("theta0 v3 bootstrap seed identity drifted")
    return seed


def _file_entries(
    manifest: Mapping[str, object], *, source: str
) -> Mapping[str, Mapping[str, object]]:
    return cal._file_entry_map(manifest["files"], f"{source}.files")


def _load_dependencies(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
) -> _Dependencies:
    protocol = load_relationship_product_horizon_theta0_v3_bootstrap_protocol()
    source_pin = cal._mapping(protocol.payload["source_v4_admission"], "source_v4_admission")
    source_root = pathlib.Path(source_v4_admission_root)
    source_manifest_raw = cal._require_raw_sha(
        source_root / "manifest.json",
        source_pin["manifest_raw_sha256"],
        "source-v4 admission manifest",
    )
    source_manifest = cal._parse_json_bytes(
        source_manifest_raw, source="source-v4 admission manifest"
    )
    if source_manifest_raw != cal._canonical_bytes(source_manifest):
        raise ValueError("source-v4 admission manifest must use canonical bytes")
    if (
        source_manifest["protocol_id"] != source_pin["protocol_id"]
        or source_manifest["artifact_id"] != source_pin["artifact_id"]
        or source_manifest["source_protocol_id"] != source_pin["source_protocol_id"]
        or source_manifest["public_plan_sha256"] != source_pin["public_plan_sha256"]
        or source_manifest["sealed_bundle_sha256"]
        != source_pin["sealed_evaluator_bundle_sha256"]
        or source_manifest["root_count"] != 112
        or source_manifest["onboarding_session_count"] != 448
        or source_manifest["status"] != "campaign_input_admitted_execution_not_authorized"
    ):
        raise ValueError("theta0 v3 source-v4 admission envelope drifted")
    if source_manifest["artifact_id"] != sha256_json(
        {key: value for key, value in source_manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("source-v4 admission content identity drifted")
    source_files = _file_entries(source_manifest, source="source-v4 admission")
    for relative_field, digest_field in (
        ("source_protocol_relative_path", "source_protocol_raw_sha256"),
        ("public_plan_relative_path", "public_plan_raw_sha256"),
        ("sealed_evaluator_relative_path", "sealed_evaluator_raw_sha256"),
        ("commitment_index_relative_path", "commitment_index_raw_sha256"),
    ):
        relative = cal._text(source_pin[relative_field], relative_field)
        if source_files[relative]["raw_sha256"] != source_pin[digest_field]:
            raise ValueError(f"theta0 v3 source-v4 file pin drifted: {relative}")
    public_raw = cal._require_raw_sha(
        source_root / cal._text(source_pin["public_plan_relative_path"], "public_plan_relative_path"),
        source_pin["public_plan_raw_sha256"],
        "source-v4 public plan",
    )
    public_payload = cal._parse_json_bytes(public_raw, source="source-v4 public plan")
    if public_raw != cal._canonical_bytes(public_payload):
        raise ValueError("source-v4 public plan must use canonical bytes")
    public_view = RelationshipProductHorizonPublicView.from_payload(public_payload)
    if (
        public_view.schema_version != source_pin["public_plan_schema_version"]
        or public_view.protocol_id != source_pin["source_protocol_id"]
        or public_view.public_plan_sha256 != source_pin["public_plan_sha256"]
    ):
        raise ValueError("theta0 v3 public-view owner replay drifted")

    reader_pin = cal._mapping(protocol.payload["development_reader"], "development_reader")
    development_root = pathlib.Path(reader_root)
    reader_manifest_raw = cal._require_raw_sha(
        development_root / "manifest.json",
        reader_pin["manifest_raw_sha256"],
        "development reader manifest",
    )
    reader_manifest = cal._parse_json_bytes(
        reader_manifest_raw, source="development reader manifest"
    )
    if reader_manifest_raw != cal._canonical_bytes(reader_manifest):
        raise ValueError("development reader manifest must use canonical bytes")
    if (
        reader_manifest["protocol_id"] != reader_pin["protocol_id"]
        or reader_manifest["artifact_id"] != reader_pin["artifact_id"]
        or reader_manifest["embedding_table_artifact_id"]
        != reader_pin["embedding_table_artifact_id"]
        or reader_manifest["reader_artifact_id"] != reader_pin["reader_artifact_id"]
        or reader_manifest["source_v4_sealed_file_read_count"] != 0
        or reader_manifest["challenge_label_file_read_count"] != 0
        or reader_manifest["status"] != "development_unqualified_reader_materialized"
        or reader_manifest["claims"]["condition_reader_qualified"] is not False
    ):
        raise ValueError("theta0 v3 development reader envelope drifted")
    if reader_manifest["artifact_id"] != sha256_json(
        {key: value for key, value in reader_manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("development reader content identity drifted")
    reader_files = _file_entries(reader_manifest, source="development reader")
    if (
        reader_files["embedding_table.json"]["raw_sha256"]
        != reader_pin["embedding_table_raw_sha256"]
        or reader_files["reader_artifact.json"]["raw_sha256"]
        != reader_pin["reader_artifact_raw_sha256"]
    ):
        raise ValueError("theta0 v3 development reader file pin drifted")
    table_path = development_root / "embedding_table.json"
    cal._require_raw_sha(
        table_path,
        reader_pin["embedding_table_raw_sha256"],
        "development embedding table",
    )
    table = load_precomputed_public_embedding_table(table_path)
    if table.artifact_id != reader_pin["embedding_table_artifact_id"]:
        raise ValueError("theta0 v3 embedding table identity drifted")
    reader_raw = cal._require_raw_sha(
        development_root / "reader_artifact.json",
        reader_pin["reader_artifact_raw_sha256"],
        "development reader artifact",
    )
    reader_artifact = FrozenLinearRelationshipConditionReaderArtifact.from_json(
        reader_raw.decode("utf-8")
    )
    if reader_artifact.artifact_id != reader_pin["reader_artifact_id"]:
        raise ValueError("theta0 v3 reader artifact identity drifted")
    forecast_runtime = FrozenLinearRelationshipPreferenceForecastRuntime(
        reader=FrozenLinearRelationshipConditionReaderRuntime(
            artifact=reader_artifact,
            embedder=PrecomputedPublicSemanticEmbedder(table),
        )
    )
    return _Dependencies(
        protocol=protocol,
        source_root=source_root,
        public_view=public_view,
        forecast_runtime=forecast_runtime,
    )


class _SelectedBranchEnvironmentScope:
    """Open sealed source-v4 only after the owner's durable preaction barrier."""

    def __init__(self, *, dependencies: _Dependencies) -> None:
        source_pin = cal._mapping(
            dependencies.protocol.payload["source_v4_admission"],
            "source_v4_admission",
        )
        source_path = dependencies.source_root / cal._text(
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
            raise ValueError("theta0 v3 source protocol identity drifted")

        evaluator_path = dependencies.source_root / cal._text(
            source_pin["sealed_evaluator_relative_path"],
            "sealed_evaluator_relative_path",
        )
        evaluator_raw = cal._require_raw_sha(
            evaluator_path,
            source_pin["sealed_evaluator_raw_sha256"],
            "source-v4 sealed evaluator",
        )
        evaluator_payload = cal._parse_json_bytes(
            evaluator_raw, source="source-v4 sealed evaluator"
        )
        if evaluator_raw != cal._canonical_bytes(evaluator_payload):
            raise ValueError("source-v4 sealed evaluator must use canonical bytes")
        evaluator = build_relationship_product_horizon_evaluator_bundle(source_protocol)
        if (
            evaluator.sealed_bundle_sha256
            != source_pin["sealed_evaluator_bundle_sha256"]
            or cal._canonical_bytes(evaluator.to_payload()) != evaluator_raw
        ):
            raise ValueError("theta0 v3 sealed evaluator owner replay drifted")

        commitment_path = dependencies.source_root / cal._text(
            source_pin["commitment_index_relative_path"],
            "commitment_index_relative_path",
        )
        commitment_raw = cal._require_raw_sha(
            commitment_path,
            source_pin["commitment_index_raw_sha256"],
            "source-v4 action commitment index",
        )
        commitment_payload = cal._parse_json_bytes(
            commitment_raw, source="source-v4 action commitment index"
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
            or commitment_payload["decision_count"] != 5376
            or commitment_payload["commitment_count"]
            != source_pin["commitment_count"]
        ):
            raise ValueError("theta0 v3 commitment index identity drifted")
        branch_ids: dict[tuple[str, str], str] = {}
        for raw_decision in cal._list(
            commitment_payload["decision_branch_commitments"],
            "decision_branch_commitments",
        ):
            decision = cal._mapping(raw_decision, "decision commitment")
            cal._exact_keys(
                decision,
                {"decision_id", "branches"},
                "decision commitment",
            )
            decision_id = cal._text(decision["decision_id"], "decision_id")
            branches = cal._list(decision["branches"], "branches")
            if len(branches) != 3:
                raise ValueError("source-v4 decision must contain three branches")
            for raw_branch in branches:
                branch = cal._mapping(raw_branch, "commitment branch")
                cal._exact_keys(
                    branch,
                    {"selected_action_id", "commitment_id"},
                    "commitment branch",
                )
                key = (
                    decision_id,
                    cal._text(branch["selected_action_id"], "selected_action_id"),
                )
                if key in branch_ids:
                    raise ValueError("theta0 v3 commitment branch identity reused")
                branch_ids[key] = cal._digest(
                    branch["commitment_id"], "commitment_id"
                )
        if len(branch_ids) != source_pin["commitment_count"]:
            raise ValueError("theta0 v3 commitment branch inventory drifted")
        if tuple(
            (item.root_index, item.subject_id, item.public_trajectory_sha256)
            for item in evaluator.root_manifests
        ) != tuple(
            (index, root.subject_id, root.public_trajectory_sha256)
            for index, root in enumerate(dependencies.public_view.roots)
        ):
            raise ValueError("theta0 v3 public/evaluator root join drifted")
        self._evaluator = evaluator
        self._branch_ids = branch_ids

    def settle(
        self,
        *,
        public_root: HorizonPublicRoot,
        public_decision: HorizonPublicDecisionSession,
        selected_action: RelationshipAction,
    ) -> _SafeSelectedBranchOutcome:
        commitment = build_relationship_product_horizon_source_action_commitment(
            self._evaluator,
            subject_id=public_root.subject_id,
            decision_id=public_decision.decision_id,
            action=selected_action,
        )
        commitment_id = cal._digest(commitment["commitment_id"], "commitment_id")
        expected = self._branch_ids.get(
            (public_decision.decision_id, selected_action.value)
        )
        if commitment_id != expected:
            raise ValueError("theta0 v3 selected branch commitment drifted")
        preimage = cal._mapping(commitment["preimage"], "commitment preimage")
        if (
            preimage["subject_id"] != public_root.subject_id
            or preimage["decision_id"] != public_decision.decision_id
            or preimage["selected_action_id"] != selected_action.value
            or preimage["sealed_evaluator_bundle_sha256"]
            != self._evaluator.sealed_bundle_sha256
        ):
            raise ValueError("theta0 v3 selected branch preimage drifted")
        return _SafeSelectedBranchOutcome(
            environment_subject_id=cal._text(preimage["subject_id"], "subject_id"),
            selected_action_id=cal._text(
                preimage["selected_action_id"], "selected_action_id"
            ),
            typed_outcome_id=cal._text(
                preimage["typed_outcome_id"], "typed_outcome_id"
            ),
            rendered_user_reaction=cal._text(
                preimage["rendered_user_reaction"], "rendered_user_reaction"
            ),
            environment_evidence_ref=cal._text(
                preimage["environment_evidence_ref"], "environment_evidence_ref"
            ),
            environment_version=cal._text(
                preimage["environment_version"], "environment_version"
            ),
            commitment_id=commitment_id,
        )


def _request(
    *, subject_id: str, decision: HorizonPublicDecisionSession
) -> RelationshipProductPreActionRequest:
    action_turn = 4 + decision.decision_index * 2
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


def _credit_timestamp(root_index: int, decision_index: int) -> int:
    return root_index * 20 + 5 + 2 * decision_index


def _temporal_delivery_timestamp(root_index: int, decision_index: int) -> int:
    return root_index * 20 + 4 + 2 * decision_index


def _canonical_bytes(payload: object) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError("theta0 v3 could not execute git") from exc
    if check and completed.returncode != 0:
        raise ValueError(
            "theta0 v3 git lineage check failed: "
            f"args={args!r}; stderr={completed.stderr.strip()!r}"
        )
    return completed


def _verify_implementation_checkout(
    *,
    protocol: RelationshipProductHorizonTheta0V3BootstrapProtocol,
    implementation_git_commit: str,
) -> _ImplementationCheckoutReceipt:
    commit = cal._git_commit(implementation_git_commit)
    expected_owner = (_REPOSITORY_ROOT / _OWNER_RELATIVE_PATH).resolve(strict=True)
    if os.path.normcase(str(expected_owner)) != os.path.normcase(
        str(pathlib.Path(__file__).resolve(strict=True))
    ):
        raise ValueError("theta0 v3 owner is not loaded from the repository closure")
    repository = pathlib.Path(
        _run_git("rev-parse", "--show-toplevel").stdout.strip()
    ).resolve(strict=True)
    if os.path.normcase(str(repository)) != os.path.normcase(
        str(_REPOSITORY_ROOT.resolve(strict=True))
    ):
        raise ValueError("theta0 v3 repository root identity drifted")
    resolved_commit = _run_git(
        "rev-parse", "--verify", f"{commit}^{{commit}}"
    ).stdout.strip()
    if resolved_commit != commit:
        raise ValueError("theta0 v3 implementation commit does not exist exactly")
    head = _run_git("rev-parse", "HEAD").stdout.strip()
    if head != commit:
        raise ValueError("theta0 v3 implementation commit does not match HEAD")

    implementation = cal._mapping(
        protocol.payload["implementation_lineage"], "implementation_lineage"
    )
    owned_paths = tuple(cal._list(implementation["owned_paths"], "owned_paths"))
    if owned_paths != _IMPLEMENTATION_OWNED_PATHS:
        raise ValueError("theta0 v3 owned-path lineage drifted")
    tracked = set(
        _run_git("ls-files", "--", *owned_paths).stdout.splitlines()
    )
    if tracked != set(owned_paths):
        raise ValueError("theta0 v3 implementation closure is not fully tracked")
    owned_diff = _run_git(
        "diff", "--quiet", commit, "--", *owned_paths, check=False
    )
    if owned_diff.returncode != 0:
        if owned_diff.returncode == 1:
            raise ValueError(
                "theta0 v3 owned implementation differs from frozen commit"
            )
        raise RuntimeError("theta0 v3 git diff failed for owned implementation")
    clean_scope = tuple(
        cal._list(
            implementation["materialize_clean_scope"],
            "materialize_clean_scope",
        )
    )
    dirty = _run_git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *clean_scope,
    ).stdout
    if dirty:
        raise ValueError(
            "theta0 v3 materialization code scope differs from frozen HEAD"
        )
    owned_blob_ids = tuple(
        (path, _run_git("rev-parse", f"{commit}:{path}").stdout.strip())
        for path in owned_paths
    )
    if any(not blob_id for _path, blob_id in owned_blob_ids):
        raise ValueError("theta0 v3 implementation blob identity is empty")
    return _ImplementationCheckoutReceipt(
        implementation_git_commit=commit,
        owned_blob_ids=owned_blob_ids,
    )


def _verify_historical_implementation_lineage(
    *,
    protocol: RelationshipProductHorizonTheta0V3BootstrapProtocol,
    implementation_git_commit: str,
) -> _ImplementationCheckoutReceipt:
    """Resolve the frozen historical blobs without claiming a live checkout.

    This deliberately does not weaken ``_verify_implementation_checkout``.
    The historical commit is used only as an input identity for a later
    cross-commit compatibility replay.  Historical execution acceptance must
    be supplied independently by the handoff owner.
    """

    commit = cal._git_commit(implementation_git_commit)
    expected_owner = (_REPOSITORY_ROOT / _OWNER_RELATIVE_PATH).resolve(strict=True)
    if os.path.normcase(str(expected_owner)) != os.path.normcase(
        str(pathlib.Path(__file__).resolve(strict=True))
    ):
        raise ValueError("theta0 v3 owner is not loaded from the repository closure")
    repository = pathlib.Path(
        _run_git("rev-parse", "--show-toplevel").stdout.strip()
    ).resolve(strict=True)
    if os.path.normcase(str(repository)) != os.path.normcase(
        str(_REPOSITORY_ROOT.resolve(strict=True))
    ):
        raise ValueError("theta0 v3 repository root identity drifted")
    resolved_commit = _run_git(
        "rev-parse", "--verify", f"{commit}^{{commit}}"
    ).stdout.strip()
    if resolved_commit != commit:
        raise ValueError("theta0 v3 historical implementation commit does not exist")

    implementation = cal._mapping(
        protocol.payload["implementation_lineage"], "implementation_lineage"
    )
    owned_paths = tuple(cal._list(implementation["owned_paths"], "owned_paths"))
    if owned_paths != _IMPLEMENTATION_OWNED_PATHS:
        raise ValueError("theta0 v3 historical owned-path lineage drifted")
    owned_blob_ids = tuple(
        (path, _run_git("rev-parse", f"{commit}:{path}").stdout.strip())
        for path in owned_paths
    )
    if any(not blob_id for _path, blob_id in owned_blob_ids):
        raise ValueError("theta0 v3 historical implementation blob identity is empty")
    return _ImplementationCheckoutReceipt(
        implementation_git_commit=commit,
        owned_blob_ids=owned_blob_ids,
    )


def _require_disjoint_output_root(
    *, output_root: pathlib.Path, input_roots: tuple[pathlib.Path, ...]
) -> None:
    output = os.path.normcase(str(output_root.resolve(strict=False)))
    for raw_input in input_roots:
        frozen_input = os.path.normcase(str(raw_input.resolve(strict=True)))
        try:
            common = os.path.commonpath((output, frozen_input))
        except ValueError:
            # Different Windows volumes cannot overlap, so they are disjoint by construction.
            continue
        if common in {output, frozen_input}:
            raise ValueError(
                "theta0 v3 output root must be disjoint from frozen input roots"
            )


def _write_and_reopen_exact(path: pathlib.Path, raw: bytes) -> None:
    with path.open("x+b") as handle:
        written = handle.write(raw)
        if written != len(raw):
            raise OSError("theta0 v3 create-only writer performed a short write")
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        if handle.read() != raw:
            raise OSError("theta0 v3 same-handle readback drifted")
    if cal._read_regular(path) != raw:
        raise OSError("theta0 v3 close/reopen readback drifted")


def _require_closed_trace_exact(*, path: pathlib.Path, ledger: _Ledger) -> None:
    if cal._read_regular(path) != ledger.raw_bytes:
        raise OSError("theta0 v3 closed trace bytes drifted from the fsynced ledger")


def _durable_parent_receipt(
    *,
    protocol_id: str,
    implementation_git_commit: str,
    implementation_checkout: _ImplementationCheckoutReceipt,
    parent_schedule: RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
    persisted_raw: bytes,
) -> _DurableParentScheduleReceipt:
    if persisted_raw != _canonical_bytes(parent_schedule.to_payload()):
        raise ValueError("theta0 v3 persisted parent schedule bytes drifted")
    receipt = _DurableParentScheduleReceipt(
        protocol_id=protocol_id,
        implementation_git_commit=implementation_git_commit,
        parent_schedule_artifact_id=parent_schedule.artifact_id,
        parent_schedule_raw_sha256=_sha256_bytes(persisted_raw),
        parent_schedule_raw_bytes=len(persisted_raw),
        ordered_child_schedule_ids_sha256=sha256_json(
            [
                item.child_schedule_artifact.artifact_id
                for item in parent_schedule.segments
            ]
        ),
        implementation_owned_blob_ids_sha256=(
            implementation_checkout.owned_blob_ids_sha256
        ),
    )
    receipt.validate(
        protocol_id=protocol_id,
        implementation_git_commit=implementation_git_commit,
        parent_schedule=parent_schedule,
        implementation_checkout=implementation_checkout,
    )
    return receipt


def _activate_parent_receipt(
    *,
    ledger: _Ledger,
    durable: _DurableParentScheduleReceipt,
    protocol_id: str,
    implementation_git_commit: str,
    implementation_checkout: _ImplementationCheckoutReceipt,
    parent_schedule: RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
) -> _ActiveParentScheduleReceipt:
    if ledger.next_physical_sequence_index != 0:
        raise ValueError("parent durable receipt must be the first ledger row")
    durable.validate(
        protocol_id=protocol_id,
        implementation_git_commit=implementation_git_commit,
        parent_schedule=parent_schedule,
        implementation_checkout=implementation_checkout,
    )
    row = ledger.append(
        record_type="parent_schedule_durable",
        payload={
            "protocol_id": protocol_id,
            "implementation_git_commit": implementation_git_commit,
            "implementation_owned_blob_ids": [
                {"path": path, "git_blob_id": blob_id}
                for path, blob_id in implementation_checkout.owned_blob_ids
            ],
            "implementation_owned_blob_ids_sha256": (
                durable.implementation_owned_blob_ids_sha256
            ),
            "implementation_head_matched_before_output_creation": True,
            "implementation_clean_scope_matched_before_output_creation": True,
            "implementation_checkout_checks_are_cooperative_owner_receipts": True,
            "parent_schedule_artifact_id": durable.parent_schedule_artifact_id,
            "parent_schedule_raw_sha256": durable.parent_schedule_raw_sha256,
            "parent_schedule_raw_bytes": durable.parent_schedule_raw_bytes,
            "ordered_child_schedule_ids_sha256": (
                durable.ordered_child_schedule_ids_sha256
            ),
            "child_schedule_count": len(parent_schedule.segments),
            "entry_count": len(parent_schedule.flattened_entries),
            "onboarding_count_before_receipt": 0,
            "forecast_count_before_receipt": 0,
            "outcome_count_before_receipt": 0,
            "child_transition_count_before_receipt": 0,
            "windows_directory_entry_durability_attested": False,
            "os_process_secrecy_claim": False,
        },
    )
    return _ActiveParentScheduleReceipt(
        durable=durable,
        implementation_checkout=implementation_checkout,
        ledger_row_id=cal._digest(row["row_id"], "parent ledger row_id"),
        physical_sequence_index=cal._integer(
            row["physical_sequence_index"], "parent physical_sequence_index"
        ),
    )


def _require_active_parent(
    *,
    active: _ActiveParentScheduleReceipt,
    ledger: _Ledger,
    protocol_id: str,
    implementation_git_commit: str,
    parent_schedule: RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
) -> None:
    active.durable.validate(
        protocol_id=protocol_id,
        implementation_git_commit=implementation_git_commit,
        parent_schedule=parent_schedule,
        implementation_checkout=active.implementation_checkout,
    )
    if active.physical_sequence_index != 0 or ledger.next_physical_sequence_index < 1:
        raise RuntimeError("theta0 v3 parent receipt is not active")


def _terminal_failure_reasons(
    *,
    completed_root_count: int,
    onboarding_count: int,
    onboarding_write_change_count: int,
    preaction_count: int,
    postaction_count: int,
    owner_handoff_count: int,
    owner_writeback_change_count: int,
    child_collection_count: int,
    unique_child_collection_count: int,
    unique_credit_count: int,
    federated_credit_count: int,
    apply_child_batch_count: int,
    apply_child_transition_count: int,
    apply_credit_count: int,
    apply_atomic_commit_count: int,
    apply_update_count_delta: int,
    apply_informative_update_count_delta: int,
    apply_cap_hit_count: int,
    withhold_child_transition_count: int,
    withhold_atomic_commit_count: int,
    withhold_update_count_delta: int,
    withhold_checkpoint_unchanged: bool,
    terminal_parameter_finite: bool,
    terminal_parameter_nonzero: bool,
) -> tuple[str, ...]:
    checks = (
        (completed_root_count != 112, "completed_root_count_not_112"),
        (onboarding_count != 448, "onboarding_count_not_448"),
        (
            onboarding_write_change_count != 448,
            "onboarding_write_change_count_not_448",
        ),
        (preaction_count != 896, "preaction_count_not_896"),
        (postaction_count != 896, "postaction_count_not_896"),
        (owner_handoff_count != 784, "owner_handoff_count_not_784"),
        (
            owner_writeback_change_count != 896,
            "owner_writeback_change_count_not_896",
        ),
        (child_collection_count != 112, "child_collection_count_not_112"),
        (
            unique_child_collection_count != 112,
            "child_collection_ids_not_unique",
        ),
        (unique_credit_count != 896, "unique_credit_count_not_896"),
        (federated_credit_count != 896, "federated_credit_count_not_896"),
        (apply_child_batch_count != 112, "apply_child_batch_count_not_112"),
        (
            apply_child_transition_count != 0,
            "apply_child_transition_count_not_zero",
        ),
        (apply_credit_count != 896, "apply_credit_count_not_896"),
        (apply_atomic_commit_count != 1, "apply_atomic_commit_count_not_one"),
        (apply_update_count_delta != 896, "apply_update_count_delta_not_896"),
        (
            apply_informative_update_count_delta < 1,
            "apply_informative_update_count_below_one",
        ),
        (apply_cap_hit_count != 0, "apply_cap_hit_count_not_zero"),
        (
            withhold_child_transition_count != 0,
            "withhold_child_transition_count_not_zero",
        ),
        (
            withhold_atomic_commit_count != 0,
            "withhold_atomic_commit_count_not_zero",
        ),
        (
            withhold_update_count_delta != 0,
            "withhold_update_count_delta_not_zero",
        ),
        (not withhold_checkpoint_unchanged, "withhold_checkpoint_changed"),
        (not terminal_parameter_finite, "terminal_parameter_not_finite"),
        (not terminal_parameter_nonzero, "terminal_parameter_all_zero"),
    )
    return tuple(reason for failed, reason in checks if failed)


async def _run_bootstrap(
    *,
    dependencies: _Dependencies,
    seed_artifact: RelationshipActionGateV2Artifact,
    parent_schedule: RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
    child_schedules: tuple[RelationshipActionGateV2AssignmentScheduleArtifact, ...],
    durable_parent: _DurableParentScheduleReceipt,
    implementation_checkout: _ImplementationCheckoutReceipt,
    implementation_git_commit: str,
    ledger: _Ledger,
) -> _BootstrapReplay:
    protocol_id = dependencies.protocol.protocol_id
    active_parent = _activate_parent_receipt(
        ledger=ledger,
        durable=durable_parent,
        protocol_id=protocol_id,
        implementation_git_commit=implementation_git_commit,
        implementation_checkout=implementation_checkout,
        parent_schedule=parent_schedule,
    )
    frozen_policy = RelationshipActionGateV2(
        artifact=seed_artifact
    ).freeze_for_evaluation()
    if (
        frozen_policy.checkpoint.update_count != 0
        or frozen_policy.checkpoint.informative_update_count != 0
        or frozen_policy.checkpoint.processed_credit_ids
    ):
        raise RuntimeError("theta0 v3 forced collection requires one cold seed")
    pulse_authorization = RelationshipProductPulseAuthorization(
        authorization_id=f"relationship-product-horizon-theta0-v3:{protocol_id}",
        allowed_policy_artifact_id=seed_artifact.artifact_id,
        allowed_policy_artifact_version=2,
    )
    if len(child_schedules) != len(dependencies.public_view.roots):
        raise ValueError("theta0 v3 child schedule/root membership drifted")

    environment_scope: _SelectedBranchEnvironmentScope | None = None
    child_collections: list[RelationshipProductV2SegmentedCollectedCreditBatch] = []
    completed_root_count = 0
    onboarding_count = 0
    onboarding_write_change_count = 0
    preaction_count = 0
    postaction_count = 0
    owner_handoff_count = 0
    owner_writeback_change_count = 0
    previous_temporal_delivery_timestamp = -1
    previous_credit_timestamp = -1
    delivered_action_counts: Counter[str] = Counter()
    forecast_ids: set[str] = set()
    exposure_ids: set[str] = set()
    credit_ids: set[str] = set()
    settlement_ids: set[str] = set()
    commitment_ids: set[str] = set()

    for root_index, (root, child_schedule) in enumerate(
        zip(dependencies.public_view.roots, child_schedules, strict=True)
    ):
        _require_active_parent(
            active=active_parent,
            ledger=ledger,
            protocol_id=protocol_id,
            implementation_git_commit=implementation_git_commit,
            parent_schedule=parent_schedule,
        )
        owner_persistence = None
        ledger.append(
            record_type="root_begin",
            payload={
                "parent_durable_row_id": active_parent.ledger_row_id,
                "root_sequence_index": root_index,
                "subject_id": root.subject_id,
                "child_schedule_artifact_id": child_schedule.artifact_id,
                "prior_owner_persistence_sha256": None,
                "onboarding_count_before_root": onboarding_count,
                "forecast_count_before_root": preaction_count,
                "outcome_count_before_root": postaction_count,
                "child_transition_count": 0,
            },
        )
        for onboarding in root.onboarding_sessions:
            _require_active_parent(
                active=active_parent,
                ledger=ledger,
                protocol_id=protocol_id,
                implementation_git_commit=implementation_git_commit,
                parent_schedule=parent_schedule,
            )
            prior_sha = (
                None
                if owner_persistence is None
                else social_record_store_persistence_sha256(owner_persistence)
            )
            appended = await append_relationship_product_onboarding(
                owner_persistence_snapshot=owner_persistence,
                onboarding=RelationshipProductOnboardingInput(
                    session_id=onboarding.session_id,
                    session_index=onboarding.session_index,
                    turn_index=onboarding.session_index,
                    public_observation=onboarding.user_utterance,
                    action_id=onboarding.exposed_action_id,
                    observed_outcome_id=onboarding.observed_outcome_id,
                    reaction_summary=onboarding.rendered_user_reaction,
                    evidence_ref=f"public-onboarding:{sha256_json(onboarding.to_payload())}",
                ),
            )
            owner_persistence = appended.owner_persistence_snapshot
            post_sha = social_record_store_persistence_sha256(owner_persistence)
            onboarding_write_change_count += int(post_sha != prior_sha)
            ledger.append(
                record_type="onboarding_append",
                payload={
                    "parent_durable_row_id": active_parent.ledger_row_id,
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "session_index": onboarding.session_index,
                    "session_id": onboarding.session_id,
                    "public_onboarding_sha256": sha256_json(onboarding.to_payload()),
                    "prior_owner_persistence_sha256": prior_sha,
                    "post_owner_persistence_sha256": post_sha,
                    "child_transition_count": 0,
                },
            )
            onboarding_count += 1
        if owner_persistence is None or len(root.onboarding_sessions) != 4:
            raise RuntimeError("theta0 v3 root did not replay four onboarding writes")
        segment_start = owner_persistence
        segment_start_sha = social_record_store_persistence_sha256(segment_start)
        settlements = []
        prior_postaction_sha: str | None = None

        for decision in root.decision_sessions[:8]:
            _require_active_parent(
                active=active_parent,
                ledger=ledger,
                protocol_id=protocol_id,
                implementation_git_commit=implementation_git_commit,
                parent_schedule=parent_schedule,
            )
            owner_input_sha = social_record_store_persistence_sha256(owner_persistence)
            if decision.decision_index == 0:
                if owner_input_sha != segment_start_sha:
                    raise RuntimeError("theta0 v3 first preaction lost onboarding state")
            else:
                if owner_input_sha != prior_postaction_sha:
                    raise RuntimeError("theta0 v3 owner handoff broke within root")
                owner_handoff_count += 1
            entry = child_schedule.entry_for_decision(decision.decision_id)
            assignment = RelationshipActionGateV2AssignmentReceipt(
                schedule_artifact=child_schedule,
                schedule_entry=entry,
            )
            authorization = RelationshipProductV2ForcedCollectionAuthorization(
                pulse_authorization=pulse_authorization,
                frozen_policy=frozen_policy,
                assignment=assignment,
            )
            temporal_delivery_timestamp = _temporal_delivery_timestamp(
                root_index,
                decision.decision_index,
            )
            if temporal_delivery_timestamp <= previous_temporal_delivery_timestamp:
                raise RuntimeError(
                    "theta0 v3 temporal delivery timestamps are not increasing"
                )
            previous_temporal_delivery_timestamp = temporal_delivery_timestamp
            preaction = await prepare_relationship_product_v2_forced_collection_preaction(
                request=_request(subject_id=root.subject_id, decision=decision),
                owner_persistence_snapshot=owner_persistence,
                forecast_runtime=dependencies.forecast_runtime,
                authorization=authorization,
                substrate_snapshot=cal._placeholder_substrate(),
                temporal_delivery_timestamp_ms=temporal_delivery_timestamp,
            )
            if (
                preaction.frozen_policy != frozen_policy
                or preaction.forced_exposure.assignment != assignment
                or preaction.execution_receipt.temporal_delivery.timestamp_ms
                != temporal_delivery_timestamp
                or preaction.delivered_action_id
                != (
                    preaction.forecast.recommended_action_id
                    if entry.assignment_role
                    is RelationshipActionGateV2AssignmentRole.CANDIDATE
                    else RelationshipAction.NEUTRAL_NOOP.value
                )
            ):
                raise RuntimeError("theta0 v3 v2 forced preaction drifted")
            if (
                preaction.forecast.forecast_id in forecast_ids
                or preaction.forced_exposure.exposure_id in exposure_ids
            ):
                raise RuntimeError("theta0 v3 preaction identity was reused")
            forecast_ids.add(preaction.forecast.forecast_id)
            exposure_ids.add(preaction.forced_exposure.exposure_id)
            ledger.append(
                record_type="preaction",
                payload={
                    "parent_durable_row_id": active_parent.ledger_row_id,
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "decision_index": decision.decision_index,
                    "decision_id": decision.decision_id,
                    "child_schedule_artifact_id": child_schedule.artifact_id,
                    "assignment_receipt_id": (
                        preaction.forced_exposure.assignment_receipt_id
                    ),
                    "assignment_role": entry.assignment_role.value,
                    "forecast_id": preaction.forecast.forecast_id,
                    "forced_exposure_id": preaction.forced_exposure.exposure_id,
                    "forced_receipt_id": preaction.execution_receipt.receipt_id,
                    "temporal_delivery_timestamp_ms": temporal_delivery_timestamp,
                    "delivered_action_id": preaction.delivered_action_id,
                    "owner_input_persistence_sha256": owner_input_sha,
                    "owner_forecast_poststate_sha256": (
                        social_record_store_persistence_sha256(
                            preaction.owner_persistence_snapshot
                        )
                    ),
                    "selected_branch_opened": False,
                    "sealed_truth_passed_to_forecast": False,
                    "child_transition_count": 0,
                },
            )
            preaction_count += 1
            if environment_scope is None:
                _require_active_parent(
                    active=active_parent,
                    ledger=ledger,
                    protocol_id=protocol_id,
                    implementation_git_commit=implementation_git_commit,
                    parent_schedule=parent_schedule,
                )
                environment_scope = _SelectedBranchEnvironmentScope(
                    dependencies=dependencies
                )
            branch = environment_scope.settle(
                public_root=root,
                public_decision=decision,
                selected_action=RelationshipAction(preaction.delivered_action_id),
            )
            if (
                branch.selected_action_id != preaction.delivered_action_id
                or branch.commitment_id in commitment_ids
            ):
                raise RuntimeError("theta0 v3 selected environment branch drifted")
            commitment_ids.add(branch.commitment_id)
            timestamp = _credit_timestamp(root_index, decision.decision_index)
            if timestamp - temporal_delivery_timestamp != 1:
                raise RuntimeError(
                    "theta0 v3 temporal delivery did not precede credit by one"
                )
            if timestamp <= previous_credit_timestamp:
                raise RuntimeError("theta0 v3 credit timestamps are not increasing")
            previous_credit_timestamp = timestamp
            action_turn = 4 + 2 * decision.decision_index
            settlement_input = replace(
                cal._settlement_input(
                    subject_scope=root.subject_id,
                    decision=decision,
                    forecast_id=preaction.forecast.forecast_id,
                    selected_action_id=preaction.delivered_action_id,
                    environment_outcome=branch,
                    action_turn=action_turn,
                    credit_timestamp=timestamp,
                ),
                apply_credit_to_gate=False,
            )
            settled = await settle_relationship_product_v2_forced_collection(
                preaction=preaction,
                settlement_input=settlement_input,
            )
            common = settled.common_baseline_credit
            if (
                settled.credit_applied_to_gate
                or settled.collection_gate_update_delta != 0
                or common.forecast.forecast_id != preaction.forecast.forecast_id
                or common.parent_action_credit.timestamp_ms != timestamp
                or common.record_id in credit_ids
                or settled.settlement.settlement_id in settlement_ids
            ):
                raise RuntimeError("theta0 v3 PE/common-credit settlement drifted")
            credit_ids.add(common.record_id)
            settlement_ids.add(settled.settlement.settlement_id)
            delivered_action_counts[preaction.delivered_action_id] += 1
            owner_preaction_sha = social_record_store_persistence_sha256(
                preaction.owner_persistence_snapshot
            )
            owner_postaction_sha = social_record_store_persistence_sha256(
                settled.owner_persistence_snapshot
            )
            owner_writeback_change_count += int(
                owner_preaction_sha != owner_postaction_sha
            )
            owner_persistence = settled.owner_persistence_snapshot
            prior_postaction_sha = owner_postaction_sha
            settlements.append(settled)
            ledger.append(
                record_type="postaction",
                payload={
                    "parent_durable_row_id": active_parent.ledger_row_id,
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "decision_index": decision.decision_index,
                    "decision_id": decision.decision_id,
                    "delivered_action_id": preaction.delivered_action_id,
                    "selected_branch_commitment_id": branch.commitment_id,
                    "typed_outcome_id": branch.typed_outcome_id,
                    "environment_evidence_ref": branch.environment_evidence_ref,
                    "settlement_id": settled.settlement.settlement_id,
                    "common_baseline_credit": common.to_payload(),
                    "owner_preaction_persistence_sha256": owner_preaction_sha,
                    "owner_postaction_persistence_sha256": owner_postaction_sha,
                    "selected_branch_opened_after_current_preaction_fsync": True,
                    "postaction_fsynced_before_next_preaction": True,
                    "credit_applied_online": False,
                    "child_transition_count": 0,
                    "evaluation_or_judge_feedback_received": False,
                },
            )
            postaction_count += 1

        segment = RelationshipProductV2CollectionSegment(
            segment_scope_id=root.subject_id,
            segment_start_owner_persistence_snapshot=segment_start,
            settlements=tuple(settlements),
        )
        child_collection = build_relationship_product_v2_segmented_collected_credit_batch(
            (segment,)
        )
        if (
            child_collection.gate_batch.schedule_artifact != child_schedule
            or len(child_collection.gate_batch.credits) != 8
        ):
            raise RuntimeError("theta0 v3 child collection did not exhaust its schedule")
        child_collections.append(child_collection)
        ledger.append(
            record_type="root_terminal",
            payload={
                "parent_durable_row_id": active_parent.ledger_row_id,
                "root_sequence_index": root_index,
                "subject_id": root.subject_id,
                "child_schedule_artifact_id": child_schedule.artifact_id,
                "segment_start_owner_persistence_sha256": segment_start_sha,
                "terminal_owner_persistence_sha256": (
                    social_record_store_persistence_sha256(owner_persistence)
                ),
                "onboarding_from_literal_none_count": 4,
                "child_collection": child_collection.to_payload(),
                "child_gate_batch": child_collection.gate_batch.to_payload(),
                "child_transition_count": 0,
            },
        )
        completed_root_count += 1

    if environment_scope is None:
        raise RuntimeError("theta0 v3 opened no selected-branch environment")
    federated_collection = build_relationship_product_v2_federated_collected_credit_batch(
        federated_schedule_artifact=parent_schedule,
        child_collected_batches=tuple(child_collections),
    )
    matched = commit_relationship_product_v2_federated_matched_gate_transitions(
        artifact=seed_artifact,
        collected_batch=federated_collection,
    )
    applied = matched.applied.gate_receipt
    withheld = matched.withheld.gate_receipt
    terminal_weights = matched.applied.terminal_checkpoint.weights
    terminal_parameter_finite = all(math.isfinite(value) for value in terminal_weights)
    terminal_parameter_nonzero = any(value != 0.0 for value in terminal_weights)
    child_transitions = _ChildTransitionObservation(
        apply_count=applied.child_transition_count,
        withhold_count=withheld.child_transition_count,
    )
    failure_reasons = _terminal_failure_reasons(
        completed_root_count=completed_root_count,
        onboarding_count=onboarding_count,
        onboarding_write_change_count=onboarding_write_change_count,
        preaction_count=preaction_count,
        postaction_count=postaction_count,
        owner_handoff_count=owner_handoff_count,
        owner_writeback_change_count=owner_writeback_change_count,
        child_collection_count=len(child_collections),
        unique_child_collection_count=len(
            {item.collection_id for item in child_collections}
        ),
        unique_credit_count=len(credit_ids),
        federated_credit_count=len(federated_collection.gate_batch.credits),
        apply_child_batch_count=applied.child_batch_count,
        apply_child_transition_count=applied.child_transition_count,
        apply_credit_count=applied.credit_count,
        apply_atomic_commit_count=applied.atomic_commit_count,
        apply_update_count_delta=applied.update_count_delta,
        apply_informative_update_count_delta=(
            applied.informative_update_count_delta
        ),
        apply_cap_hit_count=applied.cap_hit_count,
        withhold_child_transition_count=withheld.child_transition_count,
        withhold_atomic_commit_count=withheld.atomic_commit_count,
        withhold_update_count_delta=withheld.update_count_delta,
        withhold_checkpoint_unchanged=(
            matched.withheld.terminal_checkpoint == frozen_policy.checkpoint
        ),
        terminal_parameter_finite=terminal_parameter_finite,
        terminal_parameter_nonzero=terminal_parameter_nonzero,
    )
    learned_theta0 = None
    if not failure_reasons:
        learned_theta0 = RelationshipActionGateV2Artifact.create_learned_theta0_from_federation(
            parent_artifact=seed_artifact,
            source_batch=federated_collection.gate_batch,
            apply_receipt=applied,
        )
        cold = RelationshipActionGateV2(artifact=learned_theta0).export_checkpoint()
        if (
            cold.weights != matched.applied.terminal_checkpoint.weights
            or cold.update_count != 0
            or cold.informative_update_count != 0
            or cold.processed_credit_ids
        ):
            raise RuntimeError("theta0 v3 cold condensation replay drifted")
    status = _SUCCESS_STATUS if learned_theta0 is not None else _FAIL_STATUS
    ledger.append(
        record_type="parent_transition_pair_committed",
        payload={
            "parent_durable_row_id": active_parent.ledger_row_id,
            "federated_collection_id": federated_collection.collection_id,
            "federated_gate_batch_id": federated_collection.gate_batch.batch_id,
            "pulse_matched_transitions_id": matched.transitions_id,
            "gate_matched_transitions_id": matched.gate_matched_transitions.transitions_id,
            "apply_receipt": applied.to_payload(),
            "withhold_receipt": withheld.to_payload(),
            "apply_terminal_checkpoint": matched.applied.terminal_checkpoint.to_payload(),
            "withhold_terminal_checkpoint": matched.withheld.terminal_checkpoint.to_payload(),
            "parent_transition_pair_count": 1,
            "apply_child_transition_count": child_transitions.apply_count,
            "withhold_child_transition_count": child_transitions.withhold_count,
            "child_transition_count": child_transitions.total_count,
        },
    )
    ledger.append(
        record_type="terminal",
        payload={
            "parent_durable_row_id": active_parent.ledger_row_id,
            "completed_root_count": completed_root_count,
            "onboarding_count": onboarding_count,
            "onboarding_write_change_count": onboarding_write_change_count,
            "preaction_count": preaction_count,
            "postaction_count": postaction_count,
            "owner_handoff_count": owner_handoff_count,
            "owner_writeback_change_count": owner_writeback_change_count,
            "unique_forecast_count": len(forecast_ids),
            "unique_exposure_count": len(exposure_ids),
            "unique_credit_count": len(credit_ids),
            "unique_settlement_count": len(settlement_ids),
            "unique_selected_branch_commitment_count": len(commitment_ids),
            "unique_child_collection_count": len(
                {item.collection_id for item in child_collections}
            ),
            "delivered_action_counts": dict(sorted(delivered_action_counts.items())),
            "parent_transition_pair_count": 1,
            "apply_child_transition_count": child_transitions.apply_count,
            "withhold_child_transition_count": child_transitions.withhold_count,
            "child_transition_count": child_transitions.total_count,
            "terminal_failure_reasons": list(failure_reasons),
            "published_theta0_artifact_id": (
                None if learned_theta0 is None else learned_theta0.artifact_id
            ),
            "terminal_status": status,
            "model_output_count": 0,
            "cuda_execution_count": 0,
            "campaign_decision_count": 0,
            "effect_estimand_count": 0,
        },
    )
    return _BootstrapReplay(
        implementation_checkout=implementation_checkout,
        seed_artifact=seed_artifact,
        parent_schedule=parent_schedule,
        child_collections=tuple(child_collections),
        federated_collection=federated_collection,
        matched_transitions=matched,
        learned_theta0=learned_theta0,
        parent_ledger_row_id=active_parent.ledger_row_id,
        completed_root_count=completed_root_count,
        onboarding_count=onboarding_count,
        onboarding_write_change_count=onboarding_write_change_count,
        preaction_count=preaction_count,
        postaction_count=postaction_count,
        owner_handoff_count=owner_handoff_count,
        owner_writeback_change_count=owner_writeback_change_count,
        child_transition_observation=child_transitions,
        delivered_action_counts=dict(sorted(delivered_action_counts.items())),
        terminal_failure_reasons=failure_reasons,
        terminal_status=status,
    )


def _validate_frozen_public_inputs(
    *,
    dependencies: _Dependencies,
    seed_artifact: RelationshipActionGateV2Artifact,
    parent_schedule: RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
) -> None:
    schedule_pin = cal._mapping(
        dependencies.protocol.payload["federated_schedule"],
        "federated_schedule",
    )
    parent_raw = _canonical_bytes(parent_schedule.to_payload())
    ordered_child_ids = [
        item.child_schedule_artifact.artifact_id
        for item in parent_schedule.segments
    ]
    if (
        parent_schedule.source_artifact_id != schedule_pin["source_artifact_id"]
        or parent_schedule.schedule_scope_id
        != schedule_pin["parent_schedule_scope_id"]
        or parent_schedule.artifact_id
        != schedule_pin["expected_parent_schedule_artifact_id"]
        or _sha256_bytes(parent_raw)
        != schedule_pin["expected_parent_schedule_raw_sha256"]
        or len(parent_raw) != schedule_pin["expected_parent_schedule_raw_bytes"]
        or sha256_json(ordered_child_ids)
        != schedule_pin["ordered_child_schedule_ids_sha256"]
        or len(parent_schedule.segments) != 112
        or len(parent_schedule.flattened_entries) != 896
    ):
        raise ValueError("theta0 v3 frozen parent schedule pin drifted")
    gate_pin = cal._mapping(dependencies.protocol.payload["gate_v2"], "gate_v2")
    if seed_artifact.artifact_id != gate_pin["expected_seed_artifact_id"]:
        raise ValueError("theta0 v3 frozen seed pin drifted")


def _transition_bundle_payload(replay: _BootstrapReplay) -> Mapping[str, object]:
    applied = replay.matched_transitions.applied
    withheld = replay.matched_transitions.withheld
    return {
        "schema_version": THETA0_V3_BOOTSTRAP_TRANSITION_BUNDLE_SCHEMA_VERSION,
        "seed_artifact": replay.seed_artifact.to_payload(),
        "parent_schedule_artifact_id": replay.parent_schedule.artifact_id,
        "ordered_child_collection_ids": [
            item.collection_id for item in replay.child_collections
        ],
        "child_collections": [
            item.to_payload() for item in replay.child_collections
        ],
        "child_gate_batches": [
            item.gate_batch.to_payload() for item in replay.child_collections
        ],
        "federated_collection": replay.federated_collection.to_payload(),
        "federated_gate_batch": replay.federated_collection.gate_batch.to_payload(),
        "pulse_matched_transitions": replay.matched_transitions.to_payload(),
        "gate_matched_transitions": (
            replay.matched_transitions.gate_matched_transitions.to_payload()
        ),
        "apply_transition": applied.to_payload(),
        "apply_receipt": applied.gate_receipt.to_payload(),
        "apply_terminal_checkpoint": applied.terminal_checkpoint.to_payload(),
        "withhold_transition": withheld.to_payload(),
        "withhold_receipt": withheld.gate_receipt.to_payload(),
        "withhold_terminal_checkpoint": withheld.terminal_checkpoint.to_payload(),
        "parent_transition_pair_count": 1,
        "apply_child_transition_count": (
            replay.child_transition_observation.apply_count
        ),
        "withhold_child_transition_count": (
            replay.child_transition_observation.withhold_count
        ),
        "child_transition_count": replay.child_transition_observation.total_count,
        "child_batch_count": len(
            replay.federated_collection.gate_batch.child_batches
        ),
        "credit_count": len(replay.federated_collection.gate_batch.credits),
        "learned_theta0": (
            None
            if replay.learned_theta0 is None
            else replay.learned_theta0.to_payload()
        ),
        "terminal_failure_reasons": list(replay.terminal_failure_reasons),
        "terminal_status": replay.terminal_status,
    }


def _write_replay_files(*, root: pathlib.Path, replay: _BootstrapReplay) -> None:
    _write_and_reopen_exact(
        root / _TRANSITION_FILENAME,
        _canonical_bytes(_transition_bundle_payload(replay)),
    )
    if replay.learned_theta0 is not None:
        _write_and_reopen_exact(
            root / _THETA_FILENAME,
            _canonical_bytes(replay.learned_theta0.to_payload()),
        )


def _build_manifest(
    *,
    root: pathlib.Path,
    dependencies: _Dependencies,
    replay: _BootstrapReplay,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    paths = [
        "protocol.json",
        _SCHEDULE_FILENAME,
        _TRACE_FILENAME,
        _TRANSITION_FILENAME,
    ]
    if replay.learned_theta0 is not None:
        paths.append(_THETA_FILENAME)
    files = []
    for relative in paths:
        raw = cal._read_regular(root / relative)
        files.append(
            {
                "path": relative,
                "raw_bytes": len(raw),
                "raw_sha256": _sha256_bytes(raw),
            }
        )
    applied = replay.matched_transitions.applied
    withheld = replay.matched_transitions.withheld
    root_reset_closed = (
        replay.completed_root_count == 112
        and replay.onboarding_count == 448
        and replay.onboarding_write_change_count == 448
    )
    claims = {
        "parent_preoutcome_durable_owner_path_receipt_established": True,
        "root_reset_replay_established": root_reset_closed,
        "no_child_transition_accepted_lineage_established": (
            replay.child_transition_observation.accepted_lineage_has_no_child_transition
        ),
        "development_theta0_v3_materialized": replay.learned_theta0 is not None,
        "treatment_reachability_admitted": False,
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
    }
    core = {
        "schema_version": THETA0_V3_BOOTSTRAP_MANIFEST_SCHEMA_VERSION,
        "protocol_id": dependencies.protocol.protocol_id,
        "protocol_raw_sha256": dependencies.protocol.raw_sha256,
        "implementation_git_commit": implementation_git_commit,
        "implementation_owned_blob_ids": [
            {"path": path, "git_blob_id": blob_id}
            for path, blob_id in replay.implementation_checkout.owned_blob_ids
        ],
        "implementation_owned_blob_ids_sha256": (
            replay.implementation_checkout.owned_blob_ids_sha256
        ),
        "implementation_head_matched_before_output_creation_owner_receipt": True,
        "implementation_clean_scope_before_output_creation_owner_receipt": True,
        "source_v4_admission_artifact_id": dependencies.protocol.payload[
            "source_v4_admission"
        ]["artifact_id"],
        "source_v4_public_plan_sha256": dependencies.public_view.public_plan_sha256,
        "development_reader_artifact_id": dependencies.protocol.payload[
            "development_reader"
        ]["artifact_id"],
        "condition_reader_qualified": False,
        "seed_artifact_id": replay.seed_artifact.artifact_id,
        "seed_checkpoint_content_sha256": RelationshipActionGateV2(
            artifact=replay.seed_artifact
        ).export_checkpoint().content_sha256,
        "parent_schedule_artifact_id": replay.parent_schedule.artifact_id,
        "parent_schedule_raw_sha256": _sha256_bytes(
            cal._read_regular(root / _SCHEDULE_FILENAME)
        ),
        "parent_durable_ledger_row_id": replay.parent_ledger_row_id,
        "child_schedule_count": len(replay.parent_schedule.segments),
        "child_collection_count": len(replay.child_collections),
        "federated_collection_id": replay.federated_collection.collection_id,
        "federated_gate_batch_id": replay.federated_collection.gate_batch.batch_id,
        "pulse_matched_transitions_id": replay.matched_transitions.transitions_id,
        "gate_matched_transitions_id": (
            replay.matched_transitions.gate_matched_transitions.transitions_id
        ),
        "apply_transition_id": applied.transition_id,
        "apply_receipt_id": applied.gate_receipt.receipt_id,
        "apply_terminal_checkpoint_content_sha256": (
            applied.terminal_checkpoint.content_sha256
        ),
        "apply_update_count_delta": applied.gate_receipt.update_count_delta,
        "apply_informative_update_count_delta": (
            applied.gate_receipt.informative_update_count_delta
        ),
        "apply_atomic_commit_count": applied.gate_receipt.atomic_commit_count,
        "apply_cap_hit_count": applied.gate_receipt.cap_hit_count,
        "withhold_transition_id": withheld.transition_id,
        "withhold_receipt_id": withheld.gate_receipt.receipt_id,
        "withhold_terminal_checkpoint_content_sha256": (
            withheld.terminal_checkpoint.content_sha256
        ),
        "withhold_update_count_delta": withheld.gate_receipt.update_count_delta,
        "withhold_atomic_commit_count": withheld.gate_receipt.atomic_commit_count,
        "parent_transition_pair_count": 1,
        "apply_child_transition_count": (
            replay.child_transition_observation.apply_count
        ),
        "withhold_child_transition_count": (
            replay.child_transition_observation.withhold_count
        ),
        "child_transition_count": replay.child_transition_observation.total_count,
        "completed_root_count": replay.completed_root_count,
        "onboarding_count": replay.onboarding_count,
        "onboarding_write_change_count": replay.onboarding_write_change_count,
        "preaction_count": replay.preaction_count,
        "postaction_count": replay.postaction_count,
        "owner_handoff_count": replay.owner_handoff_count,
        "owner_writeback_change_count": replay.owner_writeback_change_count,
        "temporal_delivery_timestamp_clock": (
            "protocol_frozen_offline_logical_milliseconds"
        ),
        "temporal_delivery_timestamp_formula": (
            "root_index_times_20_plus_4_plus_2_times_decision_index"
        ),
        "temporal_delivery_timestamp_count": replay.preaction_count,
        "temporal_delivery_timestamp_min": _temporal_delivery_timestamp(0, 0),
        "temporal_delivery_timestamp_max": _temporal_delivery_timestamp(111, 7),
        "temporal_delivery_precedes_credit_timestamp_by_ms": 1,
        "credit_count": len(replay.federated_collection.gate_batch.credits),
        "delivered_action_counts": replay.delivered_action_counts,
        "published_theta0_artifact_id": (
            None
            if replay.learned_theta0 is None
            else replay.learned_theta0.artifact_id
        ),
        "terminal_failure_reasons": list(replay.terminal_failure_reasons),
        "parent_schedule_create_only_fsync_reopen_exact_owner_receipt": True,
        "parent_receipt_precedes_all_onboarding_forecast_and_outcome_owner_trace": True,
        "sealed_source_opened_after_first_preaction_fsync_owner_trace": True,
        "historical_fsync_independently_reproven_by_validate_existing": False,
        "accepted_lineage_no_child_transition": (
            replay.child_transition_observation.accepted_lineage_has_no_child_transition
        ),
        "accepted_lineage_no_child_transition_is_not_os_secrecy": True,
        "windows_directory_entry_durability_attested": False,
        "legacy_theta0_v2_input_count": 0,
        "legacy_forced_common_outcome_or_credit_file_read_count": 0,
        "model_output_count": 0,
        "cuda_execution_count": 0,
        "campaign_decision_count": 0,
        "effect_estimand_count": 0,
        "files": files,
        "status": replay.terminal_status,
        "claims": claims,
        "claim_boundary": dependencies.protocol.payload["claim_boundary"],
    }
    return {"artifact_id": sha256_json(core), **core}


def materialize_relationship_product_horizon_theta0_v3_bootstrap(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    protocol = load_relationship_product_horizon_theta0_v3_bootstrap_protocol()
    implementation_checkout = _verify_implementation_checkout(
        protocol=protocol,
        implementation_git_commit=implementation_git_commit,
    )
    commit = implementation_checkout.implementation_git_commit
    root = pathlib.Path(output_dir)
    _require_disjoint_output_root(
        output_root=root,
        input_roots=(
            pathlib.Path(source_v4_admission_root),
            pathlib.Path(reader_root),
            *_OUTPUT_FORBIDDEN_REPOSITORY_ROOTS,
        ),
    )
    if root.exists():
        raise FileExistsError(f"theta0 v3 bootstrap root is create-only: {root}")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
    )
    seed = _build_seed(dependencies.protocol)
    parent_schedule, child_schedules = _build_federated_schedule(
        dependencies.public_view
    )
    _validate_frozen_public_inputs(
        dependencies=dependencies,
        seed_artifact=seed,
        parent_schedule=parent_schedule,
    )
    root.mkdir(parents=True, exist_ok=False)
    _write_and_reopen_exact(root / "protocol.json", dependencies.protocol.raw_bytes)
    parent_raw = _canonical_bytes(parent_schedule.to_payload())
    _write_and_reopen_exact(root / _SCHEDULE_FILENAME, parent_raw)
    durable_parent = _durable_parent_receipt(
        protocol_id=dependencies.protocol.protocol_id,
        implementation_git_commit=commit,
        implementation_checkout=implementation_checkout,
        parent_schedule=parent_schedule,
        persisted_raw=cal._read_regular(root / _SCHEDULE_FILENAME),
    )
    sink = cal._FsyncTraceSink(root / _TRACE_FILENAME)
    ledger = _Ledger(sink)
    try:
        replay = asyncio.run(
            _run_bootstrap(
                dependencies=dependencies,
                seed_artifact=seed,
                parent_schedule=parent_schedule,
                child_schedules=child_schedules,
                durable_parent=durable_parent,
                implementation_checkout=implementation_checkout,
                implementation_git_commit=commit,
                ledger=ledger,
            )
        )
    finally:
        sink.close()
    _require_closed_trace_exact(path=root / _TRACE_FILENAME, ledger=ledger)
    _write_replay_files(root=root, replay=replay)
    manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        replay=replay,
        implementation_git_commit=commit,
    )
    _write_and_reopen_exact(root / _MANIFEST_FILENAME, _canonical_bytes(manifest))
    return manifest


def _filesystem_fingerprint(root: pathlib.Path) -> Mapping[str, tuple[int, int, str]]:
    return {
        relative: (
            (root / relative).stat().st_size,
            (root / relative).stat().st_mtime_ns,
            _sha256_bytes(cal._read_regular(root / relative)),
        )
        for relative in sorted(cal._regular_file_inventory(root))
    }


def _validate_artifact(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
    cross_commit_compatibility_replay: bool = False,
) -> tuple[Mapping[str, object], _BootstrapReplay]:
    external_protocol = cal._digest(expected_protocol_id, "expected_protocol_id")
    external_artifact = cal._digest(expected_artifact_id, "expected_artifact_id")
    root = pathlib.Path(output_dir)
    _require_disjoint_output_root(
        output_root=root,
        input_roots=(
            pathlib.Path(source_v4_admission_root),
            pathlib.Path(reader_root),
            *_OUTPUT_FORBIDDEN_REPOSITORY_ROOTS,
        ),
    )
    before = _filesystem_fingerprint(root)
    manifest_raw = cal._read_regular(root / _MANIFEST_FILENAME)
    manifest = cal._parse_json_bytes(manifest_raw, source="theta0 v3 manifest")
    if manifest_raw != _canonical_bytes(manifest):
        raise ValueError("theta0 v3 manifest must use canonical bytes")
    if (
        manifest["protocol_id"] != external_protocol
        or manifest["artifact_id"] != external_artifact
        or manifest["artifact_id"]
        != sha256_json(
            {key: value for key, value in manifest.items() if key != "artifact_id"}
        )
    ):
        raise ValueError("theta0 v3 external identity drifted")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
    )
    if dependencies.protocol.protocol_id != external_protocol:
        raise ValueError("theta0 v3 packaged protocol identity drifted")
    if cal._read_regular(root / "protocol.json") != dependencies.protocol.raw_bytes:
        raise ValueError("theta0 v3 persisted protocol bytes drifted")
    seed = _build_seed(dependencies.protocol)
    parent_schedule, child_schedules = _build_federated_schedule(
        dependencies.public_view
    )
    _validate_frozen_public_inputs(
        dependencies=dependencies,
        seed_artifact=seed,
        parent_schedule=parent_schedule,
    )
    parent_raw = cal._read_regular(root / _SCHEDULE_FILENAME)
    if parent_raw != _canonical_bytes(parent_schedule.to_payload()):
        raise ValueError("theta0 v3 parent schedule bytes drifted")
    commit = cal._git_commit(manifest["implementation_git_commit"])
    if cross_commit_compatibility_replay:
        implementation_checkout = _verify_historical_implementation_lineage(
            protocol=dependencies.protocol,
            implementation_git_commit=commit,
        )
    else:
        implementation_checkout = _verify_implementation_checkout(
            protocol=dependencies.protocol,
            implementation_git_commit=commit,
        )
    expected_owned_blob_ids = [
        {"path": path, "git_blob_id": blob_id}
        for path, blob_id in implementation_checkout.owned_blob_ids
    ]
    if (
        manifest["implementation_owned_blob_ids"] != expected_owned_blob_ids
        or manifest["implementation_owned_blob_ids_sha256"]
        != implementation_checkout.owned_blob_ids_sha256
    ):
        raise ValueError("theta0 v3 implementation blob lineage drifted")
    durable_parent = _durable_parent_receipt(
        protocol_id=dependencies.protocol.protocol_id,
        implementation_git_commit=commit,
        implementation_checkout=implementation_checkout,
        parent_schedule=parent_schedule,
        persisted_raw=parent_raw,
    )
    sink = cal._MemoryTraceSink()
    ledger = _Ledger(sink)
    replay = asyncio.run(
        _run_bootstrap(
            dependencies=dependencies,
            seed_artifact=seed,
            parent_schedule=parent_schedule,
            child_schedules=child_schedules,
            durable_parent=durable_parent,
            implementation_checkout=implementation_checkout,
            implementation_git_commit=commit,
            ledger=ledger,
        )
    )
    if cal._read_regular(root / _TRACE_FILENAME) != ledger.raw_bytes:
        raise ValueError("theta0 v3 deterministic ledger bytes drifted")
    if cal._read_regular(root / _TRANSITION_FILENAME) != _canonical_bytes(
        _transition_bundle_payload(replay)
    ):
        raise ValueError("theta0 v3 transition bundle bytes drifted")
    if replay.learned_theta0 is not None:
        if cal._read_regular(root / _THETA_FILENAME) != _canonical_bytes(
            replay.learned_theta0.to_payload()
        ):
            raise ValueError("theta0 v3 learned artifact bytes drifted")
    expected_files = (
        _SUCCESS_OUTPUT_FILES
        if replay.learned_theta0 is not None
        else _BASE_OUTPUT_FILES
    )
    if cal._regular_file_inventory(root) != expected_files:
        raise ValueError("theta0 v3 output file inventory drifted")
    expected_manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        replay=replay,
        implementation_git_commit=commit,
    )
    if manifest != expected_manifest or manifest_raw != _canonical_bytes(
        expected_manifest
    ):
        raise ValueError("theta0 v3 manifest content drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("theta0 v3 artifact identity drifted")
    after = _filesystem_fingerprint(root)
    if after != before:
        raise RuntimeError("theta0 v3 validate-existing modified the artifact")
    return manifest, replay


def validate_relationship_product_horizon_theta0_v3_bootstrap(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    manifest, _replay = _validate_artifact(
        source_v4_admission_root=source_v4_admission_root,
        reader_root=reader_root,
        output_dir=output_dir,
        expected_protocol_id=expected_protocol_id,
        expected_artifact_id=expected_artifact_id,
    )
    return manifest


def load_relationship_product_horizon_theta0_v3_bundle(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> RelationshipProductHorizonTheta0V3Bundle:
    manifest, replay = _validate_artifact(
        source_v4_admission_root=source_v4_admission_root,
        reader_root=reader_root,
        output_dir=output_dir,
        expected_protocol_id=expected_protocol_id,
        expected_artifact_id=expected_artifact_id,
    )
    if replay.learned_theta0 is None:
        raise ValueError("theta0 v3 failed artifact has no consumable theta0")
    return RelationshipProductHorizonTheta0V3Bundle(
        manifest=manifest,
        theta0_artifact=replay.learned_theta0,
        federated_collection=replay.federated_collection,
        matched_transitions=replay.matched_transitions,
    )


def _replay_relationship_product_horizon_theta0_v3_bundle_for_cross_commit_handoff(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> RelationshipProductHorizonTheta0V3Bundle:
    """Rebuild the full typed graph under current code for a handoff owner.

    This is a compatibility replay, not a replacement for the frozen
    protocol's exact-HEAD ``validate-existing``.  A consumer must separately
    bind an accepted historical validation receipt before using the result.
    """

    manifest, replay = _validate_artifact(
        source_v4_admission_root=source_v4_admission_root,
        reader_root=reader_root,
        output_dir=output_dir,
        expected_protocol_id=expected_protocol_id,
        expected_artifact_id=expected_artifact_id,
        cross_commit_compatibility_replay=True,
    )
    if replay.learned_theta0 is None:
        raise ValueError("theta0 v3 failed artifact has no handoff-compatible theta0")
    return RelationshipProductHorizonTheta0V3Bundle(
        manifest=manifest,
        theta0_artifact=replay.learned_theta0,
        federated_collection=replay.federated_collection,
        matched_transitions=replay.matched_transitions,
    )


__all__ = [
    "RelationshipProductHorizonTheta0V3BootstrapProtocol",
    "RelationshipProductHorizonTheta0V3Bundle",
    "THETA0_V3_BOOTSTRAP_MANIFEST_SCHEMA_VERSION",
    "THETA0_V3_BOOTSTRAP_PROTOCOL_ID",
    "THETA0_V3_BOOTSTRAP_PROTOCOL_SCHEMA_VERSION",
    "load_relationship_product_horizon_theta0_v3_bootstrap_protocol",
    "load_relationship_product_horizon_theta0_v3_bundle",
    "materialize_relationship_product_horizon_theta0_v3_bootstrap",
    "relationship_product_horizon_theta0_v3_bootstrap_protocol_path",
    "validate_relationship_product_horizon_theta0_v3_bootstrap",
]
