"""Development-only 112-root Product Horizon causal campaign.

This owner consumes the already frozen forced common-batch artifact and runs
three immutable evaluation arms.  It estimates only synthetic, transductive
development contrasts.  It never qualifies the reader, invokes a model or
CUDA, applies evaluation credit to a gate, or authorizes formal evidence.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
import pathlib
import time
from typing import BinaryIO, Callable, Mapping, Protocol, Sequence

from lifeform_domain_emogpt.lab.contracts import sha256_json
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    HorizonPublicDecisionSession,
    HorizonPublicRoot,
)
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RELATIONSHIP_PRODUCT_EXECUTOR_SCHEMA_VERSION,
    RelationshipProductExecutorCommand,
    RelationshipProductExecutorReceipt,
    RelationshipProductFrozenPreActionSnapshot,
    RelationshipProductFrozenPulseAuthorization,
    RelationshipProductFrozenSettlementSnapshot,
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    RelationshipProductSettlementInput,
    RelationshipProductTemporalDelivery,
    prepare_relationship_product_frozen_preaction,
    settle_relationship_product_frozen_pulse,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_evolution import (
    relationship_product_horizon_dynamic_collection_prefix as dynamic,
)
from lifeform_evolution import relationship_product_horizon_theta0_calibration as cal
from lifeform_evolution.relationship_product_horizon_forced_common_batch import (
    RelationshipProductHorizonCampaignArm,
    RelationshipProductHorizonCampaignArmInitialization,
    RelationshipProductHorizonForcedCampaignInputs,
    load_relationship_product_horizon_forced_campaign_inputs,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.credit import derive_preference_action_forecast_credit_records
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot
from volvence_zero.social import (
    PreferenceAboutOtherModule,
    PreferenceActionForecastRequest,
    SocialRecordStore,
    settle_preference_action_forecast,
    social_record_store_persistence_sha256,
)
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    PreferenceActionForecastSettlement,
    PreferenceActionOutcomeEvidence,
    SocialPredictionError,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialScopeKind,
    preference_action_forecast_from_payload,
    preference_action_forecast_to_payload,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind
from volvence_zero.temporal_types import (
    TemporalActionAdvisoryProposal,
    TemporalActionAdvisoryStatus,
)


DEVELOPMENT_CAMPAIGN_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-development-campaign-protocol.v1"
)
DEVELOPMENT_CAMPAIGN_PLAN_SCHEMA_VERSION = (
    "relationship-product-horizon-development-campaign-plan.v1"
)
DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION = (
    "relationship-product-horizon-development-campaign-trace.v1"
)
DEVELOPMENT_CAMPAIGN_TERMINAL_STATE_SCHEMA_VERSION = (
    "relationship-product-horizon-development-campaign-terminal-state.v1"
)
DEVELOPMENT_CAMPAIGN_REPORT_SCHEMA_VERSION = (
    "relationship-product-horizon-development-campaign-report.v1"
)
DEVELOPMENT_CAMPAIGN_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-development-campaign-manifest.v1"
)
DEVELOPMENT_CAMPAIGN_BARRIER_SCHEMA_VERSION = (
    "relationship-product-horizon-development-campaign-barrier.v1"
)

_PROTOCOL_FILENAME = "relationship_product_horizon_development_campaign_v1.json"
_PLAN_FILENAME = "campaign_plan.json"
_TRACE_FILENAME = "campaign_trace.jsonl"
_TERMINAL_STATE_FILENAME = "root_arm_terminal_states.jsonl"
_REPORT_FILENAME = "report.json"
_OUTPUT_FILES = frozenset(
    {
        "protocol.json",
        _PLAN_FILENAME,
        _TRACE_FILENAME,
        _TERMINAL_STATE_FILENAME,
        _REPORT_FILENAME,
        "manifest.json",
    }
)
_ARM_IDS = (
    RelationshipProductHorizonCampaignArm.FULL,
    RelationshipProductHorizonCampaignArm.FROZEN_THETA0,
    RelationshipProductHorizonCampaignArm.STRICT_NOOP,
)
_ARM_ORDERS = (
    _ARM_IDS,
    (_ARM_IDS[0], _ARM_IDS[2], _ARM_IDS[1]),
    (_ARM_IDS[1], _ARM_IDS[0], _ARM_IDS[2]),
    (_ARM_IDS[1], _ARM_IDS[2], _ARM_IDS[0]),
    (_ARM_IDS[2], _ARM_IDS[0], _ARM_IDS[1]),
    (_ARM_IDS[2], _ARM_IDS[1], _ARM_IDS[0]),
)
_SEGMENTS = (
    ("post_reversal", tuple(range(8, 16))),
    ("correction", tuple(range(16, 24))),
    ("post_correction", tuple(range(24, 32))),
    ("return_after_gap", tuple(range(32, 40))),
    ("mixed_stress", tuple(range(40, 48))),
)
_EVALUATION_INDICES = tuple(range(8, 48))
_INTERLOCUTOR_ID = "primary"
_POSITIVE_OUTCOMES = frozenset(
    {DialogueExternalOutcomeKind.HELPED, DialogueExternalOutcomeKind.FELT_HEARD}
)
_SUCCESS_BOTH = (
    "development_campaign_completed_both_contrasts_go_candidate_no_effect_claim"
)
_SUCCESS_SINGLE = (
    "development_campaign_completed_single_contrast_go_candidate_no_effect_claim"
)
_COMPLETE_STOP = "development_campaign_completed_stop_no_effect_claim"
_COMPLETE_INVALID = "development_campaign_completed_contrast_invalid_no_claim"
_EXECUTOR_COMMAND_PREFIX = "relationship-product-executor-command-sha256:"
_EXECUTOR_RECEIPT_PREFIX = "relationship-product-executor-receipt-sha256:"
_LEARNABLE_CONTRAST_ID = "learnable_full_minus_frozen_theta0"
_STEERABLE_CONTRAST_ID = "steerable_frozen_theta0_minus_strict_noop"
_BOOTSTRAP_REPLICATE_ZERO_FIRST_TWELVE = (
    15,
    80,
    50,
    109,
    99,
    79,
    56,
    99,
    25,
    103,
    104,
    36,
)

# Filled from the final packaged bytes before the protocol/implementation
# commit.  A non-matching custom path fails closed.
_EXPECTED_PROTOCOL_ID = (
    "bc4c0882c6b00f445534d0a2136d8018da688ccecd6b73c2a08066faadbd587c"
)
_EXPECTED_PROTOCOL_RAW_SHA256 = (
    "9374a71ba45ce7d16a0778c76618da0a21f9da0bf357f1352a008cd8499b19fa"
)


@dataclass(frozen=True)
class RelationshipProductHorizonDevelopmentCampaignProtocol:
    payload: Mapping[str, object]
    raw_bytes: bytes
    protocol_id: str
    raw_sha256: str


@dataclass(frozen=True)
class _Dependencies:
    protocol: RelationshipProductHorizonDevelopmentCampaignProtocol
    inputs: RelationshipProductHorizonForcedCampaignInputs
    source_v4_admission_root: pathlib.Path
    reader_root: pathlib.Path
    theta0_v2_root: pathlib.Path
    scanner_root: pathlib.Path
    dynamic_root: pathlib.Path


@dataclass(frozen=True)
class _DurableRows:
    start_index: int
    end_index: int
    row_ids: tuple[str, ...]
    rows_raw_sha256: str
    byte_offset_start: int
    byte_offset_end: int
    stream_prefix_raw_sha256: str


@dataclass(frozen=True)
class _PreactionBarrierReceipt:
    barrier_id: str
    receipt_row_id: str
    root_sequence_index: int
    decision_index: int
    arm_order: tuple[RelationshipProductHorizonCampaignArm, ...]
    preaction_receipt_ids: tuple[str, ...]
    stream_prefix_raw_sha256: str


@dataclass(frozen=True)
class _DurablePreactions:
    barrier: _PreactionBarrierReceipt
    preactions: tuple[
        tuple[
            RelationshipProductHorizonCampaignArm,
            RelationshipProductFrozenPreActionSnapshot,
        ],
        ...,
    ]


@dataclass(frozen=True)
class _ArmSettlement:
    arm_id: RelationshipProductHorizonCampaignArm
    outcome: dynamic.RelationshipProductHorizonSelectedBranchOutcome
    settled: RelationshipProductFrozenSettlementSnapshot


@dataclass(frozen=True)
class _DecisionRecord:
    root_sequence_index: int
    decision_index: int
    segment_id: str
    arm_id: RelationshipProductHorizonCampaignArm
    candidate_action: RelationshipAction
    delivered_action: RelationshipAction
    outcome: DialogueExternalOutcomeKind


@dataclass(frozen=True)
class _CampaignReplay:
    report: Mapping[str, object]
    trace_row_count: int
    trace_raw_bytes: int
    trace_raw_sha256: str
    terminal_state_row_count: int
    terminal_state_raw_bytes: int
    terminal_state_raw_sha256: str


class _RowSink(Protocol):
    @property
    def row_count(self) -> int: ...

    @property
    def raw_bytes(self) -> int: ...

    @property
    def raw_sha256(self) -> str: ...

    def append_many_fsync(
        self, rows: Sequence[Mapping[str, object]]
    ) -> _DurableRows: ...

    def close(self) -> None: ...


class _DigestRowSink:
    """Streaming canonical JSONL digest with no linear raw-byte cache."""

    def __init__(self) -> None:
        self._row_count = 0
        self._raw_bytes = 0
        self._digest = hashlib.sha256()

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def raw_bytes(self) -> int:
        return self._raw_bytes

    @property
    def raw_sha256(self) -> str:
        return self._digest.hexdigest()

    def _write(self, raw: bytes) -> None:
        del raw

    def _sync(self) -> None:
        return

    def append_many_fsync(
        self, rows: Sequence[Mapping[str, object]]
    ) -> _DurableRows:
        if not rows:
            raise ValueError("durable row group cannot be empty")
        start_index = self._row_count
        start_offset = self._raw_bytes
        group_digest = hashlib.sha256()
        row_ids: list[str] = []
        for payload in rows:
            core = {"physical_sequence_index": self._row_count, **payload}
            if "row_id" in core:
                raise ValueError("row_id is owned by the streaming sink")
            row_id = sha256_json(core)
            raw = cal._canonical_bytes({"row_id": row_id, **core})
            self._write(raw)
            self._digest.update(raw)
            group_digest.update(raw)
            self._raw_bytes += len(raw)
            self._row_count += 1
            row_ids.append(row_id)
        self._sync()
        return _DurableRows(
            start_index=start_index,
            end_index=self._row_count - 1,
            row_ids=tuple(row_ids),
            rows_raw_sha256=group_digest.hexdigest(),
            byte_offset_start=start_offset,
            byte_offset_end=self._raw_bytes,
            stream_prefix_raw_sha256=self._digest.hexdigest(),
        )

    def close(self) -> None:
        return


class _CreateOnlyStreamingJsonlSink(_DigestRowSink):
    """Create-only sink; every public append group is physically fsynced."""

    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        self._handle: BinaryIO = pathlib.Path(path).open("xb")
        self._closed = False

    def _write(self, raw: bytes) -> None:
        written = self._handle.write(raw)
        if written != len(raw):
            raise OSError("short campaign JSONL write")

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


def relationship_product_horizon_development_campaign_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).with_name("protocols") / _PROTOCOL_FILENAME


def load_relationship_product_horizon_development_campaign_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonDevelopmentCampaignProtocol:
    source = pathlib.Path(
        path or relationship_product_horizon_development_campaign_protocol_path()
    )
    raw = source.read_bytes()
    payload = cal._parse_json_bytes(raw, source="development campaign protocol")
    cal._exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "purpose",
            "upstream_forced_inputs",
            "design",
            "runtime_order",
            "trace_contract",
            "outcomes",
            "estimands",
            "bootstrap",
            "icc",
            "power_marker",
            "ledger_boundary",
            "mechanism_gates",
            "decision_rule",
            "terminal_statuses",
            "claims",
            "claim_boundary",
        },
        "development campaign protocol",
    )
    protocol_id = sha256_json(payload)
    raw_sha = cal._sha256_bytes(raw)
    if (
        payload["schema_version"] != DEVELOPMENT_CAMPAIGN_PROTOCOL_SCHEMA_VERSION
        or payload["evidence_tier"] != "development"
        or payload["owner"]
        != "lifeform_evolution.relationship_product_horizon_development_campaign"
        or payload["purpose"]
        != "learnable_and_steerable_direction_variance_icc_cost_screen"
        or protocol_id != _EXPECTED_PROTOCOL_ID
        or raw_sha != _EXPECTED_PROTOCOL_RAW_SHA256
    ):
        raise ValueError("development campaign protocol identity drifted")
    _validate_protocol_contract(payload)
    return RelationshipProductHorizonDevelopmentCampaignProtocol(
        payload=payload,
        raw_bytes=raw,
        protocol_id=protocol_id,
        raw_sha256=raw_sha,
    )


def _validate_protocol_contract(payload: Mapping[str, object]) -> None:
    upstream = cal._mapping(payload["upstream_forced_inputs"], "upstream inputs")
    design = cal._mapping(payload["design"], "design")
    runtime = cal._mapping(payload["runtime_order"], "runtime order")
    trace = cal._mapping(payload["trace_contract"], "trace contract")
    bootstrap = cal._mapping(payload["bootstrap"], "bootstrap")
    claims = cal._mapping(payload["claims"], "claims")
    ledger = cal._mapping(payload["ledger_boundary"], "ledger boundary")
    if (
        design["root_count"] != 112
        or design["arm_ids"] != [item.value for item in _ARM_IDS]
        or design["evaluation_decision_indices"] != list(_EVALUATION_INDICES)
        or design["evaluation_decision_count_per_arm"] != 40
        or design["evaluation_credit_applied_to_gate"] is not False
        or design["evaluation_gate_update_count_per_arm"] != 0
        or design["evaluation_credit_timestamp_formula"]
        != "root_sequence_index_times_100_plus_5_plus_2_times_decision_index"
        or design["rehearsal_enabled"] is not False
        or design["rehearsal_required"] is not False
        or runtime["output_inventory"]
        != [
            "protocol.json",
            _PLAN_FILENAME,
            _TRACE_FILENAME,
            _TERMINAL_STATE_FILENAME,
            _REPORT_FILENAME,
            "manifest.json",
        ]
        or runtime["resume_or_partial_completion_authorized"] is not False
        or runtime["technical_missingness_allowed"] is not False
        or runtime["model_invocation_count"] != 0
        or runtime["cuda_execution_count"] != 0
        or trace["total_row_count"] != 36066
        or trace["terminal_state_row_count"] != 336
        or trace["record_field_sets_sha256"]
        != "5a212ca8bcec0e9efad56a997bafed7a55581283ad4845e1601195c461c48617"
        or bootstrap["replicate_count"] != 20000
        or bootstrap["replicate_zero_first_twelve_root_indices"]
        != list(_BOOTSTRAP_REPLICATE_ZERO_FIRST_TWELVE)
        or bootstrap["simultaneous_max_error_quantile_zero_based_order_index"]
        != 18999
        or claims["development_campaign_execution_authorized"] is not True
        or claims["confirmatory_effect_tested"] is not False
        or claims["learnable_effect"] is not False
        or claims["steerable_effect"] is not False
        or ledger["account"] != "product_horizon_development_campaign"
        or ledger["a1_reactive_source_qualification_evidence_inherited"] is not False
        or ledger["a2_msc_budget_or_dyad_evidence_inherited"] is not False
    ):
        raise ValueError("development campaign frozen contract drifted")
    expected_claim_fields = {
        "development_campaign_execution_authorized",
        "development_campaign_completed",
        "development_contrast_estimated",
        "development_contrasts_estimated_by_id",
        "confirmatory_effect_tested",
        "power_prereg_design_authorized",
        "development_go_candidate_by_contrast",
        "power_prereg_design_authorized_by_contrast",
        "reader_qualified",
        "appendable_effect",
        "readable_effect",
        "learnable_effect",
        "steerable_effect",
        "four_able_complete",
        "formal_evidence_authorized",
        "unseen_evidence_authorized",
        "integrated_horizon_authorized",
        "human_validation_complete",
        "production_active",
    }
    contrast_ids = {_LEARNABLE_CONTRAST_ID, _STEERABLE_CONTRAST_ID}
    if (
        set(claims) != expected_claim_fields
        or set(
            cal._mapping(
                claims["development_contrasts_estimated_by_id"],
                "development contrast claims",
            )
        )
        != contrast_ids
        or set(
            cal._mapping(
                claims["development_go_candidate_by_contrast"],
                "development GO candidate claims",
            )
        )
        != contrast_ids
        or set(
            cal._mapping(
                claims["power_prereg_design_authorized_by_contrast"],
                "power prereg claims",
            )
        )
        != contrast_ids
    ):
        raise ValueError("development campaign claim vector drifted")
    lineage = cal._list(upstream["campaign_input_lineage"], "campaign lineage")
    names = tuple(cal._text(cal._mapping(item, "lineage item")["name"], "name") for item in lineage)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("development campaign lineage must be sorted and unique")


def _load_dependencies(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    dynamic_root: pathlib.Path,
    forced_common_batch_root: pathlib.Path,
) -> _Dependencies:
    protocol = load_relationship_product_horizon_development_campaign_protocol()
    pin = cal._mapping(protocol.payload["upstream_forced_inputs"], "upstream inputs")
    inputs = load_relationship_product_horizon_forced_campaign_inputs(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
        scanner_root=pathlib.Path(scanner_root),
        dynamic_root=pathlib.Path(dynamic_root),
        forced_common_batch_root=pathlib.Path(forced_common_batch_root),
        expected_forced_protocol_id=cal._digest(
            pin["forced_protocol_id"], "forced_protocol_id"
        ),
        expected_forced_artifact_id=cal._digest(
            pin["forced_artifact_id"], "forced_artifact_id"
        ),
    )
    expected_lineage = tuple(
        (
            cal._text(cal._mapping(item, "lineage item")["name"], "lineage name"),
            cal._text(cal._mapping(item, "lineage item")["value"], "lineage value"),
        )
        for item in cal._list(pin["campaign_input_lineage"], "campaign lineage")
    )
    actual_lineage = tuple((item.name, item.value) for item in inputs.lineage)
    if (
        inputs.forced_protocol_id != pin["forced_protocol_id"]
        or inputs.forced_protocol_raw_sha256 != pin["forced_protocol_raw_sha256"]
        or inputs.forced_artifact_id != pin["forced_artifact_id"]
        or inputs.forced_manifest_raw_sha256 != pin["forced_manifest_raw_sha256"]
        or inputs.lineage_schema_version
        != pin["campaign_input_lineage_schema_version"]
        or inputs.lineage_id != pin["campaign_input_lineage_id"]
        or actual_lineage != expected_lineage
        or inputs.public_plan_sha256
        != dict(actual_lineage)["source_v4_public_plan_sha256"]
        or len(inputs.roots) != 112
    ):
        raise ValueError("development campaign forced input lineage drifted")
    return _Dependencies(
        protocol=protocol,
        inputs=inputs,
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
        scanner_root=pathlib.Path(scanner_root),
        dynamic_root=pathlib.Path(dynamic_root),
    )


def _segment_id(decision_index: int) -> str:
    for segment_id, indices in _SEGMENTS:
        if decision_index in indices:
            return segment_id
    raise ValueError("decision index is outside the evaluation segments")


def _arm_order(root_sequence_index: int) -> tuple[RelationshipProductHorizonCampaignArm, ...]:
    return _ARM_ORDERS[root_sequence_index % len(_ARM_ORDERS)]


def _build_campaign_plan(
    *, dependencies: _Dependencies
) -> Mapping[str, object]:
    roots = []
    for item in dependencies.inputs.roots:
        if item.root_sequence_index != len(roots):
            raise ValueError("campaign root order drifted")
        roots.append(
            {
                "root_sequence_index": item.root_sequence_index,
                "subject_id": item.public_root.subject_id,
                "public_trajectory_sha256": item.public_root.public_trajectory_sha256,
                "common_terminal_owner_persistence_sha256": (
                    item.common_terminal_owner_persistence_sha256
                ),
                "schedule_artifact_id": item.schedule_artifact_id,
                "forced_transition_raw_sha256": item.transition_raw_sha256,
                "arm_order": [arm.value for arm in _arm_order(item.root_sequence_index)],
                "evaluation_decisions": [
                    {
                        "decision_index": decision.decision_index,
                        "segment_id": _segment_id(decision.decision_index),
                        "session_id": decision.session_id,
                        "decision_id": decision.decision_id,
                        "public_decision_sha256": sha256_json(decision.to_payload()),
                    }
                    for decision in item.public_root.decision_sessions[8:]
                ],
            }
        )
    core = {
        "schema_version": DEVELOPMENT_CAMPAIGN_PLAN_SCHEMA_VERSION,
        "protocol_id": dependencies.protocol.protocol_id,
        "campaign_input_lineage_id": dependencies.inputs.lineage_id,
        "public_plan_sha256": dependencies.inputs.public_plan_sha256,
        "root_count": 112,
        "arm_count": 3,
        "evaluation_decision_count_per_arm": 40,
        "roots": roots,
        "sealed_source_fields_present": False,
    }
    return {"plan_id": sha256_json(core), **core}


def _write_create_only_fsynced(path: pathlib.Path, raw: bytes) -> None:
    with pathlib.Path(path).open("xb") as handle:
        written = handle.write(raw)
        if written != len(raw):
            raise OSError(f"short create-only write: {path}")
        handle.flush()
        os.fsync(handle.fileno())


def _write_and_reopen_exact(path: pathlib.Path, raw: bytes) -> None:
    _write_create_only_fsynced(path, raw)
    if cal._read_regular(path) != raw:
        raise RuntimeError(f"persisted create-only bytes drifted: {path.name}")


def _owner_snapshot_payload(
    snapshot: OwnerPersistenceSnapshot,
) -> Mapping[str, object]:
    if not isinstance(snapshot, OwnerPersistenceSnapshot):
        raise TypeError("owner snapshot has unexpected type")
    return {
        "owner_name": snapshot.owner_name,
        "schema_version": snapshot.schema_version,
        "payload": snapshot.payload,
        "description": snapshot.description,
    }


def _owner_snapshot_from_payload(payload: object) -> OwnerPersistenceSnapshot:
    raw = cal._mapping(payload, "campaign owner persistence envelope")
    cal._exact_keys(
        raw,
        {"owner_name", "schema_version", "payload", "description"},
        "campaign owner persistence envelope",
    )
    description = raw["description"]
    if not isinstance(description, str):
        raise ValueError("campaign owner description must be text")
    snapshot = OwnerPersistenceSnapshot(
        owner_name=cal._text(raw["owner_name"], "owner_name"),
        schema_version=cal._integer(raw["schema_version"], "schema_version"),
        payload=cal._mapping(raw["payload"], "owner payload"),
        description=description,
    )
    store = SocialRecordStore()
    store.hydrate_from_persistence(snapshot)
    if store.export_persistence_snapshot() != snapshot:
        raise ValueError("campaign owner persistence did not roundtrip exactly")
    return snapshot


def _request(
    *,
    root: HorizonPublicRoot,
    decision: HorizonPublicDecisionSession,
) -> RelationshipProductPreActionRequest:
    action_turn = 4 + 2 * decision.decision_index
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
            session_scope=root.subject_id,
        ),
        outcome_turn_index=action_turn + 1,
    )


def _placeholder_substrate() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="relationship-product-horizon-development-campaign-placeholder",
        is_frozen=True,
        surface_kind=SurfaceKind.PLACEHOLDER,
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="development campaign typed executor action surface",
    )


def _authorization(
    *,
    protocol_id: str,
    root_sequence_index: int,
    arm_id: RelationshipProductHorizonCampaignArm,
    initialization: RelationshipProductHorizonCampaignArmInitialization,
) -> RelationshipProductFrozenPulseAuthorization:
    policy = initialization.frozen_policy
    pulse = RelationshipProductPulseAuthorization(
        authorization_id=(
            "relationship-product-horizon-development-campaign:"
            f"{protocol_id}:root:{root_sequence_index:03d}:arm:{arm_id.value}"
        ),
        allowed_policy_artifact_id=policy.artifact.artifact_id,
        allowed_policy_artifact_version=policy.artifact.artifact_version,
    )
    return RelationshipProductFrozenPulseAuthorization(
        pulse_authorization=pulse,
        allowed_frozen_policy_id=policy.policy_id,
        allowed_checkpoint_content_sha256=policy.checkpoint.content_sha256,
    )


def _arm_initialization_payload(
    *,
    arm: RelationshipProductHorizonCampaignArm,
    initialization: RelationshipProductHorizonCampaignArmInitialization,
) -> Mapping[str, object]:
    return {
        "arm_id": arm.value,
        "batch_id": initialization.batch.batch_id,
        "batch_receipt_id": initialization.batch_receipt.receipt_id,
        "batch_disposition": initialization.batch_receipt.disposition.value,
        "frozen_policy_id": initialization.frozen_policy.policy_id,
        "checkpoint_content_sha256": (
            initialization.frozen_policy.checkpoint.content_sha256
        ),
        "checkpoint_update_count": (
            initialization.frozen_policy.checkpoint.update_count
        ),
        "executor_disposition": initialization.executor_disposition.value,
    }


def _full_policy_differs_from_cold(
    *,
    full_policy_id: object,
    full_checkpoint_content_sha256: object,
    cold_policy_id: object,
    cold_checkpoint_content_sha256: object,
) -> bool:
    return bool(
        full_policy_id != cold_policy_id
        and full_checkpoint_content_sha256 != cold_checkpoint_content_sha256
    )


def _credit_timestamp(root_sequence_index: int, decision_index: int) -> int:
    return root_sequence_index * 100 + 5 + 2 * decision_index


def _settlement_input(
    *,
    root: HorizonPublicRoot,
    decision: HorizonPublicDecisionSession,
    preaction: RelationshipProductFrozenPreActionSnapshot,
    outcome: dynamic.RelationshipProductHorizonSelectedBranchOutcome,
    credit_timestamp_ms: int,
) -> RelationshipProductSettlementInput:
    action = RelationshipAction(preaction.delivered_action_id)
    selected_action = outcome.selected_action
    typed_outcome = outcome.typed_outcome
    if selected_action is not action:
        raise ValueError("selected branch action differs from executor delivery")
    if not isinstance(typed_outcome, DialogueExternalOutcomeKind):
        raise TypeError("selected branch outcome is not typed")
    evidence_ref = outcome.environment_evidence_ref
    reaction = outcome.rendered_user_reaction
    evidence_id = f"relationship-product-outcome:{decision.decision_id}"
    action_turn = 4 + 2 * decision.decision_index
    return RelationshipProductSettlementInput(
        external_outcome=DialogueExternalOutcomeEvidence(
            evidence_id=evidence_id,
            turn_index=action_turn + 1,
            kind=typed_outcome,
            source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
            confidence=1.0,
            evidence_ref=evidence_ref,
            description=reaction,
            session_scope=root.subject_id,
            action_turn_index=action_turn,
            forecast_id=preaction.forecast.forecast_id,
            decision_id=decision.decision_id,
            action_id=action.value,
        ),
        owner_outcome_evidence=PreferenceActionOutcomeEvidence(
            evidence_id=evidence_id,
            interlocutor_id=_INTERLOCUTOR_ID,
            observation_summary=decision.current_input,
            action_id=action.value,
            observed_outcome_id=typed_outcome.value,
            reaction_summary=reaction,
            source_turn=action_turn + 1,
            evidence_refs=(evidence_ref,),
        ),
        credit_timestamp_ms=credit_timestamp_ms,
        apply_credit_to_gate=False,
    )


def _external_outcome_evidence_payload(
    evidence: DialogueExternalOutcomeEvidence,
) -> Mapping[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "turn_index": evidence.turn_index,
        "kind": evidence.kind.value,
        "source": evidence.source.value,
        "confidence_hex": evidence.confidence.hex(),
        "evidence_ref": evidence.evidence_ref,
        "description": evidence.description,
        "session_scope": evidence.session_scope,
        "action_turn_index": evidence.action_turn_index,
        "forecast_id": evidence.forecast_id,
        "decision_id": evidence.decision_id,
        "action_id": evidence.action_id,
        "typing_qualification_id": evidence.typing_qualification_id,
        "typing_qualification_sha256": evidence.typing_qualification_sha256,
        "typing_runtime_id": evidence.typing_runtime_id,
        "typing_schema_version": evidence.typing_schema_version,
    }


def _owner_outcome_evidence_payload(
    evidence: PreferenceActionOutcomeEvidence,
) -> Mapping[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "interlocutor_id": evidence.interlocutor_id,
        "observation_summary": evidence.observation_summary,
        "action_id": evidence.action_id,
        "observed_outcome_id": evidence.observed_outcome_id,
        "reaction_summary": evidence.reaction_summary,
        "source_turn": evidence.source_turn,
        "evidence_refs": list(evidence.evidence_refs),
    }


def _forecast_settlement_payload(
    settlement: PreferenceActionForecastSettlement,
) -> Mapping[str, object]:
    return {
        "settlement_id": settlement.settlement_id,
        "forecast_id": settlement.forecast_id,
        "decision_id": settlement.decision_id,
        "session_scope": settlement.session_scope,
        "interlocutor_id": settlement.interlocutor_id,
        "action_id": settlement.action_id,
        "observed_outcome_id": settlement.observed_outcome_id,
        "predicted_probability_hex": settlement.predicted_probability.hex(),
        "negative_log_likelihood_hex": (
            settlement.negative_log_likelihood.hex()
        ),
        "outcome": settlement.outcome.value,
        "magnitude_hex": settlement.magnitude.hex(),
        "source_evidence_id": settlement.source_evidence_id,
        "forecast_issued_turn": settlement.forecast_issued_turn,
        "observed_turn": settlement.observed_turn,
        "evidence_confidence_hex": settlement.evidence_confidence.hex(),
        "expected_utility_hex": settlement.expected_utility.hex(),
        "observed_utility_hex": settlement.observed_utility.hex(),
        "signed_utility_prediction_error_hex": (
            settlement.signed_utility_prediction_error.hex()
        ),
    }


def _preaction_payload(
    *,
    root_sequence_index: int,
    root: HorizonPublicRoot,
    decision: HorizonPublicDecisionSession,
    arm_id: RelationshipProductHorizonCampaignArm,
    physical_arm_order_index: int,
    owner_input_sha256: str,
    initialization: RelationshipProductHorizonCampaignArmInitialization,
    preaction: RelationshipProductFrozenPreActionSnapshot,
) -> Mapping[str, object]:
    forecast_payload = preference_action_forecast_to_payload(preaction.forecast)
    receipt = preaction.execution_receipt
    command = receipt.command
    checkpoint = preaction.frozen_policy.checkpoint
    owner_preaction_sha = social_record_store_persistence_sha256(
        preaction.owner_persistence_snapshot
    )
    if (
        command.owner_prestate_sha256 != owner_preaction_sha
        or command.executor_disposition is not initialization.executor_disposition
        or preaction.frozen_policy != initialization.frozen_policy
        or receipt.delivered_action_id
        != receipt.temporal_delivery.active_abstract_action
    ):
        raise RuntimeError("campaign preaction executor lineage drifted")
    return {
        "schema_version": DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION,
        "record_type": "preaction",
        "root_sequence_index": root_sequence_index,
        "subject_id": root.subject_id,
        "public_trajectory_sha256": root.public_trajectory_sha256,
        "decision_index": decision.decision_index,
        "segment_id": _segment_id(decision.decision_index),
        "session_id": decision.session_id,
        "decision_id": decision.decision_id,
        "arm_id": arm_id.value,
        "physical_arm_order_index": physical_arm_order_index,
        "owner_input_persistence_sha256": owner_input_sha256,
        "owner_preaction_persistence_sha256": owner_preaction_sha,
        "forecast_id": preaction.forecast.forecast_id,
        "forecast_sha256": _pulse_payload_sha256(forecast_payload),
        "frozen_policy_id": preaction.frozen_policy.policy_id,
        "checkpoint_content_sha256": checkpoint.content_sha256,
        "checkpoint_update_count": checkpoint.update_count,
        "transition_batch_id": (
            preaction.frozen_policy.transition_batch.batch_id
            if preaction.frozen_policy.transition_batch is not None
            else None
        ),
        "transition_receipt_id": (
            preaction.frozen_policy.transition_receipt.receipt_id
            if preaction.frozen_policy.transition_receipt is not None
            else None
        ),
        "frozen_decision_sha256": sha256_json(preaction.frozen_decision.to_payload()),
        "gate_action": preaction.frozen_decision.decision.gate_action.value,
        "candidate_action_id": command.candidate_action_id,
        "command_id": command.command_id,
        "authorization_id": command.authorization.authorization_id,
        "executor_receipt_id": receipt.receipt_id,
        "executor_receipt": receipt.to_payload(),
        "executor_disposition": command.executor_disposition.value,
        "executor_status": receipt.executor_status.value,
        "delivered_action_id": receipt.delivered_action_id,
        "executed_nonnoop": (
            receipt.delivered_action_id != RelationshipAction.NEUTRAL_NOOP.value
        ),
        "branch_opened": False,
        "evaluation_or_judge_feedback_received": False,
    }


def _mint_preaction_barrier(
    *,
    sink: _RowSink,
    protocol_id: str,
    root_sequence_index: int,
    decision_index: int,
    arm_order: tuple[RelationshipProductHorizonCampaignArm, ...],
    preactions: tuple[
        tuple[
            RelationshipProductHorizonCampaignArm,
            RelationshipProductFrozenPreActionSnapshot,
        ],
        ...,
    ],
    durable_preactions: _DurableRows,
) -> _PreactionBarrierReceipt:
    receipt_ids = tuple(item.execution_receipt.receipt_id for _, item in preactions)
    core = {
        "barrier_schema_version": DEVELOPMENT_CAMPAIGN_BARRIER_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "root_sequence_index": root_sequence_index,
        "decision_index": decision_index,
        "arm_order": [arm.value for arm in arm_order],
        "preaction_row_ids": list(durable_preactions.row_ids),
        "preaction_executor_receipt_ids": list(receipt_ids),
        "preaction_rows_start_index": durable_preactions.start_index,
        "preaction_rows_end_index": durable_preactions.end_index,
        "preaction_rows_raw_sha256": durable_preactions.rows_raw_sha256,
        "durable_prefix_byte_count_before_receipt": (
            durable_preactions.byte_offset_end
        ),
        "durable_prefix_raw_sha256_before_receipt": (
            durable_preactions.stream_prefix_raw_sha256
        ),
    }
    barrier_id = sha256_json(core)
    durable_receipt = sink.append_many_fsync(
        (
            {
                "schema_version": DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION,
                "record_type": "preaction_group_fsync",
                "barrier_id": barrier_id,
                **core,
            },
        )
    )
    return _PreactionBarrierReceipt(
        barrier_id=barrier_id,
        receipt_row_id=durable_receipt.row_ids[0],
        root_sequence_index=root_sequence_index,
        decision_index=decision_index,
        arm_order=arm_order,
        preaction_receipt_ids=receipt_ids,
        stream_prefix_raw_sha256=durable_receipt.stream_prefix_raw_sha256,
    )


class _OnceOnlySelectedBranchSettlementOwner:
    """Consume one durable triplet before any action-conditioned truth lookup."""

    def __init__(
        self,
        *,
        environment_opener: Callable[
            [], dynamic.RelationshipProductHorizonSelectedBranchEnvironment
        ],
    ) -> None:
        self._environment_opener = environment_opener
        self._environment: (
            dynamic.RelationshipProductHorizonSelectedBranchEnvironment | None
        ) = None
        self._consumed: set[tuple[int, int]] = set()
        self._next_slot = 0
        self._failed = False

    async def settle_triplet(
        self,
        *,
        durable: _DurablePreactions,
        root: HorizonPublicRoot,
        decision: HorizonPublicDecisionSession,
        root_sequence_index: int,
    ) -> tuple[_ArmSettlement, ...]:
        if self._failed:
            raise RuntimeError("selected-branch settlement owner is failed closed")
        slot = (root_sequence_index, decision.decision_index)
        expected_slot = root_sequence_index * 40 + decision.decision_index - 8
        if (
            expected_slot != self._next_slot
            or slot in self._consumed
            or durable.barrier.root_sequence_index != root_sequence_index
            or durable.barrier.decision_index != decision.decision_index
            or tuple(arm for arm, _ in durable.preactions)
            != durable.barrier.arm_order
            or tuple(
                preaction.execution_receipt.receipt_id
                for _, preaction in durable.preactions
            )
            != durable.barrier.preaction_receipt_ids
        ):
            raise ValueError("durable preaction barrier or campaign slot drifted")
        self._consumed.add(slot)
        self._next_slot += 1
        try:
            if self._environment is None:
                self._environment = self._environment_opener()
            outcomes: dict[
                RelationshipAction,
                dynamic.RelationshipProductHorizonSelectedBranchOutcome,
            ] = {}
            for _, preaction in durable.preactions:
                action = RelationshipAction(preaction.delivered_action_id)
                if action not in outcomes:
                    outcomes[action] = self._environment.settle(
                        public_root=root,
                        public_decision=decision,
                        selected_action=action,
                    )
            timestamp = _credit_timestamp(root_sequence_index, decision.decision_index)
            settled_items: list[_ArmSettlement] = []
            for arm_id, preaction in durable.preactions:
                action = RelationshipAction(preaction.delivered_action_id)
                outcome = outcomes[action]
                settlement_input = _settlement_input(
                    root=root,
                    decision=decision,
                    preaction=preaction,
                    outcome=outcome,
                    credit_timestamp_ms=timestamp,
                )
                settled = await settle_relationship_product_frozen_pulse(
                    preaction=preaction,
                    settlement_input=settlement_input,
                )
                if (
                    settled.credit_applied_to_gate
                    or settled.evaluation_gate_update_delta != 0
                    or settled.gate_checkpoint != preaction.frozen_policy.checkpoint
                    or settled.settlement.action_id != action.value
                    or settled.credit.abstract_action_id != action.value
                    or settled.credit.prediction_id != preaction.forecast.forecast_id
                    or settled.credit.timestamp_ms != timestamp
                    or settled.settlement_input.external_outcome.action_id
                    != action.value
                    or settled.settlement_input.owner_outcome_evidence.action_id
                    != action.value
                ):
                    raise RuntimeError("campaign settlement PE-credit lineage drifted")
                settled_items.append(
                    _ArmSettlement(arm_id=arm_id, outcome=outcome, settled=settled)
                )
            return tuple(settled_items)
        except Exception as exc:
            self._failed = True
            raise RuntimeError(
                f"selected-branch settlement failed at root/decision {slot}"
            ) from exc


def _postaction_payload(
    *,
    protocol_id: str,
    barrier: _PreactionBarrierReceipt,
    root_sequence_index: int,
    root: HorizonPublicRoot,
    decision: HorizonPublicDecisionSession,
    physical_arm_order_index: int,
    item: _ArmSettlement,
) -> Mapping[str, object]:
    settled = item.settled
    outcome = item.outcome
    preaction = settled.preaction
    owner_preaction_sha = social_record_store_persistence_sha256(
        preaction.owner_persistence_snapshot
    )
    owner_postaction_sha = social_record_store_persistence_sha256(
        settled.owner_persistence_snapshot
    )
    return {
        "schema_version": DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION,
        "record_type": "postaction",
        "protocol_id": protocol_id,
        "preaction_barrier_id": barrier.barrier_id,
        "preaction_barrier_receipt_row_id": barrier.receipt_row_id,
        "root_sequence_index": root_sequence_index,
        "subject_id": root.subject_id,
        "decision_index": decision.decision_index,
        "segment_id": _segment_id(decision.decision_index),
        "session_id": decision.session_id,
        "decision_id": decision.decision_id,
        "arm_id": item.arm_id.value,
        "physical_arm_order_index": physical_arm_order_index,
        "forecast_id": preaction.forecast.forecast_id,
        "executor_receipt_id": preaction.execution_receipt.receipt_id,
        "candidate_action_id": preaction.execution_receipt.command.candidate_action_id,
        "delivered_action_id": preaction.delivered_action_id,
        "environment_subject_id": outcome.environment_subject_id,
        "selected_branch_action_id": outcome.selected_action.value,
        "selected_branch_commitment_id": outcome.commitment_id,
        "typed_outcome_id": outcome.typed_outcome.value,
        "rendered_user_reaction_sha256": cal._sha256_text(
            outcome.rendered_user_reaction
        ),
        "environment_evidence_ref": outcome.environment_evidence_ref,
        "environment_version": outcome.environment_version,
        "settlement_id": settled.settlement.settlement_id,
        "external_outcome_evidence": _external_outcome_evidence_payload(
            settled.settlement_input.external_outcome
        ),
        "owner_outcome_evidence": _owner_outcome_evidence_payload(
            settled.settlement_input.owner_outcome_evidence
        ),
        "settlement": _forecast_settlement_payload(settled.settlement),
        "social_prediction_error": cal._social_pe_payload(
            settled.social_prediction_error_snapshot.value
        ),
        "credit": cal._credit_payload(settled.credit),
        "credit_applied_to_gate": False,
        "evaluation_gate_update_delta": 0,
        "checkpoint_content_sha256": settled.gate_checkpoint.content_sha256,
        "checkpoint_update_count": settled.gate_checkpoint.update_count,
        "owner_preaction_persistence_sha256": owner_preaction_sha,
        "owner_postaction_persistence_sha256": owner_postaction_sha,
        "owner_writeback_changed_persistence": (
            owner_preaction_sha != owner_postaction_sha
        ),
        "evaluation_or_judge_feedback_received": False,
    }


def _append_postaction_receipt(
    *,
    sink: _RowSink,
    protocol_id: str,
    barrier: _PreactionBarrierReceipt,
    root_sequence_index: int,
    decision_index: int,
    durable_postactions: _DurableRows,
) -> _DurableRows:
    core = {
        "barrier_schema_version": DEVELOPMENT_CAMPAIGN_BARRIER_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "preaction_barrier_id": barrier.barrier_id,
        "root_sequence_index": root_sequence_index,
        "decision_index": decision_index,
        "postaction_row_ids": list(durable_postactions.row_ids),
        "postaction_rows_start_index": durable_postactions.start_index,
        "postaction_rows_end_index": durable_postactions.end_index,
        "postaction_rows_raw_sha256": durable_postactions.rows_raw_sha256,
        "durable_prefix_byte_count_before_receipt": (
            durable_postactions.byte_offset_end
        ),
        "durable_prefix_raw_sha256_before_receipt": (
            durable_postactions.stream_prefix_raw_sha256
        ),
    }
    return sink.append_many_fsync(
        (
            {
                "schema_version": DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION,
                "record_type": "postaction_group_fsync",
                "postaction_receipt_id": sha256_json(core),
                **core,
            },
        )
    )


async def _run_campaign(
    *,
    dependencies: _Dependencies,
    plan: Mapping[str, object],
    trace_sink: _RowSink,
    terminal_state_sink: _RowSink,
) -> _CampaignReplay:
    protocol = dependencies.protocol
    lineage = {item.name: item.value for item in dependencies.inputs.lineage}
    trace_sink.append_many_fsync(
        (
            {
                "schema_version": DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION,
                "record_type": "header",
                "protocol_id": protocol.protocol_id,
                "protocol_raw_sha256": protocol.raw_sha256,
                "plan_id": plan["plan_id"],
                "campaign_input_lineage_id": dependencies.inputs.lineage_id,
                "forced_protocol_id": dependencies.inputs.forced_protocol_id,
                "forced_artifact_id": dependencies.inputs.forced_artifact_id,
                "public_plan_sha256": dependencies.inputs.public_plan_sha256,
                "root_count": 112,
                "arm_ids": [item.value for item in _ARM_IDS],
                "evaluation_decision_indices": list(_EVALUATION_INDICES),
                "credit_timestamp_formula": (
                    "root_sequence_index_times_100_plus_5_plus_2_times_"
                    "decision_index"
                ),
                "rehearsal_enabled": False,
                "selected_branch_environment_opened": False,
                "model_invocation_count": 0,
                "cuda_execution_count": 0,
            },
        )
    )

    environment_owner = _OnceOnlySelectedBranchSettlementOwner(
        environment_opener=lambda: dynamic.open_relationship_product_horizon_selected_branch_environment(
            source_v4_admission_root=dependencies.source_v4_admission_root,
            reader_root=dependencies.reader_root,
            theta0_v2_root=dependencies.theta0_v2_root,
            scanner_root=dependencies.scanner_root,
            dynamic_collection_prefix_root=dependencies.dynamic_root,
            expected_dynamic_protocol_id=lineage["dynamic_protocol_id"],
            expected_dynamic_artifact_id=lineage["dynamic_artifact_id"],
        )
    )
    records: list[_DecisionRecord] = []
    action_counts = {arm.value: Counter() for arm in _ARM_IDS}
    learnable_divergence_count = 0
    steerable_divergence_count = 0
    frozen_nonnoop_count = 0
    later_owner_handoff_count = 0
    owner_writeback_change_count = 0
    branch_ids: set[str] = set()
    settlement_slots: set[tuple[str, str]] = set()
    credit_slots: set[tuple[str, str]] = set()
    prior_credit_timestamp = -1

    for root_input in dependencies.inputs.roots:
        root_index = root_input.root_sequence_index
        root = root_input.public_root
        physical_order = _arm_order(root_index)
        initializations = {
            item.arm_id: item for item in root_input.fresh_arm_initializations()
        }
        if tuple(initializations) != _ARM_IDS:
            raise RuntimeError("campaign arm initialization order drifted")
        if len({id(item.owner_persistence_snapshot) for item in initializations.values()}) != 3:
            raise RuntimeError("campaign arms share mutable owner snapshots")
        if len({id(item.forecast_runtime) for item in initializations.values()}) != 3:
            raise RuntimeError("campaign arms share reader runtime caches")
        if any(
            item.starting_owner_persistence_sha256
            != root_input.common_terminal_owner_persistence_sha256
            for item in initializations.values()
        ):
            raise RuntimeError("campaign root did not start from one common owner")
        owners = {
            arm: item.owner_persistence_snapshot
            for arm, item in initializations.items()
        }
        prior_post_sha: dict[RelationshipProductHorizonCampaignArm, str] = {}
        trace_sink.append_many_fsync(
            (
                {
                    "schema_version": DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION,
                    "record_type": "root_start",
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "public_trajectory_sha256": root.public_trajectory_sha256,
                    "common_terminal_owner_persistence_sha256": (
                        root_input.common_terminal_owner_persistence_sha256
                    ),
                    "forced_transition_raw_sha256": (
                        root_input.transition_raw_sha256
                    ),
                    "arm_order": [item.value for item in physical_order],
                    "arm_initializations": [
                        _arm_initialization_payload(
                            arm=arm,
                            initialization=initializations[arm],
                        )
                        for arm in physical_order
                    ],
                },
            )
        )

        for decision in root.decision_sessions[8:]:
            timestamp = _credit_timestamp(root_index, decision.decision_index)
            if timestamp <= prior_credit_timestamp:
                raise RuntimeError("campaign credit timestamp is not strictly increasing")
            prior_credit_timestamp = timestamp
            prepared: list[
                tuple[
                    RelationshipProductHorizonCampaignArm,
                    RelationshipProductFrozenPreActionSnapshot,
                    str,
                ]
            ] = []
            pre_rows: list[Mapping[str, object]] = []
            request = _request(root=root, decision=decision)
            for physical_index, arm in enumerate(physical_order):
                initialization = initializations[arm]
                owner_input = owners[arm]
                owner_input_sha = social_record_store_persistence_sha256(owner_input)
                if decision.decision_index == 8:
                    if owner_input_sha != root_input.common_terminal_owner_persistence_sha256:
                        raise RuntimeError("first evaluation owner is not common terminal")
                else:
                    if owner_input_sha != prior_post_sha[arm]:
                        raise RuntimeError("later evaluation owner handoff drifted")
                    later_owner_handoff_count += 1
                preaction = await prepare_relationship_product_frozen_preaction(
                    request=request,
                    owner_persistence_snapshot=owner_input,
                    forecast_runtime=initialization.forecast_runtime,
                    frozen_policy=initialization.frozen_policy,
                    executor_disposition=initialization.executor_disposition,
                    authorization=_authorization(
                        protocol_id=protocol.protocol_id,
                        root_sequence_index=root_index,
                        arm_id=arm,
                        initialization=initialization,
                    ),
                    substrate_snapshot=_placeholder_substrate(),
                )
                prepared.append((arm, preaction, owner_input_sha))
                pre_rows.append(
                    _preaction_payload(
                        root_sequence_index=root_index,
                        root=root,
                        decision=decision,
                        arm_id=arm,
                        physical_arm_order_index=physical_index,
                        owner_input_sha256=owner_input_sha,
                        initialization=initialization,
                        preaction=preaction,
                    )
                )
            durable_pre_rows = trace_sink.append_many_fsync(tuple(pre_rows))
            preactions = tuple((arm, preaction) for arm, preaction, _ in prepared)
            barrier = _mint_preaction_barrier(
                sink=trace_sink,
                protocol_id=protocol.protocol_id,
                root_sequence_index=root_index,
                decision_index=decision.decision_index,
                arm_order=physical_order,
                preactions=preactions,
                durable_preactions=durable_pre_rows,
            )
            settlements = await environment_owner.settle_triplet(
                durable=_DurablePreactions(barrier=barrier, preactions=preactions),
                root=root,
                decision=decision,
                root_sequence_index=root_index,
            )
            post_rows = tuple(
                _postaction_payload(
                    protocol_id=protocol.protocol_id,
                    barrier=barrier,
                    root_sequence_index=root_index,
                    root=root,
                    decision=decision,
                    physical_arm_order_index=physical_index,
                    item=item,
                )
                for physical_index, item in enumerate(settlements)
            )
            durable_post_rows = trace_sink.append_many_fsync(post_rows)
            _append_postaction_receipt(
                sink=trace_sink,
                protocol_id=protocol.protocol_id,
                barrier=barrier,
                root_sequence_index=root_index,
                decision_index=decision.decision_index,
                durable_postactions=durable_post_rows,
            )

            delivered = {
                item.arm_id: RelationshipAction(item.settled.preaction.delivered_action_id)
                for item in settlements
            }
            if delivered[RelationshipProductHorizonCampaignArm.STRICT_NOOP] is not RelationshipAction.NEUTRAL_NOOP:
                raise RuntimeError("strict_noop arm delivered a non-noop action")
            learnable_divergence_count += int(
                delivered[RelationshipProductHorizonCampaignArm.FULL]
                is not delivered[RelationshipProductHorizonCampaignArm.FROZEN_THETA0]
            )
            steerable_divergence_count += int(
                delivered[RelationshipProductHorizonCampaignArm.FROZEN_THETA0]
                is not delivered[RelationshipProductHorizonCampaignArm.STRICT_NOOP]
            )
            frozen_nonnoop_count += int(
                delivered[RelationshipProductHorizonCampaignArm.FROZEN_THETA0]
                is not RelationshipAction.NEUTRAL_NOOP
            )
            for item in settlements:
                arm = item.arm_id
                settled = item.settled
                outcome_kind = item.outcome.typed_outcome
                candidate = RelationshipAction(
                    settled.preaction.execution_receipt.command.candidate_action_id
                )
                action = RelationshipAction(settled.preaction.delivered_action_id)
                records.append(
                    _DecisionRecord(
                        root_sequence_index=root_index,
                        decision_index=decision.decision_index,
                        segment_id=_segment_id(decision.decision_index),
                        arm_id=arm,
                        candidate_action=candidate,
                        delivered_action=action,
                        outcome=outcome_kind,
                    )
                )
                action_counts[arm.value][action.value] += 1
                branch_ids.add(item.outcome.commitment_id)
                settlement_slot = (arm.value, settled.settlement.settlement_id)
                credit_slot = (arm.value, settled.credit.record_id)
                if settlement_slot in settlement_slots or credit_slot in credit_slots:
                    raise RuntimeError("campaign settlement or credit slot was reused")
                settlement_slots.add(settlement_slot)
                credit_slots.add(credit_slot)
                post_sha = social_record_store_persistence_sha256(
                    settled.owner_persistence_snapshot
                )
                if post_sha != social_record_store_persistence_sha256(
                    settled.preaction.owner_persistence_snapshot
                ):
                    owner_writeback_change_count += 1
                owners[arm] = settled.owner_persistence_snapshot
                prior_post_sha[arm] = post_sha

        terminal_rows: list[Mapping[str, object]] = []
        for arm in physical_order:
            owner = owners[arm]
            payload = _owner_snapshot_payload(owner)
            roundtripped = _owner_snapshot_from_payload(payload)
            if roundtripped != owner:
                raise RuntimeError("campaign terminal owner failed exact roundtrip")
            terminal_rows.append(
                {
                    "schema_version": (
                        DEVELOPMENT_CAMPAIGN_TERMINAL_STATE_SCHEMA_VERSION
                    ),
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "arm_id": arm.value,
                    "physical_arm_order_index": physical_order.index(arm),
                    "terminal_owner_persistence_sha256": (
                        social_record_store_persistence_sha256(owner)
                    ),
                    "terminal_owner_persistence": payload,
                    "frozen_policy_id": initializations[arm].frozen_policy.policy_id,
                    "checkpoint_content_sha256": (
                        initializations[arm].frozen_policy.checkpoint.content_sha256
                    ),
                    "checkpoint_update_count": (
                        initializations[arm].frozen_policy.checkpoint.update_count
                    ),
                    "evaluation_gate_update_count": 0,
                }
            )
        durable_terminal = terminal_state_sink.append_many_fsync(tuple(terminal_rows))
        trace_sink.append_many_fsync(
            (
                {
                    "schema_version": DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION,
                    "record_type": "root_terminal",
                    "root_sequence_index": root_index,
                    "subject_id": root.subject_id,
                    "evaluation_decision_count_per_arm": 40,
                    "terminal_state_row_ids": list(durable_terminal.row_ids),
                    "terminal_state_prefix_raw_sha256": (
                        durable_terminal.stream_prefix_raw_sha256
                    ),
                    "terminal_owner_persistence_sha256_by_arm": {
                        arm.value: social_record_store_persistence_sha256(owners[arm])
                        for arm in _ARM_IDS
                    },
                    "evaluation_gate_update_count": 0,
                },
            )
        )

    mechanism = {
        "complete_root_count": 112,
        "complete_root_arm_count": 336,
        "complete_evaluation_slot_count": len(records),
        "later_owner_handoff_count": later_owner_handoff_count,
        "terminal_owner_roundtrip_count": terminal_state_sink.row_count,
        "owner_writeback_change_count": owner_writeback_change_count,
        "strict_noop_nonnoop_count": sum(
            count
            for action, count in action_counts[
                RelationshipProductHorizonCampaignArm.STRICT_NOOP.value
            ].items()
            if action != RelationshipAction.NEUTRAL_NOOP.value
        ),
        "evaluation_credit_applied_count": 0,
        "evaluation_gate_update_count": 0,
        "learnable_actual_action_divergence_count": learnable_divergence_count,
        "steerable_actual_action_divergence_count": steerable_divergence_count,
        "frozen_theta0_physical_nonnoop_count": frozen_nonnoop_count,
        "full_learned_policy_differs_from_cold_root_count": sum(
            int(
                _full_policy_differs_from_cold(
                    full_policy_id=item.full_policy_id,
                    full_checkpoint_content_sha256=(
                        item.full_checkpoint_content_sha256
                    ),
                    cold_policy_id=item.cold_frozen_policy_id,
                    cold_checkpoint_content_sha256=lineage[
                        "cold_checkpoint_content_sha256"
                    ],
                )
            )
            for item in dependencies.inputs.roots
        ),
        "selected_branch_unique_content_count": len(branch_ids),
        "settlement_slot_count": len(settlement_slots),
        "credit_slot_count": len(credit_slots),
        "actual_action_counts_by_arm": {
            arm: dict(sorted(counts.items()))
            for arm, counts in sorted(action_counts.items())
        },
    }
    _validate_complete_mechanism_contract(mechanism)
    report = _build_report(
        protocol=protocol,
        plan_id=cal._text(plan["plan_id"], "plan_id"),
        lineage_id=dependencies.inputs.lineage_id,
        records=tuple(records),
        mechanism=mechanism,
    )
    trace_sink.append_many_fsync(
        (
            {
                "schema_version": DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION,
                "record_type": "terminal",
                "protocol_id": protocol.protocol_id,
                "plan_id": plan["plan_id"],
                "report_id": report["report_id"],
                "completed_root_count": 112,
                "completed_root_arm_count": 336,
                "completed_evaluation_slot_count": len(records),
                "preaction_count": 13440,
                "preaction_group_fsync_count": 4480,
                "postaction_count": 13440,
                "postaction_group_fsync_count": 4480,
                "terminal_state_row_count": terminal_state_sink.row_count,
                "mechanism": mechanism,
                "status": report["status"],
                "rehearsal_executed": False,
                "model_invocation_count": 0,
                "cuda_execution_count": 0,
            },
        )
    )
    if trace_sink.row_count != 36066 or terminal_state_sink.row_count != 336:
        raise RuntimeError("campaign terminal row inventory drifted")
    return _CampaignReplay(
        report=report,
        trace_row_count=trace_sink.row_count,
        trace_raw_bytes=trace_sink.raw_bytes,
        trace_raw_sha256=trace_sink.raw_sha256,
        terminal_state_row_count=terminal_state_sink.row_count,
        terminal_state_raw_bytes=terminal_state_sink.raw_bytes,
        terminal_state_raw_sha256=terminal_state_sink.raw_sha256,
    )


def _validate_complete_mechanism_contract(mechanism: Mapping[str, object]) -> None:
    if (
        mechanism["complete_root_count"] != 112
        or mechanism["complete_root_arm_count"] != 336
        or mechanism["complete_evaluation_slot_count"] != 13440
        or mechanism["later_owner_handoff_count"] != 13104
        or mechanism["terminal_owner_roundtrip_count"] != 336
        or mechanism["strict_noop_nonnoop_count"] != 0
        or mechanism["evaluation_credit_applied_count"] != 0
        or mechanism["evaluation_gate_update_count"] != 0
        or mechanism["settlement_slot_count"] != 13440
        or mechanism["credit_slot_count"] != 13440
    ):
        raise RuntimeError("campaign complete mechanism contract failed")


def _float_hex(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("campaign statistic must be finite")
    return value.hex()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("campaign statistic cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _bootstrap_root_indices(
    *,
    replicate_index: int,
    root_count: int,
    seed_hex: str,
    domain: str,
) -> tuple[int, ...]:
    seed = bytes.fromhex(seed_hex)
    if len(seed) != 8:
        raise ValueError("campaign bootstrap seed must decode to eight bytes")

    def field(raw: bytes) -> bytes:
        return len(raw).to_bytes(4, "big") + raw

    prefix = b"".join(
        (
            field(seed),
            field(domain.encode("utf-8")),
            field(str(root_count).encode("utf-8")),
            field(str(replicate_index).encode("utf-8")),
        )
    )
    result: list[int] = []
    for draw_index in range(root_count):
        rejection_index = 0
        while True:
            digest = hashlib.sha256(
                prefix
                + field(str(draw_index).encode("utf-8"))
                + field(str(rejection_index).encode("utf-8"))
            ).digest()
            accepted: int | None = None
            for word_index in range(8):
                value = int.from_bytes(
                    digest[word_index * 4 : (word_index + 1) * 4],
                    "big",
                )
                if value < 4_294_967_264:
                    accepted = value % root_count
                    break
            if accepted is not None:
                result.append(accepted)
                break
            rejection_index += 1
    return tuple(result)


def _root_metric_table(
    records: Sequence[_DecisionRecord],
) -> tuple[
    Mapping[str, tuple[tuple[float, ...], ...]],
    Mapping[str, Mapping[str, object]],
]:
    slots = {
        (item.root_sequence_index, item.decision_index, item.arm_id): item
        for item in records
    }
    if len(slots) != 13440:
        raise ValueError("campaign analysis decision slot inventory drifted")
    contrast_arms = {
        "learnable_full_minus_frozen_theta0": (
            RelationshipProductHorizonCampaignArm.FULL,
            RelationshipProductHorizonCampaignArm.FROZEN_THETA0,
        ),
        "steerable_frozen_theta0_minus_strict_noop": (
            RelationshipProductHorizonCampaignArm.FROZEN_THETA0,
            RelationshipProductHorizonCampaignArm.STRICT_NOOP,
        ),
    }
    tables: dict[str, tuple[tuple[float, ...], ...]] = {}
    arm_summary: dict[str, Mapping[str, object]] = {}
    for arm in _ARM_IDS:
        arm_records = tuple(item for item in records if item.arm_id is arm)
        outcome_counts = Counter(item.outcome.value for item in arm_records)
        segment_counts = {
            segment_id: Counter(
                item.outcome.value
                for item in arm_records
                if item.segment_id == segment_id
            )
            for segment_id, _ in _SEGMENTS
        }
        arm_summary[arm.value] = {
            "decision_count": len(arm_records),
            "outcome_counts": {
                outcome.value: outcome_counts[outcome.value]
                for outcome in RELATIONSHIP_OUTCOMES
            },
            "positive_rate_hex": _float_hex(
                sum(
                    outcome_counts[item.value] for item in _POSITIVE_OUTCOMES
                )
                / 4480
            ),
            "over_directive_rate_hex": _float_hex(
                outcome_counts[DialogueExternalOutcomeKind.OVER_DIRECTIVE.value]
                / 4480
            ),
            "segment_outcome_counts": {
                segment_id: {
                    outcome.value: segment_counts[segment_id][outcome.value]
                    for outcome in RELATIONSHIP_OUTCOMES
                }
                for segment_id, _ in _SEGMENTS
            },
        }
    for contrast_id, (target, reference) in contrast_arms.items():
        root_rows: list[tuple[float, ...]] = []
        for root_index in range(112):
            differences: list[float] = []
            for decision_index in _EVALUATION_INDICES:
                target_item = slots[(root_index, decision_index, target)]
                reference_item = slots[(root_index, decision_index, reference)]
                target_positive = float(target_item.outcome in _POSITIVE_OUTCOMES)
                reference_positive = float(reference_item.outcome in _POSITIVE_OUTCOMES)
                differences.append(target_positive - reference_positive)
            root_rows.append(tuple(differences))
        tables[contrast_id] = tuple(root_rows)
    return tables, arm_summary


def _bootstrap_statistics(
    *,
    root_metrics: Mapping[str, tuple[float, ...]],
    protocol: RelationshipProductHorizonDevelopmentCampaignProtocol,
) -> Mapping[str, object]:
    bootstrap = cal._mapping(protocol.payload["bootstrap"], "bootstrap")
    seed_hex = cal._text(bootstrap["seed_hex"], "bootstrap seed")
    domain = cal._text(bootstrap["domain"], "bootstrap domain")
    replicate_count = cal._integer(
        bootstrap["replicate_count"], "bootstrap replicate_count"
    )
    if any(len(values) != 112 for values in root_metrics.values()):
        raise ValueError("bootstrap root metric inventory drifted")
    observed = {key: _mean(values) for key, values in root_metrics.items()}
    distributions = {key: [] for key in root_metrics}
    index_stream_digest = hashlib.sha256()
    for replicate_index in range(replicate_count):
        indices = _bootstrap_root_indices(
            replicate_index=replicate_index,
            root_count=112,
            seed_hex=seed_hex,
            domain=domain,
        )
        index_stream_digest.update(bytes(indices))
        for key, root_values in root_metrics.items():
            distributions[key].append(
                math.fsum(root_values[index] for index in indices) / 112
            )
    return {
        "observed": observed,
        "distributions": distributions,
        "index_stream_sha256": index_stream_digest.hexdigest(),
        "accepted_root_index_count": replicate_count * 112,
    }


def _icc_and_power(
    root_rows: tuple[tuple[float, ...], ...],
    *,
    protocol: RelationshipProductHorizonDevelopmentCampaignProtocol,
) -> Mapping[str, object]:
    root_count = len(root_rows)
    decision_count = len(root_rows[0])
    position_means = tuple(
        math.fsum(root_rows[root][column] for root in range(root_count))
        / root_count
        for column in range(decision_count)
    )
    centered = tuple(
        tuple(row[column] - position_means[column] for column in range(decision_count))
        for row in root_rows
    )
    centered_root_means = tuple(_mean(row) for row in centered)
    centered_grand = _mean(centered_root_means)
    bms = decision_count / (root_count - 1) * math.fsum(
        (value - centered_grand) ** 2 for value in centered_root_means
    )
    wms = 1 / (root_count * (decision_count - 1)) * math.fsum(
        (value - centered_root_means[root]) ** 2
        for root, row in enumerate(centered)
        for value in row
    )
    denominator = bms + (decision_count - 1) * wms
    if denominator == 0.0:
        icc_raw: float | None = None
        icc_planning: float | None = None
        icc_reason: str | None = "zero_total_variance_icc_undefined"
    else:
        icc_raw = (bms - wms) / denominator
        icc_planning = min(1.0, max(0.0, icc_raw))
        icc_reason = None
    raw_root_means = tuple(_mean(row) for row in root_rows)
    raw_grand = _mean(raw_root_means)
    variance = math.fsum(
        (value - raw_grand) ** 2 for value in raw_root_means
    ) / (root_count - 1)
    standard_error = math.sqrt(variance / root_count)
    power = cal._mapping(protocol.payload["power_marker"], "power marker")
    zsum = float(cal._text(power["z_alpha_two_sided"], "z alpha")) + float(
        cal._text(power["z_power"], "z power")
    )
    markers = []
    for assumed_effect_raw in cal._list(power["assumed_effects"], "assumed effects"):
        assumed_effect = float(assumed_effect_raw)
        if variance == 0.0:
            marker: int | None = None
            reason: str | None = "zero_observed_root_effect_variance"
        else:
            marker = math.ceil(zsum**2 * variance / assumed_effect**2)
            reason = None
        markers.append(
            {
                "assumed_effect_hex": _float_hex(assumed_effect),
                "root_count_marker": marker,
                "null_reason": reason,
            }
        )
    return {
        "position_centering_applied": True,
        "between_mean_square_hex": _float_hex(bms),
        "within_mean_square_hex": _float_hex(wms),
        "icc_raw_hex": _float_hex(icc_raw) if icc_raw is not None else None,
        "icc_planning_hex": (
            _float_hex(icc_planning) if icc_planning is not None else None
        ),
        "icc_null_reason": icc_reason,
        "root_effect_variance_hex": _float_hex(variance),
        "root_effect_standard_error_hex": _float_hex(standard_error),
        "power_markers": markers,
        "formal_sample_size_selected": False,
    }


def _effect_class(point: float, simultaneous_lower: float) -> str:
    if point <= 0.0:
        return "directionally_nonpositive"
    if point < 0.05:
        return "directionally_positive_below_practical_floor"
    if simultaneous_lower <= 0.0:
        return "at_or_above_practical_floor_interval_inconclusive"
    return "at_or_above_practical_floor_positive_bound"


def _contrast_mechanism_valid(
    *,
    contrast_id: str,
    mechanism: Mapping[str, object],
) -> bool:
    frozen_theta0_is_a_real_steering_arm = (
        mechanism["frozen_theta0_physical_nonnoop_count"] >= 1
        and mechanism["steerable_actual_action_divergence_count"] >= 1
    )
    if contrast_id == _LEARNABLE_CONTRAST_ID:
        return bool(
            mechanism["learnable_actual_action_divergence_count"] >= 1
            and mechanism["full_learned_policy_differs_from_cold_root_count"] >= 1
            and frozen_theta0_is_a_real_steering_arm
        )
    if contrast_id == _STEERABLE_CONTRAST_ID:
        return bool(frozen_theta0_is_a_real_steering_arm)
    raise ValueError(f"unknown development contrast: {contrast_id}")


def _terminal_claims(
    contrast_reports: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    estimated_by_id = {
        contrast_id: bool(contrast_reports[contrast_id]["mechanism_valid"])
        for contrast_id in (_LEARNABLE_CONTRAST_ID, _STEERABLE_CONTRAST_ID)
    }
    go_candidate_by_id = {
        contrast_id: bool(
            contrast_reports[contrast_id]["development_go_candidate"]
        )
        for contrast_id in (_LEARNABLE_CONTRAST_ID, _STEERABLE_CONTRAST_ID)
    }
    global_power_prereg_authorized = all(estimated_by_id.values()) and any(
        go_candidate_by_id.values()
    )
    power_prereg_by_id = {
        contrast_id: global_power_prereg_authorized and go_candidate
        for contrast_id, go_candidate in go_candidate_by_id.items()
    }
    return {
        "development_campaign_execution_authorized": False,
        "development_campaign_completed": True,
        "development_contrast_estimated": any(estimated_by_id.values()),
        "development_contrasts_estimated_by_id": estimated_by_id,
        "confirmatory_effect_tested": False,
        "power_prereg_design_authorized": global_power_prereg_authorized,
        "development_go_candidate_by_contrast": go_candidate_by_id,
        "power_prereg_design_authorized_by_contrast": power_prereg_by_id,
        "reader_qualified": False,
        "appendable_effect": False,
        "readable_effect": False,
        "learnable_effect": False,
        "steerable_effect": False,
        "four_able_complete": False,
        "formal_evidence_authorized": False,
        "unseen_evidence_authorized": False,
        "integrated_horizon_authorized": False,
        "human_validation_complete": False,
        "production_active": False,
    }


def _contrast_status(
    *,
    mechanism_valid: bool,
    effect_class: str,
    durability_passed: bool,
    safety_passed: bool,
) -> str:
    if not mechanism_valid:
        return "arm_degeneracy_invalid_contrast_no_claim"
    if effect_class == "directionally_nonpositive":
        return "directionally_nonpositive_stop_scaleup_no_effect_claim"
    if effect_class == "directionally_positive_below_practical_floor":
        return (
            "directionally_positive_below_practical_floor_stop_scaleup_"
            "no_effect_claim"
        )
    if effect_class == "at_or_above_practical_floor_interval_inconclusive":
        return (
            "direction_at_or_above_floor_but_interval_inconclusive_"
            "no_effect_claim"
        )
    if not durability_passed:
        return "durability_guard_failed_stop_scaleup_no_effect_claim"
    if not safety_passed:
        return "safety_guard_failed_stop_scaleup_no_effect_claim"
    return "development_go_candidate_no_effect_claim"


def _build_report(
    *,
    protocol: RelationshipProductHorizonDevelopmentCampaignProtocol,
    plan_id: str,
    lineage_id: str,
    records: Sequence[_DecisionRecord],
    mechanism: Mapping[str, object],
) -> Mapping[str, object]:
    tables, arm_summary = _root_metric_table(records)
    safety_tables: dict[str, tuple[float, ...]] = {}
    record_slots = {
        (item.root_sequence_index, item.decision_index, item.arm_id): item
        for item in records
    }
    contrast_arms = {
        _LEARNABLE_CONTRAST_ID: (
            RelationshipProductHorizonCampaignArm.FULL,
            RelationshipProductHorizonCampaignArm.FROZEN_THETA0,
        ),
        _STEERABLE_CONTRAST_ID: (
            RelationshipProductHorizonCampaignArm.FROZEN_THETA0,
            RelationshipProductHorizonCampaignArm.STRICT_NOOP,
        ),
    }
    for contrast_id, (target, reference) in contrast_arms.items():
        safety_tables[contrast_id] = tuple(
            _mean(
                tuple(
                    float(
                        record_slots[(root_index, decision_index, target)].outcome
                        is DialogueExternalOutcomeKind.OVER_DIRECTIVE
                    )
                    - float(
                        record_slots[(root_index, decision_index, reference)].outcome
                        is DialogueExternalOutcomeKind.OVER_DIRECTIVE
                    )
                    for decision_index in _EVALUATION_INDICES
                )
            )
            for root_index in range(112)
        )
    root_metrics: dict[str, tuple[float, ...]] = {}
    for contrast_id, root_rows in tables.items():
        root_metrics[f"{contrast_id}.overall"] = tuple(
            _mean(row) for row in root_rows
        )
        for segment_id, indices in _SEGMENTS:
            local = tuple(index - 8 for index in indices)
            root_metrics[f"{contrast_id}.segment.{segment_id}"] = tuple(
                _mean(tuple(row[index] for index in local)) for row in root_rows
            )
        root_metrics[f"{contrast_id}.safety"] = safety_tables[contrast_id]
    boot = _bootstrap_statistics(root_metrics=root_metrics, protocol=protocol)
    observed = boot["observed"]
    distributions = boot["distributions"]
    q_index = 18999
    marginal_low = 499
    marginal_high = 19499
    primary_errors = []
    safety_errors = []
    for replicate_index in range(20000):
        primary_errors.append(
            max(
                observed[f"{contrast_id}.overall"]
                - distributions[f"{contrast_id}.overall"][replicate_index]
                for contrast_id in contrast_arms
            )
        )
        safety_errors.append(
            max(
                distributions[f"{contrast_id}.safety"][replicate_index]
                - observed[f"{contrast_id}.safety"]
                for contrast_id in contrast_arms
            )
        )
    primary_q = sorted(primary_errors)[q_index]
    safety_q = sorted(safety_errors)[q_index]
    contrast_reports: dict[str, Mapping[str, object]] = {}
    go_count = 0
    invalid_count = 0
    for contrast_id in contrast_arms:
        overall_key = f"{contrast_id}.overall"
        safety_key = f"{contrast_id}.safety"
        point = observed[overall_key]
        simultaneous_lower = point - primary_q
        safety_point = observed[safety_key]
        simultaneous_safety_upper = safety_point + safety_q
        segment_reports = {}
        segment_points = {}
        for segment_id, _ in _SEGMENTS:
            key = f"{contrast_id}.segment.{segment_id}"
            values = sorted(distributions[key])
            segment_points[segment_id] = observed[key]
            segment_reports[segment_id] = {
                "point_estimate_hex": _float_hex(observed[key]),
                "report_only_percentile_95_lower_hex": _float_hex(
                    values[marginal_low]
                ),
                "report_only_percentile_95_upper_hex": _float_hex(
                    values[marginal_high]
                ),
            }
        durability_passed = (
            sum(value >= 0.0 for value in segment_points.values()) >= 4
            and segment_points["return_after_gap"] > 0.0
            and segment_points["mixed_stress"] > 0.0
        )
        safety_passed = simultaneous_safety_upper <= 0.02
        mechanism_valid = _contrast_mechanism_valid(
            contrast_id=contrast_id,
            mechanism=mechanism,
        )
        classification = _effect_class(point, simultaneous_lower)
        primary_passed = (
            classification == "at_or_above_practical_floor_positive_bound"
        )
        go = mechanism_valid and primary_passed and durability_passed and safety_passed
        status = _contrast_status(
            mechanism_valid=mechanism_valid,
            effect_class=classification,
            durability_passed=durability_passed,
            safety_passed=safety_passed,
        )
        go_count += int(go)
        invalid_count += int(not mechanism_valid)
        overall_distribution = sorted(distributions[overall_key])
        safety_distribution = sorted(distributions[safety_key])
        failed_guards = []
        if not durability_passed:
            failed_guards.append("durability_guard_failed")
        if not safety_passed:
            failed_guards.append("safety_guard_failed")
        contrast_reports[contrast_id] = {
            "mechanism_valid": mechanism_valid,
            "effect_class": classification,
            "primary": {
                "point_estimate_hex": _float_hex(point),
                "practical_floor_hex": _float_hex(0.05),
                "simultaneous_95_lower_hex": _float_hex(simultaneous_lower),
                "report_only_percentile_95_lower_hex": _float_hex(
                    overall_distribution[marginal_low]
                ),
                "report_only_percentile_95_upper_hex": _float_hex(
                    overall_distribution[marginal_high]
                ),
                "gate_passed": primary_passed,
            },
            "segments": segment_reports,
            "durability": {
                "nonnegative_segment_count": sum(
                    value >= 0.0 for value in segment_points.values()
                ),
                "return_after_gap_strictly_positive": (
                    segment_points["return_after_gap"] > 0.0
                ),
                "mixed_stress_strictly_positive": (
                    segment_points["mixed_stress"] > 0.0
                ),
                "point_estimate_gate_passed": durability_passed,
            },
            "safety": {
                "point_risk_difference_hex": _float_hex(safety_point),
                "simultaneous_95_upper_hex": _float_hex(
                    simultaneous_safety_upper
                ),
                "margin_hex": _float_hex(0.02),
                "report_only_percentile_95_lower_hex": _float_hex(
                    safety_distribution[marginal_low]
                ),
                "report_only_percentile_95_upper_hex": _float_hex(
                    safety_distribution[marginal_high]
                ),
                "gate_passed": safety_passed,
            },
            "failed_guards": failed_guards,
            "icc_and_power_marker": _icc_and_power(
                tables[contrast_id], protocol=protocol
            ),
            "development_go_candidate": go,
            "status": status,
        }
    if invalid_count:
        status = _COMPLETE_INVALID
    elif go_count == 2:
        status = _SUCCESS_BOTH
    elif go_count == 1:
        status = _SUCCESS_SINGLE
    else:
        status = _COMPLETE_STOP
    claims = _terminal_claims(contrast_reports)
    core = {
        "schema_version": DEVELOPMENT_CAMPAIGN_REPORT_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "plan_id": plan_id,
        "campaign_input_lineage_id": lineage_id,
        "analysis_unit": "synthetic_root",
        "root_count": 112,
        "arm_summary": arm_summary,
        "mechanism": mechanism,
        "bootstrap": {
            "contract_id": "paired-whole-root-sha256-stream-bootstrap.v1",
            "replicate_count": 20000,
            "accepted_root_index_count": boot["accepted_root_index_count"],
            "root_index_stream_sha256": boot["index_stream_sha256"],
            "shared_resamples_across_all_endpoints": True,
            "simultaneous_primary_max_error_quantile_hex": _float_hex(primary_q),
            "simultaneous_safety_max_error_quantile_hex": _float_hex(safety_q),
            "confirmatory_coverage_claim": False,
        },
        "contrasts": contrast_reports,
        "status": status,
        "claims": claims,
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    return {"report_id": sha256_json(core), **core}


@dataclass(frozen=True)
class _PersistedRow:
    payload: Mapping[str, object]
    raw: bytes
    byte_offset_start: int
    byte_offset_end: int
    prefix_raw_sha256_before: str
    prefix_raw_sha256_after: str


class _CanonicalJsonlReader:
    def __init__(self, path: pathlib.Path) -> None:
        self._handle = pathlib.Path(path).open("rb")
        self._digest = hashlib.sha256()
        self._byte_count = 0
        self._row_count = 0
        self._closed = False

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def raw_bytes(self) -> int:
        return self._byte_count

    @property
    def raw_sha256(self) -> str:
        return self._digest.hexdigest()

    def next(self, *, source: str) -> _PersistedRow:
        raw = self._handle.readline()
        if not raw:
            raise ValueError(f"{source} ended before its frozen inventory")
        if not raw.endswith(b"\n") or raw == b"\n":
            raise ValueError(f"{source} row is not one canonical JSON line")
        payload = cal._parse_json_bytes(raw, source=source)
        if raw != cal._canonical_bytes(payload):
            raise ValueError(f"{source} row bytes are not canonical")
        prefix_before = self._digest.hexdigest()
        start = self._byte_count
        row_id = cal._digest(payload.get("row_id"), f"{source}.row_id")
        if payload.get("physical_sequence_index") != self._row_count:
            raise ValueError(f"{source} physical sequence drifted")
        core = {key: value for key, value in payload.items() if key != "row_id"}
        if sha256_json(core) != row_id:
            raise ValueError(f"{source} row identity drifted")
        self._digest.update(raw)
        self._byte_count += len(raw)
        self._row_count += 1
        return _PersistedRow(
            payload=payload,
            raw=raw,
            byte_offset_start=start,
            byte_offset_end=self._byte_count,
            prefix_raw_sha256_before=prefix_before,
            prefix_raw_sha256_after=self._digest.hexdigest(),
        )

    def require_eof(self, *, source: str) -> None:
        if self._handle.read(1):
            raise ValueError(f"{source} contains extra rows")

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


def _require_trace_row(
    row: _PersistedRow,
    *,
    record_type: str,
    expected_fields: frozenset[str],
) -> Mapping[str, object]:
    payload = row.payload
    cal._exact_keys(payload, set(expected_fields), f"campaign {record_type} row")
    if (
        payload["schema_version"] != DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION
        or payload["record_type"] != record_type
    ):
        raise ValueError(f"campaign {record_type} row identity drifted")
    return payload


_TRACE_BASE = frozenset(
    {"row_id", "physical_sequence_index", "schema_version", "record_type"}
)
_TRACE_FIELDS = {
    "header": _TRACE_BASE
    | {
        "protocol_id",
        "protocol_raw_sha256",
        "plan_id",
        "campaign_input_lineage_id",
        "forced_protocol_id",
        "forced_artifact_id",
        "public_plan_sha256",
        "root_count",
        "arm_ids",
        "evaluation_decision_indices",
        "credit_timestamp_formula",
        "rehearsal_enabled",
        "selected_branch_environment_opened",
        "model_invocation_count",
        "cuda_execution_count",
    },
    "root_start": _TRACE_BASE
    | {
        "root_sequence_index",
        "subject_id",
        "public_trajectory_sha256",
        "common_terminal_owner_persistence_sha256",
        "forced_transition_raw_sha256",
        "arm_order",
        "arm_initializations",
    },
        "preaction": _TRACE_BASE
    | {
        "root_sequence_index",
        "subject_id",
        "public_trajectory_sha256",
        "decision_index",
        "segment_id",
        "session_id",
        "decision_id",
        "arm_id",
        "physical_arm_order_index",
        "owner_input_persistence_sha256",
        "owner_preaction_persistence_sha256",
        "forecast_id",
        "forecast_sha256",
        "frozen_policy_id",
        "checkpoint_content_sha256",
        "checkpoint_update_count",
        "transition_batch_id",
        "transition_receipt_id",
        "frozen_decision_sha256",
        "gate_action",
        "candidate_action_id",
        "command_id",
        "authorization_id",
        "executor_receipt_id",
        "executor_receipt",
        "executor_disposition",
        "executor_status",
        "delivered_action_id",
        "executed_nonnoop",
        "branch_opened",
        "evaluation_or_judge_feedback_received",
    },
    "preaction_group_fsync": _TRACE_BASE
    | {
        "barrier_id",
        "barrier_schema_version",
        "protocol_id",
        "root_sequence_index",
        "decision_index",
        "arm_order",
        "preaction_row_ids",
        "preaction_executor_receipt_ids",
        "preaction_rows_start_index",
        "preaction_rows_end_index",
        "preaction_rows_raw_sha256",
        "durable_prefix_byte_count_before_receipt",
        "durable_prefix_raw_sha256_before_receipt",
    },
    "postaction": _TRACE_BASE
    | {
        "protocol_id",
        "preaction_barrier_id",
        "preaction_barrier_receipt_row_id",
        "root_sequence_index",
        "subject_id",
        "decision_index",
        "segment_id",
        "session_id",
        "decision_id",
        "arm_id",
        "physical_arm_order_index",
        "forecast_id",
        "executor_receipt_id",
        "candidate_action_id",
        "delivered_action_id",
        "environment_subject_id",
        "selected_branch_action_id",
        "selected_branch_commitment_id",
        "typed_outcome_id",
        "rendered_user_reaction_sha256",
        "environment_evidence_ref",
        "environment_version",
        "settlement_id",
        "external_outcome_evidence",
        "owner_outcome_evidence",
        "settlement",
        "social_prediction_error",
        "credit",
        "credit_applied_to_gate",
        "evaluation_gate_update_delta",
        "checkpoint_content_sha256",
        "checkpoint_update_count",
        "owner_preaction_persistence_sha256",
        "owner_postaction_persistence_sha256",
        "owner_writeback_changed_persistence",
        "evaluation_or_judge_feedback_received",
    },
    "postaction_group_fsync": _TRACE_BASE
    | {
        "postaction_receipt_id",
        "barrier_schema_version",
        "protocol_id",
        "preaction_barrier_id",
        "root_sequence_index",
        "decision_index",
        "postaction_row_ids",
        "postaction_rows_start_index",
        "postaction_rows_end_index",
        "postaction_rows_raw_sha256",
        "durable_prefix_byte_count_before_receipt",
        "durable_prefix_raw_sha256_before_receipt",
    },
    "root_terminal": _TRACE_BASE
    | {
        "root_sequence_index",
        "subject_id",
        "evaluation_decision_count_per_arm",
        "terminal_state_row_ids",
        "terminal_state_prefix_raw_sha256",
        "terminal_owner_persistence_sha256_by_arm",
        "evaluation_gate_update_count",
    },
    "terminal": _TRACE_BASE
    | {
        "protocol_id",
        "plan_id",
        "report_id",
        "completed_root_count",
        "completed_root_arm_count",
        "completed_evaluation_slot_count",
        "preaction_count",
        "preaction_group_fsync_count",
        "postaction_count",
        "postaction_group_fsync_count",
        "terminal_state_row_count",
        "mechanism",
        "status",
        "rehearsal_executed",
        "model_invocation_count",
        "cuda_execution_count",
    },
}
_TERMINAL_STATE_FIELDS = frozenset(
    {
        "row_id",
        "physical_sequence_index",
        "schema_version",
        "root_sequence_index",
        "subject_id",
        "arm_id",
        "physical_arm_order_index",
        "terminal_owner_persistence_sha256",
        "terminal_owner_persistence",
        "frozen_policy_id",
        "checkpoint_content_sha256",
        "checkpoint_update_count",
        "evaluation_gate_update_count",
    }
)


@dataclass(frozen=True)
class _PersistedAnalysis:
    report: Mapping[str, object]
    trace_row_count: int
    trace_raw_bytes: int
    trace_raw_sha256: str
    terminal_state_row_count: int
    terminal_state_raw_bytes: int
    terminal_state_raw_sha256: str


def _read_terminal_states(
    *,
    path: pathlib.Path,
    plan: Mapping[str, object],
) -> tuple[
    Mapping[tuple[int, RelationshipProductHorizonCampaignArm], Mapping[str, object]],
    int,
    int,
    str,
]:
    reader = _CanonicalJsonlReader(path)
    states: dict[
        tuple[int, RelationshipProductHorizonCampaignArm], Mapping[str, object]
    ] = {}
    try:
        plan_roots = cal._list(plan["roots"], "campaign plan roots")
        for raw_root in plan_roots:
            plan_root = cal._mapping(raw_root, "campaign plan root")
            root_index = cal._integer(
                plan_root["root_sequence_index"], "root_sequence_index"
            )
            subject_id = cal._text(plan_root["subject_id"], "subject_id")
            arm_order = tuple(
                RelationshipProductHorizonCampaignArm(value)
                for value in cal._list(plan_root["arm_order"], "arm_order")
            )
            for physical_index, arm in enumerate(arm_order):
                persisted = reader.next(source="campaign terminal state")
                payload = persisted.payload
                cal._exact_keys(
                    payload,
                    set(_TERMINAL_STATE_FIELDS),
                    "campaign terminal state",
                )
                if (
                    payload["schema_version"]
                    != DEVELOPMENT_CAMPAIGN_TERMINAL_STATE_SCHEMA_VERSION
                    or payload["root_sequence_index"] != root_index
                    or payload["subject_id"] != subject_id
                    or payload["arm_id"] != arm.value
                    or payload["physical_arm_order_index"] != physical_index
                    or payload["evaluation_gate_update_count"] != 0
                ):
                    raise ValueError("campaign terminal state slot drifted")
                owner = _owner_snapshot_from_payload(
                    payload["terminal_owner_persistence"]
                )
                if (
                    social_record_store_persistence_sha256(owner)
                    != payload["terminal_owner_persistence_sha256"]
                ):
                    raise ValueError("campaign terminal owner hash drifted")
                states[(root_index, arm)] = {
                    **payload,
                    "row_id": payload["row_id"],
                    "prefix_raw_sha256_after": (
                        persisted.prefix_raw_sha256_after
                    ),
                }
        reader.require_eof(source="campaign terminal state")
        if reader.row_count != 336:
            raise ValueError("campaign terminal state row count drifted")
        return states, reader.row_count, reader.raw_bytes, reader.raw_sha256
    finally:
        reader.close()


def _verify_group_receipt(
    *,
    receipt: _PersistedRow,
    members: Sequence[_PersistedRow],
    prefix: str,
) -> None:
    payload = receipt.payload
    row_ids_field = f"{prefix}_row_ids"
    start_field = f"{prefix}_rows_start_index"
    end_field = f"{prefix}_rows_end_index"
    raw_field = f"{prefix}_rows_raw_sha256"
    if (
        payload[row_ids_field] != [item.payload["row_id"] for item in members]
        or payload[start_field]
        != members[0].payload["physical_sequence_index"]
        or payload[end_field]
        != members[-1].payload["physical_sequence_index"]
        or payload[raw_field]
        != cal._sha256_bytes(b"".join(item.raw for item in members))
        or payload["durable_prefix_byte_count_before_receipt"]
        != members[-1].byte_offset_end
        or payload["durable_prefix_raw_sha256_before_receipt"]
        != receipt.prefix_raw_sha256_before
        or receipt.prefix_raw_sha256_before
        != members[-1].prefix_raw_sha256_after
    ):
        raise ValueError(f"campaign {prefix} durable group receipt drifted")


def _expected_persisted_transition_binding(
    *,
    arm: RelationshipProductHorizonCampaignArm,
    initialization: Mapping[str, object],
) -> tuple[object, object]:
    if arm is RelationshipProductHorizonCampaignArm.FULL:
        return initialization["batch_id"], initialization["batch_receipt_id"]
    return None, None


def _pulse_payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _temporal_advisory_from_payload(
    payload: object,
) -> TemporalActionAdvisoryProposal:
    raw = cal._mapping(payload, "campaign temporal advisory")
    cal._exact_keys(
        raw,
        {
            "advisory_id",
            "decision_id",
            "prediction_id",
            "action_id",
            "confidence",
            "policy_artifact_id",
            "policy_artifact_version",
            "evidence_refs",
            "rationale_codes",
            "evaluator_only",
            "active_authorized",
        },
        "campaign temporal advisory",
    )
    return TemporalActionAdvisoryProposal(
        advisory_id=cal._text(raw["advisory_id"], "advisory_id"),
        decision_id=cal._text(raw["decision_id"], "advisory decision_id"),
        prediction_id=cal._text(
            raw["prediction_id"], "advisory prediction_id"
        ),
        action_id=cal._text(raw["action_id"], "advisory action_id"),
        confidence=raw["confidence"],
        policy_artifact_id=cal._text(
            raw["policy_artifact_id"], "advisory policy_artifact_id"
        ),
        policy_artifact_version=cal._integer(
            raw["policy_artifact_version"], "advisory policy_artifact_version"
        ),
        evidence_refs=tuple(
            cal._text(item, "advisory evidence_ref")
            for item in cal._list(raw["evidence_refs"], "advisory evidence_refs")
        ),
        rationale_codes=tuple(
            cal._text(item, "advisory rationale_code")
            for item in cal._list(raw["rationale_codes"], "advisory rationale_codes")
        ),
        evaluator_only=cal._boolean(
            raw["evaluator_only"], "advisory evaluator_only"
        ),
        active_authorized=cal._boolean(
            raw["active_authorized"], "advisory active_authorized"
        ),
    )


def _temporal_delivery_from_payload(
    payload: object,
) -> RelationshipProductTemporalDelivery:
    raw = cal._mapping(payload, "campaign temporal delivery")
    cal._exact_keys(
        raw,
        {
            "slot_name",
            "owner",
            "version",
            "timestamp_ms",
            "active_abstract_action",
            "controller_params_hash",
            "action_family_version",
            "action_advisory_id",
            "action_advisory_status",
        },
        "campaign temporal delivery",
    )
    return RelationshipProductTemporalDelivery(
        slot_name=cal._text(raw["slot_name"], "temporal slot_name"),
        owner=cal._text(raw["owner"], "temporal owner"),
        version=cal._integer(raw["version"], "temporal version"),
        timestamp_ms=cal._integer(raw["timestamp_ms"], "temporal timestamp_ms"),
        active_abstract_action=cal._text(
            raw["active_abstract_action"], "temporal active action"
        ),
        controller_params_hash=cal._text(
            raw["controller_params_hash"], "temporal controller_params_hash"
        ),
        action_family_version=cal._integer(
            raw["action_family_version"], "temporal action_family_version"
        ),
        action_advisory_id=cal._text(
            raw["action_advisory_id"], "temporal action_advisory_id"
        ),
        action_advisory_status=TemporalActionAdvisoryStatus(
            cal._text(
                raw["action_advisory_status"], "temporal action_advisory_status"
            )
        ),
    )


def _validate_persisted_executor_receipt(
    *,
    preaction: Mapping[str, object],
    protocol_id: str,
    root_sequence_index: int,
    arm: RelationshipProductHorizonCampaignArm,
    initialization: RelationshipProductHorizonCampaignArmInitialization,
) -> PreferenceActionForecast:
    receipt = cal._mapping(
        preaction["executor_receipt"], "campaign executor receipt"
    )
    cal._exact_keys(
        receipt,
        {
            "receipt_id",
            "schema_version",
            "command",
            "authorization_id",
            "frozen_policy_id",
            "theta0_artifact_id",
            "owner_prestate_sha256",
            "checkpoint_content_sha256_before",
            "checkpoint_content_sha256_after",
            "policy_update_count_before",
            "policy_update_count_after",
            "evaluation_gate_update_delta",
            "pending_decision_count_before",
            "pending_decision_count_after",
            "forecast_sha256",
            "frozen_decision",
            "gate_selected_action_id",
            "intervention_candidate_action_id",
            "candidate_advisory",
            "executor_disposition",
            "executor_apply_bit",
            "executor_status",
            "candidate_non_noop",
            "candidate_applied",
            "strict_noop_substituted",
            "delivered_advisory",
            "delivered_action_id",
            "executed_non_noop",
            "action_diverged",
            "temporal_projection",
            "evaluator_or_judge_feedback_received",
        },
        "campaign executor receipt",
    )
    receipt_core = {
        key: value for key, value in receipt.items() if key != "receipt_id"
    }
    receipt_id = cal._text(receipt["receipt_id"], "executor receipt_id")
    if receipt_id != f"{_EXECUTOR_RECEIPT_PREFIX}{_pulse_payload_sha256(receipt_core)}":
        raise ValueError("campaign executor receipt identity drifted")

    command = cal._mapping(receipt["command"], "campaign executor command")
    cal._exact_keys(
        command,
        {
            "command_id",
            "schema_version",
            "forecast",
            "forecast_sha256",
            "frozen_policy",
            "frozen_decision",
            "authorization",
            "owner_prestate_sha256",
            "executor_disposition",
        },
        "campaign executor command",
    )
    command_core = {
        key: value for key, value in command.items() if key != "command_id"
    }
    command_id = cal._text(command["command_id"], "executor command_id")
    if command_id != f"{_EXECUTOR_COMMAND_PREFIX}{_pulse_payload_sha256(command_core)}":
        raise ValueError("campaign executor command identity drifted")

    forecast_payload = cal._mapping(command["forecast"], "campaign forecast")
    forecast = preference_action_forecast_from_payload(forecast_payload)
    frozen_policy = cal._mapping(
        command["frozen_policy"], "campaign frozen policy"
    )
    frozen_decision = cal._mapping(
        command["frozen_decision"], "campaign frozen decision"
    )
    decision = cal._mapping(
        frozen_decision["decision"], "campaign gate decision"
    )
    authorization = cal._mapping(
        command["authorization"], "campaign frozen authorization"
    )
    pulse_authorization = cal._mapping(
        authorization["pulse_authorization"], "campaign pulse authorization"
    )
    candidate_advisory = cal._mapping(
        receipt["candidate_advisory"], "campaign candidate advisory"
    )
    delivered_advisory = cal._mapping(
        receipt["delivered_advisory"], "campaign delivered advisory"
    )
    temporal_projection = cal._mapping(
        receipt["temporal_projection"], "campaign temporal projection"
    )
    delivered_action = cal._text(
        receipt["delivered_action_id"], "executor delivered action"
    )
    disposition = cal._text(
        receipt["executor_disposition"], "executor disposition"
    )
    if disposition not in {"apply_candidate", "force_strict_noop"}:
        raise ValueError("campaign executor disposition is outside frozen surface")
    candidate_nonnoop = cal._boolean(
        receipt["candidate_non_noop"], "candidate_non_noop"
    )
    expected_status = (
        "strict_noop"
        if disposition == "force_strict_noop"
        else "applied_candidate"
        if candidate_nonnoop
        else "gate_noop"
    )
    expected_delivered = (
        RelationshipAction.NEUTRAL_NOOP.value
        if disposition == "force_strict_noop"
        else cal._text(
            receipt["gate_selected_action_id"], "gate selected action"
        )
    )
    if (
        receipt["schema_version"] != RELATIONSHIP_PRODUCT_EXECUTOR_SCHEMA_VERSION
        or command["schema_version"]
        != RELATIONSHIP_PRODUCT_EXECUTOR_SCHEMA_VERSION
        or receipt_id != preaction["executor_receipt_id"]
        or command_id != preaction["command_id"]
        or receipt["command"] != command
        or preference_action_forecast_to_payload(forecast) != forecast_payload
        or command["forecast_sha256"] != _pulse_payload_sha256(forecast_payload)
        or receipt["forecast_sha256"] != _pulse_payload_sha256(forecast_payload)
        or receipt["forecast_sha256"] != preaction["forecast_sha256"]
        or forecast.forecast_id != preaction["forecast_id"]
        or frozen_policy["policy_id"] != preaction["frozen_policy_id"]
        or frozen_policy["checkpoint_content_sha256"]
        != preaction["checkpoint_content_sha256"]
        or frozen_policy["checkpoint_update_count"]
        != preaction["checkpoint_update_count"]
        or frozen_policy["transition_batch_id"]
        != preaction["transition_batch_id"]
        or frozen_policy["transition_receipt_id"]
        != preaction["transition_receipt_id"]
        or sha256_json(frozen_decision)
        != preaction["frozen_decision_sha256"]
        or frozen_decision != receipt["frozen_decision"]
        or decision["gate_action"] != preaction["gate_action"]
        or decision["selected_action_id"] != preaction["candidate_action_id"]
        or pulse_authorization["authorization_id"]
        != preaction["authorization_id"]
        or receipt["authorization_id"] != preaction["authorization_id"]
        or receipt["frozen_policy_id"] != preaction["frozen_policy_id"]
        or receipt["owner_prestate_sha256"]
        != preaction["owner_preaction_persistence_sha256"]
        or receipt["checkpoint_content_sha256_before"]
        != preaction["checkpoint_content_sha256"]
        or receipt["checkpoint_content_sha256_after"]
        != preaction["checkpoint_content_sha256"]
        or receipt["policy_update_count_before"]
        != preaction["checkpoint_update_count"]
        or receipt["policy_update_count_after"]
        != preaction["checkpoint_update_count"]
        or receipt["evaluation_gate_update_delta"] != 0
        or receipt["pending_decision_count_before"]
        != receipt["pending_decision_count_after"]
        or receipt["gate_selected_action_id"]
        != preaction["candidate_action_id"]
        or receipt["intervention_candidate_action_id"]
        != preaction["candidate_action_id"]
        or candidate_advisory["action_id"] != preaction["candidate_action_id"]
        or candidate_nonnoop
        is not (
            decision["gate_action"] == "steer"
            and preaction["candidate_action_id"]
            != RelationshipAction.NEUTRAL_NOOP.value
        )
        or command["executor_disposition"] != disposition
        or disposition != preaction["executor_disposition"]
        or receipt["executor_apply_bit"] is not (
            disposition == "apply_candidate"
        )
        or receipt["candidate_applied"] is not (
            disposition == "apply_candidate"
        )
        or receipt["strict_noop_substituted"] is not (
            disposition == "force_strict_noop"
        )
        or receipt["executor_status"] != expected_status
        or receipt["executor_status"] != preaction["executor_status"]
        or delivered_action != expected_delivered
        or delivered_action != preaction["delivered_action_id"]
        or delivered_advisory["action_id"] != delivered_action
        or temporal_projection["active_abstract_action"] != delivered_action
        or temporal_projection["action_advisory_id"]
        != delivered_advisory["advisory_id"]
        or receipt["executed_non_noop"] is not (
            delivered_action != RelationshipAction.NEUTRAL_NOOP.value
        )
        or receipt["executed_non_noop"] is not preaction["executed_nonnoop"]
        or receipt["action_diverged"] is not (
            delivered_action != preaction["candidate_action_id"]
        )
        or receipt["evaluator_or_judge_feedback_received"] is not False
    ):
        raise ValueError("campaign executor receipt exact join drifted")
    expected_command = RelationshipProductExecutorCommand(
        forecast=forecast,
        frozen_policy=initialization.frozen_policy,
        frozen_decision=initialization.frozen_policy.decide(forecast),
        authorization=_authorization(
            protocol_id=protocol_id,
            root_sequence_index=root_sequence_index,
            arm_id=arm,
            initialization=initialization,
        ),
        owner_prestate_sha256=cal._digest(
            preaction["owner_preaction_persistence_sha256"],
            "owner_preaction_persistence_sha256",
        ),
        executor_disposition=initialization.executor_disposition,
    )
    expected_receipt = RelationshipProductExecutorReceipt(
        command=expected_command,
        candidate_advisory=_temporal_advisory_from_payload(
            receipt["candidate_advisory"]
        ),
        delivered_advisory=_temporal_advisory_from_payload(
            receipt["delivered_advisory"]
        ),
        temporal_delivery=_temporal_delivery_from_payload(
            receipt["temporal_projection"]
        ),
    )
    if (
        expected_command.to_payload() != command
        or expected_command.command_id != command_id
        or expected_receipt.to_payload() != receipt
        or expected_receipt.receipt_id != receipt_id
    ):
        raise ValueError("campaign executor owner replay drifted")
    return forecast


def _validate_persisted_settlement_join(
    *,
    postaction: Mapping[str, object],
    preaction: Mapping[str, object],
    forecast: PreferenceActionForecast,
    root: HorizonPublicRoot,
    decision: HorizonPublicDecisionSession,
    root_sequence_index: int,
) -> DialogueExternalOutcomeKind:
    external_payload = cal._mapping(
        postaction["external_outcome_evidence"],
        "campaign external outcome evidence",
    )
    cal._exact_keys(
        external_payload,
        set(
            _external_outcome_evidence_payload(
                DialogueExternalOutcomeEvidence(
                    evidence_id="shape",
                    turn_index=0,
                    kind=DialogueExternalOutcomeKind.HELPED,
                    source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
                    confidence=1.0,
                    evidence_ref="shape",
                )
            )
        ),
        "campaign external outcome evidence",
    )
    outcome = DialogueExternalOutcomeKind(
        cal._text(external_payload["kind"], "external outcome kind")
    )
    if outcome not in RELATIONSHIP_OUTCOMES:
        raise ValueError("campaign outcome is outside the frozen product surface")
    typing_values = tuple(
        external_payload[field]
        for field in (
            "typing_qualification_id",
            "typing_qualification_sha256",
            "typing_runtime_id",
            "typing_schema_version",
        )
    )
    if any(not isinstance(value, str) for value in typing_values) or any(
        typing_values
    ):
        raise ValueError("environment outcome cannot carry typing lineage")
    external_evidence = DialogueExternalOutcomeEvidence(
        evidence_id=cal._text(external_payload["evidence_id"], "evidence_id"),
        turn_index=cal._integer(external_payload["turn_index"], "turn_index"),
        kind=outcome,
        source=DialogueExternalOutcomeEvidenceSource(
            cal._text(external_payload["source"], "external outcome source")
        ),
        confidence=float.fromhex(
            cal._text(external_payload["confidence_hex"], "confidence_hex")
        ),
        evidence_ref=cal._text(
            external_payload["evidence_ref"], "external evidence_ref"
        ),
        description=cal._text(
            external_payload["description"], "external description"
        ),
        session_scope=cal._text(
            external_payload["session_scope"], "external session_scope"
        ),
        action_turn_index=cal._integer(
            external_payload["action_turn_index"], "external action_turn_index"
        ),
        forecast_id=cal._text(
            external_payload["forecast_id"], "external forecast_id"
        ),
        decision_id=cal._text(
            external_payload["decision_id"], "external decision_id"
        ),
        action_id=cal._text(
            external_payload["action_id"], "external action_id"
        ),
        typing_qualification_id=typing_values[0],
        typing_qualification_sha256=typing_values[1],
        typing_runtime_id=typing_values[2],
        typing_schema_version=typing_values[3],
    )
    branch_outcome = dynamic.RelationshipProductHorizonSelectedBranchOutcome(
        environment_subject_id=cal._text(
            postaction["environment_subject_id"], "environment_subject_id"
        ),
        selected_action=RelationshipAction(
            cal._text(
                postaction["selected_branch_action_id"],
                "selected_branch_action_id",
            )
        ),
        typed_outcome=outcome,
        rendered_user_reaction=external_evidence.description,
        environment_evidence_ref=cal._text(
            postaction["environment_evidence_ref"], "environment_evidence_ref"
        ),
        environment_version=cal._text(
            postaction["environment_version"], "environment_version"
        ),
        commitment_id=cal._digest(
            postaction["selected_branch_commitment_id"],
            "selected_branch_commitment_id",
        ),
    )
    owner_payload = cal._mapping(
        postaction["owner_outcome_evidence"],
        "campaign owner outcome evidence",
    )
    cal._exact_keys(
        owner_payload,
        {
            "evidence_id",
            "interlocutor_id",
            "observation_summary",
            "action_id",
            "observed_outcome_id",
            "reaction_summary",
            "source_turn",
            "evidence_refs",
        },
        "campaign owner outcome evidence",
    )
    owner_evidence = PreferenceActionOutcomeEvidence(
        evidence_id=cal._text(owner_payload["evidence_id"], "owner evidence_id"),
        interlocutor_id=cal._text(
            owner_payload["interlocutor_id"], "owner interlocutor_id"
        ),
        observation_summary=cal._text(
            owner_payload["observation_summary"], "owner observation_summary"
        ),
        action_id=cal._text(owner_payload["action_id"], "owner action_id"),
        observed_outcome_id=cal._text(
            owner_payload["observed_outcome_id"], "owner observed_outcome_id"
        ),
        reaction_summary=cal._text(
            owner_payload["reaction_summary"], "owner reaction_summary"
        ),
        source_turn=cal._integer(
            owner_payload["source_turn"], "owner source_turn"
        ),
        evidence_refs=tuple(
            cal._text(item, "owner evidence_ref")
            for item in cal._list(
                owner_payload["evidence_refs"], "owner evidence_refs"
            )
        ),
    )
    action_turn = 4 + 2 * decision.decision_index
    if (
        _external_outcome_evidence_payload(external_evidence) != external_payload
        or _owner_outcome_evidence_payload(owner_evidence) != owner_payload
        or external_evidence.evidence_id
        != f"relationship-product-outcome:{decision.decision_id}"
        or branch_outcome.environment_subject_id != root.subject_id
        or branch_outcome.selected_action.value
        != preaction["delivered_action_id"]
        or branch_outcome.typed_outcome is not external_evidence.kind
        or branch_outcome.environment_evidence_ref
        != external_evidence.evidence_ref
        or external_evidence.turn_index != action_turn + 1
        or external_evidence.source
        is not DialogueExternalOutcomeEvidenceSource.ENVIRONMENT
        or external_evidence.confidence != 1.0
        or external_evidence.evidence_ref
        != postaction["environment_evidence_ref"]
        or cal._sha256_text(external_evidence.description)
        != postaction["rendered_user_reaction_sha256"]
        or external_evidence.session_scope != root.subject_id
        or external_evidence.action_turn_index != action_turn
        or external_evidence.forecast_id != forecast.forecast_id
        or external_evidence.decision_id != decision.decision_id
        or external_evidence.action_id != postaction["delivered_action_id"]
        or external_evidence.kind.value != postaction["typed_outcome_id"]
        or any(
            (
                external_evidence.typing_qualification_id,
                external_evidence.typing_qualification_sha256,
                external_evidence.typing_runtime_id,
                external_evidence.typing_schema_version,
            )
        )
        or owner_evidence.evidence_id != external_evidence.evidence_id
        or owner_evidence.interlocutor_id != _INTERLOCUTOR_ID
        or owner_evidence.observation_summary != decision.current_input
        or owner_evidence.action_id != external_evidence.action_id
        or owner_evidence.observed_outcome_id != external_evidence.kind.value
        or owner_evidence.reaction_summary != external_evidence.description
        or owner_evidence.source_turn != external_evidence.turn_index
        or owner_evidence.evidence_refs != (external_evidence.evidence_ref,)
    ):
        raise ValueError("campaign source-to-owner evidence join drifted")

    expected_settlement = settle_preference_action_forecast(
        forecast=forecast,
        evidence=external_evidence,
    )
    settlement_payload = cal._mapping(
        postaction["settlement"], "campaign forecast settlement"
    )
    if (
        settlement_payload != _forecast_settlement_payload(expected_settlement)
        or postaction["settlement_id"] != expected_settlement.settlement_id
    ):
        raise ValueError("campaign owner settlement exact derivation drifted")

    pe_payload = cal._mapping(
        postaction["social_prediction_error"], "campaign social PE"
    )
    cal._exact_keys(
        pe_payload,
        {"description", "errors"},
        "campaign social PE",
    )
    errors = tuple(
        cal._mapping(item, "campaign social PE error")
        for item in cal._list(pe_payload["errors"], "campaign social PE errors")
    )
    for item in errors:
        cal._exact_keys(
            item,
            {
                "error_id",
                "prediction_id",
                "kind",
                "outcome",
                "magnitude_hex",
                "owner",
                "scope_kind",
                "scope_id",
                "evidence",
            },
            "campaign social PE error",
        )
    expected_error_id = f"social-pe:{expected_settlement.settlement_id}"
    expected_error_evidence = (
        f"forecast_settlement:{expected_settlement.settlement_id}",
        f"external_outcome:{expected_settlement.source_evidence_id}",
        f"action:{expected_settlement.action_id}",
        f"observed_outcome:{expected_settlement.observed_outcome_id}",
        "predicted_probability="
        f"{expected_settlement.predicted_probability:.12f}",
        "negative_log_likelihood="
        f"{expected_settlement.negative_log_likelihood:.12f}",
        "signed_utility_prediction_error="
        f"{expected_settlement.signed_utility_prediction_error:.12f}",
    )
    expected_error_payload = {
        "error_id": expected_error_id,
        "prediction_id": expected_settlement.forecast_id,
        "kind": SocialPredictionKind.PREFERENCE_ABOUT_OTHER.value,
        "outcome": expected_settlement.outcome.value,
        "magnitude_hex": expected_settlement.magnitude.hex(),
        "owner": PreferenceAboutOtherModule.owner,
        "scope_kind": SocialScopeKind.INTERLOCUTOR.value,
        "scope_id": expected_settlement.interlocutor_id,
        "evidence": list(expected_error_evidence),
    }
    matching_errors = tuple(
        item for item in errors if item.get("error_id") == expected_error_id
    )
    if matching_errors != (expected_error_payload,):
        raise ValueError("campaign settlement-to-social-PE join drifted")
    social_error = SocialPredictionError(
        error_id=expected_error_id,
        prediction_id=expected_settlement.forecast_id,
        kind=SocialPredictionKind.PREFERENCE_ABOUT_OTHER,
        outcome=SocialPredictionOutcome(expected_settlement.outcome.value),
        magnitude=expected_settlement.magnitude,
        owner=PreferenceAboutOtherModule.owner,
        scope_kind=SocialScopeKind.INTERLOCUTOR,
        scope_id=expected_settlement.interlocutor_id,
        evidence=expected_error_evidence,
    )
    expected_credits = derive_preference_action_forecast_credit_records(
        settlements=(expected_settlement,),
        social_errors=(social_error,),
        settled_at_turn=expected_settlement.observed_turn,
        timestamp_ms=_credit_timestamp(root_sequence_index, decision.decision_index),
    )
    if len(expected_credits) != 1 or cal._credit_payload(
        expected_credits[0]
    ) != cal._mapping(postaction["credit"], "campaign credit"):
        raise ValueError("campaign social-PE-to-credit exact derivation drifted")
    return outcome


def _analyze_persisted_evidence(
    *,
    dependencies: _Dependencies,
    plan: Mapping[str, object],
    trace_path: pathlib.Path,
    terminal_state_path: pathlib.Path,
) -> _PersistedAnalysis:
    states, state_count, state_bytes, state_sha = _read_terminal_states(
        path=terminal_state_path,
        plan=plan,
    )
    reader = _CanonicalJsonlReader(trace_path)
    records: list[_DecisionRecord] = []
    action_counts = {arm.value: Counter() for arm in _ARM_IDS}
    settlement_slots: set[tuple[str, str]] = set()
    credit_slots: set[tuple[str, str]] = set()
    branch_ids: set[str] = set()
    learnable_divergence_count = 0
    steerable_divergence_count = 0
    frozen_nonnoop_count = 0
    later_handoff_count = 0
    writeback_count = 0
    full_diff_root_count = 0
    terminal_payload: Mapping[str, object] | None = None
    try:
        header_row = reader.next(source="campaign trace header")
        header = _require_trace_row(
            header_row,
            record_type="header",
            expected_fields=_TRACE_FIELDS["header"],
        )
        if (
            header["protocol_id"] != dependencies.protocol.protocol_id
            or header["protocol_raw_sha256"] != dependencies.protocol.raw_sha256
            or header["plan_id"] != plan["plan_id"]
            or header["campaign_input_lineage_id"]
            != dependencies.inputs.lineage_id
            or header["forced_protocol_id"]
            != dependencies.inputs.forced_protocol_id
            or header["forced_artifact_id"]
            != dependencies.inputs.forced_artifact_id
            or header["public_plan_sha256"]
            != dependencies.inputs.public_plan_sha256
            or header["root_count"] != 112
            or header["arm_ids"] != [item.value for item in _ARM_IDS]
            or header["evaluation_decision_indices"] != list(_EVALUATION_INDICES)
            or header["credit_timestamp_formula"]
            != (
                "root_sequence_index_times_100_plus_5_plus_2_times_"
                "decision_index"
            )
            or header["rehearsal_enabled"] is not False
            or header["selected_branch_environment_opened"] is not False
            or header["model_invocation_count"] != 0
            or header["cuda_execution_count"] != 0
        ):
            raise ValueError("campaign trace header drifted")
        plan_roots = cal._list(plan["roots"], "campaign plan roots")
        prior_post_sha: dict[
            tuple[int, RelationshipProductHorizonCampaignArm], str
        ] = {}
        for raw_plan_root in plan_roots:
            plan_root = cal._mapping(raw_plan_root, "campaign plan root")
            root_index = cal._integer(
                plan_root["root_sequence_index"], "root_sequence_index"
            )
            subject_id = cal._text(plan_root["subject_id"], "subject_id")
            root_input = dependencies.inputs.roots[root_index]
            root = root_input.public_root
            if (
                root_input.root_sequence_index != root_index
                or root.subject_id != subject_id
            ):
                raise ValueError("campaign forced input root order drifted")
            arm_order = tuple(
                RelationshipProductHorizonCampaignArm(value)
                for value in cal._list(plan_root["arm_order"], "arm_order")
            )
            root_start_row = reader.next(source="campaign root start")
            root_start = _require_trace_row(
                root_start_row,
                record_type="root_start",
                expected_fields=_TRACE_FIELDS["root_start"],
            )
            if (
                root_start["root_sequence_index"] != root_index
                or root_start["subject_id"] != subject_id
                or root_start["public_trajectory_sha256"]
                != plan_root["public_trajectory_sha256"]
                or root_start["common_terminal_owner_persistence_sha256"]
                != plan_root["common_terminal_owner_persistence_sha256"]
                or root_start["forced_transition_raw_sha256"]
                != plan_root["forced_transition_raw_sha256"]
                or root_start["forced_transition_raw_sha256"]
                != root_input.transition_raw_sha256
                or root_start["arm_order"] != [arm.value for arm in arm_order]
            ):
                raise ValueError("campaign root start join drifted")
            initializations = cal._list(
                root_start["arm_initializations"], "arm initializations"
            )
            if len(initializations) != 3:
                raise ValueError("campaign root initialization count drifted")
            initialization_by_arm = {
                RelationshipProductHorizonCampaignArm(
                    cal._text(cal._mapping(item, "arm initialization")["arm_id"], "arm_id")
                ): cal._mapping(item, "arm initialization")
                for item in initializations
            }
            if tuple(initialization_by_arm) != arm_order:
                raise ValueError("campaign physical arm initialization order drifted")
            fresh_initialization_by_arm = {
                item.arm_id: item for item in root_input.fresh_arm_initializations()
            }
            if (
                tuple(fresh_initialization_by_arm) != _ARM_IDS
                or initializations
                != [
                    _arm_initialization_payload(
                        arm=arm,
                        initialization=fresh_initialization_by_arm[arm],
                    )
                    for arm in arm_order
                ]
            ):
                raise ValueError("campaign root initialization input join drifted")
            initialization_fields = {
                "arm_id",
                "batch_id",
                "batch_receipt_id",
                "batch_disposition",
                "frozen_policy_id",
                "checkpoint_content_sha256",
                "checkpoint_update_count",
                "executor_disposition",
            }
            for arm, initialization in initialization_by_arm.items():
                cal._exact_keys(
                    initialization,
                    initialization_fields,
                    f"campaign {arm.value} initialization",
                )
            if (
                initialization_by_arm[RelationshipProductHorizonCampaignArm.FULL][
                    "checkpoint_update_count"
                ]
                != 8
                or initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.FROZEN_THETA0
                ]["checkpoint_update_count"]
                != 0
                or initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.STRICT_NOOP
                ]["checkpoint_update_count"]
                != 0
                or initialization_by_arm[RelationshipProductHorizonCampaignArm.FULL][
                    "batch_disposition"
                ]
                != "apply"
                or initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.FROZEN_THETA0
                ]["batch_disposition"]
                != "withhold"
                or initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.STRICT_NOOP
                ]["batch_disposition"]
                != "withhold"
                or initialization_by_arm[RelationshipProductHorizonCampaignArm.FULL][
                    "executor_disposition"
                ]
                != "apply_candidate"
                or initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.FROZEN_THETA0
                ]["executor_disposition"]
                != "apply_candidate"
                or initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.STRICT_NOOP
                ]["executor_disposition"]
                != "force_strict_noop"
                or len(
                    {
                        initialization_by_arm[arm]["batch_id"]
                        for arm in _ARM_IDS
                    }
                )
                != 1
                or initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.FROZEN_THETA0
                ]["batch_receipt_id"]
                != initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.STRICT_NOOP
                ]["batch_receipt_id"]
                or initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.FROZEN_THETA0
                ]["frozen_policy_id"]
                != initialization_by_arm[
                    RelationshipProductHorizonCampaignArm.STRICT_NOOP
                ]["frozen_policy_id"]
            ):
                raise ValueError("campaign root checkpoint isolation drifted")
            full_diff_root_count += int(
                _full_policy_differs_from_cold(
                    full_policy_id=initialization_by_arm[
                        RelationshipProductHorizonCampaignArm.FULL
                    ]["frozen_policy_id"],
                    full_checkpoint_content_sha256=initialization_by_arm[
                        RelationshipProductHorizonCampaignArm.FULL
                    ]["checkpoint_content_sha256"],
                    cold_policy_id=initialization_by_arm[
                        RelationshipProductHorizonCampaignArm.FROZEN_THETA0
                    ]["frozen_policy_id"],
                    cold_checkpoint_content_sha256=initialization_by_arm[
                        RelationshipProductHorizonCampaignArm.FROZEN_THETA0
                    ]["checkpoint_content_sha256"],
                )
            )
            decisions = cal._list(
                plan_root["evaluation_decisions"], "evaluation decisions"
            )
            for raw_plan_decision in decisions:
                plan_decision = cal._mapping(
                    raw_plan_decision, "campaign plan decision"
                )
                decision_index = cal._integer(
                    plan_decision["decision_index"], "decision_index"
                )
                public_decision = root.decision_sessions[decision_index]
                if (
                    public_decision.decision_index != decision_index
                    or public_decision.session_id != plan_decision["session_id"]
                    or public_decision.decision_id != plan_decision["decision_id"]
                    or sha256_json(public_decision.to_payload())
                    != plan_decision["public_decision_sha256"]
                ):
                    raise ValueError("campaign public decision join drifted")
                pre_members: list[_PersistedRow] = []
                pre_by_arm: dict[
                    RelationshipProductHorizonCampaignArm, Mapping[str, object]
                ] = {}
                pre_forecast_by_arm: dict[
                    RelationshipProductHorizonCampaignArm,
                    PreferenceActionForecast,
                ] = {}
                for physical_index, arm in enumerate(arm_order):
                    row = reader.next(source="campaign preaction")
                    pre = _require_trace_row(
                        row,
                        record_type="preaction",
                        expected_fields=_TRACE_FIELDS["preaction"],
                    )
                    (
                        expected_transition_batch_id,
                        expected_transition_receipt_id,
                    ) = _expected_persisted_transition_binding(
                        arm=arm,
                        initialization=initialization_by_arm[arm],
                    )
                    if (
                        pre["root_sequence_index"] != root_index
                        or pre["subject_id"] != subject_id
                        or pre["public_trajectory_sha256"]
                        != root.public_trajectory_sha256
                        or pre["decision_index"] != decision_index
                        or pre["segment_id"] != plan_decision["segment_id"]
                        or pre["session_id"] != plan_decision["session_id"]
                        or pre["decision_id"] != plan_decision["decision_id"]
                        or pre["arm_id"] != arm.value
                        or pre["physical_arm_order_index"] != physical_index
                        or pre["branch_opened"] is not False
                        or pre["evaluation_or_judge_feedback_received"] is not False
                        or pre["frozen_policy_id"]
                        != initialization_by_arm[arm]["frozen_policy_id"]
                        or pre["checkpoint_content_sha256"]
                        != initialization_by_arm[arm]["checkpoint_content_sha256"]
                        or pre["checkpoint_update_count"]
                        != initialization_by_arm[arm]["checkpoint_update_count"]
                        or pre["transition_batch_id"]
                        != expected_transition_batch_id
                        or pre["transition_receipt_id"]
                        != expected_transition_receipt_id
                        or pre["executor_disposition"]
                        != initialization_by_arm[arm]["executor_disposition"]
                        or pre["authorization_id"]
                        != (
                            "relationship-product-horizon-development-campaign:"
                            f"{dependencies.protocol.protocol_id}:root:"
                            f"{root_index:03d}:arm:{arm.value}"
                        )
                    ):
                        raise ValueError("campaign preaction slot drifted")
                    expected_owner = (
                        root_start["common_terminal_owner_persistence_sha256"]
                        if decision_index == 8
                        else prior_post_sha[(root_index, arm)]
                    )
                    if pre["owner_input_persistence_sha256"] != expected_owner:
                        raise ValueError("campaign owner handoff drifted")
                    if decision_index != 8:
                        later_handoff_count += 1
                    pre_forecast_by_arm[arm] = (
                        _validate_persisted_executor_receipt(
                            preaction=pre,
                            protocol_id=dependencies.protocol.protocol_id,
                            root_sequence_index=root_index,
                            arm=arm,
                            initialization=fresh_initialization_by_arm[arm],
                        )
                    )
                    pre_members.append(row)
                    pre_by_arm[arm] = pre
                receipt_row = reader.next(source="campaign preaction receipt")
                receipt = _require_trace_row(
                    receipt_row,
                    record_type="preaction_group_fsync",
                    expected_fields=_TRACE_FIELDS["preaction_group_fsync"],
                )
                _verify_group_receipt(
                    receipt=receipt_row, members=pre_members, prefix="preaction"
                )
                barrier_core = {
                    key: value
                    for key, value in receipt.items()
                    if key
                    not in {
                        "row_id",
                        "physical_sequence_index",
                        "schema_version",
                        "record_type",
                        "barrier_id",
                    }
                }
                if (
                    receipt["barrier_id"] != sha256_json(barrier_core)
                    or receipt["protocol_id"] != dependencies.protocol.protocol_id
                    or receipt["root_sequence_index"] != root_index
                    or receipt["decision_index"] != decision_index
                    or receipt["arm_order"] != [arm.value for arm in arm_order]
                    or receipt["preaction_executor_receipt_ids"]
                    != [pre_by_arm[arm]["executor_receipt_id"] for arm in arm_order]
                ):
                    raise ValueError("campaign preaction barrier identity drifted")
                post_members: list[_PersistedRow] = []
                post_by_arm: dict[
                    RelationshipProductHorizonCampaignArm, Mapping[str, object]
                ] = {}
                branch_signature_by_action: dict[
                    RelationshipAction, tuple[object, ...]
                ] = {}
                for physical_index, arm in enumerate(arm_order):
                    row = reader.next(source="campaign postaction")
                    post = _require_trace_row(
                        row,
                        record_type="postaction",
                        expected_fields=_TRACE_FIELDS["postaction"],
                    )
                    pre = pre_by_arm[arm]
                    credit = cal._mapping(post["credit"], "campaign credit")
                    if (
                        post["protocol_id"] != dependencies.protocol.protocol_id
                        or post["preaction_barrier_id"] != receipt["barrier_id"]
                        or post["preaction_barrier_receipt_row_id"]
                        != receipt["row_id"]
                        or post["root_sequence_index"] != root_index
                        or post["subject_id"] != subject_id
                        or post["decision_index"] != decision_index
                        or post["segment_id"] != plan_decision["segment_id"]
                        or post["session_id"] != plan_decision["session_id"]
                        or post["decision_id"] != plan_decision["decision_id"]
                        or post["arm_id"] != arm.value
                        or post["physical_arm_order_index"] != physical_index
                        or post["forecast_id"] != pre["forecast_id"]
                        or post["executor_receipt_id"]
                        != pre["executor_receipt_id"]
                        or post["candidate_action_id"] != pre["candidate_action_id"]
                        or post["delivered_action_id"] != pre["delivered_action_id"]
                        or post["selected_branch_action_id"]
                        != pre["delivered_action_id"]
                        or post["environment_subject_id"] != subject_id
                        or post["owner_preaction_persistence_sha256"]
                        != pre["owner_preaction_persistence_sha256"]
                        or post["checkpoint_content_sha256"]
                        != pre["checkpoint_content_sha256"]
                        or post["checkpoint_update_count"]
                        != pre["checkpoint_update_count"]
                        or post["credit_applied_to_gate"] is not False
                        or post["evaluation_gate_update_delta"] != 0
                        or post["evaluation_or_judge_feedback_received"] is not False
                        or credit["abstract_action_id"] != post["delivered_action_id"]
                        or credit["prediction_id"] != post["forecast_id"]
                        or credit["timestamp_ms"]
                        != _credit_timestamp(root_index, decision_index)
                    ):
                        raise ValueError("campaign postaction exact join drifted")
                    candidate = RelationshipAction(post["candidate_action_id"])
                    delivered = RelationshipAction(post["delivered_action_id"])
                    outcome = _validate_persisted_settlement_join(
                        postaction=post,
                        preaction=pre,
                        forecast=pre_forecast_by_arm[arm],
                        root=root,
                        decision=public_decision,
                        root_sequence_index=root_index,
                    )
                    if (
                        arm is RelationshipProductHorizonCampaignArm.STRICT_NOOP
                        and delivered is not RelationshipAction.NEUTRAL_NOOP
                    ):
                        raise ValueError("campaign strict arm delivered non-noop")
                    branch_signature = (
                        post["selected_branch_commitment_id"],
                        post["typed_outcome_id"],
                        post["rendered_user_reaction_sha256"],
                        post["environment_evidence_ref"],
                        post["environment_version"],
                    )
                    prior_branch_signature = branch_signature_by_action.setdefault(
                        delivered,
                        branch_signature,
                    )
                    if prior_branch_signature != branch_signature:
                        raise ValueError(
                            "campaign same-action selected branch was not reused"
                        )
                    records.append(
                        _DecisionRecord(
                            root_sequence_index=root_index,
                            decision_index=decision_index,
                            segment_id=cal._text(post["segment_id"], "segment_id"),
                            arm_id=arm,
                            candidate_action=candidate,
                            delivered_action=delivered,
                            outcome=outcome,
                        )
                    )
                    action_counts[arm.value][delivered.value] += 1
                    branch_ids.add(
                        cal._digest(
                            post["selected_branch_commitment_id"],
                            "selected branch commitment",
                        )
                    )
                    settlement_slot = (
                        arm.value,
                        cal._text(post["settlement_id"], "settlement_id"),
                    )
                    credit_slot = (
                        arm.value,
                        cal._text(credit["record_id"], "credit record_id"),
                    )
                    if settlement_slot in settlement_slots or credit_slot in credit_slots:
                        raise ValueError("campaign persisted content slot reused")
                    settlement_slots.add(settlement_slot)
                    credit_slots.add(credit_slot)
                    if post["owner_writeback_changed_persistence"] is not (
                        post["owner_preaction_persistence_sha256"]
                        != post["owner_postaction_persistence_sha256"]
                    ):
                        raise ValueError("campaign owner writeback flag drifted")
                    writeback_count += int(
                        post["owner_writeback_changed_persistence"]
                    )
                    prior_post_sha[(root_index, arm)] = cal._digest(
                        post["owner_postaction_persistence_sha256"],
                        "owner postaction persistence",
                    )
                    post_members.append(row)
                    post_by_arm[arm] = post
                post_receipt_row = reader.next(
                    source="campaign postaction receipt"
                )
                post_receipt = _require_trace_row(
                    post_receipt_row,
                    record_type="postaction_group_fsync",
                    expected_fields=_TRACE_FIELDS["postaction_group_fsync"],
                )
                _verify_group_receipt(
                    receipt=post_receipt_row,
                    members=post_members,
                    prefix="postaction",
                )
                post_core = {
                    key: value
                    for key, value in post_receipt.items()
                    if key
                    not in {
                        "row_id",
                        "physical_sequence_index",
                        "schema_version",
                        "record_type",
                        "postaction_receipt_id",
                    }
                }
                if (
                    post_receipt["postaction_receipt_id"] != sha256_json(post_core)
                    or post_receipt["preaction_barrier_id"]
                    != receipt["barrier_id"]
                    or post_receipt["root_sequence_index"] != root_index
                    or post_receipt["decision_index"] != decision_index
                ):
                    raise ValueError("campaign postaction receipt identity drifted")
                full_action = RelationshipAction(
                    post_by_arm[RelationshipProductHorizonCampaignArm.FULL][
                        "delivered_action_id"
                    ]
                )
                frozen_action = RelationshipAction(
                    post_by_arm[
                        RelationshipProductHorizonCampaignArm.FROZEN_THETA0
                    ]["delivered_action_id"]
                )
                strict_action = RelationshipAction(
                    post_by_arm[RelationshipProductHorizonCampaignArm.STRICT_NOOP][
                        "delivered_action_id"
                    ]
                )
                learnable_divergence_count += int(full_action is not frozen_action)
                steerable_divergence_count += int(frozen_action is not strict_action)
                frozen_nonnoop_count += int(
                    frozen_action is not RelationshipAction.NEUTRAL_NOOP
                )
            root_terminal_row = reader.next(source="campaign root terminal")
            root_terminal = _require_trace_row(
                root_terminal_row,
                record_type="root_terminal",
                expected_fields=_TRACE_FIELDS["root_terminal"],
            )
            state_rows = [states[(root_index, arm)] for arm in arm_order]
            if (
                root_terminal["root_sequence_index"] != root_index
                or root_terminal["subject_id"] != subject_id
                or root_terminal["evaluation_decision_count_per_arm"] != 40
                or root_terminal["terminal_state_row_ids"]
                != [item["row_id"] for item in state_rows]
                or root_terminal["terminal_state_prefix_raw_sha256"]
                != state_rows[-1]["prefix_raw_sha256_after"]
                or root_terminal["terminal_owner_persistence_sha256_by_arm"]
                != {
                    arm.value: states[(root_index, arm)][
                        "terminal_owner_persistence_sha256"
                    ]
                    for arm in _ARM_IDS
                }
                or root_terminal["evaluation_gate_update_count"] != 0
            ):
                raise ValueError("campaign root terminal join drifted")
            for arm in _ARM_IDS:
                if (
                    states[(root_index, arm)][
                        "terminal_owner_persistence_sha256"
                    ]
                    != prior_post_sha[(root_index, arm)]
                    or states[(root_index, arm)]["frozen_policy_id"]
                    != initialization_by_arm[arm]["frozen_policy_id"]
                    or states[(root_index, arm)]["checkpoint_content_sha256"]
                    != initialization_by_arm[arm]["checkpoint_content_sha256"]
                    or states[(root_index, arm)]["checkpoint_update_count"]
                    != initialization_by_arm[arm]["checkpoint_update_count"]
                    or states[(root_index, arm)]["evaluation_gate_update_count"]
                    != 0
                ):
                    raise ValueError("campaign final owner/policy handoff drifted")
        terminal_row = reader.next(source="campaign terminal")
        terminal_payload = _require_trace_row(
            terminal_row,
            record_type="terminal",
            expected_fields=_TRACE_FIELDS["terminal"],
        )
        reader.require_eof(source="campaign trace")
        if reader.row_count != 36066:
            raise ValueError("campaign trace row count drifted")
    finally:
        reader.close()
    if terminal_payload is None:
        raise RuntimeError("campaign trace produced no terminal payload")
    strict_nonnoop = sum(
        count
        for action, count in action_counts[
            RelationshipProductHorizonCampaignArm.STRICT_NOOP.value
        ].items()
        if action != RelationshipAction.NEUTRAL_NOOP.value
    )
    mechanism = {
        "complete_root_count": 112,
        "complete_root_arm_count": 336,
        "complete_evaluation_slot_count": len(records),
        "later_owner_handoff_count": later_handoff_count,
        "terminal_owner_roundtrip_count": state_count,
        "owner_writeback_change_count": writeback_count,
        "strict_noop_nonnoop_count": strict_nonnoop,
        "evaluation_credit_applied_count": 0,
        "evaluation_gate_update_count": 0,
        "learnable_actual_action_divergence_count": learnable_divergence_count,
        "steerable_actual_action_divergence_count": steerable_divergence_count,
        "frozen_theta0_physical_nonnoop_count": frozen_nonnoop_count,
        "full_learned_policy_differs_from_cold_root_count": full_diff_root_count,
        "selected_branch_unique_content_count": len(branch_ids),
        "settlement_slot_count": len(settlement_slots),
        "credit_slot_count": len(credit_slots),
        "actual_action_counts_by_arm": {
            arm: dict(sorted(counts.items()))
            for arm, counts in sorted(action_counts.items())
        },
    }
    _validate_complete_mechanism_contract(mechanism)
    report = _build_report(
        protocol=dependencies.protocol,
        plan_id=cal._text(plan["plan_id"], "plan_id"),
        lineage_id=dependencies.inputs.lineage_id,
        records=tuple(records),
        mechanism=mechanism,
    )
    if (
        terminal_payload["protocol_id"] != dependencies.protocol.protocol_id
        or terminal_payload["plan_id"] != plan["plan_id"]
        or terminal_payload["report_id"] != report["report_id"]
        or terminal_payload["completed_root_count"] != 112
        or terminal_payload["completed_root_arm_count"] != 336
        or terminal_payload["completed_evaluation_slot_count"] != 13440
        or terminal_payload["preaction_count"] != 13440
        or terminal_payload["preaction_group_fsync_count"] != 4480
        or terminal_payload["postaction_count"] != 13440
        or terminal_payload["postaction_group_fsync_count"] != 4480
        or terminal_payload["terminal_state_row_count"] != 336
        or terminal_payload["mechanism"] != mechanism
        or terminal_payload["status"] != report["status"]
        or terminal_payload["rehearsal_executed"] is not False
        or terminal_payload["model_invocation_count"] != 0
        or terminal_payload["cuda_execution_count"] != 0
    ):
        raise ValueError("campaign terminal report join drifted")
    return _PersistedAnalysis(
        report=report,
        trace_row_count=reader.row_count,
        trace_raw_bytes=reader.raw_bytes,
        trace_raw_sha256=reader.raw_sha256,
        terminal_state_row_count=state_count,
        terminal_state_raw_bytes=state_bytes,
        terminal_state_raw_sha256=state_sha,
    )


def _file_entries(
    manifest: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    entries: dict[str, Mapping[str, object]] = {}
    for raw_item in cal._list(manifest["files"], "campaign manifest files"):
        item = cal._mapping(raw_item, "campaign manifest file")
        cal._exact_keys(
            item,
            {"path", "raw_bytes", "raw_sha256"},
            "campaign manifest file",
        )
        relative = cal._text(item["path"], "campaign file path")
        if relative in entries:
            raise ValueError("campaign manifest file path reused")
        entries[relative] = item
    return entries


def _build_manifest(
    *,
    root: pathlib.Path,
    dependencies: _Dependencies,
    plan: Mapping[str, object],
    analysis: _PersistedAnalysis,
    implementation_git_commit: str,
    execution_elapsed_seconds_hex: str,
) -> Mapping[str, object]:
    files = []
    for relative in (
        "protocol.json",
        _PLAN_FILENAME,
        _TRACE_FILENAME,
        _TERMINAL_STATE_FILENAME,
        _REPORT_FILENAME,
    ):
        raw = cal._read_regular(root / relative)
        files.append(
            {
                "path": relative,
                "raw_bytes": len(raw),
                "raw_sha256": cal._sha256_bytes(raw),
            }
        )
    report = analysis.report
    claims = {
        "development_campaign_materialized": True,
        **cal._mapping(report["claims"], "campaign report claims"),
    }
    core = {
        "schema_version": DEVELOPMENT_CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        "protocol_id": dependencies.protocol.protocol_id,
        "protocol_raw_sha256": dependencies.protocol.raw_sha256,
        "implementation_git_commit": implementation_git_commit,
        "campaign_input_lineage_id": dependencies.inputs.lineage_id,
        "forced_protocol_id": dependencies.inputs.forced_protocol_id,
        "forced_artifact_id": dependencies.inputs.forced_artifact_id,
        "forced_manifest_raw_sha256": (
            dependencies.inputs.forced_manifest_raw_sha256
        ),
        "public_plan_sha256": dependencies.inputs.public_plan_sha256,
        "plan_id": plan["plan_id"],
        "report_id": report["report_id"],
        "trace_row_count": analysis.trace_row_count,
        "terminal_state_row_count": analysis.terminal_state_row_count,
        "execution_elapsed_seconds_hex": execution_elapsed_seconds_hex,
        "rehearsal_executed": False,
        "model_invocation_count": 0,
        "cuda_execution_count": 0,
        "files": files,
        "status": report["status"],
        "claims": claims,
        "claim_boundary": dependencies.protocol.payload["claim_boundary"],
    }
    return {"artifact_id": sha256_json(core), **core}


def materialize_relationship_product_horizon_development_campaign(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    dynamic_root: pathlib.Path,
    forced_common_batch_root: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    commit = cal._git_commit(implementation_git_commit)
    root = pathlib.Path(output_dir)
    if root.exists():
        raise FileExistsError(f"development campaign root is create-only: {root}")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
        scanner_root=pathlib.Path(scanner_root),
        dynamic_root=pathlib.Path(dynamic_root),
        forced_common_batch_root=pathlib.Path(forced_common_batch_root),
    )
    plan = _build_campaign_plan(dependencies=dependencies)
    root.mkdir(parents=True, exist_ok=False)
    _write_and_reopen_exact(root / "protocol.json", dependencies.protocol.raw_bytes)
    _write_and_reopen_exact(root / _PLAN_FILENAME, cal._canonical_bytes(plan))
    trace_sink = _CreateOnlyStreamingJsonlSink(root / _TRACE_FILENAME)
    try:
        state_sink = _CreateOnlyStreamingJsonlSink(root / _TERMINAL_STATE_FILENAME)
    except Exception:
        trace_sink.close()
        raise
    started = time.monotonic()
    try:
        replay = asyncio.run(
            _run_campaign(
                dependencies=dependencies,
                plan=plan,
                trace_sink=trace_sink,
                terminal_state_sink=state_sink,
            )
        )
    finally:
        trace_sink.close()
        state_sink.close()
    elapsed_hex = _float_hex(time.monotonic() - started)
    analysis = _analyze_persisted_evidence(
        dependencies=dependencies,
        plan=plan,
        trace_path=root / _TRACE_FILENAME,
        terminal_state_path=root / _TERMINAL_STATE_FILENAME,
    )
    if (
        replay.report != analysis.report
        or replay.trace_row_count != analysis.trace_row_count
        or replay.trace_raw_bytes != analysis.trace_raw_bytes
        or replay.trace_raw_sha256 != analysis.trace_raw_sha256
        or replay.terminal_state_row_count != analysis.terminal_state_row_count
        or replay.terminal_state_raw_bytes != analysis.terminal_state_raw_bytes
        or replay.terminal_state_raw_sha256
        != analysis.terminal_state_raw_sha256
    ):
        raise RuntimeError("campaign persisted evidence differs from live replay")
    _write_and_reopen_exact(
        root / _REPORT_FILENAME,
        cal._canonical_bytes(analysis.report),
    )
    manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        plan=plan,
        analysis=analysis,
        implementation_git_commit=commit,
        execution_elapsed_seconds_hex=elapsed_hex,
    )
    _write_and_reopen_exact(root / "manifest.json", cal._canonical_bytes(manifest))
    if cal._regular_file_inventory(root) != _OUTPUT_FILES:
        raise RuntimeError("development campaign final output inventory drifted")
    return manifest


def validate_relationship_product_horizon_development_campaign(
    *,
    source_v4_admission_root: pathlib.Path,
    reader_root: pathlib.Path,
    theta0_v2_root: pathlib.Path,
    scanner_root: pathlib.Path,
    dynamic_root: pathlib.Path,
    forced_common_batch_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    external_protocol = cal._digest(expected_protocol_id, "expected_protocol_id")
    external_artifact = cal._digest(expected_artifact_id, "expected_artifact_id")
    root = pathlib.Path(output_dir)
    if cal._regular_file_inventory(root) != _OUTPUT_FILES:
        raise ValueError("development campaign output inventory drifted")
    manifest_raw = cal._read_regular(root / "manifest.json")
    manifest = cal._parse_json_bytes(
        manifest_raw, source="development campaign manifest"
    )
    if manifest_raw != cal._canonical_bytes(manifest):
        raise ValueError("development campaign manifest must use canonical bytes")
    if manifest["protocol_id"] != external_protocol:
        raise ValueError("external development campaign protocol ID drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("external development campaign artifact ID drifted")
    if manifest["artifact_id"] != sha256_json(
        {key: value for key, value in manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("development campaign artifact identity drifted")
    dependencies = _load_dependencies(
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        reader_root=pathlib.Path(reader_root),
        theta0_v2_root=pathlib.Path(theta0_v2_root),
        scanner_root=pathlib.Path(scanner_root),
        dynamic_root=pathlib.Path(dynamic_root),
        forced_common_batch_root=pathlib.Path(forced_common_batch_root),
    )
    if dependencies.protocol.protocol_id != external_protocol:
        raise ValueError("packaged development campaign protocol ID drifted")
    if cal._read_regular(root / "protocol.json") != dependencies.protocol.raw_bytes:
        raise ValueError("persisted development campaign protocol bytes drifted")
    plan = _build_campaign_plan(dependencies=dependencies)
    if cal._read_regular(root / _PLAN_FILENAME) != cal._canonical_bytes(plan):
        raise ValueError("persisted development campaign plan bytes drifted")
    files = _file_entries(manifest)
    if frozenset(files) != _OUTPUT_FILES - {"manifest.json"}:
        raise ValueError("development campaign manifest file inventory drifted")
    for relative, entry in files.items():
        raw = cal._read_regular(root / relative)
        if (
            len(raw) != cal._integer(entry["raw_bytes"], f"{relative}.raw_bytes")
            or cal._sha256_bytes(raw)
            != cal._digest(entry["raw_sha256"], f"{relative}.raw_sha256")
        ):
            raise ValueError(f"development campaign file bytes drifted: {relative}")
    analysis = _analyze_persisted_evidence(
        dependencies=dependencies,
        plan=plan,
        trace_path=root / _TRACE_FILENAME,
        terminal_state_path=root / _TERMINAL_STATE_FILENAME,
    )
    if cal._read_regular(root / _REPORT_FILENAME) != cal._canonical_bytes(
        analysis.report
    ):
        raise ValueError("persisted development campaign report bytes drifted")
    elapsed_hex = cal._text(
        manifest["execution_elapsed_seconds_hex"],
        "execution_elapsed_seconds_hex",
    )
    elapsed = float.fromhex(elapsed_hex)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("development campaign elapsed time is invalid")
    expected_manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        plan=plan,
        analysis=analysis,
        implementation_git_commit=cal._git_commit(
            manifest["implementation_git_commit"]
        ),
        execution_elapsed_seconds_hex=elapsed_hex,
    )
    if manifest != expected_manifest or manifest_raw != cal._canonical_bytes(
        expected_manifest
    ):
        raise ValueError("development campaign manifest content drifted")
    if manifest["artifact_id"] != external_artifact:
        raise ValueError("development campaign artifact identity drifted")
    return manifest


__all__ = [
    "DEVELOPMENT_CAMPAIGN_MANIFEST_SCHEMA_VERSION",
    "DEVELOPMENT_CAMPAIGN_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_CAMPAIGN_PROTOCOL_SCHEMA_VERSION",
    "DEVELOPMENT_CAMPAIGN_REPORT_SCHEMA_VERSION",
    "DEVELOPMENT_CAMPAIGN_TERMINAL_STATE_SCHEMA_VERSION",
    "DEVELOPMENT_CAMPAIGN_TRACE_SCHEMA_VERSION",
    "RelationshipProductHorizonDevelopmentCampaignProtocol",
    "load_relationship_product_horizon_development_campaign_protocol",
    "materialize_relationship_product_horizon_development_campaign",
    "relationship_product_horizon_development_campaign_protocol_path",
    "validate_relationship_product_horizon_development_campaign",
]
