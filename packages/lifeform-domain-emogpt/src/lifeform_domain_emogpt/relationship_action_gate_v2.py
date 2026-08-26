"""Versioned, assignment-aware relationship-action gate.

The v1 gate is immutable evidence lineage.  This module introduces a separate
operator for new Product Horizon development protocols.  It deliberately has
no free bias, excludes the constant ``typed_source_support`` feature, and
learns from forced action assignments with a frozen centred-design objective
instead of treating forced actions as samples from the gate's own policy.

Only replay-checked
:class:`~volvence_zero.credit.RelationshipActionCommonBaselineCredit` values
are accepted.  The fixed-balanced objective is a development feature score,
not a randomized or causal-effect claim.  Evaluation, judge, hidden-condition,
and raw-text inputs are outside this API.
"""

from __future__ import annotations

import math
from dataclasses import InitVar, dataclass
from enum import Enum

from volvence_zero.credit import RelationshipActionCommonBaselineCredit
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    preference_action_forecast_from_payload,
    preference_action_forecast_to_payload,
)
from volvence_zero.temporal_types import TemporalActionAdvisoryProposal

from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RelationshipAction,
)
from lifeform_domain_emogpt import relationship_action_gate as legacy


RELATIONSHIP_ACTION_GATE_V2_ARTIFACT_SCHEMA_VERSION = "relationship-action-gate-artifact.v2"
RELATIONSHIP_ACTION_GATE_V2_CHECKPOINT_SCHEMA_VERSION = "relationship-action-gate-checkpoint.v2"
RELATIONSHIP_ACTION_GATE_V2_DECISION_SCHEMA_VERSION = "relationship-action-gate-decision.v2"
RELATIONSHIP_ACTION_GATE_V2_FROZEN_DECISION_SCHEMA_VERSION = "relationship-action-gate-frozen-decision.v2"
RELATIONSHIP_ACTION_GATE_V2_FROZEN_POLICY_SCHEMA_VERSION = "relationship-action-gate-frozen-policy.v2"
RELATIONSHIP_ACTION_GATE_V2_FORCED_EXPOSURE_SCHEMA_VERSION = "relationship-action-gate-forced-exposure.v2"
RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_RECEIPT_SCHEMA_VERSION = "relationship-action-gate-assignment-receipt.v2"
RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_SCHEDULE_ENTRY_SCHEMA_VERSION = (
    "relationship-action-gate-assignment-schedule-entry.v2"
)
RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_SCHEDULE_SCHEMA_VERSION = "relationship-action-gate-assignment-schedule.v2"
RELATIONSHIP_ACTION_GATE_V2_CREDIT_BATCH_SCHEMA_VERSION = "relationship-action-gate-credit-batch.v2"
RELATIONSHIP_ACTION_GATE_V2_BATCH_PLAN_SCHEMA_VERSION = "relationship-action-gate-batch-plan.v2"
RELATIONSHIP_ACTION_GATE_V2_BATCH_RECEIPT_SCHEMA_VERSION = "relationship-action-gate-batch-receipt.v2"
RELATIONSHIP_ACTION_GATE_V2_OPERATOR_ID = "bias-free-centred-assignment-logistic-gate.v2"
RELATIONSHIP_ACTION_GATE_V2_OBJECTIVE_ID = "common-noop-credit-times-half-centred-assignment-feature-moment.v1"
RELATIONSHIP_ACTION_GATE_V2_THRESHOLD_RULE = "steer_probability_strictly_greater_than_0.5"
RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER = (
    "forecast_confidence_centered",
    "recommended_positive_mass_centered",
    "recommended_positive_margin_over_noop",
    "recommended_entropy_certainty_centered",
)
_FEATURE_COUNT = len(RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER)
_ARTIFACT_PREFIX = "relationship-action-gate-v2-artifact-sha256:"
_BATCH_PREFIX = "relationship-action-gate-v2-credit-batch-sha256:"
_PLAN_PREFIX = "relationship-action-gate-v2-batch-plan-sha256:"
_RECEIPT_PREFIX = "relationship-action-gate-v2-batch-receipt-sha256:"
_POLICY_PREFIX = "relationship-action-gate-v2-frozen-policy-sha256:"
_EXPOSURE_PREFIX = "relationship-action-gate-v2-forced-exposure-sha256:"
_ASSIGNMENT_PREFIX = "relationship-action-gate-v2-assignment-sha256:"
_ASSIGNMENT_SCHEDULE_ENTRY_PREFIX = "relationship-action-gate-v2-assignment-schedule-entry-sha256:"
_ASSIGNMENT_SCHEDULE_PREFIX = "relationship-action-gate-v2-assignment-schedule-sha256:"


class RelationshipActionGateV2ArtifactKind(str, Enum):
    BOOTSTRAP_SEED = "bootstrap_seed"
    LEARNED_THETA0 = "learned_theta0"


class RelationshipActionGateV2AssignmentDesign(str, Enum):
    """Design identity carried by every forced exposure.

    ``FIXED_BALANCED_HALF`` is a finite-population development objective, not
    a randomized propensity claim.  A future randomized design requires a new
    schema with a separately owned randomization mechanism and propensity
    proof; adding an enum value here would not provide either.
    """

    FIXED_BALANCED_HALF = "fixed-balanced-half.v1"


class RelationshipActionGateV2AssignmentRole(str, Enum):
    CANDIDATE = "candidate"
    NEUTRAL_NOOP = "neutral_noop"


@dataclass(frozen=True)
class RelationshipActionGateV2AssignmentScheduleEntry:
    """One fixed-balanced member; an external owner proves pre-outcome timing."""

    decision_id: str
    sequence_index: int
    assignment_role: RelationshipActionGateV2AssignmentRole
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_SCHEDULE_ENTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        legacy._require_text(self.decision_id, "decision_id")
        if type(self.sequence_index) is not int or self.sequence_index < 0:
            raise ValueError("assignment sequence_index must be non-negative")
        if type(self.assignment_role) is not RelationshipActionGateV2AssignmentRole:
            raise TypeError("assignment_role must be RelationshipActionGateV2AssignmentRole")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_SCHEDULE_ENTRY_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 schedule entry schema mismatch")

    @property
    def entry_id(self) -> str:
        return f"{_ASSIGNMENT_SCHEDULE_ENTRY_PREFIX}{legacy._canonical_sha256(self._core_payload())}"

    def to_payload(self) -> dict[str, object]:
        return {"entry_id": self.entry_id, **self._core_payload()}

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> "RelationshipActionGateV2AssignmentScheduleEntry":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "entry_id",
                "schema_version",
                "decision_id",
                "sequence_index",
                "assignment_role",
            },
            source="relationship action gate v2 assignment schedule entry",
        )
        entry = cls(
            decision_id=legacy._payload_text(raw, "decision_id"),
            sequence_index=legacy._payload_int(raw, "sequence_index"),
            assignment_role=RelationshipActionGateV2AssignmentRole(legacy._payload_text(raw, "assignment_role")),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )
        if legacy._payload_text(raw, "entry_id") != entry.entry_id:
            raise ValueError("relationship action gate v2 schedule entry_id mismatch")
        return entry

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "sequence_index": self.sequence_index,
            "assignment_role": self.assignment_role.value,
        }


@dataclass(frozen=True)
class RelationshipActionGateV2AssignmentScheduleArtifact:
    """Complete fixed-balanced schedule; all member content enters its ID.

    Content identity proves membership, not when the artifact was created.
    """

    source_artifact_id: str
    schedule_scope_id: str
    entries: tuple[RelationshipActionGateV2AssignmentScheduleEntry, ...]
    assignment_design: RelationshipActionGateV2AssignmentDesign = (
        RelationshipActionGateV2AssignmentDesign.FIXED_BALANCED_HALF
    )
    centering_fraction_hex: str = (0.5).hex()
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        legacy._require_content_addressed_id(
            self.source_artifact_id,
            "source_artifact_id",
        )
        legacy._require_text(self.schedule_scope_id, "schedule_scope_id")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("assignment schedule entries must be a non-empty exact tuple")
        if any(type(item) is not RelationshipActionGateV2AssignmentScheduleEntry for item in self.entries):
            raise TypeError("assignment schedule entries have an invalid item type")
        if type(self.assignment_design) is not RelationshipActionGateV2AssignmentDesign:
            raise TypeError("assignment_design must be RelationshipActionGateV2AssignmentDesign")
        if self.assignment_design is not RelationshipActionGateV2AssignmentDesign.FIXED_BALANCED_HALF:
            raise ValueError("gate v2 supports only fixed-balanced development schedules")
        if (
            legacy._finite_float_from_hex(
                self.centering_fraction_hex,
                "centering_fraction_hex",
            )
            != 0.5
        ):
            raise ValueError("relationship action gate v2 requires exact half centering")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_SCHEDULE_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 schedule schema mismatch")
        if tuple(item.sequence_index for item in self.entries) != tuple(range(len(self.entries))):
            raise ValueError("assignment schedule sequence must be contiguous")
        decision_ids = tuple(item.decision_id for item in self.entries)
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("assignment schedule decision ids must be unique")
        candidate_count = sum(
            item.assignment_role is RelationshipActionGateV2AssignmentRole.CANDIDATE for item in self.entries
        )
        if candidate_count * 2 != len(self.entries):
            raise ValueError("assignment schedule must be exactly candidate/noop balanced")

    @property
    def artifact_id(self) -> str:
        return f"{_ASSIGNMENT_SCHEDULE_PREFIX}{legacy._canonical_sha256(self._core_payload())}"

    def entry_for_decision(
        self,
        decision_id: str,
    ) -> RelationshipActionGateV2AssignmentScheduleEntry:
        legacy._require_text(decision_id, "decision_id")
        matches = tuple(item for item in self.entries if item.decision_id == decision_id)
        if len(matches) != 1:
            raise ValueError("assignment schedule requires one exact decision member")
        return matches[0]

    def to_payload(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self._core_payload()}

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> "RelationshipActionGateV2AssignmentScheduleArtifact":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "artifact_id",
                "schema_version",
                "source_artifact_id",
                "schedule_scope_id",
                "assignment_design",
                "centering_fraction_hex",
                "entries",
            },
            source="relationship action gate v2 assignment schedule",
        )
        raw_entries = raw["entries"]
        if type(raw_entries) is not list or not raw_entries:
            raise ValueError("assignment schedule entries must be a non-empty array")
        schedule = cls(
            source_artifact_id=legacy._payload_text(raw, "source_artifact_id"),
            schedule_scope_id=legacy._payload_text(raw, "schedule_scope_id"),
            entries=tuple(RelationshipActionGateV2AssignmentScheduleEntry.from_payload(item) for item in raw_entries),
            assignment_design=RelationshipActionGateV2AssignmentDesign(legacy._payload_text(raw, "assignment_design")),
            centering_fraction_hex=legacy._payload_text(
                raw,
                "centering_fraction_hex",
            ),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )
        if legacy._payload_text(raw, "artifact_id") != schedule.artifact_id:
            raise ValueError("relationship action gate v2 schedule artifact_id mismatch")
        return schedule

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_id": self.source_artifact_id,
            "schedule_scope_id": self.schedule_scope_id,
            "assignment_design": self.assignment_design.value,
            "centering_fraction_hex": self.centering_fraction_hex,
            "entries": [item.to_payload() for item in self.entries],
        }


@dataclass(frozen=True)
class RelationshipActionGateV2AssignmentReceipt:
    """Exact membership in one complete typed assignment schedule."""

    schedule_artifact: RelationshipActionGateV2AssignmentScheduleArtifact
    schedule_entry: RelationshipActionGateV2AssignmentScheduleEntry
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schedule_artifact) is not (RelationshipActionGateV2AssignmentScheduleArtifact):
            raise TypeError("schedule_artifact must be RelationshipActionGateV2AssignmentScheduleArtifact")
        if type(self.schedule_entry) is not RelationshipActionGateV2AssignmentScheduleEntry:
            raise TypeError("schedule_entry must be RelationshipActionGateV2AssignmentScheduleEntry")
        if self.schedule_artifact.entry_for_decision(self.schedule_entry.decision_id) != self.schedule_entry:
            raise ValueError("assignment receipt is not an exact schedule member")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_RECEIPT_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 assignment schema mismatch")

    @property
    def schedule_artifact_id(self) -> str:
        return self.schedule_artifact.artifact_id

    @property
    def schedule_entry_id(self) -> str:
        return self.schedule_entry.entry_id

    @property
    def decision_id(self) -> str:
        return self.schedule_entry.decision_id

    @property
    def sequence_index(self) -> int:
        return self.schedule_entry.sequence_index

    @property
    def assignment_design(self) -> RelationshipActionGateV2AssignmentDesign:
        return self.schedule_artifact.assignment_design

    @property
    def assignment_role(self) -> RelationshipActionGateV2AssignmentRole:
        return self.schedule_entry.assignment_role

    @property
    def centering_fraction_hex(self) -> str:
        return self.schedule_artifact.centering_fraction_hex

    @property
    def assignment_id(self) -> str:
        return f"{_ASSIGNMENT_PREFIX}{legacy._canonical_sha256(self._core_payload())}"

    def to_payload(self) -> dict[str, object]:
        return {"assignment_id": self.assignment_id, **self._core_payload()}

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> "RelationshipActionGateV2AssignmentReceipt":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "assignment_id",
                "schema_version",
                "schedule_artifact",
                "schedule_entry",
            },
            source="relationship action gate v2 assignment receipt",
        )
        receipt = cls(
            schedule_artifact=(
                RelationshipActionGateV2AssignmentScheduleArtifact.from_payload(raw["schedule_artifact"])
            ),
            schedule_entry=RelationshipActionGateV2AssignmentScheduleEntry.from_payload(raw["schedule_entry"]),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )
        if legacy._payload_text(raw, "assignment_id") != receipt.assignment_id:
            raise ValueError("relationship action gate v2 assignment_id mismatch")
        return receipt

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schedule_artifact": self.schedule_artifact.to_payload(),
            "schedule_entry": self.schedule_entry.to_payload(),
        }


@dataclass(frozen=True)
class RelationshipActionGateV2Artifact:
    """Content-addressed parameters and both frozen learning time scales."""

    artifact_id: str
    artifact_kind: RelationshipActionGateV2ArtifactKind
    weights_hex: tuple[str, ...]
    bootstrap_learning_rate_hex: str
    online_learning_rate_hex: str
    max_abs_parameter_hex: str
    bootstrap_source_artifact_id: str
    source_parent_artifact_id: str | None
    source_credit_batch_id: str | None
    source_apply_receipt_id: str | None
    source_checkpoint_content_sha256: str | None
    _source_parent: InitVar[object | None] = None
    _source_batch: InitVar[object | None] = None
    _source_apply_receipt: InitVar[object | None] = None
    operator_id: str = RELATIONSHIP_ACTION_GATE_V2_OPERATOR_ID
    objective_id: str = RELATIONSHIP_ACTION_GATE_V2_OBJECTIVE_ID
    feature_order: tuple[str, ...] = RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER
    threshold_rule: str = RELATIONSHIP_ACTION_GATE_V2_THRESHOLD_RULE
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_ARTIFACT_SCHEMA_VERSION

    def __post_init__(
        self,
        _source_parent: object | None,
        _source_batch: object | None,
        _source_apply_receipt: object | None,
    ) -> None:
        legacy._require_text(self.artifact_id, "artifact_id")
        if type(self.artifact_kind) is not RelationshipActionGateV2ArtifactKind:
            raise TypeError("artifact_kind must be RelationshipActionGateV2ArtifactKind")
        if self.operator_id != RELATIONSHIP_ACTION_GATE_V2_OPERATOR_ID:
            raise ValueError("relationship action gate v2 operator_id mismatch")
        if self.objective_id != RELATIONSHIP_ACTION_GATE_V2_OBJECTIVE_ID:
            raise ValueError("relationship action gate v2 objective_id mismatch")
        if self.feature_order != RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER:
            raise ValueError("relationship action gate v2 feature_order mismatch")
        if self.threshold_rule != RELATIONSHIP_ACTION_GATE_V2_THRESHOLD_RULE:
            raise ValueError("relationship action gate v2 threshold_rule mismatch")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 artifact schema mismatch")
        if type(self.weights_hex) is not tuple or len(self.weights_hex) != _FEATURE_COUNT:
            raise ValueError(f"weights_hex must contain {_FEATURE_COUNT} values")
        if any(type(value) is not str for value in self.weights_hex):
            raise TypeError("weights_hex must contain exact strings")
        if type(self.feature_order) is not tuple or any(type(value) is not str for value in self.feature_order):
            raise TypeError("feature_order must be an exact tuple of strings")
        weights = tuple(
            legacy._finite_float_from_hex(value, f"weights_hex[{index}]")
            for index, value in enumerate(self.weights_hex)
        )
        bootstrap_rate = legacy._finite_float_from_hex(
            self.bootstrap_learning_rate_hex,
            "bootstrap_learning_rate_hex",
        )
        online_rate = legacy._finite_float_from_hex(
            self.online_learning_rate_hex,
            "online_learning_rate_hex",
        )
        cap = legacy._finite_float_from_hex(
            self.max_abs_parameter_hex,
            "max_abs_parameter_hex",
        )
        if not 0.0 < bootstrap_rate <= 0.5:
            raise ValueError("bootstrap learning rate must be in (0, 0.5]")
        if not 0.0 < online_rate <= 0.5:
            raise ValueError("online learning rate must be in (0, 0.5]")
        if not 0.5 <= cap <= 8.0:
            raise ValueError("max_abs_parameter must be in [0.5, 8]")
        if any(abs(value) > cap for value in weights):
            raise ValueError("relationship action gate v2 weights exceed cap")
        legacy._require_content_addressed_id(
            self.bootstrap_source_artifact_id,
            "bootstrap_source_artifact_id",
        )
        if self.artifact_kind is RelationshipActionGateV2ArtifactKind.BOOTSTRAP_SEED:
            if any(value != 0.0 for value in weights):
                raise ValueError("bootstrap seed weights must be exactly zero")
            if any(
                value is not None
                for value in (
                    self.source_parent_artifact_id,
                    self.source_credit_batch_id,
                    self.source_apply_receipt_id,
                    self.source_checkpoint_content_sha256,
                )
            ):
                raise ValueError("bootstrap seed cannot cite a learned transition")
            if any(
                component is not None
                for component in (
                    _source_parent,
                    _source_batch,
                    _source_apply_receipt,
                )
            ):
                raise ValueError("bootstrap seed loading cannot cite a learned transition")
        else:
            if not any(value != 0.0 for value in weights):
                raise ValueError("learned theta0 weights must contain a nonzero value")
            for field_name, value in (
                ("source_parent_artifact_id", self.source_parent_artifact_id),
                ("source_credit_batch_id", self.source_credit_batch_id),
                ("source_apply_receipt_id", self.source_apply_receipt_id),
            ):
                if value is None:
                    raise ValueError(f"learned theta0 requires {field_name}")
                legacy._require_content_addressed_id(value, field_name)
            if self.source_checkpoint_content_sha256 is None:
                raise ValueError("learned theta0 requires a source checkpoint")
            legacy._require_sha256(
                self.source_checkpoint_content_sha256,
                "source_checkpoint_content_sha256",
            )
            if (
                type(_source_parent) is not RelationshipActionGateV2Artifact
                or type(_source_batch) is not RelationshipActionGateV2CreditBatch
                or type(_source_apply_receipt) is not RelationshipActionGateV2BatchReceipt
            ):
                raise ValueError("learned theta0 requires full source transition components")
            self.validate_source_transition(
                parent_artifact=_source_parent,
                source_batch=_source_batch,
                apply_receipt=_source_apply_receipt,
            )
        expected = f"{_ARTIFACT_PREFIX}{legacy._canonical_sha256(self._core_payload())}"
        if self.artifact_id != expected:
            raise ValueError("relationship action gate v2 artifact_id mismatch")

    @classmethod
    def create_bootstrap_seed(
        cls,
        *,
        bootstrap_learning_rate: float,
        online_learning_rate: float,
        max_abs_parameter: float,
        bootstrap_source_artifact_id: str,
    ) -> "RelationshipActionGateV2Artifact":
        return cls._create(
            artifact_kind=RelationshipActionGateV2ArtifactKind.BOOTSTRAP_SEED,
            weights=(0.0,) * _FEATURE_COUNT,
            bootstrap_learning_rate=bootstrap_learning_rate,
            online_learning_rate=online_learning_rate,
            max_abs_parameter=max_abs_parameter,
            bootstrap_source_artifact_id=bootstrap_source_artifact_id,
            source_parent_artifact_id=None,
            source_credit_batch_id=None,
            source_apply_receipt_id=None,
            source_checkpoint_content_sha256=None,
        )

    @classmethod
    def create_learned_theta0(
        cls,
        *,
        parent_artifact: "RelationshipActionGateV2Artifact",
        source_batch: "RelationshipActionGateV2CreditBatch",
        apply_receipt: "RelationshipActionGateV2BatchReceipt",
    ) -> "RelationshipActionGateV2Artifact":
        if type(parent_artifact) is not RelationshipActionGateV2Artifact:
            raise TypeError("parent_artifact must be RelationshipActionGateV2Artifact")
        if parent_artifact.artifact_kind is not RelationshipActionGateV2ArtifactKind.BOOTSTRAP_SEED:
            raise ValueError("learned theta0 parent must be a bootstrap seed")
        if type(source_batch) is not RelationshipActionGateV2CreditBatch:
            raise TypeError("source_batch must be RelationshipActionGateV2CreditBatch")
        if type(apply_receipt) is not RelationshipActionGateV2BatchReceipt:
            raise TypeError("apply_receipt must be RelationshipActionGateV2BatchReceipt")
        if apply_receipt.disposition is not legacy.RelationshipActionGateBatchDisposition.APPLY:
            raise ValueError("learned theta0 requires an exact APPLY receipt")
        replayed = RelationshipActionGateV2.from_applied_credit_batch(
            parent_artifact,
            batch=source_batch,
            receipt=apply_receipt,
        )
        source_checkpoint = replayed.export_checkpoint()
        if source_checkpoint.update_count < 1:
            raise ValueError("learned theta0 source checkpoint has no processed credit")
        if source_checkpoint.informative_update_count < 1:
            raise ValueError("learned theta0 source checkpoint has no information")
        if not any(value != 0.0 for value in source_checkpoint.weights):
            raise ValueError("learned theta0 source checkpoint has zero net update")
        artifact = cls._create(
            artifact_kind=RelationshipActionGateV2ArtifactKind.LEARNED_THETA0,
            weights=source_checkpoint.weights,
            bootstrap_learning_rate=parent_artifact.bootstrap_learning_rate,
            online_learning_rate=parent_artifact.online_learning_rate,
            max_abs_parameter=parent_artifact.max_abs_parameter,
            bootstrap_source_artifact_id=(parent_artifact.bootstrap_source_artifact_id),
            source_parent_artifact_id=parent_artifact.artifact_id,
            source_credit_batch_id=source_batch.batch_id,
            source_apply_receipt_id=apply_receipt.receipt_id,
            source_checkpoint_content_sha256=source_checkpoint.content_sha256,
            source_parent=parent_artifact,
            source_batch=source_batch,
            source_apply_receipt=apply_receipt,
        )
        return artifact

    @classmethod
    def _create(
        cls,
        *,
        artifact_kind: RelationshipActionGateV2ArtifactKind,
        weights: tuple[float, ...],
        bootstrap_learning_rate: float,
        online_learning_rate: float,
        max_abs_parameter: float,
        bootstrap_source_artifact_id: str,
        source_parent_artifact_id: str | None,
        source_credit_batch_id: str | None,
        source_apply_receipt_id: str | None,
        source_checkpoint_content_sha256: str | None,
        source_parent: "RelationshipActionGateV2Artifact | None" = None,
        source_batch: "RelationshipActionGateV2CreditBatch | None" = None,
        source_apply_receipt: "RelationshipActionGateV2BatchReceipt | None" = None,
    ) -> "RelationshipActionGateV2Artifact":
        if len(weights) != _FEATURE_COUNT or any(
            isinstance(value, bool) or not math.isfinite(value) for value in weights
        ):
            raise ValueError("relationship action gate v2 weights are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in (
                bootstrap_learning_rate,
                online_learning_rate,
                max_abs_parameter,
            )
        ):
            raise ValueError("relationship action gate v2 rates/cap must be numbers")
        normalized_weights = tuple(float(value) for value in weights)
        bootstrap_rate = float(bootstrap_learning_rate)
        online_rate = float(online_learning_rate)
        cap = float(max_abs_parameter)
        core = {
            "schema_version": RELATIONSHIP_ACTION_GATE_V2_ARTIFACT_SCHEMA_VERSION,
            "operator_id": RELATIONSHIP_ACTION_GATE_V2_OPERATOR_ID,
            "objective_id": RELATIONSHIP_ACTION_GATE_V2_OBJECTIVE_ID,
            "feature_order": list(RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER),
            "threshold_rule": RELATIONSHIP_ACTION_GATE_V2_THRESHOLD_RULE,
            "artifact_kind": artifact_kind.value,
            "weights_hex": [value.hex() for value in normalized_weights],
            "bootstrap_learning_rate_hex": bootstrap_rate.hex(),
            "online_learning_rate_hex": online_rate.hex(),
            "max_abs_parameter_hex": cap.hex(),
            "bootstrap_source_artifact_id": bootstrap_source_artifact_id,
            "source_parent_artifact_id": source_parent_artifact_id,
            "source_credit_batch_id": source_credit_batch_id,
            "source_apply_receipt_id": source_apply_receipt_id,
            "source_checkpoint_content_sha256": source_checkpoint_content_sha256,
        }
        return cls(
            artifact_id=f"{_ARTIFACT_PREFIX}{legacy._canonical_sha256(core)}",
            **{
                "artifact_kind": artifact_kind,
                "weights_hex": tuple(core["weights_hex"]),
                "bootstrap_learning_rate_hex": core["bootstrap_learning_rate_hex"],
                "online_learning_rate_hex": core["online_learning_rate_hex"],
                "max_abs_parameter_hex": core["max_abs_parameter_hex"],
                "bootstrap_source_artifact_id": bootstrap_source_artifact_id,
                "source_parent_artifact_id": source_parent_artifact_id,
                "source_credit_batch_id": source_credit_batch_id,
                "source_apply_receipt_id": source_apply_receipt_id,
                "source_checkpoint_content_sha256": source_checkpoint_content_sha256,
                "_source_parent": source_parent,
                "_source_batch": source_batch,
                "_source_apply_receipt": source_apply_receipt,
            },
        )

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(float.fromhex(value) for value in self.weights_hex)

    @property
    def bootstrap_learning_rate(self) -> float:
        return float.fromhex(self.bootstrap_learning_rate_hex)

    @property
    def online_learning_rate(self) -> float:
        return float.fromhex(self.online_learning_rate_hex)

    @property
    def active_learning_rate(self) -> float:
        if self.artifact_kind is RelationshipActionGateV2ArtifactKind.BOOTSTRAP_SEED:
            return self.bootstrap_learning_rate
        return self.online_learning_rate

    @property
    def max_abs_parameter(self) -> float:
        return float.fromhex(self.max_abs_parameter_hex)

    def validate_source_checkpoint(
        self,
        checkpoint: "RelationshipActionGateV2Checkpoint",
    ) -> None:
        if type(checkpoint) is not RelationshipActionGateV2Checkpoint:
            raise TypeError("checkpoint must be RelationshipActionGateV2Checkpoint")
        if self.artifact_kind is not RelationshipActionGateV2ArtifactKind.LEARNED_THETA0:
            raise ValueError("bootstrap seed has no learned source checkpoint")
        if (
            checkpoint.content_sha256 != self.source_checkpoint_content_sha256
            or checkpoint.weights != self.weights
            or checkpoint.update_count < 1
            or checkpoint.informative_update_count < 1
        ):
            raise ValueError("relationship action gate v2 source checkpoint drifted")

    def validate_source_transition(
        self,
        *,
        parent_artifact: "RelationshipActionGateV2Artifact",
        source_batch: "RelationshipActionGateV2CreditBatch",
        apply_receipt: "RelationshipActionGateV2BatchReceipt",
    ) -> None:
        if type(parent_artifact) is not RelationshipActionGateV2Artifact:
            raise TypeError("parent_artifact must be RelationshipActionGateV2Artifact")
        if type(source_batch) is not RelationshipActionGateV2CreditBatch:
            raise TypeError("source_batch must be RelationshipActionGateV2CreditBatch")
        if type(apply_receipt) is not RelationshipActionGateV2BatchReceipt:
            raise TypeError("apply_receipt must be RelationshipActionGateV2BatchReceipt")
        if self.artifact_kind is not RelationshipActionGateV2ArtifactKind.LEARNED_THETA0:
            raise ValueError("bootstrap seed has no learned source transition")
        if parent_artifact.artifact_kind is not RelationshipActionGateV2ArtifactKind.BOOTSTRAP_SEED:
            raise ValueError("learned theta0 source parent must be a bootstrap seed")
        if (
            parent_artifact.artifact_id != self.source_parent_artifact_id
            or parent_artifact.bootstrap_source_artifact_id != self.bootstrap_source_artifact_id
            or source_batch.batch_id != self.source_credit_batch_id
            or apply_receipt.receipt_id != self.source_apply_receipt_id
        ):
            raise ValueError("relationship action gate v2 source transition drifted")
        replayed = RelationshipActionGateV2.from_applied_credit_batch(
            parent_artifact,
            batch=source_batch,
            receipt=apply_receipt,
        )
        self.validate_source_checkpoint(replayed.export_checkpoint())

    def to_payload(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self._core_payload()}

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        parent_artifact: "RelationshipActionGateV2Artifact | None" = None,
        source_batch: "RelationshipActionGateV2CreditBatch | None" = None,
        apply_receipt: "RelationshipActionGateV2BatchReceipt | None" = None,
    ) -> "RelationshipActionGateV2Artifact":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "artifact_id",
                "schema_version",
                "operator_id",
                "objective_id",
                "feature_order",
                "threshold_rule",
                "artifact_kind",
                "weights_hex",
                "bootstrap_learning_rate_hex",
                "online_learning_rate_hex",
                "max_abs_parameter_hex",
                "bootstrap_source_artifact_id",
                "source_parent_artifact_id",
                "source_credit_batch_id",
                "source_apply_receipt_id",
                "source_checkpoint_content_sha256",
            },
            source="relationship action gate v2 artifact",
        )
        weights = raw["weights_hex"]
        features = raw["feature_order"]
        if not isinstance(weights, list) or any(not isinstance(v, str) for v in weights):
            raise ValueError("weights_hex must be an array of strings")
        if not isinstance(features, list) or any(not isinstance(v, str) for v in features):
            raise ValueError("feature_order must be an array of strings")
        optional_lineage = {
            key: raw[key]
            for key in (
                "source_parent_artifact_id",
                "source_credit_batch_id",
                "source_apply_receipt_id",
                "source_checkpoint_content_sha256",
            )
        }
        if any(value is not None and type(value) is not str for value in optional_lineage.values()):
            raise ValueError("source transition identity must be a string or null")
        return cls(
            artifact_id=legacy._payload_text(raw, "artifact_id"),
            artifact_kind=RelationshipActionGateV2ArtifactKind(legacy._payload_text(raw, "artifact_kind")),
            weights_hex=tuple(weights),
            bootstrap_learning_rate_hex=legacy._payload_text(raw, "bootstrap_learning_rate_hex"),
            online_learning_rate_hex=legacy._payload_text(raw, "online_learning_rate_hex"),
            max_abs_parameter_hex=legacy._payload_text(raw, "max_abs_parameter_hex"),
            bootstrap_source_artifact_id=legacy._payload_text(raw, "bootstrap_source_artifact_id"),
            source_parent_artifact_id=optional_lineage["source_parent_artifact_id"],
            source_credit_batch_id=optional_lineage["source_credit_batch_id"],
            source_apply_receipt_id=optional_lineage["source_apply_receipt_id"],
            source_checkpoint_content_sha256=optional_lineage["source_checkpoint_content_sha256"],
            _source_parent=parent_artifact,
            _source_batch=source_batch,
            _source_apply_receipt=apply_receipt,
            operator_id=legacy._payload_text(raw, "operator_id"),
            objective_id=legacy._payload_text(raw, "objective_id"),
            feature_order=tuple(features),
            threshold_rule=legacy._payload_text(raw, "threshold_rule"),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operator_id": self.operator_id,
            "objective_id": self.objective_id,
            "feature_order": list(self.feature_order),
            "threshold_rule": self.threshold_rule,
            "artifact_kind": self.artifact_kind.value,
            "weights_hex": list(self.weights_hex),
            "bootstrap_learning_rate_hex": self.bootstrap_learning_rate_hex,
            "online_learning_rate_hex": self.online_learning_rate_hex,
            "max_abs_parameter_hex": self.max_abs_parameter_hex,
            "bootstrap_source_artifact_id": self.bootstrap_source_artifact_id,
            "source_parent_artifact_id": self.source_parent_artifact_id,
            "source_credit_batch_id": self.source_credit_batch_id,
            "source_apply_receipt_id": self.source_apply_receipt_id,
            "source_checkpoint_content_sha256": self.source_checkpoint_content_sha256,
        }


@dataclass(frozen=True)
class RelationshipActionGateV2Checkpoint:
    artifact_id: str
    weights: tuple[float, ...]
    update_count: int
    informative_update_count: int
    processed_credit_ids: tuple[str, ...]
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        legacy._require_text(self.artifact_id, "artifact_id")
        if type(self.weights) is not tuple or len(self.weights) != _FEATURE_COUNT:
            raise ValueError("relationship action gate v2 checkpoint weights are invalid")
        if any(type(value) is not float or not math.isfinite(value) for value in self.weights):
            raise ValueError("relationship action gate v2 checkpoint weights are invalid")
        for name, value in (
            ("update_count", self.update_count),
            ("informative_update_count", self.informative_update_count),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.informative_update_count > self.update_count:
            raise ValueError("informative updates cannot exceed processed updates")
        if type(self.processed_credit_ids) is not tuple or any(
            type(value) is not str for value in self.processed_credit_ids
        ):
            raise TypeError("processed_credit_ids must be an exact tuple of strings")
        if len(self.processed_credit_ids) != self.update_count:
            raise ValueError("processed credit count must equal update_count")
        if tuple(sorted(self.processed_credit_ids)) != self.processed_credit_ids:
            raise ValueError("processed_credit_ids must use canonical order")
        if len(set(self.processed_credit_ids)) != len(self.processed_credit_ids):
            raise ValueError("processed_credit_ids must be unique")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 checkpoint schema mismatch")

    @property
    def content_sha256(self) -> str:
        return legacy._canonical_sha256(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "weights_hex": [value.hex() for value in self.weights],
            "update_count": self.update_count,
            "informative_update_count": self.informative_update_count,
            "processed_credit_ids": list(self.processed_credit_ids),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipActionGateV2Checkpoint":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "schema_version",
                "artifact_id",
                "weights_hex",
                "update_count",
                "informative_update_count",
                "processed_credit_ids",
            },
            source="relationship action gate v2 checkpoint",
        )
        weights = raw["weights_hex"]
        if not isinstance(weights, list) or any(not isinstance(v, str) for v in weights):
            raise ValueError("weights_hex must be an array of strings")
        return cls(
            artifact_id=legacy._payload_text(raw, "artifact_id"),
            weights=tuple(
                legacy._finite_float_from_hex(value, f"weights_hex[{index}]") for index, value in enumerate(weights)
            ),
            update_count=legacy._payload_int(raw, "update_count"),
            informative_update_count=legacy._payload_int(raw, "informative_update_count"),
            processed_credit_ids=legacy._payload_text_tuple(raw, "processed_credit_ids", allow_empty=True),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )


@dataclass(frozen=True)
class RelationshipActionGateV2Decision:
    decision_id: str
    forecast_id: str
    gate_action: legacy.RelationshipGateAction
    selected_action_id: str
    recommended_action_id: str
    steer_probability: float
    features: tuple[float, ...]
    artifact_id: str
    update_count: int
    evidence_refs: tuple[str, ...]
    rationale_codes: tuple[str, ...]
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("decision_id", self.decision_id),
            ("forecast_id", self.forecast_id),
            ("selected_action_id", self.selected_action_id),
            ("recommended_action_id", self.recommended_action_id),
            ("artifact_id", self.artifact_id),
        ):
            legacy._require_text(value, name)
        action_surface = {action.value for action in RELATIONSHIP_ACTIONS}
        if self.selected_action_id not in action_surface:
            raise ValueError("selected_action_id is outside relationship action surface")
        if self.recommended_action_id not in action_surface:
            raise ValueError("recommended_action_id is outside relationship action surface")
        if type(self.gate_action) is not legacy.RelationshipGateAction:
            raise TypeError("gate_action must be RelationshipGateAction")
        if (
            type(self.steer_probability) is not float
            or not math.isfinite(self.steer_probability)
            or not 0.0 <= self.steer_probability <= 1.0
        ):
            raise ValueError("steer_probability must be finite and in [0, 1]")
        if type(self.features) is not tuple or len(self.features) != _FEATURE_COUNT:
            raise ValueError("relationship action gate v2 features are invalid")
        if any(
            type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in self.features
        ):
            raise ValueError("relationship action gate v2 features are invalid")
        if type(self.update_count) is not int or self.update_count < 0:
            raise ValueError("update_count must be a non-negative integer")
        for field_name, values in (
            ("evidence_refs", self.evidence_refs),
            ("rationale_codes", self.rationale_codes),
        ):
            if type(values) is not tuple or any(type(value) is not str for value in values):
                raise TypeError(f"{field_name} must be an exact tuple of strings")
        legacy._require_non_empty_unique(self.evidence_refs, "evidence_refs")
        legacy._require_non_empty_unique(self.rationale_codes, "rationale_codes")
        expected_gate_action = (
            legacy.RelationshipGateAction.STEER if self.steer_probability > 0.5 else legacy.RelationshipGateAction.NOOP
        )
        if self.gate_action is not expected_gate_action:
            raise ValueError("gate action does not match the strict probability threshold")
        if (
            self.gate_action is legacy.RelationshipGateAction.NOOP
            and self.selected_action_id != RelationshipAction.NEUTRAL_NOOP.value
        ):
            raise ValueError("noop decision must select neutral_noop")
        expected_selected = (
            self.recommended_action_id
            if self.gate_action is legacy.RelationshipGateAction.STEER
            else RelationshipAction.NEUTRAL_NOOP.value
        )
        if self.selected_action_id != expected_selected:
            raise ValueError("selected action does not match gate action")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_DECISION_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 decision schema mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "forecast_id": self.forecast_id,
            "gate_action": self.gate_action.value,
            "selected_action_id": self.selected_action_id,
            "recommended_action_id": self.recommended_action_id,
            "steer_probability_hex": self.steer_probability.hex(),
            "features_hex": [value.hex() for value in self.features],
            "artifact_id": self.artifact_id,
            "update_count": self.update_count,
            "evidence_refs": list(self.evidence_refs),
            "rationale_codes": list(self.rationale_codes),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipActionGateV2Decision":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "schema_version",
                "decision_id",
                "forecast_id",
                "gate_action",
                "selected_action_id",
                "recommended_action_id",
                "steer_probability_hex",
                "features_hex",
                "artifact_id",
                "update_count",
                "evidence_refs",
                "rationale_codes",
            },
            source="relationship action gate v2 decision",
        )
        features = raw["features_hex"]
        if not isinstance(features, list) or any(not isinstance(v, str) for v in features):
            raise ValueError("features_hex must be an array of strings")
        return cls(
            decision_id=legacy._payload_text(raw, "decision_id"),
            forecast_id=legacy._payload_text(raw, "forecast_id"),
            gate_action=legacy.RelationshipGateAction(legacy._payload_text(raw, "gate_action")),
            selected_action_id=legacy._payload_text(raw, "selected_action_id"),
            recommended_action_id=legacy._payload_text(raw, "recommended_action_id"),
            steer_probability=legacy._finite_float_from_hex(
                legacy._payload_text(raw, "steer_probability_hex"),
                "steer_probability_hex",
            ),
            features=tuple(
                legacy._finite_float_from_hex(value, f"features_hex[{index}]") for index, value in enumerate(features)
            ),
            artifact_id=legacy._payload_text(raw, "artifact_id"),
            update_count=legacy._payload_int(raw, "update_count"),
            evidence_refs=legacy._payload_text_tuple(raw, "evidence_refs"),
            rationale_codes=legacy._payload_text_tuple(raw, "rationale_codes"),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )


@dataclass(frozen=True)
class RelationshipActionGateV2FrozenDecision:
    decision: RelationshipActionGateV2Decision
    checkpoint_content_sha256: str
    frozen_policy_id: str
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_FROZEN_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.decision) is not RelationshipActionGateV2Decision:
            raise TypeError("decision must be RelationshipActionGateV2Decision")
        legacy._require_sha256(self.checkpoint_content_sha256, "checkpoint_content_sha256")
        legacy._require_content_addressed_id(
            self.frozen_policy_id,
            "frozen_policy_id",
        )
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_FROZEN_DECISION_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 frozen decision schema mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_policy_id": self.frozen_policy_id,
            "checkpoint_content_sha256": self.checkpoint_content_sha256,
            "decision": self.decision.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipActionGateV2FrozenDecision":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "schema_version",
                "frozen_policy_id",
                "checkpoint_content_sha256",
                "decision",
            },
            source="relationship action gate v2 frozen decision",
        )
        return cls(
            decision=RelationshipActionGateV2Decision.from_payload(raw["decision"]),
            checkpoint_content_sha256=legacy._payload_text(raw, "checkpoint_content_sha256"),
            frozen_policy_id=legacy._payload_text(raw, "frozen_policy_id"),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )


@dataclass(frozen=True)
class RelationshipActionGateV2ForcedExposure:
    forecast: PreferenceActionForecast
    frozen_decision: RelationshipActionGateV2FrozenDecision
    assignment: RelationshipActionGateV2AssignmentReceipt
    delivered_action_id: str
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_FORCED_EXPOSURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.forecast) is not PreferenceActionForecast:
            raise TypeError("forecast must be PreferenceActionForecast")
        _require_exact_forecast_shape(self.forecast)
        if type(self.frozen_decision) is not RelationshipActionGateV2FrozenDecision:
            raise TypeError("frozen_decision must be RelationshipActionGateV2FrozenDecision")
        if type(self.assignment) is not RelationshipActionGateV2AssignmentReceipt:
            raise TypeError("assignment must be RelationshipActionGateV2AssignmentReceipt")
        legacy._require_text(self.delivered_action_id, "delivered_action_id")
        decision = self.frozen_decision.decision
        if (
            decision.forecast_id != self.forecast.forecast_id
            or decision.decision_id != self.forecast.decision_id
            or decision.recommended_action_id != self.forecast.recommended_action_id
        ):
            raise ValueError("forced exposure forecast/decision lineage mismatch")
        if self.assignment.decision_id != self.forecast.decision_id:
            raise ValueError("forced exposure assignment lineage mismatch")
        expected_delivery = (
            self.forecast.recommended_action_id
            if self.assignment.assignment_role is RelationshipActionGateV2AssignmentRole.CANDIDATE
            else RelationshipAction.NEUTRAL_NOOP.value
        )
        if self.delivered_action_id != expected_delivery:
            raise ValueError("delivered action does not match frozen assignment role")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_FORCED_EXPOSURE_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 exposure schema mismatch")

    @property
    def sequence_index(self) -> int:
        return self.assignment.sequence_index

    @property
    def artifact_id(self) -> str:
        return self.frozen_decision.decision.artifact_id

    @property
    def schedule_artifact(self) -> RelationshipActionGateV2AssignmentScheduleArtifact:
        return self.assignment.schedule_artifact

    @property
    def schedule_entry(self) -> RelationshipActionGateV2AssignmentScheduleEntry:
        return self.assignment.schedule_entry

    @property
    def candidate_is_nonnoop(self) -> bool:
        return self.forecast.recommended_action_id != RelationshipAction.NEUTRAL_NOOP.value

    @property
    def informative(self) -> bool:
        return self.candidate_is_nonnoop

    @property
    def assignment_design(self) -> RelationshipActionGateV2AssignmentDesign:
        return self.assignment.assignment_design

    @property
    def assignment_role(self) -> RelationshipActionGateV2AssignmentRole:
        return self.assignment.assignment_role

    @property
    def assignment_receipt_id(self) -> str:
        return self.assignment.assignment_id

    @property
    def treatment_indicator(self) -> float:
        return 1.0 if self.assignment.assignment_role is RelationshipActionGateV2AssignmentRole.CANDIDATE else 0.0

    @property
    def centered_assignment(self) -> float:
        return self.treatment_indicator - float.fromhex(self.assignment.centering_fraction_hex)

    @property
    def exposure_id(self) -> str:
        return f"{_EXPOSURE_PREFIX}{legacy._canonical_sha256(self._core_payload())}"

    def to_payload(self) -> dict[str, object]:
        return {"exposure_id": self.exposure_id, **self._core_payload()}

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipActionGateV2ForcedExposure":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "exposure_id",
                "schema_version",
                "forecast",
                "frozen_decision",
                "assignment",
                "delivered_action_id",
            },
            source="relationship action gate v2 forced exposure",
        )
        exposure = cls(
            forecast=preference_action_forecast_from_payload(raw["forecast"]),
            frozen_decision=RelationshipActionGateV2FrozenDecision.from_payload(raw["frozen_decision"]),
            assignment=RelationshipActionGateV2AssignmentReceipt.from_payload(raw["assignment"]),
            delivered_action_id=legacy._payload_text(raw, "delivered_action_id"),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )
        if legacy._payload_text(raw, "exposure_id") != exposure.exposure_id:
            raise ValueError("relationship action gate v2 exposure_id mismatch")
        return exposure

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "forecast": preference_action_forecast_to_payload(self.forecast),
            "frozen_decision": self.frozen_decision.to_payload(),
            "assignment": self.assignment.to_payload(),
            "delivered_action_id": self.delivered_action_id,
        }


@dataclass(frozen=True)
class RelationshipActionGateV2CreditBatch:
    exposures: tuple[RelationshipActionGateV2ForcedExposure, ...]
    credits: tuple[RelationshipActionCommonBaselineCredit, ...]
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_CREDIT_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.exposures) is not tuple or any(
            type(item) is not RelationshipActionGateV2ForcedExposure for item in self.exposures
        ):
            raise TypeError("v2 exposures must be an exact tuple of forced exposures")
        if type(self.credits) is not tuple or any(
            type(item) is not RelationshipActionCommonBaselineCredit for item in self.credits
        ):
            raise TypeError("v2 credits must be an exact tuple of common-baseline credits")
        if not self.exposures or len(self.exposures) != len(self.credits):
            raise ValueError("v2 credit batch requires equal non-empty entries")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_CREDIT_BATCH_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 batch schema mismatch")
        if tuple(item.sequence_index for item in self.exposures) != tuple(range(len(self.exposures))):
            raise ValueError("v2 credit batch sequence must be contiguous")
        if len({item.exposure_id for item in self.exposures}) != len(self.exposures):
            raise ValueError("v2 credit batch exposure ids must be unique")
        if len({item.assignment_receipt_id for item in self.exposures}) != len(self.exposures):
            raise ValueError("v2 assignment receipt ids must be unique")
        schedule = self.exposures[0].schedule_artifact
        if any(item.schedule_artifact != schedule for item in self.exposures):
            raise ValueError("v2 credit batch must share one complete schedule")
        if tuple(item.schedule_entry for item in self.exposures) != schedule.entries:
            raise ValueError("v2 credit batch must consume the complete schedule in order")
        if len({item.artifact_id for item in self.exposures}) != 1:
            raise ValueError("v2 credit batch must share one artifact")
        if len({item.frozen_decision.checkpoint_content_sha256 for item in self.exposures}) != 1:
            raise ValueError("v2 credit batch must share one cold checkpoint")
        if len({item.frozen_decision.frozen_policy_id for item in self.exposures}) != 1:
            raise ValueError("v2 credit batch must share one frozen policy")
        if len({item.assignment_design for item in self.exposures}) != 1:
            raise ValueError("v2 credit batch must share one assignment design")
        credit_ids: list[str] = []
        timestamps: list[int] = []
        decision_ids: list[str] = []
        forecast_ids: list[str] = []
        for exposure, credit in zip(self.exposures, self.credits, strict=True):
            if credit.forecast != exposure.forecast:
                raise ValueError("v2 common credit forecast lineage mismatch")
            if credit.external_evidence.action_id != exposure.delivered_action_id:
                raise ValueError("v2 common credit delivered-action lineage mismatch")
            if credit.settlement.action_id != exposure.delivered_action_id:
                raise ValueError("v2 common credit settlement action mismatch")
            if credit.parent_action_credit.abstract_action_id != exposure.delivered_action_id:
                raise ValueError("v2 parent credit delivered-action lineage mismatch")
            credit_ids.append(credit.record_id)
            timestamps.append(credit.parent_action_credit.timestamp_ms)
            decision_ids.append(exposure.forecast.decision_id)
            forecast_ids.append(exposure.forecast.forecast_id)
        if len(set(credit_ids)) != len(credit_ids):
            raise ValueError("v2 credit ids must be unique")
        if any(left >= right for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError("v2 credit timestamps must be strictly increasing")
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("v2 decision ids must be unique")
        if len(set(forecast_ids)) != len(forecast_ids):
            raise ValueError("v2 forecast ids must be unique")

    @property
    def artifact_id(self) -> str:
        return self.exposures[0].artifact_id

    @property
    def base_checkpoint_content_sha256(self) -> str:
        return self.exposures[0].frozen_decision.checkpoint_content_sha256

    @property
    def schedule_artifact(self) -> RelationshipActionGateV2AssignmentScheduleArtifact:
        return self.exposures[0].schedule_artifact

    @property
    def assignment_design(self) -> RelationshipActionGateV2AssignmentDesign:
        return self.exposures[0].assignment_design

    @property
    def informative_count(self) -> int:
        return sum(item.informative for item in self.exposures)

    @property
    def zero_information_count(self) -> int:
        return len(self.exposures) - self.informative_count

    @property
    def batch_id(self) -> str:
        return f"{_BATCH_PREFIX}{legacy._canonical_sha256(self._core_payload())}"

    def to_payload(self) -> dict[str, object]:
        return {"batch_id": self.batch_id, **self._core_payload()}

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        full_common_credits: tuple[RelationshipActionCommonBaselineCredit, ...],
    ) -> "RelationshipActionGateV2CreditBatch":
        if type(full_common_credits) is not tuple or any(
            type(item) is not RelationshipActionCommonBaselineCredit for item in full_common_credits
        ):
            raise TypeError("full_common_credits must contain exact typed credits")
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "batch_id",
                "schema_version",
                "artifact_id",
                "base_checkpoint_content_sha256",
                "schedule_artifact_id",
                "assignment_design",
                "entries",
            },
            source="relationship action gate v2 credit batch",
        )
        entries = raw["entries"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("v2 credit batch entries must be non-empty")
        if len(entries) != len(full_common_credits):
            raise ValueError("v2 full common credit count does not match projection")
        exposures: list[RelationshipActionGateV2ForcedExposure] = []
        for index, entry in enumerate(entries):
            item = legacy._require_exact_mapping(
                entry,
                expected={"exposure", "credit"},
                source=f"relationship action gate v2 batch entry {index}",
            )
            exposures.append(RelationshipActionGateV2ForcedExposure.from_payload(item["exposure"]))
            if item["credit"] != full_common_credits[index].to_payload():
                raise ValueError("v2 common credit audit projection mismatch")
        batch = cls(
            exposures=tuple(exposures),
            credits=full_common_credits,
            schema_version=legacy._payload_text(raw, "schema_version"),
        )
        if legacy._payload_text(raw, "artifact_id") != batch.artifact_id:
            raise ValueError("v2 batch artifact identity mismatch")
        if legacy._payload_text(raw, "base_checkpoint_content_sha256") != (batch.base_checkpoint_content_sha256):
            raise ValueError("v2 batch checkpoint identity mismatch")
        if legacy._payload_text(raw, "schedule_artifact_id") != (batch.schedule_artifact.artifact_id):
            raise ValueError("v2 batch schedule identity mismatch")
        if legacy._payload_text(raw, "assignment_design") != batch.assignment_design.value:
            raise ValueError("v2 batch assignment design mismatch")
        if legacy._payload_text(raw, "batch_id") != batch.batch_id:
            raise ValueError("v2 batch_id mismatch")
        return batch

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "base_checkpoint_content_sha256": self.base_checkpoint_content_sha256,
            "schedule_artifact_id": self.schedule_artifact.artifact_id,
            "assignment_design": self.assignment_design.value,
            "entries": [
                {
                    "exposure": exposure.to_payload(),
                    "credit": credit.to_payload(),
                }
                for exposure, credit in zip(self.exposures, self.credits, strict=True)
            ],
        }


@dataclass(frozen=True)
class RelationshipActionGateV2BatchPlan:
    batch: RelationshipActionGateV2CreditBatch
    pre_checkpoint: RelationshipActionGateV2Checkpoint
    candidate_checkpoint: RelationshipActionGateV2Checkpoint
    informative_count: int
    zero_information_count: int
    informative_candidate_count: int
    informative_noop_count: int
    cap_hit_count: int
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_BATCH_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.batch) is not RelationshipActionGateV2CreditBatch:
            raise TypeError("batch must be RelationshipActionGateV2CreditBatch")
        if type(self.pre_checkpoint) is not RelationshipActionGateV2Checkpoint:
            raise TypeError("pre_checkpoint must be RelationshipActionGateV2Checkpoint")
        if type(self.candidate_checkpoint) is not RelationshipActionGateV2Checkpoint:
            raise TypeError("candidate_checkpoint must be RelationshipActionGateV2Checkpoint")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_BATCH_PLAN_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 plan schema mismatch")
        counts = (
            self.informative_count,
            self.zero_information_count,
            self.informative_candidate_count,
            self.informative_noop_count,
            self.cap_hit_count,
        )
        if any(type(v) is not int or v < 0 for v in counts):
            raise ValueError("relationship action gate v2 plan counts are invalid")
        if self.informative_count != self.batch.informative_count:
            raise ValueError("v2 plan informative count mismatch")
        if self.zero_information_count != self.batch.zero_information_count:
            raise ValueError("v2 plan zero-information count mismatch")
        if self.informative_candidate_count + self.informative_noop_count != self.informative_count:
            raise ValueError("v2 plan informative assignment counts mismatch")
        if self.pre_checkpoint.content_sha256 != self.batch.base_checkpoint_content_sha256:
            raise ValueError("v2 plan pre-checkpoint does not match batch")
        if self.pre_checkpoint.artifact_id != self.batch.artifact_id:
            raise ValueError("v2 plan artifact does not match batch")
        if self.candidate_checkpoint.artifact_id != self.pre_checkpoint.artifact_id:
            raise ValueError("v2 plan candidate artifact mismatch")
        if self.candidate_checkpoint.update_count != self.pre_checkpoint.update_count + len(self.batch.credits):
            raise ValueError("v2 plan candidate update count mismatch")
        if self.candidate_checkpoint.informative_update_count != (
            self.pre_checkpoint.informative_update_count + self.informative_count
        ):
            raise ValueError("v2 plan candidate informative count mismatch")

    @property
    def plan_id(self) -> str:
        return f"{_PLAN_PREFIX}{legacy._canonical_sha256(self._core_payload())}"

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch.batch_id,
            "pre_checkpoint_content_sha256": self.pre_checkpoint.content_sha256,
            "candidate_checkpoint_content_sha256": self.candidate_checkpoint.content_sha256,
            "informative_count": self.informative_count,
            "zero_information_count": self.zero_information_count,
            "informative_candidate_count": self.informative_candidate_count,
            "informative_noop_count": self.informative_noop_count,
            "cap_hit_count": self.cap_hit_count,
        }


@dataclass(frozen=True)
class RelationshipActionGateV2BatchReceipt:
    batch_id: str
    plan_id: str
    disposition: legacy.RelationshipActionGateBatchDisposition
    pre_checkpoint_content_sha256: str
    candidate_checkpoint_content_sha256: str
    post_checkpoint_content_sha256: str
    credit_count: int
    informative_count: int
    zero_information_count: int
    informative_candidate_count: int
    informative_noop_count: int
    cap_hit_count: int
    weight_delta_hex: tuple[str, ...]
    update_count_delta: int
    informative_update_count_delta: int
    atomic_commit_count: int
    applied_credit_ids: tuple[str, ...]
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_BATCH_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        legacy._require_text(self.batch_id, "batch_id")
        legacy._require_text(self.plan_id, "plan_id")
        if type(self.disposition) is not legacy.RelationshipActionGateBatchDisposition:
            raise TypeError("disposition must be RelationshipActionGateBatchDisposition")
        for name, value in (
            ("pre_checkpoint_content_sha256", self.pre_checkpoint_content_sha256),
            ("candidate_checkpoint_content_sha256", self.candidate_checkpoint_content_sha256),
            ("post_checkpoint_content_sha256", self.post_checkpoint_content_sha256),
        ):
            legacy._require_sha256(value, name)
        counts = (
            self.credit_count,
            self.informative_count,
            self.zero_information_count,
            self.informative_candidate_count,
            self.informative_noop_count,
            self.cap_hit_count,
            self.update_count_delta,
            self.informative_update_count_delta,
            self.atomic_commit_count,
        )
        if any(type(v) is not int or v < 0 for v in counts):
            raise ValueError("relationship action gate v2 receipt counts are invalid")
        if self.credit_count < 1:
            raise ValueError("v2 receipt credit_count must be positive")
        if self.informative_count + self.zero_information_count != self.credit_count:
            raise ValueError("v2 receipt information counts do not close")
        if self.informative_candidate_count + self.informative_noop_count != self.informative_count:
            raise ValueError("v2 receipt assignment counts do not close")
        if type(self.weight_delta_hex) is not tuple or len(self.weight_delta_hex) != _FEATURE_COUNT:
            raise ValueError("v2 receipt weight delta length mismatch")
        if any(type(value) is not str for value in self.weight_delta_hex):
            raise TypeError("v2 receipt weight deltas must be exact strings")
        if type(self.applied_credit_ids) is not tuple or any(
            type(value) is not str for value in self.applied_credit_ids
        ):
            raise TypeError("v2 receipt applied credit ids must be an exact tuple")
        deltas = tuple(
            legacy._finite_float_from_hex(value, f"weight_delta_hex[{index}]")
            for index, value in enumerate(self.weight_delta_hex)
        )
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_BATCH_RECEIPT_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 receipt schema mismatch")
        if self.disposition is legacy.RelationshipActionGateBatchDisposition.APPLY:
            if self.post_checkpoint_content_sha256 != self.candidate_checkpoint_content_sha256:
                raise ValueError("v2 APPLY post checkpoint must equal candidate")
            if self.update_count_delta != self.credit_count:
                raise ValueError("v2 APPLY update count mismatch")
            if self.informative_update_count_delta != self.informative_count:
                raise ValueError("v2 APPLY informative count mismatch")
            if self.atomic_commit_count != 1:
                raise ValueError("v2 APPLY must commit exactly once")
            if len(self.applied_credit_ids) != self.credit_count:
                raise ValueError("v2 APPLY credit id count mismatch")
            legacy._require_non_empty_unique(self.applied_credit_ids, "applied_credit_ids")
        else:
            if self.post_checkpoint_content_sha256 != self.pre_checkpoint_content_sha256:
                raise ValueError("v2 WITHHOLD must preserve pre checkpoint")
            if any((self.update_count_delta, self.informative_update_count_delta, self.atomic_commit_count)):
                raise ValueError("v2 WITHHOLD cannot report updates")
            if self.applied_credit_ids:
                raise ValueError("v2 WITHHOLD cannot report applied credits")
            if any(value != 0.0 for value in deltas):
                raise ValueError("v2 WITHHOLD parameter delta must be zero")

    @property
    def weight_delta(self) -> tuple[float, ...]:
        return tuple(float.fromhex(value) for value in self.weight_delta_hex)

    @property
    def receipt_id(self) -> str:
        return f"{_RECEIPT_PREFIX}{legacy._canonical_sha256(self.to_payload(include_receipt_id=False))}"

    def to_payload(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        core: dict[str, object] = {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "plan_id": self.plan_id,
            "disposition": self.disposition.value,
            "pre_checkpoint_content_sha256": self.pre_checkpoint_content_sha256,
            "candidate_checkpoint_content_sha256": self.candidate_checkpoint_content_sha256,
            "post_checkpoint_content_sha256": self.post_checkpoint_content_sha256,
            "credit_count": self.credit_count,
            "informative_count": self.informative_count,
            "zero_information_count": self.zero_information_count,
            "informative_candidate_count": self.informative_candidate_count,
            "informative_noop_count": self.informative_noop_count,
            "cap_hit_count": self.cap_hit_count,
            "weight_delta_hex": list(self.weight_delta_hex),
            "update_count_delta": self.update_count_delta,
            "informative_update_count_delta": self.informative_update_count_delta,
            "atomic_commit_count": self.atomic_commit_count,
            "applied_credit_ids": list(self.applied_credit_ids),
        }
        return {"receipt_id": self.receipt_id, **core} if include_receipt_id else core

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipActionGateV2BatchReceipt":
        raw = legacy._require_exact_mapping(
            payload,
            expected={
                "receipt_id",
                "schema_version",
                "batch_id",
                "plan_id",
                "disposition",
                "pre_checkpoint_content_sha256",
                "candidate_checkpoint_content_sha256",
                "post_checkpoint_content_sha256",
                "credit_count",
                "informative_count",
                "zero_information_count",
                "informative_candidate_count",
                "informative_noop_count",
                "cap_hit_count",
                "weight_delta_hex",
                "update_count_delta",
                "informative_update_count_delta",
                "atomic_commit_count",
                "applied_credit_ids",
            },
            source="relationship action gate v2 batch receipt",
        )
        deltas = raw["weight_delta_hex"]
        if not isinstance(deltas, list) or any(not isinstance(v, str) for v in deltas):
            raise ValueError("weight_delta_hex must be an array of strings")
        receipt = cls(
            batch_id=legacy._payload_text(raw, "batch_id"),
            plan_id=legacy._payload_text(raw, "plan_id"),
            disposition=legacy.RelationshipActionGateBatchDisposition(legacy._payload_text(raw, "disposition")),
            pre_checkpoint_content_sha256=legacy._payload_text(raw, "pre_checkpoint_content_sha256"),
            candidate_checkpoint_content_sha256=legacy._payload_text(raw, "candidate_checkpoint_content_sha256"),
            post_checkpoint_content_sha256=legacy._payload_text(raw, "post_checkpoint_content_sha256"),
            credit_count=legacy._payload_int(raw, "credit_count"),
            informative_count=legacy._payload_int(raw, "informative_count"),
            zero_information_count=legacy._payload_int(raw, "zero_information_count"),
            informative_candidate_count=legacy._payload_int(raw, "informative_candidate_count"),
            informative_noop_count=legacy._payload_int(raw, "informative_noop_count"),
            cap_hit_count=legacy._payload_int(raw, "cap_hit_count"),
            weight_delta_hex=tuple(deltas),
            update_count_delta=legacy._payload_int(raw, "update_count_delta"),
            informative_update_count_delta=legacy._payload_int(raw, "informative_update_count_delta"),
            atomic_commit_count=legacy._payload_int(raw, "atomic_commit_count"),
            applied_credit_ids=legacy._payload_text_tuple(raw, "applied_credit_ids", allow_empty=True),
            schema_version=legacy._payload_text(raw, "schema_version"),
        )
        if legacy._payload_text(raw, "receipt_id") != receipt.receipt_id:
            raise ValueError("relationship action gate v2 receipt_id mismatch")
        return receipt


@dataclass(frozen=True)
class RelationshipActionGateV2FrozenPolicy:
    artifact: RelationshipActionGateV2Artifact
    checkpoint: RelationshipActionGateV2Checkpoint
    transition_batch: RelationshipActionGateV2CreditBatch | None = None
    transition_receipt: RelationshipActionGateV2BatchReceipt | None = None
    schema_version: str = RELATIONSHIP_ACTION_GATE_V2_FROZEN_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.artifact) is not RelationshipActionGateV2Artifact:
            raise TypeError("artifact must be RelationshipActionGateV2Artifact")
        if type(self.checkpoint) is not RelationshipActionGateV2Checkpoint:
            raise TypeError("checkpoint must be RelationshipActionGateV2Checkpoint")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_V2_FROZEN_POLICY_SCHEMA_VERSION:
            raise ValueError("relationship action gate v2 frozen policy schema mismatch")
        if self.checkpoint.artifact_id != self.artifact.artifact_id:
            raise ValueError("v2 frozen policy artifact/checkpoint mismatch")
        if any(abs(value) > self.artifact.max_abs_parameter for value in self.checkpoint.weights):
            raise ValueError("v2 frozen policy checkpoint exceeds parameter cap")
        if (self.transition_batch is None) is not (self.transition_receipt is None):
            raise ValueError("v2 frozen policy transition batch/receipt must be paired")
        if self.transition_batch is None:
            if self.checkpoint.update_count != 0:
                raise ValueError("non-cold v2 policy requires a transition")
            if self.checkpoint.weights != self.artifact.weights:
                raise ValueError("cold v2 policy parameters differ from artifact")
        else:
            if type(self.transition_batch) is not RelationshipActionGateV2CreditBatch:
                raise ValueError("learned v2 policy requires transition batch")
            if type(self.transition_receipt) is not RelationshipActionGateV2BatchReceipt:
                raise ValueError("learned v2 policy requires transition receipt")
            replayed = RelationshipActionGateV2.from_credit_batch_transition(
                self.artifact,
                batch=self.transition_batch,
                receipt=self.transition_receipt,
            )
            if replayed.export_checkpoint() != self.checkpoint:
                raise ValueError("learned v2 policy differs from exact replay")

    @property
    def policy_id(self) -> str:
        return f"{_POLICY_PREFIX}{legacy._canonical_sha256(self._core_payload())}"

    def decide(self, forecast: PreferenceActionForecast) -> RelationshipActionGateV2FrozenDecision:
        return _frozen_decide(
            artifact=self.artifact,
            checkpoint=self.checkpoint,
            policy_id=self.policy_id,
            forecast=forecast,
        )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact": self.artifact.to_payload(),
            "checkpoint": self.checkpoint.to_payload(),
            "transition_batch_id": None if self.transition_batch is None else self.transition_batch.batch_id,
            "transition_receipt_id": None if self.transition_receipt is None else self.transition_receipt.receipt_id,
        }


class RelationshipActionGateV2:
    """Bias-free bounded gate with one atomic common-credit transition."""

    def __init__(
        self,
        *,
        artifact: RelationshipActionGateV2Artifact,
        checkpoint: RelationshipActionGateV2Checkpoint | None = None,
    ) -> None:
        if type(artifact) is not RelationshipActionGateV2Artifact:
            raise TypeError("artifact must be RelationshipActionGateV2Artifact")
        if checkpoint is not None and type(checkpoint) is not RelationshipActionGateV2Checkpoint:
            raise TypeError("checkpoint must be RelationshipActionGateV2Checkpoint")
        self._artifact = artifact
        self._checkpoint = checkpoint or RelationshipActionGateV2Checkpoint(
            artifact_id=artifact.artifact_id,
            weights=artifact.weights,
            update_count=0,
            informative_update_count=0,
            processed_credit_ids=(),
        )
        self._transition_batch: RelationshipActionGateV2CreditBatch | None = None
        self._transition_receipt: RelationshipActionGateV2BatchReceipt | None = None
        self._validate_checkpoint(self._checkpoint)
        if checkpoint is not None and checkpoint.update_count != 0:
            raise ValueError("non-cold v2 checkpoint requires exact batch plus APPLY receipt replay")

    @classmethod
    def from_credit_batch_transition(
        cls,
        artifact: RelationshipActionGateV2Artifact,
        *,
        batch: RelationshipActionGateV2CreditBatch,
        receipt: RelationshipActionGateV2BatchReceipt,
    ) -> "RelationshipActionGateV2":
        if type(receipt) is not RelationshipActionGateV2BatchReceipt:
            raise TypeError("receipt must be RelationshipActionGateV2BatchReceipt")
        gate = cls(artifact=artifact)
        plan = gate.plan_credit_batch(batch)
        expected = gate.commit_credit_batch(
            plan,
            disposition=receipt.disposition,
        )
        if expected != receipt:
            raise ValueError("v2 persisted transition receipt differs from exact replay")
        return gate

    @classmethod
    def from_applied_credit_batch(
        cls,
        artifact: RelationshipActionGateV2Artifact,
        *,
        batch: RelationshipActionGateV2CreditBatch,
        receipt: RelationshipActionGateV2BatchReceipt,
    ) -> "RelationshipActionGateV2":
        if type(receipt) is not RelationshipActionGateV2BatchReceipt:
            raise TypeError("receipt must be RelationshipActionGateV2BatchReceipt")
        if receipt.disposition is not legacy.RelationshipActionGateBatchDisposition.APPLY:
            raise ValueError("v2 applied replay requires an APPLY receipt")
        return cls.from_credit_batch_transition(
            artifact,
            batch=batch,
            receipt=receipt,
        )

    @property
    def artifact(self) -> RelationshipActionGateV2Artifact:
        return self._artifact

    def export_checkpoint(self) -> RelationshipActionGateV2Checkpoint:
        return self._checkpoint

    def freeze_for_evaluation(self) -> RelationshipActionGateV2FrozenPolicy:
        return RelationshipActionGateV2FrozenPolicy(
            artifact=self._artifact,
            checkpoint=self._checkpoint,
            transition_batch=self._transition_batch,
            transition_receipt=self._transition_receipt,
        )

    def record_forced_exposure(
        self,
        forecast: PreferenceActionForecast,
        *,
        assignment: RelationshipActionGateV2AssignmentReceipt,
        delivered_action_id: str,
    ) -> RelationshipActionGateV2ForcedExposure:
        if type(assignment) is not RelationshipActionGateV2AssignmentReceipt:
            raise TypeError("assignment must be RelationshipActionGateV2AssignmentReceipt")
        if self._checkpoint.update_count != 0 or self._transition_receipt is not None:
            raise ValueError("v2 forced exposure requires a cold phase checkpoint")
        before = self.export_checkpoint()
        frozen = self.freeze_for_evaluation().decide(forecast)
        exposure = RelationshipActionGateV2ForcedExposure(
            forecast=forecast,
            frozen_decision=frozen,
            assignment=assignment,
            delivered_action_id=delivered_action_id,
        )
        if self.export_checkpoint() != before:
            raise RuntimeError("recording a v2 forced exposure mutated the gate")
        return exposure

    def plan_credit_batch(
        self,
        batch: RelationshipActionGateV2CreditBatch,
    ) -> RelationshipActionGateV2BatchPlan:
        if type(batch) is not RelationshipActionGateV2CreditBatch:
            raise TypeError("batch must be RelationshipActionGateV2CreditBatch")
        pre = self.export_checkpoint()
        if pre.update_count != 0 or self._transition_receipt is not None:
            raise ValueError("v2 gate permits exactly one transition from a cold checkpoint")
        if batch.artifact_id != self._artifact.artifact_id:
            raise ValueError("v2 batch artifact mismatch")
        if batch.base_checkpoint_content_sha256 != pre.content_sha256:
            raise ValueError("v2 batch checkpoint is stale")
        if any(credit.record_id in pre.processed_credit_ids for credit in batch.credits):
            raise ValueError("v2 common-baseline credit was already processed")
        policy = self.freeze_for_evaluation()
        for exposure in batch.exposures:
            if policy.decide(exposure.forecast) != exposure.frozen_decision:
                raise ValueError("v2 batch contains forged frozen decision")

        weights = pre.weights
        cap = self._artifact.max_abs_parameter
        rate = self._artifact.active_learning_rate
        cap_hits = 0
        informative_candidate = 0
        informative_noop = 0
        for exposure, credit in zip(batch.exposures, batch.credits, strict=True):
            if not exposure.informative:
                continue
            if exposure.assignment_role is RelationshipActionGateV2AssignmentRole.CANDIDATE:
                informative_candidate += 1
            else:
                informative_noop += 1
            scale = rate * credit.credit_value * exposure.centered_assignment
            candidate_values = tuple(
                weight + scale * feature
                for weight, feature in zip(
                    weights,
                    exposure.frozen_decision.decision.features,
                    strict=True,
                )
            )
            cap_hits += sum(abs(value) > cap for value in candidate_values)
            weights = tuple(max(-cap, min(cap, value)) for value in candidate_values)
        processed = tuple(sorted({*pre.processed_credit_ids, *(item.record_id for item in batch.credits)}))
        candidate = RelationshipActionGateV2Checkpoint(
            artifact_id=pre.artifact_id,
            weights=weights,
            update_count=pre.update_count + len(batch.credits),
            informative_update_count=(pre.informative_update_count + batch.informative_count),
            processed_credit_ids=processed,
        )
        return RelationshipActionGateV2BatchPlan(
            batch=batch,
            pre_checkpoint=pre,
            candidate_checkpoint=candidate,
            informative_count=batch.informative_count,
            zero_information_count=batch.zero_information_count,
            informative_candidate_count=informative_candidate,
            informative_noop_count=informative_noop,
            cap_hit_count=cap_hits,
        )

    def commit_credit_batch(
        self,
        plan: RelationshipActionGateV2BatchPlan,
        *,
        disposition: legacy.RelationshipActionGateBatchDisposition,
    ) -> RelationshipActionGateV2BatchReceipt:
        if type(plan) is not RelationshipActionGateV2BatchPlan:
            raise TypeError("plan must be RelationshipActionGateV2BatchPlan")
        if type(disposition) is not legacy.RelationshipActionGateBatchDisposition:
            raise TypeError("disposition must be RelationshipActionGateBatchDisposition")
        if self._checkpoint.update_count != 0 or self._transition_receipt is not None:
            raise ValueError("v2 gate transition was already committed")
        expected = self.plan_credit_batch(plan.batch)
        if expected != plan:
            raise ValueError("v2 batch plan differs from current pure transition")
        pre = plan.pre_checkpoint
        candidate = plan.candidate_checkpoint
        if disposition is legacy.RelationshipActionGateBatchDisposition.APPLY:
            post = candidate
            delta = tuple(after - before for before, after in zip(pre.weights, post.weights, strict=True))
            update_delta = len(plan.batch.credits)
            informative_delta = plan.informative_count
            atomic = 1
            applied = tuple(item.record_id for item in plan.batch.credits)
        else:
            post = pre
            delta = (0.0,) * _FEATURE_COUNT
            update_delta = 0
            informative_delta = 0
            atomic = 0
            applied = ()
        receipt = RelationshipActionGateV2BatchReceipt(
            batch_id=plan.batch.batch_id,
            plan_id=plan.plan_id,
            disposition=disposition,
            pre_checkpoint_content_sha256=pre.content_sha256,
            candidate_checkpoint_content_sha256=candidate.content_sha256,
            post_checkpoint_content_sha256=post.content_sha256,
            credit_count=len(plan.batch.credits),
            informative_count=plan.informative_count,
            zero_information_count=plan.zero_information_count,
            informative_candidate_count=plan.informative_candidate_count,
            informative_noop_count=plan.informative_noop_count,
            cap_hit_count=plan.cap_hit_count,
            weight_delta_hex=tuple(value.hex() for value in delta),
            update_count_delta=update_delta,
            informative_update_count_delta=informative_delta,
            atomic_commit_count=atomic,
            applied_credit_ids=applied,
        )
        self._checkpoint = post
        self._transition_batch = plan.batch
        self._transition_receipt = receipt
        return receipt

    def _validate_checkpoint(self, checkpoint: RelationshipActionGateV2Checkpoint) -> None:
        if type(checkpoint) is not RelationshipActionGateV2Checkpoint:
            raise TypeError("checkpoint must be RelationshipActionGateV2Checkpoint")
        if checkpoint.artifact_id != self._artifact.artifact_id:
            raise ValueError("v2 checkpoint artifact mismatch")
        if any(abs(value) > self._artifact.max_abs_parameter for value in checkpoint.weights):
            raise ValueError("v2 checkpoint exceeds parameter cap")
        if checkpoint.update_count == 0 and checkpoint.weights != self._artifact.weights:
            raise ValueError("cold v2 checkpoint differs from artifact parameters")


def relationship_action_gate_v2_features(
    forecast: PreferenceActionForecast,
) -> tuple[float, ...]:
    """Publish four bounded, centred owner features with no constant term."""

    _require_exact_forecast_shape(forecast)
    confidence, positive_mass, margin, certainty, _support = legacy.relationship_action_gate_features(forecast)
    return (
        2.0 * confidence - 1.0,
        2.0 * positive_mass - 1.0,
        margin,
        2.0 * certainty - 1.0,
    )


def _require_exact_forecast_shape(forecast: PreferenceActionForecast) -> None:
    """Reject mutable/subclass shapes using the public owner codec."""

    if type(forecast) is not PreferenceActionForecast:
        raise TypeError("forecast must be an exact PreferenceActionForecast")
    replayed = preference_action_forecast_from_payload(preference_action_forecast_to_payload(forecast))
    if replayed != forecast:
        raise TypeError("forecast must use the immutable canonical owner shape")


def temporal_action_advisory_from_gate_v2_decision(
    frozen_decision: RelationshipActionGateV2FrozenDecision,
    *,
    frozen_policy: RelationshipActionGateV2FrozenPolicy,
    forecast: PreferenceActionForecast,
) -> TemporalActionAdvisoryProposal:
    if type(frozen_decision) is not RelationshipActionGateV2FrozenDecision:
        raise TypeError("frozen_decision must be RelationshipActionGateV2FrozenDecision")
    if type(frozen_policy) is not RelationshipActionGateV2FrozenPolicy:
        raise TypeError("frozen_policy must be RelationshipActionGateV2FrozenPolicy")
    _require_exact_forecast_shape(forecast)
    if frozen_policy.decide(forecast) != frozen_decision:
        raise ValueError("frozen decision differs from exact policy replay")
    decision = frozen_decision.decision
    confidence = (
        decision.steer_probability
        if decision.gate_action is legacy.RelationshipGateAction.STEER
        else 1.0 - decision.steer_probability
    )
    checkpoint_ref = f"relationship-action-gate-v2-checkpoint-sha256:{frozen_decision.checkpoint_content_sha256}"
    forecast_payload = preference_action_forecast_to_payload(forecast)
    forecast_ref = f"preference-action-forecast-sha256:{legacy._canonical_sha256(forecast_payload)}"
    advisory_lineage = {
        "forecast": forecast_payload,
        "frozen_decision": frozen_decision.to_payload(),
    }
    return TemporalActionAdvisoryProposal(
        advisory_id=(f"relationship-action-advisory-v2-sha256:{legacy._canonical_sha256(advisory_lineage)}"),
        decision_id=decision.decision_id,
        prediction_id=decision.forecast_id,
        action_id=decision.selected_action_id,
        confidence=confidence,
        policy_artifact_id=decision.artifact_id,
        policy_artifact_version=2,
        evidence_refs=tuple(
            dict.fromkeys(
                (
                    *decision.evidence_refs,
                    frozen_decision.frozen_policy_id,
                    checkpoint_ref,
                    forecast_ref,
                )
            )
        ),
        rationale_codes=(
            *decision.rationale_codes,
            f"gate:{decision.gate_action.value}",
            "lineage:frozen-policy-and-checkpoint-bound",
        ),
        evaluator_only=False,
        active_authorized=False,
    )


def _frozen_decide(
    *,
    artifact: RelationshipActionGateV2Artifact,
    checkpoint: RelationshipActionGateV2Checkpoint,
    policy_id: str,
    forecast: PreferenceActionForecast,
) -> RelationshipActionGateV2FrozenDecision:
    features = relationship_action_gate_v2_features(forecast)
    logit = math.fsum(weight * feature for weight, feature in zip(checkpoint.weights, features, strict=True))
    probability = _sigmoid(logit)
    gate_action = legacy.RelationshipGateAction.STEER if probability > 0.5 else legacy.RelationshipGateAction.NOOP
    selected = (
        forecast.recommended_action_id
        if gate_action is legacy.RelationshipGateAction.STEER
        else RelationshipAction.NEUTRAL_NOOP.value
    )
    decision = RelationshipActionGateV2Decision(
        decision_id=forecast.decision_id,
        forecast_id=forecast.forecast_id,
        gate_action=gate_action,
        selected_action_id=selected,
        recommended_action_id=forecast.recommended_action_id,
        steer_probability=probability,
        features=features,
        artifact_id=artifact.artifact_id,
        update_count=checkpoint.update_count,
        evidence_refs=tuple(dict.fromkeys((*forecast.source_record_ids, *forecast.evidence))),
        rationale_codes=(
            "policy:bias-free-centred-assignment-logistic-gate-v2",
            "inputs:typed-owner-forecast-only",
            "learning:common-noop-credit-development-feature-moment-only",
        ),
    )
    return RelationshipActionGateV2FrozenDecision(
        decision=decision,
        checkpoint_content_sha256=checkpoint.content_sha256,
        frozen_policy_id=policy_id,
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


__all__ = [
    "RELATIONSHIP_ACTION_GATE_V2_ARTIFACT_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_RECEIPT_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_SCHEDULE_ENTRY_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_ASSIGNMENT_SCHEDULE_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_BATCH_PLAN_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_BATCH_RECEIPT_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_CHECKPOINT_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_CREDIT_BATCH_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_DECISION_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_FEATURE_ORDER",
    "RELATIONSHIP_ACTION_GATE_V2_FORCED_EXPOSURE_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_FROZEN_DECISION_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_FROZEN_POLICY_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_V2_OBJECTIVE_ID",
    "RELATIONSHIP_ACTION_GATE_V2_OPERATOR_ID",
    "RELATIONSHIP_ACTION_GATE_V2_THRESHOLD_RULE",
    "RelationshipActionGateV2",
    "RelationshipActionGateV2Artifact",
    "RelationshipActionGateV2ArtifactKind",
    "RelationshipActionGateV2AssignmentDesign",
    "RelationshipActionGateV2AssignmentReceipt",
    "RelationshipActionGateV2AssignmentRole",
    "RelationshipActionGateV2AssignmentScheduleArtifact",
    "RelationshipActionGateV2AssignmentScheduleEntry",
    "RelationshipActionGateV2BatchPlan",
    "RelationshipActionGateV2BatchReceipt",
    "RelationshipActionGateV2Checkpoint",
    "RelationshipActionGateV2CreditBatch",
    "RelationshipActionGateV2Decision",
    "RelationshipActionGateV2ForcedExposure",
    "RelationshipActionGateV2FrozenDecision",
    "RelationshipActionGateV2FrozenPolicy",
    "relationship_action_gate_v2_features",
    "temporal_action_advisory_from_gate_v2_decision",
]
