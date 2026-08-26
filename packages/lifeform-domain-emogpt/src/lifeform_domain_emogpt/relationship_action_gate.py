"""Bounded relationship-action gate for the companion vertical.

The preference owner freezes an action-conditional forecast before acting.
This gate only decides whether to adopt that recommendation (``steer``) or
fall back to ``neutral_noop``.  It never reads raw dialogue, evaluator scores,
or hidden Relationship Lab truth.  Online updates accept only the dedicated
PE-derived credit level and leave the forecast reader and expression executor
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

from volvence_zero.credit import CreditRecord
from volvence_zero.memory import Track
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    preference_action_forecast_to_payload,
)
from volvence_zero.temporal_types import TemporalActionAdvisoryProposal

from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)


RELATIONSHIP_ACTION_GATE_SCHEMA_VERSION = "relationship-action-gate.v1"
RELATIONSHIP_ACTION_CREDIT_LEVEL = "relationship_action_prediction_error"
RELATIONSHIP_ACTION_GATE_THETA0_SCHEMA_VERSION = (
    "relationship-action-gate-theta0-artifact.v1"
)
RELATIONSHIP_ACTION_GATE_FROZEN_POLICY_SCHEMA_VERSION = (
    "relationship-action-gate-frozen-policy.v1"
)
RELATIONSHIP_ACTION_GATE_FORCED_EXPOSURE_SCHEMA_VERSION = (
    "relationship-action-gate-forced-exposure.v1"
)
RELATIONSHIP_ACTION_GATE_CREDIT_BATCH_SCHEMA_VERSION = (
    "relationship-action-gate-credit-batch.v1"
)
RELATIONSHIP_ACTION_GATE_BATCH_PLAN_SCHEMA_VERSION = (
    "relationship-action-gate-batch-plan.v1"
)
RELATIONSHIP_ACTION_GATE_BATCH_RECEIPT_SCHEMA_VERSION = (
    "relationship-action-gate-batch-receipt.v1"
)
RELATIONSHIP_ACTION_GATE_OPERATOR_ID = "bounded-logistic-gate.v1"
RELATIONSHIP_ACTION_GATE_THRESHOLD_RULE = "steer_probability_strictly_greater_than_0.5"
RELATIONSHIP_ACTION_GATE_FEATURE_ORDER = (
    "forecast_confidence",
    "recommended_positive_mass",
    "recommended_positive_margin_over_noop",
    "recommended_entropy_certainty",
    "typed_source_support",
)
_FEATURE_COUNT = len(RELATIONSHIP_ACTION_GATE_FEATURE_ORDER)
_POSITIVE_OUTCOMES = frozenset({"helped", "felt_heard"})
_THETA0_ARTIFACT_PREFIX = "relationship-action-gate-theta0-sha256:"
_RELATIONSHIP_ACTION_CREDIT_RECORD_PREFIX = "relationship-action-pe-credit:"
_RELATIONSHIP_ACTION_SOCIAL_PE_SOURCE_PREFIX = "social_pe:social-pe:"


class RelationshipActionGateMode(str, Enum):
    LEARNED = "learned"
    NOOP = "noop"
    ALWAYS = "always"
    RANDOM = "random"
    ORACLE = "oracle"


class RelationshipGateAction(str, Enum):
    NOOP = "noop"
    STEER = "steer"


@dataclass(frozen=True)
class RelationshipActionGateArtifact:
    """Frozen gate shape; only its bounded online state may change."""

    artifact_id: str = "relationship-action-gate-zero-init"
    artifact_version: int = 1
    initial_weights: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    initial_bias: float = 0.0
    learning_rate: float = 0.25
    max_abs_parameter: float = 4.0
    schema_version: str = RELATIONSHIP_ACTION_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact_id")
        if self.artifact_version < 1:
            raise ValueError("artifact_version must be >= 1")
        if len(self.initial_weights) != _FEATURE_COUNT:
            raise ValueError(f"initial_weights must contain {_FEATURE_COUNT} values")
        for value in (*self.initial_weights, self.initial_bias):
            if not math.isfinite(value):
                raise ValueError("gate parameters must be finite")
        if not math.isfinite(self.learning_rate) or not 0.0 < self.learning_rate <= 0.5:
            raise ValueError("learning_rate must be finite and in (0, 0.5]")
        if (
            not math.isfinite(self.max_abs_parameter)
            or not 0.5 <= self.max_abs_parameter <= 8.0
        ):
            raise ValueError("max_abs_parameter must be finite and in [0.5, 8]")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_SCHEMA_VERSION:
            raise ValueError("relationship action gate schema version mismatch")


@dataclass(frozen=True)
class RelationshipActionGateTheta0Artifact:
    """Content-addressed, non-zero cold gate used only by new experiments.

    This is deliberately separate from :class:`RelationshipActionGateArtifact`.
    The legacy v1 artifact and every historical Decision/Checkpoint payload stay
    byte-for-byte compatible; a theta0 artifact materializes a legacy-shaped
    runtime artifact only when a new experiment explicitly opts in.
    """

    artifact_id: str
    weights_hex: tuple[str, ...]
    bias_hex: str
    learning_rate_hex: str
    max_abs_parameter_hex: str
    source_checkpoint_content_sha256: str
    source_batch_artifact_id: str
    operator_id: str = RELATIONSHIP_ACTION_GATE_OPERATOR_ID
    feature_order: tuple[str, ...] = RELATIONSHIP_ACTION_GATE_FEATURE_ORDER
    threshold_rule: str = RELATIONSHIP_ACTION_GATE_THRESHOLD_RULE
    schema_version: str = RELATIONSHIP_ACTION_GATE_THETA0_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact_id")
        _require_sha256(
            self.source_checkpoint_content_sha256,
            "source_checkpoint_content_sha256",
        )
        _require_content_addressed_id(
            self.source_batch_artifact_id,
            "source_batch_artifact_id",
        )
        if self.operator_id != RELATIONSHIP_ACTION_GATE_OPERATOR_ID:
            raise ValueError("theta0 operator_id mismatch")
        if self.feature_order != RELATIONSHIP_ACTION_GATE_FEATURE_ORDER:
            raise ValueError("theta0 feature_order mismatch")
        if self.threshold_rule != RELATIONSHIP_ACTION_GATE_THRESHOLD_RULE:
            raise ValueError("theta0 threshold_rule mismatch")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_THETA0_SCHEMA_VERSION:
            raise ValueError("theta0 schema_version mismatch")
        if len(self.weights_hex) != _FEATURE_COUNT:
            raise ValueError(f"theta0 weights must contain {_FEATURE_COUNT} values")
        weights = tuple(
            _finite_float_from_hex(value, f"weights_hex[{index}]")
            for index, value in enumerate(self.weights_hex)
        )
        bias = _finite_float_from_hex(self.bias_hex, "bias_hex")
        learning_rate = _finite_float_from_hex(
            self.learning_rate_hex,
            "learning_rate_hex",
        )
        max_abs_parameter = _finite_float_from_hex(
            self.max_abs_parameter_hex,
            "max_abs_parameter_hex",
        )
        if not any(value != 0.0 for value in (*weights, bias)):
            raise ValueError("theta0 parameters must contain at least one non-zero value")
        if not 0.0 < learning_rate <= 0.5:
            raise ValueError("theta0 learning_rate must be in (0, 0.5]")
        if not 0.5 <= max_abs_parameter <= 8.0:
            raise ValueError("theta0 max_abs_parameter must be in [0.5, 8]")
        if any(abs(value) > max_abs_parameter for value in (*weights, bias)):
            raise ValueError("theta0 parameters exceed max_abs_parameter")
        expected_id = f"{_THETA0_ARTIFACT_PREFIX}{_canonical_sha256(self._core_payload())}"
        if self.artifact_id != expected_id:
            raise ValueError("theta0 artifact_id does not match canonical content")

    @classmethod
    def create(
        cls,
        *,
        source_checkpoint: RelationshipActionGateCheckpoint,
        learning_rate: float,
        max_abs_parameter: float,
        source_batch_artifact_id: str,
    ) -> "RelationshipActionGateTheta0Artifact":
        if not isinstance(source_checkpoint, RelationshipActionGateCheckpoint):
            raise TypeError(
                "source_checkpoint must be RelationshipActionGateCheckpoint"
            )
        if source_checkpoint.pending_decisions:
            raise ValueError("theta0 source checkpoint cannot contain pending decisions")
        if source_checkpoint.update_count < 1:
            raise ValueError("theta0 source checkpoint must contain applied PE credit")
        if len(source_checkpoint.processed_credit_ids) != source_checkpoint.update_count:
            raise ValueError("theta0 source checkpoint credit/update counts do not match")
        weights = source_checkpoint.weights
        bias = source_checkpoint.bias
        weights_hex = tuple(
            _finite_float_hex(value, f"weights[{index}]")
            for index, value in enumerate(weights)
        )
        bias_hex = _finite_float_hex(bias, "bias")
        learning_rate_hex = _finite_float_hex(learning_rate, "learning_rate")
        max_abs_parameter_hex = _finite_float_hex(
            max_abs_parameter,
            "max_abs_parameter",
        )
        core = {
            "schema_version": RELATIONSHIP_ACTION_GATE_THETA0_SCHEMA_VERSION,
            "operator_id": RELATIONSHIP_ACTION_GATE_OPERATOR_ID,
            "feature_order": list(RELATIONSHIP_ACTION_GATE_FEATURE_ORDER),
            "threshold_rule": RELATIONSHIP_ACTION_GATE_THRESHOLD_RULE,
            "weights_hex": list(weights_hex),
            "bias_hex": bias_hex,
            "learning_rate_hex": learning_rate_hex,
            "max_abs_parameter_hex": max_abs_parameter_hex,
            "source_checkpoint_content_sha256": source_checkpoint.content_sha256,
            "source_batch_artifact_id": source_batch_artifact_id,
        }
        artifact_id = f"{_THETA0_ARTIFACT_PREFIX}{_canonical_sha256(core)}"
        artifact = cls(
            artifact_id=artifact_id,
            weights_hex=weights_hex,
            bias_hex=bias_hex,
            learning_rate_hex=learning_rate_hex,
            max_abs_parameter_hex=max_abs_parameter_hex,
            source_checkpoint_content_sha256=source_checkpoint.content_sha256,
            source_batch_artifact_id=source_batch_artifact_id,
        )
        artifact.validate_source_checkpoint(source_checkpoint)
        return artifact

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(float.fromhex(value) for value in self.weights_hex)

    @property
    def bias(self) -> float:
        return float.fromhex(self.bias_hex)

    @property
    def learning_rate(self) -> float:
        return float.fromhex(self.learning_rate_hex)

    @property
    def max_abs_parameter(self) -> float:
        return float.fromhex(self.max_abs_parameter_hex)

    def to_runtime_artifact(self) -> RelationshipActionGateArtifact:
        return RelationshipActionGateArtifact(
            artifact_id=self.artifact_id,
            artifact_version=2,
            initial_weights=self.weights,
            initial_bias=self.bias,
            learning_rate=self.learning_rate,
            max_abs_parameter=self.max_abs_parameter,
        )

    def validate_source_checkpoint(
        self,
        source_checkpoint: RelationshipActionGateCheckpoint,
    ) -> None:
        """Verify that the frozen theta0 is exactly the cited learned checkpoint."""

        if not isinstance(source_checkpoint, RelationshipActionGateCheckpoint):
            raise TypeError(
                "source_checkpoint must be RelationshipActionGateCheckpoint"
            )
        if source_checkpoint.content_sha256 != self.source_checkpoint_content_sha256:
            raise ValueError("theta0 source checkpoint content hash mismatch")
        if source_checkpoint.pending_decisions:
            raise ValueError("theta0 source checkpoint cannot contain pending decisions")
        if source_checkpoint.update_count < 1:
            raise ValueError("theta0 source checkpoint must contain applied PE credit")
        if len(source_checkpoint.processed_credit_ids) != source_checkpoint.update_count:
            raise ValueError("theta0 source checkpoint credit/update counts do not match")
        if source_checkpoint.weights != self.weights or source_checkpoint.bias != self.bias:
            raise ValueError("theta0 parameters differ from the cited source checkpoint")

    def to_payload(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self._core_payload()}

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operator_id": self.operator_id,
            "feature_order": list(self.feature_order),
            "threshold_rule": self.threshold_rule,
            "weights_hex": list(self.weights_hex),
            "bias_hex": self.bias_hex,
            "learning_rate_hex": self.learning_rate_hex,
            "max_abs_parameter_hex": self.max_abs_parameter_hex,
            "source_checkpoint_content_sha256": (
                self.source_checkpoint_content_sha256
            ),
            "source_batch_artifact_id": self.source_batch_artifact_id,
        }


@dataclass(frozen=True)
class RelationshipActionGateDecision:
    decision_id: str
    forecast_id: str
    gate_action: RelationshipGateAction
    selected_action_id: str
    recommended_action_id: str
    steer_probability: float
    features: tuple[float, ...]
    mode: RelationshipActionGateMode
    artifact_id: str
    artifact_version: int
    update_count: int
    evidence_refs: tuple[str, ...]
    rationale_codes: tuple[str, ...]
    evaluator_only: bool = False

    def __post_init__(self) -> None:
        for field_name, value in (
            ("decision_id", self.decision_id),
            ("forecast_id", self.forecast_id),
            ("selected_action_id", self.selected_action_id),
            ("recommended_action_id", self.recommended_action_id),
            ("artifact_id", self.artifact_id),
        ):
            _require_text(value, field_name)
        action_surface = {action.value for action in RELATIONSHIP_ACTIONS}
        if self.selected_action_id not in action_surface:
            raise ValueError("selected_action_id is outside the relationship action surface")
        if self.recommended_action_id not in action_surface:
            raise ValueError("recommended_action_id is outside the relationship action surface")
        if not 0.0 <= self.steer_probability <= 1.0:
            raise ValueError("steer_probability must be in [0, 1]")
        if len(self.features) != _FEATURE_COUNT or any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in self.features
        ):
            raise ValueError("features must be five finite values in [-1, 1]")
        if self.artifact_version < 1 or self.update_count < 0:
            raise ValueError("artifact_version must be >= 1 and update_count >= 0")
        _require_non_empty_unique(self.evidence_refs, "evidence_refs")
        _require_non_empty_unique(self.rationale_codes, "rationale_codes")
        if self.mode is RelationshipActionGateMode.ORACLE and not self.evaluator_only:
            raise ValueError("oracle decisions must be evaluator_only")
        if self.mode is not RelationshipActionGateMode.ORACLE and self.evaluator_only:
            raise ValueError("only oracle decisions may be evaluator_only")
        if (
            self.gate_action is RelationshipGateAction.NOOP
            and self.selected_action_id
            != RelationshipAction.NEUTRAL_NOOP.value
        ):
            raise ValueError("noop gate action must select neutral_noop")

    def to_payload(self) -> dict[str, object]:
        """Return the canonical JSON-compatible pending-decision payload."""

        return {
            "decision_id": self.decision_id,
            "forecast_id": self.forecast_id,
            "gate_action": self.gate_action.value,
            "selected_action_id": self.selected_action_id,
            "recommended_action_id": self.recommended_action_id,
            "steer_probability": self.steer_probability,
            "features": list(self.features),
            "mode": self.mode.value,
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "update_count": self.update_count,
            "evidence_refs": list(self.evidence_refs),
            "rationale_codes": list(self.rationale_codes),
            "evaluator_only": self.evaluator_only,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipActionGateDecision":
        """Restore one pending decision from its exact canonical shape."""

        raw = _require_exact_mapping(
            payload,
            expected={
                "decision_id",
                "forecast_id",
                "gate_action",
                "selected_action_id",
                "recommended_action_id",
                "steer_probability",
                "features",
                "mode",
                "artifact_id",
                "artifact_version",
                "update_count",
                "evidence_refs",
                "rationale_codes",
                "evaluator_only",
            },
            source="relationship action gate decision",
        )
        return cls(
            decision_id=_payload_text(raw, "decision_id"),
            forecast_id=_payload_text(raw, "forecast_id"),
            gate_action=RelationshipGateAction(
                _payload_text(raw, "gate_action")
            ),
            selected_action_id=_payload_text(raw, "selected_action_id"),
            recommended_action_id=_payload_text(
                raw,
                "recommended_action_id",
            ),
            steer_probability=_payload_float(raw, "steer_probability"),
            features=_payload_float_tuple(raw, "features"),
            mode=RelationshipActionGateMode(_payload_text(raw, "mode")),
            artifact_id=_payload_text(raw, "artifact_id"),
            artifact_version=_payload_int(raw, "artifact_version"),
            update_count=_payload_int(raw, "update_count"),
            evidence_refs=_payload_text_tuple(raw, "evidence_refs"),
            rationale_codes=_payload_text_tuple(raw, "rationale_codes"),
            evaluator_only=_payload_bool(raw, "evaluator_only"),
        )


@dataclass(frozen=True)
class RelationshipActionGateUpdate:
    credit_record_id: str
    forecast_id: str
    selected_action_id: str
    credit_value: float
    old_state_sha256: str
    new_state_sha256: str
    update_count: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("credit_record_id", self.credit_record_id),
            ("forecast_id", self.forecast_id),
            ("selected_action_id", self.selected_action_id),
            ("old_state_sha256", self.old_state_sha256),
            ("new_state_sha256", self.new_state_sha256),
        ):
            _require_text(value, field_name)
        if not math.isfinite(self.credit_value) or not -1.0 <= self.credit_value <= 1.0:
            raise ValueError("credit_value must be in [-1, 1]")
        if self.update_count < 1:
            raise ValueError("update_count must be >= 1")


@dataclass(frozen=True)
class RelationshipActionGateCheckpoint:
    artifact_id: str
    artifact_version: int
    weights: tuple[float, ...]
    bias: float
    update_count: int
    processed_credit_ids: tuple[str, ...]
    pending_decisions: tuple[RelationshipActionGateDecision, ...]
    schema_version: str = RELATIONSHIP_ACTION_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact_id")
        if self.artifact_version < 1 or self.update_count < 0:
            raise ValueError("checkpoint version/count is invalid")
        if len(self.weights) != _FEATURE_COUNT or any(
            not math.isfinite(value) for value in self.weights
        ):
            raise ValueError("checkpoint weights are invalid")
        if not math.isfinite(self.bias):
            raise ValueError("checkpoint bias must be finite")
        if len(set(self.processed_credit_ids)) != len(self.processed_credit_ids):
            raise ValueError("processed_credit_ids must be unique")
        forecast_ids = tuple(item.forecast_id for item in self.pending_decisions)
        if len(set(forecast_ids)) != len(forecast_ids):
            raise ValueError("pending decisions must have unique forecast ids")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_SCHEMA_VERSION:
            raise ValueError("relationship action gate checkpoint schema mismatch")

    @property
    def checkpoint_sha256(self) -> str:
        return _state_digest(self.weights, self.bias, self.update_count)

    @property
    def content_sha256(self) -> str:
        """Bind the full checkpoint without changing its historical payload."""

        return _canonical_sha256(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        """Return the canonical JSON-compatible full checkpoint payload.

        ``checkpoint_sha256`` intentionally remains the historical bounded
        parameter-state digest.  Callers that persist this full payload bind
        its canonical bytes in their containing artifact manifest.
        """

        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "weights": list(self.weights),
            "bias": self.bias,
            "update_count": self.update_count,
            "processed_credit_ids": list(self.processed_credit_ids),
            "pending_decisions": [
                decision.to_payload() for decision in self.pending_decisions
            ],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipActionGateCheckpoint":
        """Restore a checkpoint from canonical or legacy service payload.

        The closed-alpha service historically added ``content_sha256`` around
        this exact body.  That one legacy envelope is accepted and verified;
        every other missing or extra field fails loudly.
        """

        raw = _checkpoint_payload_body(payload)
        pending_raw = raw["pending_decisions"]
        if not isinstance(pending_raw, list):
            raise ValueError("pending_decisions must be an array of objects")
        pending = tuple(
            RelationshipActionGateDecision.from_payload(item)
            for item in pending_raw
        )
        processed = _payload_text_tuple(
            raw,
            "processed_credit_ids",
            allow_empty=True,
        )
        if processed != tuple(sorted(processed)):
            raise ValueError("processed_credit_ids must use canonical order")
        if tuple(item.forecast_id for item in pending) != tuple(
            sorted(item.forecast_id for item in pending)
        ):
            raise ValueError("pending_decisions must use canonical forecast order")
        return cls(
            artifact_id=_payload_text(raw, "artifact_id"),
            artifact_version=_payload_int(raw, "artifact_version"),
            weights=_payload_float_tuple(raw, "weights"),
            bias=_payload_float(raw, "bias"),
            update_count=_payload_int(raw, "update_count"),
            processed_credit_ids=processed,
            pending_decisions=pending,
            schema_version=_payload_text(raw, "schema_version"),
        )


@dataclass(frozen=True)
class RelationshipActionGateFrozenDecision:
    """Pure learned decision plus the exact checkpoint that produced it."""

    decision: RelationshipActionGateDecision
    checkpoint_content_sha256: str
    frozen_policy_id: str
    schema_version: str = RELATIONSHIP_ACTION_GATE_FROZEN_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RelationshipActionGateDecision):
            raise TypeError("decision must be RelationshipActionGateDecision")
        if self.decision.mode is not RelationshipActionGateMode.LEARNED:
            raise ValueError("frozen policy only emits learned gate decisions")
        if self.decision.evaluator_only:
            raise ValueError("frozen policy decision cannot be evaluator_only")
        _require_sha256(
            self.checkpoint_content_sha256,
            "checkpoint_content_sha256",
        )
        _require_text(self.frozen_policy_id, "frozen_policy_id")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_FROZEN_POLICY_SCHEMA_VERSION:
            raise ValueError("frozen decision schema_version mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_policy_id": self.frozen_policy_id,
            "checkpoint_content_sha256": self.checkpoint_content_sha256,
            "decision": self.decision.to_payload(),
        }


@dataclass(frozen=True)
class RelationshipActionGateFrozenPolicy:
    """Immutable decision surface; calling ``decide`` cannot create pending state."""

    artifact: RelationshipActionGateArtifact
    checkpoint: RelationshipActionGateCheckpoint
    random_seed: str
    theta0_artifact: RelationshipActionGateTheta0Artifact | None = None
    transition_batch: RelationshipActionGateCreditBatch | None = None
    transition_receipt: RelationshipActionGateBatchReceipt | None = None
    schema_version: str = RELATIONSHIP_ACTION_GATE_FROZEN_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, RelationshipActionGateArtifact):
            raise TypeError("artifact must be RelationshipActionGateArtifact")
        if not isinstance(self.checkpoint, RelationshipActionGateCheckpoint):
            raise TypeError("checkpoint must be RelationshipActionGateCheckpoint")
        _require_text(self.random_seed, "random_seed")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_FROZEN_POLICY_SCHEMA_VERSION:
            raise ValueError("frozen policy schema_version mismatch")
        _validate_checkpoint_against_artifact(self.checkpoint, self.artifact)
        if self.checkpoint.pending_decisions:
            raise ValueError("frozen policy requires an empty pending-decision set")
        if self.theta0_artifact is None:
            if self.artifact.artifact_id.startswith(_THETA0_ARTIFACT_PREFIX):
                raise ValueError("theta0 frozen policy requires its sidecar artifact")
            if self.transition_batch is not None or self.transition_receipt is not None:
                raise ValueError("legacy frozen policy cannot cite a theta0 transition")
        else:
            _validate_theta0_runtime_artifact(self.theta0_artifact, self.artifact)
            if self.checkpoint.update_count == 0:
                if self.transition_batch is not None or self.transition_receipt is not None:
                    raise ValueError("cold theta0 policy cannot cite an applied transition")
                _validate_theta0_checkpoint(
                    self.theta0_artifact,
                    self.checkpoint,
                    require_cold=True,
                )
            else:
                if not isinstance(
                    self.transition_batch,
                    RelationshipActionGateCreditBatch,
                ) or not isinstance(
                    self.transition_receipt,
                    RelationshipActionGateBatchReceipt,
                ):
                    raise ValueError(
                        "learned theta0 policy requires exact batch and APPLY receipt"
                    )
                replayed = RelationshipActionGate.from_applied_credit_batch(
                    self.theta0_artifact,
                    batch=self.transition_batch,
                    receipt=self.transition_receipt,
                    random_seed=self.random_seed,
                )
                if replayed.export_checkpoint() != self.checkpoint:
                    raise ValueError(
                        "learned theta0 policy checkpoint differs from exact batch replay"
                    )

    @property
    def policy_id(self) -> str:
        return (
            "relationship-action-gate-frozen-policy-sha256:"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def decide(
        self,
        forecast: PreferenceActionForecast,
    ) -> RelationshipActionGateFrozenDecision:
        gate = RelationshipActionGate(
            artifact=self.artifact,
            checkpoint=self.checkpoint,
            random_seed=self.random_seed,
        )
        decision = gate._decide(
            forecast,
            mode=RelationshipActionGateMode.LEARNED,
            oracle_action_id=None,
            evaluator_only=False,
            record_pending=False,
        )
        return RelationshipActionGateFrozenDecision(
            decision=decision,
            checkpoint_content_sha256=self.checkpoint.content_sha256,
            frozen_policy_id=self.policy_id,
        )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact.artifact_id,
            "artifact_version": self.artifact.artifact_version,
            "checkpoint_content_sha256": self.checkpoint.content_sha256,
            "random_seed": self.random_seed,
            "theta0_artifact_id": (
                self.theta0_artifact.artifact_id
                if self.theta0_artifact is not None
                else None
            ),
            "transition_batch_id": (
                self.transition_batch.batch_id
                if self.transition_batch is not None
                else None
            ),
            "transition_receipt_id": (
                self.transition_receipt.receipt_id
                if self.transition_receipt is not None
                else None
            ),
        }


@dataclass(frozen=True)
class RelationshipActionGateForcedExposure:
    """One forced learning opportunity observed under an unchanged theta0."""

    sequence_index: int
    forecast: PreferenceActionForecast
    frozen_decision: RelationshipActionGateFrozenDecision
    forced_action_id: str
    theta0_artifact_id: str
    schema_version: str = RELATIONSHIP_ACTION_GATE_FORCED_EXPOSURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.sequence_index, bool) or not isinstance(
            self.sequence_index,
            int,
        ):
            raise ValueError("sequence_index must be an integer")
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be >= 0")
        if not isinstance(self.forecast, PreferenceActionForecast):
            raise TypeError("forecast must be PreferenceActionForecast")
        if not isinstance(self.frozen_decision, RelationshipActionGateFrozenDecision):
            raise TypeError("frozen_decision must be RelationshipActionGateFrozenDecision")
        _require_text(self.forced_action_id, "forced_action_id")
        _require_text(self.theta0_artifact_id, "theta0_artifact_id")
        decision = self.frozen_decision.decision
        if (
            decision.forecast_id != self.forecast.forecast_id
            or decision.decision_id != self.forecast.decision_id
            or decision.recommended_action_id != self.forecast.recommended_action_id
        ):
            raise ValueError("forced exposure decision/forecast lineage mismatch")
        supported_actions = {
            decision.recommended_action_id,
            RelationshipAction.NEUTRAL_NOOP.value,
        }
        if self.forced_action_id not in supported_actions:
            raise ValueError(
                "forced action must be the owner recommendation or neutral_noop"
            )
        if self.theta0_artifact_id != decision.artifact_id:
            raise ValueError("forced exposure theta0 artifact lineage mismatch")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_FORCED_EXPOSURE_SCHEMA_VERSION:
            raise ValueError("forced exposure schema_version mismatch")

    @property
    def exposure_id(self) -> str:
        return (
            "relationship-action-gate-forced-exposure-sha256:"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def to_payload(self) -> dict[str, object]:
        return {"exposure_id": self.exposure_id, **self._core_payload()}

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence_index": self.sequence_index,
            "theta0_artifact_id": self.theta0_artifact_id,
            "forced_action_id": self.forced_action_id,
            "forecast": preference_action_forecast_to_payload(self.forecast),
            "frozen_decision": self.frozen_decision.to_payload(),
        }


@dataclass(frozen=True)
class RelationshipActionGateCreditBatch:
    """Chronological PE-credit batch shared byte-for-byte by both arms."""

    exposures: tuple[RelationshipActionGateForcedExposure, ...]
    credits: tuple[CreditRecord, ...]
    schema_version: str = RELATIONSHIP_ACTION_GATE_CREDIT_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.exposures or len(self.exposures) != len(self.credits):
            raise ValueError("credit batch requires equal non-empty exposures and credits")
        if self.schema_version != RELATIONSHIP_ACTION_GATE_CREDIT_BATCH_SCHEMA_VERSION:
            raise ValueError("credit batch schema_version mismatch")
        if tuple(item.sequence_index for item in self.exposures) != tuple(
            range(len(self.exposures))
        ):
            raise ValueError("credit batch exposures must use contiguous chronological order")
        base_hashes = {
            item.frozen_decision.checkpoint_content_sha256
            for item in self.exposures
        }
        artifact_ids = {item.theta0_artifact_id for item in self.exposures}
        policy_ids = {
            item.frozen_decision.frozen_policy_id for item in self.exposures
        }
        if len(base_hashes) != 1 or len(artifact_ids) != 1 or len(policy_ids) != 1:
            raise ValueError("credit batch exposures must share one cold theta0 policy")
        if len({item.exposure_id for item in self.exposures}) != len(self.exposures):
            raise ValueError("credit batch exposure_id values must be unique")
        decision_ids = tuple(
            item.frozen_decision.decision.decision_id for item in self.exposures
        )
        forecast_ids = tuple(
            item.frozen_decision.decision.forecast_id for item in self.exposures
        )
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("credit batch decision_id values must be unique")
        if len(set(forecast_ids)) != len(forecast_ids):
            raise ValueError("credit batch forecast_id values must be unique")
        credit_ids: list[str] = []
        timestamps: list[int] = []
        for exposure, credit in zip(self.exposures, self.credits, strict=True):
            _validate_relationship_credit_record(
                credit,
                require_complete_lineage=True,
            )
            decision = exposure.frozen_decision.decision
            if credit.prediction_id != decision.forecast_id:
                raise ValueError("credit batch prediction lineage mismatch")
            if credit.abstract_action_id != exposure.forced_action_id:
                raise ValueError("credit batch forced-action lineage mismatch")
            credit_ids.append(credit.record_id)
            timestamps.append(credit.timestamp_ms)
        if len(set(credit_ids)) != len(credit_ids):
            raise ValueError("credit batch record_id values must be unique")
        if timestamps != sorted(timestamps):
            raise ValueError("credit batch timestamps must be chronological")

    @property
    def base_checkpoint_content_sha256(self) -> str:
        return self.exposures[0].frozen_decision.checkpoint_content_sha256

    @property
    def theta0_artifact_id(self) -> str:
        return self.exposures[0].theta0_artifact_id

    @property
    def batch_id(self) -> str:
        return (
            "relationship-action-gate-credit-batch-sha256:"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def to_payload(self) -> dict[str, object]:
        return {"batch_id": self.batch_id, **self._core_payload()}

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "theta0_artifact_id": self.theta0_artifact_id,
            "base_checkpoint_content_sha256": self.base_checkpoint_content_sha256,
            "entries": [
                {
                    "exposure": exposure.to_payload(),
                    "credit": _credit_record_payload(credit),
                }
                for exposure, credit in zip(self.exposures, self.credits, strict=True)
            ],
        }


@dataclass(frozen=True)
class RelationshipActionGateBatchPlan:
    """Pure candidate transition; constructing it never mutates the gate."""

    batch: RelationshipActionGateCreditBatch
    pre_checkpoint: RelationshipActionGateCheckpoint
    candidate_checkpoint: RelationshipActionGateCheckpoint
    schema_version: str = RELATIONSHIP_ACTION_GATE_BATCH_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.batch, RelationshipActionGateCreditBatch):
            raise TypeError("batch must be RelationshipActionGateCreditBatch")
        if not isinstance(self.pre_checkpoint, RelationshipActionGateCheckpoint):
            raise TypeError("pre_checkpoint must be RelationshipActionGateCheckpoint")
        if not isinstance(
            self.candidate_checkpoint,
            RelationshipActionGateCheckpoint,
        ):
            raise TypeError(
                "candidate_checkpoint must be RelationshipActionGateCheckpoint"
            )
        if self.schema_version != RELATIONSHIP_ACTION_GATE_BATCH_PLAN_SCHEMA_VERSION:
            raise ValueError("batch plan schema_version mismatch")
        if self.pre_checkpoint.content_sha256 != self.batch.base_checkpoint_content_sha256:
            raise ValueError("batch plan pre-checkpoint does not match batch base")
        if self.pre_checkpoint.artifact_id != self.batch.theta0_artifact_id:
            raise ValueError("batch plan theta0 artifact mismatch")
        if self.candidate_checkpoint.artifact_id != self.pre_checkpoint.artifact_id:
            raise ValueError("batch plan candidate artifact mismatch")
        if (
            self.candidate_checkpoint.artifact_version
            != self.pre_checkpoint.artifact_version
        ):
            raise ValueError("batch plan candidate artifact version mismatch")
        if self.pre_checkpoint.pending_decisions:
            raise ValueError("batch plan requires no pending legacy decisions")
        if self.candidate_checkpoint.pending_decisions:
            raise ValueError("batch plan candidate cannot contain pending decisions")
        if self.candidate_checkpoint.update_count != (
            self.pre_checkpoint.update_count + len(self.batch.credits)
        ):
            raise ValueError("batch plan candidate update_count mismatch")
        expected_credit_ids = set(self.pre_checkpoint.processed_credit_ids) | {
            credit.record_id for credit in self.batch.credits
        }
        if set(self.candidate_checkpoint.processed_credit_ids) != expected_credit_ids:
            raise ValueError("batch plan candidate processed-credit set mismatch")

    @property
    def plan_id(self) -> str:
        return (
            "relationship-action-gate-batch-plan-sha256:"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch.batch_id,
            "pre_checkpoint_content_sha256": self.pre_checkpoint.content_sha256,
            "candidate_checkpoint_content_sha256": (
                self.candidate_checkpoint.content_sha256
            ),
        }


class RelationshipActionGateBatchDisposition(str, Enum):
    APPLY = "apply"
    WITHHOLD = "withhold"


@dataclass(frozen=True)
class RelationshipActionGateBatchReceipt:
    batch_id: str
    plan_id: str
    disposition: RelationshipActionGateBatchDisposition
    pre_checkpoint_content_sha256: str
    candidate_checkpoint_content_sha256: str
    post_checkpoint_content_sha256: str
    credit_count: int
    weight_delta: tuple[float, ...]
    bias_delta: float
    update_count_delta: int
    atomic_commit_count: int
    applied_credit_ids: tuple[str, ...]
    schema_version: str = RELATIONSHIP_ACTION_GATE_BATCH_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, RelationshipActionGateBatchDisposition):
            raise TypeError("disposition must be RelationshipActionGateBatchDisposition")
        for field_name, value in (
            ("batch_id", self.batch_id),
            ("plan_id", self.plan_id),
        ):
            _require_text(value, field_name)
        for field_name, value in (
            ("pre_checkpoint_content_sha256", self.pre_checkpoint_content_sha256),
            (
                "candidate_checkpoint_content_sha256",
                self.candidate_checkpoint_content_sha256,
            ),
            ("post_checkpoint_content_sha256", self.post_checkpoint_content_sha256),
        ):
            _require_sha256(value, field_name)
        if self.schema_version != RELATIONSHIP_ACTION_GATE_BATCH_RECEIPT_SCHEMA_VERSION:
            raise ValueError("batch receipt schema_version mismatch")
        if isinstance(self.credit_count, bool) or not isinstance(self.credit_count, int):
            raise ValueError("batch receipt credit_count must be an integer")
        if self.credit_count < 1:
            raise ValueError("batch receipt credit_count must be >= 1")
        if len(self.weight_delta) != _FEATURE_COUNT or any(
            isinstance(value, bool) or not math.isfinite(value)
            for value in self.weight_delta
        ):
            raise ValueError("batch receipt weight_delta is invalid")
        if isinstance(self.bias_delta, bool) or not math.isfinite(self.bias_delta):
            raise ValueError("batch receipt bias_delta must be finite")
        for field_name, value in (
            ("update_count_delta", self.update_count_delta),
            ("atomic_commit_count", self.atomic_commit_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"batch receipt {field_name} must be an integer")
        if self.disposition is RelationshipActionGateBatchDisposition.APPLY:
            if self.post_checkpoint_content_sha256 != (
                self.candidate_checkpoint_content_sha256
            ):
                raise ValueError("applied batch post checkpoint must equal candidate")
            if self.update_count_delta != self.credit_count:
                raise ValueError("applied batch update_count_delta mismatch")
            if self.atomic_commit_count != 1:
                raise ValueError("applied batch must have one atomic commit")
            if len(self.applied_credit_ids) != self.credit_count:
                raise ValueError("applied batch credit-id count mismatch")
            _require_non_empty_unique(
                self.applied_credit_ids,
                "applied_credit_ids",
            )
        elif self.disposition is RelationshipActionGateBatchDisposition.WITHHOLD:
            if self.post_checkpoint_content_sha256 != self.pre_checkpoint_content_sha256:
                raise ValueError("withheld batch must preserve the exact checkpoint")
            if self.update_count_delta != 0 or self.atomic_commit_count != 0:
                raise ValueError("withheld batch cannot update or commit")
            if self.applied_credit_ids:
                raise ValueError("withheld batch cannot report applied credits")
            if any(value != 0.0 for value in (*self.weight_delta, self.bias_delta)):
                raise ValueError("withheld batch parameter delta must be zero")
        else:
            raise ValueError("unknown batch disposition")

    @property
    def receipt_id(self) -> str:
        return (
            "relationship-action-gate-batch-receipt-sha256:"
            f"{_canonical_sha256(self.to_payload(include_receipt_id=False))}"
        )

    def to_payload(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "plan_id": self.plan_id,
            "disposition": self.disposition.value,
            "pre_checkpoint_content_sha256": self.pre_checkpoint_content_sha256,
            "candidate_checkpoint_content_sha256": (
                self.candidate_checkpoint_content_sha256
            ),
            "post_checkpoint_content_sha256": self.post_checkpoint_content_sha256,
            "credit_count": self.credit_count,
            "weight_delta": list(self.weight_delta),
            "bias_delta": self.bias_delta,
            "update_count_delta": self.update_count_delta,
            "atomic_commit_count": self.atomic_commit_count,
            "applied_credit_ids": list(self.applied_credit_ids),
        }
        if include_receipt_id:
            return {"receipt_id": self.receipt_id, **payload}
        return payload


@dataclass(frozen=True)
class _RelationshipActionGateRuntimeState:
    weights: tuple[float, ...]
    bias: float
    update_count: int
    processed_credit_ids: frozenset[str]
    pending_decisions: tuple[RelationshipActionGateDecision, ...]
    applied_batch: RelationshipActionGateCreditBatch | None = None
    applied_batch_receipt: RelationshipActionGateBatchReceipt | None = None


class RelationshipActionGate:
    """Session-medium bounded Bernoulli gate over a frozen forecast reader."""

    def __init__(
        self,
        *,
        artifact: RelationshipActionGateArtifact | None = None,
        checkpoint: RelationshipActionGateCheckpoint | None = None,
        random_seed: str = "relationship-action-random-control-v1",
    ) -> None:
        self._artifact = artifact or RelationshipActionGateArtifact()
        self._theta0_artifact: RelationshipActionGateTheta0Artifact | None = None
        _require_text(random_seed, "random_seed")
        self._random_seed = random_seed
        self._state = _RelationshipActionGateRuntimeState(
            weights=self._artifact.initial_weights,
            bias=self._artifact.initial_bias,
            update_count=0,
            processed_credit_ids=frozenset(),
            pending_decisions=(),
        )
        if checkpoint is not None:
            self.restore(checkpoint)

    @classmethod
    def from_theta0(
        cls,
        theta0_artifact: RelationshipActionGateTheta0Artifact,
        *,
        checkpoint: RelationshipActionGateCheckpoint | None = None,
        random_seed: str = "relationship-action-random-control-v1",
    ) -> "RelationshipActionGate":
        """Opt a new experiment into a non-zero content-addressed theta0."""

        if not isinstance(theta0_artifact, RelationshipActionGateTheta0Artifact):
            raise TypeError("theta0_artifact must be RelationshipActionGateTheta0Artifact")
        runtime_artifact = theta0_artifact.to_runtime_artifact()
        gate = cls(
            artifact=runtime_artifact,
            checkpoint=checkpoint,
            random_seed=random_seed,
        )
        gate._theta0_artifact = theta0_artifact
        _validate_theta0_runtime_artifact(theta0_artifact, runtime_artifact)
        _validate_theta0_checkpoint(
            theta0_artifact,
            gate.export_checkpoint(),
            require_cold=True,
        )
        return gate

    @classmethod
    def from_applied_credit_batch(
        cls,
        theta0_artifact: RelationshipActionGateTheta0Artifact,
        *,
        batch: RelationshipActionGateCreditBatch,
        receipt: RelationshipActionGateBatchReceipt,
        random_seed: str = "relationship-action-random-control-v1",
    ) -> "RelationshipActionGate":
        """Rebuild a learned state only by replaying its exact applied batch."""

        if not isinstance(receipt, RelationshipActionGateBatchReceipt):
            raise TypeError("receipt must be RelationshipActionGateBatchReceipt")
        if receipt.disposition is not RelationshipActionGateBatchDisposition.APPLY:
            raise ValueError("only an applied batch receipt can restore learned theta0")
        gate = cls.from_theta0(theta0_artifact, random_seed=random_seed)
        plan = gate.plan_credit_batch(batch)
        replayed_receipt = gate.commit_credit_batch(
            plan,
            disposition=RelationshipActionGateBatchDisposition.APPLY,
        )
        if receipt != replayed_receipt:
            raise ValueError("applied batch receipt does not match exact owner replay")
        return gate

    @property
    def update_count(self) -> int:
        return self._state.update_count

    @property
    def parameter_state(self) -> tuple[tuple[float, ...], float]:
        return self._state.weights, self._state.bias

    def decide(
        self,
        forecast: PreferenceActionForecast,
        *,
        mode: RelationshipActionGateMode = RelationshipActionGateMode.LEARNED,
        oracle_action_id: str | None = None,
        evaluator_only: bool = False,
    ) -> RelationshipActionGateDecision:
        return self._decide(
            forecast,
            mode=mode,
            oracle_action_id=oracle_action_id,
            evaluator_only=evaluator_only,
            record_pending=True,
        )

    def _decide(
        self,
        forecast: PreferenceActionForecast,
        *,
        mode: RelationshipActionGateMode,
        oracle_action_id: str | None,
        evaluator_only: bool,
        record_pending: bool,
    ) -> RelationshipActionGateDecision:
        features = relationship_action_gate_features(forecast)
        probability = _sigmoid(
            math.fsum(
                weight * value
                for weight, value in zip(self._state.weights, features, strict=True)
            )
            + self._state.bias
        )
        gate_action: RelationshipGateAction
        selected_action_id: str
        rationale: tuple[str, ...]
        if mode is RelationshipActionGateMode.NOOP:
            _reject_oracle_inputs(oracle_action_id, evaluator_only)
            gate_action = RelationshipGateAction.NOOP
            selected_action_id = RelationshipAction.NEUTRAL_NOOP.value
            rationale = ("control:no-op",)
        elif mode is RelationshipActionGateMode.ALWAYS:
            _reject_oracle_inputs(oracle_action_id, evaluator_only)
            gate_action = RelationshipGateAction.STEER
            selected_action_id = forecast.recommended_action_id
            rationale = ("control:always-adopt-owner-forecast",)
        elif mode is RelationshipActionGateMode.RANDOM:
            _reject_oracle_inputs(oracle_action_id, evaluator_only)
            draw = _deterministic_draw(self._random_seed, forecast.decision_id)
            gate_action = (
                RelationshipGateAction.STEER
                if draw < 0.5
                else RelationshipGateAction.NOOP
            )
            selected_action_id = (
                forecast.recommended_action_id
                if gate_action is RelationshipGateAction.STEER
                else RelationshipAction.NEUTRAL_NOOP.value
            )
            rationale = ("control:deterministic-random", f"draw_bucket:{int(draw * 1000)}")
        elif mode is RelationshipActionGateMode.ORACLE:
            if not evaluator_only:
                raise ValueError("oracle mode requires evaluator_only=True")
            if oracle_action_id is None:
                raise ValueError("oracle mode requires oracle_action_id")
            try:
                selected_action_id = RelationshipAction(oracle_action_id).value
            except ValueError as exc:
                raise ValueError("oracle_action_id is outside the action surface") from exc
            gate_action = (
                RelationshipGateAction.NOOP
                if selected_action_id == RelationshipAction.NEUTRAL_NOOP.value
                else RelationshipGateAction.STEER
            )
            rationale = ("control:evaluator-only-oracle",)
        else:
            _reject_oracle_inputs(oracle_action_id, evaluator_only)
            gate_action = (
                RelationshipGateAction.STEER
                if probability > 0.5
                else RelationshipGateAction.NOOP
            )
            selected_action_id = (
                forecast.recommended_action_id
                if gate_action is RelationshipGateAction.STEER
                else RelationshipAction.NEUTRAL_NOOP.value
            )
            rationale = (
                "policy:bounded-logistic-gate",
                "inputs:typed-owner-forecast-only",
                "learning:pe-credit-only",
            )
        decision = RelationshipActionGateDecision(
            decision_id=forecast.decision_id,
            forecast_id=forecast.forecast_id,
            gate_action=gate_action,
            selected_action_id=selected_action_id,
            recommended_action_id=forecast.recommended_action_id,
            steer_probability=probability,
            features=features,
            mode=mode,
            artifact_id=self._artifact.artifact_id,
            artifact_version=self._artifact.artifact_version,
            update_count=self._state.update_count,
            evidence_refs=(
                *dict.fromkeys(
                    (
                        *forecast.source_record_ids,
                        *forecast.evidence,
                    )
                ),
            ),
            rationale_codes=rationale,
            evaluator_only=evaluator_only,
        )
        if mode is RelationshipActionGateMode.LEARNED and record_pending:
            pending = {
                item.forecast_id: item for item in self._state.pending_decisions
            }
            existing = pending.get(forecast.forecast_id)
            if existing is not None and existing != decision:
                raise ValueError("forecast already has a different pending gate decision")
            if existing is None:
                pending[forecast.forecast_id] = decision
                self._state = replace(
                    self._state,
                    pending_decisions=tuple(pending[key] for key in sorted(pending)),
                )
        return decision

    def freeze_for_evaluation(self) -> RelationshipActionGateFrozenPolicy:
        """Publish a pure immutable policy without freezing legacy APIs."""

        return RelationshipActionGateFrozenPolicy(
            artifact=self._artifact,
            checkpoint=self.export_checkpoint(),
            random_seed=self._random_seed,
            theta0_artifact=self._theta0_artifact,
            transition_batch=self._state.applied_batch,
            transition_receipt=self._state.applied_batch_receipt,
        )

    def validate_frozen_theta0(self) -> RelationshipActionGateCheckpoint:
        """Fail closed unless this gate is the exact cold non-zero theta0."""

        if self._theta0_artifact is None:
            raise ValueError("gate was not instantiated from a theta0 artifact")
        checkpoint = self.export_checkpoint()
        _validate_theta0_runtime_artifact(self._theta0_artifact, self._artifact)
        _validate_theta0_checkpoint(
            self._theta0_artifact,
            checkpoint,
            require_cold=True,
        )
        return checkpoint

    def record_forced_exposure(
        self,
        forecast: PreferenceActionForecast,
        *,
        forced_action_id: str,
        sequence_index: int,
    ) -> RelationshipActionGateForcedExposure:
        """Record a cold-policy opportunity without pending state or mutation."""

        before = self.validate_frozen_theta0()
        frozen_decision = self.freeze_for_evaluation().decide(forecast)
        exposure = RelationshipActionGateForcedExposure(
            sequence_index=sequence_index,
            forecast=forecast,
            frozen_decision=frozen_decision,
            forced_action_id=forced_action_id,
            theta0_artifact_id=self._artifact.artifact_id,
        )
        if self.export_checkpoint() != before:
            raise RuntimeError("forced exposure changed relationship action gate state")
        return exposure

    def plan_credit_batch(
        self,
        batch: RelationshipActionGateCreditBatch,
    ) -> RelationshipActionGateBatchPlan:
        """Validate a full PE batch and compute its candidate state purely."""

        if not isinstance(batch, RelationshipActionGateCreditBatch):
            raise TypeError("batch must be RelationshipActionGateCreditBatch")
        pre = self.validate_frozen_theta0()
        if batch.theta0_artifact_id != self._artifact.artifact_id:
            raise ValueError("credit batch theta0 artifact mismatch")
        if batch.base_checkpoint_content_sha256 != pre.content_sha256:
            raise ValueError("credit batch base checkpoint is stale")
        if any(
            credit.record_id in self._state.processed_credit_ids
            for credit in batch.credits
        ):
            raise ValueError("relationship action credit was already processed")
        frozen_policy = self.freeze_for_evaluation()
        for exposure in batch.exposures:
            if frozen_policy.decide(exposure.forecast) != exposure.frozen_decision:
                raise ValueError(
                    "credit batch contains a forged or stale frozen decision"
                )

        weights = pre.weights
        bias = pre.bias
        cap = self._artifact.max_abs_parameter
        for exposure, credit in zip(batch.exposures, batch.credits, strict=True):
            decision = exposure.frozen_decision.decision
            action_indicator = (
                0.0
                if exposure.forced_action_id
                == RelationshipAction.NEUTRAL_NOOP.value
                else 1.0
            )
            gradient_scale = (
                self._artifact.learning_rate
                * credit.credit_value
                * (action_indicator - decision.steer_probability)
            )
            weights = tuple(
                max(-cap, min(cap, weight + gradient_scale * feature))
                for weight, feature in zip(
                    weights,
                    decision.features,
                    strict=True,
                )
            )
            bias = max(-cap, min(cap, bias + gradient_scale))
        candidate = RelationshipActionGateCheckpoint(
            artifact_id=pre.artifact_id,
            artifact_version=pre.artifact_version,
            weights=weights,
            bias=bias,
            update_count=pre.update_count + len(batch.credits),
            processed_credit_ids=tuple(
                sorted(
                    {
                        *pre.processed_credit_ids,
                        *(credit.record_id for credit in batch.credits),
                    }
                )
            ),
            pending_decisions=(),
        )
        return RelationshipActionGateBatchPlan(
            batch=batch,
            pre_checkpoint=pre,
            candidate_checkpoint=candidate,
        )

    def commit_credit_batch(
        self,
        plan: RelationshipActionGateBatchPlan,
        *,
        disposition: RelationshipActionGateBatchDisposition,
    ) -> RelationshipActionGateBatchReceipt:
        """Commit a prevalidated batch once, or preserve the exact cold state."""

        if not isinstance(plan, RelationshipActionGateBatchPlan):
            raise TypeError("plan must be RelationshipActionGateBatchPlan")
        if not isinstance(disposition, RelationshipActionGateBatchDisposition):
            raise TypeError("disposition must be RelationshipActionGateBatchDisposition")
        expected_plan = self.plan_credit_batch(plan.batch)
        if plan != expected_plan:
            raise ValueError("batch plan does not match the current pure transition")
        pre = plan.pre_checkpoint
        candidate = plan.candidate_checkpoint
        if disposition is RelationshipActionGateBatchDisposition.APPLY:
            post = candidate
            weight_delta = tuple(
                after - before
                for before, after in zip(pre.weights, post.weights, strict=True)
            )
            bias_delta = post.bias - pre.bias
            update_count_delta = post.update_count - pre.update_count
            atomic_commit_count = 1
            applied_credit_ids = tuple(
                credit.record_id for credit in plan.batch.credits
            )
        else:
            post = pre
            weight_delta = (0.0,) * _FEATURE_COUNT
            bias_delta = 0.0
            update_count_delta = 0
            atomic_commit_count = 0
            applied_credit_ids = ()
        receipt = RelationshipActionGateBatchReceipt(
            batch_id=plan.batch.batch_id,
            plan_id=plan.plan_id,
            disposition=disposition,
            pre_checkpoint_content_sha256=pre.content_sha256,
            candidate_checkpoint_content_sha256=candidate.content_sha256,
            post_checkpoint_content_sha256=post.content_sha256,
            credit_count=len(plan.batch.credits),
            weight_delta=weight_delta,
            bias_delta=bias_delta,
            update_count_delta=update_count_delta,
            atomic_commit_count=atomic_commit_count,
            applied_credit_ids=applied_credit_ids,
        )
        if disposition is RelationshipActionGateBatchDisposition.APPLY:
            self._state = replace(
                _runtime_state_from_checkpoint(post),
                applied_batch=plan.batch,
                applied_batch_receipt=receipt,
            )
        return receipt

    def observe_credit(self, credit: CreditRecord) -> RelationshipActionGateUpdate:
        _validate_relationship_credit_record(credit)
        if credit.record_id in self._state.processed_credit_ids:
            raise ValueError("relationship action credit was already processed")
        pending = {
            item.forecast_id: item for item in self._state.pending_decisions
        }
        try:
            decision = pending[credit.prediction_id]
        except KeyError as exc:
            raise ValueError("relationship action credit has no pending learned decision") from exc
        if credit.abstract_action_id != decision.selected_action_id:
            raise ValueError("relationship action credit action lineage mismatch")
        old_digest = _state_digest(
            self._state.weights,
            self._state.bias,
            self._state.update_count,
        )
        action_indicator = (
            1.0 if decision.gate_action is RelationshipGateAction.STEER else 0.0
        )
        gradient_scale = (
            self._artifact.learning_rate
            * credit.credit_value
            * (action_indicator - decision.steer_probability)
        )
        cap = self._artifact.max_abs_parameter
        weights = tuple(
            max(-cap, min(cap, weight + gradient_scale * feature))
            for weight, feature in zip(
                self._state.weights,
                decision.features,
                strict=True,
            )
        )
        bias = max(-cap, min(cap, self._state.bias + gradient_scale))
        update_count = self._state.update_count + 1
        del pending[credit.prediction_id]
        candidate_state = _RelationshipActionGateRuntimeState(
            weights=weights,
            bias=bias,
            update_count=update_count,
            processed_credit_ids=(
                self._state.processed_credit_ids | {credit.record_id}
            ),
            pending_decisions=tuple(pending[key] for key in sorted(pending)),
            applied_batch=None,
            applied_batch_receipt=None,
        )
        new_digest = _state_digest(weights, bias, update_count)
        update = RelationshipActionGateUpdate(
            credit_record_id=credit.record_id,
            forecast_id=credit.prediction_id,
            selected_action_id=credit.abstract_action_id,
            credit_value=credit.credit_value,
            old_state_sha256=old_digest,
            new_state_sha256=new_digest,
            update_count=update_count,
        )
        self._state = candidate_state
        return update

    def export_checkpoint(self) -> RelationshipActionGateCheckpoint:
        return RelationshipActionGateCheckpoint(
            artifact_id=self._artifact.artifact_id,
            artifact_version=self._artifact.artifact_version,
            weights=self._state.weights,
            bias=self._state.bias,
            update_count=self._state.update_count,
            processed_credit_ids=tuple(sorted(self._state.processed_credit_ids)),
            pending_decisions=self._state.pending_decisions,
        )

    def restore(self, checkpoint: RelationshipActionGateCheckpoint) -> None:
        if not isinstance(checkpoint, RelationshipActionGateCheckpoint):
            raise TypeError("checkpoint must be RelationshipActionGateCheckpoint")
        _validate_checkpoint_against_artifact(checkpoint, self._artifact)
        if self._theta0_artifact is not None:
            _validate_theta0_checkpoint(
                self._theta0_artifact,
                checkpoint,
                require_cold=True,
            )
        candidate_state = _runtime_state_from_checkpoint(checkpoint)
        self._state = candidate_state


def relationship_action_gate_features(
    forecast: PreferenceActionForecast,
) -> tuple[float, ...]:
    """Read five bounded features from the frozen forecast, never raw text."""

    expected_actions = tuple(action.value for action in RELATIONSHIP_ACTIONS)
    observed_actions = tuple(
        candidate.action_id for candidate in forecast.candidate_predictions
    )
    if set(observed_actions) != set(expected_actions) or len(observed_actions) != len(
        expected_actions
    ):
        raise ValueError("forecast action surface does not match relationship v1")
    expected_outcomes = tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES)
    by_action = {
        candidate.action_id: candidate for candidate in forecast.candidate_predictions
    }
    positive_mass: dict[str, float] = {}
    for action_id, candidate in by_action.items():
        if tuple(item.outcome_id for item in candidate.outcomes) != expected_outcomes:
            raise ValueError("forecast outcome surface does not match relationship v1")
        positive_mass[action_id] = math.fsum(
            item.probability
            for item in candidate.outcomes
            if item.outcome_id in _POSITIVE_OUTCOMES
        )
    recommended = by_action[forecast.recommended_action_id]
    entropy = -math.fsum(
        item.probability * math.log(max(item.probability, 1e-12))
        for item in recommended.outcomes
    )
    certainty = 1.0 - entropy / math.log(len(recommended.outcomes))
    positive_margin_over_noop = positive_mass[forecast.recommended_action_id] - positive_mass[
        RelationshipAction.NEUTRAL_NOOP.value
    ]
    support = min(1.0, len(forecast.source_record_ids) / 4.0)
    return (
        forecast.confidence,
        positive_mass[forecast.recommended_action_id],
        max(-1.0, min(1.0, positive_margin_over_noop)),
        max(0.0, min(1.0, certainty)),
        support,
    )


def temporal_action_advisory_from_gate_decision(
    decision: RelationshipActionGateDecision,
) -> TemporalActionAdvisoryProposal:
    """Translate a gate decision into the self-temporal collaborator contract.

    P3/P4 decisions are deliberately not ACTIVE-authorized.  ``self_temporal``
    can therefore record them under SHADOW, while an accidental ACTIVE wiring
    fails loudly instead of changing user-visible expression.
    """

    confidence = (
        decision.steer_probability
        if decision.gate_action is RelationshipGateAction.STEER
        else 1.0 - decision.steer_probability
    )
    return TemporalActionAdvisoryProposal(
        advisory_id=f"relationship-action-advisory:{decision.decision_id}",
        decision_id=decision.decision_id,
        prediction_id=decision.forecast_id,
        action_id=decision.selected_action_id,
        confidence=confidence,
        policy_artifact_id=decision.artifact_id,
        policy_artifact_version=decision.artifact_version,
        evidence_refs=decision.evidence_refs,
        rationale_codes=(
            *decision.rationale_codes,
            f"gate:{decision.gate_action.value}",
            f"mode:{decision.mode.value}",
        ),
        evaluator_only=decision.evaluator_only,
        active_authorized=False,
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a canonical lowercase SHA-256")


def _require_content_addressed_id(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    prefix, separator, digest = value.rpartition(":")
    if (
        not separator
        or not prefix
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field_name} must end in a canonical SHA-256")


def _finite_float_hex(value: object, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric.hex()


def _finite_float_from_hex(value: str, field_name: str) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a canonical finite float hex string")
    try:
        numeric = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a canonical finite float hex string"
        ) from exc
    if not math.isfinite(numeric) or numeric.hex() != value:
        raise ValueError(f"{field_name} must be canonical and finite")
    return numeric


def _validate_checkpoint_against_artifact(
    checkpoint: RelationshipActionGateCheckpoint,
    artifact: RelationshipActionGateArtifact,
) -> None:
    if checkpoint.artifact_id != artifact.artifact_id:
        raise ValueError("relationship action gate artifact_id mismatch")
    if checkpoint.artifact_version != artifact.artifact_version:
        raise ValueError("relationship action gate artifact_version mismatch")
    cap = artifact.max_abs_parameter
    if any(abs(value) > cap for value in (*checkpoint.weights, checkpoint.bias)):
        raise ValueError("relationship action gate checkpoint exceeds parameter cap")


def _validate_theta0_runtime_artifact(
    theta0_artifact: RelationshipActionGateTheta0Artifact,
    runtime_artifact: RelationshipActionGateArtifact,
) -> None:
    if not isinstance(theta0_artifact, RelationshipActionGateTheta0Artifact):
        raise TypeError("theta0_artifact must be RelationshipActionGateTheta0Artifact")
    if not isinstance(runtime_artifact, RelationshipActionGateArtifact):
        raise TypeError("runtime_artifact must be RelationshipActionGateArtifact")
    expected = theta0_artifact.to_runtime_artifact()
    if runtime_artifact != expected:
        raise ValueError("theta0 runtime artifact does not match content-addressed source")


def _validate_theta0_checkpoint(
    theta0_artifact: RelationshipActionGateTheta0Artifact,
    checkpoint: RelationshipActionGateCheckpoint,
    *,
    require_cold: bool,
) -> None:
    runtime_artifact = theta0_artifact.to_runtime_artifact()
    _validate_checkpoint_against_artifact(checkpoint, runtime_artifact)
    if checkpoint.pending_decisions:
        raise ValueError("theta0 experiment checkpoint cannot contain pending decisions")
    if len(checkpoint.processed_credit_ids) != checkpoint.update_count:
        raise ValueError("theta0 checkpoint credit/update counts do not match")
    if require_cold:
        if checkpoint.update_count != 0 or checkpoint.processed_credit_ids:
            raise ValueError("theta0 cold checkpoint has already consumed credit")
        if (
            checkpoint.weights != runtime_artifact.initial_weights
            or checkpoint.bias != runtime_artifact.initial_bias
        ):
            raise ValueError("theta0 cold checkpoint parameters differ from the artifact")


def _runtime_state_from_checkpoint(
    checkpoint: RelationshipActionGateCheckpoint,
) -> _RelationshipActionGateRuntimeState:
    return _RelationshipActionGateRuntimeState(
        weights=checkpoint.weights,
        bias=checkpoint.bias,
        update_count=checkpoint.update_count,
        processed_credit_ids=frozenset(checkpoint.processed_credit_ids),
        pending_decisions=tuple(
            sorted(checkpoint.pending_decisions, key=lambda item: item.forecast_id)
        ),
    )


def _validate_relationship_credit_record(
    credit: CreditRecord,
    *,
    require_complete_lineage: bool = False,
) -> None:
    if not isinstance(credit, CreditRecord):
        raise TypeError("credit must be CreditRecord")
    if credit.level != RELATIONSHIP_ACTION_CREDIT_LEVEL:
        raise ValueError(
            "relationship action gate only accepts PE-derived relationship credit"
        )
    if credit.track is not Track.SELF:
        raise ValueError("relationship action credit must be self-track")
    if not math.isfinite(credit.credit_value) or not -1.0 <= credit.credit_value <= 1.0:
        raise ValueError("relationship action credit_value must be finite and in [-1, 1]")
    if not require_complete_lineage:
        return
    for field_name, value in (
        ("record_id", credit.record_id),
        ("source_event", credit.source_event),
        ("context", credit.context),
        ("prediction_id", credit.prediction_id),
        ("environment_outcome_id", credit.environment_outcome_id),
        ("abstract_action_id", credit.abstract_action_id),
    ):
        _require_text(value, f"credit.{field_name}")
    if not credit.record_id.startswith(_RELATIONSHIP_ACTION_CREDIT_RECORD_PREFIX):
        raise ValueError("relationship action credit record_id bypassed its owner")
    record_suffix = credit.record_id.removeprefix(
        _RELATIONSHIP_ACTION_CREDIT_RECORD_PREFIX
    )
    if not credit.source_event.startswith(_RELATIONSHIP_ACTION_SOCIAL_PE_SOURCE_PREFIX):
        raise ValueError("relationship action credit bypassed the social PE owner")
    source_suffix = credit.source_event.removeprefix(
        _RELATIONSHIP_ACTION_SOCIAL_PE_SOURCE_PREFIX
    )
    if not record_suffix or record_suffix != source_suffix:
        raise ValueError("relationship action credit record/social-PE lineage mismatch")
    if isinstance(credit.timestamp_ms, bool) or not isinstance(credit.timestamp_ms, int):
        raise ValueError("credit.timestamp_ms must be an integer")
    if credit.timestamp_ms < 0:
        raise ValueError("credit.timestamp_ms must be >= 0")
    action_surface = {action.value for action in RELATIONSHIP_ACTIONS}
    if credit.abstract_action_id not in action_surface:
        raise ValueError("relationship action credit action is outside the action surface")
    if any(not isinstance(value, str) or not value.strip() for value in credit.conditioning_bank_set):
        raise ValueError("credit.conditioning_bank_set must contain non-empty strings")
    if len(set(credit.conditioning_bank_set)) != len(credit.conditioning_bank_set):
        raise ValueError("credit.conditioning_bank_set must be unique")
    fingerprint_bank_ids: list[str] = []
    for index, pair in enumerate(credit.conditioning_bank_fingerprints):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                "credit.conditioning_bank_fingerprints must contain string pairs"
            )
        bank_id, fingerprint = pair
        _require_text(bank_id, f"credit.conditioning_bank_fingerprints[{index}].bank_id")
        _require_text(
            fingerprint,
            f"credit.conditioning_bank_fingerprints[{index}].fingerprint",
        )
        fingerprint_bank_ids.append(bank_id)
    if len(set(fingerprint_bank_ids)) != len(fingerprint_bank_ids):
        raise ValueError("credit conditioning-bank fingerprint ids must be unique")


def _credit_record_payload(credit: CreditRecord) -> dict[str, object]:
    return {
        "record_id": credit.record_id,
        "level": credit.level,
        "track": credit.track.value,
        "source_event": credit.source_event,
        "credit_value": credit.credit_value,
        "context": credit.context,
        "timestamp_ms": credit.timestamp_ms,
        "prediction_id": credit.prediction_id,
        "environment_event_id": credit.environment_event_id,
        "environment_outcome_id": credit.environment_outcome_id,
        "segment_id": credit.segment_id,
        "abstract_action_id": credit.abstract_action_id,
        "conditioning_bank_set": list(credit.conditioning_bank_set),
        "conditioning_bank_fingerprints": [
            list(pair) for pair in credit.conditioning_bank_fingerprints
        ],
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _deterministic_draw(seed: str, decision_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{decision_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _state_digest(weights: tuple[float, ...], bias: float, update_count: int) -> str:
    payload = json.dumps(
        {
            "weights": list(weights),
            "bias": bias,
            "update_count": update_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reject_oracle_inputs(
    oracle_action_id: str | None,
    evaluator_only: bool,
) -> None:
    if oracle_action_id is not None or evaluator_only:
        raise ValueError("oracle inputs are only valid in oracle mode")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_empty_unique(values: tuple[str, ...], field_name: str) -> None:
    if not values or any(not value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")


def _require_exact_mapping(
    payload: object,
    *,
    expected: set[str],
    source: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} must be an object")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{source} keys must be strings")
    observed = set(payload)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{source} fields do not match schema; "
            f"missing={missing}, extra={extra}"
        )
    return payload


def _checkpoint_payload_body(payload: object) -> Mapping[str, object]:
    expected = {
        "schema_version",
        "artifact_id",
        "artifact_version",
        "weights",
        "bias",
        "update_count",
        "processed_credit_ids",
        "pending_decisions",
    }
    if not isinstance(payload, Mapping):
        raise ValueError("relationship action gate checkpoint must be an object")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError(
            "relationship action gate checkpoint keys must be strings"
        )
    observed = set(payload)
    if observed == expected:
        return payload
    legacy_expected = {*expected, "content_sha256"}
    if observed != legacy_expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - legacy_expected)
        raise ValueError(
            "relationship action gate checkpoint fields do not match schema; "
            f"missing={missing}, extra={extra}"
        )
    stored = _payload_text(payload, "content_sha256")
    body = {key: payload[key] for key in expected}
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    computed = hashlib.sha256(encoded).hexdigest()
    if stored != computed:
        raise ValueError(
            "relationship action gate checkpoint content hash mismatch"
        )
    return body


def _payload_text(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _payload_int(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _payload_float(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _payload_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _payload_text_tuple(
    payload: Mapping[str, object],
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{key} must be an array of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{key} must be non-empty")
    return tuple(value)


def _payload_float_tuple(
    payload: Mapping[str, object],
    key: str,
) -> tuple[float, ...]:
    value = payload[key]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int | float)
        for item in value
    ):
        raise ValueError(f"{key} must be an array of numbers")
    return tuple(float(item) for item in value)


__all__ = [
    "RELATIONSHIP_ACTION_CREDIT_LEVEL",
    "RELATIONSHIP_ACTION_GATE_BATCH_PLAN_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_BATCH_RECEIPT_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_CREDIT_BATCH_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_FEATURE_ORDER",
    "RELATIONSHIP_ACTION_GATE_FORCED_EXPOSURE_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_FROZEN_POLICY_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_OPERATOR_ID",
    "RELATIONSHIP_ACTION_GATE_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_THETA0_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_GATE_THRESHOLD_RULE",
    "RelationshipActionGate",
    "RelationshipActionGateArtifact",
    "RelationshipActionGateBatchDisposition",
    "RelationshipActionGateBatchPlan",
    "RelationshipActionGateBatchReceipt",
    "RelationshipActionGateCheckpoint",
    "RelationshipActionGateCreditBatch",
    "RelationshipActionGateDecision",
    "RelationshipActionGateForcedExposure",
    "RelationshipActionGateFrozenDecision",
    "RelationshipActionGateFrozenPolicy",
    "RelationshipActionGateMode",
    "RelationshipActionGateTheta0Artifact",
    "RelationshipActionGateUpdate",
    "RelationshipGateAction",
    "relationship_action_gate_features",
    "temporal_action_advisory_from_gate_decision",
]
