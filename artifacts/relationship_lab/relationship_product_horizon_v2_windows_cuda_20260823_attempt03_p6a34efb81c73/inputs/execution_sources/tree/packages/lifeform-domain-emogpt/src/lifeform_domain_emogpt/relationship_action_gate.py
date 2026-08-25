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
from dataclasses import dataclass
from enum import Enum

from volvence_zero.credit import CreditRecord
from volvence_zero.memory import Track
from volvence_zero.social_cognition import PreferenceActionForecast
from volvence_zero.temporal_types import TemporalActionAdvisoryProposal

from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)


RELATIONSHIP_ACTION_GATE_SCHEMA_VERSION = "relationship-action-gate.v1"
RELATIONSHIP_ACTION_CREDIT_LEVEL = "relationship_action_prediction_error"
_FEATURE_COUNT = 5
_POSITIVE_OUTCOMES = frozenset({"helped", "felt_heard"})


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
        if not -1.0 <= self.credit_value <= 1.0:
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
        _require_text(random_seed, "random_seed")
        self._random_seed = random_seed
        self._weights = self._artifact.initial_weights
        self._bias = self._artifact.initial_bias
        self._update_count = 0
        self._processed_credit_ids: set[str] = set()
        self._pending: dict[str, RelationshipActionGateDecision] = {}
        if checkpoint is not None:
            self.restore(checkpoint)

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def parameter_state(self) -> tuple[tuple[float, ...], float]:
        return self._weights, self._bias

    def decide(
        self,
        forecast: PreferenceActionForecast,
        *,
        mode: RelationshipActionGateMode = RelationshipActionGateMode.LEARNED,
        oracle_action_id: str | None = None,
        evaluator_only: bool = False,
    ) -> RelationshipActionGateDecision:
        features = relationship_action_gate_features(forecast)
        probability = _sigmoid(
            math.fsum(weight * value for weight, value in zip(self._weights, features, strict=True))
            + self._bias
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
            update_count=self._update_count,
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
        if mode is RelationshipActionGateMode.LEARNED:
            existing = self._pending.get(forecast.forecast_id)
            if existing is not None and existing != decision:
                raise ValueError("forecast already has a different pending gate decision")
            self._pending[forecast.forecast_id] = decision
        return decision

    def observe_credit(self, credit: CreditRecord) -> RelationshipActionGateUpdate:
        if credit.level != RELATIONSHIP_ACTION_CREDIT_LEVEL:
            raise ValueError("relationship action gate only accepts PE-derived relationship credit")
        if credit.track is not Track.SELF:
            raise ValueError("relationship action credit must be self-track")
        if credit.record_id in self._processed_credit_ids:
            raise ValueError("relationship action credit was already processed")
        try:
            decision = self._pending[credit.prediction_id]
        except KeyError as exc:
            raise ValueError("relationship action credit has no pending learned decision") from exc
        if credit.abstract_action_id != decision.selected_action_id:
            raise ValueError("relationship action credit action lineage mismatch")
        old_digest = _state_digest(self._weights, self._bias, self._update_count)
        action_indicator = (
            1.0 if decision.gate_action is RelationshipGateAction.STEER else 0.0
        )
        gradient_scale = (
            self._artifact.learning_rate
            * credit.credit_value
            * (action_indicator - decision.steer_probability)
        )
        cap = self._artifact.max_abs_parameter
        self._weights = tuple(
            max(-cap, min(cap, weight + gradient_scale * feature))
            for weight, feature in zip(self._weights, decision.features, strict=True)
        )
        self._bias = max(-cap, min(cap, self._bias + gradient_scale))
        self._update_count += 1
        self._processed_credit_ids.add(credit.record_id)
        del self._pending[credit.prediction_id]
        new_digest = _state_digest(self._weights, self._bias, self._update_count)
        return RelationshipActionGateUpdate(
            credit_record_id=credit.record_id,
            forecast_id=credit.prediction_id,
            selected_action_id=credit.abstract_action_id,
            credit_value=credit.credit_value,
            old_state_sha256=old_digest,
            new_state_sha256=new_digest,
            update_count=self._update_count,
        )

    def export_checkpoint(self) -> RelationshipActionGateCheckpoint:
        return RelationshipActionGateCheckpoint(
            artifact_id=self._artifact.artifact_id,
            artifact_version=self._artifact.artifact_version,
            weights=self._weights,
            bias=self._bias,
            update_count=self._update_count,
            processed_credit_ids=tuple(sorted(self._processed_credit_ids)),
            pending_decisions=tuple(
                self._pending[key] for key in sorted(self._pending)
            ),
        )

    def restore(self, checkpoint: RelationshipActionGateCheckpoint) -> None:
        if checkpoint.artifact_id != self._artifact.artifact_id:
            raise ValueError("relationship action gate artifact_id mismatch")
        if checkpoint.artifact_version != self._artifact.artifact_version:
            raise ValueError("relationship action gate artifact_version mismatch")
        cap = self._artifact.max_abs_parameter
        if any(abs(value) > cap for value in (*checkpoint.weights, checkpoint.bias)):
            raise ValueError("relationship action gate checkpoint exceeds parameter cap")
        self._weights = checkpoint.weights
        self._bias = checkpoint.bias
        self._update_count = checkpoint.update_count
        self._processed_credit_ids = set(checkpoint.processed_credit_ids)
        self._pending = {
            decision.forecast_id: decision
            for decision in checkpoint.pending_decisions
        }


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
    "RELATIONSHIP_ACTION_GATE_SCHEMA_VERSION",
    "RelationshipActionGate",
    "RelationshipActionGateArtifact",
    "RelationshipActionGateCheckpoint",
    "RelationshipActionGateDecision",
    "RelationshipActionGateMode",
    "RelationshipActionGateUpdate",
    "RelationshipGateAction",
    "relationship_action_gate_features",
    "temporal_action_advisory_from_gate_decision",
]
