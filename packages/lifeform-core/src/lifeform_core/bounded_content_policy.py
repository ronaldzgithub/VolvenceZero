"""Reusable bounded content-position policy for vertical Brain adapters.

The policy sees only adapter-published numeric features and opaque entry IDs.
It can promote at most one non-leading entry to the first position or preserve
the owner retrieval order exactly.  Vertical adapters retain ownership of
memory retrieval, PE/credit settlement, checkpoint scope and persistence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

from lifeform_core.bounded_policy import (
    BoundedPolicyCandidate,
    BoundedPolicyDecision,
    BoundedPolicyRankedCandidate,
    apply_bounded_policy_credit,
    rank_and_gate_bounded_policy,
)


CONTENT_POLICY_CHECKPOINT_SCHEMA_VERSION = "bounded-content-policy-checkpoint.v1"
CONTENT_POLICY_DECISION_SCHEMA_VERSION = "bounded-content-policy-decision.v1"
CONTENT_POLICY_CREDIT_SCHEMA_VERSION = "bounded-content-policy-credit.v1"
CONTENT_POLICY_UPDATE_SCHEMA_VERSION = "bounded-content-policy-update.v1"
CONTENT_POLICY_ACTION_KEY = "promote_first"
CONTENT_POLICY_NOOP_CANDIDATE_ID = "bounded-content-policy:no-op"


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_text(name: str, value: object, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{name} must be at most {maximum} characters")
    return value


def _require_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _require_string_tuple(
    name: str,
    values: object,
    *,
    maximum: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) > maximum:
        raise ValueError(f"{name} has too many entries")
    for value in values:
        _require_text(name, value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be unique")
    return values


def _strict_mapping(
    name: str,
    value: object,
    *,
    fields: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has invalid shape")
    return value


def _array(name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return tuple(value)


@dataclass(frozen=True)
class BoundedContentCandidate:
    """One opaque memory entry projected by its owning adapter."""

    entry_id: str
    feature_values: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _require_text("entry_id", self.entry_id)
        if not isinstance(self.feature_values, tuple) or not self.feature_values:
            raise ValueError("feature_values must be a non-empty tuple")
        names = tuple(name for name, _ in self.feature_values)
        _require_string_tuple(
            "feature names",
            names,
            maximum=32,
            allow_empty=False,
        )
        for _, value in self.feature_values:
            numeric = _require_finite("feature value", value)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError("content policy features must be in [0, 1]")

    def to_json(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "feature_values": [
                {"name": name, "value": float(value)}
                for name, value in self.feature_values
            ],
        }


@dataclass(frozen=True)
class BoundedContentPolicyCheckpoint:
    checkpoint_id: str
    content_sha256: str
    artifact_id: str
    feature_order: tuple[str, ...]
    ranking_weights: tuple[float, ...]
    intervention_weights: tuple[float, ...]
    intervention_bias: float
    learning_rate: float
    max_abs_parameter: float
    update_count: int = 0
    processed_credit_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text("checkpoint_id", self.checkpoint_id)
        if self.checkpoint_id != f"bounded-content-policy-checkpoint:{self.content_sha256}":
            raise ValueError("checkpoint_id must match content_sha256")
        if len(self.content_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.content_sha256
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        _require_text("artifact_id", self.artifact_id)
        _require_string_tuple(
            "feature_order",
            self.feature_order,
            maximum=32,
            allow_empty=False,
        )
        if len(self.ranking_weights) != len(self.feature_order):
            raise ValueError("ranking_weights must match feature_order")
        if len(self.intervention_weights) != len(self.feature_order):
            raise ValueError("intervention_weights must match feature_order")
        cap = _require_finite("max_abs_parameter", self.max_abs_parameter)
        if not 0.5 <= cap <= 8.0:
            raise ValueError("max_abs_parameter must be in [0.5, 8]")
        for value in (
            *self.ranking_weights,
            *self.intervention_weights,
            self.intervention_bias,
        ):
            if abs(_require_finite("policy parameter", value)) > cap:
                raise ValueError("policy parameter exceeds max_abs_parameter")
        rate = _require_finite("learning_rate", self.learning_rate)
        if not 0.0 < rate <= 0.5:
            raise ValueError("learning_rate must be in (0, 0.5]")
        if (
            isinstance(self.update_count, bool)
            or not isinstance(self.update_count, int)
            or self.update_count < 0
        ):
            raise ValueError("update_count must be a non-negative integer")
        _require_string_tuple(
            "processed_credit_ids",
            self.processed_credit_ids,
            maximum=4_096,
        )
        if self.update_count != len(self.processed_credit_ids):
            raise ValueError("processed credit count must equal update_count")

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        feature_order: tuple[str, ...],
        ranking_weights: tuple[float, ...],
        intervention_weights: tuple[float, ...],
        intervention_bias: float,
        learning_rate: float = 0.12,
        max_abs_parameter: float = 4.0,
        update_count: int = 0,
        processed_credit_ids: tuple[str, ...] = (),
    ) -> "BoundedContentPolicyCheckpoint":
        core = {
            "schema_version": CONTENT_POLICY_CHECKPOINT_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "feature_order": list(feature_order),
            "ranking_weights": list(ranking_weights),
            "intervention_weights": list(intervention_weights),
            "intervention_bias": intervention_bias,
            "learning_rate": learning_rate,
            "max_abs_parameter": max_abs_parameter,
            "update_count": update_count,
            "processed_credit_ids": list(processed_credit_ids),
        }
        digest = _digest(core)
        return cls(
            checkpoint_id=f"bounded-content-policy-checkpoint:{digest}",
            content_sha256=digest,
            artifact_id=artifact_id,
            feature_order=feature_order,
            ranking_weights=ranking_weights,
            intervention_weights=intervention_weights,
            intervention_bias=intervention_bias,
            learning_rate=learning_rate,
            max_abs_parameter=max_abs_parameter,
            update_count=update_count,
            processed_credit_ids=processed_credit_ids,
        )

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "BoundedContentPolicyCheckpoint":
        expected = {
            "schema_version",
            "checkpoint_id",
            "content_sha256",
            "artifact_id",
            "feature_order",
            "ranking_weights",
            "intervention_weights",
            "intervention_bias",
            "learning_rate",
            "max_abs_parameter",
            "update_count",
            "processed_credit_ids",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("bounded content checkpoint has invalid shape")
        if payload["schema_version"] != CONTENT_POLICY_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported bounded content checkpoint schema")
        checkpoint = cls(
            checkpoint_id=payload["checkpoint_id"],
            content_sha256=payload["content_sha256"],
            artifact_id=payload["artifact_id"],
            feature_order=tuple(payload["feature_order"]),
            ranking_weights=tuple(payload["ranking_weights"]),
            intervention_weights=tuple(payload["intervention_weights"]),
            intervention_bias=payload["intervention_bias"],
            learning_rate=payload["learning_rate"],
            max_abs_parameter=payload["max_abs_parameter"],
            update_count=payload["update_count"],
            processed_credit_ids=tuple(payload["processed_credit_ids"]),
        )
        core = checkpoint.to_json()
        core.pop("checkpoint_id")
        core.pop("content_sha256")
        if _digest(core) != checkpoint.content_sha256:
            raise ValueError("bounded content checkpoint digest mismatch")
        return checkpoint

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTENT_POLICY_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "content_sha256": self.content_sha256,
            "artifact_id": self.artifact_id,
            "feature_order": list(self.feature_order),
            "ranking_weights": [float(value) for value in self.ranking_weights],
            "intervention_weights": [
                float(value) for value in self.intervention_weights
            ],
            "intervention_bias": float(self.intervention_bias),
            "learning_rate": float(self.learning_rate),
            "max_abs_parameter": float(self.max_abs_parameter),
            "update_count": self.update_count,
            "processed_credit_ids": list(self.processed_credit_ids),
        }


@dataclass(frozen=True)
class BoundedContentRankedCandidate:
    entry_id: str
    rank: int
    policy_score: float
    selection_probability: float
    feature_values: tuple[tuple[str, float], ...]

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "BoundedContentRankedCandidate":
        fields = frozenset(
            {
                "entry_id",
                "rank",
                "policy_score",
                "selection_probability",
                "feature_values",
            }
        )
        _strict_mapping("ranked candidate", payload, fields=fields)
        features: list[tuple[str, float]] = []
        for item in _array("feature_values", payload["feature_values"]):
            row = _strict_mapping(
                "feature_values[]",
                item,
                fields=frozenset({"name", "value"}),
            )
            features.append((row["name"], row["value"]))
        return cls(
            entry_id=payload["entry_id"],
            rank=payload["rank"],
            policy_score=payload["policy_score"],
            selection_probability=payload["selection_probability"],
            feature_values=tuple(features),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "rank": self.rank,
            "policy_score": float(self.policy_score),
            "selection_probability": float(self.selection_probability),
            "feature_values": [
                {"name": name, "value": float(value)}
                for name, value in self.feature_values
            ],
        }


@dataclass(frozen=True)
class BoundedContentPolicyDecision:
    policy_decision_id: str
    content_sha256: str
    checkpoint_id: str
    checkpoint_update_count: int
    source_prediction_id: str
    input_entry_ids: tuple[str, ...]
    output_entry_ids: tuple[str, ...]
    recommended_entry_id: str
    selected_entry_id: str
    intervention_probability: float
    intervened: bool
    ranked_candidates: tuple[BoundedContentRankedCandidate, ...]

    def __post_init__(self) -> None:
        if self.policy_decision_id != f"bounded-content-policy-decision:{self.content_sha256}":
            raise ValueError("policy_decision_id must match content_sha256")
        _require_text("source_prediction_id", self.source_prediction_id)
        _require_string_tuple(
            "input_entry_ids",
            self.input_entry_ids,
            maximum=80,
            allow_empty=False,
        )
        _require_string_tuple(
            "output_entry_ids",
            self.output_entry_ids,
            maximum=80,
            allow_empty=False,
        )
        if set(self.input_entry_ids) != set(self.output_entry_ids):
            raise ValueError("content policy output must preserve the input set")
        if self.intervened:
            if not self.selected_entry_id:
                raise ValueError("intervention requires selected_entry_id")
            if self.output_entry_ids[0] != self.selected_entry_id:
                raise ValueError("selected entry must be promoted to first position")
        elif self.output_entry_ids != self.input_entry_ids or self.selected_entry_id:
            raise ValueError("NOOP must preserve exact owner order")

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "BoundedContentPolicyDecision":
        fields = frozenset(
            {
                "schema_version",
                "policy_decision_id",
                "content_sha256",
                "checkpoint_id",
                "checkpoint_update_count",
                "source_prediction_id",
                "input_entry_ids",
                "output_entry_ids",
                "recommended_entry_id",
                "selected_entry_id",
                "intervention_probability",
                "intervened",
                "ranked_candidates",
            }
        )
        _strict_mapping("content policy decision", payload, fields=fields)
        if payload["schema_version"] != CONTENT_POLICY_DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported bounded content decision schema")
        decision = cls(
            policy_decision_id=payload["policy_decision_id"],
            content_sha256=payload["content_sha256"],
            checkpoint_id=payload["checkpoint_id"],
            checkpoint_update_count=payload["checkpoint_update_count"],
            source_prediction_id=payload["source_prediction_id"],
            input_entry_ids=tuple(_array("input_entry_ids", payload["input_entry_ids"])),
            output_entry_ids=tuple(
                _array("output_entry_ids", payload["output_entry_ids"])
            ),
            recommended_entry_id=payload["recommended_entry_id"],
            selected_entry_id=payload["selected_entry_id"],
            intervention_probability=payload["intervention_probability"],
            intervened=payload["intervened"],
            ranked_candidates=tuple(
                BoundedContentRankedCandidate.from_json(
                    _strict_mapping(
                        "ranked_candidates[]",
                        item,
                        fields=frozenset(
                            {
                                "entry_id",
                                "rank",
                                "policy_score",
                                "selection_probability",
                                "feature_values",
                            }
                        ),
                    )
                )
                for item in _array(
                    "ranked_candidates",
                    payload["ranked_candidates"],
                )
            ),
        )
        core = decision.to_json()
        core.pop("policy_decision_id")
        core.pop("content_sha256")
        if _digest(core) != decision.content_sha256:
            raise ValueError("bounded content decision digest mismatch")
        return decision

    @classmethod
    def create(
        cls,
        *,
        checkpoint: BoundedContentPolicyCheckpoint,
        source_prediction_id: str,
        input_entry_ids: tuple[str, ...],
        output_entry_ids: tuple[str, ...],
        recommended_entry_id: str,
        selected_entry_id: str,
        intervention_probability: float,
        intervened: bool,
        ranked_candidates: tuple[BoundedContentRankedCandidate, ...],
    ) -> "BoundedContentPolicyDecision":
        core = {
            "schema_version": CONTENT_POLICY_DECISION_SCHEMA_VERSION,
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_update_count": checkpoint.update_count,
            "source_prediction_id": source_prediction_id,
            "input_entry_ids": list(input_entry_ids),
            "output_entry_ids": list(output_entry_ids),
            "recommended_entry_id": recommended_entry_id,
            "selected_entry_id": selected_entry_id,
            "intervention_probability": intervention_probability,
            "intervened": intervened,
            "ranked_candidates": [item.to_json() for item in ranked_candidates],
        }
        digest = _digest(core)
        return cls(
            policy_decision_id=f"bounded-content-policy-decision:{digest}",
            content_sha256=digest,
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_update_count=checkpoint.update_count,
            source_prediction_id=source_prediction_id,
            input_entry_ids=input_entry_ids,
            output_entry_ids=output_entry_ids,
            recommended_entry_id=recommended_entry_id,
            selected_entry_id=selected_entry_id,
            intervention_probability=intervention_probability,
            intervened=intervened,
            ranked_candidates=ranked_candidates,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTENT_POLICY_DECISION_SCHEMA_VERSION,
            "policy_decision_id": self.policy_decision_id,
            "content_sha256": self.content_sha256,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_update_count": self.checkpoint_update_count,
            "source_prediction_id": self.source_prediction_id,
            "input_entry_ids": list(self.input_entry_ids),
            "output_entry_ids": list(self.output_entry_ids),
            "recommended_entry_id": self.recommended_entry_id,
            "selected_entry_id": self.selected_entry_id,
            "intervention_probability": float(self.intervention_probability),
            "intervened": self.intervened,
            "ranked_candidates": [item.to_json() for item in self.ranked_candidates],
        }


@dataclass(frozen=True)
class BoundedContentPolicyCredit:
    credit_id: str
    policy_decision_id: str
    credited_candidate_id: str
    prediction_id: str
    settlement_ref: str
    signed_prediction_error: float
    source_credit_record_ids: tuple[str, ...]
    observed_at_ms: int

    def __post_init__(self) -> None:
        _require_text("credit_id", self.credit_id)
        _require_text("policy_decision_id", self.policy_decision_id)
        _require_text("credited_candidate_id", self.credited_candidate_id)
        _require_text("prediction_id", self.prediction_id)
        _require_text("settlement_ref", self.settlement_ref)
        signed = _require_finite(
            "signed_prediction_error",
            self.signed_prediction_error,
        )
        if not -1.0 <= signed <= 1.0:
            raise ValueError("signed_prediction_error must be in [-1, 1]")
        _require_string_tuple(
            "source_credit_record_ids",
            self.source_credit_record_ids,
            maximum=16,
            allow_empty=False,
        )
        if len(self.source_credit_record_ids) != 4:
            raise ValueError("content policy requires exactly four PE credits")
        if (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 0
        ):
            raise ValueError("observed_at_ms must be a non-negative integer")

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "BoundedContentPolicyCredit":
        fields = frozenset(
            {
                "schema_version",
                "credit_id",
                "policy_decision_id",
                "credited_candidate_id",
                "prediction_id",
                "settlement_ref",
                "signed_prediction_error",
                "source_credit_record_ids",
                "observed_at_ms",
            }
        )
        _strict_mapping("content policy credit", payload, fields=fields)
        if payload["schema_version"] != CONTENT_POLICY_CREDIT_SCHEMA_VERSION:
            raise ValueError("unsupported bounded content credit schema")
        credit = cls(
            credit_id=payload["credit_id"],
            policy_decision_id=payload["policy_decision_id"],
            credited_candidate_id=payload["credited_candidate_id"],
            prediction_id=payload["prediction_id"],
            settlement_ref=payload["settlement_ref"],
            signed_prediction_error=payload["signed_prediction_error"],
            source_credit_record_ids=tuple(
                _array(
                    "source_credit_record_ids",
                    payload["source_credit_record_ids"],
                )
            ),
            observed_at_ms=payload["observed_at_ms"],
        )
        expected = cls.create(
            policy_decision_id=credit.policy_decision_id,
            credited_candidate_id=credit.credited_candidate_id,
            prediction_id=credit.prediction_id,
            settlement_ref=credit.settlement_ref,
            signed_prediction_error=credit.signed_prediction_error,
            source_credit_record_ids=credit.source_credit_record_ids,
            observed_at_ms=credit.observed_at_ms,
        )
        if credit.credit_id != expected.credit_id:
            raise ValueError("bounded content credit digest mismatch")
        return credit

    @classmethod
    def create(
        cls,
        *,
        policy_decision_id: str,
        credited_candidate_id: str,
        prediction_id: str,
        settlement_ref: str,
        signed_prediction_error: float,
        source_credit_record_ids: tuple[str, ...],
        observed_at_ms: int,
    ) -> "BoundedContentPolicyCredit":
        core = {
            "schema_version": CONTENT_POLICY_CREDIT_SCHEMA_VERSION,
            "policy_decision_id": policy_decision_id,
            "credited_candidate_id": credited_candidate_id,
            "prediction_id": prediction_id,
            "settlement_ref": settlement_ref,
            "signed_prediction_error": signed_prediction_error,
            "source_credit_record_ids": list(source_credit_record_ids),
            "observed_at_ms": observed_at_ms,
        }
        return cls(
            credit_id=f"bounded-content-policy-credit:{_digest(core)}",
            policy_decision_id=policy_decision_id,
            credited_candidate_id=credited_candidate_id,
            prediction_id=prediction_id,
            settlement_ref=settlement_ref,
            signed_prediction_error=signed_prediction_error,
            source_credit_record_ids=source_credit_record_ids,
            observed_at_ms=observed_at_ms,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTENT_POLICY_CREDIT_SCHEMA_VERSION,
            "credit_id": self.credit_id,
            "policy_decision_id": self.policy_decision_id,
            "credited_candidate_id": self.credited_candidate_id,
            "prediction_id": self.prediction_id,
            "settlement_ref": self.settlement_ref,
            "signed_prediction_error": float(self.signed_prediction_error),
            "source_credit_record_ids": list(self.source_credit_record_ids),
            "observed_at_ms": self.observed_at_ms,
        }


@dataclass(frozen=True)
class BoundedContentPolicyUpdateReceipt:
    update_id: str
    credit_id: str
    policy_decision_id: str
    previous_checkpoint_id: str
    next_checkpoint_id: str
    parameter_delta_l2: float
    update_count: int

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "BoundedContentPolicyUpdateReceipt":
        fields = frozenset(
            {
                "schema_version",
                "update_id",
                "credit_id",
                "policy_decision_id",
                "previous_checkpoint_id",
                "next_checkpoint_id",
                "parameter_delta_l2",
                "update_count",
            }
        )
        _strict_mapping("content policy update", payload, fields=fields)
        if payload["schema_version"] != CONTENT_POLICY_UPDATE_SCHEMA_VERSION:
            raise ValueError("unsupported bounded content update schema")
        receipt = cls(
            update_id=payload["update_id"],
            credit_id=payload["credit_id"],
            policy_decision_id=payload["policy_decision_id"],
            previous_checkpoint_id=payload["previous_checkpoint_id"],
            next_checkpoint_id=payload["next_checkpoint_id"],
            parameter_delta_l2=payload["parameter_delta_l2"],
            update_count=payload["update_count"],
        )
        core = receipt.to_json()
        core.pop("update_id")
        if receipt.update_id != f"bounded-content-policy-update:{_digest(core)}":
            raise ValueError("bounded content update digest mismatch")
        return receipt

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": CONTENT_POLICY_UPDATE_SCHEMA_VERSION,
            "update_id": self.update_id,
            "credit_id": self.credit_id,
            "policy_decision_id": self.policy_decision_id,
            "previous_checkpoint_id": self.previous_checkpoint_id,
            "next_checkpoint_id": self.next_checkpoint_id,
            "parameter_delta_l2": float(self.parameter_delta_l2),
            "update_count": self.update_count,
        }


class BoundedContentPolicy:
    """Shared policy mechanism; adapters own every semantic input and output."""

    def decide(
        self,
        *,
        owner_order: tuple[str, ...],
        challengers: tuple[BoundedContentCandidate, ...],
        source_prediction_id: str,
        checkpoint: BoundedContentPolicyCheckpoint,
    ) -> BoundedContentPolicyDecision | None:
        if len(owner_order) < 2:
            return None
        _require_string_tuple(
            "owner_order",
            owner_order,
            maximum=80,
            allow_empty=False,
        )
        if tuple(item.entry_id for item in challengers) != owner_order[1:]:
            raise ValueError("challengers must preserve the non-leading owner order")
        for candidate in challengers:
            if tuple(name for name, _ in candidate.feature_values) != checkpoint.feature_order:
                raise ValueError("candidate feature order drift")
        raw = rank_and_gate_bounded_policy(
            candidates=tuple(
                BoundedPolicyCandidate(
                    candidate_id=item.entry_id,
                    action_key=CONTENT_POLICY_ACTION_KEY,
                    feature_values=tuple(value for _, value in item.feature_values),
                )
                for item in challengers
            ),
            action_weights=((CONTENT_POLICY_ACTION_KEY, checkpoint.ranking_weights),),
            intervention_weights=checkpoint.intervention_weights,
            intervention_bias=checkpoint.intervention_bias,
            maximum_candidates=len(challengers),
        )
        selected = raw.selected_candidate_id
        output_order = owner_order
        if raw.intervenes:
            output_order = (selected, *(item for item in owner_order if item != selected))
        by_id = {item.entry_id: item for item in challengers}
        ranked = tuple(
            BoundedContentRankedCandidate(
                entry_id=item.candidate_id,
                rank=item.rank,
                policy_score=item.policy_score,
                selection_probability=item.selection_probability,
                feature_values=by_id[item.candidate_id].feature_values,
            )
            for item in raw.ranked_candidates
        )
        return BoundedContentPolicyDecision.create(
            checkpoint=checkpoint,
            source_prediction_id=source_prediction_id,
            input_entry_ids=owner_order,
            output_entry_ids=output_order,
            recommended_entry_id=raw.recommended_candidate_id,
            selected_entry_id=selected,
            intervention_probability=raw.intervention_probability,
            intervened=raw.intervenes,
            ranked_candidates=ranked,
        )

    def observe_credit(
        self,
        *,
        checkpoint: BoundedContentPolicyCheckpoint,
        decision: BoundedContentPolicyDecision,
        credit: BoundedContentPolicyCredit,
    ) -> tuple[
        BoundedContentPolicyCheckpoint,
        BoundedContentPolicyUpdateReceipt,
    ]:
        if credit.credit_id in checkpoint.processed_credit_ids:
            raise ValueError("content policy credit was already processed")
        if credit.policy_decision_id != decision.policy_decision_id:
            raise ValueError("content policy credit decision mismatch")
        if credit.prediction_id != decision.source_prediction_id:
            raise ValueError("content policy credit prediction mismatch")
        raw_decision = BoundedPolicyDecision(
            ranked_candidates=tuple(
                BoundedPolicyRankedCandidate(
                    candidate_id=item.entry_id,
                    action_key=CONTENT_POLICY_ACTION_KEY,
                    rank=item.rank,
                    policy_score=item.policy_score,
                    selection_probability=item.selection_probability,
                    feature_values=tuple(value for _, value in item.feature_values),
                )
                for item in decision.ranked_candidates
            ),
            recommended_candidate_id=decision.recommended_entry_id,
            selected_candidate_id=decision.selected_entry_id,
            intervention_probability=decision.intervention_probability,
            intervenes=decision.intervened,
        )
        update = apply_bounded_policy_credit(
            action_weights=((CONTENT_POLICY_ACTION_KEY, checkpoint.ranking_weights),),
            intervention_weights=checkpoint.intervention_weights,
            intervention_bias=checkpoint.intervention_bias,
            decision=raw_decision,
            credited_candidate_id=credit.credited_candidate_id,
            noop_candidate_id=CONTENT_POLICY_NOOP_CANDIDATE_ID,
            signed_credit=credit.signed_prediction_error,
            learning_rate=checkpoint.learning_rate,
            max_abs_parameter=checkpoint.max_abs_parameter,
        )
        next_checkpoint = BoundedContentPolicyCheckpoint.create(
            artifact_id=checkpoint.artifact_id,
            feature_order=checkpoint.feature_order,
            ranking_weights=update.action_weights[0][1],
            intervention_weights=update.intervention_weights,
            intervention_bias=update.intervention_bias,
            learning_rate=checkpoint.learning_rate,
            max_abs_parameter=checkpoint.max_abs_parameter,
            update_count=checkpoint.update_count + 1,
            processed_credit_ids=(*checkpoint.processed_credit_ids, credit.credit_id),
        )
        receipt_core = {
            "schema_version": CONTENT_POLICY_UPDATE_SCHEMA_VERSION,
            "credit_id": credit.credit_id,
            "policy_decision_id": decision.policy_decision_id,
            "previous_checkpoint_id": checkpoint.checkpoint_id,
            "next_checkpoint_id": next_checkpoint.checkpoint_id,
            "parameter_delta_l2": update.parameter_delta_l2,
            "update_count": next_checkpoint.update_count,
        }
        receipt = BoundedContentPolicyUpdateReceipt(
            update_id=f"bounded-content-policy-update:{_digest(receipt_core)}",
            credit_id=credit.credit_id,
            policy_decision_id=decision.policy_decision_id,
            previous_checkpoint_id=checkpoint.checkpoint_id,
            next_checkpoint_id=next_checkpoint.checkpoint_id,
            parameter_delta_l2=update.parameter_delta_l2,
            update_count=next_checkpoint.update_count,
        )
        return next_checkpoint, receipt


def default_bounded_content_policy_checkpoint(
    *,
    artifact_id: str,
    feature_order: tuple[str, ...],
) -> BoundedContentPolicyCheckpoint:
    """Return the reviewed non-zero theta0 shared by vertical adapters."""

    if len(feature_order) != 5:
        raise ValueError("bounded content policy v1 requires five typed features")
    return BoundedContentPolicyCheckpoint.create(
        artifact_id=artifact_id,
        feature_order=feature_order,
        ranking_weights=(0.25, 0.90, 0.55, 0.70, 0.20),
        intervention_weights=(0.10, 0.45, 0.30, 0.35, 0.55),
        intervention_bias=-0.85,
    )


__all__ = (
    "BoundedContentCandidate",
    "BoundedContentPolicy",
    "BoundedContentPolicyCheckpoint",
    "BoundedContentPolicyCredit",
    "BoundedContentPolicyDecision",
    "BoundedContentPolicyUpdateReceipt",
    "CONTENT_POLICY_NOOP_CANDIDATE_ID",
    "default_bounded_content_policy_checkpoint",
)
