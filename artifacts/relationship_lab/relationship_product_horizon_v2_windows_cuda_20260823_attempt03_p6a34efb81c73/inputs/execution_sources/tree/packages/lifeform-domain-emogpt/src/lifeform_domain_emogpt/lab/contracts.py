"""Frozen Relationship Lab evidence contracts.

This module belongs to the relationship vertical's *offline* lab.  It does
not define a runtime owner or a kernel snapshot.  The pre-action decision is
kept separate from the settled sidecar so a consumer cannot manufacture a
prediction after seeing the environment outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind

from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)


RELATIONSHIP_DECISION_TRACE_SCHEMA_VERSION = "relationship-decision-trace.v1"
RELATIONSHIP_PREACTION_SCHEMA_VERSION = "relationship-preaction-decision.v1"


class RelationshipDatasetSplit(str, Enum):
    """Content split assigned at mirrored-pair level."""

    TRAIN = "train"
    VALIDATION = "validation"
    HELDOUT = "heldout"


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def canonical_json(payload: object) -> str:
    """Return the byte-stable JSON representation used for content ids."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _as_text(value: object, field_name: str) -> str:
    _require_text(value, field_name)
    assert isinstance(value, str)
    return value


def _as_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _as_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")


def _timestamp(value: str, field_name: str) -> datetime:
    _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    *,
    source: str,
) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(
            f"{source} fields do not match schema; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class OutcomeProbability:
    """One typed external outcome probability."""

    outcome_kind: DialogueExternalOutcomeKind
    probability: float

    def __post_init__(self) -> None:
        if self.outcome_kind not in RELATIONSHIP_OUTCOMES:
            raise ValueError(
                f"unsupported Relationship Lab outcome: {self.outcome_kind.value}"
            )
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("outcome probability must be finite and in [0, 1]")

    def to_payload(self) -> dict[str, object]:
        return {
            "outcome_kind": self.outcome_kind.value,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class CandidateOutcomePrediction:
    """Action-conditional prediction frozen before an action is executed."""

    action_id: RelationshipAction
    outcomes: tuple[OutcomeProbability, ...]

    def __post_init__(self) -> None:
        observed = tuple(item.outcome_kind for item in self.outcomes)
        if observed != RELATIONSHIP_OUTCOMES:
            raise ValueError(
                "candidate prediction must contain every relationship outcome "
                "exactly once in canonical order"
            )
        total = math.fsum(item.probability for item in self.outcomes)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"candidate prediction probabilities must sum to 1.0, got {total}"
            )

    def probability_of(self, kind: DialogueExternalOutcomeKind) -> float:
        for outcome in self.outcomes:
            if outcome.outcome_kind is kind:
                return outcome.probability
        raise KeyError(kind)

    def to_payload(self) -> dict[str, object]:
        return {
            "action_id": self.action_id.value,
            "outcomes": [item.to_payload() for item in self.outcomes],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CandidateOutcomePrediction":
        if not isinstance(payload, dict):
            raise ValueError("candidate prediction must be an object")
        _require_exact_keys(payload, {"action_id", "outcomes"}, source="prediction")
        raw_outcomes = payload["outcomes"]
        if not isinstance(raw_outcomes, list):
            raise ValueError("prediction.outcomes must be an array")
        outcomes: list[OutcomeProbability] = []
        for index, raw in enumerate(raw_outcomes):
            if not isinstance(raw, dict):
                raise ValueError(f"prediction.outcomes[{index}] must be an object")
            _require_exact_keys(
                raw,
                {"outcome_kind", "probability"},
                source=f"prediction.outcomes[{index}]",
            )
            outcomes.append(
                OutcomeProbability(
                    outcome_kind=DialogueExternalOutcomeKind(
                        _as_text(
                            raw["outcome_kind"],
                            f"prediction.outcomes[{index}].outcome_kind",
                        )
                    ),
                    probability=_as_float(
                        raw["probability"],
                        f"prediction.outcomes[{index}].probability",
                    ),
                )
            )
        return cls(
            action_id=RelationshipAction(
                _as_text(payload["action_id"], "prediction.action_id")
            ),
            outcomes=tuple(outcomes),
        )


@dataclass(frozen=True)
class RelationshipModelLineage:
    """Frozen model/prompt/generation identity for one decision."""

    model_id: str
    weights_sha256: str
    prompt_sha256: str
    generation_config_sha256: str
    seed: int

    def __post_init__(self) -> None:
        _require_text(self.model_id, "lineage.model_id")
        _require_sha256(self.weights_sha256, "lineage.weights_sha256")
        _require_sha256(self.prompt_sha256, "lineage.prompt_sha256")
        _require_sha256(
            self.generation_config_sha256,
            "lineage.generation_config_sha256",
        )
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("lineage.seed must be a non-negative integer")

    def to_payload(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "prompt_sha256": self.prompt_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "seed": self.seed,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "RelationshipModelLineage":
        if not isinstance(payload, dict):
            raise ValueError("lineage must be an object")
        expected = {
            "model_id",
            "weights_sha256",
            "prompt_sha256",
            "generation_config_sha256",
            "seed",
        }
        _require_exact_keys(payload, expected, source="lineage")
        return cls(
            model_id=_as_text(payload["model_id"], "lineage.model_id"),
            weights_sha256=_as_text(
                payload["weights_sha256"], "lineage.weights_sha256"
            ),
            prompt_sha256=_as_text(
                payload["prompt_sha256"], "lineage.prompt_sha256"
            ),
            generation_config_sha256=_as_text(
                payload["generation_config_sha256"],
                "lineage.generation_config_sha256",
            ),
            seed=_as_int(payload["seed"], "lineage.seed"),
        )


@dataclass(frozen=True)
class PreActionRelationshipDecision:
    """The auditable bet frozen before the environment reveals an outcome."""

    decision_id: str
    pre_action_timestamp: str
    candidate_predictions: tuple[CandidateOutcomePrediction, ...]
    chosen_action_id: RelationshipAction
    source_snapshot_hashes: tuple[str, ...]
    lineage: RelationshipModelLineage
    schema_version: str = RELATIONSHIP_PREACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PREACTION_SCHEMA_VERSION:
            raise ValueError(
                "pre-action schema_version must be "
                f"{RELATIONSHIP_PREACTION_SCHEMA_VERSION!r}"
            )
        _require_text(self.decision_id, "decision_id")
        _timestamp(self.pre_action_timestamp, "pre_action_timestamp")
        actions = tuple(item.action_id for item in self.candidate_predictions)
        if actions != RELATIONSHIP_ACTIONS:
            raise ValueError(
                "candidate predictions must cover the closed action surface in "
                "canonical order"
            )
        if self.chosen_action_id not in actions:
            raise ValueError("chosen action must be one of the candidate actions")
        if self.source_snapshot_hashes != tuple(
            sorted(set(self.source_snapshot_hashes))
        ):
            raise ValueError("source_snapshot_hashes must be sorted and unique")
        for index, digest in enumerate(self.source_snapshot_hashes):
            _require_sha256(digest, f"source_snapshot_hashes[{index}]")

    @property
    def candidate_action_ids(self) -> tuple[RelationshipAction, ...]:
        return tuple(item.action_id for item in self.candidate_predictions)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "pre_action_timestamp": self.pre_action_timestamp,
            "candidate_predictions": [
                item.to_payload() for item in self.candidate_predictions
            ],
            "chosen_action_id": self.chosen_action_id.value,
            "source_snapshot_hashes": list(self.source_snapshot_hashes),
            "lineage": self.lineage.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "PreActionRelationshipDecision":
        if not isinstance(payload, dict):
            raise ValueError("pre_action must be an object")
        expected = {
            "schema_version",
            "decision_id",
            "pre_action_timestamp",
            "candidate_predictions",
            "chosen_action_id",
            "source_snapshot_hashes",
            "lineage",
        }
        _require_exact_keys(payload, expected, source="pre_action")
        raw_predictions = payload["candidate_predictions"]
        raw_hashes = payload["source_snapshot_hashes"]
        if not isinstance(raw_predictions, list):
            raise ValueError("pre_action.candidate_predictions must be an array")
        if not isinstance(raw_hashes, list):
            raise ValueError("pre_action.source_snapshot_hashes must be an array")
        return cls(
            schema_version=_as_text(
                payload["schema_version"], "pre_action.schema_version"
            ),
            decision_id=_as_text(payload["decision_id"], "pre_action.decision_id"),
            pre_action_timestamp=_as_text(
                payload["pre_action_timestamp"],
                "pre_action.pre_action_timestamp",
            ),
            candidate_predictions=tuple(
                CandidateOutcomePrediction.from_payload(item)
                for item in raw_predictions
            ),
            chosen_action_id=RelationshipAction(
                _as_text(payload["chosen_action_id"], "pre_action.chosen_action_id")
            ),
            source_snapshot_hashes=tuple(
                _as_text(item, f"pre_action.source_snapshot_hashes[{index}]")
                for index, item in enumerate(raw_hashes)
            ),
            lineage=RelationshipModelLineage.from_payload(payload["lineage"]),
        )


@dataclass(frozen=True)
class RelationshipDecisionTrace:
    """Settled, content-addressed evidence sidecar for one decision.

    ``sealed_latent_dynamic_id`` and the observed outcome are deliberately
    present only in this post-action artifact.  A system-under-test receives a
    :class:`RelationshipObservation` payload from ``dataset.py`` instead.
    """

    trajectory_sha256: str
    user_scope_hash: str
    scenario_family: str
    surface_scene_id: str
    split: RelationshipDatasetSplit
    sealed_latent_dynamic_id: str
    pre_action: PreActionRelationshipDecision
    observed_typed_outcome: DialogueExternalOutcomeKind
    outcome_observed_at: str
    environment_evidence_ref: str
    prediction_error_ref: str | None = None
    credit_refs: tuple[str, ...] = ()
    next_state_hash: str | None = None
    schema_version: str = RELATIONSHIP_DECISION_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_DECISION_TRACE_SCHEMA_VERSION:
            raise ValueError(
                "decision trace schema_version must be "
                f"{RELATIONSHIP_DECISION_TRACE_SCHEMA_VERSION!r}"
            )
        _require_sha256(self.trajectory_sha256, "trajectory_sha256")
        _require_sha256(self.user_scope_hash, "user_scope_hash")
        _require_text(self.scenario_family, "scenario_family")
        _require_text(self.surface_scene_id, "surface_scene_id")
        _require_text(
            self.sealed_latent_dynamic_id,
            "sealed_latent_dynamic_id",
        )
        if self.observed_typed_outcome not in RELATIONSHIP_OUTCOMES:
            raise ValueError("observed_typed_outcome is outside the lab vocabulary")
        if _timestamp(
            self.outcome_observed_at,
            "outcome_observed_at",
        ) <= _timestamp(
            self.pre_action.pre_action_timestamp,
            "pre_action_timestamp",
        ):
            raise ValueError("outcome must be observed strictly after the pre-action bet")
        _require_sha256(self.environment_evidence_ref, "environment_evidence_ref")
        if self.prediction_error_ref is not None:
            _require_text(self.prediction_error_ref, "prediction_error_ref")
        if self.credit_refs != tuple(sorted(set(self.credit_refs))):
            raise ValueError("credit_refs must be sorted and unique")
        for index, ref in enumerate(self.credit_refs):
            _require_text(ref, f"credit_refs[{index}]")
        if self.next_state_hash is not None:
            _require_sha256(self.next_state_hash, "next_state_hash")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trajectory_sha256": self.trajectory_sha256,
            "user_scope_hash": self.user_scope_hash,
            "scenario_family": self.scenario_family,
            "surface_scene_id": self.surface_scene_id,
            "split": self.split.value,
            "sealed_latent_dynamic_id": self.sealed_latent_dynamic_id,
            "pre_action": self.pre_action.to_payload(),
            "observed_typed_outcome": self.observed_typed_outcome.value,
            "outcome_observed_at": self.outcome_observed_at,
            "environment_evidence_ref": self.environment_evidence_ref,
            "prediction_error_ref": self.prediction_error_ref,
            "credit_refs": list(self.credit_refs),
            "next_state_hash": self.next_state_hash,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipDecisionTrace":
        raw = json.loads(encoded)
        if not isinstance(raw, dict):
            raise ValueError("decision trace must be a JSON object")
        expected = {
            "artifact_id",
            "schema_version",
            "trajectory_sha256",
            "user_scope_hash",
            "scenario_family",
            "surface_scene_id",
            "split",
            "sealed_latent_dynamic_id",
            "pre_action",
            "observed_typed_outcome",
            "outcome_observed_at",
            "environment_evidence_ref",
            "prediction_error_ref",
            "credit_refs",
            "next_state_hash",
        }
        _require_exact_keys(raw, expected, source="decision_trace")
        raw_credit_refs = raw["credit_refs"]
        if not isinstance(raw_credit_refs, list):
            raise ValueError("decision_trace.credit_refs must be an array")
        artifact_id = _as_text(raw.pop("artifact_id"), "artifact_id")
        trace = cls(
            schema_version=_as_text(
                raw["schema_version"], "decision_trace.schema_version"
            ),
            trajectory_sha256=_as_text(
                raw["trajectory_sha256"], "decision_trace.trajectory_sha256"
            ),
            user_scope_hash=_as_text(
                raw["user_scope_hash"], "decision_trace.user_scope_hash"
            ),
            scenario_family=_as_text(
                raw["scenario_family"], "decision_trace.scenario_family"
            ),
            surface_scene_id=_as_text(
                raw["surface_scene_id"], "decision_trace.surface_scene_id"
            ),
            split=RelationshipDatasetSplit(
                _as_text(raw["split"], "decision_trace.split")
            ),
            sealed_latent_dynamic_id=_as_text(
                raw["sealed_latent_dynamic_id"],
                "decision_trace.sealed_latent_dynamic_id",
            ),
            pre_action=PreActionRelationshipDecision.from_payload(raw["pre_action"]),
            observed_typed_outcome=DialogueExternalOutcomeKind(
                _as_text(
                    raw["observed_typed_outcome"],
                    "decision_trace.observed_typed_outcome",
                )
            ),
            outcome_observed_at=_as_text(
                raw["outcome_observed_at"],
                "decision_trace.outcome_observed_at",
            ),
            environment_evidence_ref=_as_text(
                raw["environment_evidence_ref"],
                "decision_trace.environment_evidence_ref",
            ),
            prediction_error_ref=(
                _as_text(
                    raw["prediction_error_ref"],
                    "decision_trace.prediction_error_ref",
                )
                if raw["prediction_error_ref"] is not None
                else None
            ),
            credit_refs=tuple(
                _as_text(item, f"decision_trace.credit_refs[{index}]")
                for index, item in enumerate(raw_credit_refs)
            ),
            next_state_hash=(
                _as_text(
                    raw["next_state_hash"],
                    "decision_trace.next_state_hash",
                )
                if raw["next_state_hash"] is not None
                else None
            ),
        )
        _require_sha256(artifact_id, "artifact_id")
        if artifact_id != trace.artifact_id:
            raise ValueError(
                "decision trace artifact_id does not match canonical payload"
            )
        return trace


__all__ = [
    "CandidateOutcomePrediction",
    "OutcomeProbability",
    "PreActionRelationshipDecision",
    "RELATIONSHIP_ACTIONS",
    "RELATIONSHIP_DECISION_TRACE_SCHEMA_VERSION",
    "RELATIONSHIP_OUTCOMES",
    "RELATIONSHIP_PREACTION_SCHEMA_VERSION",
    "RelationshipAction",
    "RelationshipDatasetSplit",
    "RelationshipDecisionTrace",
    "RelationshipModelLineage",
    "canonical_json",
    "sha256_json",
]
