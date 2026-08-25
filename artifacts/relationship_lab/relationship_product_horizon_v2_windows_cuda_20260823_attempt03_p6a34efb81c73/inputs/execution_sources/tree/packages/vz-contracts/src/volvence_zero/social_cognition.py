"""Social cognition contracts (R16-R20).

This module is deliberately data-only. It lives in ``vz-contracts`` so
kernel owners, lifeform-side readouts, and evidence tooling can share the
same immutable shapes without reversing package dependencies.

SSOT split (oss-relationship-representation-standard.md, Phase A1): the
core ToM *representation* — :class:`OtherMindRecord` and its kind / status
enums — lives in ``companion_standard.social_cognition`` (the public
Relationship Representation Standard) and is re-exported here. Everything
else (owner snapshots with runtime diagnostics, social prediction / PE
records, common-ground, group state, lift helpers) is runtime contract and
stays private in this module.

The first landed slice covers R16 scaffolding: multi-party identity scope,
pre-action social predictions, and typed social prediction error records.
Runtime owners and propagation wiring are added in later phases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import TYPE_CHECKING

from companion_standard.social_cognition import (  # noqa: F401
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
)

if TYPE_CHECKING:
    from volvence_zero.llm_proposal_diagnostics import LLMProposalAttemptCounters


PRIMARY_INTERLOCUTOR_ID = "primary"
SELF_INTERLOCUTOR_ID = "self"
MAX_COMMON_GROUND_RECURSION_DEPTH = 2


class SocialScopeKind(str, Enum):
    """Scope for a social state or memory claim."""

    INTERLOCUTOR = "interlocutor"
    DYAD = "dyad"
    GROUP = "group"


class SocialPredictionKind(str, Enum):
    """Typed prediction classes emitted before a social action."""

    IDENTITY_ATTRIBUTION = "identity_attribution"
    AUDIENCE_SCOPE = "audience_scope"
    MEMORY_VISIBILITY = "memory_visibility"
    RELATIONSHIP_ATTRIBUTION = "relationship_attribution"
    ROLE_ASSIGNMENT = "role_assignment"
    COMMON_GROUND_RESOLUTION = "common_ground_resolution"
    GROUP_COMMITMENT_DURABILITY = "group_commitment_durability"
    BELIEF_ABOUT_OTHER = "belief_about_other"
    INTENT_ABOUT_OTHER = "intent_about_other"
    FEELING_ABOUT_OTHER = "feeling_about_other"
    PREFERENCE_ABOUT_OTHER = "preference_about_other"


class SocialPredictionOutcome(str, Enum):
    """Outcome class for a previously emitted social prediction."""

    CONFIRMED = "confirmed"
    DISCONFIRMED = "disconfirmed"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InterlocutorIdentity:
    interlocutor_id: str
    display_name: str | None = None
    aliases: tuple[str, ...] = ()
    confidence: float = 1.0
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("interlocutor_id", self.interlocutor_id)
        _require_confidence("confidence", self.confidence)
        _require_unique_non_empty("aliases", self.aliases)
        _require_non_empty_items("evidence", self.evidence)


@dataclass(frozen=True)
class SocialPrediction:
    prediction_id: str
    kind: SocialPredictionKind
    scope_kind: SocialScopeKind
    scope_id: str
    subject_ids: tuple[str, ...]
    audience_ids: tuple[str, ...]
    predicted_outcome: str
    confidence: float
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("prediction_id", self.prediction_id)
        _require_non_empty("scope_id", self.scope_id)
        _require_non_empty_unique_tuple("subject_ids", self.subject_ids)
        _require_non_empty_unique_tuple("audience_ids", self.audience_ids)
        _require_non_empty("predicted_outcome", self.predicted_outcome)
        _require_confidence("confidence", self.confidence)
        _require_non_empty_items("evidence", self.evidence)


@dataclass(frozen=True)
class SocialActionOutcomeProbability:
    """One typed outcome probability under a candidate social action.

    The kernel contract deliberately keeps ``outcome_id`` domain-neutral.
    A relationship vertical may freeze a closed vocabulary such as
    ``helped`` / ``felt_heard`` / ``missed`` / ``over_directive`` without
    making that product vocabulary a dependency of ``vz-contracts``.
    """

    outcome_id: str
    probability: float

    def __post_init__(self) -> None:
        _require_non_empty("outcome_id", self.outcome_id)
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError("probability must be finite and in [0, 1]")


@dataclass(frozen=True)
class SocialActionCandidatePrediction:
    """Pre-action outcome distribution for one candidate social action."""

    action_id: str
    outcomes: tuple[SocialActionOutcomeProbability, ...]

    def __post_init__(self) -> None:
        _require_non_empty("action_id", self.action_id)
        outcome_ids = tuple(outcome.outcome_id for outcome in self.outcomes)
        _require_non_empty_unique_tuple("outcomes.outcome_id", outcome_ids)
        total = math.fsum(outcome.probability for outcome in self.outcomes)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"candidate action outcome probabilities must sum to 1.0, got {total}")


@dataclass(frozen=True)
class PreferenceActionOutcomeEvidence:
    """Typed, already-observed action outcome owned by preference-about-other.

    This is past evidence, not a forecast settlement and not a learning signal.
    Keeping the action and outcome typed prevents a forecast collaborator from
    reparsing owner prose or receiving the raw transcript again after hydration.
    """

    evidence_id: str
    interlocutor_id: str
    observation_summary: str
    action_id: str
    observed_outcome_id: str
    reaction_summary: str
    source_turn: int
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("evidence_id", self.evidence_id),
            ("interlocutor_id", self.interlocutor_id),
            ("observation_summary", self.observation_summary),
            ("action_id", self.action_id),
            ("observed_outcome_id", self.observed_outcome_id),
            ("reaction_summary", self.reaction_summary),
        ):
            _require_non_empty(field_name, value)
        if isinstance(self.source_turn, bool) or not isinstance(self.source_turn, int):
            raise ValueError("source_turn must be an integer")
        if self.source_turn < 0:
            raise ValueError("source_turn must be >= 0")
        _require_non_empty_unique_tuple("evidence_refs", self.evidence_refs)


class PreferenceActionOutcomeMutationOperation(str, Enum):
    """User-directed changes to one persisted preference-action outcome."""

    CORRECT = "correct"
    REDACT = "redact"


@dataclass(frozen=True)
class PreferenceActionOutcomeMutation:
    """Optimistic-concurrency command consumed by the preference owner.

    A correction replaces the typed interpretation of an already-observed
    outcome. A redaction removes both that outcome and its paired owner record.
    Neither operation is PE, reward, evaluation, or a learning signal.
    """

    mutation_id: str
    target_evidence_id: str
    expected_evidence_sha256: str
    operation: PreferenceActionOutcomeMutationOperation
    requested_turn: int
    evidence_refs: tuple[str, ...]
    replacement: PreferenceActionOutcomeEvidence | None = None

    def __post_init__(self) -> None:
        _require_non_empty("mutation_id", self.mutation_id)
        _require_non_empty("target_evidence_id", self.target_evidence_id)
        if not isinstance(
            self.operation,
            PreferenceActionOutcomeMutationOperation,
        ):
            raise ValueError("operation must be a mutation operation enum")
        _require_sha256(
            "expected_evidence_sha256",
            self.expected_evidence_sha256,
        )
        if isinstance(self.requested_turn, bool) or not isinstance(
            self.requested_turn,
            int,
        ):
            raise ValueError("requested_turn must be an integer")
        if self.requested_turn < 0:
            raise ValueError("requested_turn must be >= 0")
        _require_opaque_refs("evidence_refs", self.evidence_refs)
        if self.operation is PreferenceActionOutcomeMutationOperation.CORRECT:
            if self.replacement is None:
                raise ValueError("correct mutation requires replacement evidence")
            if self.replacement.evidence_id != self.target_evidence_id:
                raise ValueError("correction replacement evidence_id must match target_evidence_id")
            if self.replacement.source_turn > self.requested_turn:
                raise ValueError("correction replacement cannot originate after requested_turn")
        elif self.replacement is not None:
            raise ValueError("redact mutation cannot carry replacement evidence")


@dataclass(frozen=True)
class PreferenceActionOutcomeMutationReceipt:
    """Content-safe audit/tombstone published after an owner mutation."""

    mutation_id: str
    command_sha256: str
    target_evidence_id: str
    operation: PreferenceActionOutcomeMutationOperation
    before_evidence_sha256: str
    after_evidence_sha256: str | None
    applied_turn: int
    invalidated_forecast_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty("mutation_id", self.mutation_id)
        _require_sha256("command_sha256", self.command_sha256)
        _require_non_empty("target_evidence_id", self.target_evidence_id)
        if not isinstance(
            self.operation,
            PreferenceActionOutcomeMutationOperation,
        ):
            raise ValueError("operation must be a mutation operation enum")
        _require_sha256(
            "before_evidence_sha256",
            self.before_evidence_sha256,
        )
        if self.operation is PreferenceActionOutcomeMutationOperation.CORRECT:
            if self.after_evidence_sha256 is None:
                raise ValueError("correct receipt requires after_evidence_sha256")
            _require_sha256(
                "after_evidence_sha256",
                self.after_evidence_sha256,
            )
        elif self.after_evidence_sha256 is not None:
            raise ValueError("redact receipt cannot carry after_evidence_sha256")
        if isinstance(self.applied_turn, bool) or not isinstance(
            self.applied_turn,
            int,
        ):
            raise ValueError("applied_turn must be an integer")
        if self.applied_turn < 0:
            raise ValueError("applied_turn must be >= 0")
        _require_unique_non_empty(
            "invalidated_forecast_ids",
            self.invalidated_forecast_ids,
        )
        _require_opaque_refs("evidence_refs", self.evidence_refs)


def preference_action_outcome_evidence_sha256(
    evidence: PreferenceActionOutcomeEvidence,
) -> str:
    """Return the canonical content hash used for optimistic mutation checks."""

    return _sha256_json(_preference_action_outcome_evidence_payload(evidence))


def preference_action_outcome_mutation_sha256(
    mutation: PreferenceActionOutcomeMutation,
) -> str:
    """Return a canonical command hash without retaining raw command text."""

    return _sha256_json(
        {
            "mutation_id": mutation.mutation_id,
            "target_evidence_id": mutation.target_evidence_id,
            "expected_evidence_sha256": mutation.expected_evidence_sha256,
            "operation": mutation.operation.value,
            "requested_turn": mutation.requested_turn,
            "evidence_refs": list(mutation.evidence_refs),
            "replacement": (
                _preference_action_outcome_evidence_payload(mutation.replacement)
                if mutation.replacement is not None
                else None
            ),
        }
    )


@dataclass(frozen=True)
class RelationshipConditionReadout:
    """Named pre-action relationship condition emitted by a frozen reader.

    The readout is owner-visible state, not evaluator truth.  It binds a
    human-readable abstract condition to the exact reader artifact and hashed
    public observation that produced it, so downstream policy code never has
    to recover the condition by parsing evidence prose.
    """

    condition_label: str
    confidence: float
    normalized_margin: float
    candidate_scores: tuple[tuple[str, float], ...]
    reader_artifact_id: str
    source_observation_sha256: str

    def __post_init__(self) -> None:
        _require_non_empty("condition_label", self.condition_label)
        _require_sha256("reader_artifact_id", self.reader_artifact_id)
        _require_sha256(
            "source_observation_sha256",
            self.source_observation_sha256,
        )
        for field_name, value in (
            ("confidence", self.confidence),
            ("normalized_margin", self.normalized_margin),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{field_name} must be finite and in [0, 1]")
        labels = tuple(label for label, _ in self.candidate_scores)
        _require_non_empty_unique_tuple("candidate_scores.label", labels)
        if len(labels) < 2:
            raise ValueError("candidate_scores requires at least two conditions")
        if self.condition_label not in labels:
            raise ValueError("condition_label must name one candidate score")
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not -1.0 <= score <= 1.0
            for _, score in self.candidate_scores
        ):
            raise ValueError("candidate scores must be finite cosine values in [-1, 1]")
        ordered = sorted(
            self.candidate_scores,
            key=lambda item: item[1],
            reverse=True,
        )
        if ordered[0][0] != self.condition_label:
            raise ValueError("condition_label must be the top-scoring candidate")
        expected_margin = min(
            1.0,
            max(0.0, (ordered[0][1] - ordered[1][1]) / 2.0),
        )
        if not math.isclose(
            self.normalized_margin,
            expected_margin,
            abs_tol=1e-12,
        ):
            raise ValueError("normalized_margin does not match candidate scores")


@dataclass(frozen=True)
class PreferenceActionForecast:
    """Owner-authored candidate-action forecast frozen before acting.

    This is a readout carried by :class:`PreferenceAboutOtherSnapshot`, not a
    second relationship-state owner.  It contains no observed outcome,
    evaluator label, reward, or credit.  Those may only be attached by a
    later post-action evidence record that references ``forecast_id``.
    """

    forecast_id: str
    decision_id: str
    interlocutor_id: str
    candidate_predictions: tuple[SocialActionCandidatePrediction, ...]
    recommended_action_id: str
    confidence: float
    source_record_ids: tuple[str, ...]
    issued_turn: int
    evidence: tuple[str, ...]
    # Empty is the backwards-compatible standalone/probe shape. P3 exact
    # settlement requires a non-empty session scope and rejects unscoped joins.
    session_scope: str = ""
    # P2d: optional named semantic readout.  ``None`` preserves every older
    # forecast; when present it is authored by this owner from a frozen reader
    # proposal and persists with the pending forecast.
    condition_readout: RelationshipConditionReadout | None = None

    def __post_init__(self) -> None:
        _require_non_empty("forecast_id", self.forecast_id)
        _require_non_empty("decision_id", self.decision_id)
        _require_non_empty("interlocutor_id", self.interlocutor_id)
        action_ids = tuple(prediction.action_id for prediction in self.candidate_predictions)
        if len(action_ids) < 2:
            raise ValueError("candidate_predictions must contain at least two social actions")
        _require_unique_non_empty(
            "candidate_predictions.action_id",
            action_ids,
        )
        outcome_vocabularies = {
            tuple(outcome.outcome_id for outcome in prediction.outcomes) for prediction in self.candidate_predictions
        }
        if len(outcome_vocabularies) != 1:
            raise ValueError("candidate predictions must share one ordered outcome vocabulary")
        if self.recommended_action_id not in action_ids:
            raise ValueError("recommended_action_id must name one of the candidate actions")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
        ):
            raise ValueError("confidence must be a finite number")
        _require_confidence("confidence", self.confidence)
        _require_unique_non_empty("source_record_ids", self.source_record_ids)
        if isinstance(self.issued_turn, bool) or not isinstance(self.issued_turn, int):
            raise ValueError("issued_turn must be an integer")
        if self.issued_turn < 0:
            raise ValueError("issued_turn must be >= 0")
        _require_non_empty_unique_tuple("evidence", self.evidence)


def preference_action_forecast_to_payload(
    forecast: PreferenceActionForecast,
) -> dict[str, object]:
    """Serialize one frozen forecast without adding owner-persistence policy.

    The payload deliberately matches the ``SocialRecordStore`` schema-v4
    forecast member byte shape. Artifact envelopes and their schema/version
    fields belong to the publisher of that artifact; this codec owns only the
    lossless immutable value shared by owners and evidence tooling.
    """

    if not isinstance(forecast, PreferenceActionForecast):
        raise TypeError("forecast must be a PreferenceActionForecast")
    return {
        "forecast_id": forecast.forecast_id,
        "decision_id": forecast.decision_id,
        "interlocutor_id": forecast.interlocutor_id,
        "candidate_predictions": [
            {
                "action_id": candidate.action_id,
                "outcomes": [
                    {
                        "outcome_id": outcome.outcome_id,
                        "probability": outcome.probability,
                    }
                    for outcome in candidate.outcomes
                ],
            }
            for candidate in forecast.candidate_predictions
        ],
        "recommended_action_id": forecast.recommended_action_id,
        "confidence": forecast.confidence,
        "source_record_ids": list(forecast.source_record_ids),
        "issued_turn": forecast.issued_turn,
        "evidence": list(forecast.evidence),
        "session_scope": forecast.session_scope,
        "condition_readout": (
            None
            if forecast.condition_readout is None
            else {
                "condition_label": forecast.condition_readout.condition_label,
                "confidence": forecast.condition_readout.confidence,
                "normalized_margin": forecast.condition_readout.normalized_margin,
                "candidate_scores": [
                    {"label": label, "score": score}
                    for label, score in forecast.condition_readout.candidate_scores
                ],
                "reader_artifact_id": forecast.condition_readout.reader_artifact_id,
                "source_observation_sha256": (
                    forecast.condition_readout.source_observation_sha256
                ),
            }
        ),
    }


def preference_action_forecast_from_payload(
    payload: object,
) -> PreferenceActionForecast:
    """Strictly reconstruct one frozen forecast from its canonical payload.

    This is intentionally not a compatibility decoder: unknown/missing keys,
    tuple-for-array substitutions, boolean numerics, and scalar coercions all
    fail loudly. Persistence owners must adapt their explicitly supported
    legacy schemas before calling this function.
    """

    raw = _forecast_payload_mapping(
        payload,
        expected={
            "forecast_id",
            "decision_id",
            "interlocutor_id",
            "candidate_predictions",
            "recommended_action_id",
            "confidence",
            "source_record_ids",
            "issued_turn",
            "evidence",
            "session_scope",
            "condition_readout",
        },
        field_name="PreferenceActionForecast",
    )
    candidates = tuple(
        _preference_action_candidate_from_payload(item, index=index)
        for index, item in enumerate(
            _forecast_payload_list(
                raw["candidate_predictions"],
                field_name="PreferenceActionForecast.candidate_predictions",
            )
        )
    )
    source_record_ids = tuple(
        _forecast_payload_text(
            item,
            field_name=f"PreferenceActionForecast.source_record_ids[{index}]",
        )
        for index, item in enumerate(
            _forecast_payload_list(
                raw["source_record_ids"],
                field_name="PreferenceActionForecast.source_record_ids",
            )
        )
    )
    evidence = tuple(
        _forecast_payload_text(
            item,
            field_name=f"PreferenceActionForecast.evidence[{index}]",
        )
        for index, item in enumerate(
            _forecast_payload_list(
                raw["evidence"],
                field_name="PreferenceActionForecast.evidence",
            )
        )
    )
    condition_payload = raw["condition_readout"]
    condition_readout = (
        None
        if condition_payload is None
        else _relationship_condition_readout_from_payload(condition_payload)
    )
    return PreferenceActionForecast(
        forecast_id=_forecast_payload_text(
            raw["forecast_id"],
            field_name="PreferenceActionForecast.forecast_id",
        ),
        decision_id=_forecast_payload_text(
            raw["decision_id"],
            field_name="PreferenceActionForecast.decision_id",
        ),
        interlocutor_id=_forecast_payload_text(
            raw["interlocutor_id"],
            field_name="PreferenceActionForecast.interlocutor_id",
        ),
        candidate_predictions=candidates,
        recommended_action_id=_forecast_payload_text(
            raw["recommended_action_id"],
            field_name="PreferenceActionForecast.recommended_action_id",
        ),
        confidence=_forecast_payload_number(
            raw["confidence"],
            field_name="PreferenceActionForecast.confidence",
        ),
        source_record_ids=source_record_ids,
        issued_turn=_forecast_payload_integer(
            raw["issued_turn"],
            field_name="PreferenceActionForecast.issued_turn",
        ),
        evidence=evidence,
        session_scope=_forecast_payload_text(
            raw["session_scope"],
            field_name="PreferenceActionForecast.session_scope",
            allow_empty=True,
        ),
        condition_readout=condition_readout,
    )


def _preference_action_candidate_from_payload(
    payload: object,
    *,
    index: int,
) -> SocialActionCandidatePrediction:
    field_name = f"PreferenceActionForecast.candidate_predictions[{index}]"
    raw = _forecast_payload_mapping(
        payload,
        expected={"action_id", "outcomes"},
        field_name=field_name,
    )
    outcomes = tuple(
        _preference_action_outcome_probability_from_payload(
            item,
            field_name=f"{field_name}.outcomes[{outcome_index}]",
        )
        for outcome_index, item in enumerate(
            _forecast_payload_list(
                raw["outcomes"],
                field_name=f"{field_name}.outcomes",
            )
        )
    )
    return SocialActionCandidatePrediction(
        action_id=_forecast_payload_text(
            raw["action_id"],
            field_name=f"{field_name}.action_id",
        ),
        outcomes=outcomes,
    )


def _preference_action_outcome_probability_from_payload(
    payload: object,
    *,
    field_name: str,
) -> SocialActionOutcomeProbability:
    raw = _forecast_payload_mapping(
        payload,
        expected={"outcome_id", "probability"},
        field_name=field_name,
    )
    return SocialActionOutcomeProbability(
        outcome_id=_forecast_payload_text(
            raw["outcome_id"],
            field_name=f"{field_name}.outcome_id",
        ),
        probability=_forecast_payload_number(
            raw["probability"],
            field_name=f"{field_name}.probability",
        ),
    )


def _relationship_condition_readout_from_payload(
    payload: object,
) -> RelationshipConditionReadout:
    field_name = "PreferenceActionForecast.condition_readout"
    raw = _forecast_payload_mapping(
        payload,
        expected={
            "condition_label",
            "confidence",
            "normalized_margin",
            "candidate_scores",
            "reader_artifact_id",
            "source_observation_sha256",
        },
        field_name=field_name,
    )
    candidate_scores: list[tuple[str, float]] = []
    for index, item in enumerate(
        _forecast_payload_list(
            raw["candidate_scores"],
            field_name=f"{field_name}.candidate_scores",
        )
    ):
        score_field = f"{field_name}.candidate_scores[{index}]"
        score_payload = _forecast_payload_mapping(
            item,
            expected={"label", "score"},
            field_name=score_field,
        )
        candidate_scores.append(
            (
                _forecast_payload_text(
                    score_payload["label"],
                    field_name=f"{score_field}.label",
                ),
                _forecast_payload_number(
                    score_payload["score"],
                    field_name=f"{score_field}.score",
                ),
            )
        )
    return RelationshipConditionReadout(
        condition_label=_forecast_payload_text(
            raw["condition_label"],
            field_name=f"{field_name}.condition_label",
        ),
        confidence=_forecast_payload_number(
            raw["confidence"],
            field_name=f"{field_name}.confidence",
        ),
        normalized_margin=_forecast_payload_number(
            raw["normalized_margin"],
            field_name=f"{field_name}.normalized_margin",
        ),
        candidate_scores=tuple(candidate_scores),
        reader_artifact_id=_forecast_payload_text(
            raw["reader_artifact_id"],
            field_name=f"{field_name}.reader_artifact_id",
        ),
        source_observation_sha256=_forecast_payload_text(
            raw["source_observation_sha256"],
            field_name=f"{field_name}.source_observation_sha256",
        ),
    )


def _forecast_payload_mapping(
    value: object,
    *,
    expected: set[str],
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(
            f"{field_name} fields do not match schema; "
            f"missing={missing}, extra={extra}"
        )
    return value


def _forecast_payload_list(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be an array")
    return value


def _forecast_payload_text(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "text" if allow_empty else "non-empty text"
        raise TypeError(f"{field_name} must be {qualifier}")
    return value


def _forecast_payload_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(f"{field_name} must be finite") from exc
    if not finite:
        raise ValueError(f"{field_name} must be finite")
    return value


def _forecast_payload_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


@dataclass(frozen=True)
class PreferenceActionForecastSettlement:
    """Owner-authored exact settlement of one pre-action forecast."""

    settlement_id: str
    forecast_id: str
    decision_id: str
    session_scope: str
    interlocutor_id: str
    action_id: str
    observed_outcome_id: str
    predicted_probability: float
    negative_log_likelihood: float
    outcome: SocialPredictionOutcome
    magnitude: float
    source_evidence_id: str
    forecast_issued_turn: int
    observed_turn: int
    evidence_confidence: float = 1.0
    expected_utility: float = 0.0
    observed_utility: float = 0.0
    signed_utility_prediction_error: float = 0.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("settlement_id", self.settlement_id),
            ("forecast_id", self.forecast_id),
            ("decision_id", self.decision_id),
            ("session_scope", self.session_scope),
            ("interlocutor_id", self.interlocutor_id),
            ("action_id", self.action_id),
            ("observed_outcome_id", self.observed_outcome_id),
            ("source_evidence_id", self.source_evidence_id),
        ):
            _require_non_empty(field_name, value)
        for field_name, value in (
            ("predicted_probability", self.predicted_probability),
            ("magnitude", self.magnitude),
            ("evidence_confidence", self.evidence_confidence),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{field_name} must be finite and in [0, 1]")
        for field_name, value in (
            ("expected_utility", self.expected_utility),
            ("observed_utility", self.observed_utility),
            (
                "signed_utility_prediction_error",
                self.signed_utility_prediction_error,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not -1.0 <= value <= 1.0
            ):
                raise ValueError(f"{field_name} must be finite and in [-1, 1]")
        if (
            isinstance(self.negative_log_likelihood, bool)
            or not isinstance(self.negative_log_likelihood, (int, float))
            or not math.isfinite(self.negative_log_likelihood)
            or self.negative_log_likelihood < 0.0
        ):
            raise ValueError("negative_log_likelihood must be finite and >= 0")
        for field_name, value in (
            ("forecast_issued_turn", self.forecast_issued_turn),
            ("observed_turn", self.observed_turn),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.observed_turn < self.forecast_issued_turn:
            raise ValueError("observed_turn cannot precede forecast_issued_turn")


@dataclass(frozen=True)
class SocialPredictionError:
    error_id: str
    prediction_id: str
    kind: SocialPredictionKind
    outcome: SocialPredictionOutcome
    magnitude: float
    owner: str
    scope_kind: SocialScopeKind
    scope_id: str
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty("error_id", self.error_id)
        _require_non_empty("prediction_id", self.prediction_id)
        _require_unit_interval("magnitude", self.magnitude)
        _require_non_empty("owner", self.owner)
        _require_non_empty("scope_id", self.scope_id)
        _require_unique_non_empty("evidence", self.evidence)


@dataclass(frozen=True)
class SocialPredictionSnapshot:
    predictions: tuple[SocialPrediction, ...]
    description: str

    def __post_init__(self) -> None:
        prediction_ids = tuple(prediction.prediction_id for prediction in self.predictions)
        _require_unique_non_empty("predictions.prediction_id", prediction_ids)
        _require_non_empty("description", self.description)


@dataclass(frozen=True)
class SocialPredictionErrorSnapshot:
    errors: tuple[SocialPredictionError, ...]
    description: str

    def __post_init__(self) -> None:
        error_ids = tuple(error.error_id for error in self.errors)
        _require_unique_non_empty("errors.error_id", error_ids)
        _require_non_empty("description", self.description)


@dataclass(frozen=True)
class MemorySocialPESignal:
    """Typed PE signal published by ``MemoryModule`` itself.

    R8 SSOT contract: only the owning ``MemoryModule`` writes this
    record into its own ``MemorySnapshot.social_pe_signals``. Downstream
    social prediction / error owners lift each signal into
    :class:`SocialPrediction` / :class:`SocialPredictionError` via the
    pure helpers below; they never reconstruct it from raw memory
    fields and they never borrow another owner's name on the resulting
    public records.

    The signal carries both the pre-action prediction shape (so the
    aggregator can publish a stable :class:`SocialPrediction`) and the
    optional settled outcome (so the error owner can publish a stable
    :class:`SocialPredictionError`). When ``outcome`` is ``None`` the
    signal is prediction-only and the error owner skips it.
    """

    signal_id: str
    prediction_id: str
    source_owner: str
    prediction_kind: SocialPredictionKind
    scope_kind: SocialScopeKind
    scope_id: str
    subject_ids: tuple[str, ...]
    audience_ids: tuple[str, ...]
    predicted_outcome: str
    confidence: float
    outcome: SocialPredictionOutcome | None = None
    magnitude: float = 0.0
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("signal_id", self.signal_id)
        _require_non_empty("prediction_id", self.prediction_id)
        _require_non_empty("source_owner", self.source_owner)
        _require_non_empty("scope_id", self.scope_id)
        _require_non_empty_unique_tuple("subject_ids", self.subject_ids)
        _require_non_empty_unique_tuple("audience_ids", self.audience_ids)
        _require_non_empty("predicted_outcome", self.predicted_outcome)
        _require_confidence("confidence", self.confidence)
        _require_unit_interval("magnitude", self.magnitude)
        _require_unique_non_empty("evidence", self.evidence)
        if self.outcome is None and self.evidence:
            raise ValueError("MemorySocialPESignal.evidence must be empty when outcome is None")
        if self.outcome is not None and not self.evidence:
            raise ValueError("MemorySocialPESignal.evidence must be non-empty when outcome is set")


def build_memory_visibility_signals(
    *,
    source_owner: str,
    sequence_index: int,
    active_subject_scope: tuple[str, ...],
    retrieved_count: int,
    suppressed_evidence: tuple[str, ...],
    audience_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,),
    pre_action_confidence: float = 0.6,
) -> tuple[MemorySocialPESignal, ...]:
    """Build typed memory-visibility PE signals for one retrieval cycle.

    Pure functional contract helper. The owning ``MemoryModule`` calls
    this once per ``process`` after running scoped retrieval and forwards
    the result through ``MemorySnapshot.social_pe_signals``. A single
    signal is emitted when the active multi-party scope is non-default;
    the signal carries an outcome (``DISCONFIRMED`` + magnitude) only
    when cross-scope memory entries were actually suppressed.

    Returns an empty tuple when the scope is default or empty so the
    caller can pass the result straight through without branching.
    """

    if not active_subject_scope:
        return ()
    if active_subject_scope == (PRIMARY_INTERLOCUTOR_ID,):
        return ()

    scope_id = active_subject_scope[0]
    seq_token = f"v{sequence_index}"
    prediction_id = f"memory_visibility:{scope_id}:{seq_token}"
    signal_id = f"memory_visibility_pe:{scope_id}:{seq_token}"

    suppressed_count = len(suppressed_evidence)
    if suppressed_count > 0:
        evaluated_total = retrieved_count + suppressed_count
        magnitude = suppressed_count / evaluated_total if evaluated_total > 0 else 1.0
        magnitude = min(1.0, max(0.0, magnitude))
        outcome: SocialPredictionOutcome | None = SocialPredictionOutcome.DISCONFIRMED
        evidence = suppressed_evidence
    else:
        outcome = None
        magnitude = 0.0
        evidence = ()

    return (
        MemorySocialPESignal(
            signal_id=signal_id,
            prediction_id=prediction_id,
            source_owner=source_owner,
            prediction_kind=SocialPredictionKind.MEMORY_VISIBILITY,
            scope_kind=SocialScopeKind.INTERLOCUTOR,
            scope_id=scope_id,
            subject_ids=active_subject_scope,
            audience_ids=audience_ids,
            predicted_outcome="memory_subjects_match_active_subjects",
            confidence=pre_action_confidence,
            outcome=outcome,
            magnitude=magnitude,
            evidence=evidence,
        ),
    )


def social_prediction_from_memory_signal(
    signal: MemorySocialPESignal,
    *,
    extra_evidence: tuple[str, ...] = (),
) -> SocialPrediction:
    """Lift a memory PE signal to a public :class:`SocialPrediction`.

    Used by the social-prediction aggregator. ``extra_evidence`` lets the
    aggregator append contextual evidence (e.g. retrieved-count summary)
    without touching the owner's signal.
    """

    return SocialPrediction(
        prediction_id=signal.prediction_id,
        kind=signal.prediction_kind,
        scope_kind=signal.scope_kind,
        scope_id=signal.scope_id,
        subject_ids=signal.subject_ids,
        audience_ids=signal.audience_ids,
        predicted_outcome=signal.predicted_outcome,
        confidence=signal.confidence,
        evidence=tuple(extra_evidence),
    )


def social_prediction_error_from_memory_signal(
    signal: MemorySocialPESignal,
) -> SocialPredictionError | None:
    """Lift a settled memory PE signal to a public :class:`SocialPredictionError`.

    Returns ``None`` when the signal is prediction-only
    (``outcome is None``). The resulting error's ``owner`` field comes
    from the signal's ``source_owner``, so the SSOT contract is
    preserved: the memory module owns the PE source, this helper only
    converts the typed signal into the public PE record.
    """

    if signal.outcome is None:
        return None
    return SocialPredictionError(
        error_id=signal.signal_id,
        prediction_id=signal.prediction_id,
        kind=signal.prediction_kind,
        outcome=signal.outcome,
        magnitude=signal.magnitude,
        owner=signal.source_owner,
        scope_kind=signal.scope_kind,
        scope_id=signal.scope_id,
        evidence=signal.evidence,
    )


@dataclass(frozen=True)
class BeliefAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str
    proposal_diagnostics: "LLMProposalAttemptCounters | None" = None
    # W1.C (CP-16): owner-settled outcomes for predictions this owner
    # issued on a PRIOR turn. Only the owning module constructs these;
    # SocialPredictionErrorModule forwards them without reconstruction.
    settled_errors: tuple[SocialPredictionError, ...] = ()

    def __post_init__(self) -> None:
        _validate_other_mind_snapshot(
            snapshot_name="BeliefAboutOtherSnapshot",
            expected_kind=OtherMindRecordKind.BELIEF,
            records=self.records,
            active_predictions=self.active_predictions,
            control_signal=self.control_signal,
            description=self.description,
            settled_errors=self.settled_errors,
        )


@dataclass(frozen=True)
class IntentAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str
    proposal_diagnostics: "LLMProposalAttemptCounters | None" = None
    settled_errors: tuple[SocialPredictionError, ...] = ()

    def __post_init__(self) -> None:
        _validate_other_mind_snapshot(
            snapshot_name="IntentAboutOtherSnapshot",
            expected_kind=OtherMindRecordKind.INTENT,
            records=self.records,
            active_predictions=self.active_predictions,
            control_signal=self.control_signal,
            description=self.description,
            settled_errors=self.settled_errors,
        )


@dataclass(frozen=True)
class FeelingAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str
    proposal_diagnostics: "LLMProposalAttemptCounters | None" = None
    settled_errors: tuple[SocialPredictionError, ...] = ()

    def __post_init__(self) -> None:
        _validate_other_mind_snapshot(
            snapshot_name="FeelingAboutOtherSnapshot",
            expected_kind=OtherMindRecordKind.FEELING,
            records=self.records,
            active_predictions=self.active_predictions,
            control_signal=self.control_signal,
            description=self.description,
            settled_errors=self.settled_errors,
        )


@dataclass(frozen=True)
class PreferenceAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str
    proposal_diagnostics: "LLMProposalAttemptCounters | None" = None
    settled_errors: tuple[SocialPredictionError, ...] = ()
    # P2a contract-only surface. The existing owner publishes an empty tuple
    # until the P2 SHADOW producer is wired; expression/planner consumers do
    # not read this field.
    action_forecasts: tuple[PreferenceActionForecast, ...] = ()
    # P2c: past typed action/outcome evidence retained by this owner. This is
    # distinct from a future observed outcome and contains no PE/credit.
    action_outcome_evidence: tuple[PreferenceActionOutcomeEvidence, ...] = ()
    # P3: exact post-action joins authored by this owner. Kept separate from
    # action_forecasts so a consumer never splices labels into a pre-action row.
    forecast_settlements: tuple[PreferenceActionForecastSettlement, ...] = ()
    # P4.2: user-directed correction/redaction audit. Redaction receipts are
    # durable tombstones and deliberately contain hashes/refs rather than the
    # removed observation or reaction text.
    action_outcome_mutation_receipts: tuple[PreferenceActionOutcomeMutationReceipt, ...] = ()

    def __post_init__(self) -> None:
        _validate_other_mind_snapshot(
            snapshot_name="PreferenceAboutOtherSnapshot",
            expected_kind=OtherMindRecordKind.PREFERENCE,
            records=self.records,
            active_predictions=self.active_predictions,
            control_signal=self.control_signal,
            description=self.description,
            settled_errors=self.settled_errors,
        )
        _validate_preference_action_forecasts(
            records=self.records,
            forecasts=self.action_forecasts,
        )
        _validate_preference_action_outcome_evidence(
            records=self.records,
            action_outcomes=self.action_outcome_evidence,
        )
        _validate_preference_forecast_settlements(self.forecast_settlements)
        _validate_preference_action_outcome_mutation_receipts(
            records=self.records,
            action_outcomes=self.action_outcome_evidence,
            receipts=self.action_outcome_mutation_receipts,
        )


@dataclass(frozen=True)
class ToMInterlocutorRecordCount:
    """COG-2 evidence readout keyed by interlocutor.

    Counts are derived from public ToM owner snapshots only. Benchmark and
    evaluation code should consume this readout instead of traversing owner
    internals or rebuilding per-person state from raw text.
    """

    interlocutor_id: str
    belief_count: int = 0
    intent_count: int = 0
    feeling_count: int = 0
    preference_count: int = 0

    @property
    def total_count(self) -> int:
        return self.belief_count + self.intent_count + self.feeling_count + self.preference_count

    def __post_init__(self) -> None:
        _require_non_empty("interlocutor_id", self.interlocutor_id)
        for field_name, value in (
            ("belief_count", self.belief_count),
            ("intent_count", self.intent_count),
            ("feeling_count", self.feeling_count),
            ("preference_count", self.preference_count),
        ):
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0, got {value!r}")


def tom_record_counts_by_interlocutor(
    *,
    belief: BeliefAboutOtherSnapshot | None = None,
    intent: IntentAboutOtherSnapshot | None = None,
    feeling: FeelingAboutOtherSnapshot | None = None,
    preference: PreferenceAboutOtherSnapshot | None = None,
) -> tuple[ToMInterlocutorRecordCount, ...]:
    """Aggregate public ToM records by interlocutor id.

    This is a pure readout helper for COG-2 evidence. It does not own ToM
    state and it does not infer anything from text.
    """

    counts: dict[str, dict[str, int]] = {}

    def bump(interlocutor_id: str, key: str) -> None:
        bucket = counts.setdefault(
            interlocutor_id,
            {
                "belief_count": 0,
                "intent_count": 0,
                "feeling_count": 0,
                "preference_count": 0,
            },
        )
        bucket[key] += 1

    if belief is not None:
        for record in belief.records:
            bump(record.interlocutor_id, "belief_count")
    if intent is not None:
        for record in intent.records:
            bump(record.interlocutor_id, "intent_count")
    if feeling is not None:
        for record in feeling.records:
            bump(record.interlocutor_id, "feeling_count")
    if preference is not None:
        for record in preference.records:
            bump(record.interlocutor_id, "preference_count")

    return tuple(
        ToMInterlocutorRecordCount(
            interlocutor_id=interlocutor_id,
            belief_count=values["belief_count"],
            intent_count=values["intent_count"],
            feeling_count=values["feeling_count"],
            preference_count=values["preference_count"],
        )
        for interlocutor_id, values in sorted(counts.items())
    )


@dataclass(frozen=True)
class ConversationalRoleSnapshot:
    active_speaker_id: str
    addressee_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    witness_ids: tuple[str, ...]
    overhearer_ids: tuple[str, ...]
    group_audience_ids: tuple[str, ...]
    role_confidence: float
    active_predictions: tuple[SocialPrediction, ...]
    description: str

    def __post_init__(self) -> None:
        _require_non_empty("active_speaker_id", self.active_speaker_id)
        _require_non_empty_unique_tuple("addressee_ids", self.addressee_ids)
        _require_non_empty_unique_tuple("subject_ids", self.subject_ids)
        _require_unique_non_empty("witness_ids", self.witness_ids)
        _require_unique_non_empty("overhearer_ids", self.overhearer_ids)
        _require_unique_non_empty("group_audience_ids", self.group_audience_ids)
        _require_confidence("role_confidence", self.role_confidence)
        prediction_ids = tuple(prediction.prediction_id for prediction in self.active_predictions)
        _require_unique_non_empty("active_predictions.prediction_id", prediction_ids)
        _require_non_empty("description", self.description)


@dataclass(frozen=True)
class CommonGroundAtom:
    atom_id: str
    scope_id: str
    scope_kind: SocialScopeKind
    summary: str
    recursion_depth: int
    confidence: float
    accepted_by_ids: tuple[str, ...]
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty("atom_id", self.atom_id)
        _require_non_empty("scope_id", self.scope_id)
        if self.scope_kind not in {SocialScopeKind.DYAD, SocialScopeKind.GROUP}:
            raise ValueError(f"CommonGroundAtom.scope_kind must be dyad or group; got {self.scope_kind.value}")
        _require_non_empty("summary", self.summary)
        if self.recursion_depth < 0 or self.recursion_depth > MAX_COMMON_GROUND_RECURSION_DEPTH:
            raise ValueError(
                "recursion_depth must be between 0 and "
                f"{MAX_COMMON_GROUND_RECURSION_DEPTH}, got {self.recursion_depth!r}"
            )
        _require_confidence("confidence", self.confidence)
        _require_non_empty_unique_tuple("accepted_by_ids", self.accepted_by_ids)
        _require_unique_non_empty("evidence", self.evidence)


@dataclass(frozen=True)
class CommonGroundSnapshot:
    dyad_atoms: tuple[CommonGroundAtom, ...]
    group_atoms: tuple[CommonGroundAtom, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str
    proposal_diagnostics: "LLMProposalAttemptCounters | None" = None
    # W1.C (CP-17): owner-settled outcomes for common-ground predictions
    # issued on a prior turn (confirm / disconfirm from this turn's atoms).
    settled_errors: tuple[SocialPredictionError, ...] = ()

    def __post_init__(self) -> None:
        _validate_common_ground_atoms("dyad_atoms", self.dyad_atoms, SocialScopeKind.DYAD)
        _validate_common_ground_atoms("group_atoms", self.group_atoms, SocialScopeKind.GROUP)
        prediction_ids = tuple(prediction.prediction_id for prediction in self.active_predictions)
        _require_unique_non_empty("active_predictions.prediction_id", prediction_ids)
        settled_ids = tuple(error.error_id for error in self.settled_errors)
        _require_unique_non_empty("settled_errors.error_id", settled_ids)
        _require_unit_interval("control_signal", self.control_signal)
        _require_non_empty("description", self.description)


@dataclass(frozen=True)
class GroupIdentity:
    group_id: str
    member_ids: tuple[str, ...]
    display_name: str | None = None
    confidence: float = 1.0
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("group_id", self.group_id)
        _require_non_empty_unique_tuple("member_ids", self.member_ids)
        if self.display_name is not None:
            _require_non_empty("display_name", self.display_name)
        _require_confidence("confidence", self.confidence)
        _require_unique_non_empty("evidence", self.evidence)


@dataclass(frozen=True)
class GroupSnapshot:
    groups: tuple[GroupIdentity, ...]
    active_group_id: str | None
    joint_attention: tuple[str, ...]
    joint_commitments: tuple[str, ...]
    group_regime_id: str | None
    active_predictions: tuple[SocialPrediction, ...]
    description: str
    # G1 (CP-18): owner-settled outcomes for group-durability predictions
    # issued on a prior turn (same settlement contract the ToM /
    # common-ground owners use). SocialPredictionErrorModule forwards
    # these; the group-level PE is never reconstructed downstream.
    settled_errors: tuple[SocialPredictionError, ...] = ()
    # Learned commitment-durability score for the active group
    # ([0,1], store-held, PE-settlement updated). 0.5 = uninformed
    # prior; published so consumers read the learned state instead of
    # rebuilding it from settlement history.
    group_durability_score: float = 0.5

    def __post_init__(self) -> None:
        group_ids = tuple(group.group_id for group in self.groups)
        _require_unique_non_empty("groups.group_id", group_ids)
        if self.active_group_id is not None:
            _require_non_empty("active_group_id", self.active_group_id)
            if self.active_group_id not in group_ids:
                raise ValueError(f"active_group_id must refer to one of groups.group_id; got {self.active_group_id!r}")
        _require_unique_non_empty("joint_attention", self.joint_attention)
        _require_unique_non_empty("joint_commitments", self.joint_commitments)
        if self.group_regime_id is not None:
            _require_non_empty("group_regime_id", self.group_regime_id)
        prediction_ids = tuple(prediction.prediction_id for prediction in self.active_predictions)
        _require_unique_non_empty("active_predictions.prediction_id", prediction_ids)
        settled_ids = tuple(error.error_id for error in self.settled_errors)
        _require_unique_non_empty("settled_errors.error_id", settled_ids)
        _require_unit_interval("group_durability_score", self.group_durability_score)
        _require_non_empty("description", self.description)


@dataclass(frozen=True)
class MultiPartyIdentitySnapshot:
    active_speaker_id: str
    addressee_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    audience_ids: tuple[str, ...]
    interlocutors: tuple[InterlocutorIdentity, ...]
    identity_predictions: tuple[SocialPrediction, ...]
    description: str

    def __post_init__(self) -> None:
        _require_non_empty("active_speaker_id", self.active_speaker_id)
        _require_non_empty_unique_tuple("addressee_ids", self.addressee_ids)
        _require_non_empty_unique_tuple("subject_ids", self.subject_ids)
        _require_non_empty_unique_tuple("audience_ids", self.audience_ids)
        _require_non_empty("description", self.description)
        identity_ids = tuple(identity.interlocutor_id for identity in self.interlocutors)
        _require_unique_non_empty("interlocutors.interlocutor_id", identity_ids)
        if self.active_speaker_id not in identity_ids:
            raise ValueError("MultiPartyIdentitySnapshot.active_speaker_id must be present in interlocutors")


def build_primary_multi_party_identity_snapshot(
    *,
    description: str = "Single-interlocutor compatibility identity scope.",
) -> MultiPartyIdentitySnapshot:
    """Return the neutral single-party compatibility snapshot.

    ``primary`` is a migration key used while flat single-user state is
    retired. It is not a claim that future social cognition is single-party.
    """

    primary = InterlocutorIdentity(
        interlocutor_id=PRIMARY_INTERLOCUTOR_ID,
        display_name=None,
        aliases=(),
        confidence=1.0,
        evidence=("single-party compatibility default",),
    )
    return MultiPartyIdentitySnapshot(
        active_speaker_id=PRIMARY_INTERLOCUTOR_ID,
        addressee_ids=(SELF_INTERLOCUTOR_ID,),
        subject_ids=(PRIMARY_INTERLOCUTOR_ID,),
        audience_ids=(SELF_INTERLOCUTOR_ID,),
        interlocutors=(primary,),
        identity_predictions=(),
        description=description,
    )


def build_primary_conversational_role_snapshot(
    *,
    description: str = "Single-interlocutor compatibility conversational role.",
) -> ConversationalRoleSnapshot:
    return ConversationalRoleSnapshot(
        active_speaker_id=PRIMARY_INTERLOCUTOR_ID,
        addressee_ids=(SELF_INTERLOCUTOR_ID,),
        subject_ids=(PRIMARY_INTERLOCUTOR_ID,),
        witness_ids=(),
        overhearer_ids=(),
        group_audience_ids=(),
        role_confidence=1.0,
        active_predictions=(),
        description=description,
    )


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_non_empty_items(field_name: str, values: tuple[str, ...]) -> None:
    for value in values:
        if not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty")


def _require_unique_non_empty(field_name: str, values: tuple[str, ...]) -> None:
    _require_non_empty_items(field_name, values)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} entries must be unique")


def _require_non_empty_unique_tuple(field_name: str, values: tuple[str, ...]) -> None:
    if not values:
        raise ValueError(f"{field_name} must contain at least one entry")
    _require_unique_non_empty(field_name, values)


def _require_opaque_refs(field_name: str, values: tuple[str, ...]) -> None:
    _require_non_empty_unique_tuple(field_name, values)
    if any(any(character.isspace() for character in value) for value in values):
        raise ValueError(f"{field_name} entries must be opaque references")


def _require_sha256(field_name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")


def _preference_action_outcome_evidence_payload(
    evidence: PreferenceActionOutcomeEvidence,
) -> dict[str, object]:
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


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_confidence(field_name: str, value: float) -> None:
    _require_unit_interval(field_name, value)


def _require_unit_interval(field_name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")


def _validate_other_mind_snapshot(
    *,
    snapshot_name: str,
    expected_kind: OtherMindRecordKind,
    records: tuple[OtherMindRecord, ...],
    active_predictions: tuple[SocialPrediction, ...],
    control_signal: float,
    description: str,
    settled_errors: tuple[SocialPredictionError, ...] = (),
) -> None:
    record_ids = tuple(record.record_id for record in records)
    _require_unique_non_empty(f"{snapshot_name}.records.record_id", record_ids)
    for record in records:
        if record.kind is not expected_kind:
            raise ValueError(
                f"{snapshot_name} records must have kind={expected_kind.value}; "
                f"got {record.kind.value} for {record.record_id!r}"
            )
    prediction_ids = tuple(prediction.prediction_id for prediction in active_predictions)
    _require_unique_non_empty(f"{snapshot_name}.active_predictions.prediction_id", prediction_ids)
    settled_ids = tuple(error.error_id for error in settled_errors)
    _require_unique_non_empty(f"{snapshot_name}.settled_errors.error_id", settled_ids)
    _require_unit_interval("control_signal", control_signal)
    _require_non_empty("description", description)


def _validate_preference_action_forecasts(
    *,
    records: tuple[OtherMindRecord, ...],
    forecasts: tuple[PreferenceActionForecast, ...],
) -> None:
    forecast_ids = tuple(forecast.forecast_id for forecast in forecasts)
    _require_unique_non_empty(
        "PreferenceAboutOtherSnapshot.action_forecasts.forecast_id",
        forecast_ids,
    )
    decision_scopes = tuple((forecast.decision_id, forecast.interlocutor_id) for forecast in forecasts)
    if len(set(decision_scopes)) != len(decision_scopes):
        raise ValueError(
            "PreferenceAboutOtherSnapshot.action_forecasts must contain at most "
            "one forecast per decision and interlocutor"
        )
    records_by_id = {record.record_id: record for record in records}
    for forecast in forecasts:
        for record_id in forecast.source_record_ids:
            try:
                record = records_by_id[record_id]
            except KeyError as exc:
                raise ValueError(
                    "PreferenceAboutOtherSnapshot.action_forecasts source_record_ids "
                    f"references unknown record {record_id!r}"
                ) from exc
            if record.interlocutor_id != forecast.interlocutor_id:
                raise ValueError(
                    "PreferenceAboutOtherSnapshot.action_forecasts cannot source a record from another interlocutor"
                )
            if record.source_turn > forecast.issued_turn:
                raise ValueError(
                    "PreferenceAboutOtherSnapshot.action_forecasts cannot source a record from a future turn"
                )


def _validate_preference_action_outcome_evidence(
    *,
    records: tuple[OtherMindRecord, ...],
    action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
) -> None:
    evidence_ids = tuple(item.evidence_id for item in action_outcomes)
    _require_unique_non_empty(
        "PreferenceAboutOtherSnapshot.action_outcome_evidence.evidence_id",
        evidence_ids,
    )
    records_by_id = {record.record_id: record for record in records}
    for item in action_outcomes:
        try:
            record = records_by_id[item.evidence_id]
        except KeyError as exc:
            raise ValueError(
                "PreferenceAboutOtherSnapshot.action_outcome_evidence must "
                f"reference an owner record; missing {item.evidence_id!r}"
            ) from exc
        if record.interlocutor_id != item.interlocutor_id:
            raise ValueError(
                "PreferenceAboutOtherSnapshot.action_outcome_evidence cannot "
                "reference a record from another interlocutor"
            )
        if record.source_turn != item.source_turn:
            raise ValueError(
                "PreferenceAboutOtherSnapshot.action_outcome_evidence source_turn must match its owner record"
            )


def _validate_preference_forecast_settlements(
    settlements: tuple[PreferenceActionForecastSettlement, ...],
) -> None:
    settlement_ids = tuple(item.settlement_id for item in settlements)
    _require_unique_non_empty(
        "PreferenceAboutOtherSnapshot.forecast_settlements.settlement_id",
        settlement_ids,
    )
    forecast_ids = tuple(item.forecast_id for item in settlements)
    if len(set(forecast_ids)) != len(forecast_ids):
        raise ValueError("PreferenceAboutOtherSnapshot.forecast_settlements may settle a forecast at most once")


def _validate_preference_action_outcome_mutation_receipts(
    *,
    records: tuple[OtherMindRecord, ...],
    action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    receipts: tuple[PreferenceActionOutcomeMutationReceipt, ...],
) -> None:
    mutation_ids = tuple(receipt.mutation_id for receipt in receipts)
    _require_unique_non_empty(
        "PreferenceAboutOtherSnapshot.action_outcome_mutation_receipts.mutation_id",
        mutation_ids,
    )
    latest_by_target: dict[str, PreferenceActionOutcomeMutationReceipt] = {}
    for receipt in receipts:
        latest_by_target[receipt.target_evidence_id] = receipt
    record_ids = {record.record_id for record in records}
    action_outcomes_by_id = {item.evidence_id: item for item in action_outcomes}
    for target_evidence_id, receipt in latest_by_target.items():
        current = action_outcomes_by_id.get(target_evidence_id)
        if receipt.operation is PreferenceActionOutcomeMutationOperation.REDACT:
            if current is not None or target_evidence_id in record_ids:
                raise ValueError("redacted preference action outcome or record cannot remain in snapshot")
            continue
        if current is not None and (
            preference_action_outcome_evidence_sha256(current) != receipt.after_evidence_sha256
        ):
            raise ValueError("corrected preference action outcome hash does not match latest receipt")


def _validate_common_ground_atoms(
    field_name: str,
    atoms: tuple[CommonGroundAtom, ...],
    expected_kind: SocialScopeKind,
) -> None:
    atom_ids = tuple(atom.atom_id for atom in atoms)
    _require_unique_non_empty(f"{field_name}.atom_id", atom_ids)
    for atom in atoms:
        if atom.scope_kind is not expected_kind:
            raise ValueError(
                f"{field_name} must contain {expected_kind.value} atoms; "
                f"got {atom.scope_kind.value} for {atom.atom_id!r}"
            )


__all__ = [
    "MAX_COMMON_GROUND_RECURSION_DEPTH",
    "PRIMARY_INTERLOCUTOR_ID",
    "SELF_INTERLOCUTOR_ID",
    "BeliefAboutOtherSnapshot",
    "ConversationalRoleSnapshot",
    "CommonGroundAtom",
    "CommonGroundSnapshot",
    "FeelingAboutOtherSnapshot",
    "GroupIdentity",
    "GroupSnapshot",
    "InterlocutorIdentity",
    "IntentAboutOtherSnapshot",
    "MemorySocialPESignal",
    "MultiPartyIdentitySnapshot",
    "OtherMindRecord",
    "OtherMindRecordKind",
    "OtherMindRecordStatus",
    "PreferenceActionOutcomeEvidence",
    "PreferenceActionOutcomeMutation",
    "PreferenceActionOutcomeMutationOperation",
    "PreferenceActionOutcomeMutationReceipt",
    "PreferenceActionForecast",
    "PreferenceActionForecastSettlement",
    "PreferenceAboutOtherSnapshot",
    "RelationshipConditionReadout",
    "SocialActionCandidatePrediction",
    "SocialActionOutcomeProbability",
    "SocialPrediction",
    "SocialPredictionError",
    "SocialPredictionErrorSnapshot",
    "SocialPredictionKind",
    "SocialPredictionOutcome",
    "SocialPredictionSnapshot",
    "SocialScopeKind",
    "ToMInterlocutorRecordCount",
    "build_memory_visibility_signals",
    "build_primary_conversational_role_snapshot",
    "build_primary_multi_party_identity_snapshot",
    "social_prediction_error_from_memory_signal",
    "social_prediction_from_memory_signal",
    "preference_action_outcome_evidence_sha256",
    "preference_action_outcome_mutation_sha256",
    "preference_action_forecast_from_payload",
    "preference_action_forecast_to_payload",
    "tom_record_counts_by_interlocutor",
]
