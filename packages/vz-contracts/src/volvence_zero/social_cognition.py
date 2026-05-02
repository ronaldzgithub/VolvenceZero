"""Social cognition contracts (R16-R20).

This module is deliberately data-only. It lives in ``vz-contracts`` so
kernel owners, lifeform-side readouts, and evidence tooling can share the
same immutable shapes without reversing package dependencies.

The first landed slice covers R16 scaffolding: multi-party identity scope,
pre-action social predictions, and typed social prediction error records.
Runtime owners and propagation wiring are added in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


PRIMARY_INTERLOCUTOR_ID = "primary"
SELF_INTERLOCUTOR_ID = "self"


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


class SocialPredictionOutcome(str, Enum):
    """Outcome class for a previously emitted social prediction."""

    CONFIRMED = "confirmed"
    DISCONFIRMED = "disconfirmed"
    STALE = "stale"
    UNKNOWN = "unknown"


class OtherMindRecordKind(str, Enum):
    """Four distinct Theory-of-Mind state kinds (R17)."""

    BELIEF = "belief"
    INTENT = "intent"
    FEELING = "feeling"
    PREFERENCE = "preference"


class OtherMindRecordStatus(str, Enum):
    """Lifecycle state for an inferred other-mind record."""

    ACTIVE = "active"
    CONTESTED = "contested"
    RETIRED = "retired"


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
class OtherMindRecord:
    record_id: str
    interlocutor_id: str
    kind: OtherMindRecordKind
    summary: str
    detail: str
    confidence: float
    status: OtherMindRecordStatus
    source_turn: int
    prediction_error_refs: tuple[str, ...]
    evidence: str

    def __post_init__(self) -> None:
        _require_non_empty("record_id", self.record_id)
        _require_non_empty("interlocutor_id", self.interlocutor_id)
        _require_non_empty("summary", self.summary)
        _require_non_empty("detail", self.detail)
        _require_confidence("confidence", self.confidence)
        if self.source_turn < 0:
            raise ValueError(f"source_turn must be >= 0, got {self.source_turn!r}")
        _require_unique_non_empty("prediction_error_refs", self.prediction_error_refs)
        _require_non_empty("evidence", self.evidence)


@dataclass(frozen=True)
class BeliefAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str

    def __post_init__(self) -> None:
        _validate_other_mind_snapshot(
            snapshot_name="BeliefAboutOtherSnapshot",
            expected_kind=OtherMindRecordKind.BELIEF,
            records=self.records,
            active_predictions=self.active_predictions,
            control_signal=self.control_signal,
            description=self.description,
        )


@dataclass(frozen=True)
class IntentAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str

    def __post_init__(self) -> None:
        _validate_other_mind_snapshot(
            snapshot_name="IntentAboutOtherSnapshot",
            expected_kind=OtherMindRecordKind.INTENT,
            records=self.records,
            active_predictions=self.active_predictions,
            control_signal=self.control_signal,
            description=self.description,
        )


@dataclass(frozen=True)
class FeelingAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str

    def __post_init__(self) -> None:
        _validate_other_mind_snapshot(
            snapshot_name="FeelingAboutOtherSnapshot",
            expected_kind=OtherMindRecordKind.FEELING,
            records=self.records,
            active_predictions=self.active_predictions,
            control_signal=self.control_signal,
            description=self.description,
        )


@dataclass(frozen=True)
class PreferenceAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str

    def __post_init__(self) -> None:
        _validate_other_mind_snapshot(
            snapshot_name="PreferenceAboutOtherSnapshot",
            expected_kind=OtherMindRecordKind.PREFERENCE,
            records=self.records,
            active_predictions=self.active_predictions,
            control_signal=self.control_signal,
            description=self.description,
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
        prediction_ids = tuple(
            prediction.prediction_id for prediction in self.active_predictions
        )
        _require_unique_non_empty("active_predictions.prediction_id", prediction_ids)
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
            raise ValueError(
                "MultiPartyIdentitySnapshot.active_speaker_id must be present "
                "in interlocutors"
            )


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
) -> None:
    record_ids = tuple(record.record_id for record in records)
    _require_unique_non_empty(f"{snapshot_name}.records.record_id", record_ids)
    for record in records:
        if record.kind is not expected_kind:
            raise ValueError(
                f"{snapshot_name} records must have kind={expected_kind.value}; "
                f"got {record.kind.value} for {record.record_id!r}"
            )
    prediction_ids = tuple(
        prediction.prediction_id for prediction in active_predictions
    )
    _require_unique_non_empty(
        f"{snapshot_name}.active_predictions.prediction_id", prediction_ids
    )
    _require_unit_interval("control_signal", control_signal)
    _require_non_empty("description", description)


__all__ = [
    "PRIMARY_INTERLOCUTOR_ID",
    "SELF_INTERLOCUTOR_ID",
    "BeliefAboutOtherSnapshot",
    "ConversationalRoleSnapshot",
    "FeelingAboutOtherSnapshot",
    "InterlocutorIdentity",
    "IntentAboutOtherSnapshot",
    "MultiPartyIdentitySnapshot",
    "OtherMindRecord",
    "OtherMindRecordKind",
    "OtherMindRecordStatus",
    "PreferenceAboutOtherSnapshot",
    "SocialPrediction",
    "SocialPredictionError",
    "SocialPredictionErrorSnapshot",
    "SocialPredictionKind",
    "SocialPredictionOutcome",
    "SocialPredictionSnapshot",
    "SocialScopeKind",
    "build_primary_conversational_role_snapshot",
    "build_primary_multi_party_identity_snapshot",
]
