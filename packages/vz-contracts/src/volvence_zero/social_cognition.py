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
            raise ValueError(
                "MemorySocialPESignal.evidence must be empty when outcome is None"
            )
        if self.outcome is not None and not self.evidence:
            raise ValueError(
                "MemorySocialPESignal.evidence must be non-empty when outcome is set"
            )


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
        magnitude = (
            suppressed_count / evaluated_total if evaluated_total > 0 else 1.0
        )
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
            raise ValueError(
                "CommonGroundAtom.scope_kind must be dyad or group; "
                f"got {self.scope_kind.value}"
            )
        _require_non_empty("summary", self.summary)
        if (
            self.recursion_depth < 0
            or self.recursion_depth > MAX_COMMON_GROUND_RECURSION_DEPTH
        ):
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

    def __post_init__(self) -> None:
        _validate_common_ground_atoms("dyad_atoms", self.dyad_atoms, SocialScopeKind.DYAD)
        _validate_common_ground_atoms("group_atoms", self.group_atoms, SocialScopeKind.GROUP)
        prediction_ids = tuple(
            prediction.prediction_id for prediction in self.active_predictions
        )
        _require_unique_non_empty("active_predictions.prediction_id", prediction_ids)
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

    def __post_init__(self) -> None:
        group_ids = tuple(group.group_id for group in self.groups)
        _require_unique_non_empty("groups.group_id", group_ids)
        if self.active_group_id is not None:
            _require_non_empty("active_group_id", self.active_group_id)
            if self.active_group_id not in group_ids:
                raise ValueError(
                    "active_group_id must refer to one of groups.group_id; "
                    f"got {self.active_group_id!r}"
                )
        _require_unique_non_empty("joint_attention", self.joint_attention)
        _require_unique_non_empty("joint_commitments", self.joint_commitments)
        if self.group_regime_id is not None:
            _require_non_empty("group_regime_id", self.group_regime_id)
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
    "PreferenceAboutOtherSnapshot",
    "SocialPrediction",
    "SocialPredictionError",
    "SocialPredictionErrorSnapshot",
    "SocialPredictionKind",
    "SocialPredictionOutcome",
    "SocialPredictionSnapshot",
    "SocialScopeKind",
    "build_memory_visibility_signals",
    "build_primary_conversational_role_snapshot",
    "build_primary_multi_party_identity_snapshot",
    "social_prediction_error_from_memory_signal",
    "social_prediction_from_memory_signal",
]
