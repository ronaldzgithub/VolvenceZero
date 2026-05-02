"""Contract tests for Social Cognition Learning Layer scaffolding (R16)."""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    BeliefAboutOtherSnapshot,
    ConversationalRoleSnapshot,
    FeelingAboutOtherSnapshot,
    InterlocutorIdentity,
    IntentAboutOtherSnapshot,
    MultiPartyIdentitySnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceAboutOtherSnapshot,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionErrorSnapshot,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialPredictionSnapshot,
    SocialScopeKind,
    build_primary_conversational_role_snapshot,
    build_primary_multi_party_identity_snapshot,
)


def test_social_prediction_kind_values_are_exhaustive() -> None:
    assert set(SocialPredictionKind) == {
        SocialPredictionKind.IDENTITY_ATTRIBUTION,
        SocialPredictionKind.AUDIENCE_SCOPE,
        SocialPredictionKind.MEMORY_VISIBILITY,
        SocialPredictionKind.RELATIONSHIP_ATTRIBUTION,
        SocialPredictionKind.ROLE_ASSIGNMENT,
        SocialPredictionKind.COMMON_GROUND_RESOLUTION,
        SocialPredictionKind.GROUP_COMMITMENT_DURABILITY,
    }


def test_social_scope_kind_values_are_exhaustive() -> None:
    assert set(SocialScopeKind) == {
        SocialScopeKind.INTERLOCUTOR,
        SocialScopeKind.DYAD,
        SocialScopeKind.GROUP,
    }


def test_social_prediction_outcome_values_are_exhaustive() -> None:
    assert set(SocialPredictionOutcome) == {
        SocialPredictionOutcome.CONFIRMED,
        SocialPredictionOutcome.DISCONFIRMED,
        SocialPredictionOutcome.STALE,
        SocialPredictionOutcome.UNKNOWN,
    }


def test_other_mind_record_kind_values_are_exhaustive() -> None:
    assert set(OtherMindRecordKind) == {
        OtherMindRecordKind.BELIEF,
        OtherMindRecordKind.INTENT,
        OtherMindRecordKind.FEELING,
        OtherMindRecordKind.PREFERENCE,
    }


def test_other_mind_record_status_values_are_exhaustive() -> None:
    assert set(OtherMindRecordStatus) == {
        OtherMindRecordStatus.ACTIVE,
        OtherMindRecordStatus.CONTESTED,
        OtherMindRecordStatus.RETIRED,
    }


def test_primary_identity_snapshot_is_frozen_and_single_party_compatible() -> None:
    snapshot = build_primary_multi_party_identity_snapshot()

    assert dataclasses.is_dataclass(snapshot)
    assert snapshot.__dataclass_params__.frozen
    assert snapshot.active_speaker_id == PRIMARY_INTERLOCUTOR_ID
    assert snapshot.addressee_ids == (SELF_INTERLOCUTOR_ID,)
    assert snapshot.subject_ids == (PRIMARY_INTERLOCUTOR_ID,)
    assert snapshot.audience_ids == (SELF_INTERLOCUTOR_ID,)
    assert len(snapshot.interlocutors) == 1
    assert snapshot.interlocutors[0].interlocutor_id == PRIMARY_INTERLOCUTOR_ID


def test_primary_conversational_role_snapshot_is_frozen_and_single_party_compatible() -> None:
    snapshot = build_primary_conversational_role_snapshot()

    assert dataclasses.is_dataclass(snapshot)
    assert snapshot.__dataclass_params__.frozen
    assert snapshot.active_speaker_id == PRIMARY_INTERLOCUTOR_ID
    assert snapshot.addressee_ids == (SELF_INTERLOCUTOR_ID,)
    assert snapshot.subject_ids == (PRIMARY_INTERLOCUTOR_ID,)
    assert snapshot.witness_ids == ()
    assert snapshot.overhearer_ids == ()
    assert snapshot.group_audience_ids == ()
    assert snapshot.role_confidence == 1.0
    assert snapshot.active_predictions == ()


def test_conversational_role_rejects_invalid_required_scope() -> None:
    with pytest.raises(ValueError, match="addressee_ids"):
        ConversationalRoleSnapshot(
            active_speaker_id="alice",
            addressee_ids=(),
            subject_ids=("alice",),
            witness_ids=(),
            overhearer_ids=(),
            group_audience_ids=(),
            role_confidence=0.8,
            active_predictions=(),
            description="missing addressee",
        )
    with pytest.raises(ValueError, match="subject_ids"):
        ConversationalRoleSnapshot(
            active_speaker_id="alice",
            addressee_ids=("self",),
            subject_ids=("alice", "alice"),
            witness_ids=(),
            overhearer_ids=(),
            group_audience_ids=(),
            role_confidence=0.8,
            active_predictions=(),
            description="duplicate subject",
        )


def test_conversational_role_rejects_invalid_optional_scope_and_confidence() -> None:
    with pytest.raises(ValueError, match="witness_ids"):
        ConversationalRoleSnapshot(
            active_speaker_id="alice",
            addressee_ids=("self",),
            subject_ids=("alice",),
            witness_ids=("bob", "bob"),
            overhearer_ids=(),
            group_audience_ids=(),
            role_confidence=0.8,
            active_predictions=(),
            description="duplicate witness",
        )
    with pytest.raises(ValueError, match="role_confidence"):
        ConversationalRoleSnapshot(
            active_speaker_id="alice",
            addressee_ids=("self",),
            subject_ids=("alice",),
            witness_ids=(),
            overhearer_ids=(),
            group_audience_ids=(),
            role_confidence=1.2,
            active_predictions=(),
            description="bad confidence",
        )


def test_conversational_role_rejects_duplicate_prediction_ids() -> None:
    prediction = SocialPrediction(
        prediction_id="role-prediction",
        kind=SocialPredictionKind.ROLE_ASSIGNMENT,
        scope_kind=SocialScopeKind.INTERLOCUTOR,
        scope_id="alice",
        subject_ids=("alice",),
        audience_ids=("self",),
        predicted_outcome="Alice is the active speaker.",
        confidence=0.8,
    )

    with pytest.raises(ValueError, match="prediction_id"):
        ConversationalRoleSnapshot(
            active_speaker_id="alice",
            addressee_ids=("self",),
            subject_ids=("alice",),
            witness_ids=(),
            overhearer_ids=(),
            group_audience_ids=(),
            role_confidence=0.8,
            active_predictions=(prediction, prediction),
            description="duplicate predictions",
        )


def test_interlocutor_identity_rejects_empty_and_duplicate_values() -> None:
    with pytest.raises(ValueError, match="interlocutor_id"):
        InterlocutorIdentity(interlocutor_id=" ")

    with pytest.raises(ValueError, match="aliases"):
        InterlocutorIdentity(interlocutor_id="alice", aliases=("a", "a"))

    with pytest.raises(ValueError, match="confidence"):
        InterlocutorIdentity(interlocutor_id="alice", confidence=1.1)


def test_multi_party_identity_requires_active_speaker_identity() -> None:
    alice = InterlocutorIdentity(interlocutor_id="alice")

    with pytest.raises(ValueError, match="active_speaker_id"):
        MultiPartyIdentitySnapshot(
            active_speaker_id="bob",
            addressee_ids=("self",),
            subject_ids=("alice",),
            audience_ids=("self",),
            interlocutors=(alice,),
            identity_predictions=(),
            description="speaker mismatch",
        )


def test_multi_party_identity_rejects_duplicate_scope_ids() -> None:
    alice = InterlocutorIdentity(interlocutor_id="alice")

    with pytest.raises(ValueError, match="audience_ids"):
        MultiPartyIdentitySnapshot(
            active_speaker_id="alice",
            addressee_ids=("self",),
            subject_ids=("alice",),
            audience_ids=("self", "self"),
            interlocutors=(alice,),
            identity_predictions=(),
            description="duplicate audience",
        )


def test_social_prediction_requires_scope_and_confidence() -> None:
    with pytest.raises(ValueError, match="subject_ids"):
        SocialPrediction(
            prediction_id="p1",
            kind=SocialPredictionKind.IDENTITY_ATTRIBUTION,
            scope_kind=SocialScopeKind.INTERLOCUTOR,
            scope_id="alice",
            subject_ids=(),
            audience_ids=("self",),
            predicted_outcome="state belongs to Alice",
            confidence=0.8,
        )

    with pytest.raises(ValueError, match="confidence"):
        SocialPrediction(
            prediction_id="p1",
            kind=SocialPredictionKind.IDENTITY_ATTRIBUTION,
            scope_kind=SocialScopeKind.INTERLOCUTOR,
            scope_id="alice",
            subject_ids=("alice",),
            audience_ids=("self",),
            predicted_outcome="state belongs to Alice",
            confidence=-0.1,
        )


def test_social_prediction_error_is_typed_and_bounded() -> None:
    error = SocialPredictionError(
        error_id="e1",
        prediction_id="p1",
        kind=SocialPredictionKind.MEMORY_VISIBILITY,
        outcome=SocialPredictionOutcome.DISCONFIRMED,
        magnitude=0.75,
        owner="MultiPartyIdentityModule",
        scope_kind=SocialScopeKind.INTERLOCUTOR,
        scope_id="alice",
        evidence=("private memory was exposed to Bob",),
    )

    assert error.kind is SocialPredictionKind.MEMORY_VISIBILITY
    assert error.outcome is SocialPredictionOutcome.DISCONFIRMED

    with pytest.raises(ValueError, match="magnitude"):
        SocialPredictionError(
            error_id="e2",
            prediction_id="p1",
            kind=SocialPredictionKind.MEMORY_VISIBILITY,
            outcome=SocialPredictionOutcome.DISCONFIRMED,
            magnitude=1.5,
            owner="MultiPartyIdentityModule",
            scope_kind=SocialScopeKind.INTERLOCUTOR,
            scope_id="alice",
            evidence=("out of range",),
        )


def _other_mind_record(
    *,
    record_id: str = "tom:belief:1",
    kind: OtherMindRecordKind = OtherMindRecordKind.BELIEF,
) -> OtherMindRecord:
    return OtherMindRecord(
        record_id=record_id,
        interlocutor_id="alice",
        kind=kind,
        summary="Alice believes the meeting is tomorrow.",
        detail="Alice stated the meeting was on Tuesday after reading the invite.",
        confidence=0.72,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=3,
        prediction_error_refs=("social-pe:1",),
        evidence="Explicit utterance plus calendar discussion.",
    )


def test_other_mind_record_is_keyed_and_validated() -> None:
    record = _other_mind_record()

    assert dataclasses.is_dataclass(record)
    assert record.__dataclass_params__.frozen
    assert record.interlocutor_id == "alice"
    assert record.kind is OtherMindRecordKind.BELIEF

    with pytest.raises(ValueError, match="interlocutor_id"):
        _other_mind_record().__class__(
            record_id="tom:bad:1",
            interlocutor_id=" ",
            kind=OtherMindRecordKind.BELIEF,
            summary="summary",
            detail="detail",
            confidence=0.5,
            status=OtherMindRecordStatus.ACTIVE,
            source_turn=0,
            prediction_error_refs=(),
            evidence="evidence",
        )
    with pytest.raises(ValueError, match="confidence"):
        _other_mind_record(kind=OtherMindRecordKind.INTENT).__class__(
            record_id="tom:bad:2",
            interlocutor_id="alice",
            kind=OtherMindRecordKind.INTENT,
            summary="summary",
            detail="detail",
            confidence=1.1,
            status=OtherMindRecordStatus.ACTIVE,
            source_turn=0,
            prediction_error_refs=(),
            evidence="evidence",
        )


def test_other_mind_snapshots_accept_empty_shadow_scaffolds() -> None:
    assert BeliefAboutOtherSnapshot((), (), 0.0, "empty belief scaffold").records == ()
    assert IntentAboutOtherSnapshot((), (), 0.0, "empty intent scaffold").records == ()
    assert FeelingAboutOtherSnapshot((), (), 0.0, "empty feeling scaffold").records == ()
    assert PreferenceAboutOtherSnapshot((), (), 0.0, "empty preference scaffold").records == ()


def test_other_mind_snapshots_reject_wrong_record_kind() -> None:
    wrong = _other_mind_record(kind=OtherMindRecordKind.INTENT)

    with pytest.raises(ValueError, match="kind=belief"):
        BeliefAboutOtherSnapshot(
            records=(wrong,),
            active_predictions=(),
            control_signal=0.1,
            description="wrong kind",
        )


def test_other_mind_snapshots_reject_duplicate_record_ids() -> None:
    record = _other_mind_record(record_id="tom:belief:duplicate")

    with pytest.raises(ValueError, match="record_id"):
        BeliefAboutOtherSnapshot(
            records=(record, record),
            active_predictions=(),
            control_signal=0.1,
            description="duplicate records",
        )


def test_social_prediction_snapshots_accept_empty_shadow_scaffolds() -> None:
    predictions = SocialPredictionSnapshot(
        predictions=(),
        description="empty shadow social prediction scaffold",
    )
    errors = SocialPredictionErrorSnapshot(
        errors=(),
        description="empty shadow social prediction error scaffold",
    )

    assert predictions.predictions == ()
    assert errors.errors == ()


def test_social_prediction_snapshot_rejects_duplicate_prediction_ids() -> None:
    prediction = SocialPrediction(
        prediction_id="p1",
        kind=SocialPredictionKind.IDENTITY_ATTRIBUTION,
        scope_kind=SocialScopeKind.INTERLOCUTOR,
        scope_id="alice",
        subject_ids=("alice",),
        audience_ids=("self",),
        predicted_outcome="state belongs to Alice",
        confidence=0.8,
    )

    with pytest.raises(ValueError, match="prediction_id"):
        SocialPredictionSnapshot(
            predictions=(prediction, prediction),
            description="duplicate predictions",
        )


def test_social_prediction_error_snapshot_rejects_duplicate_error_ids() -> None:
    error = SocialPredictionError(
        error_id="e1",
        prediction_id="p1",
        kind=SocialPredictionKind.MEMORY_VISIBILITY,
        outcome=SocialPredictionOutcome.DISCONFIRMED,
        magnitude=0.5,
        owner="MultiPartyIdentityModule",
        scope_kind=SocialScopeKind.INTERLOCUTOR,
        scope_id="alice",
        evidence=("scope mismatch",),
    )

    with pytest.raises(ValueError, match="error_id"):
        SocialPredictionErrorSnapshot(
            errors=(error, error),
            description="duplicate errors",
        )
