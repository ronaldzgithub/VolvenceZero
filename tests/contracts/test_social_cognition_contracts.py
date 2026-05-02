"""Contract tests for Social Cognition Learning Layer scaffolding (R16)."""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    InterlocutorIdentity,
    MultiPartyIdentitySnapshot,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionErrorSnapshot,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialPredictionSnapshot,
    SocialScopeKind,
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
