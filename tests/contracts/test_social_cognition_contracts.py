"""Contract tests for Social Cognition Learning Layer scaffolding (R16)."""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    BeliefAboutOtherSnapshot,
    CommonGroundAtom,
    CommonGroundSnapshot,
    ConversationalRoleSnapshot,
    FeelingAboutOtherSnapshot,
    GroupIdentity,
    GroupSnapshot,
    InterlocutorIdentity,
    IntentAboutOtherSnapshot,
    MultiPartyIdentitySnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceAboutOtherSnapshot,
    MAX_COMMON_GROUND_RECURSION_DEPTH,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionErrorSnapshot,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialPredictionSnapshot,
    SocialScopeKind,
    ToMInterlocutorRecordCount,
    build_primary_conversational_role_snapshot,
    build_primary_multi_party_identity_snapshot,
    tom_record_counts_by_interlocutor,
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


def _common_ground_atom(
    *,
    atom_id: str = "cg:dyad:alice-bob:1",
    scope_kind: SocialScopeKind = SocialScopeKind.DYAD,
    recursion_depth: int = MAX_COMMON_GROUND_RECURSION_DEPTH,
) -> CommonGroundAtom:
    return CommonGroundAtom(
        atom_id=atom_id,
        scope_id="alice:bob" if scope_kind is SocialScopeKind.DYAD else "team:1",
        scope_kind=scope_kind,
        summary="Alice and Bob both know the plan changed.",
        recursion_depth=recursion_depth,
        confidence=0.7,
        accepted_by_ids=("alice", "bob"),
        evidence=("explicit confirmation",),
    )


def test_common_ground_atom_accepts_dyad_and_group_scopes() -> None:
    dyad = _common_ground_atom(scope_kind=SocialScopeKind.DYAD)
    group = _common_ground_atom(atom_id="cg:group:1", scope_kind=SocialScopeKind.GROUP)

    assert dyad.scope_kind is SocialScopeKind.DYAD
    assert group.scope_kind is SocialScopeKind.GROUP
    assert dyad.recursion_depth == MAX_COMMON_GROUND_RECURSION_DEPTH


def test_common_ground_atom_rejects_invalid_scope_or_depth() -> None:
    with pytest.raises(ValueError, match="scope_kind"):
        _common_ground_atom(scope_kind=SocialScopeKind.INTERLOCUTOR)
    with pytest.raises(ValueError, match="recursion_depth"):
        _common_ground_atom(recursion_depth=MAX_COMMON_GROUND_RECURSION_DEPTH + 1)
    with pytest.raises(ValueError, match="accepted_by_ids"):
        CommonGroundAtom(
            atom_id="cg:bad:accepted-by",
            scope_id="alice:bob",
            scope_kind=SocialScopeKind.DYAD,
            summary="shared assumption",
            recursion_depth=1,
            confidence=0.7,
            accepted_by_ids=("alice", "alice"),
            evidence=("duplicate accepted-by",),
        )


def test_common_ground_snapshot_validates_scope_buckets_and_prediction_ids() -> None:
    dyad = _common_ground_atom(scope_kind=SocialScopeKind.DYAD)
    group = _common_ground_atom(atom_id="cg:group:1", scope_kind=SocialScopeKind.GROUP)
    prediction = SocialPrediction(
        prediction_id="cg:prediction:1",
        kind=SocialPredictionKind.COMMON_GROUND_RESOLUTION,
        scope_kind=SocialScopeKind.DYAD,
        scope_id="alice:bob",
        subject_ids=("alice", "bob"),
        audience_ids=("alice", "bob"),
        predicted_outcome="Both can resolve the reference.",
        confidence=0.8,
    )

    snapshot = CommonGroundSnapshot(
        dyad_atoms=(dyad,),
        group_atoms=(group,),
        active_predictions=(prediction,),
        control_signal=0.4,
        description="common ground snapshot",
    )
    assert len(snapshot.dyad_atoms) == 1
    assert len(snapshot.group_atoms) == 1

    with pytest.raises(ValueError, match="dyad_atoms"):
        CommonGroundSnapshot(
            dyad_atoms=(group,),
            group_atoms=(),
            active_predictions=(),
            control_signal=0.0,
            description="wrong bucket",
        )
    with pytest.raises(ValueError, match="prediction_id"):
        CommonGroundSnapshot(
            dyad_atoms=(),
            group_atoms=(),
            active_predictions=(prediction, prediction),
            control_signal=0.0,
            description="duplicate prediction",
        )


def test_group_identity_validates_membership_and_confidence() -> None:
    group = GroupIdentity(
        group_id="group:launch",
        member_ids=("alice", "bob"),
        display_name="Launch group",
        confidence=0.8,
        evidence=("host membership list",),
    )

    assert group.group_id == "group:launch"
    assert group.member_ids == ("alice", "bob")

    with pytest.raises(ValueError, match="member_ids"):
        GroupIdentity(group_id="group:bad", member_ids=("alice", "alice"))
    with pytest.raises(ValueError, match="confidence"):
        GroupIdentity(group_id="group:bad", member_ids=("alice",), confidence=1.1)


def test_group_snapshot_validates_active_group_and_predictions() -> None:
    group = GroupIdentity(group_id="group:launch", member_ids=("alice", "bob"))
    prediction = SocialPrediction(
        prediction_id="group:prediction:1",
        kind=SocialPredictionKind.GROUP_COMMITMENT_DURABILITY,
        scope_kind=SocialScopeKind.GROUP,
        scope_id="group:launch",
        subject_ids=("alice", "bob"),
        audience_ids=("alice", "bob"),
        predicted_outcome="Joint commitment remains active.",
        confidence=0.8,
    )
    snapshot = GroupSnapshot(
        groups=(group,),
        active_group_id="group:launch",
        joint_attention=("launch-plan",),
        joint_commitments=("commitment:ship",),
        group_regime_id="problem_solving",
        active_predictions=(prediction,),
        description="group snapshot",
    )

    assert snapshot.active_group_id == "group:launch"
    assert snapshot.joint_commitments == ("commitment:ship",)

    with pytest.raises(ValueError, match="active_group_id"):
        GroupSnapshot(
            groups=(group,),
            active_group_id="group:missing",
            joint_attention=(),
            joint_commitments=(),
            group_regime_id=None,
            active_predictions=(),
            description="bad active group",
        )
    with pytest.raises(ValueError, match="prediction_id"):
        GroupSnapshot(
            groups=(group,),
            active_group_id="group:launch",
            joint_attention=(),
            joint_commitments=(),
            group_regime_id=None,
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


def test_tom_record_counts_by_interlocutor_uses_public_owner_snapshots() -> None:
    alice_belief = _other_mind_record(record_id="belief:alice")
    bob_belief = dataclasses.replace(
        _other_mind_record(record_id="belief:bob"),
        interlocutor_id="bob",
    )
    bob_intent = dataclasses.replace(
        _other_mind_record(
            record_id="intent:bob",
            kind=OtherMindRecordKind.INTENT,
        ),
        interlocutor_id="bob",
    )
    bob_feeling = dataclasses.replace(
        _other_mind_record(
            record_id="feeling:bob",
            kind=OtherMindRecordKind.FEELING,
        ),
        interlocutor_id="bob",
    )

    counts = tom_record_counts_by_interlocutor(
        belief=BeliefAboutOtherSnapshot(
            records=(alice_belief, bob_belief),
            active_predictions=(),
            control_signal=0.1,
            description="beliefs",
        ),
        intent=IntentAboutOtherSnapshot(
            records=(bob_intent,),
            active_predictions=(),
            control_signal=0.1,
            description="intents",
        ),
        feeling=FeelingAboutOtherSnapshot(
            records=(bob_feeling,),
            active_predictions=(),
            control_signal=0.1,
            description="feelings",
        ),
    )

    assert counts == (
        ToMInterlocutorRecordCount(
            interlocutor_id="alice",
            belief_count=1,
        ),
        ToMInterlocutorRecordCount(
            interlocutor_id="bob",
            belief_count=1,
            intent_count=1,
            feeling_count=1,
        ),
    )
    assert counts[1].total_count == 3


def test_tom_interlocutor_record_count_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="belief_count"):
        ToMInterlocutorRecordCount(interlocutor_id="alice", belief_count=-1)


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
