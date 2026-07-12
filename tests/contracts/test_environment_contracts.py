"""Contract tests for Environment Interface runtime shapes."""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.environment import (
    EnvironmentActorRef,
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentFrame,
    EnvironmentOutcome,
    build_environment_event,
    build_primary_environment_frame,
    build_user_input_environment_event,
)
from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
)


def test_environment_event_kind_values_are_exhaustive() -> None:
    assert set(EnvironmentEventKind) == {
        EnvironmentEventKind.USER_INPUT,
        EnvironmentEventKind.SYSTEM_TICK,
        EnvironmentEventKind.SCENE_EVENT,
        EnvironmentEventKind.TOOL_RESULT,
        EnvironmentEventKind.INGESTION,
        EnvironmentEventKind.APPRENTICE,
        EnvironmentEventKind.INTERNAL_DRIVE,
        EnvironmentEventKind.FOLLOWUP_DUE,
    }


def test_primary_environment_frame_is_frozen_and_single_party_compatible() -> None:
    frame = build_primary_environment_frame()

    assert dataclasses.is_dataclass(frame)
    assert frame.__dataclass_params__.frozen
    assert frame.actor_id == PRIMARY_INTERLOCUTOR_ID
    assert frame.active_speaker_id == PRIMARY_INTERLOCUTOR_ID
    assert frame.addressee_ids == (SELF_INTERLOCUTOR_ID,)
    assert frame.subject_ids == (PRIMARY_INTERLOCUTOR_ID,)
    assert frame.audience_ids == (SELF_INTERLOCUTOR_ID,)


def test_environment_frame_rejects_empty_and_duplicate_scope_ids() -> None:
    actor = EnvironmentActorRef(actor_id="alice")

    with pytest.raises(ValueError, match="actor_id"):
        EnvironmentActorRef(actor_id=" ")

    with pytest.raises(ValueError, match="subject_ids"):
        EnvironmentFrame(
            actor=actor,
            active_speaker_id="alice",
            addressee_ids=("self",),
            subject_ids=(),
            audience_ids=("self",),
        )

    with pytest.raises(ValueError, match="audience_ids"):
        EnvironmentFrame(
            actor=actor,
            active_speaker_id="alice",
            addressee_ids=("self",),
            subject_ids=("alice",),
            audience_ids=("self", "self"),
        )


def test_user_input_environment_event_exposes_frame_fields() -> None:
    event = build_user_input_environment_event(
        event_id="evt-1",
        user_input="hello",
        scene_id="scene-1",
        timestamp_ms=12,
    )

    assert dataclasses.is_dataclass(event)
    assert event.__dataclass_params__.frozen
    assert event.event_kind is EnvironmentEventKind.USER_INPUT
    assert event.trigger_kind == EnvironmentEventKind.USER_INPUT.value
    assert event.actor_id == PRIMARY_INTERLOCUTOR_ID
    assert event.subject_ids == (PRIMARY_INTERLOCUTOR_ID,)
    assert event.payload_summary == "hello"


@pytest.mark.parametrize(
    "event_kind",
    (
        EnvironmentEventKind.SYSTEM_TICK,
        EnvironmentEventKind.SCENE_EVENT,
        EnvironmentEventKind.INTERNAL_DRIVE,
        EnvironmentEventKind.FOLLOWUP_DUE,
    ),
)
def test_generic_builder_preserves_canonical_lifecycle_event_kind(
    event_kind: EnvironmentEventKind,
) -> None:
    event = build_environment_event(
        event_id=f"evt-{event_kind.value}",
        event_kind=event_kind,
        trigger_kind=event_kind.value,
        payload_summary="typed lifecycle payload",
        scene_id="scene-1",
        timestamp_ms=12,
        provenance=f"test:{event_kind.value}",
        consent_context=("proactive-contact-allowed",),
    )

    assert event.event_kind is event_kind
    assert event.trigger_kind == event_kind.value
    assert event.provenance == f"test:{event_kind.value}"
    assert event.consent_context == ("proactive-contact-allowed",)


def test_environment_event_rejects_invalid_metadata() -> None:
    frame = build_primary_environment_frame()

    with pytest.raises(ValueError, match="event_id"):
        EnvironmentEvent(
            event_id=" ",
            event_kind=EnvironmentEventKind.USER_INPUT,
            trigger_kind="user_input",
            frame=frame,
            scene_id="scene-1",
            timestamp_ms=0,
            provenance="test",
        )

    with pytest.raises(ValueError, match="timestamp_ms"):
        EnvironmentEvent(
            event_id="evt-1",
            event_kind=EnvironmentEventKind.USER_INPUT,
            trigger_kind="user_input",
            frame=frame,
            scene_id="scene-1",
            timestamp_ms=-1,
            provenance="test",
        )


def test_environment_outcome_is_traceable_and_bounded() -> None:
    outcome = EnvironmentOutcome(
        outcome_id="out-1",
        event_id="evt-1",
        outcome_kind=EnvironmentEventKind.TOOL_RESULT,
        action_id="tool-call-1",
        status="success",
        summary="tool completed",
        detail="detail",
        confidence=0.9,
        prediction_id="prediction-1",
        evidence=("tool result event",),
    )

    assert outcome.event_id == "evt-1"
    assert outcome.prediction_id == "prediction-1"

    with pytest.raises(ValueError, match="confidence"):
        EnvironmentOutcome(
            outcome_id="out-2",
            event_id="evt-1",
            outcome_kind=EnvironmentEventKind.TOOL_RESULT,
            action_id="tool-call-1",
            status="success",
            summary="tool completed",
            detail="detail",
            confidence=1.5,
        )


def test_environment_outcome_observable_fields_are_minimal_and_validated() -> None:
    outcome = EnvironmentOutcome(
        outcome_id="out-3",
        event_id="evt-1",
        outcome_kind=EnvironmentEventKind.TOOL_RESULT,
        action_id="tool-call-1",
        status="success",
        summary="tool completed",
        detail="detail",
        latency_ms=42,
        monetary_cost=0.2,
        reversibility="costly",
        environment_state_delta_kind="filesystem_read",
    )

    assert outcome.latency_ms == 42
    assert outcome.monetary_cost == 0.2
    assert outcome.reversibility == "costly"
    assert outcome.environment_state_delta_kind == "filesystem_read"

    with pytest.raises(ValueError, match="latency_ms"):
        EnvironmentOutcome(
            outcome_id="out-4",
            event_id="evt-1",
            outcome_kind=EnvironmentEventKind.TOOL_RESULT,
            action_id="tool-call-1",
            status="success",
            summary="tool completed",
            detail="detail",
            latency_ms=-1,
        )

    with pytest.raises(ValueError, match="reversibility"):
        EnvironmentOutcome(
            outcome_id="out-5",
            event_id="evt-1",
            outcome_kind=EnvironmentEventKind.TOOL_RESULT,
            action_id="tool-call-1",
            status="success",
            summary="tool completed",
            detail="detail",
            reversibility="unsafe",
        )
