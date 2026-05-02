"""Conversational role owner scaffold (R18).

Slice 2 publishes a SHADOW-only ``primary/self`` compatibility role
snapshot. Later slices will consume EnvironmentEvent role fields and role
proposals; this slice only establishes ownership and wiring.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.environment import EnvironmentEvent
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    ConversationalRoleSnapshot,
    SocialPrediction,
    SocialPredictionKind,
    SocialScopeKind,
    build_primary_conversational_role_snapshot,
)


class ConversationalRoleModule(RuntimeModule[ConversationalRoleSnapshot]):
    slot_name = "conversational_role"
    owner = "ConversationalRoleModule"
    value_type = ConversationalRoleSnapshot
    dependencies = ("multi_party_identity",)
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        environment_event: EnvironmentEvent | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._environment_event = environment_event

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ConversationalRoleSnapshot]:
        del upstream
        if self._environment_event is not None:
            frame = self._environment_event.frame
            prediction = SocialPrediction(
                prediction_id=f"{self._environment_event.event_id}:role-assignment",
                kind=SocialPredictionKind.ROLE_ASSIGNMENT,
                scope_kind=SocialScopeKind.INTERLOCUTOR,
                scope_id=frame.active_speaker_id,
                subject_ids=frame.subject_ids,
                audience_ids=frame.audience_ids,
                predicted_outcome=(
                    f"active_speaker={frame.active_speaker_id}; "
                    f"addressees={','.join(frame.addressee_ids)}; "
                    f"subjects={','.join(frame.subject_ids)}"
                ),
                confidence=1.0,
                evidence=(self._environment_event.event_id,),
            )
            return self.publish(
                ConversationalRoleSnapshot(
                    active_speaker_id=frame.active_speaker_id,
                    addressee_ids=frame.addressee_ids,
                    subject_ids=frame.subject_ids,
                    witness_ids=(),
                    overhearer_ids=(),
                    group_audience_ids=(),
                    role_confidence=1.0,
                    active_predictions=(prediction,),
                    description=(
                        "R18 SHADOW scaffold: conversational role consumed "
                        f"from EnvironmentEvent {self._environment_event.event_id}."
                    ),
                )
            )
        return self.publish(
            build_primary_conversational_role_snapshot(
                description=(
                    "R18 SHADOW scaffold: single-interlocutor compatibility "
                    "conversational role under primary/self."
                )
            )
        )


__all__ = ["ConversationalRoleModule"]
