"""Multi-party identity owner scaffold (R16).

Slice 2 keeps behaviour unchanged: the owner publishes a SHADOW-only
``primary`` compatibility snapshot. Later R16 slices will replace this
with real speaker / audience / subject inference and social PE wiring.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.environment import EnvironmentEvent
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    InterlocutorIdentity,
    MultiPartyIdentitySnapshot,
    SocialPredictionError,
    SocialPredictionErrorSnapshot,
    SocialPredictionSnapshot,
    build_primary_multi_party_identity_snapshot,
)


class MultiPartyIdentityModule(RuntimeModule[MultiPartyIdentitySnapshot]):
    """Publishes the R16 identity scope contract.

    The initial implementation intentionally has no upstream dependencies:
    it is a compatibility scaffold that exposes the new contract in SHADOW
    without changing existing flat ``user_model`` / ``relationship_state``
    consumers.
    """

    slot_name = "multi_party_identity"
    owner = "MultiPartyIdentityModule"
    value_type = MultiPartyIdentitySnapshot
    dependencies: tuple[str, ...] = ()
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
    ) -> Snapshot[MultiPartyIdentitySnapshot]:
        if self._environment_event is not None:
            frame = self._environment_event.frame
            identities = {
                frame.active_speaker_id,
                *frame.addressee_ids,
                *frame.subject_ids,
                *frame.audience_ids,
            }
            return self.publish(
                MultiPartyIdentitySnapshot(
                    active_speaker_id=frame.active_speaker_id,
                    addressee_ids=frame.addressee_ids,
                    subject_ids=frame.subject_ids,
                    audience_ids=frame.audience_ids,
                    interlocutors=tuple(
                        InterlocutorIdentity(
                            interlocutor_id=interlocutor_id,
                            evidence=(self._environment_event.event_id,),
                        )
                        for interlocutor_id in sorted(identities)
                    ),
                    identity_predictions=(),
                    description=(
                        "R16 SHADOW scaffold: identity scope consumed from "
                        f"EnvironmentEvent {self._environment_event.event_id}."
                    ),
                )
            )
        return self.publish(
            build_primary_multi_party_identity_snapshot(
                description=(
                    "R16 SHADOW scaffold: single-interlocutor compatibility "
                    "identity scope under 'primary'."
                )
            )
        )


class SocialPredictionAggregateModule(RuntimeModule[SocialPredictionSnapshot]):
    """Publishes the pre-action social prediction aggregate.

    Slice 6 is intentionally empty: it establishes ownership and wiring so
    later slices can add misattribution predictions without changing the
    public snapshot shape.
    """

    slot_name = "social_prediction"
    owner = "SocialPredictionAggregateModule"
    value_type = SocialPredictionSnapshot
    dependencies = ("multi_party_identity",)
    default_wiring_level = WiringLevel.SHADOW

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[SocialPredictionSnapshot]:
        identity_snapshot = upstream.get("multi_party_identity")
        identity_available = isinstance(
            identity_snapshot.value, MultiPartyIdentitySnapshot
        ) if identity_snapshot is not None else False
        suffix = "identity=available" if identity_available else "identity=compatibility-fallback"
        return self.publish(
            SocialPredictionSnapshot(
                predictions=(),
                description=f"R16 SHADOW scaffold: no social predictions emitted yet; {suffix}.",
            )
        )


class SocialPredictionErrorModule(RuntimeModule[SocialPredictionErrorSnapshot]):
    """Publishes typed social PE records derived from social predictions."""

    slot_name = "social_prediction_error"
    owner = "SocialPredictionErrorModule"
    value_type = SocialPredictionErrorSnapshot
    dependencies = ("social_prediction",)
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        pending_errors: tuple[SocialPredictionError, ...] = (),
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._pending_errors = pending_errors

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[SocialPredictionErrorSnapshot]:
        prediction_snapshot = upstream.get("social_prediction")
        prediction_available = isinstance(
            prediction_snapshot.value, SocialPredictionSnapshot
        ) if prediction_snapshot is not None else False
        suffix = (
            "prediction=available"
            if prediction_available
            else "prediction=compatibility-fallback"
        )
        return self.publish(
            SocialPredictionErrorSnapshot(
                errors=self._pending_errors,
                description=(
                    f"R16 SHADOW scaffold: social prediction errors="
                    f"{len(self._pending_errors)}; {suffix}."
                ),
            )
        )


__all__ = [
    "MultiPartyIdentityModule",
    "SocialPredictionAggregateModule",
    "SocialPredictionErrorModule",
]
