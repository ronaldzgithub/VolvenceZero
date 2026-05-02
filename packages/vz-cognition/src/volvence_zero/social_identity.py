"""Multi-party identity owner scaffold (R16).

Slice 2 keeps behaviour unchanged: the owner publishes a SHADOW-only
``primary`` compatibility snapshot. Later R16 slices will replace this
with real speaker / audience / subject inference and social PE wiring.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    MultiPartyIdentitySnapshot,
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

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[MultiPartyIdentitySnapshot]:
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
                errors=(),
                description=f"R16 SHADOW scaffold: no social prediction errors emitted yet; {suffix}.",
            )
        )


__all__ = [
    "MultiPartyIdentityModule",
    "SocialPredictionAggregateModule",
    "SocialPredictionErrorModule",
]
