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


__all__ = ["MultiPartyIdentityModule"]
