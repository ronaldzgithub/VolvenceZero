"""Lifeform-level identity seed for the dual-track Self surface.

Cross-wheel immutable contract for ``DualTrackModule`` (in
``vz-cognition.dual_track``). The seed is the lifeform-side
counterpart to ``BehaviorProtocol.IdentityAssertion`` (in
``vz-contracts.behavior_protocol``):

    * ``IdentityAssertion`` declares **what a protocol requires**
      from the lifeform's identity.
    * ``IdentitySeed`` declares **what the lifeform actually is**
      at construction time.

The activation gate in
``vz-application.protocol_runtime.activation`` matches one against
the other (subset for required traits; absent for forbidden
traits) to decide whether a protocol can activate for this
lifeform.

Lifecycle / scope:

* The seed is a **frozen lifeform-construction** input — set once
  per lifeform (or per session, for tests). Production is
  expected to derive it from a vertical bundle (e.g.
  ``build_growth_advisor_identity_seed(profile)``) and attach it
  via ``LifeformConfig`` (lifeform-core wiring lands in packet
  1.3'''; packet 1.3'' threads the seed through
  ``run_final_wiring_turn`` directly).
* Once attached, the seed flows into
  ``DualTrackModule.__init__`` and is read each turn by
  ``derive_track_state`` to populate
  ``TrackState.traits`` for the SELF track.
* The seed is intentionally **not mutable from the protocol
  layer**. Protocols can require / forbid traits, but they can
  not declare new ones into the lifeform — that path would let
  a malicious / drifting protocol reshape identity, which is
  exactly the R7 hazard the spec warns against (`docs/specs/dual-track-learning.md`
  + `docs/next_gen_emogpt.md` R7 — "关系连续性不是问题解决的副作用").
* Future ``IdentityRevision`` flow (R6 reflection-driven trait
  evolution) is out of scope for packet 1.3''; the seed is
  frozen on construction.

Why ``vz-contracts`` rather than ``vz-cognition.dual_track``:

* Both kernel (``DualTrackModule``) and lifeform-side / vertical
  fixture (``build_growth_advisor_identity_seed``) construct the
  type.
* ``vz-contracts`` is the canonical home for cross-wheel data
  shapes (mirrors ``BehaviorProtocol`` itself).
* Avoids forcing ``lifeform-domain-*`` wheels to declare a new
  ``vz-cognition`` dependency just to import this dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentitySeed:
    """Frozen identity descriptors for the lifeform's SELF track.

    Consumed by:

    * ``volvence_zero.dual_track.DualTrackModule.__init__`` (packet
      1.3'') — populates ``TrackState.traits`` for the SELF track
      on every snapshot.
    * Indirectly by
      ``volvence_zero.protocol_runtime.activation
      ._compute_identity_gate`` — checks
      ``BehaviorProtocol.IdentityAssertion.requires_self_traits``
      against ``self_track.traits``.

    Fields:

    * ``traits``: stable identity descriptors of the lifeform's
      Self track (e.g. ``("warm_peer_register", "long_horizon")``).
      Match key with
      ``BehaviorProtocol.IdentityAssertion.requires_self_traits``
      and ``forbidden_self_traits``.
    * ``description``: short human-readable summary of the
      identity for debug / audit. Optional.

    Validation: ``traits`` must be unique within the seed.
    """

    traits: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if len(set(self.traits)) != len(self.traits):
            raise ValueError(
                f"IdentitySeed.traits must be unique, got {self.traits!r}"
            )
        for trait in self.traits:
            if not trait.strip():
                raise ValueError(
                    "IdentitySeed.traits entries must be non-empty strings"
                )


__all__ = ["IdentitySeed"]
