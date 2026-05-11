"""Build the IdentitySeed for the growth-advisor archetype (packet 1.3'').

Mirrors the existing fixture-side compile pattern:

* ``compiler.build_growth_advisor_package`` →
  ``DomainExperiencePackage`` (knowledge / case / playbook /
  boundary records).
* ``compiler.build_growth_advisor_vitals_bootstrap`` →
  ``VitalsBootstrap`` (drive set).
* ``fixture_uptake.growth_advisor_profile_to_behavior_protocol`` →
  ``BehaviorProtocol`` (protocol-runtime declaration).
* **packet 1.3'**, this module: → ``IdentitySeed`` (Self-track
  trait surface).

The seed populates ``DualTrackSnapshot.self_track.traits`` so the
protocol identity gate (R7 self-trait branch) can do real
subset / forbidden checks instead of falling back to
``self_traits_populator_pending``.

Trait derivation policy for cheng_laoshi:

The reviewed profile already declares the lifeform's intended
identity stance through ``IdentityAssertion.requires_self_traits``
on every ``BehaviorProtocol`` it produces — those are the traits
the protocol expects the lifeform to carry. The fixture
intentionally returns the **same** trait set on the seed, so:

* The seed asserts that the lifeform actually IS the growth-
  advisor archetype (warm peer-mom register; long-horizon
  relationship horizon).
* The protocol gate matches its own ``requires_self_traits`` and
  passes.
* Other (hypothetical, hostile) protocols requiring different
  traits — e.g. ``aggressive_sales`` — would now actually be
  filtered out by the gate; before packet 1.3'' the gate fell
  back to permissive.

This is **not** circular: the seed is set ONCE at lifeform
construction and frozen for the session. Protocols cannot mutate
the seed; only assert against it. Future packets can move trait
synthesis from the fixture into a R6 reflection / persona-source
layer without changing this signature.
"""

from __future__ import annotations

from volvence_zero.identity_seed import IdentitySeed

from lifeform_domain_growth_advisor.profile import GrowthAdvisorProfile


_GROWTH_ADVISOR_TRAITS: tuple[str, ...] = (
    "warm_peer_register",
    "long_horizon",
)


def build_growth_advisor_identity_seed(
    profile: GrowthAdvisorProfile,
) -> IdentitySeed:
    """Compile growth-advisor profile metadata into an ``IdentitySeed``.

    Returns a frozen ``IdentitySeed`` whose ``traits`` match the
    growth-advisor archetype's canonical persona descriptors. The
    profile argument is currently consumed only for the
    ``description`` field (so audit logs pin the seed to its source
    profile); the trait list itself is canonical for the archetype
    and identical across profiles in this vertical.
    """

    return IdentitySeed(
        traits=_GROWTH_ADVISOR_TRAITS,
        description=(
            f"Growth-advisor identity seed compiled from "
            f"{profile.profile_id!r}: warm peer-mom register + "
            f"long-horizon companion stance."
        ),
    )


__all__ = [
    "build_growth_advisor_identity_seed",
]
