"""Compile a cultivation seed into an Identity Core BehaviorProtocol.

First-principles placement: in Volvence Zero the "流派 / school" of an
agent lives in the Behavior Protocol Runtime, NOT in a regime label and
NOT in raw ingested corpus. The cultivated expert's stable identity is
expressed as an **Identity Core protocol**: a reviewed, ACTIVE
``BehaviorProtocol`` that

* asserts the persona's identity (``IdentityAssertion``),
* carries the operator-supplied value boundaries as a hard-block
  ``boundary_contracts`` union (the "不被用户浅薄认知带偏" floor), and
* declares a single school-coherence ``StrategyPrior`` that biases the
  agent to assess every newly-researched theory against the school it is
  forming, integrate the consistent parts and down-weight/reject the
  inconsistent ones.

This protocol is the constant layer above the active mixture: researched
theories enter the mixture as *other* protocols and compete on PE-driven
utility, while this anchor keeps a minimum activation floor so the school
identity never drops out. Conflict resolution itself is owned by the
kernel's protocol runtime (soft-blend / PE arbitration / boundary union /
slow reflection) — this module only builds the typed anchor; it never
decides matches by keyword.
"""

from __future__ import annotations

from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    FailureSignal,
    IdentityAssertion,
    ProtocolSourceKind,
    ReviewStatus,
    StrategyPrior,
    SuccessSignal,
    TemporalArc,
)

from lifeform_cultivation.curriculum import CultivationSeed

# The anchor keeps a minimum activation weight so the school identity is
# never fully out-competed by a transiently high-utility researched
# theory. It is deliberately modest: the mixture should still be able to
# foreground a specialised protocol for a given turn, but never drop the
# core to zero.
IDENTITY_CORE_WEIGHT_FLOOR = 0.2


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")
    return out or "expert"


def build_identity_core_protocol(seed: CultivationSeed) -> BehaviorProtocol:
    """Build the ACTIVE Identity Core protocol for a cultivation seed.

    The returned protocol is review_status=ACTIVE because it is
    operator-seeded (the rough persona the operator hands in IS the
    reviewed identity); researched-theory protocols, by contrast, enter
    via DocumentUptake at SHADOW and earn their place through PE.
    """

    slug = _slug(seed.display_name or seed.domain)
    protocol_id = f"cultivation-identity:{slug}"

    boundary_contracts = tuple(
        BoundaryContract(
            boundary_id=f"{protocol_id}:boundary:{i}",
            description=text,
            # Typed signal source (enum value), never a user-text keyword
            # (no-keyword-matching invariant). A boundary violation event
            # is the canonical hard-block trigger.
            trigger_reasons=(BehaviorProtocolSignalSource.BOUNDARY_VIOLATION_FIRED.value,),
            blocked_topics=(),
            refer_out_required=False,
            severity=BoundarySeverity.HARD_BLOCK,
        )
        for i, text in enumerate(seed.value_boundaries)
        if text.strip()
    )

    school_strategy = StrategyPrior(
        rule_id=f"{protocol_id}:school-coherence",
        problem_pattern="new theory evaluated against the forming school",
        recommended_ordering=(
            "assess-consistency-with-formed-school",
            "integrate-consistent-claims",
            "down-weight-or-reject-inconsistent-claims",
        ),
        recommended_pacing="deliberate",
        avoid_patterns=("mixing-incompatible-schools-without-reconciliation",),
    )

    success = (
        SuccessSignal(
            signal_id=f"{protocol_id}:coherent-no-rupture",
            description=(
                "No rupture while reasoning from the formed school — the "
                "expert stayed internally consistent under this protocol."
            ),
            measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
        ),
    )
    failure = (
        FailureSignal(
            signal_id=f"{protocol_id}:incoherence-rupture",
            description=(
                "Rupture while reasoning from the formed school — a "
                "contradiction was surfaced that the school did not resolve."
            ),
            measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
        ),
    )

    return BehaviorProtocol(
        protocol_id=protocol_id,
        version="1.0.0",
        advisor_name=seed.display_name or seed.role_archetype or "cultivated-expert",
        description=(
            f"Identity Core for cultivated {seed.role_archetype or 'expert'} "
            f"in {seed.domain}. Objective: {seed.single_school_objective}"
        ),
        source_kind=ProtocolSourceKind.API_INJECTION,
        source_locator=f"cultivation:identity:{slug}",
        identity_assertion=IdentityAssertion(
            requires_self_traits=(),
            forbidden_self_traits=(),
            required_regime_compatibility=(),
        ),
        boundary_contracts=boundary_contracts,
        activation_conditions=ActivationConditions(
            minimum_weight_floor=IDENTITY_CORE_WEIGHT_FLOOR,
        ),
        strategy_priors=(school_strategy,),
        temporal_arc=TemporalArc(),
        success_signals=success,
        failure_signals=failure,
        review_status=ReviewStatus.ACTIVE,
    )


def is_identity_core(protocol_id: str) -> bool:
    return protocol_id.startswith("cultivation-identity:")


__all__ = [
    "IDENTITY_CORE_WEIGHT_FLOOR",
    "build_identity_core_protocol",
    "is_identity_core",
]
