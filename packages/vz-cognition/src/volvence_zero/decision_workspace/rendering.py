"""What the system is entitled to say about a decision.

This module answers one question — *given the current valuation, which
claims are licensed?* — and it answers it as a typed verdict rather than
as prompt wording. The distinction matters: an instruction in a prompt
("don't overclaim") is a request; a claim licence the assembler must
consult is a constraint.

Two failures this exists to make impossible:

**Stating a winner when the intervals overlap.** "现在收益最高的是先分开
三个月" reads as a computed result. If the option intervals overlap, no
such result exists, and the honest sentence is about robustness and
reversibility — which is also the *true* reason that option leads.
``ClaimLicence.comparative`` is only granted on strict separation.

**Stating an unverified quantity as fact.** An equity valuation nobody
has checked is not a number, it is a placeholder. Any figure whose
estimate carries no evidence ref is licensed only as
``unverified`` — the assembler must mark it, and cannot quietly promote
it by rewording.

Neither rule is about tone. Both are about whether the underlying
computation supports the sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.decision_workspace.valuation import ValuationResult
from volvence_zero.regime import ParticipationLevel

# What kind of statement about the ranking is supported.
CLAIM_NONE = "none"
# No separation: may say an option is more robust / more reversible /
# preserves more choices. May NOT say it scores highest.
CLAIM_ROBUSTNESS = "robustness"
# Strict interval separation: a comparative claim is supported.
CLAIM_COMPARATIVE = "comparative"


@dataclass(frozen=True)
class ClaimLicence:
    """Which statements the current valuation supports."""

    claim_kind: str
    subject_ref: str | None
    # Quantities that may only appear marked as unverified, keyed by the
    # dimension they score.
    unverified_dimension_refs: tuple[str, ...] = ()
    # Unknowns that must remain visible in any stated conclusion. A
    # conclusion that silently drops the thing it depends on is worse
    # than no conclusion.
    must_surface_unknown_refs: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def may_state_a_winner(self) -> bool:
        return self.claim_kind == CLAIM_COMPARATIVE

    def permits(self, quantity_dimension_ref: str) -> bool:
        """Whether a figure for this dimension may be stated as fact."""
        return quantity_dimension_ref not in self.unverified_dimension_refs


def licence_for(
    result: ValuationResult, *, safety_hold: bool = False
) -> ClaimLicence:
    """Derive the claim licence from a valuation.

    ``safety_hold`` short-circuits everything. When the boundary owner
    has raised a safety band, no ranking claim is licensed at all —
    regardless of how well separated the intervals are, and regardless
    of how the user weighted the dimensions. Safety is not a dimension
    that can be outvoted by the others; that is the whole point of it
    sitting above the ranking rather than inside it.
    """
    if safety_hold:
        return ClaimLicence(
            claim_kind=CLAIM_NONE,
            subject_ref=None,
            rationale="licence: withheld (safety hold above ranking)",
        )
    unverified = tuple(
        sorted(
            {
                ref
                for option in result.options
                for ref in option.unsupported_dimension_refs
            }
        )
    )
    must_surface = tuple(
        sorted({u.unknown_ref for u in result.unknowns if u.is_worth_asking})
    )
    if not result.options:
        return ClaimLicence(
            claim_kind=CLAIM_NONE,
            subject_ref=None,
            unverified_dimension_refs=unverified,
            must_surface_unknown_refs=must_surface,
            rationale="licence: nothing valued",
        )
    if result.separated:
        return ClaimLicence(
            claim_kind=CLAIM_COMPARATIVE,
            subject_ref=result.leader_ref,
            unverified_dimension_refs=unverified,
            must_surface_unknown_refs=must_surface,
            rationale="licence: comparative (top interval clears the field)",
        )
    return ClaimLicence(
        claim_kind=CLAIM_ROBUSTNESS,
        subject_ref=result.most_robust_ref,
        unverified_dimension_refs=unverified,
        must_surface_unknown_refs=must_surface,
        rationale=(
            "licence: robustness only (intervals overlap; a comparative "
            "claim would assert a result the arithmetic does not have)"
        ),
    )


# ---------------------------------------------------------------------------
# Render plan
#
# The split here is deliberate: *what may be said* is decided in this tier,
# next to the arithmetic that licenses it; *how it is worded* belongs to the
# expression layer. Putting the constraint in the prompt instead would make
# it a suggestion, and the failure would be invisible — fluent text that
# quietly asserts a result the numbers do not support.
#
# It also means the contract tests can assert the constraint without
# asserting the wording, so rephrasing a sentence never silently relaxes
# what the system is entitled to claim.
# ---------------------------------------------------------------------------

PANORAMA_TIER_BRIEF = "brief"
PANORAMA_TIER_STRUCTURED = "structured"


@dataclass(frozen=True)
class PanoramaRenderPlan:
    """What the expression layer may put on screen this turn."""

    tier: str
    option_count: int
    dimension_count: int
    open_unknown_count: int
    # Licence-derived. ``claim_kind`` is CLAIM_NONE / CLAIM_ROBUSTNESS /
    # CLAIM_COMPARATIVE; ``subject_ref`` names the option the claim is
    # about, or None when no claim is licensed at all.
    claim_kind: str = CLAIM_NONE
    subject_ref: str | None = None
    # Unknowns that must remain visible in whatever is said. A conclusion
    # that silently drops the thing it depends on is worse than no
    # conclusion, because it reads as settled.
    surface_unknown_refs: tuple[str, ...] = ()
    # Dimensions whose figures may only appear marked as unverified.
    unverified_dimension_refs: tuple[str, ...] = ()
    # The single most useful thing to ask next, or None when nothing left
    # to learn could change the ranking — the termination condition.
    next_question_ref: str | None = None
    safety_hold: bool = False

    @property
    def may_state_a_winner(self) -> bool:
        return self.claim_kind == CLAIM_COMPARATIVE

    @property
    def may_rank_at_all(self) -> bool:
        return self.claim_kind != CLAIM_NONE


def plan_panorama_render(
    workspace: object, valuation: ValuationResult | None = None
) -> PanoramaRenderPlan | None:
    """Turn a published workspace into a render plan, or ``None``.

    ``None`` means nothing about a decision goes on screen this turn —
    the gate was closed. That is the common case by a wide margin, and
    it is a real absence rather than an empty section: there is nothing
    for a downstream renderer to accidentally expand.

    The BRIEF tier deliberately carries counts but no ranking claim. At
    that tier the system has noticed a decision is taking shape and can
    say so; it has not earned the right to lay one out.
    """
    engagement = getattr(workspace, "engagement", None)
    if engagement is None or engagement is ParticipationLevel.SILENT:
        return None
    safety_hold = bool(getattr(workspace, "safety_hold", False))
    options = getattr(workspace, "options", ())
    unknowns = getattr(workspace, "unknowns", ())
    dimensions = getattr(workspace, "dimension_refs", ())
    if engagement is ParticipationLevel.BRIEF:
        return PanoramaRenderPlan(
            tier=PANORAMA_TIER_BRIEF,
            option_count=len(options),
            dimension_count=0,
            open_unknown_count=len(unknowns),
            claim_kind=CLAIM_NONE,
            safety_hold=safety_hold,
        )
    licence = (
        licence_for(valuation, safety_hold=safety_hold)
        if valuation is not None
        else ClaimLicence(
            claim_kind=CLAIM_NONE,
            subject_ref=None,
            rationale="licence: no valuation available",
        )
    )
    next_question = (
        valuation.next_unknown_to_resolve() if valuation is not None else None
    )
    return PanoramaRenderPlan(
        tier=PANORAMA_TIER_STRUCTURED,
        option_count=len(options),
        dimension_count=len(dimensions),
        open_unknown_count=len(unknowns),
        claim_kind=licence.claim_kind,
        subject_ref=licence.subject_ref,
        surface_unknown_refs=licence.must_surface_unknown_refs,
        unverified_dimension_refs=licence.unverified_dimension_refs,
        next_question_ref=next_question,
        safety_hold=safety_hold,
    )
