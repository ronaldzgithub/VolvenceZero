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
