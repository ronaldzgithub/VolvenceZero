"""Research affordances for the growth-advisor vertical.

One read-only TOOL descriptor, ``research_public_company``, plus the
backend protocol a host implements to serve it.

Why this exists as an affordance rather than as a helper the response
layer calls: a decision panorama that quotes a valuation has to be able
to say where that number came from. Routing the lookup through the
affordance layer is what makes the answer arrive as an
``EnvironmentOutcome`` with provenance attached, land in
``belief_assumption``, and become citable by ``decision_workspace``. A
side channel would produce the same sentence with none of the lineage —
and the sentence is the dangerous part.

Two boundaries are built into the descriptor rather than left to the
caller:

**Public-company scope only.** Researching a specific private
individual — someone's spouse, their employer, their finances — is a
different act with different consequences, and it is exactly what a
"help me decide about my marriage" conversation would invite. The
descriptor's parameters take a company identifier, not a person, and
``when_not_to_use`` says so in the text the model actually reads.

**Consent-gated and regime-blocked.** The tool requires an explicit
``public_research`` grant, and is blocked in the emotional-support and
repair regimes. Someone in acute distress is not asking to have their
partner investigated, and a system that starts looking things up at that
moment has misread the room in a way that is hard to take back.
"""

from __future__ import annotations

from lifeform_domain_growth_advisor.research_affordances.descriptors import (
    CONSENT_PUBLIC_RESEARCH,
    RESEARCH_AFFORDANCE_DESCRIPTORS,
    RESEARCH_PUBLIC_COMPANY,
)

__all__ = [
    "CONSENT_PUBLIC_RESEARCH",
    "RESEARCH_AFFORDANCE_DESCRIPTORS",
    "RESEARCH_PUBLIC_COMPANY",
]
