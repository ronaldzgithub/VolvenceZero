"""Descriptor for the public-company research affordance.

The output schema is the load-bearing part. It requires every returned
claim to carry ``source`` / ``as_of`` / ``scope``, because a claim
missing any of the three is not evidence — and the downstream adapter
enforces that by capping such a claim's confidence below the belief
threshold, so it surfaces as an open question instead of a fact.

Making the schema demand provenance means a backend that cannot supply
it fails schema validation at the boundary rather than silently
returning authoritative-sounding prose.
"""

from __future__ import annotations

from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceSafety,
)


CONSENT_PUBLIC_RESEARCH = "public_research"
"""Consent grant for looking up public information about a company.

Deliberately narrow. It does not cover looking up a person, and there
is intentionally no companion grant that would: a host that wants to
research an individual should have to add that capability explicitly,
with its own review, rather than inherit it from a grant the user gave
for something else.
"""


# A research call during acute distress or an active rupture reads as
# the system investigating the user's life instead of staying with them.
# The block is conservative on purpose: the panorama gate may well be
# open in these regimes (a hard decision can coexist with distress), and
# the two judgements are independent.
_RESEARCH_BLOCKED_REGIMES: tuple[str, ...] = (
    "casual_social",
    "emotional_support",
    "repair_and_deescalation",
)


_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "statement": {"type": "string"},
        # The three fields that make a claim checkable. Required, not
        # optional-with-a-default: a backend that cannot say where a
        # figure came from should fail here, loudly, rather than emit
        # an empty string that reads as "no source needed".
        "source": {"type": "string"},
        "as_of": {"type": "string"},
        "scope": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "claim_id",
        "statement",
        "source",
        "as_of",
        "scope",
        "confidence",
    ],
    "additionalProperties": False,
}


RESEARCH_PUBLIC_COMPANY = AffordanceDescriptor(
    name="research_public_company",
    kind=AffordanceKind.TOOL,
    version="0.1.0",
    display_name="Research a public company",
    description=(
        "Look up publicly available information about a named company — "
        "funding history, disclosed valuation ranges, stage, sector "
        "comparables — and return it as individually sourced claims."
    ),
    when_to_use=(
        "Use when a decision under discussion hinges on a company's "
        "situation that neither the user nor the system currently knows: "
        "what stage it is at, whether a valuation has been disclosed, how "
        "comparable firms in its sector have fared. Prefer this over "
        "reasoning from an unverified impression of the company, and "
        "prefer it specifically when the unresolved question is the one "
        "that would change which option looks best."
    ),
    when_not_to_use=(
        "Do not use to research a private individual — a spouse, a "
        "colleague, an employer as a person. That is a different act with "
        "different consequences, and this tool does not do it. Do not use "
        "to substitute for documents the user can obtain directly, such "
        "as their own equity paperwork; the right move there is to say "
        "the question is unresolved and who can resolve it. Do not use "
        "while the user is in acute distress, and do not use to produce a "
        "number when no disclosed figure exists — an absent figure is an "
        "open question, not something to estimate."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            # A company, never a person. The parameter name and its
            # description are part of what the model reads when deciding
            # how to call this.
            "company": {"type": "string"},
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
        },
        "required": ["company"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "claims": {"type": "array", "items": _CLAIM_SCHEMA},
            # Questions the lookup could not answer. These matter as much
            # as the claims: an unanswerable question stays an open
            # unknown rather than quietly disappearing from the panorama.
            "unanswered": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["claims", "unanswered"],
        "additionalProperties": False,
    },
    cost_model=AffordanceCost(
        latency_class=AffordanceLatencyClass.SLOW,
        monetary_class=AffordanceMonetaryClass.LOW,
    ),
    safety_model=AffordanceSafety(
        requires_user_confirmation=False,
        irreversible=False,
        requires_consent_grant=(CONSENT_PUBLIC_RESEARCH,),
        blocked_in_regimes=_RESEARCH_BLOCKED_REGIMES,
        # Auditable even though it is read-only: this call reaches
        # outside the conversation on the user's behalf, about a subject
        # adjacent to their private life. Someone should be able to see
        # afterwards exactly what was looked up and when.
        audit_required=True,
    ),
    preconditions=("scene.is_open",),
    affordance_tags=("research", "public-record", "decision-support"),
    examples=(
        "research_public_company(company='Acme Robotics')",
        (
            "research_public_company(company='Acme Robotics', "
            "questions=['latest disclosed round', 'sector comparables'])"
        ),
    ),
    source_path=(
        "lifeform_domain_growth_advisor.research_affordances.descriptors:"
        "research_public_company"
    ),
)


RESEARCH_AFFORDANCE_DESCRIPTORS: tuple[AffordanceDescriptor, ...] = (
    RESEARCH_PUBLIC_COMPANY,
)
