"""LLM-assisted extraction for character profiles + narrative arcs.

This sub-package wraps an LLM text-generation provider with strict
JSON parsing + a mandatory human-review path. The output is always
a *candidate* — not a runnable artifact — until a reviewer accepts
it.

Usage:

    from lifeform_domain_character.extraction import (
        extract_profile_candidate,
        review_profile_candidate,
    )

    candidate = extract_profile_candidate(novel_text, llm_runtime=...)
    profile = review_profile_candidate(candidate, reviewer="me",
                                       accepted_fields=..., modifications=...)

Why a forced review step:

1. ``CharacterSoulProfile.reviewed_by`` is a mandatory non-empty
   field on the schema; an unreviewed candidate cannot become a
   profile by accident.
2. LLM outputs vary across runs; "review" makes the operator the
   accountable party for any specific saved profile.
3. The candidate carries ``requires_review`` listing fields the LLM
   was uncertain about; the reviewer addresses those explicitly.
"""

from __future__ import annotations

from lifeform_domain_character.extraction.profile_llm import (
    ReviewedProfileCandidate,
    extract_profile_candidate,
    review_profile_candidate,
)
from lifeform_domain_character.extraction.scene_llm import (
    NarrativeArcCandidate,
    extract_arc_candidate,
    review_arc_candidate,
)


__all__ = [
    "NarrativeArcCandidate",
    "ReviewedProfileCandidate",
    "extract_arc_candidate",
    "extract_profile_candidate",
    "review_arc_candidate",
    "review_profile_candidate",
]
