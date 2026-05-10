"""Bridge from a :class:`MetadataDigest` into the existing coverage map flow.

The wheel's existing :func:`build_figure_coverage_map` already builds
a :class:`FigureCoverageMap` from a :class:`HistoricalFigureProfile`
+ a :class:`FigureRetrievalIndex`. This module adds a typed bridge
that takes a metadata digest and produces an **enriched profile** —
a profile whose ``domain_coverage_seed`` is widened with metadata
labels and whose ``boundary_priors`` carry post-lifespan
out-of-scope topics derived from the lifespan.

The bridge is **pure**: it neither mutates the input profile nor
touches the retrieval / LoRA pipelines. Callers feed the enriched
profile into the existing coverage map builder unchanged.

T3 严禁污染 invariant:
    The metadata layer never touches the retrieval index or the
    LoRA training data. The bridge here only widens the profile's
    in-domain coverage seed and out-of-scope boundary topics —
    explicit places where reviewer-curated metadata is expected to
    flow.
"""

from __future__ import annotations

from dataclasses import replace as _replace

from lifeform_domain_figure.metadata.records import MetadataDigest
from lifeform_domain_figure.profile import (
    FigureBoundaryPrior,
    HistoricalFigureProfile,
)


_POST_LIFESPAN_BOUNDARY_ID = "metadata:post-lifespan"


def _post_lifespan_boundary(
    *,
    death_year: int,
    figure_id: str,
) -> FigureBoundaryPrior:
    return FigureBoundaryPrior(
        boundary_id=f"{_POST_LIFESPAN_BOUNDARY_ID}:{figure_id}",
        regime_id=None,
        trigger_reasons=("query-about-event-after-figure-lifespan",),
        answer_depth_limit_hint="absolute",
        clarification_required=False,
        refer_out_required=False,
        blocked_topics=("post-mortem-events",),
        required_disclaimers=(
            f"Figure died in {death_year}; statements about events after "
            f"that date are not theirs.",
        ),
        confidence=0.99,
        description=(
            f"Metadata-derived absolute boundary on events after "
            f"{death_year}. Auto-generated from a Wikidata lifespan record."
        ),
        out_of_scope_topics=(
            f"post_{death_year}_events",
            "contemporary_events",
        ),
    )


def enrich_profile_with_metadata(
    profile: HistoricalFigureProfile,
    digest: MetadataDigest,
) -> HistoricalFigureProfile:
    """Return a new profile with metadata-derived widenings applied.

    Two widenings:

    1. ``domain_coverage_seed`` is unioned with every in-domain hint
       label from ``digest.coverage_hints`` (deduplicated, original
       order preserved).
    2. ``boundary_priors`` gets one extra entry capturing the
       post-lifespan out-of-scope boundary, **only** when the digest
       has a lifespan with a known ``death_year`` and the profile
       does not already declare an absolute post-lifespan boundary.

    All other fields pass through unchanged. The reviewer / curation
    flow is then expected to re-call
    :func:`lifeform_domain_figure.build_figure_coverage_map` on the
    enriched profile so the L4 centroids reflect the widened seed.
    """

    if digest.figure_id != profile.profile_id:
        raise ValueError(
            f"enrich_profile_with_metadata: digest.figure_id "
            f"{digest.figure_id!r} does not match profile.profile_id "
            f"{profile.profile_id!r}"
        )
    seen_seed = set(profile.domain_coverage_seed)
    extra_seed: list[str] = []
    for hint in digest.coverage_hints:
        if hint.is_out_of_scope:
            continue
        if hint.label in seen_seed:
            continue
        seen_seed.add(hint.label)
        extra_seed.append(hint.label)
    new_coverage_seed = profile.domain_coverage_seed + tuple(extra_seed)
    new_boundaries = profile.boundary_priors
    if digest.lifespan is not None and digest.lifespan.death_year is not None:
        already_declared = any(
            boundary.answer_depth_limit_hint == "absolute"
            and any(
                topic.startswith("post_")
                for topic in boundary.out_of_scope_topics
            )
            for boundary in profile.boundary_priors
        )
        if not already_declared:
            new_boundaries = profile.boundary_priors + (
                _post_lifespan_boundary(
                    death_year=digest.lifespan.death_year,
                    figure_id=profile.profile_id,
                ),
            )
    return _replace(
        profile,
        domain_coverage_seed=new_coverage_seed,
        boundary_priors=new_boundaries,
        version=f"{profile.version}+metadata:{digest.fingerprint[:8]}",
    )


__all__ = [
    "enrich_profile_with_metadata",
]
