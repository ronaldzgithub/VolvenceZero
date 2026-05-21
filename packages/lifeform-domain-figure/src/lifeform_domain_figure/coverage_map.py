"""Coverage map — the L4 not-known refusal backbone.

Builds a frozen :class:`FigureCoverageMap` artifact that the runtime
:class:`lifeform_expression.ScopeRefuser` (P3.2) consumes to decide
whether a user query falls inside the figure's documented coverage,
matches a profile boundary's out-of-scope topics, or sits in the
unmapped beyond.

Decision is purely **semantic distance based**: cosine similarity of
the query embedding against (a) corpus-derived in-domain centroids
and (b) reviewer-declared out-of-scope centroids. No keyword
matching is used (``no-keyword-matching-hacks.mdc``).

Three-state decision keeps the L4 contract enforceable:

* ``IN_DOMAIN``         — query lands within the documented coverage;
                          the runtime may proceed with grounded
                          generation (subject to L3 evidence checks).
* ``BOUNDARY_BLOCKED``  — query matches one of the profile's
                          ``out_of_scope_topics`` for some boundary;
                          the runtime must refuse / refer-out / soft
                          disclaim per the active ``coverage_policy``.
* ``OUT_OF_DOMAIN``     — query is outside in-domain and does not
                          match a declared boundary — the polite
                          "I never wrote about this" response.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum

from lifeform_domain_figure.profile import HistoricalFigureProfile
from lifeform_domain_figure.retrieval_index import (
    FigureRetrievalIndex,
    _cosine,
    _hashing_embedding,
    _tokenize,
)


_OUT_OF_SCOPE_TOPIC_NORMALIZER = re.compile(r"[_\-]+")


class CoverageDecision(str, Enum):
    """Three-state outcome of :meth:`FigureCoverageMap.classify_query`."""

    IN_DOMAIN = "in_domain"
    BOUNDARY_BLOCKED = "boundary_blocked"
    OUT_OF_DOMAIN = "out_of_domain"


@dataclass(frozen=True)
class CoverageClassification:
    """Per-query classification result returned by the coverage map."""

    decision: CoverageDecision
    closest_in_domain_label: str
    closest_in_domain_score: float
    closest_out_of_scope_label: str
    closest_out_of_scope_score: float
    rationale: str


@dataclass(frozen=True)
class _DomainCentroid:
    """Internal: a labelled centroid embedding for one in-domain topic."""

    label: str
    embedding: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class _OutOfScopeCentroid:
    """Internal: a labelled centroid embedding for a boundary out-of-scope topic."""

    label: str
    embedding: tuple[float, ...]
    boundary_id: str


@dataclass(frozen=True)
class FigureCoverageMap:
    """Frozen, deterministic in-corpus / out-of-scope coverage map.

    Built once by :func:`build_figure_coverage_map`; serialized inside
    the eventual ``FigureArtifactBundle``; consumed read-only by the
    runtime ``ScopeRefuser``.
    """

    figure_id: str
    domain_centroids: tuple[_DomainCentroid, ...]
    out_of_scope_centroids: tuple[_OutOfScopeCentroid, ...]
    in_domain_threshold: float
    boundary_threshold: float
    integrity_hash: str

    def __post_init__(self) -> None:
        if not self.figure_id.strip():
            raise ValueError("FigureCoverageMap.figure_id must be non-empty")
        if not self.domain_centroids:
            raise ValueError(
                "FigureCoverageMap.domain_centroids must be non-empty; "
                "the L4 contract needs at least one in-domain centroid."
            )
        if not 0.0 <= self.in_domain_threshold <= 1.0:
            raise ValueError(
                f"FigureCoverageMap.in_domain_threshold must be in [0,1], "
                f"got {self.in_domain_threshold!r}"
            )
        if not 0.0 <= self.boundary_threshold <= 1.0:
            raise ValueError(
                f"FigureCoverageMap.boundary_threshold must be in [0,1], "
                f"got {self.boundary_threshold!r}"
            )

    def classify_query(
        self,
        query: str,
        *,
        retrieval_index: "FigureRetrievalIndex | None" = None,
        retrieval_floor: float = 0.18,
        retrieval_top_k: int = 5,
    ) -> CoverageClassification:
        """Classify ``query`` as IN_DOMAIN, BOUNDARY_BLOCKED, or OUT_OF_DOMAIN.

        The boundary check runs first so that explicit reviewer
        declarations (e.g., ``post_1955_events``) take precedence
        over corpus-derived in-domain similarity.

        ``retrieval_index`` (debt #39): optional retrieval-augmented
        floor pass. When supplied, a query that falls *below* the
        in-domain centroid threshold gets a second chance via top-K
        cosine against the actual corpus chunks. If
        ``max(top_k cosine_score) >= retrieval_floor``, the query is
        upgraded to IN_DOMAIN with a rationale citing the chunk id.
        This fixes the Wave K Einstein prod bug where in-corpus
        relativity / postulate / theory questions were being L4-refused
        because the per-knowledge-seed centroid embedding was too
        narrow for the corpus's actual paraphrase coverage. The
        boundary check is not re-evaluated by the floor pass —
        explicit reviewer out-of-scope declarations still win.
        """

        query_tokens = _tokenize(query)
        query_embedding = _hashing_embedding(query_tokens)
        if not any(query_embedding):
            return CoverageClassification(
                decision=CoverageDecision.OUT_OF_DOMAIN,
                closest_in_domain_label="",
                closest_in_domain_score=0.0,
                closest_out_of_scope_label="",
                closest_out_of_scope_score=0.0,
                rationale=(
                    "Query produced no usable tokens after tokenization "
                    "(after dropping function words). Treated as out-of-domain."
                ),
            )
        in_domain_label = ""
        in_domain_score = 0.0
        for centroid in self.domain_centroids:
            score = _cosine(query_embedding, centroid.embedding)
            if score > in_domain_score:
                in_domain_score = score
                in_domain_label = centroid.label
        out_label = ""
        out_score = 0.0
        for centroid in self.out_of_scope_centroids:
            score = _cosine(query_embedding, centroid.embedding)
            if score > out_score:
                out_score = score
                out_label = centroid.label
        if out_score >= self.boundary_threshold and out_score >= in_domain_score:
            return CoverageClassification(
                decision=CoverageDecision.BOUNDARY_BLOCKED,
                closest_in_domain_label=in_domain_label,
                closest_in_domain_score=in_domain_score,
                closest_out_of_scope_label=out_label,
                closest_out_of_scope_score=out_score,
                rationale=(
                    f"Query semantically matches the reviewer-declared "
                    f"out-of-scope topic {out_label!r} (cosine={out_score:.3f} "
                    f">= boundary_threshold={self.boundary_threshold:.3f})."
                ),
            )
        if in_domain_score >= self.in_domain_threshold:
            return CoverageClassification(
                decision=CoverageDecision.IN_DOMAIN,
                closest_in_domain_label=in_domain_label,
                closest_in_domain_score=in_domain_score,
                closest_out_of_scope_label=out_label,
                closest_out_of_scope_score=out_score,
                rationale=(
                    f"Query falls inside the in-domain centroid "
                    f"{in_domain_label!r} (cosine={in_domain_score:.3f} "
                    f">= in_domain_threshold={self.in_domain_threshold:.3f})."
                ),
            )
        # Retrieval-augmented floor pass (debt #39). A query may miss
        # every static centroid yet still match real corpus content —
        # e.g. an Einstein 1916 GR paraphrase that lexically diverges
        # from the curated centroid title but cosine-matches a real
        # chunk. We only enter this path when the caller has explicitly
        # supplied a retrieval_index; the legacy two-centroid behaviour
        # is preserved for callers that don't pass it.
        if retrieval_index is not None:
            evidences = retrieval_index.retrieve(query, top_k=retrieval_top_k)
            best_floor_cosine = 0.0
            best_chunk_id = ""
            for ev in evidences:
                if ev.cosine_score > best_floor_cosine:
                    best_floor_cosine = ev.cosine_score
                    best_chunk_id = ev.chunk_id
            if best_floor_cosine >= retrieval_floor:
                return CoverageClassification(
                    decision=CoverageDecision.IN_DOMAIN,
                    closest_in_domain_label=f"retrieval_floor:{best_chunk_id}",
                    closest_in_domain_score=best_floor_cosine,
                    closest_out_of_scope_label=out_label,
                    closest_out_of_scope_score=out_score,
                    rationale=(
                        f"Query missed every static in-domain centroid "
                        f"(best cosine={in_domain_score:.3f} < "
                        f"{self.in_domain_threshold:.3f}) but matched "
                        f"real corpus chunk {best_chunk_id!r} via the "
                        f"retrieval-augmented floor (cosine={best_floor_cosine:.3f} "
                        f">= retrieval_floor={retrieval_floor:.3f})."
                    ),
                )
        return CoverageClassification(
            decision=CoverageDecision.OUT_OF_DOMAIN,
            closest_in_domain_label=in_domain_label,
            closest_in_domain_score=in_domain_score,
            closest_out_of_scope_label=out_label,
            closest_out_of_scope_score=out_score,
            rationale=(
                f"Query did not clear in-domain threshold "
                f"(cosine={in_domain_score:.3f} < {self.in_domain_threshold:.3f}) "
                f"and did not match any out-of-scope topic above "
                f"{self.boundary_threshold:.3f}."
            ),
        )


def _normalise_topic_label(topic: str) -> str:
    """Convert ``post_1955_events`` / ``medical-diagnosis`` into a token stream.

    Out-of-scope topics are declared by reviewers as snake_case or
    kebab-case identifiers; the centroid embedding has to come from
    actual word tokens, so we split the identifier into words before
    embedding.
    """

    return _OUT_OF_SCOPE_TOPIC_NORMALIZER.sub(" ", topic).strip()


def build_figure_coverage_map(
    *,
    figure_id: str,
    profile: HistoricalFigureProfile,
    retrieval_index: FigureRetrievalIndex,
    in_domain_threshold: float = 0.18,
    boundary_threshold: float = 0.16,
) -> FigureCoverageMap:
    """Build a :class:`FigureCoverageMap` from profile + retrieval index.

    In-domain centroids combine two sources:

    1. Per-domain centroid from the profile's :attr:`domain_coverage_seed`
       (each seed becomes one centroid; embedding seeded from the
       normalised seed label).
    2. Per-knowledge-seed centroid from each knowledge seed's title +
       summary embedding (the load-bearing source — these are
       reviewer-curated topical anchors).

    The corpus-wide mean embedding is **deliberately not** used as a
    centroid: it is too flat (sum of every hashing bucket the corpus
    touches) and inflates similarity for unrelated queries via hash
    collisions. The L4 contract demands the false-positive rate stay
    low, which a flat mean centroid breaks.

    Out-of-scope centroids come from the union of every boundary's
    :attr:`out_of_scope_topics` declarations.
    """

    if not 0.0 < in_domain_threshold <= 1.0:
        raise ValueError(
            f"build_figure_coverage_map: in_domain_threshold must be in "
            f"(0,1], got {in_domain_threshold!r}"
        )
    if not 0.0 < boundary_threshold <= 1.0:
        raise ValueError(
            f"build_figure_coverage_map: boundary_threshold must be in "
            f"(0,1], got {boundary_threshold!r}"
        )

    centroids: list[_DomainCentroid] = []

    for domain_label in profile.domain_coverage_seed:
        normalised = _normalise_topic_label(domain_label)
        embedding = _hashing_embedding(_tokenize(normalised))
        if any(embedding):
            centroids.append(
                _DomainCentroid(
                    label=domain_label,
                    embedding=embedding,
                    source="profile.domain_coverage_seed",
                )
            )

    for seed in profile.knowledge_seeds:
        text = f"{seed.title}. {seed.summary} {seed.snippet}"
        embedding = _hashing_embedding(_tokenize(text))
        if any(embedding):
            centroids.append(
                _DomainCentroid(
                    label=f"{seed.domain}::{seed.seed_id}",
                    embedding=embedding,
                    source="profile.knowledge_seeds",
                )
            )

    # The retrieval index is part of the integrity hash but its mean
    # is intentionally not used as a centroid; see docstring.
    if not centroids:
        raise ValueError(
            "build_figure_coverage_map: no in-domain centroids could be "
            "derived from the profile or retrieval index. Refusing to "
            "build an empty coverage map."
        )

    out_of_scope_centroids: list[_OutOfScopeCentroid] = []
    seen_labels: set[str] = set()
    for boundary in profile.boundary_priors:
        for topic in boundary.out_of_scope_topics:
            if topic in seen_labels:
                continue
            seen_labels.add(topic)
            normalised = _normalise_topic_label(topic)
            embedding = _hashing_embedding(_tokenize(normalised))
            if not any(embedding):
                continue
            out_of_scope_centroids.append(
                _OutOfScopeCentroid(
                    label=topic,
                    embedding=embedding,
                    boundary_id=boundary.boundary_id,
                )
            )

    integrity_payload = (
        figure_id,
        tuple((c.label, c.source) for c in centroids),
        tuple((c.label, c.boundary_id) for c in out_of_scope_centroids),
        round(in_domain_threshold, 6),
        round(boundary_threshold, 6),
        retrieval_index.integrity_hash,
    )
    integrity_hash = hashlib.sha256(
        repr(integrity_payload).encode("utf-8")
    ).hexdigest()
    return FigureCoverageMap(
        figure_id=figure_id,
        domain_centroids=tuple(centroids),
        out_of_scope_centroids=tuple(out_of_scope_centroids),
        in_domain_threshold=in_domain_threshold,
        boundary_threshold=boundary_threshold,
        integrity_hash=integrity_hash,
    )


__all__ = [
    "CoverageClassification",
    "CoverageDecision",
    "FigureCoverageMap",
    "build_figure_coverage_map",
]
