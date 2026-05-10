"""Cross-source deduplication helpers for primary-source corpora.

When the same letter / paper / lecture is digitised twice (once on
Wikisource, once on the Internet Archive) the figure vertical must
not double-count it: the retrieval index would over-rank duplicates
and the L3 evidence pointer would surface conflicting locators for
the same underlying text.

The kernel-side ``IngestionEnvelope`` already enforces unique
``chunk_id`` *within* an envelope. This module enforces the next
layer: *across* envelopes / source kinds, no two records share the
same byte-level text content (sha256 over UTF-8 text).

Strategy:

* Take a tuple of ``IngestionEnvelope`` instances (the output of
  :func:`lifeform_domain_figure.build_figure_ingestion_envelope`).
* Compute the SHA-256 over each chunk's UTF-8 text.
* Group chunks by hash; pick the **canonical** representative as the
  one whose locator declares the highest-trust source kind (papers
  > letters > lectures > notebooks — papers are the most-curated
  ground truth; notebooks are working drafts).
* Emit a typed report so reviewers see what was collapsed and why.

This module does **not** mutate envelopes. It returns a typed
:class:`DedupReport` plus a tuple of canonical chunk ids the
downstream retrieval index builder can use to filter input.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lifeform_ingestion.envelope import IngestionEnvelope


_LOCATOR_TRUST_RANK: tuple[str, ...] = ("paper", "letter", "lecture", "notebook")


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _trust_rank(locator: str) -> int:
    head = locator.split(":", 1)[0]
    if head in _LOCATOR_TRUST_RANK:
        return _LOCATOR_TRUST_RANK.index(head)
    # Unknown locator prefixes get the lowest trust so a reviewer
    # adding a new locator format does not silently outrank papers.
    return len(_LOCATOR_TRUST_RANK)


@dataclass(frozen=True)
class DuplicateGroup:
    """One group of byte-identical chunks across envelopes."""

    text_sha256: str
    canonical_chunk_id: str
    canonical_locator: str
    canonical_envelope_id: str
    duplicate_chunk_ids: tuple[str, ...]
    duplicate_locators: tuple[str, ...]


@dataclass(frozen=True)
class DedupReport:
    """Result of running :func:`compute_dedup_report` over envelopes."""

    total_chunks: int
    unique_chunks: int
    duplicate_groups: tuple[DuplicateGroup, ...]
    canonical_chunk_ids: frozenset[str]

    @property
    def duplicate_chunk_count(self) -> int:
        return self.total_chunks - self.unique_chunks


def compute_dedup_report(
    envelopes: tuple[IngestionEnvelope, ...],
) -> DedupReport:
    """Return a typed dedup report over the given ingestion envelopes.

    Each chunk participates in exactly one group; chunks with no
    duplicates form a one-element group. Only the *canonical* chunk
    id of each group ends up in :attr:`DedupReport.canonical_chunk_ids`,
    so a downstream consumer can filter envelope chunks by membership
    in that frozenset.

    Empty envelopes input is rejected loudly: a caller that has
    nothing to dedup should not be in this module.
    """

    if not envelopes:
        raise ValueError(
            "compute_dedup_report: envelopes tuple must be non-empty"
        )
    by_hash: dict[str, list[tuple[str, str, str]]] = {}
    total_chunks = 0
    for envelope in envelopes:
        for chunk in envelope.successful_chunks:
            total_chunks += 1
            sha = _text_sha(chunk.text)
            by_hash.setdefault(sha, []).append(
                (chunk.chunk_id, chunk.locator, envelope.envelope_id)
            )
    duplicate_groups: list[DuplicateGroup] = []
    canonical_ids: set[str] = set()
    for sha, members in by_hash.items():
        members_sorted = sorted(members, key=lambda m: (_trust_rank(m[1]), m[0]))
        canonical = members_sorted[0]
        duplicates = members_sorted[1:]
        canonical_ids.add(canonical[0])
        if not duplicates:
            continue
        duplicate_groups.append(
            DuplicateGroup(
                text_sha256=sha,
                canonical_chunk_id=canonical[0],
                canonical_locator=canonical[1],
                canonical_envelope_id=canonical[2],
                duplicate_chunk_ids=tuple(member[0] for member in duplicates),
                duplicate_locators=tuple(member[1] for member in duplicates),
            )
        )
    return DedupReport(
        total_chunks=total_chunks,
        unique_chunks=len(by_hash),
        duplicate_groups=tuple(duplicate_groups),
        canonical_chunk_ids=frozenset(canonical_ids),
    )


__all__ = [
    "DedupReport",
    "DuplicateGroup",
    "compute_dedup_report",
]
