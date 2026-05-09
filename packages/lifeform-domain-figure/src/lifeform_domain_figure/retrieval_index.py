"""Citation-grounded retrieval index — the L3 backbone.

Builds a chunk-level retrieval index from a tuple of
:class:`IngestionEnvelope` instances produced by the corpus
adapters in :mod:`lifeform_domain_figure.corpus`. The output is a
frozen :class:`FigureRetrievalIndex` artifact that travels inside
the eventual ``FigureArtifactBundle`` (P2.3) and is consumed by the
runtime ``GroundedDecoder`` (P3.1) to verify every substantive
assertion has at least one citation-quality evidence pointer.

Design choices:

* **Self-contained scoring.** BM25 + a 256-dim hashing embedding
  combined with tunable weights, both implemented in pure stdlib.
  The figure vertical does NOT import ``vz-substrate`` to share
  ``semantic_feature_surface_from_text`` — adding that dependency
  would couple the artifact to a kernel sub-package and the scoring
  primitives we need are 30 lines.
* **Frozen, deterministic.** All tunable parameters land on the
  index dataclass at build time. ``retrieve(query)`` is a pure
  function of (index, query); replays from a saved bundle reproduce
  the same evidence rankings byte-for-byte.
* **Citation locators are first-class.** Every retrieval result
  carries the originating chunk's locator unchanged. ``GroundedDecoder``
  surfaces this locator verbatim in the L3 evidence pointer.
* **No keyword heuristics.** Scoring depends only on word-frequency
  statistics and hashing-embedding cosine similarity; no decisions
  branch on specific surface tokens (``no-keyword-matching-hacks.mdc``).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from lifeform_ingestion.envelope import IngestionEnvelope


_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u4e00-\u9fff]+")
_EMBEDDING_DIM = 256
_BM25_K1 = 1.5
_BM25_B = 0.75
# Tokens with IDF below this floor are treated as effective stopwords:
# they appear in too large a fraction of the corpus to contribute
# discriminating signal. We drop them at scoring time so a query like
# "...in 2024" cannot spike BM25 on the word "in" alone in a small
# corpus where IDF damping is otherwise insufficient.
_BM25_IDF_FLOOR = 0.30
# Hard stopword list for English function words. With small corpora
# the IDF floor alone is not enough to suppress "with" / "and" /
# "this" type words, because their document frequency is moderate
# rather than universal. We drop them at *both* index and query
# time so they never contribute to BM25 or to the hashing-embedding
# cosine score. The list is intentionally short and conservative —
# only words with no plausible content meaning. Future verticals
# focused on Chinese / German / etc. should extend this with a
# per-language list rather than rewrite the indexer.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "these", "those",
        "are", "was", "were", "been", "being", "have", "has", "had",
        "not", "but", "any", "all", "some", "from", "which", "what",
        "will", "would", "could", "should", "shall", "may", "might",
        "you", "your", "yours", "they", "them", "their", "his", "her",
        "him", "she", "our", "out", "into", "onto", "than", "then",
        "there", "here", "where", "when", "such", "only", "even",
        "also", "more", "most", "very", "much", "many", "few", "one",
        "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten",
    }
)


_MIN_TOKEN_LEN = 3


def _tokenize(text: str) -> tuple[str, ...]:
    """Tokenize for indexing / scoring.

    Tokens shorter than :data:`_MIN_TOKEN_LEN` are dropped: across
    every supported language the surviving function-word burden of
    1-2 character tokens (English ``in / of / a / to``, German ``in /
    zu``, French ``à / le``) introduces BM25 noise that swamps real
    evidence in small corpora. The cutoff is chosen to still admit
    short content words like ``why``, ``law``, ``one`` and Chinese
    bigrams while rejecting structural particles.
    """

    out: list[str] = []
    for token in _WORD_RE.findall(text):
        lowered = token.lower()
        if len(lowered) < _MIN_TOKEN_LEN:
            continue
        if lowered in _STOPWORDS:
            continue
        out.append(lowered)
    return tuple(out)


def _hashing_embedding(tokens: tuple[str, ...], *, dim: int = _EMBEDDING_DIM) -> tuple[float, ...]:
    """Deterministic 256-dim hashing embedding with signed accumulation.

    Mirrors the shape of ``vz-substrate``'s
    ``_semantic_embedding`` without taking a dependency on that
    module. Two tokens hashed to the same bucket can cancel by sign
    so the embedding is not just an OR of buckets.
    """

    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0] * dim
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-9:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class RetrievalEvidence:
    """One scored evidence pointer surfaced by :meth:`FigureRetrievalIndex.retrieve`."""

    chunk_id: str
    locator: str
    text: str
    score: float
    bm25_score: float
    cosine_score: float
    source_envelope_id: str

    @property
    def citation(self) -> str:
        """A short canonical citation string for L3 evidence pointers."""
        return f"{self.locator} | {self.source_envelope_id}"


@dataclass(frozen=True)
class _ChunkRecord:
    """Per-chunk index payload (private to the index)."""

    envelope_id: str
    chunk_id: str
    locator: str
    text: str
    tokens: tuple[str, ...]
    term_freqs: tuple[tuple[str, int], ...]
    embedding: tuple[float, ...]
    chunk_length: int
    confidence: float


@dataclass(frozen=True)
class FigureRetrievalIndex:
    """Frozen, deterministic citation-grounded retrieval index.

    Built once from one or more ingestion envelopes; saved as part
    of a :class:`FigureArtifactBundle`; consumed read-only by
    ``GroundedDecoder``.
    """

    figure_id: str
    chunk_records: tuple[_ChunkRecord, ...]
    inverse_doc_freq: tuple[tuple[str, float], ...]
    avg_chunk_length: float
    integrity_hash: str
    bm25_weight: float = 0.6
    cosine_weight: float = 0.4

    def __post_init__(self) -> None:
        if not self.figure_id.strip():
            raise ValueError("FigureRetrievalIndex.figure_id must be non-empty")
        if not self.chunk_records:
            raise ValueError(
                "FigureRetrievalIndex.chunk_records must be non-empty; "
                "refusing to build an empty retrieval index."
            )
        if not 0.0 <= self.bm25_weight <= 1.0:
            raise ValueError(
                f"FigureRetrievalIndex.bm25_weight must be in [0,1], "
                f"got {self.bm25_weight!r}"
            )
        if not 0.0 <= self.cosine_weight <= 1.0:
            raise ValueError(
                f"FigureRetrievalIndex.cosine_weight must be in [0,1], "
                f"got {self.cosine_weight!r}"
            )

    @property
    def total_chunks(self) -> int:
        return len(self.chunk_records)

    def retrieve(self, query: str, top_k: int = 5) -> tuple[RetrievalEvidence, ...]:
        """Return up to ``top_k`` ranked :class:`RetrievalEvidence` for ``query``.

        Combines BM25 and cosine similarity according to the
        index's weights. Empty / whitespace-only queries return an
        empty tuple — the caller is expected to fail loud on that
        upstream rather than receive a meaningless result.
        """

        if top_k <= 0:
            raise ValueError(
                f"FigureRetrievalIndex.retrieve top_k must be > 0, got {top_k!r}"
            )
        query_tokens = _tokenize(query)
        if not query_tokens:
            return ()
        query_embedding = _hashing_embedding(query_tokens)
        idf_map = dict(self.inverse_doc_freq)
        results: list[RetrievalEvidence] = []
        for record in self.chunk_records:
            term_freq_map = dict(record.term_freqs)
            bm25 = _bm25_score(
                query_tokens=query_tokens,
                term_freqs=term_freq_map,
                idf_map=idf_map,
                chunk_length=record.chunk_length,
                avg_chunk_length=self.avg_chunk_length,
            )
            cosine = _cosine(query_embedding, record.embedding)
            combined = self.bm25_weight * bm25 + self.cosine_weight * cosine
            combined *= record.confidence
            results.append(
                RetrievalEvidence(
                    chunk_id=record.chunk_id,
                    locator=record.locator,
                    text=record.text,
                    score=combined,
                    bm25_score=bm25,
                    cosine_score=cosine,
                    source_envelope_id=record.envelope_id,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return tuple(results[:top_k])

    def assertion_is_supported(
        self,
        assertion: str,
        *,
        score_threshold: float = 0.18,
        cosine_floor: float = 0.05,
        top_k: int = 3,
    ) -> tuple[RetrievalEvidence, ...]:
        """Return supporting evidence for an assertion, or empty if none.

        Used by :class:`lifeform_expression.GroundedDecoder` (P3.1) as
        the L3 enforcement primitive: an empty return value is the
        signal to fail loudly (refuse the assertion / regenerate /
        fall through to the scope refuser), per
        ``no-swallow-errors-no-hasattr-abuse.mdc``.

        Both ``score_threshold`` (combined BM25+cosine) AND
        ``cosine_floor`` (raw semantic alignment) must clear. The
        cosine floor exists because, in small corpora, BM25 alone can
        spike on a single term that survives IDF damping; requiring
        a non-trivial semantic match prevents an off-topic assertion
        from claiming evidence on a low-information word collision.
        """

        evidence = self.retrieve(assertion, top_k=top_k)
        return tuple(
            item
            for item in evidence
            if item.score >= score_threshold and item.cosine_score >= cosine_floor
        )


def build_figure_retrieval_index(
    *,
    figure_id: str,
    envelopes: tuple[IngestionEnvelope, ...],
    bm25_weight: float = 0.6,
    cosine_weight: float = 0.4,
) -> FigureRetrievalIndex:
    """Build a :class:`FigureRetrievalIndex` from ingestion envelopes.

    All chunks across all envelopes are folded into one index. Chunks
    with non-empty ``parse_error`` are dropped — the resulting index
    must only point at clean, ingestible primary-source text.
    """

    if not envelopes:
        raise ValueError(
            "build_figure_retrieval_index: envelopes tuple must be non-empty"
        )
    if not figure_id.strip():
        raise ValueError(
            "build_figure_retrieval_index: figure_id must be non-empty"
        )
    chunk_records: list[_ChunkRecord] = []
    document_frequencies: dict[str, int] = {}
    for envelope in envelopes:
        for chunk in envelope.successful_chunks:
            tokens = _tokenize(chunk.text)
            if not tokens:
                continue
            term_freqs: dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
            for token in set(tokens):
                document_frequencies[token] = (
                    document_frequencies.get(token, 0) + 1
                )
            embedding = _hashing_embedding(tokens)
            chunk_records.append(
                _ChunkRecord(
                    envelope_id=envelope.envelope_id,
                    chunk_id=chunk.chunk_id,
                    locator=chunk.locator,
                    text=chunk.text,
                    tokens=tokens,
                    term_freqs=tuple(sorted(term_freqs.items())),
                    embedding=embedding,
                    chunk_length=len(tokens),
                    confidence=chunk.confidence,
                )
            )
    if not chunk_records:
        raise ValueError(
            "build_figure_retrieval_index: no successful chunks across all "
            "envelopes — refusing to build an empty retrieval index."
        )
    avg_chunk_length = sum(rec.chunk_length for rec in chunk_records) / len(chunk_records)
    total_chunks = len(chunk_records)
    idf_pairs: list[tuple[str, float]] = []
    for token, doc_freq in document_frequencies.items():
        idf = math.log(1.0 + (total_chunks - doc_freq + 0.5) / (doc_freq + 0.5))
        idf_pairs.append((token, idf))
    idf_pairs.sort()
    integrity_payload = (
        figure_id,
        tuple(rec.envelope_id for rec in chunk_records),
        tuple(rec.chunk_id for rec in chunk_records),
        round(avg_chunk_length, 6),
        round(bm25_weight, 6),
        round(cosine_weight, 6),
    )
    integrity_hash = hashlib.sha256(
        repr(integrity_payload).encode("utf-8")
    ).hexdigest()
    return FigureRetrievalIndex(
        figure_id=figure_id,
        chunk_records=tuple(chunk_records),
        inverse_doc_freq=tuple(idf_pairs),
        avg_chunk_length=avg_chunk_length,
        integrity_hash=integrity_hash,
        bm25_weight=bm25_weight,
        cosine_weight=cosine_weight,
    )


def _bm25_score(
    *,
    query_tokens: tuple[str, ...],
    term_freqs: dict[str, int],
    idf_map: dict[str, float],
    chunk_length: int,
    avg_chunk_length: float,
) -> float:
    if avg_chunk_length <= 0.0:
        return 0.0
    score = 0.0
    seen: set[str] = set()
    for token in query_tokens:
        if token in seen:
            continue
        seen.add(token)
        idf = idf_map.get(token, 0.0)
        # Drop very common tokens (idf below floor) — they saturate
        # BM25 in small corpora without contributing real evidence.
        if idf < _BM25_IDF_FLOOR:
            continue
        tf = term_freqs.get(token, 0)
        if tf == 0:
            continue
        norm = 1.0 - _BM25_B + _BM25_B * (chunk_length / avg_chunk_length)
        score += idf * (tf * (_BM25_K1 + 1)) / (tf + _BM25_K1 * norm)
    return score


__all__ = [
    "FigureRetrievalIndex",
    "RetrievalEvidence",
    "build_figure_retrieval_index",
]
