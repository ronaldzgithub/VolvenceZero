"""Style prior — the L1 voice-shape backbone.

Builds a frozen :class:`FigureStylePrior` artifact that captures the
**statistical voice shape** of a figure's primary corpus: top-word
frequencies, character / word n-gram distributions, sentence-length
percentiles, and a curated term list. The runtime
:class:`lifeform_expression.StylePriorInjector` (P3.3) consumes this
artifact to bias the LLM's decoder toward the figure's lexical
fingerprint without retraining the base.

Crucially this is **read-only data** — no model weights, no training,
no GPU. The artifact ships inside :class:`FigureArtifactBundle` and
travels through DLaaS adopt to the runtime.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from lifeform_ingestion.envelope import IngestionEnvelope


_SENTENCE_END_RE = re.compile(r"[.!?][\s\n]+")
_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u4e00-\u9fff]+")
_DEFAULT_TOP_WORDS = 200
_DEFAULT_TOP_BIGRAMS = 100
_DEFAULT_NGRAM_DIM = 256


@dataclass(frozen=True)
class _NGramEntry:
    """One n-gram with its observed frequency.

    Stored as a tuple in the artifact so frequency tables are
    deterministically iterable.
    """

    ngram: str
    count: int
    frequency: float


@dataclass(frozen=True)
class FigureStylePrior:
    """Statistical voice-shape artifact for a figure.

    All fields are deterministic functions of the input corpus. The
    runtime injector consumes the artifact read-only:

    * :attr:`top_words` and :attr:`top_bigrams` drive logit bias.
    * :attr:`sentence_length_percentiles` drives length-control
      hints fed to the planner.
    * :attr:`term_list` is the explicit reviewer / corpus-derived
      glossary that should appear at higher rate than baseline.
    """

    figure_id: str
    total_tokens: int
    total_chunks: int
    vocabulary_size: int
    top_words: tuple[_NGramEntry, ...]
    top_bigrams: tuple[_NGramEntry, ...]
    sentence_length_percentiles: tuple[
        tuple[str, float], ...
    ]  # (("p10", x), ("p50", y), ...)
    term_list: tuple[str, ...]
    integrity_hash: str

    def __post_init__(self) -> None:
        if not self.figure_id.strip():
            raise ValueError("FigureStylePrior.figure_id must be non-empty")
        if self.total_tokens <= 0:
            raise ValueError(
                f"FigureStylePrior.total_tokens must be > 0, "
                f"got {self.total_tokens!r}"
            )
        if self.total_chunks <= 0:
            raise ValueError(
                f"FigureStylePrior.total_chunks must be > 0, "
                f"got {self.total_chunks!r}"
            )
        if not self.top_words:
            raise ValueError(
                "FigureStylePrior.top_words must be non-empty; the L1 "
                "contract needs at least one frequency entry."
            )

    def lookup_word_frequency(self, word: str) -> float:
        """Return the frequency of ``word`` in the prior, or 0.0 if absent."""
        target = word.lower()
        for entry in self.top_words:
            if entry.ngram == target:
                return entry.frequency
        return 0.0


def build_figure_style_prior(
    *,
    figure_id: str,
    envelopes: tuple[IngestionEnvelope, ...],
    extra_terms: tuple[str, ...] = (),
    top_words: int = _DEFAULT_TOP_WORDS,
    top_bigrams: int = _DEFAULT_TOP_BIGRAMS,
) -> FigureStylePrior:
    """Build a :class:`FigureStylePrior` from corpus envelopes.

    Statistics are computed on tokens of length >= 3 (per the
    retrieval index discipline) so the resulting prior is not
    dominated by structural particles. ``extra_terms`` lets callers
    inject reviewer-curated glossary that should always appear in
    :attr:`term_list` regardless of corpus frequency.
    """

    if not envelopes:
        raise ValueError(
            "build_figure_style_prior: envelopes tuple must be non-empty"
        )
    if not figure_id.strip():
        raise ValueError(
            "build_figure_style_prior: figure_id must be non-empty"
        )
    word_counts: dict[str, int] = {}
    bigram_counts: dict[str, int] = {}
    sentence_lengths: list[int] = []
    total_chunks = 0
    total_tokens = 0
    for envelope in envelopes:
        for chunk in envelope.successful_chunks:
            total_chunks += 1
            tokens = [match.group(0).lower() for match in _WORD_RE.finditer(chunk.text)]
            tokens = [token for token in tokens if len(token) >= 3]
            total_tokens += len(tokens)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
            for first, second in zip(tokens, tokens[1:]):
                bigram = f"{first} {second}"
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            for sentence in _SENTENCE_END_RE.split(chunk.text):
                stripped = sentence.strip()
                if not stripped:
                    continue
                sentence_lengths.append(len(stripped))
    if total_tokens == 0:
        raise ValueError(
            "build_figure_style_prior: corpus produced no tokens after "
            "filtering; refusing to build an empty style prior."
        )
    top_word_entries = _frequency_top(word_counts, top=top_words, total=total_tokens)
    bigram_total = sum(bigram_counts.values())
    top_bigram_entries = _frequency_top(
        bigram_counts, top=top_bigrams, total=max(bigram_total, 1)
    )
    percentiles = _length_percentiles(sentence_lengths)
    term_list_set: list[str] = []
    seen: set[str] = set()
    for term in extra_terms:
        normalised = term.strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            term_list_set.append(normalised)
    for entry in top_word_entries[:50]:
        if entry.ngram not in seen:
            seen.add(entry.ngram)
            term_list_set.append(entry.ngram)
    integrity_payload = (
        figure_id,
        total_tokens,
        total_chunks,
        len(word_counts),
        tuple((entry.ngram, entry.count) for entry in top_word_entries),
        tuple((entry.ngram, entry.count) for entry in top_bigram_entries),
        percentiles,
        tuple(term_list_set),
    )
    integrity_hash = hashlib.sha256(
        repr(integrity_payload).encode("utf-8")
    ).hexdigest()
    return FigureStylePrior(
        figure_id=figure_id,
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        vocabulary_size=len(word_counts),
        top_words=top_word_entries,
        top_bigrams=top_bigram_entries,
        sentence_length_percentiles=percentiles,
        term_list=tuple(term_list_set),
        integrity_hash=integrity_hash,
    )


def _frequency_top(
    counts: dict[str, int],
    *,
    top: int,
    total: int,
) -> tuple[_NGramEntry, ...]:
    sorted_items = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    chosen = sorted_items[: max(1, top)]
    return tuple(
        _NGramEntry(
            ngram=ngram,
            count=count,
            frequency=count / total if total > 0 else 0.0,
        )
        for ngram, count in chosen
    )


def _length_percentiles(
    sentence_lengths: list[int],
) -> tuple[tuple[str, float], ...]:
    if not sentence_lengths:
        return (
            ("p10", 0.0),
            ("p50", 0.0),
            ("p90", 0.0),
        )
    sorted_lengths = sorted(sentence_lengths)
    return (
        ("p10", float(_percentile(sorted_lengths, 0.10))),
        ("p50", float(_percentile(sorted_lengths, 0.50))),
        ("p90", float(_percentile(sorted_lengths, 0.90))),
    )


def _percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[index]


__all__ = [
    "FigureStylePrior",
    "build_figure_style_prior",
]
