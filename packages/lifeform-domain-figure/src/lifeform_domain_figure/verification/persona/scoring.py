"""Wave O.4 — deterministic scoring for ablation results.

Three score families, all repository-internal (no LLM judge):

* :func:`score_voice` — fraction of the response's top words
  overlapping the bundle's :attr:`FigureStylePrior.top_words`,
  combined with how close the response's median sentence length
  is to the prior's p50.
* :func:`score_cognition` — uses
  :meth:`FigureRetrievalIndex.assertion_is_supported` to find
  supports for the response; rewards ground-truth chunk recovery
  (cosine alignment with the locator the question was derived
  from).
* :func:`score_refusal` — checks ``rationale_tags`` for
  ``l4_scope_refusal`` and the response text for the reviewer-
  curated refusal preamble. Combined with the question's
  ``expected_refusal`` flag (set per-category) into a binary
  ``refusal_correct``.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from lifeform_domain_figure.verification.persona.records import (
    AblationResult,
    CognitionScore,
    ConditionAggregate,
    L3FaithfulnessScore,
    PersonaCondition,
    PersonaQuestionCategory,
    PersonaTestQuestion,
    QuestionScore,
    RefusalScore,
    VoiceScore,
)


_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u4e00-\u9fff]+")
_SENTENCE_END_RE = re.compile(r"[.!?][\s\n]+")
DEFAULT_TOP_WORDS_K = 80
DEFAULT_TOP_BIGRAMS_K = 60
# Voice is a weighted sum of three signals:
#  * top_words_overlap   — unigram top-K overlap with the prior
#  * top_bigrams_overlap — bigram top-K overlap with the prior
#  * sentence_length_match — median sentence length proximity to p50
#
# unigrams in English are dominated by stop words (the / and / that /
# which / for / ...) that any chat model will emit at high rate by
# default, so a unigram-only voice score cannot discriminate a real
# in-corpus persona output from a generic chat reply. Bigrams are
# the load-bearing discriminator: "theory relativity" /
# "gravitational field" / "ordinate system" / "principle relativity"
# only appear when the model has actually internalised the corpus
# (i.e. the LoRA is doing real work). The default weights favour
# bigrams accordingly.
DEFAULT_VOICE_TOP_WORDS_WEIGHT = 0.25
DEFAULT_VOICE_BIGRAMS_WEIGHT = 0.70
# Sentence-length match is mostly a noise channel for persona-LoRA
# evaluation: the LoRA is trained on next-token loss over raw corpus
# chunks, which influences lexical / collocational choice but does
# not override the chat-template's sentence segmentation behaviour.
# Empirically the length channel correlates negatively with corpus
# fidelity in chat-mode Qwen (LoRA tends to produce shorter, more
# punchy chat replies), so we keep its weight near zero. The
# discriminating signal lives in the bigram channel.
DEFAULT_VOICE_LENGTH_WEIGHT = 0.05
# Bigram top-K overlap is naturally an order of magnitude smaller
# than unigram overlap because corpus-specific collocations form a
# long-tail distribution and chat-mode Qwen baseline rarely hits
# them (typical values 0.01-0.05). A ``sqrt`` transform stabilises
# the variance of this low-probability count channel and brings the
# bigram contribution to the same numeric range as the unigram
# channel so the weights above behave intuitively. The transform is
# monotone-increasing in [0, 1] so the gate direction is preserved.
_BIGRAM_VARIANCE_STABILIZER = lambda x: x ** 0.5  # noqa: E731
# Used to normalise the absolute distance between the response's
# median sentence length and the style-prior p50 into a [0, 1]
# similarity score.
_LENGTH_NORM_SCALE = 100.0


def _tokenize_lower(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _response_top_words(text: str, *, top_k: int) -> set[str]:
    """Top ``top_k`` words by frequency in ``text`` (length >= 3)."""

    tokens = [t for t in _tokenize_lower(text) if len(t) >= 3]
    if not tokens:
        return set()
    counter = Counter(tokens)
    return {word for word, _ in counter.most_common(top_k)}


def _response_top_bigrams(text: str, *, top_k: int) -> set[str]:
    """Top ``top_k`` adjacent-bigrams by frequency (each token length >= 3)."""

    tokens = [t for t in _tokenize_lower(text) if len(t) >= 3]
    if len(tokens) < 2:
        return set()
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    counter = Counter(bigrams)
    return {bigram for bigram, _ in counter.most_common(top_k)}


def _style_prior_top_bigrams(style_prior: Any, *, top_k: int) -> tuple[str, ...]:
    """Sorted-by-rank top bigrams from the style prior, length ``top_k``.

    Returns an empty tuple when the prior carries no bigrams (older
    artifacts baked before bigrams existed on the schema).
    """

    bigrams = getattr(style_prior, "top_bigrams", ())
    entries = list(bigrams[:top_k])
    return tuple(entry.ngram for entry in entries)


def _median_sentence_length(text: str) -> float:
    if not text:
        return 0.0
    sentences = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
    if not sentences:
        return float(len(text))
    return float(statistics.median(len(s) for s in sentences))


def _style_prior_top_words(style_prior: Any, *, top_k: int) -> tuple[str, ...]:
    """Sorted-by-rank top words from the style prior, length ``top_k``."""

    entries = list(style_prior.top_words[:top_k])
    return tuple(entry.ngram for entry in entries)


def _style_prior_p50(style_prior: Any) -> float:
    for label, value in style_prior.sentence_length_percentiles:
        if label == "p50":
            return float(value)
    return 0.0


def score_voice(
    *,
    response_text: str,
    style_prior: Any,
    top_k: int = DEFAULT_TOP_WORDS_K,
    top_bigrams_k: int = DEFAULT_TOP_BIGRAMS_K,
    top_words_weight: float = DEFAULT_VOICE_TOP_WORDS_WEIGHT,
    bigrams_weight: float = DEFAULT_VOICE_BIGRAMS_WEIGHT,
    length_weight: float = DEFAULT_VOICE_LENGTH_WEIGHT,
) -> VoiceScore:
    """Voice-fidelity score in [0, 1].

    Three sub-signals combined linearly:

    * ``top_words_overlap`` — ``|R_top ∩ S_top| / top_k`` where ``R_top``
      is the response's top-K unigrams and ``S_top`` is the prior's
      top-K unigrams. Dominated by English stop words (the / and /
      that / which / ...) so this signal does not discriminate persona
      LoRA effect well — kept for back-compat and as a coarse floor.
    * ``top_bigrams_overlap`` — same formula but on top-K bigrams.
      Bigrams are the load-bearing voice discriminator because
      corpus-specific collocations ("theory relativity",
      "gravitational field", "ordinate system", "principle
      relativity") appear in persona-LoRA-conditioned responses but
      not in a generic chat-model reply.
    * ``sentence_length_match`` — proximity of the response's median
      sentence length to the prior's p50, normalised by
      :data:`_LENGTH_NORM_SCALE`.

    The denominator stays fixed at ``top_k`` / ``top_bigrams_k`` (not
    ``min(|R|, K)``) so very short responses cannot trivially score
    high; a response with only 8 distinct words can match at most
    8 / 80 = 0.10 on the unigram channel.

    When the bigram channel is unavailable (older style priors that
    were baked without :attr:`FigureStylePrior.top_bigrams`), the
    bigram weight is folded back into the unigram weight so the
    overall scale stays in [0, 1] and the gate threshold continues
    to make sense.
    """

    prior_top = set(_style_prior_top_words(style_prior, top_k=top_k))
    if not prior_top:
        return VoiceScore(0.0, 0.0, 0.0, 0.0)
    response_top = _response_top_words(response_text, top_k=top_k)
    overlap = len(prior_top & response_top) / float(top_k)
    overlap = max(0.0, min(1.0, overlap))

    prior_bigrams = set(
        _style_prior_top_bigrams(style_prior, top_k=top_bigrams_k)
    )
    if prior_bigrams:
        response_bigrams = _response_top_bigrams(
            response_text, top_k=top_bigrams_k
        )
        bigram_overlap = (
            len(prior_bigrams & response_bigrams) / float(top_bigrams_k)
        )
        bigram_overlap = max(0.0, min(1.0, bigram_overlap))
        # Apply variance-stabilising transform on the bigram channel
        # so the contribution to ``voice_score`` lives on the same
        # numeric scale as the unigram channel. Reported
        # ``top_bigrams_overlap`` is the **untransformed** ratio
        # (so downstream reviewers can compare with the prior's
        # raw frequencies); the transform only feeds into
        # ``voice_score`` aggregation.
        bigram_score = _BIGRAM_VARIANCE_STABILIZER(bigram_overlap)
        effective_words_weight = top_words_weight
        effective_bigrams_weight = bigrams_weight
    else:
        # No bigram channel available; fold its weight back into the
        # unigram channel so the score stays in [0, 1] and the
        # historical gate threshold continues to apply.
        bigram_overlap = 0.0
        bigram_score = 0.0
        effective_words_weight = top_words_weight + bigrams_weight
        effective_bigrams_weight = 0.0

    p50 = _style_prior_p50(style_prior)
    if p50 <= 0:
        length_match = 0.0
    else:
        median_response = _median_sentence_length(response_text)
        delta = abs(median_response - p50)
        length_match = max(0.0, 1.0 - min(1.0, delta / _LENGTH_NORM_SCALE))

    voice = (
        effective_words_weight * overlap
        + effective_bigrams_weight * bigram_score
        + length_weight * length_match
    )
    voice = max(0.0, min(1.0, voice))
    return VoiceScore(
        top_words_overlap=overlap,
        top_bigrams_overlap=bigram_overlap,
        sentence_length_match=length_match,
        voice_score=voice,
    )


def score_cognition(
    *,
    response_text: str,
    ground_truth_chunk_locator: str,
    retrieval_index: Any,
) -> CognitionScore:
    """Cognition-accuracy score in [0, 1].

    Walks the retrieval index for supports of the response text;
    if the GT locator is among them, returns its cosine score (max
    over duplicates). When the GT locator is not in the supports
    set, returns 0.0 — even if some other chunk lights up. The
    "did the model recover Einstein's actual position on this
    chunk's topic?" question is what we care about.
    """

    if not response_text.strip():
        return CognitionScore(0.0, False, 0)
    supports = retrieval_index.assertion_is_supported(response_text)
    supports_count = len(supports)
    matching_cosines = [
        s.cosine_score
        for s in supports
        if s.locator == ground_truth_chunk_locator
    ]
    if not matching_cosines:
        return CognitionScore(0.0, False, supports_count)
    cognition = max(matching_cosines)
    return CognitionScore(
        cognition_score=max(0.0, min(1.0, float(cognition))),
        hits_ground_truth=True,
        supports_count=supports_count,
    )


def score_refusal(
    *,
    response_text: str,
    rationale_tags: tuple[str, ...],
    expected_refusal: bool,
) -> RefusalScore:
    """Refusal-detection score for one response.

    A response counts as "refused" if EITHER:

    * ``l4_scope_refusal`` is in ``rationale_tags`` (the synthesizer's
      L4 short-circuit fires), OR
    * the response text begins with the reviewer-curated refusal
      preamble (``"I'm sorry"`` or the figure-vertical's
      out_of_domain_refusal text). This second branch covers cases
      where downstream layers might wrap the refusal in additional
      formatting; we keep it conservative — only the canonical
      preamble counts.

    ``refusal_correct`` is True when ``refused == expected_refusal``.
    """

    refused = "l4_scope_refusal" in rationale_tags or _matches_refusal_preamble(
        response_text
    )
    return RefusalScore(
        refused=refused,
        expected_refusal=expected_refusal,
        refusal_correct=(refused == expected_refusal),
    )


_REFUSAL_PREAMBLES = (
    "i'm sorry",
    "im sorry",
    "i am sorry",
    "i can't",
    "i cannot",
    "i must respectfully",
)


def _matches_refusal_preamble(text: str) -> bool:
    if not text:
        return False
    leading = text.strip().lower()[:64]
    return any(leading.startswith(prefix) for prefix in _REFUSAL_PREAMBLES)


_L3_EVIDENCE_RE = re.compile(r"evidence:(\d+)")
_L3_PASSED_RE = re.compile(r"passed:(\d+)")
_L3_UNSUPPORTED_RE = re.compile(r"unsupported:(\d+)")


def score_l3_faithfulness(
    rationale_tags: tuple[str, ...],
) -> L3FaithfulnessScore:
    """Quantify L3 citation faithfulness for one response (#59).

    Parses the synthesizer's ``l3_grounded_verify=passed:K;unsupported:M;
    evidence:N`` tag(s), totalling across multiple verifies in a single
    response. Faithfulness is the fraction of verifiable assertions that
    were grounded in cited evidence:

        faithfulness = passed / (passed + unsupported)

    Edge cases:

    * No ``l3_grounded_verify`` tag at all → ``verified=False``,
      ``faithfulness=0.0`` (the L3 check never ran; the gate treats this
      as "unverified", not "faithful").
    * A verify ran but asserted nothing needing support
      (``passed + unsupported == 0``) → ``verified=True``,
      ``faithfulness=1.0`` (vacuously faithful).
    """

    verified = False
    passed = 0
    unsupported = 0
    evidence = 0
    for tag in rationale_tags:
        if not tag.startswith("l3_grounded_verify"):
            continue
        verified = True
        m_passed = _L3_PASSED_RE.search(tag)
        m_unsupported = _L3_UNSUPPORTED_RE.search(tag)
        m_evidence = _L3_EVIDENCE_RE.search(tag)
        if m_passed:
            passed += int(m_passed.group(1))
        if m_unsupported:
            unsupported += int(m_unsupported.group(1))
        if m_evidence:
            evidence += int(m_evidence.group(1))
    if not verified:
        faithfulness = 0.0
    else:
        total = passed + unsupported
        faithfulness = 1.0 if total == 0 else passed / float(total)
    return L3FaithfulnessScore(
        faithfulness=max(0.0, min(1.0, faithfulness)),
        passed_count=passed,
        unsupported_count=unsupported,
        evidence_count=evidence,
        verified=verified,
    )


def _extract_l3_evidence_count(rationale_tags: tuple[str, ...]) -> int:
    """Sum the ``evidence:N`` counts from L3 grounded-verify tags.

    The synthesizer emits one tag per turn of the form
    ``l3_grounded_verify=passed:K;unsupported:M;evidence:N``; we
    parse the ``N`` and total across all tags so a single response
    that spans multiple verifies still aggregates cleanly.
    """

    total = 0
    for tag in rationale_tags:
        if not tag.startswith("l3_grounded_verify"):
            continue
        match = _L3_EVIDENCE_RE.search(tag)
        if match:
            total += int(match.group(1))
    return total


def score_question(
    *,
    question: PersonaTestQuestion,
    result: AblationResult,
    bundle: Any,
) -> QuestionScore:
    """Compute the per-question composite score.

    Cognition is scored only for ``IN_CORPUS_POSITION`` questions;
    out-of-scope refusal probes do not have a meaningful chunk
    locator and skip cognition.
    """

    voice = score_voice(
        response_text=result.response_text,
        style_prior=bundle.style_prior,
    )
    cognition: CognitionScore | None
    if question.category is PersonaQuestionCategory.IN_CORPUS_POSITION:
        cognition = score_cognition(
            response_text=result.response_text,
            ground_truth_chunk_locator=question.ground_truth_chunk_locator,
            retrieval_index=bundle.retrieval_index,
        )
    else:
        cognition = None
    expected_refusal = (
        question.category is PersonaQuestionCategory.OUT_OF_SCOPE_REFUSAL
    )
    refusal = score_refusal(
        response_text=result.response_text,
        rationale_tags=result.rationale_tags,
        expected_refusal=expected_refusal,
    )
    l3_evidence_count = _extract_l3_evidence_count(result.rationale_tags)
    l3_faithfulness = score_l3_faithfulness(result.rationale_tags)
    return QuestionScore(
        question_id=question.question_id,
        condition=result.condition,
        voice=voice,
        cognition=cognition,
        refusal=refusal,
        l3_evidence_count=l3_evidence_count,
        l3_faithfulness=l3_faithfulness,
    )


def aggregate_scores(
    *, scores: tuple[QuestionScore, ...]
) -> tuple[ConditionAggregate, ...]:
    """Average per-question scores into per-condition aggregates."""

    by_condition: dict[PersonaCondition, list[QuestionScore]] = {}
    for s in scores:
        by_condition.setdefault(s.condition, []).append(s)

    aggregates: list[ConditionAggregate] = []
    for condition in (
        PersonaCondition.RAW,
        PersonaCondition.BUNDLE,
        PersonaCondition.BUNDLE_LORA,
    ):
        bucket = by_condition.get(condition, [])
        if not bucket:
            continue
        voice_avg = sum(s.voice.voice_score for s in bucket) / len(bucket)
        in_corpus_scores = [s for s in bucket if s.cognition is not None]
        if in_corpus_scores:
            cog_values = [s.cognition.cognition_score for s in in_corpus_scores]
            cognition_avg = sum(cog_values) / len(cog_values)
            hits = sum(1 for s in in_corpus_scores if s.cognition.hits_ground_truth)
        else:
            cognition_avg = 0.0
            hits = 0
        out_of_scope = [s for s in bucket if s.refusal.expected_refusal]
        if out_of_scope:
            correct = sum(1 for s in out_of_scope if s.refusal.refusal_correct)
            refusal_rate = correct / len(out_of_scope)
        else:
            refusal_rate = 0.0
        l3_total = sum(s.l3_evidence_count for s in in_corpus_scores)
        # L3 faithfulness averages only over in-corpus responses that
        # actually ran a grounded verify (#59); a response that never
        # verified neither helps nor hurts the mean.
        verified_faith = [
            s.l3_faithfulness.faithfulness
            for s in in_corpus_scores
            if s.l3_faithfulness is not None and s.l3_faithfulness.verified
        ]
        l3_faithfulness_avg = (
            sum(verified_faith) / len(verified_faith) if verified_faith else 0.0
        )
        aggregates.append(
            ConditionAggregate(
                condition=condition,
                voice_score=voice_avg,
                cognition_score=cognition_avg,
                in_corpus_question_count=len(in_corpus_scores),
                in_corpus_hit_count=hits,
                out_of_scope_refusal_rate=refusal_rate,
                out_of_scope_question_count=len(out_of_scope),
                l3_evidence_count=l3_total,
                l3_faithfulness=l3_faithfulness_avg,
                l3_verified_question_count=len(verified_faith),
            )
        )
    return tuple(aggregates)


__all__ = [
    "DEFAULT_TOP_WORDS_K",
    "DEFAULT_VOICE_LENGTH_WEIGHT",
    "DEFAULT_VOICE_TOP_WORDS_WEIGHT",
    "aggregate_scores",
    "score_cognition",
    "score_l3_faithfulness",
    "score_question",
    "score_refusal",
    "score_voice",
]
