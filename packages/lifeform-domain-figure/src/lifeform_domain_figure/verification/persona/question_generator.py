"""Wave O.1 — auto-generate test questions from a curated bundle's corpus.

Given a :class:`FigureArtifactBundle` whose ``retrieval_index`` is
populated with real corpus chunks, derive ~N test questions whose
ground truth is the chunk text.

Design choice (deterministic over LLM-generated):

The plan originally suggested using a Qwen runtime as the question
generator. We deliberately go deterministic instead because:

1. **Reproducibility (R15)**: a deterministic generator produces
   byte-identical questions across runs; an LLM generator drifts
   under temperature / sampling.
2. **CI hermetic**: the smoke test does not need a real Qwen
   download to validate the harness.
3. **Question quality matters less than ground-truth quality**:
   the bundle's ``retrieval_index.assertion_is_supported`` is the
   scorer of cognition fidelity; the question itself only needs to
   be on-topic enough to elicit a relevant response. A simple
   "What is Einstein's perspective on {topic_phrase}?" template
   does that as well as a clever LLM-written question.
"""

from __future__ import annotations

import re
from typing import Any

from lifeform_domain_figure.verification.persona.records import (
    PersonaQuestionCategory,
    PersonaTestQuestion,
)


# Stop-words borrowed from retrieval_index._STOPWORDS to keep topic
# extraction consistent with the index's tokenisation. We don't
# import the private constant directly to avoid coupling; instead we
# include a small subset that matters for topic extraction.
_TOPIC_STOPWORDS = frozenset(
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
        "nine", "ten", "between", "across", "without", "within",
        "about", "after", "before", "again", "always", "every",
        "while", "though", "however", "therefore",
    }
)
_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u4e00-\u9fff]+")
_DEFAULT_TOPIC_WORD_COUNT = 6
_DEFAULT_QUESTION_TEMPLATE = (
    "Speaking from your own primary writings, what is your perspective on "
    "the relationship between {topic}?"
)


def _extract_topic_phrase(
    text: str, *, word_count: int = _DEFAULT_TOPIC_WORD_COUNT
) -> str:
    """Pull the first ``word_count`` substantive content words from text.

    Skips stop-words and very short tokens so the resulting phrase
    carries discriminating signal. Returns a lowercased, comma-joined
    phrase suitable for templating into the question prompt.
    """

    tokens = []
    for token in _WORD_RE.findall(text):
        lowered = token.lower()
        if len(lowered) <= 3:
            continue
        if lowered in _TOPIC_STOPWORDS:
            continue
        tokens.append(lowered)
        if len(tokens) >= word_count:
            break
    return ", ".join(tokens)


def generate_in_corpus_questions(
    *,
    bundle: Any,
    figure_display_name: str = "Einstein",
    max_questions: int = 20,
    question_template: str = _DEFAULT_QUESTION_TEMPLATE,
    topic_word_count: int = _DEFAULT_TOPIC_WORD_COUNT,
) -> tuple[PersonaTestQuestion, ...]:
    """Walk the bundle's retrieval-index chunks and produce in-corpus questions.

    Iteration order is sorted by ``chunk_id`` for determinism (R15).
    The first ``max_questions`` chunks whose topic-phrase extraction
    yields ≥ 3 content words are kept; chunks below the threshold
    are skipped (they likely have too little substance to test).
    """

    del figure_display_name  # reserved for future template variants
    if not hasattr(bundle, "retrieval_index"):
        raise TypeError(
            "generate_in_corpus_questions: bundle has no retrieval_index"
        )
    index = bundle.retrieval_index
    questions: list[PersonaTestQuestion] = []
    sorted_chunks = sorted(
        index.chunk_records, key=lambda r: r.chunk_id
    )
    for chunk_index, record in enumerate(sorted_chunks):
        topic = _extract_topic_phrase(record.text, word_count=topic_word_count)
        if topic.count(",") + 1 < 3:
            # Need at least 3 content words to form a discriminating
            # topic phrase — skip thin chunks.
            continue
        prompt = question_template.format(topic=topic)
        excerpt = record.text[:240].rstrip()
        question = PersonaTestQuestion(
            question_id=f"in-corpus:{chunk_index:03d}:{record.chunk_id[-12:]}",
            prompt=prompt,
            category=PersonaQuestionCategory.IN_CORPUS_POSITION,
            ground_truth_chunk_locator=record.locator,
            ground_truth_excerpt=excerpt,
            domain_tag=_locator_kind_tag(record.locator),
        )
        questions.append(question)
        if len(questions) >= max_questions:
            break
    return tuple(questions)


def _locator_kind_tag(locator: str) -> str:
    """Pull the leading kind from a citation locator (paper/letter/...)."""

    if not locator:
        return ""
    head = locator.split(":", 1)[0]
    return head if head in {"paper", "letter", "lecture", "notebook"} else ""


__all__ = [
    "generate_in_corpus_questions",
]
