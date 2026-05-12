"""Typed records for the persona verification harness.

Every artifact the harness produces is a frozen dataclass:

* :class:`PersonaTestQuestion` — one prompt + ground truth.
* :class:`AblationResult` — one synthesizer response under one
  condition.
* :class:`VoiceScore` / :class:`CognitionScore` /
  :class:`RefusalScore` — per-response scoring breakdown.
* :class:`ConditionAggregate` — averaged scores across all
  questions for one ablation condition.
* :class:`GateResult` / :class:`PersonaVerdict` — final pass/fail
  with per-gate detail.

All fields are JSON-serialisable; `to_json()` / `from_json()`
roundtrip is the persistence shape used by the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PersonaQuestionCategory(str, Enum):
    """Closed enum of test question categories."""

    IN_CORPUS_POSITION = "in_corpus_position"
    OUT_OF_SCOPE_REFUSAL = "out_of_scope_refusal"


class PersonaCondition(str, Enum):
    """Closed enum of ablation conditions."""

    RAW = "raw"
    BUNDLE = "bundle"
    BUNDLE_LORA = "bundle_lora"


@dataclass(frozen=True)
class PersonaTestQuestion:
    """One reviewer-vetted or auto-generated test question.

    ``ground_truth_chunk_locator`` is empty for ``OUT_OF_SCOPE_REFUSAL``
    questions (no in-corpus chunk to match against). For
    ``IN_CORPUS_POSITION`` it points at the chunk the question was
    derived from; the cognition scorer asserts the response's
    retrieval-index supports include that locator.
    """

    question_id: str
    prompt: str
    category: PersonaQuestionCategory
    ground_truth_chunk_locator: str = ""
    ground_truth_excerpt: str = ""
    domain_tag: str = ""

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("PersonaTestQuestion.question_id must be non-empty")
        if not self.prompt.strip():
            raise ValueError("PersonaTestQuestion.prompt must be non-empty")
        if (
            self.category is PersonaQuestionCategory.IN_CORPUS_POSITION
            and not self.ground_truth_chunk_locator
        ):
            raise ValueError(
                "PersonaTestQuestion.ground_truth_chunk_locator must be set "
                "for in_corpus_position questions"
            )

    def to_json(self) -> dict:
        return {
            "question_id": self.question_id,
            "prompt": self.prompt,
            "category": self.category.value,
            "ground_truth_chunk_locator": self.ground_truth_chunk_locator,
            "ground_truth_excerpt": self.ground_truth_excerpt,
            "domain_tag": self.domain_tag,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "PersonaTestQuestion":
        return cls(
            question_id=str(payload["question_id"]),
            prompt=str(payload["prompt"]),
            category=PersonaQuestionCategory(payload["category"]),
            ground_truth_chunk_locator=str(
                payload.get("ground_truth_chunk_locator", "")
            ),
            ground_truth_excerpt=str(payload.get("ground_truth_excerpt", "")),
            domain_tag=str(payload.get("domain_tag", "")),
        )


@dataclass(frozen=True)
class AblationResult:
    """One synthesizer response under one condition for one question."""

    condition: PersonaCondition
    question_id: str
    response_text: str
    rationale_tags: tuple[str, ...]
    abstract_action: str
    wall_ms: int

    def to_json(self) -> dict:
        return {
            "condition": self.condition.value,
            "question_id": self.question_id,
            "response_text": self.response_text,
            "rationale_tags": list(self.rationale_tags),
            "abstract_action": self.abstract_action,
            "wall_ms": self.wall_ms,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "AblationResult":
        return cls(
            condition=PersonaCondition(payload["condition"]),
            question_id=str(payload["question_id"]),
            response_text=str(payload["response_text"]),
            rationale_tags=tuple(str(t) for t in payload.get("rationale_tags", [])),
            abstract_action=str(payload.get("abstract_action", "")),
            wall_ms=int(payload.get("wall_ms", 0)),
        )


@dataclass(frozen=True)
class VoiceScore:
    """Voice-fidelity decomposition for one response."""

    top_words_overlap: float  # in [0, 1]
    sentence_length_match: float  # in [0, 1]
    voice_score: float  # weighted combination, in [0, 1]

    def to_json(self) -> dict:
        return {
            "top_words_overlap": self.top_words_overlap,
            "sentence_length_match": self.sentence_length_match,
            "voice_score": self.voice_score,
        }


@dataclass(frozen=True)
class CognitionScore:
    """Cognition-accuracy decomposition for one response.

    ``cognition_score`` is the maximum cosine alignment between the
    response and the ground-truth chunk's retrieval-index entry, or
    0.0 when no support is found. ``hits_ground_truth`` is True
    when the retrieval index returned the GT locator AT ALL within
    the configured top_k.
    """

    cognition_score: float
    hits_ground_truth: bool
    supports_count: int

    def to_json(self) -> dict:
        return {
            "cognition_score": self.cognition_score,
            "hits_ground_truth": self.hits_ground_truth,
            "supports_count": self.supports_count,
        }


@dataclass(frozen=True)
class RefusalScore:
    """Refusal-detection result for one response.

    ``expected_refusal`` is True for OUT_OF_SCOPE_REFUSAL questions
    and False for IN_CORPUS_POSITION questions. ``refusal_correct``
    is True when ``refused == expected_refusal`` — covering both
    "correctly refused" and "correctly answered" cases.
    """

    refused: bool
    expected_refusal: bool
    refusal_correct: bool

    def to_json(self) -> dict:
        return {
            "refused": self.refused,
            "expected_refusal": self.expected_refusal,
            "refusal_correct": self.refusal_correct,
        }


@dataclass(frozen=True)
class QuestionScore:
    """Aggregate of voice / cognition / refusal for one question."""

    question_id: str
    condition: PersonaCondition
    voice: VoiceScore
    cognition: CognitionScore | None  # None for out-of-scope questions
    refusal: RefusalScore
    l3_evidence_count: int

    def to_json(self) -> dict:
        return {
            "question_id": self.question_id,
            "condition": self.condition.value,
            "voice": self.voice.to_json(),
            "cognition": self.cognition.to_json() if self.cognition else None,
            "refusal": self.refusal.to_json(),
            "l3_evidence_count": self.l3_evidence_count,
        }


@dataclass(frozen=True)
class ConditionAggregate:
    """Averaged scores across all questions for one condition."""

    condition: PersonaCondition
    voice_score: float
    cognition_score: float
    in_corpus_question_count: int
    in_corpus_hit_count: int
    out_of_scope_refusal_rate: float
    out_of_scope_question_count: int
    l3_evidence_count: int

    def to_json(self) -> dict:
        return {
            "condition": self.condition.value,
            "voice_score": self.voice_score,
            "cognition_score": self.cognition_score,
            "in_corpus_question_count": self.in_corpus_question_count,
            "in_corpus_hit_count": self.in_corpus_hit_count,
            "out_of_scope_refusal_rate": self.out_of_scope_refusal_rate,
            "out_of_scope_question_count": self.out_of_scope_question_count,
            "l3_evidence_count": self.l3_evidence_count,
        }


@dataclass(frozen=True)
class GateResult:
    """One verdict gate's pass/fail + the numeric evidence."""

    name: str
    passed: bool
    observed: float
    threshold: float
    rationale: str

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PersonaVerdict:
    """Final pass/fail with per-gate detail and per-condition aggregate."""

    figure_id: str
    bundle_id: str
    persona_lora_record_id: str
    overall_passed: bool
    gates: tuple[GateResult, ...]
    condition_aggregates: tuple[ConditionAggregate, ...]
    total_questions: int
    in_corpus_questions: int
    out_of_scope_questions: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict:
        return {
            "figure_id": self.figure_id,
            "bundle_id": self.bundle_id,
            "persona_lora_record_id": self.persona_lora_record_id,
            "overall_passed": self.overall_passed,
            "gates": [g.to_json() for g in self.gates],
            "condition_aggregates": [
                a.to_json() for a in self.condition_aggregates
            ],
            "total_questions": self.total_questions,
            "in_corpus_questions": self.in_corpus_questions,
            "out_of_scope_questions": self.out_of_scope_questions,
            "notes": list(self.notes),
        }


__all__ = [
    "AblationResult",
    "CognitionScore",
    "ConditionAggregate",
    "GateResult",
    "PersonaCondition",
    "PersonaQuestionCategory",
    "PersonaTestQuestion",
    "PersonaVerdict",
    "QuestionScore",
    "RefusalScore",
    "VoiceScore",
]
