"""LLM exam-question generation for the DLaaS eval gate.

Turns operator-supplied source material (topics / corpus excerpts /
signature cases) into scenario exam questions with rubrics and
reference answers, via the same env-configured OpenAI-compatible
endpoint the rubric judge uses (:mod:`dlaas_platform_eval.llm_grader`).

Generation is semantic (the LLM grounds questions in the supplied
material); there is no keyword-matching question bank and no stub
fallback — when the LLM env is not configured the route answers
503 ``llm_not_configured``.

R12 guard: question generation is exam authoring, a platform artifact
path. Generated questions are persisted via ``EvalStore`` only;
nothing here reads or writes kernel owners or learning state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dlaas_platform_contracts import RubricEntry

from dlaas_platform_eval import llm_grader
from dlaas_platform_eval.llm_grader import (
    EvalLLMConfig,
    EvalLLMError,
    QuestionGenerationError,
    parse_strict_json_object,
)
from dlaas_platform_eval.prompts import (
    QUESTION_GEN_SYSTEM_PROMPT,
    QUESTION_GEN_USER_TEMPLATE,
)

_LOG = logging.getLogger(__name__)

_ALLOWED_DIFFICULTIES = ("easy", "medium", "hard")
MAX_QUESTION_COUNT = 20
MIN_RUBRIC_CRITERIA = 2
MAX_RUBRIC_CRITERIA = 3


@dataclass(frozen=True)
class GeneratedQuestion:
    """One LLM-authored exam question, validated and ready to persist."""

    scenario_tag: str
    user_prompt: str
    rubric: tuple[RubricEntry, ...]
    reference_answer: str
    tags: tuple[str, ...] = ()
    difficulty: str = "medium"


@dataclass(frozen=True)
class QuestionSource:
    """Operator-supplied grounding material for question generation."""

    topics: tuple[str, ...] = ()
    corpus_excerpts: tuple[str, ...] = ()
    signature_cases: tuple[Mapping[str, str], ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (
            self.topics or self.corpus_excerpts or self.signature_cases
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "topics": list(self.topics),
            "corpus_excerpts": list(self.corpus_excerpts),
            "signature_cases": [dict(c) for c in self.signature_cases],
        }


def parse_question_source(raw: Mapping[str, Any]) -> QuestionSource:
    """Parse the ``source`` body field. Raises ``ValueError`` on shape errors."""

    topics = tuple(str(t) for t in (raw.get("topics") or ()) if str(t).strip())
    excerpts = tuple(
        str(e) for e in (raw.get("corpus_excerpts") or ()) if str(e).strip()
    )
    raw_cases = raw.get("signature_cases") or ()
    if not isinstance(raw_cases, (list, tuple)):
        raise ValueError("source.signature_cases must be a list")
    cases: list[dict[str, str]] = []
    for case in raw_cases:
        if not isinstance(case, Mapping):
            raise ValueError("each signature_case must be an object")
        cases.append(
            {
                "title": str(case.get("title", "") or ""),
                "summary": str(case.get("summary", "") or ""),
            }
        )
    return QuestionSource(
        topics=topics,
        corpus_excerpts=excerpts,
        signature_cases=tuple(cases),
    )


def generate_exam_questions(
    *,
    config: EvalLLMConfig,
    source: QuestionSource,
    count: int = 5,
    difficulty: str = "medium",
    language: str = "en",
) -> tuple[GeneratedQuestion, ...]:
    """Generate ``count`` scenario questions grounded in ``source``.

    Raises:
        QuestionGenerationError: on LLM transport failure or output
            that violates the strict JSON contract (missing fields,
            rubric outside 2-3 criteria, non-positive scores, ...).
            No silent fallback — callers surface the error.
    """

    if source.is_empty():
        raise QuestionGenerationError(
            "source material is empty: supply topics, corpus_excerpts, "
            "and/or signature_cases"
        )
    count = max(1, min(int(count), MAX_QUESTION_COUNT))
    if difficulty not in _ALLOWED_DIFFICULTIES:
        difficulty = "medium"
    user_prompt = QUESTION_GEN_USER_TEMPLATE.format(
        count=count,
        difficulty=difficulty,
        language=language or "en",
        source_json=json.dumps(
            source.to_json(), ensure_ascii=False, indent=2
        ),
    )
    try:
        # Module-attribute lookup (not a from-import binding) so tests
        # can monkeypatch ``llm_grader.chat_completion_text``.
        content = llm_grader.chat_completion_text(
            config,
            system_prompt=QUESTION_GEN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except QuestionGenerationError:
        raise
    except EvalLLMError as exc:
        raise QuestionGenerationError(str(exc)) from exc
    return _parse_generated_questions(content)


def _parse_generated_questions(
    content: str,
) -> tuple[GeneratedQuestion, ...]:
    try:
        data = parse_strict_json_object(content)
    except json.JSONDecodeError as exc:
        raise QuestionGenerationError(
            f"question-gen LLM output is not valid JSON: {exc}; "
            f"content={content[:300]!r}"
        ) from exc
    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise QuestionGenerationError(
            f"question-gen LLM output missing non-empty 'questions' "
            f"list: {content[:300]!r}"
        )
    questions: list[GeneratedQuestion] = []
    for index, raw in enumerate(raw_questions):
        if not isinstance(raw, Mapping):
            raise QuestionGenerationError(
                f"question[{index}] is not an object: {raw!r}"
            )
        user_prompt = str(raw.get("user_prompt", "") or "").strip()
        if not user_prompt:
            raise QuestionGenerationError(
                f"question[{index}] has an empty 'user_prompt'"
            )
        raw_rubric = raw.get("rubric")
        if not isinstance(raw_rubric, list) or not (
            MIN_RUBRIC_CRITERIA <= len(raw_rubric) <= MAX_RUBRIC_CRITERIA
        ):
            raise QuestionGenerationError(
                f"question[{index}] rubric must have "
                f"{MIN_RUBRIC_CRITERIA}-{MAX_RUBRIC_CRITERIA} criteria, "
                f"got {raw_rubric!r}"
            )
        rubric_entries: list[RubricEntry] = []
        for criterion_raw in raw_rubric:
            if not isinstance(criterion_raw, Mapping):
                raise QuestionGenerationError(
                    f"question[{index}] rubric entry is not an object: "
                    f"{criterion_raw!r}"
                )
            try:
                entry = RubricEntry.from_json(criterion_raw)
            except ValueError as exc:
                raise QuestionGenerationError(
                    f"question[{index}] rubric entry invalid: {exc}"
                ) from exc
            if entry.max_score <= 0 or entry.weight <= 0:
                raise QuestionGenerationError(
                    f"question[{index}] rubric entry "
                    f"{entry.criterion!r} must have positive "
                    f"max_score and weight"
                )
            rubric_entries.append(entry)
        difficulty = str(raw.get("difficulty", "medium") or "medium")
        if difficulty not in _ALLOWED_DIFFICULTIES:
            difficulty = "medium"
        questions.append(
            GeneratedQuestion(
                scenario_tag=str(
                    raw.get("scenario_tag", "generated") or "generated"
                ),
                user_prompt=user_prompt,
                rubric=tuple(rubric_entries),
                reference_answer=str(raw.get("reference_answer", "") or ""),
                tags=tuple(str(t) for t in (raw.get("tags") or ())),
                difficulty=difficulty,
            )
        )
    return tuple(questions)


__all__ = [
    "GeneratedQuestion",
    "MAX_QUESTION_COUNT",
    "QuestionSource",
    "generate_exam_questions",
    "parse_question_source",
]
