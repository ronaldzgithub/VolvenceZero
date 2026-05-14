"""Rubric grader protocol + deterministic fail-closed default.

Slice 6 ships a deterministic scorer so the eval gate is self-contained
and reproducible. Real LLM-judge backends plug in by satisfying the
:class:`RubricGrader` protocol — they are intentionally out of scope
here because:

1. R12 / OA-1 / EVO-2 require evaluation to be a **readout**: any
   gradient / reward signal flowing back into the kernel from the
   judge is a contract violation. Keeping the grader interface
   pure (input rubric + AI response → score breakdown) makes that
   easy to enforce.
2. The deterministic default never grants licenses on its own; the
   exam run can still pass when an operator explicitly supplies
   per-question scores via the public ``ai_responses`` payload, but
   automated execution against a non-graded session lands below
   the pass threshold by design.

The grader operates on individual ``RubricEntry`` lists and AI
responses; it does not know about exam runs / aggregates / licenses.
Aggregation lives in :mod:`dlaas_platform_eval.routes`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from dlaas_platform_contracts import RubricEntry


_LOG = logging.getLogger(__name__)


# Plug-in surface for an LLM-backed grader (debt #13). Implementations
# receive (rubric, ai_response, reference_answer) and return the same
# GradedSubmission shape; this keeps the runtime contract the same as
# DefaultRubricGrader so the swap is purely behavioural.
LLMGraderCallable = Callable[
    [tuple[RubricEntry, ...], str, str], "GradedSubmission"
]


@dataclass(frozen=True)
class GradedSubmission:
    """Per-question grading outcome.

    ``rubric_breakdown`` is a list of dicts (one per criterion) with
    ``criterion`` / ``score`` / ``max_score`` / ``weight`` keys. The
    aggregator computes the run-level score from these.
    """

    weighted_score: float
    rubric_breakdown: tuple[Mapping[str, Any], ...]


class RubricGrader(Protocol):
    """Plug-in interface for rubric grading."""

    def grade(
        self,
        *,
        rubric: tuple[RubricEntry, ...],
        ai_response: str,
        reference_answer: str,
    ) -> GradedSubmission: ...


class DefaultRubricGrader:
    """Deterministic fail-closed scorer.

    Scoring policy (Slice 6 default):

    * Every criterion gets ``score = max_score * 0.5`` when the
      response is non-empty; ``0.0`` when the response is empty.
    * Aggregate ``weighted_score`` is the weighted average of
      criterion scores normalised to ``[0, 1]``.

    This keeps the eval-gate happy path observable (responses move
    the score off zero) without ever automatically granting a
    license. License grants require either an explicit operator
    score override or a real grader plugged in via
    :class:`RubricGrader`.
    """

    def __init__(
        self,
        *,
        default_factor: float = 0.5,
        llm_grader_callable: LLMGraderCallable | None = None,
    ) -> None:
        """Default rubric grader with optional LLM injection (debt #13).

        ``llm_grader_callable`` (debt #13): when supplied, every
        :meth:`grade` call routes to the injected callable instead of
        the deterministic placeholder. The callable's return value is
        expected to satisfy the :class:`GradedSubmission` shape.
        Tests inject deterministic fakes to lock the wiring contract;
        production wires a real LLM judge once #13 ACTIVE lands.
        """

        if not 0.0 <= default_factor <= 1.0:
            raise ValueError(
                f"default_factor must be in [0,1], got {default_factor!r}"
            )
        self._default_factor = default_factor
        self._llm_grader = llm_grader_callable
        if self._llm_grader is None:
            _LOG.warning(
                "DefaultRubricGrader running in fallback mode "
                "(default_factor=%.2f). LLM judge not configured "
                "(debt #13); all responses score "
                "%.0f%% of max_score. License grants require operator "
                "score override or a real LLMGraderCallable injected "
                "via DefaultRubricGrader(llm_grader_callable=...).",
                self._default_factor,
                self._default_factor * 100,
            )

    def grade(
        self,
        *,
        rubric: tuple[RubricEntry, ...],
        ai_response: str,
        reference_answer: str,
    ) -> GradedSubmission:
        if self._llm_grader is not None:
            # Injected LLM grader takes over end-to-end; we trust the
            # caller's GradedSubmission shape (no defensive coercion
            # so any contract drift fails loudly).
            return self._llm_grader(rubric, ai_response, reference_answer)
        breakdown: list[dict[str, Any]] = []
        if not rubric:
            return GradedSubmission(
                weighted_score=0.0,
                rubric_breakdown=(),
            )
        responded = bool((ai_response or "").strip())
        total_weight = 0.0
        weighted_sum = 0.0
        for entry in rubric:
            score = (
                entry.max_score * self._default_factor if responded else 0.0
            )
            breakdown.append(
                {
                    "criterion": entry.criterion,
                    "score": score,
                    "max_score": entry.max_score,
                    "weight": entry.weight,
                }
            )
            if entry.max_score > 0:
                weighted_sum += entry.weight * (score / entry.max_score)
                total_weight += entry.weight
        weighted_score = (
            weighted_sum / total_weight if total_weight > 0 else 0.0
        )
        return GradedSubmission(
            weighted_score=weighted_score,
            rubric_breakdown=tuple(breakdown),
        )


__all__ = [
    "DefaultRubricGrader",
    "GradedSubmission",
    "LLMGraderCallable",
    "RubricGrader",
]
