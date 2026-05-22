"""Wave O.3 — drive the ablation grid (conditions x questions).

For each (condition, question) pair we open the
:func:`with_condition` context, build a default
:class:`ResponseContext` carrying the question prompt, and call
``synth.synthesize(context=..., assembly=_baseline_assembly)``. The
wall time is captured from ``time.monotonic_ns()`` so reviewers can
track latency regressions.

The minimal :class:`ResponseAssemblySnapshot` is the load-bearing
detail for measuring real LoRA effect: when ``assembly is None``,
:meth:`LLMResponseSynthesizer.synthesize` short-circuits to a
templated regime-based renderer that never invokes the LLM forward
path — and a verification ablation built on that path would compare
template strings rather than Qwen output, so the persona LoRA would
not be observable in the gates (debt #40 secondary cause:
verification did not invoke the runtime forward at all).

Output is a tuple of :class:`AblationResult` records, one per pair,
in the (questions x conditions) iteration order — that ordering is
guaranteed deterministic so downstream `transcript.md` rendering
is stable.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from lifeform_domain_figure.persona_runtime_surface import (
    baseline_response_assembly,
    default_response_context,
)
from lifeform_domain_figure.verification.persona.records import (
    AblationResult,
    PersonaCondition,
    PersonaTestQuestion,
)
from lifeform_domain_figure.verification.persona.runtime_conditions import (
    with_condition,
)


_DEFAULT_CONDITION_ORDER: tuple[PersonaCondition, ...] = (
    PersonaCondition.RAW,
    PersonaCondition.BUNDLE,
    PersonaCondition.BUNDLE_LORA,
)


def run_ablation(
    *,
    questions: Sequence[PersonaTestQuestion],
    runtime: Any,
    bundle: Any,
    conditions: Sequence[PersonaCondition] = _DEFAULT_CONDITION_ORDER,
) -> tuple[AblationResult, ...]:
    """Iterate ``conditions`` x ``questions`` and collect responses.

    The same ``runtime`` instance is reused across all conditions
    (see :func:`with_condition` for why). The ``bundle`` is shared
    by reference and is required for ``BUNDLE`` / ``BUNDLE_LORA``
    conditions.
    """

    if not questions:
        return ()
    if not conditions:
        raise ValueError("run_ablation: conditions must be non-empty")

    assembly = baseline_response_assembly()
    results: list[AblationResult] = []
    for condition in conditions:
        with with_condition(
            condition=condition, runtime=runtime, bundle=bundle
        ) as synth:
            for question in questions:
                started = time.monotonic_ns()
                context = default_response_context(question.prompt)
                response = synth.synthesize(
                    context=context, assembly=assembly
                )
                wall_ms = max(
                    0, (time.monotonic_ns() - started) // 1_000_000
                )
                results.append(
                    AblationResult(
                        condition=condition,
                        question_id=question.question_id,
                        response_text=response.text,
                        rationale_tags=tuple(response.rationale_tags),
                        abstract_action=response.abstract_action or "",
                        wall_ms=int(wall_ms),
                    )
                )
    return tuple(results)


__all__ = ["run_ablation"]
