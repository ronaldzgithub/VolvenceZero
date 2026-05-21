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

from volvence_zero.agent.response import ResponseContext
from volvence_zero.application.types import (
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)

from lifeform_domain_figure.verification.persona.records import (
    AblationResult,
    PersonaCondition,
    PersonaTestQuestion,
)
from lifeform_domain_figure.verification.persona.runtime_conditions import (
    with_condition,
)


def _baseline_assembly() -> ResponseAssemblySnapshot:
    """Minimal :class:`ResponseAssemblySnapshot` that routes through the LLM.

    All three ablation conditions share the **same** assembly so the
    only thing that differs across rows is the LoRA activation and the
    figure-bundle prompt enrichment — never the prompt assembly /
    constraint plan. That makes the voice / cognition delta strictly
    attributable to LoRA (BUNDLE_LORA vs BUNDLE) and to the bundle
    style prior / scope policy (BUNDLE vs RAW).

    Field choices:

    * ``response_mode=SUPPORT`` and ``answer_depth_limit="medium"``
      → no premature truncation, no clarify-mode refusal.
    * ``clarification_required=False`` /
      ``refer_out_required=False`` → no synthesizer-side
      short-circuit.
    * ``citation_mode="optional"`` → no required-citation phrasing.
    * Empty tuples for briefs / playbook / disclaimers → the prompt
      reads "answer the question" without injected boilerplate.
    * ``risk_band=LOW`` → no safety-side rephrase.

    All optional fields use the dataclass defaults
    (``ordering_driver="playbook-only"``, ``speech_plan=None``,
    etc.) so the renderer's ``_decoding_profile_for_assembly``
    falls into the deterministic ``"structure-first"`` branch.
    """

    return ResponseAssemblySnapshot(
        regime_id="default",
        regime_name="default",
        abstract_action="answer",
        response_mode=ResponseMode.SUPPORT,
        answer_depth_limit="medium",
        citation_mode="optional",
        clarification_required=False,
        refer_out_required=False,
        ordering_plan=(),
        knowledge_briefs=(),
        case_briefs=(),
        playbook_ordering=(),
        required_disclaimers=(),
        required_disclaimer_phrases=(),
        control_code=(),
        control_scale=0.0,
        max_questions=0,
        prompt_residue_summary="",
        prompt_residue_ratio=0.0,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="persona-verification minimal assembly",
    )


_DEFAULT_CONDITION_ORDER: tuple[PersonaCondition, ...] = (
    PersonaCondition.RAW,
    PersonaCondition.BUNDLE,
    PersonaCondition.BUNDLE_LORA,
)


def _default_response_context(prompt: str) -> ResponseContext:
    """Return a minimal :class:`ResponseContext` carrying the prompt.

    All non-prompt fields are set to defaults that disable the
    runtime's regime / reflection / repair branches so the
    measured response is the synthesizer's "answer this question"
    path with no controller bias.
    """

    return ResponseContext(
        regime_id="default",
        regime_name="default",
        regime_switched=False,
        abstract_action="answer",
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="continue",
        user_input=prompt,
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

    assembly = _baseline_assembly()
    results: list[AblationResult] = []
    for condition in conditions:
        with with_condition(
            condition=condition, runtime=runtime, bundle=bundle
        ) as synth:
            for question in questions:
                started = time.monotonic_ns()
                context = _default_response_context(question.prompt)
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
