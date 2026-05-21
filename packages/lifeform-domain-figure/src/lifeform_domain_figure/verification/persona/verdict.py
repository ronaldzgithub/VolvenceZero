"""Wave O.5 — verdict thresholds and gate evaluation.

Four gates, all numeric, no LLM judge:

1. ``gate_cognition_improves`` — the bundle condition's average
   cognition score must beat raw by ``cognition_delta``.
2. ``gate_voice_improves_with_lora`` — bundle_lora's voice score
   must beat bundle's voice score by ``voice_delta``. This is
   the **load-bearing** gate that the LoRA actually changed
   forward behaviour in a measurable way.
3. ``gate_refusal_works`` — bundle's out-of-scope refusal rate
   must be ≥ ``refusal_min`` (default 0.8).
4. ``gate_evidence_emerges`` — bundle (or bundle_lora) must
   surface at least one L3 grounded-verify evidence pointer
   across the in-corpus question batch.

Thresholds are explicit module-level constants so the CLI can
override them via flags and reviewers can audit the exact numbers
the verdict was computed against. They are intentionally
**conservative** — Wave K curated corpus is small, so we accept
any positive movement rather than demanding large effect sizes
(see ``risks`` section in the plan).
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_domain_figure.verification.persona.records import (
    ConditionAggregate,
    GateResult,
    PersonaCondition,
    PersonaVerdict,
)


@dataclass(frozen=True)
class VerdictThresholds:
    """Numeric thresholds for the four verdict gates.

    Set the same constants used in CI runs and in the smoke test
    so reviewers see one source of truth. CLI ``--threshold-*``
    flags override these per-run.
    """

    cognition_delta: float
    voice_delta: float
    refusal_min: float
    evidence_min: int

    def __post_init__(self) -> None:
        if self.cognition_delta < 0:
            raise ValueError("cognition_delta must be >= 0")
        if self.voice_delta < 0:
            raise ValueError("voice_delta must be >= 0")
        if not 0.0 <= self.refusal_min <= 1.0:
            raise ValueError("refusal_min must lie in [0, 1]")
        if self.evidence_min < 0:
            raise ValueError("evidence_min must be >= 0")


DEFAULT_VERDICT_THRESHOLDS = VerdictThresholds(
    cognition_delta=0.05,
    # ``voice_delta`` was 0.02 under the original unigram-dominated
    # voice formula (0.6 × top_words + 0.4 × length), where stop-word
    # overlap alone produced ±0.1 swings and LoRA could move the
    # score by a meaningful 0.02. After the move to the
    # bigram-dominated formula with sqrt-stabilised bigram channel
    # (0.25 × words + 0.70 × sqrt(bigrams) + 0.05 × length, see
    # ``scoring.score_voice`` and ``_BIGRAM_VARIANCE_STABILIZER``),
    # the bigram channel contribution lives on roughly the same
    # numeric scale as the unigram channel and the threshold 0.02
    # continues to mean what it used to: a measurable lift on the
    # discriminating channel beyond noise.
    voice_delta=0.02,
    refusal_min=0.8,
    evidence_min=1,
)


def _aggregate_for(
    aggregates: tuple[ConditionAggregate, ...],
    condition: PersonaCondition,
) -> ConditionAggregate | None:
    for agg in aggregates:
        if agg.condition is condition:
            return agg
    return None


def _gate_cognition_improves(
    *,
    aggregates: tuple[ConditionAggregate, ...],
    thresholds: VerdictThresholds,
) -> GateResult:
    raw = _aggregate_for(aggregates, PersonaCondition.RAW)
    bundle = _aggregate_for(aggregates, PersonaCondition.BUNDLE)
    if raw is None or bundle is None:
        return GateResult(
            name="gate_cognition_improves",
            passed=False,
            observed=0.0,
            threshold=thresholds.cognition_delta,
            rationale="missing raw or bundle aggregate",
        )
    delta = bundle.cognition_score - raw.cognition_score
    passed = delta >= thresholds.cognition_delta
    return GateResult(
        name="gate_cognition_improves",
        passed=passed,
        observed=delta,
        threshold=thresholds.cognition_delta,
        rationale=(
            f"bundle.cognition - raw.cognition = "
            f"{bundle.cognition_score:.4f} - {raw.cognition_score:.4f} = "
            f"{delta:.4f}"
        ),
    )


def _gate_voice_improves_with_lora(
    *,
    aggregates: tuple[ConditionAggregate, ...],
    thresholds: VerdictThresholds,
) -> GateResult:
    bundle = _aggregate_for(aggregates, PersonaCondition.BUNDLE)
    bundle_lora = _aggregate_for(aggregates, PersonaCondition.BUNDLE_LORA)
    if bundle is None or bundle_lora is None:
        return GateResult(
            name="gate_voice_improves_with_lora",
            passed=False,
            observed=0.0,
            threshold=thresholds.voice_delta,
            rationale="missing bundle or bundle_lora aggregate",
        )
    delta = bundle_lora.voice_score - bundle.voice_score
    passed = delta >= thresholds.voice_delta
    return GateResult(
        name="gate_voice_improves_with_lora",
        passed=passed,
        observed=delta,
        threshold=thresholds.voice_delta,
        rationale=(
            f"bundle_lora.voice - bundle.voice = "
            f"{bundle_lora.voice_score:.4f} - {bundle.voice_score:.4f} = "
            f"{delta:.4f}"
        ),
    )


def _gate_refusal_works(
    *,
    aggregates: tuple[ConditionAggregate, ...],
    thresholds: VerdictThresholds,
) -> GateResult:
    bundle = _aggregate_for(aggregates, PersonaCondition.BUNDLE)
    if bundle is None:
        return GateResult(
            name="gate_refusal_works",
            passed=False,
            observed=0.0,
            threshold=thresholds.refusal_min,
            rationale="missing bundle aggregate",
        )
    if bundle.out_of_scope_question_count == 0:
        return GateResult(
            name="gate_refusal_works",
            passed=False,
            observed=0.0,
            threshold=thresholds.refusal_min,
            rationale="no out-of-scope questions in batch",
        )
    rate = bundle.out_of_scope_refusal_rate
    return GateResult(
        name="gate_refusal_works",
        passed=rate >= thresholds.refusal_min,
        observed=rate,
        threshold=thresholds.refusal_min,
        rationale=(
            f"bundle correctly refused "
            f"{int(rate * bundle.out_of_scope_question_count)}/"
            f"{bundle.out_of_scope_question_count} out-of-scope probes"
        ),
    )


def _gate_evidence_emerges(
    *,
    aggregates: tuple[ConditionAggregate, ...],
    thresholds: VerdictThresholds,
) -> GateResult:
    bundle = _aggregate_for(aggregates, PersonaCondition.BUNDLE)
    bundle_lora = _aggregate_for(aggregates, PersonaCondition.BUNDLE_LORA)
    bundle_evidence = bundle.l3_evidence_count if bundle else 0
    bundle_lora_evidence = bundle_lora.l3_evidence_count if bundle_lora else 0
    observed = max(bundle_evidence, bundle_lora_evidence)
    return GateResult(
        name="gate_evidence_emerges",
        passed=observed >= thresholds.evidence_min,
        observed=float(observed),
        threshold=float(thresholds.evidence_min),
        rationale=(
            f"l3_evidence_count: bundle={bundle_evidence}, "
            f"bundle_lora={bundle_lora_evidence}"
        ),
    )


def build_persona_verdict(
    *,
    figure_id: str,
    bundle_id: str,
    persona_lora_record_id: str,
    aggregates: tuple[ConditionAggregate, ...],
    total_questions: int,
    in_corpus_questions: int,
    out_of_scope_questions: int,
    thresholds: VerdictThresholds = DEFAULT_VERDICT_THRESHOLDS,
    notes: tuple[str, ...] = (),
) -> PersonaVerdict:
    """Compose the four gates into a final :class:`PersonaVerdict`."""

    gates = (
        _gate_cognition_improves(
            aggregates=aggregates, thresholds=thresholds
        ),
        _gate_voice_improves_with_lora(
            aggregates=aggregates, thresholds=thresholds
        ),
        _gate_refusal_works(
            aggregates=aggregates, thresholds=thresholds
        ),
        _gate_evidence_emerges(
            aggregates=aggregates, thresholds=thresholds
        ),
    )
    overall = all(g.passed for g in gates)
    return PersonaVerdict(
        figure_id=figure_id,
        bundle_id=bundle_id,
        persona_lora_record_id=persona_lora_record_id,
        overall_passed=overall,
        gates=gates,
        condition_aggregates=aggregates,
        total_questions=total_questions,
        in_corpus_questions=in_corpus_questions,
        out_of_scope_questions=out_of_scope_questions,
        notes=tuple(notes),
    )


__all__ = [
    "DEFAULT_VERDICT_THRESHOLDS",
    "VerdictThresholds",
    "build_persona_verdict",
]
