"""Wave P.2 — driver CLI for the persona verification harness.

Usage:

    python -m lifeform_domain_figure.verification.persona.cli \\
        --bundle-id einstein-2026Q2-... \\
        --bundle-root data/figure_bundles \\
        --qwen-model-id sshleifer/tiny-gpt2 \\
        --output-dir artifacts/figure_verify/<run-id> \\
        --max-in-corpus-questions 20 \\
        --conditions raw,bundle,bundle_lora

Outputs (under ``--output-dir``):

* ``questions.jsonl``         — generated test questions
* ``results/<condition>.jsonl`` — per-condition ablation results
* ``scores.json``             — per-question + per-condition aggregates
* ``verdict.json``            — final 4-gate verdict
* ``transcript.md``           — human-readable side-by-side transcript

Exit codes:

* ``0`` — verdict pass (all four gates green)
* ``2`` — verdict fail (at least one gate red); files still written
* ``3`` — IO / schema / setup error before harness ran

Runtime selection: ``--runtime synthetic`` uses the in-process
:class:`SyntheticOpenWeightResidualRuntime` (CPU, no HF download —
useful for validating the harness end-to-end without a real LLM).
``--runtime transformers`` (default) loads
:class:`TransformersOpenWeightResidualRuntime` with
``--qwen-model-id`` and runs real generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from lifeform_domain_figure import load_figure_bundle
from lifeform_domain_figure.verification.persona.ablation import run_ablation
from lifeform_domain_figure.verification.persona.out_of_scope_set import (
    OUT_OF_SCOPE_REFUSAL_QUESTIONS,
)
from lifeform_domain_figure.verification.persona.question_generator import (
    generate_in_corpus_questions,
)
from lifeform_domain_figure.verification.persona.records import (
    AblationResult,
    PersonaCondition,
    PersonaTestQuestion,
    PersonaVerdict,
    QuestionScore,
)
from lifeform_domain_figure.verification.persona.scoring import (
    aggregate_scores,
    score_question,
)
from lifeform_domain_figure.verification.persona.verdict import (
    DEFAULT_VERDICT_THRESHOLDS,
    VerdictThresholds,
    build_persona_verdict,
)


EXIT_OK = 0
EXIT_VERDICT_FAIL = 2
EXIT_SETUP_ERROR = 3


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform_domain_figure.verification.persona.cli",
        description=(
            "Drive the figure-vertical persona verification harness "
            "end-to-end."
        ),
    )
    parser.add_argument(
        "--bundle-id",
        required=True,
        help="FigureArtifactBundle id (under --bundle-root) to verify.",
    )
    parser.add_argument(
        "--figure",
        default="einstein",
        help="Figure id used to look up the persona LoRA pool record.",
    )
    parser.add_argument(
        "--bundle-root",
        default="data/figure_bundles",
        help="Persisted figure bundle root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write questions / results / scores / verdict.",
    )
    parser.add_argument(
        "--runtime",
        default="transformers",
        choices=("transformers", "synthetic"),
        help=(
            "Substrate runtime backend. 'transformers' loads a real "
            "HuggingFace model via "
            "TransformersOpenWeightResidualRuntime. 'synthetic' uses "
            "the in-process simulator — useful for harness smoke tests "
            "without an HF download."
        ),
    )
    parser.add_argument(
        "--qwen-model-id",
        default="sshleifer/tiny-gpt2",
        help=(
            "HuggingFace model id (only consumed when "
            "--runtime=transformers). Default tiny-gpt2 for fast "
            "smoke; recommend Qwen/Qwen2.5-1.5B-Instruct for real run."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for the substrate runtime (default cpu).",
    )
    parser.add_argument(
        "--max-in-corpus-questions",
        type=int,
        default=20,
        help="Cap on auto-generated in-corpus questions (default 20).",
    )
    parser.add_argument(
        "--conditions",
        default="raw,bundle,bundle_lora",
        help=(
            "Comma-separated subset of {raw,bundle,bundle_lora} to run. "
            "Default runs all three."
        ),
    )
    parser.add_argument(
        "--cognition-delta",
        type=float,
        default=DEFAULT_VERDICT_THRESHOLDS.cognition_delta,
    )
    parser.add_argument(
        "--voice-delta",
        type=float,
        default=DEFAULT_VERDICT_THRESHOLDS.voice_delta,
    )
    parser.add_argument(
        "--refusal-min",
        type=float,
        default=DEFAULT_VERDICT_THRESHOLDS.refusal_min,
    )
    parser.add_argument(
        "--evidence-min",
        type=int,
        default=DEFAULT_VERDICT_THRESHOLDS.evidence_min,
    )
    parser.add_argument(
        "--questions-cache",
        default=None,
        help=(
            "Optional path to read/write generated questions. When set, "
            "an existing file is loaded verbatim; otherwise the freshly "
            "generated questions are written here for reproducibility."
        ),
    )
    return parser


def _parse_conditions(spec: str) -> tuple[PersonaCondition, ...]:
    items: list[PersonaCondition] = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        try:
            items.append(PersonaCondition(token))
        except ValueError as exc:
            raise SystemExit(
                f"--conditions: unknown value {token!r}; expected raw / "
                f"bundle / bundle_lora"
            ) from exc
    if not items:
        raise SystemExit("--conditions must contain at least one value")
    return tuple(items)


def _build_runtime(*, args: argparse.Namespace) -> Any:
    if args.runtime == "synthetic":
        from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime
        return SyntheticOpenWeightResidualRuntime()
    from volvence_zero.substrate import TransformersOpenWeightResidualRuntime
    return TransformersOpenWeightResidualRuntime(
        model_id=args.qwen_model_id,
        device=args.device,
    )


def _persona_lora_record_id(figure_id: str) -> str:
    """Look up the persona LoRA record id; return ``"absent"`` when none.

    ``BUNDLE_LORA`` condition needs a registered record to exercise
    activation. We don't fail when absent — we surface the absence
    in the verdict notes so reviewers can debug.
    """

    from volvence_zero.substrate import default_persona_lora_pool
    pool = default_persona_lora_pool()
    if not pool.has(figure_id):
        return "absent"
    record = pool.lookup(figure_id)
    return record.record_id


def _ensure_pool_has_bundle_lora(*, bundle: Any) -> str:
    """Auto-register the bundle's baked LoRA into the process pool.

    ``default_persona_lora_pool()`` is process-local. The bake CLI
    registers in its own process and then exits — fresh verify
    processes start with an empty pool and would fall through the
    BUNDLE_LORA path silently.

    This helper reads the bundle's documented ``lora`` slot
    (FigureLoRAArtifact) and registers it; no-op when the bundle
    has no baked LoRA. Returns ``record_id`` or ``"absent"`` —
    reviewers can spot the difference in the verdict notes.

    The wheel-boundary direction is preserved: this lives inside
    ``lifeform_domain_figure`` and only consumes
    ``volvence_zero.substrate`` public surface (PersonaLoRAPool)
    and ``lifeform_domain_figure`` public attrs (bundle.lora).
    """

    artifact = getattr(bundle, "lora", None)
    if artifact is None:
        return "absent"
    from volvence_zero.substrate import default_persona_lora_pool
    pool = default_persona_lora_pool()
    figure_id = artifact.figure_id
    bundle_id = bundle.bundle_id
    if pool.has(figure_id):
        existing = pool.lookup(figure_id)
        if existing.source_bundle_id == bundle_id:
            return existing.record_id
        # Different bundle source — replace so the verify run reflects
        # the bundle the CLI was actually pointed at.
        pool.deregister(figure_id)
    return pool.register(
        figure_id=figure_id,
        source_bundle_id=bundle_id,
        backend_id=artifact.backend_id,
        training_plan_hash=artifact.training_plan_hash,
        adapter_layers=artifact.adapter_layers,
        parameter_count=artifact.parameter_count,
        description=artifact.description,
    )


def _load_or_generate_questions(
    *,
    bundle: Any,
    args: argparse.Namespace,
) -> tuple[PersonaTestQuestion, ...]:
    cache_path: Path | None = (
        Path(args.questions_cache) if args.questions_cache else None
    )
    if cache_path is not None and cache_path.exists():
        rows = [
            json.loads(line)
            for line in cache_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return tuple(PersonaTestQuestion.from_json(row) for row in rows)

    in_corpus = generate_in_corpus_questions(
        bundle=bundle,
        max_questions=args.max_in_corpus_questions,
    )
    questions = in_corpus + OUT_OF_SCOPE_REFUSAL_QUESTIONS
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            "\n".join(json.dumps(q.to_json()) for q in questions) + "\n",
            encoding="utf-8",
        )
    return questions


def _write_questions(
    *, questions: Sequence[PersonaTestQuestion], output_dir: Path
) -> None:
    target = output_dir / "questions.jsonl"
    target.write_text(
        "\n".join(json.dumps(q.to_json()) for q in questions) + "\n",
        encoding="utf-8",
    )


def _write_results_per_condition(
    *,
    results: Sequence[AblationResult],
    output_dir: Path,
) -> None:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    by_condition: dict[PersonaCondition, list[AblationResult]] = {}
    for r in results:
        by_condition.setdefault(r.condition, []).append(r)
    for condition, bucket in by_condition.items():
        target = results_dir / f"{condition.value}.jsonl"
        target.write_text(
            "\n".join(json.dumps(r.to_json()) for r in bucket) + "\n",
            encoding="utf-8",
        )


def _write_scores(
    *,
    question_scores: Sequence[QuestionScore],
    aggregates: Sequence[Any],
    output_dir: Path,
) -> None:
    payload = {
        "question_scores": [s.to_json() for s in question_scores],
        "condition_aggregates": [a.to_json() for a in aggregates],
    }
    (output_dir / "scores.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _write_verdict(*, verdict: PersonaVerdict, output_dir: Path) -> None:
    (output_dir / "verdict.json").write_text(
        json.dumps(verdict.to_json(), indent=2) + "\n", encoding="utf-8"
    )


def _write_transcript(
    *,
    questions: Sequence[PersonaTestQuestion],
    results: Sequence[AblationResult],
    verdict: PersonaVerdict,
    output_dir: Path,
) -> None:
    by_question: dict[
        str, dict[PersonaCondition, AblationResult]
    ] = {}
    for r in results:
        by_question.setdefault(r.question_id, {})[r.condition] = r

    lines: list[str] = []
    lines.append(f"# Persona verification transcript — {verdict.figure_id}\n")
    lines.append(
        f"- bundle: `{verdict.bundle_id}`\n"
        f"- persona_lora_record_id: `{verdict.persona_lora_record_id}`\n"
        f"- overall: **{'PASS' if verdict.overall_passed else 'FAIL'}**\n"
        f"- total_questions: {verdict.total_questions} "
        f"(in_corpus={verdict.in_corpus_questions}, "
        f"out_of_scope={verdict.out_of_scope_questions})\n"
    )
    lines.append("\n## Gates\n")
    for gate in verdict.gates:
        mark = "PASS" if gate.passed else "FAIL"
        lines.append(
            f"- **{mark}** `{gate.name}` "
            f"observed={gate.observed:.4f} "
            f"threshold={gate.threshold:.4f} — {gate.rationale}\n"
        )
    lines.append("\n## Per-condition aggregate\n")
    for agg in verdict.condition_aggregates:
        lines.append(
            f"- `{agg.condition.value}`: voice={agg.voice_score:.3f} "
            f"cognition={agg.cognition_score:.3f} "
            f"refusal={agg.out_of_scope_refusal_rate:.2f} "
            f"l3_evidence={agg.l3_evidence_count}\n"
        )
    lines.append("\n## Side-by-side responses\n")
    for q in questions:
        lines.append(f"\n### {q.question_id} — {q.category.value}\n")
        lines.append(f"> Prompt: {q.prompt}\n")
        if q.ground_truth_excerpt:
            lines.append(
                f"> Ground truth (`{q.ground_truth_chunk_locator}`): "
                f"{q.ground_truth_excerpt[:200]}...\n"
            )
        for condition in (
            PersonaCondition.RAW,
            PersonaCondition.BUNDLE,
            PersonaCondition.BUNDLE_LORA,
        ):
            r = by_question.get(q.question_id, {}).get(condition)
            if r is None:
                continue
            lines.append(f"\n**{condition.value}** (wall_ms={r.wall_ms})\n")
            lines.append("```\n")
            lines.append(r.response_text.strip() + "\n")
            lines.append("```\n")
            if r.rationale_tags:
                lines.append(
                    "_tags_: " + ", ".join(r.rationale_tags) + "\n"
                )
    (output_dir / "transcript.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        bundle = load_figure_bundle(
            root_dir=args.bundle_root,
            bundle_id=args.bundle_id,
            figure_id=args.figure,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"persona-verify: failed to load bundle: {exc}", file=sys.stderr)
        return EXIT_SETUP_ERROR

    try:
        runtime = _build_runtime(args=args)
    except (ImportError, RuntimeError) as exc:
        print(f"persona-verify: failed to build runtime: {exc}", file=sys.stderr)
        return EXIT_SETUP_ERROR

    record_id_at_load = _ensure_pool_has_bundle_lora(bundle=bundle)

    questions = _load_or_generate_questions(bundle=bundle, args=args)
    if not questions:
        print(
            "persona-verify: no questions generated — corpus may be too "
            "thin for in-corpus question extraction",
            file=sys.stderr,
        )
        return EXIT_SETUP_ERROR
    _write_questions(questions=questions, output_dir=output_dir)

    conditions = _parse_conditions(args.conditions)
    results = run_ablation(
        questions=questions,
        runtime=runtime,
        bundle=bundle,
        conditions=conditions,
    )
    _write_results_per_condition(results=results, output_dir=output_dir)

    by_question_id = {q.question_id: q for q in questions}
    question_scores = tuple(
        score_question(
            question=by_question_id[r.question_id],
            result=r,
            bundle=bundle,
        )
        for r in results
    )
    aggregates = aggregate_scores(scores=question_scores)
    _write_scores(
        question_scores=question_scores,
        aggregates=aggregates,
        output_dir=output_dir,
    )

    record_id = _persona_lora_record_id(args.figure)
    notes: tuple[str, ...] = ()
    if record_id == "absent" and PersonaCondition.BUNDLE_LORA in conditions:
        notes = (
            "persona LoRA pool record absent at verify time AND bundle "
            "carries no baked LoRA artifact; BUNDLE_LORA condition fell "
            "through to BUNDLE behaviour. Run "
            "scripts/figure_bake_einstein_persona_lora.sh first.",
        )
    elif (
        record_id_at_load != "absent"
        and record_id != "absent"
        and PersonaCondition.BUNDLE_LORA in conditions
    ):
        notes = (
            f"persona LoRA auto-registered from bundle.lora "
            f"(record_id={record_id_at_load}).",
        )

    thresholds = VerdictThresholds(
        cognition_delta=args.cognition_delta,
        voice_delta=args.voice_delta,
        refusal_min=args.refusal_min,
        evidence_min=args.evidence_min,
    )

    in_corpus_count = sum(
        1 for q in questions if q.category.value == "in_corpus_position"
    )
    out_of_scope_count = sum(
        1 for q in questions if q.category.value == "out_of_scope_refusal"
    )
    verdict = build_persona_verdict(
        figure_id=args.figure,
        bundle_id=bundle.bundle_id,
        persona_lora_record_id=record_id,
        aggregates=aggregates,
        total_questions=len(questions),
        in_corpus_questions=in_corpus_count,
        out_of_scope_questions=out_of_scope_count,
        thresholds=thresholds,
        notes=notes,
    )
    _write_verdict(verdict=verdict, output_dir=output_dir)
    _write_transcript(
        questions=questions,
        results=results,
        verdict=verdict,
        output_dir=output_dir,
    )

    print(json.dumps(verdict.to_json(), indent=2))
    return EXIT_OK if verdict.overall_passed else EXIT_VERDICT_FAIL


# ``asdict`` is unused here but exposed for downstream callers that want
# a quick dict view of any of the dataclasses without re-implementing.
__all__ = ["main", "asdict"]


if __name__ == "__main__":
    raise SystemExit(main())
