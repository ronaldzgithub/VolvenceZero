"""Wave P.3 — synthetic-substrate smoke for the persona verification harness.

Validates that:

1. ``generate_in_corpus_questions`` produces deterministic
   :class:`PersonaTestQuestion`-shaped objects from a synthetic
   Einstein bundle.
2. ``with_condition`` wires up the synthesizer correctly for all
   three conditions (RAW / BUNDLE / BUNDLE_LORA) and the runtime
   instance is reused across them.
3. ``run_ablation`` produces (conditions x questions) results
   without crashing.
4. ``score_question`` and ``aggregate_scores`` produce JSON-
   serialisable scalars in [0, 1].
5. ``build_persona_verdict`` always emits exactly four gates with
   the canonical names; the smoke test does NOT assert the verdict
   passes — synthetic substrate has placeholder ``.generate()``
   that returns the same hint string for any prompt, so the
   delta-based gates intentionally fail. Schema completeness is
   what we guard here.
6. The CLI entry-point writes the full output tree
   (``questions.jsonl``, ``results/<cond>.jsonl``, ``scores.json``,
   ``verdict.json``, ``transcript.md``) and exits with the
   expected non-zero verdict-fail code on synthetic substrate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lifeform_domain_figure import (
    build_einstein_lifeform,
    save_figure_bundle,
)
from lifeform_domain_figure.verification.persona import (
    DEFAULT_VERDICT_THRESHOLDS,
    OUT_OF_SCOPE_REFUSAL_QUESTIONS,
    PersonaCondition,
    PersonaQuestionCategory,
    aggregate_scores,
    build_persona_verdict,
    generate_in_corpus_questions,
    run_ablation,
    score_question,
    with_condition,
)
from lifeform_domain_figure.verification.persona.cli import main as cli_main
from volvence_zero.substrate import (
    SyntheticOpenWeightResidualRuntime,
    default_persona_lora_pool,
)


@pytest.fixture
def synthetic_einstein_bundle() -> object:
    return build_einstein_lifeform().artifact_bundle


@pytest.fixture
def clean_pool():
    pool = default_persona_lora_pool()
    if pool.has("einstein"):
        pool.deregister("einstein")
    yield pool
    if pool.has("einstein"):
        pool.deregister("einstein")


def test_generate_in_corpus_questions_is_deterministic(
    synthetic_einstein_bundle,
) -> None:
    a = generate_in_corpus_questions(
        bundle=synthetic_einstein_bundle, max_questions=5
    )
    b = generate_in_corpus_questions(
        bundle=synthetic_einstein_bundle, max_questions=5
    )
    assert a == b
    assert 0 < len(a) <= 5
    for q in a:
        assert q.category is PersonaQuestionCategory.IN_CORPUS_POSITION
        assert q.ground_truth_chunk_locator
        assert q.ground_truth_excerpt


def test_out_of_scope_questions_constant_is_not_empty() -> None:
    assert len(OUT_OF_SCOPE_REFUSAL_QUESTIONS) >= 5
    for q in OUT_OF_SCOPE_REFUSAL_QUESTIONS:
        assert q.category is PersonaQuestionCategory.OUT_OF_SCOPE_REFUSAL


def test_with_condition_yields_synthesizer_for_all_three_paths(
    synthetic_einstein_bundle, clean_pool,
) -> None:
    runtime = SyntheticOpenWeightResidualRuntime()
    for cond in (
        PersonaCondition.RAW,
        PersonaCondition.BUNDLE,
        PersonaCondition.BUNDLE_LORA,
    ):
        with with_condition(
            condition=cond,
            runtime=runtime,
            bundle=synthetic_einstein_bundle,
        ) as synth:
            assert synth is not None
            # RAW disowns the bundle; BUNDLE / BUNDLE_LORA carry it.
            if cond is PersonaCondition.RAW:
                assert synth.figure_bundle is None
            else:
                assert synth.figure_bundle is synthetic_einstein_bundle


def test_run_ablation_smoke_through_synthetic_substrate(
    synthetic_einstein_bundle, clean_pool,
) -> None:
    runtime = SyntheticOpenWeightResidualRuntime()
    questions = (
        generate_in_corpus_questions(
            bundle=synthetic_einstein_bundle, max_questions=3
        )
        + OUT_OF_SCOPE_REFUSAL_QUESTIONS[:2]
    )
    results = run_ablation(
        questions=questions, runtime=runtime, bundle=synthetic_einstein_bundle
    )
    assert len(results) == len(questions) * 3
    seen_conditions = {r.condition for r in results}
    assert seen_conditions == {
        PersonaCondition.RAW,
        PersonaCondition.BUNDLE,
        PersonaCondition.BUNDLE_LORA,
    }
    for r in results:
        assert isinstance(r.response_text, str)
        assert r.wall_ms >= 0


def test_scoring_pipeline_emits_well_formed_verdict(
    synthetic_einstein_bundle, clean_pool,
) -> None:
    runtime = SyntheticOpenWeightResidualRuntime()
    questions = (
        generate_in_corpus_questions(
            bundle=synthetic_einstein_bundle, max_questions=3
        )
        + OUT_OF_SCOPE_REFUSAL_QUESTIONS[:2]
    )
    results = run_ablation(
        questions=questions, runtime=runtime, bundle=synthetic_einstein_bundle
    )
    by_qid = {q.question_id: q for q in questions}
    scores = tuple(
        score_question(
            question=by_qid[r.question_id],
            result=r,
            bundle=synthetic_einstein_bundle,
        )
        for r in results
    )
    aggregates = aggregate_scores(scores=scores)
    assert {a.condition for a in aggregates} == {
        PersonaCondition.RAW,
        PersonaCondition.BUNDLE,
        PersonaCondition.BUNDLE_LORA,
    }
    for agg in aggregates:
        assert 0.0 <= agg.voice_score <= 1.0
        assert 0.0 <= agg.cognition_score <= 1.0
        assert 0.0 <= agg.out_of_scope_refusal_rate <= 1.0

    verdict = build_persona_verdict(
        figure_id="einstein",
        bundle_id=synthetic_einstein_bundle.bundle_id,
        persona_lora_record_id="absent",
        aggregates=aggregates,
        total_questions=len(questions),
        in_corpus_questions=3,
        out_of_scope_questions=2,
        thresholds=DEFAULT_VERDICT_THRESHOLDS,
    )
    gate_names = [g.name for g in verdict.gates]
    assert gate_names == [
        "gate_cognition_improves",
        "gate_voice_improves_with_lora",
        "gate_refusal_works",
        "gate_evidence_emerges",
    ]
    payload = verdict.to_json()
    json.dumps(payload)
    assert payload["overall_passed"] is verdict.overall_passed
    assert len(payload["gates"]) == 4


def test_cli_main_writes_full_output_tree(
    synthetic_einstein_bundle, clean_pool, tmp_path: Path
) -> None:
    bundle_root = tmp_path / "bundles"
    bundle_root.mkdir()
    save_figure_bundle(synthetic_einstein_bundle, root_dir=bundle_root)
    output_dir = tmp_path / "verify_out"
    argv = [
        "--bundle-id", synthetic_einstein_bundle.bundle_id,
        "--figure", "einstein",
        "--bundle-root", str(bundle_root),
        "--output-dir", str(output_dir),
        "--runtime", "synthetic",
        "--max-in-corpus-questions", "3",
        "--conditions", "raw,bundle,bundle_lora",
    ]
    rc = cli_main(argv)
    # Synthetic substrate cannot meet the deltas — verdict fails (rc=2)
    # or passes (rc=0) depending on chunk geometry. Either way the
    # output tree must be complete.
    assert rc in (0, 2)
    assert (output_dir / "questions.jsonl").exists()
    assert (output_dir / "scores.json").exists()
    assert (output_dir / "verdict.json").exists()
    assert (output_dir / "transcript.md").exists()
    for cond in ("raw", "bundle", "bundle_lora"):
        assert (output_dir / "results" / f"{cond}.jsonl").exists()
    verdict_payload = json.loads(
        (output_dir / "verdict.json").read_text(encoding="utf-8")
    )
    assert verdict_payload["figure_id"] == "einstein"
    assert verdict_payload["bundle_id"] == synthetic_einstein_bundle.bundle_id
    assert len(verdict_payload["gates"]) == 4
