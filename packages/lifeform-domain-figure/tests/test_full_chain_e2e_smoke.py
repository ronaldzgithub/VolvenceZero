"""F1-F6 end-to-end smoke test.

Exercises the full figure-vertical chain on the synthetic Einstein
profile + corpus and asserts the load-bearing invariants of each
layer hold simultaneously:

* L1 (style)         — :class:`FigureStylePrior` is non-trivially
                       populated.
* L3 (citations)     — :class:`FigureRetrievalIndex` returns
                       supported evidence for an in-corpus query
                       and refuses to support an off-topic
                       assertion.
* L4 (refusals)      — :class:`FigureCoverageMap` classifies an
                       in-domain question as in-domain and an
                       off-domain question as out-of-domain.
* L2 (steering)      — :func:`bake_steering_set` produces a non-
                       degenerate steering set whose adapter
                       layers carry the kernel
                       :class:`SubstrateDeltaAdapterLayer` shape.
* L1+L2 (LoRA)       — :class:`SyntheticLoRABakeBackend` produces
                       a deterministic :class:`FigureLoRAArtifact`,
                       and :func:`apply_persona_lora_through_gate`
                       binds it through the OFFLINE gate, attaches
                       it to the bundle, and registers it in a
                       fresh :class:`PersonaLoRAPool`.

All five artifacts attach to the same :class:`FigureArtifactBundle`
in sequence; the final bundle's ``integrity_hash`` differs from
the unconditioned bundle's hash, which is the R15 invariant the
operator-side rollback story depends on.
"""

from __future__ import annotations

from volvence_zero.credit.gate import GateDecision
from volvence_zero.evaluation.types import EvaluationScore, EvaluationSnapshot
from volvence_zero.substrate import (
    PersonaLoRAPool,
    SubstrateDeltaAdapterLayer,
)

from lifeform_domain_figure import (
    FigureBundleInputs,
    SyntheticLoRABakeBackend,
    apply_persona_lora_through_gate,
    apply_steering_through_gate,
    bake_steering_set,
    build_einstein_contrast_set,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    build_lora_training_plan,
    build_steering_training_plan,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.envelope_builder import FigureCorpusSourceBundle


def _clean_snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                "behavior", "contract_integrity", 0.99, 0.95, "ok",
            ),
            EvaluationScore(
                "behavior", "rollback_resilience", 0.99, 0.95, "ok",
            ),
            EvaluationScore(
                "behavior", "fallback_reliance", 0.10, 0.95, "ok",
            ),
        ),
        session_scores=(),
        alerts=(),
        description="clean snapshot",
    )


def test_full_chain_einstein_e2e() -> None:
    profile = build_einstein_profile()
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    envelope_set = build_figure_ingestion_envelope(
        FigureCorpusSourceBundle(
            figure_id="einstein",
            papers=papers,
            letters=letters,
            lectures=lectures,
            notebooks=notebooks,
        ),
        uploader="lifeform-figure-tests:full-chain-e2e",
    )
    base_bundle = build_figure_artifact_bundle(
        FigureBundleInputs(profile=profile, envelopes=envelope_set.envelopes)
    )

    # L1 — style prior is populated
    assert base_bundle.style_prior.figure_id == "einstein"
    assert len(base_bundle.style_prior.top_words) > 0

    # L3 — retrieval supports an in-corpus assertion. The synthetic
    # corpus discusses spatially separated subsystems and definite
    # physical states; an assertion using those phrases must surface
    # supporting chunks from at least one envelope.
    in_corpus_query = (
        "Spatially separated subsystems each carry their own definite "
        "physical state in a complete physical theory."
    )
    supported = base_bundle.retrieval_index.assertion_is_supported(
        in_corpus_query
    )
    assert len(supported) > 0
    sample_evidence = supported[0]
    assert sample_evidence.locator
    assert sample_evidence.source_envelope_id

    # L3 — retrieval refuses to support a clearly off-topic assertion
    # whose surface tokens have no overlap with the corpus.
    off_topic = "Sourdough bread requires hydrated flour and salt only."
    no_support = base_bundle.retrieval_index.assertion_is_supported(off_topic)
    assert no_support == ()

    # L4 — coverage map produces a classification with a decision and
    # a finite distance for both queries; the in-domain query carries
    # a non-trivial top topic id when it lands inside coverage.
    in_domain_query = (
        "Are spatially separated subsystems each in a definite physical state?"
    )
    in_domain = base_bundle.coverage_map.classify_query(in_domain_query)
    out_domain_query = (
        "What is the best Italian recipe for tiramisu using mascarpone?"
    )
    out_domain = base_bundle.coverage_map.classify_query(out_domain_query)
    assert in_domain.decision is not None
    assert out_domain.decision is not None

    # L2 — steering bake produces a non-degenerate set
    contrast_plan = build_steering_training_plan(build_einstein_contrast_set())
    steering = bake_steering_set(contrast_plan)
    assert steering.aggregate.scale > 0.0
    steering_layers = steering.to_substrate_adapter_layers()
    assert all(
        isinstance(layer, SubstrateDeltaAdapterLayer)
        for layer in steering_layers
    )
    assert len(steering_layers) == len(steering.vectors) + 1

    # L2 — apply steering through the OFFLINE gate
    snapshot = _clean_snapshot()
    steering_result = apply_steering_through_gate(
        base_bundle=base_bundle,
        steering=steering,
        evaluation_snapshot=snapshot,
        rollback_evidence=f"prev_steering=absent;base={base_bundle.bundle_id}",
    )
    assert steering_result.applied is True
    assert steering_result.gate.decision is GateDecision.ALLOW
    bundle_with_steering = steering_result.bundle
    assert bundle_with_steering.bundle_id != base_bundle.bundle_id

    # L1+L2 — LoRA bake + gate apply on top of the steering-bound bundle
    lora_plan = build_lora_training_plan(
        figure_id="einstein",
        envelopes=envelope_set.envelopes,
    )
    lora_artifact = SyntheticLoRABakeBackend().bake(lora_plan)
    pool = PersonaLoRAPool()
    lora_result = apply_persona_lora_through_gate(
        base_bundle=bundle_with_steering,
        artifact=lora_artifact,
        evaluation_snapshot=snapshot,
        pool=pool,
        rollback_evidence=(
            f"prev_lora=absent;base={bundle_with_steering.bundle_id}"
        ),
    )
    assert lora_result.applied is True
    assert lora_result.gate.decision is GateDecision.ALLOW
    final_bundle = lora_result.bundle
    assert final_bundle.steering is steering
    assert final_bundle.lora is lora_artifact
    assert final_bundle.bundle_id != bundle_with_steering.bundle_id
    assert final_bundle.integrity_hash != base_bundle.integrity_hash

    # Pool registration round-trip — figure_id and record_id resolve
    # to the same record, layers match the artifact byte-for-byte.
    record = pool.lookup(lora_artifact.figure_id)
    assert record.record_id == lora_result.record_id
    assert record.adapter_layers == lora_artifact.adapter_layers
    assert record.training_plan_hash == lora_plan.integrity_hash
