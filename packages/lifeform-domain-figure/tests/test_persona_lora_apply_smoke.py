"""Smoke tests for the F6 / P6.3 persona LoRA gate-apply pipeline.

Validates:

* :func:`apply_persona_lora_through_gate` allows a clean proposal,
  registers the artifact in the supplied :class:`PersonaLoRAPool`,
  re-keys the bundle id, and surfaces the previous record id (or
  ``"absent"``).
* Re-applying with a fresh artifact reports the previous record id
  correctly (R15: rollback-observability).
* The gate refuses to run with empty ``rollback_evidence``.
* The gate rejects a figure id mismatch.
* The pool registration round-trips through both ``record_id`` and
  ``figure_id`` and exposes the same adapter layers as the
  artifact.
"""

from __future__ import annotations

import pytest

from volvence_zero.credit.gate import GateDecision
from volvence_zero.evaluation.types import EvaluationScore, EvaluationSnapshot
from volvence_zero.substrate import PersonaLoRAPool

from lifeform_domain_figure import (
    FigureBundleInputs,
    FigureLoRAArtifact,
    SyntheticLoRABakeBackend,
    apply_persona_lora_through_gate,
    build_figure_artifact_bundle,
    build_einstein_profile,
    build_figure_ingestion_envelope,
    build_lora_training_plan,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.envelope_builder import FigureCorpusSourceBundle


def _einstein_envelopes():
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    return build_figure_ingestion_envelope(
        FigureCorpusSourceBundle(
            figure_id="einstein",
            papers=papers,
            letters=letters,
            lectures=lectures,
            notebooks=notebooks,
        ),
        uploader="lifeform-figure-tests:persona-lora-apply",
    ).envelopes


def _einstein_bundle():
    return build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=build_einstein_profile(),
            envelopes=_einstein_envelopes(),
        )
    )


def _einstein_artifact(rank: int = 8) -> FigureLoRAArtifact:
    plan = build_lora_training_plan(
        figure_id="einstein",
        envelopes=_einstein_envelopes(),
        rank=rank,
    )
    return SyntheticLoRABakeBackend().bake(plan)


def _clean_evaluation_snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                "behavior", "contract_integrity", 0.99, 0.95,
                "all contracts honored",
            ),
            EvaluationScore(
                "behavior", "rollback_resilience", 0.99, 0.95,
                "rollback drill clean",
            ),
            EvaluationScore(
                "behavior", "fallback_reliance", 0.10, 0.95,
                "no fallback",
            ),
        ),
        session_scores=(),
        alerts=(),
        description="clean offline-gate snapshot",
    )


def test_apply_persona_lora_through_gate_allows_clean_proposal() -> None:
    bundle = _einstein_bundle()
    artifact = _einstein_artifact()
    pool = PersonaLoRAPool()
    result = apply_persona_lora_through_gate(
        base_bundle=bundle,
        artifact=artifact,
        evaluation_snapshot=_clean_evaluation_snapshot(),
        pool=pool,
        rollback_evidence=f"prev=absent;base={bundle.bundle_id}",
    )
    assert result.applied is True
    assert result.gate.decision is GateDecision.ALLOW
    assert result.gate.block_reasons == ()
    assert result.previous_record_id == "absent"
    assert result.record_id is not None
    assert result.bundle.lora is artifact
    assert result.bundle.bundle_id != bundle.bundle_id


def test_apply_persona_lora_registers_in_pool_under_record_and_figure_id() -> None:
    bundle = _einstein_bundle()
    artifact = _einstein_artifact()
    pool = PersonaLoRAPool()
    result = apply_persona_lora_through_gate(
        base_bundle=bundle,
        artifact=artifact,
        evaluation_snapshot=_clean_evaluation_snapshot(),
        pool=pool,
        rollback_evidence="rb",
    )
    assert pool.has(result.record_id)
    assert pool.has(artifact.figure_id)
    record_via_record_id = pool.lookup(result.record_id)
    record_via_figure_id = pool.lookup(artifact.figure_id)
    assert record_via_record_id is record_via_figure_id
    assert record_via_record_id.adapter_layers == artifact.adapter_layers
    assert record_via_record_id.training_plan_hash == artifact.training_plan_hash


def test_apply_persona_lora_through_gate_surfaces_previous_record_id() -> None:
    bundle = _einstein_bundle()
    pool = PersonaLoRAPool()
    first_artifact = _einstein_artifact(rank=4)
    first_result = apply_persona_lora_through_gate(
        base_bundle=bundle,
        artifact=first_artifact,
        evaluation_snapshot=_clean_evaluation_snapshot(),
        pool=pool,
        rollback_evidence="rb-1",
    )
    assert first_result.applied
    assert first_result.previous_record_id == "absent"
    second_artifact = _einstein_artifact(rank=8)
    second_result = apply_persona_lora_through_gate(
        base_bundle=first_result.bundle,
        artifact=second_artifact,
        evaluation_snapshot=_clean_evaluation_snapshot(),
        pool=pool,
        rollback_evidence=f"rb-2;rollback_to={first_result.record_id}",
    )
    assert second_result.applied
    assert second_result.previous_record_id == first_result.record_id


def test_apply_persona_lora_through_gate_requires_rollback_evidence() -> None:
    bundle = _einstein_bundle()
    artifact = _einstein_artifact()
    with pytest.raises(ValueError, match="rollback_evidence"):
        apply_persona_lora_through_gate(
            base_bundle=bundle,
            artifact=artifact,
            evaluation_snapshot=_clean_evaluation_snapshot(),
            pool=PersonaLoRAPool(),
            rollback_evidence="",
        )


def test_apply_persona_lora_through_gate_rejects_figure_id_mismatch() -> None:
    bundle = _einstein_bundle()
    artifact = _einstein_artifact()
    foreign = FigureLoRAArtifact(
        schema_version=artifact.schema_version,
        figure_id="not-einstein",
        backend_id=artifact.backend_id,
        rank=artifact.rank,
        target_layer_index=artifact.target_layer_index,
        adapter_layers=artifact.adapter_layers,
        training_plan_hash=artifact.training_plan_hash,
        integrity_hash=artifact.integrity_hash,
        parameter_count=artifact.parameter_count,
        description=artifact.description,
    )
    with pytest.raises(ValueError, match="figure_id"):
        apply_persona_lora_through_gate(
            base_bundle=bundle,
            artifact=foreign,
            evaluation_snapshot=_clean_evaluation_snapshot(),
            pool=PersonaLoRAPool(),
            rollback_evidence="rb",
        )


def test_pool_deregister_is_idempotent() -> None:
    pool = PersonaLoRAPool()
    bundle = _einstein_bundle()
    artifact = _einstein_artifact()
    result = apply_persona_lora_through_gate(
        base_bundle=bundle,
        artifact=artifact,
        evaluation_snapshot=_clean_evaluation_snapshot(),
        pool=pool,
        rollback_evidence="rb",
    )
    assert pool.has(result.record_id)
    pool.deregister(result.record_id)
    assert not pool.has(result.record_id)
    assert not pool.has(artifact.figure_id)
    pool.deregister(result.record_id)
