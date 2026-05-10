"""Smoke tests for the F6 / P6.1 LoRA training data + proposal schema.

Validates:

* :func:`build_lora_training_plan` mixes figure rows with the
  built-in replay buffer at the expected proportions and produces
  a deterministic integrity hash.
* The plan rejects degenerate inputs (no envelopes, all-failed
  chunks, duplicate replay ids, invalid hyper-parameters).
* :func:`build_persona_lora_proposal` builds a
  :class:`PersonaLoRAProposal` whose underlying
  :class:`ModificationProposal` has the expected target /
  desired_gate / value-hash invariants.
* :class:`PersonaLoRAProposal` rejects malformed proposals
  (wrong gate, missing rollback evidence, mismatched figure id).
"""

from __future__ import annotations

import pytest

from volvence_zero.credit.gate import ModificationGate

from lifeform_domain_figure import (
    LoRATrainingExample,
    LoRATrainingPlan,
    PersonaLoRAProposal,
    build_lora_training_plan,
    build_persona_lora_proposal,
    build_figure_ingestion_envelope,
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
        uploader="lifeform-figure-tests:lora-data-prep",
    ).envelopes


def test_build_lora_training_plan_mixes_figure_and_replay() -> None:
    envelopes = _einstein_envelopes()
    plan = build_lora_training_plan(
        figure_id="einstein",
        envelopes=envelopes,
    )
    assert isinstance(plan, LoRATrainingPlan)
    assert plan.figure_id == "einstein"
    assert plan.figure_example_count > 0
    assert plan.replay_example_count >= 6
    assert plan.total_examples == (
        plan.figure_example_count + plan.replay_example_count
    )
    figure_rows = [
        ex for ex in plan.examples if ex.source_kind == "figure"
    ]
    replay_rows = [
        ex for ex in plan.examples if ex.source_kind == "replay"
    ]
    assert len(figure_rows) == plan.figure_example_count
    assert len(replay_rows) == plan.replay_example_count
    assert all(row.weight == 0.5 for row in replay_rows)


def test_build_lora_training_plan_is_deterministic() -> None:
    envelopes = _einstein_envelopes()
    plan_a = build_lora_training_plan(figure_id="einstein", envelopes=envelopes)
    plan_b = build_lora_training_plan(figure_id="einstein", envelopes=envelopes)
    assert plan_a.integrity_hash == plan_b.integrity_hash
    assert plan_a.examples == plan_b.examples


def test_build_lora_training_plan_accepts_extra_replay() -> None:
    envelopes = _einstein_envelopes()
    extra = (
        ("replay:custom:gravity", "Falling apples accelerate at g near Earth."),
    )
    plan = build_lora_training_plan(
        figure_id="einstein",
        envelopes=envelopes,
        extra_replay_examples=extra,
    )
    assert any(
        ex.example_id == "replay:custom:gravity" for ex in plan.examples
    )


def test_build_lora_training_plan_rejects_duplicate_extra_replay() -> None:
    envelopes = _einstein_envelopes()
    extra = (
        ("replay:wikipedia:water-cycle", "Duplicate of built-in replay row"),
    )
    with pytest.raises(ValueError, match="duplicate replay_id"):
        build_lora_training_plan(
            figure_id="einstein",
            envelopes=envelopes,
            extra_replay_examples=extra,
        )


def test_build_lora_training_plan_rejects_empty_envelopes() -> None:
    with pytest.raises(ValueError, match="envelopes tuple"):
        build_lora_training_plan(figure_id="einstein", envelopes=())


def test_build_lora_training_plan_rejects_invalid_replay_weight() -> None:
    envelopes = _einstein_envelopes()
    with pytest.raises(ValueError, match="replay_weight"):
        build_lora_training_plan(
            figure_id="einstein",
            envelopes=envelopes,
            replay_weight=1.5,
        )


def test_lora_training_example_rejects_unknown_source_kind() -> None:
    with pytest.raises(ValueError, match="source_kind"):
        LoRATrainingExample(
            example_id="x",
            text="some text",
            locator="loc",
            source_kind="unknown",
            weight=1.0,
        )


def test_build_persona_lora_proposal_pins_invariants() -> None:
    envelopes = _einstein_envelopes()
    plan = build_lora_training_plan(figure_id="einstein", envelopes=envelopes)
    persona = build_persona_lora_proposal(
        figure_id="einstein",
        plan=plan,
        new_artifact_integrity_hash="abc123",
        previous_artifact_id="absent",
        rollback_evidence="rollback_to=absent;plan=" + plan.integrity_hash[:8],
    )
    assert isinstance(persona, PersonaLoRAProposal)
    assert persona.figure_id == "einstein"
    assert persona.training_plan_hash == plan.integrity_hash
    assert persona.proposal.target == "figure.persona_lora[einstein]"
    assert persona.proposal.desired_gate is ModificationGate.OFFLINE
    assert persona.proposal.is_reversible is True
    assert persona.proposal.rollback_evidence
    assert persona.proposal.old_value_hash != persona.proposal.new_value_hash


def test_build_persona_lora_proposal_requires_rollback_evidence() -> None:
    envelopes = _einstein_envelopes()
    plan = build_lora_training_plan(figure_id="einstein", envelopes=envelopes)
    with pytest.raises(ValueError, match="rollback_evidence"):
        build_persona_lora_proposal(
            figure_id="einstein",
            plan=plan,
            new_artifact_integrity_hash="abc123",
            previous_artifact_id="absent",
            rollback_evidence="",
        )


def test_build_persona_lora_proposal_rejects_figure_id_mismatch() -> None:
    envelopes = _einstein_envelopes()
    plan = build_lora_training_plan(figure_id="einstein", envelopes=envelopes)
    with pytest.raises(ValueError, match="plan.figure_id"):
        build_persona_lora_proposal(
            figure_id="not-einstein",
            plan=plan,
            new_artifact_integrity_hash="abc123",
            previous_artifact_id="absent",
            rollback_evidence="rb",
        )


def test_persona_lora_proposal_rejects_wrong_gate() -> None:
    envelopes = _einstein_envelopes()
    plan = build_lora_training_plan(figure_id="einstein", envelopes=envelopes)
    persona = build_persona_lora_proposal(
        figure_id="einstein",
        plan=plan,
        new_artifact_integrity_hash="abc123",
        previous_artifact_id="absent",
        rollback_evidence="rb",
    )
    from dataclasses import replace as _replace

    bad_proposal = _replace(
        persona.proposal,
        desired_gate=ModificationGate.ONLINE,
    )
    with pytest.raises(ValueError, match="desired_gate"):
        PersonaLoRAProposal(
            figure_id="einstein",
            previous_artifact_id="absent",
            training_plan_hash=plan.integrity_hash,
            proposal=bad_proposal,
        )
