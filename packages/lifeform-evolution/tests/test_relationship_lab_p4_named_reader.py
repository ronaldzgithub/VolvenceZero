from __future__ import annotations

import hashlib
import json

import pytest

from lifeform_domain_emogpt.relationship_condition_reader import (
    RelationshipConditionPrototype,
    RelationshipConditionReaderArtifact,
)
from lifeform_evolution.relationship_lab_p4_named_reader import (
    P4NamedReaderArm,
    run_relationship_p4_named_reader_transmission,
    validate_relationship_p4_named_reader_report_files,
    write_relationship_p4_named_reader_markdown,
    write_relationship_p4_named_reader_report,
)
from lifeform_evolution.relationship_lab_p4_pe_learning import (
    P4PeLearningArm,
    run_relationship_p4_pe_credit_learning,
    validate_relationship_p4_pe_learning_report_files,
    write_relationship_p4_pe_learning_report,
)
from lifeform_evolution.relationship_lab_packet1m_qualification import (
    RelationshipP1mArmMetrics,
    RelationshipP1mQualificationArm,
    RelationshipP1mQualificationProtocol,
    RelationshipP1mQualificationReport,
    RelationshipP1mQualificationVerdict,
    wilson_one_sided_lower,
)


_SHA = "a" * 64


class _DeterministicFixtureEmbedder:
    """Stable test double; it is never used as relationship evidence."""

    def embed(self, text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return tuple((value + 1) / 256.0 for value in digest[:16])


def _reader_artifact() -> RelationshipConditionReaderArtifact:
    return RelationshipConditionReaderArtifact(
        embedding_model_id="fixture-bge",
        embedding_weights_sha256=_SHA,
        prototypes=(
            RelationshipConditionPrototype(
                label="agency_displacement",
                summary="当事人的声音、选择或决定被别人越过和取代。",
            ),
            RelationshipConditionPrototype(
                label="belonging_erasure",
                summary="当事人在共同经历和关系网络中的位置被遗漏。",
            ),
        ),
    )


def _protocol() -> RelationshipP1mQualificationProtocol:
    return RelationshipP1mQualificationProtocol(
        frozen_at_iso="2026-08-22T15:00:00+08:00",
        source_p1k_report_artifact_id=_SHA,
        source_generation_attestation_id="b" * 64,
        source_generation_protocol_id="c" * 64,
        source_transport_id="d" * 64,
        source_seed_inventory_sha256="e" * 64,
        package_name="relationship_transfer_p1m_v1",
        dataset_fingerprint="f" * 64,
        pair_count=24,
        scene_count=48,
        qwen_model_source="Qwen/Qwen2.5-3B-Instruct",
        qwen_model_revision="main",
        qwen_model_id="fixture-qwen",
        qwen_weights_sha256="1" * 64,
        qwen_snapshot_sha256="2" * 64,
        qwen_device="cpu",
        qwen_torch_dtype="bfloat16",
        prompt_sha256="3" * 64,
        request_template_sha256="4" * 64,
        token_a_id=32,
        token_b_id=33,
        scoring_method="exact_first_assistant_token_logits_A_vs_B",
        qwen_config_sha256="5" * 64,
        bge_model_source="BAAI/bge-m3",
        bge_model_revision="main",
        bge_weights_sha256=_SHA,
        reader_artifact=_reader_artifact(),
        rag_top_k=4,
        plan_artifact_id="6" * 64,
        planned_qwen_readouts=96,
        planned_structured_readouts=48,
        qualification_inputs_observed_before_freeze=0,
        qwen_outputs_observed_before_freeze=0,
        structured_outputs_observed_before_freeze=0,
        first_qualification_attempt_only=True,
        evaluation_feedback_allowed=False,
    )


def _metric(
    arm: RelationshipP1mQualificationArm,
    *,
    correct: int,
    flips: int,
) -> RelationshipP1mArmMetrics:
    return RelationshipP1mArmMetrics(
        arm=arm,
        decisions=48,
        valid_decisions=48,
        correct_decisions=correct,
        accuracy=correct / 48,
        accuracy_wilson_lower=wilson_one_sided_lower(correct, 48),
        mirrored_pairs=24,
        pair_flips=flips,
        pair_flip_rate=flips / 24,
        pair_flip_wilson_lower=wilson_one_sided_lower(flips, 24),
    )


def _failed_p1m_report(
    protocol: RelationshipP1mQualificationProtocol,
) -> RelationshipP1mQualificationReport:
    return RelationshipP1mQualificationReport(
        created_at_iso="2026-08-22T18:00:00+08:00",
        protocol_id=protocol.protocol_id,
        plan_artifact_id=protocol.plan_artifact_id,
        dataset_fingerprint=protocol.dataset_fingerprint,
        qwen_readout_ledger_sha256="7" * 64,
        structured_readout_ledger_sha256="8" * 64,
        qwen_decision_ledger_sha256="9" * 64,
        structured_decision_ledger_sha256="0" * 64,
        arm_metrics=(
            _metric(
                RelationshipP1mQualificationArm.PROMPT_STEELMAN,
                correct=24,
                flips=0,
            ),
            _metric(
                RelationshipP1mQualificationArm.RAG_STEELMAN,
                correct=24,
                flips=0,
            ),
            _metric(
                RelationshipP1mQualificationArm.STRUCTURED_STATE,
                correct=46,
                flips=24,
            ),
        ),
        verdict=RelationshipP1mQualificationVerdict.BASELINE_TOO_WEAK,
        qualification_passed=False,
        scenario_versioning_closed=True,
        evaluation_feedback_to_system=False,
    )


async def test_named_reader_transmission_isolates_readout_and_freezes_artifacts(
    tmp_path,
) -> None:
    protocol = _protocol()
    report = await run_relationship_p4_named_reader_transmission(
        p1m_protocol=protocol,
        p1m_report=_failed_p1m_report(protocol),
        embedder=_DeterministicFixtureEmbedder(),
        embedding_model_id=protocol.reader_artifact.embedding_model_id,
        embedding_weights_sha256=protocol.bge_weights_sha256,
    )

    assert len(report.runs) == 4
    assert tuple(item.arm for item in report.summaries) == tuple(P4NamedReaderArm)
    assert all(not run.mechanism.credit_applied_to_gate for run in report.runs)
    assert all(run.mechanism.gate_update_count == 0 for run in report.runs)
    assert all(run.mechanism.process_restart_count == 11 for run in report.runs)
    legacy, named = report.summaries
    assert legacy.named_readout_count == 0
    assert named.named_readout_count == 16
    assert report.component_selected_after_p1m_observation is True
    assert report.seen_fixture_only is True
    assert report.formal_evidence_authorized is False

    report_path = write_relationship_p4_named_reader_report(
        report,
        output_dir=tmp_path,
    )
    markdown_path = write_relationship_p4_named_reader_markdown(
        report,
        output_dir=tmp_path,
    )
    validate_relationship_p4_named_reader_report_files(
        report,
        report_path=report_path,
        markdown_path=markdown_path,
    )
    assert b"\r\n" not in report_path.read_bytes()
    assert b"\r\n" not in markdown_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_relationship_p4_named_reader_report(report, output_dir=tmp_path)
    with pytest.raises(FileExistsError):
        write_relationship_p4_named_reader_markdown(report, output_dir=tmp_path)

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    raw["positive_outcome_gain"] += 1
    report_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact drift"):
        validate_relationship_p4_named_reader_report_files(
            report,
            report_path=report_path,
            markdown_path=markdown_path,
        )


async def test_pe_credit_learning_isolates_updates_and_freezes_artifacts(
    tmp_path,
) -> None:
    protocol = _protocol()
    embedder = _DeterministicFixtureEmbedder()
    source_report = await run_relationship_p4_named_reader_transmission(
        p1m_protocol=protocol,
        p1m_report=_failed_p1m_report(protocol),
        embedder=embedder,
        embedding_model_id=protocol.reader_artifact.embedding_model_id,
        embedding_weights_sha256=protocol.bge_weights_sha256,
    )

    report = await run_relationship_p4_pe_credit_learning(
        p1m_protocol=protocol,
        source_report=source_report,
        embedder=embedder,
    )

    no_credit, pe_credit = report.summaries
    assert no_credit.arm is P4PeLearningArm.NAMED_LEARNED_NO_CREDIT
    assert no_credit.steer_count == 0
    assert no_credit.credit_applied_count == 0
    assert no_credit.parameter_change_count == 0
    assert no_credit.final_update_count == 0
    assert pe_credit.arm is P4PeLearningArm.NAMED_LEARNED_PE_CREDIT
    assert pe_credit.credit_applied_count == 16
    assert pe_credit.parameter_change_count == 16
    assert pe_credit.final_update_count == 16
    assert report.causal_next_pulse_probability_change_count > 0
    assert report.evaluation_feedback_to_learning is False
    assert report.seen_fixture_only is True
    assert report.formal_evidence_authorized is False
    assert all(
        run.mechanism.gate_audits[0].pre_update_count == 0
        for run in report.runs
    )

    json_path, markdown_path = write_relationship_p4_pe_learning_report(
        report,
        output_dir=tmp_path,
    )
    validate_relationship_p4_pe_learning_report_files(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    assert b"\r\n" not in json_path.read_bytes()
    assert b"\r\n" not in markdown_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_relationship_p4_pe_learning_report(report, output_dir=tmp_path)

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw["positive_outcome_gain"] += 1
    json_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact drift"):
        validate_relationship_p4_pe_learning_report_files(
            report,
            json_path=json_path,
            markdown_path=markdown_path,
        )


async def test_named_reader_source_rejects_embedding_lineage_drift() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="weights lineage drift"):
        # Source validation fails before any owner or environment is executed.
        await run_relationship_p4_named_reader_transmission(
            p1m_protocol=protocol,
            p1m_report=_failed_p1m_report(protocol),
            embedder=_DeterministicFixtureEmbedder(),
            embedding_model_id=protocol.reader_artifact.embedding_model_id,
            embedding_weights_sha256="f" * 64,
        )
