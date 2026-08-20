from __future__ import annotations

import json
import pathlib
import runpy
from dataclasses import replace

import pytest

from companion_ref_harness import HashingEmbedder
from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
    sha256_json,
)
from lifeform_evolution.relationship_lab_contexts import (
    RelationshipP1Arm,
    RelationshipP1RagCandidateSurface,
    build_relationship_p1_context_bundle,
)
from lifeform_evolution.relationship_lab_gate0 import (
    FrozenBaselineAttestation,
    Gate0CalibrationConfig,
    run_relationship_gate0_calibration,
)
from lifeform_evolution.relationship_lab_packet1b import (
    RelationshipP1bReport,
    RelationshipP1bVerdict,
    load_relationship_packet1b_report,
)
from lifeform_evolution.relationship_lab_packet1f import (
    RelationshipP1fVerdict,
    load_relationship_packet1f_report,
)
from lifeform_evolution.relationship_lab_packet1g import (
    RelationshipP1gVerdict,
    assess_relationship_packet1g,
    load_relationship_p1g_consumer_protocol,
    load_relationship_packet1g_report,
    validate_relationship_p1g_local_lineage,
    write_relationship_packet1g_report,
)


_CREATED_AT = "2026-08-19T23:30:00+00:00"


def _source_p1f_report():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    return load_relationship_packet1f_report(
        repo_root
        / "artifacts"
        / "relationship_lab"
        / "bge_m3_packet1f_v3_public_evidence_20260820"
        / "packet1f_report.json"
    )


def _baseline(*, valid_decisions: int = 24) -> FrozenBaselineAttestation:
    protocol = load_relationship_p1g_consumer_protocol()
    return FrozenBaselineAttestation(
        arm_id="stateless",
        dataset_fingerprint=protocol.dataset_fingerprint,
        model_id=protocol.model_id,
        weights_sha256=protocol.expected_weights_sha256,
        prompt_sha256=protocol.stateless_prompt_sha256,
        generation_config_sha256=protocol.expected_generation_config_sha256,
        seed_schedule_sha256=sha256_json(protocol.baseline_seed_schedule),
        decision_ledger_sha256=sha256_json("p1g-gate0-ledger"),
        evaluated_split="calibration",
        valid_decisions=valid_decisions,
        correct_decisions=10,
        evaluated_decisions=24,
        context_tokens_total=2048,
        hidden_test_opened=False,
        frozen_at_iso=_CREATED_AT,
    )


def _arm_metrics(
    *,
    accuracy: float,
    pair_flip_rate: float,
) -> tuple[tuple[str, object], ...]:
    decisions = 8
    return tuple(
        sorted(
            {
                "accuracy": accuracy,
                "completion_tokens_total": 80,
                "correct_decisions": int(accuracy * decisions),
                "decisions": decisions,
                "pair_flip_rate": pair_flip_rate,
                "pair_groups": 4,
                "prompt_tokens_total": 800,
                "readouts": decisions,
                "valid_decisions": decisions,
                "valid_pair_groups": 4,
                "valid_rate": 1.0,
                "valid_readouts": decisions,
            }.items()
        )
    )


def _p1b_report(
    baseline: FrozenBaselineAttestation,
    *,
    verdict: RelationshipP1bVerdict,
    prompt_accuracy: float,
    prompt_pair_flip: float,
    rag_accuracy: float,
    rag_pair_flip: float,
    structured_pair_flip: float,
) -> RelationshipP1bReport:
    protocol = load_relationship_p1g_consumer_protocol()
    saturated_arms = tuple(
        arm
        for arm, accuracy in (
            ("prompt-steelman", prompt_accuracy),
            ("rag-steelman", rag_accuracy),
        )
        if accuracy > 0.875
    )
    return RelationshipP1bReport(
        created_at_iso=_CREATED_AT,
        dataset_fingerprint=protocol.dataset_fingerprint,
        context_bundle_artifact_id=sha256_json("p1g-context-bundle"),
        evaluated_context_surface_sha256=(protocol.evaluated_context_surface_sha256),
        background_templates_sha256=protocol.background_templates_sha256,
        rag_config_sha256=protocol.rag_config_sha256,
        seed_schedule_sha256=sha256_json(protocol.p1b_seed_schedule),
        p1_gate_config_sha256=protocol.p1_gate_config_sha256,
        model_id=protocol.model_id,
        weights_sha256=baseline.weights_sha256,
        generation_config_sha256=baseline.generation_config_sha256,
        gate0_baseline_attestation_id=baseline.artifact_id,
        readout_prompt_sha256=protocol.readout_prompt_sha256,
        readout_request_template_sha256=(protocol.readout_request_template_sha256),
        readout_schema_sha256=protocol.readout_schema_sha256,
        compiler_version=protocol.compiler_version,
        run_artifact_id=sha256_json("p1g-p1b-run"),
        p1_report_artifact_id=sha256_json("p1g-p1-report"),
        readout_ledger_sha256=sha256_json("p1g-readout-ledger"),
        verdict=verdict,
        p1_machinery_ready=True,
        all_readouts_valid=True,
        saturated_arms=saturated_arms,
        arm_metrics=(
            (
                "prompt-steelman",
                _arm_metrics(
                    accuracy=prompt_accuracy,
                    pair_flip_rate=prompt_pair_flip,
                ),
            ),
            (
                "rag-steelman",
                _arm_metrics(
                    accuracy=rag_accuracy,
                    pair_flip_rate=rag_pair_flip,
                ),
            ),
            (
                "structured-state",
                _arm_metrics(
                    accuracy=0.75,
                    pair_flip_rate=structured_pair_flip,
                ),
            ),
        ),
    )


def test_p1g_protocol_freezes_zero_output_v3_lineage() -> None:
    protocol = load_relationship_p1g_consumer_protocol()
    source = _source_p1f_report()
    validate_relationship_p1g_local_lineage(
        protocol,
        source_p1f_report=source,
    )
    assert protocol.package_name == RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME
    assert protocol.source_p1f_report_artifact_id == source.artifact_id
    assert source.verdict is RelationshipP1fVerdict.CONSUMER_PROTOCOL_FREEZE_CANDIDATE
    assert protocol.expected_weights_sha256 == ("3ccf77de3297aba6772fcb743af28b806d7b7c3e348cc7e8ad729fa98a4146cd")
    assert protocol.histories_per_user == 4
    assert protocol.current_message_participates
    assert protocol.all_four_histories_available
    assert protocol.rag_top_k == 4
    assert protocol.readout_profile == "v2_condition_aware"
    assert protocol.v3_qwen_outputs_observed_before_freeze == 0
    assert not protocol.formal_hidden_test_opened
    assert not protocol.p2_enabled

    with pytest.raises(ValueError, match="first v3 Qwen output"):
        replace(protocol, v3_qwen_outputs_observed_before_freeze=1)
    with pytest.raises(ValueError, match="top-k"):
        replace(protocol, rag_top_k=2)


def test_p1g_rejects_p1f_lineage_substitution() -> None:
    protocol = load_relationship_p1g_consumer_protocol()
    source = _source_p1f_report()
    substituted = replace(
        protocol,
        source_p1f_report_artifact_id=sha256_json("substituted-p1f-report"),
    )
    with pytest.raises(ValueError, match="source P1f report"):
        validate_relationship_p1g_local_lineage(
            substituted,
            source_p1f_report=source,
        )
    with pytest.raises(ValueError, match="RAG identity"):
        validate_relationship_p1g_local_lineage(
            replace(protocol, rag_model_source="example/drifted-rag"),
            source_p1f_report=source,
        )


def test_p1g_v3_contexts_publish_all_four_histories_without_truth(tmp_path) -> None:
    dataset = load_relationship_transfer_dataset(package_name=RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME)
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
        dataset=dataset,
        background_depths=(0, 8),
        rag_top_k=4,
        rag_candidate_surface=(RelationshipP1RagCandidateSurface.RELATIONSHIP_OUTCOMES_ONLY),
    )
    sealed_tokens = {item.condition_id for item in dataset.abstract_conditions} | {
        item.policy_id for item in dataset.policy_profiles
    }
    for observation in dataset.observations:
        signal_ids = {item.event_id for item in observation.histories}
        for arm in (
            RelationshipP1Arm.PROMPT_STEELMAN,
            RelationshipP1Arm.RAG_STEELMAN,
            RelationshipP1Arm.STRUCTURED_STATE,
        ):
            context = bundle.context(scene_id=observation.scene_id, arm=arm)
            assert context.context_text.count("[public relationship outcome evidence]") == 4
            assert not any(token in context.context_text for token in sealed_tokens)
        rag = bundle.context(
            scene_id=observation.scene_id,
            arm=RelationshipP1Arm.RAG_STEELMAN,
        )
        assert set(rag.source_evidence_refs) == signal_ids
        structured = bundle.context(
            scene_id=observation.scene_id,
            arm=RelationshipP1Arm.STRUCTURED_STATE,
        )
        assert len(structured.source_evidence_refs) == 4


@pytest.mark.parametrize(
    ("p1b_kwargs", "expected"),
    (
        (
            {
                "verdict": RelationshipP1bVerdict.QUALIFIED,
                "prompt_accuracy": 0.75,
                "prompt_pair_flip": 0.5,
                "rag_accuracy": 0.75,
                "rag_pair_flip": 0.5,
                "structured_pair_flip": 0.75,
            },
            RelationshipP1gVerdict.FORMAL_PREREG_FREEZE_CANDIDATE,
        ),
        (
            {
                "verdict": RelationshipP1bVerdict.DATASET_SATURATED,
                "prompt_accuracy": 1.0,
                "prompt_pair_flip": 1.0,
                "rag_accuracy": 0.75,
                "rag_pair_flip": 0.5,
                "structured_pair_flip": 0.75,
            },
            RelationshipP1gVerdict.SCENARIO_SATURATED_AFTER_EVIDENCE_REPAIR,
        ),
        (
            {
                "verdict": RelationshipP1bVerdict.BASELINE_UNDERQUALIFIED,
                "prompt_accuracy": 0.5,
                "prompt_pair_flip": 0.25,
                "rag_accuracy": 0.5,
                "rag_pair_flip": 0.25,
                "structured_pair_flip": 0.25,
            },
            RelationshipP1gVerdict.CONSUMER_STILL_UNDERQUALIFIED,
        ),
    ),
)
def test_p1g_routes_frozen_qualification_results(
    tmp_path,
    p1b_kwargs,
    expected,
) -> None:
    protocol = load_relationship_p1g_consumer_protocol()
    source = _source_p1f_report()
    baseline = _baseline()
    gate0 = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(),
        baseline=baseline,
        package_root=relationship_transfer_package_dir(protocol.package_name),
        created_at_iso=_CREATED_AT,
    )
    assert gate0.gate0_passed
    report = assess_relationship_packet1g(
        protocol=protocol,
        source_p1f_report=source,
        baseline=baseline,
        gate0_report=gate0,
        p1b_report=_p1b_report(baseline, **p1b_kwargs),
        created_at_iso=_CREATED_AT,
    )
    assert report.verdict is expected
    paths = write_relationship_packet1g_report(report, output_dir=tmp_path)
    assert load_relationship_packet1g_report(paths[0]) == report


def test_p1g_rejects_candidate_weight_drift() -> None:
    protocol = load_relationship_p1g_consumer_protocol()
    source = _source_p1f_report()
    baseline = replace(_baseline(), weights_sha256=sha256_json("drifted-weights"))
    gate0 = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(),
        baseline=baseline,
        package_root=relationship_transfer_package_dir(protocol.package_name),
        created_at_iso=_CREATED_AT,
    )
    with pytest.raises(ValueError, match="baseline diverges"):
        assess_relationship_packet1g(
            protocol=protocol,
            source_p1f_report=source,
            baseline=baseline,
            gate0_report=gate0,
            p1b_report=None,
            created_at_iso=_CREATED_AT,
        )


def test_p1g_runner_checkpoint_is_source_bound_and_scope_safe(tmp_path) -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    runner = runpy.run_path(
        str(repo_root / "scripts" / "run_relationship_lab_packet1g.py"),
        run_name="relationship_p1g_runner_test",
    )
    protocol_id = sha256_json("p1g-checkpoint-test")
    source_id = sha256_json("p1g-source-test")
    stage, artifacts = runner["_load_or_initialize_checkpoint"](
        tmp_path / "run",
        protocol_id=protocol_id,
        source_p1f_report_artifact_id=source_id,
    )
    assert stage == "initialized"
    assert artifacts == {}
    runner["_write_checkpoint"](
        tmp_path / "run",
        protocol_id=protocol_id,
        source_p1f_report_artifact_id=source_id,
        stage="gate0_running",
        artifacts={"active_gate0_dir": "gate0_candidate"},
    )
    resumed_stage, resumed = runner["_load_or_initialize_checkpoint"](
        tmp_path / "run",
        protocol_id=protocol_id,
        source_p1f_report_artifact_id=source_id,
    )
    assert resumed_stage == "gate0_running"
    assert resumed == {"active_gate0_dir": "gate0_candidate"}
    with pytest.raises(ValueError, match="another P1f source"):
        runner["_load_or_initialize_checkpoint"](
            tmp_path / "run",
            protocol_id=protocol_id,
            source_p1f_report_artifact_id=sha256_json("wrong-source"),
        )
    with pytest.raises(ValueError, match="escapes output root"):
        runner["_artifact_path"](
            tmp_path / "run",
            {"p1g_report": "../outside.json"},
            "p1g_report",
        )


def test_p1g_protocol_json_rejects_unknown_fields() -> None:
    protocol = load_relationship_p1g_consumer_protocol()
    raw = json.loads(protocol.to_json())
    raw["after_the_fact_prompt_tuning"] = True
    with pytest.raises(ValueError, match="frozen schema"):
        type(protocol).from_json(json.dumps(raw))


def test_authoritative_p1g_artifacts_round_trip_to_frozen_verdict() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    run_root = repo_root / "artifacts" / "relationship_lab" / "qwen25_3b_packet1g_v3_conditioned_top4_20260820"
    protocol = load_relationship_p1g_consumer_protocol()
    p1b = load_relationship_packet1b_report(run_root / "p1b_candidate" / "packet1b_report.json")
    report = load_relationship_packet1g_report(run_root / "packet1g_report.json")

    assert protocol.protocol_id == ("8e08d488382442f364aae102d80c268c8c23927d547f64c1e79cb0a87f0f52c6")
    assert p1b.artifact_id == ("10d120f49b442803cccec53c534e8f3c868ee644c0674439ede000d8dedd3a87")
    assert report.artifact_id == ("9d7f05b574bafb21641d22c766fe31c4656c09bf6f5e04493474eee6c694e3c8")
    assert report.consumer_protocol_id == protocol.protocol_id
    assert report.p1b_report_artifact_id == p1b.artifact_id
    assert report.verdict is RelationshipP1gVerdict.CONSUMER_STILL_UNDERQUALIFIED
    assert report.qualification_metrics == (
        ("prompt-steelman", 0.75, 0.5),
        ("rag-steelman", 0.5, 0.0),
        ("structured-state", 0.5, 0.5),
    )
