from __future__ import annotations

import json
import pathlib
import runpy
from dataclasses import replace

import pytest

from lifeform_domain_emogpt.lab import sha256_json
from lifeform_evolution.relationship_lab_gate0 import (
    FrozenBaselineAttestation,
    Gate0CalibrationConfig,
    run_relationship_gate0_calibration,
)
from lifeform_evolution.relationship_lab_packet1b import (
    RelationshipP1bReport,
    RelationshipP1bVerdict,
)
from lifeform_evolution.relationship_lab_packet1c import (
    RelationshipP1cVerdict,
    assess_relationship_packet1c,
    load_relationship_p1c_candidate_protocol,
    load_relationship_packet1c_report,
    validate_relationship_p1c_local_lineage,
    write_relationship_packet1c_report,
)


_CREATED_AT = "2026-08-19T15:00:00+00:00"


def _baseline(*, valid_decisions: int = 24) -> FrozenBaselineAttestation:
    protocol = load_relationship_p1c_candidate_protocol()
    return FrozenBaselineAttestation(
        arm_id="stateless",
        dataset_fingerprint=protocol.dataset_fingerprint,
        model_id=protocol.model_id,
        weights_sha256=sha256_json("p1c-candidate-weights"),
        prompt_sha256=protocol.stateless_prompt_sha256,
        generation_config_sha256=protocol.expected_generation_config_sha256,
        seed_schedule_sha256=sha256_json(protocol.baseline_seed_schedule),
        decision_ledger_sha256=sha256_json("p1c-gate0-ledger"),
        evaluated_split="calibration",
        valid_decisions=valid_decisions,
        correct_decisions=4,
        evaluated_decisions=24,
        context_tokens_total=2048,
        hidden_test_opened=False,
        frozen_at_iso=_CREATED_AT,
    )


def _arm_metrics(*, accuracy: float, pair_flip_rate: float) -> tuple[tuple[str, object], ...]:
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
    machinery_ready: bool = True,
) -> RelationshipP1bReport:
    protocol = load_relationship_p1c_candidate_protocol()
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
        context_bundle_artifact_id=sha256_json("p1c-context-bundle"),
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
        readout_request_template_sha256=protocol.readout_request_template_sha256,
        readout_schema_sha256=protocol.readout_schema_sha256,
        compiler_version=protocol.compiler_version,
        run_artifact_id=sha256_json("p1c-p1b-run"),
        p1_report_artifact_id=sha256_json("p1c-p1-report"),
        readout_ledger_sha256=sha256_json("p1c-readout-ledger"),
        verdict=verdict,
        p1_machinery_ready=machinery_ready,
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


def test_p1c_protocol_is_content_addressed_and_binds_local_assets() -> None:
    protocol = load_relationship_p1c_candidate_protocol()
    validate_relationship_p1c_local_lineage(protocol)
    assert protocol.model_id == "qwen2.5-3b-instruct"
    assert protocol.reference_model_id == "qwen2.5-1.5b-instruct"
    assert not protocol.formal_hidden_test_opened

    raw = json.loads(protocol.to_json())
    raw["run_config"]["rag_top_k"] = 3
    with pytest.raises(ValueError, match="RAG top-k"):
        type(protocol).from_json(json.dumps(raw))


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
            RelationshipP1cVerdict.FORMAL_PREREG_FREEZE_CANDIDATE,
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
            RelationshipP1cVerdict.VERSION_SCENARIO_DATASET_SATURATED,
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
            RelationshipP1cVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT,
        ),
    ),
)
def test_p1c_routes_the_three_qualification_forks(p1b_kwargs, expected) -> None:
    protocol = load_relationship_p1c_candidate_protocol()
    baseline = _baseline()
    gate0 = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(),
        baseline=baseline,
        created_at_iso=_CREATED_AT,
    )
    assert gate0.gate0_passed
    p1b = _p1b_report(baseline, **p1b_kwargs)
    report = assess_relationship_packet1c(
        protocol=protocol,
        baseline=baseline,
        gate0_report=gate0,
        p1b_report=p1b,
        created_at_iso=_CREATED_AT,
    )
    assert report.verdict is expected


def test_p1c_keeps_gate_and_machinery_failures_outside_qualification(tmp_path) -> None:
    protocol = load_relationship_p1c_candidate_protocol()
    invalid_baseline = _baseline(valid_decisions=23)
    failed_gate0 = run_relationship_gate0_calibration(
        baseline=invalid_baseline,
        created_at_iso=_CREATED_AT,
    )
    rejected = assess_relationship_packet1c(
        protocol=protocol,
        baseline=invalid_baseline,
        gate0_report=failed_gate0,
        p1b_report=None,
        created_at_iso=_CREATED_AT,
    )
    assert rejected.verdict is RelationshipP1cVerdict.CANDIDATE_GATE0_REJECTED

    baseline = _baseline()
    gate0 = run_relationship_gate0_calibration(
        baseline=baseline,
        created_at_iso=_CREATED_AT,
    )
    machinery_failed = _p1b_report(
        baseline,
        verdict=RelationshipP1bVerdict.BASELINE_UNDERQUALIFIED,
        prompt_accuracy=0.75,
        prompt_pair_flip=0.5,
        rag_accuracy=0.75,
        rag_pair_flip=0.5,
        structured_pair_flip=0.75,
        machinery_ready=False,
    )
    report = assess_relationship_packet1c(
        protocol=protocol,
        baseline=baseline,
        gate0_report=gate0,
        p1b_report=machinery_failed,
        created_at_iso=_CREATED_AT,
    )
    assert report.verdict is RelationshipP1cVerdict.MACHINERY_REGRESSION
    paths = write_relationship_packet1c_report(report, output_dir=tmp_path)
    assert load_relationship_packet1c_report(paths[0]) == report

    tampered = json.loads(paths[0].read_text(encoding="utf-8"))
    tampered["weights_sha256"] = sha256_json("tampered")
    paths[0].write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id mismatch"):
        load_relationship_packet1c_report(paths[0])


def test_p1c_rejects_a_p1b_verdict_that_disagrees_with_its_metrics() -> None:
    protocol = load_relationship_p1c_candidate_protocol()
    baseline = _baseline()
    gate0 = run_relationship_gate0_calibration(
        baseline=baseline,
        created_at_iso=_CREATED_AT,
    )
    underqualified = _p1b_report(
        baseline,
        verdict=RelationshipP1bVerdict.BASELINE_UNDERQUALIFIED,
        prompt_accuracy=0.5,
        prompt_pair_flip=0.25,
        rag_accuracy=0.5,
        rag_pair_flip=0.25,
        structured_pair_flip=0.25,
    )
    inconsistent = replace(
        underqualified,
        verdict=RelationshipP1bVerdict.QUALIFIED,
    )
    with pytest.raises(ValueError, match="verdict diverges"):
        assess_relationship_packet1c(
            protocol=protocol,
            baseline=baseline,
            gate0_report=gate0,
            p1b_report=inconsistent,
            created_at_iso=_CREATED_AT,
        )


def test_p1c_runner_checkpoint_is_resumable_and_scope_bound(tmp_path) -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    runner = runpy.run_path(
        str(repo_root / "scripts" / "run_relationship_lab_packet1c.py"),
        run_name="relationship_p1c_runner_test",
    )
    protocol_id = sha256_json("p1c-checkpoint-test")
    stage, artifacts = runner["_load_or_initialize_checkpoint"](
        tmp_path / "run",
        protocol_id=protocol_id,
    )
    assert stage == "initialized"
    assert artifacts == {}

    active = tmp_path / "run" / "gate0_candidate"
    active.mkdir()
    artifacts["active_gate0_dir"] = "gate0_candidate"
    runner["_write_checkpoint"](
        tmp_path / "run",
        protocol_id=protocol_id,
        stage="gate0_running",
        artifacts=artifacts,
    )
    resumed_stage, resumed_artifacts = runner["_load_or_initialize_checkpoint"](
        tmp_path / "run",
        protocol_id=protocol_id,
    )
    assert resumed_stage == "gate0_running"
    assert resumed_artifacts == artifacts
    assert runner["_next_attempt_dir"](tmp_path / "run", "gate0_candidate").name == ("gate0_candidate_attempt_2")
    with pytest.raises(ValueError, match="escapes output root"):
        runner["_artifact_path"](
            tmp_path / "run",
            {"p1c_report": "../outside.json"},
            "p1c_report",
        )
