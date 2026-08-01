from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from volvence_zero.substrate import (
    CommonAdapterBundle,
    CommonAdapterGateRecord,
    ControlBasisArtifact,
    SubstrateDeltaAdapterLayer,
    SubstrateRareHeavyCheckpoint,
    build_teacher_distilled_prefix_artifact,
    rare_heavy_checkpoint_to_json,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "train_common_adapter_model.py"
)
SPEC = importlib.util.spec_from_file_location("train_common_adapter_model", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load common adapter trainer from {SCRIPT_PATH}")
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)

MODEL_ID = "Qwen/pipeline-test"


def _carriers():
    checkpoint = SubstrateRareHeavyCheckpoint(
        checkpoint_id="common-v1",
        model_id=MODEL_ID,
        runtime_origin="hf-local",
        control_scale=0.1,
        semantic_text_weight=0.5,
        semantic_residual_weight=0.5,
        semantic_anchor_bias=(0.0,) * 5,
        update_count=1,
        source_batch_count=1,
        mean_sequence_length=2.0,
        mean_residual_magnitude=0.01,
        description="pipeline checkpoint",
        checkpoint_version=2,
        training_mode="adapter-delta-v2",
        compatibility_fingerprint="adapter-delta-v2:test",
        adapter_scale=1.0,
        adapter_parameter_count=2,
        adapter_training_loss=0.1,
        adapter_layers=(
            SubstrateDeltaAdapterLayer(
                layer_index=0,
                delta_vector=(0.01, -0.01),
                mean_abs_delta=0.01,
                description="pipeline layer",
            ),
        ),
    )
    state = build_teacher_distilled_prefix_artifact(
        model_id=MODEL_ID,
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        num_slots=1,
        bottleneck_rank=1,
        encoder_rows=((1.0,),),
        encoder_bias=(0.0,),
        key_projection=(((1.0,), (0.0,)),),
        key_bias=((0.0, 0.0),),
        value_projection=(((0.5,), (0.0,)),),
        value_bias=((0.0, 0.0),),
        reference_key_norms=(1.0,),
        reference_value_norms=(1.0,),
        norm_cap=0.1,
        source_fingerprint="pipeline-state",
        sample_count=1,
        vector_labels=("state",),
    )
    control = ControlBasisArtifact(
        model_id=MODEL_ID,
        hidden_size=2,
        basis=((1.0, 0.0),),
        layer_indices=(0,),
        layer_gains=(1.0,),
        training_mode="train-transition-pca-v1",
        source_fingerprint="pipeline-control",
        sample_count=1,
        description="pipeline control",
    )
    return checkpoint, state, control


def _sha(path: Path) -> str:
    return PIPELINE._sha256_file(path)


def _candidate(tmp_path: Path):
    checkpoint, state, control = _carriers()
    checkpoint_path = tmp_path / "rare.json"
    state_path = tmp_path / "state.json"
    state_manifest_path = tmp_path / "state.manifest.json"
    control_path = tmp_path / "control.json"
    checkpoint_path.write_text(
        rare_heavy_checkpoint_to_json(checkpoint) + "\n",
        encoding="utf-8",
    )
    state_path.write_text(state.to_json() + "\n", encoding="utf-8")
    state_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "state-kv-prefix-bake.v1",
                "artifact_id": state.artifact_id,
                "model_id": MODEL_ID,
                "weights_sha256": "a" * 64,
                "training_order": "base+rare-heavy->state-kv",
                "common_adapter_version": "common-v1",
                "rare_heavy_checkpoint_sha256": _sha(checkpoint_path),
                "rare_heavy_compatibility_fingerprint": (
                    checkpoint.compatibility_fingerprint
                ),
                "state_count": 2,
                "seed": 8,
                "epochs": 1,
                "learning_rate": 0.01,
                "num_slots": 1,
                "bottleneck_rank": 1,
                "norm_cap": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    control_path.write_text(control.to_json() + "\n", encoding="utf-8")
    compatibility = CommonAdapterBundle.build_compatibility_fingerprint(
        common_adapter_version="common-v1",
        base_model_id=MODEL_ID,
        base_model_weights_sha256="a" * 64,
        rare_heavy_checkpoint=checkpoint,
        state_kv_artifact=state,
        control_basis_artifact=control,
    )
    candidate = {
        "schema_version": PIPELINE.COMMON_ADAPTER_CANDIDATE_SCHEMA,
        "candidate_id": "pending",
        "common_adapter_version": "common-v1",
        "base_model_id": MODEL_ID,
        "base_model_weights_sha256": "a" * 64,
        "compatibility_fingerprint": compatibility,
        "rare_heavy_checkpoint": {
            "locator": checkpoint_path.name,
            "sha256": _sha(checkpoint_path),
        },
        "state_kv_artifact": {
            "locator": state_path.name,
            "sha256": _sha(state_path),
            "training_manifest_locator": state_manifest_path.name,
            "training_manifest_sha256": _sha(state_manifest_path),
        },
        "control_basis_artifact": {
            "locator": control_path.name,
            "sha256": _sha(control_path),
        },
        "training_order": ["rare-heavy", "state-kv", "offline-gate"],
        "training_provenance": {
            "traces_sha256": "b" * 64,
            "trace_count": 1,
            "seed": 7,
            "target_modules": ["q_proj"],
            "hook_layers": [0],
            "lora_rank": 1,
            "lora_alpha": 1,
            "lora_dropout": 0.0,
            "learning_rate": 0.001,
            "max_steps": 1,
            "state_kv_seed": 8,
            "state_kv_states": 2,
            "state_kv_epochs": 1,
            "state_kv_slots": 1,
            "state_kv_rank": 1,
            "state_kv_norm_cap": 0.1,
            "state_kv_learning_rate": 0.01,
        },
        "description": "pipeline candidate",
    }
    candidate["candidate_id"] = PIPELINE._candidate_id(candidate)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    held_out_path = tmp_path / "held-out.jsonl"
    held_out_rows = [
        {
            "schema_version": "adapter-held-out-case.v1",
            "case_id": f"case-{index}",
            "cohort": "relationship" if index < 6 else "safety",
            "expectation": "improve" if index < 6 else "preserve",
            "source_text": f"source-{index}",
            "continuation_text": f"continuation-{index}",
            "conditioning_state": [0.1] * 16,
            "counterfactual_conditioning_state": [0.2] * 16,
            "applied_control": [0.0],
        }
        for index in range(8)
    ]
    held_out_path.write_text(
        "".join(json.dumps(row) + "\n" for row in held_out_rows),
        encoding="utf-8",
    )
    thresholds = PIPELINE.AdapterPromotionThresholds()
    observations = tuple(
        PIPELINE.AdapterArmObservation(
            case_id=f"case-{index}",
            cohort="relationship" if index < 6 else "safety",
            expectation="improve" if index < 6 else "preserve",
            baseline_nll=1.0,
            candidate_nll=0.75 if index < 6 else 1.0,
            validation_delta=0.25 if index < 6 else 0.0,
            relative_improvement=0.25 if index < 6 else 0.0,
            counterfactual_nll=0.9 if index < 6 else 1.1,
            own_state_margin=0.15 if index < 6 else 0.1,
            token_count=2,
        )
        for index in range(8)
    )
    summary = PIPELINE.summarize_observations(
        observations=observations,
        thresholds=thresholds,
    )
    evaluation_id = PIPELINE.evaluation_id(
        subject_id=candidate["candidate_id"],
        observations=observations,
        thresholds=thresholds,
    )
    report = {
        "schema_version": PIPELINE.ADAPTER_PROMOTION_REPORT_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "subject_kind": "common-adapter-candidate",
        "subject_id": candidate["candidate_id"],
        "common_adapter_version": "common-v1",
        "base_model_id": MODEL_ID,
        "base_model_weights_sha256": "a" * 64,
        "held_out_sha256": _sha(held_out_path),
        "held_out": True,
        "source_immutable": True,
        "feedback_free": True,
        "thresholds": asdict(thresholds),
        "observations": [asdict(row) for row in observations],
        "summary": summary,
        "artifact_parameter_count": 10,
        "base_model_parameter_count": 100,
        "capacity_cost": 0.1,
        "evaluation_snapshot": {},
        "gate_reasons": [],
        "decision": "allow",
        "rollback_evidence": "restore common-v0",
    }
    report_path = tmp_path / "evaluation.json"
    PIPELINE._write_json(report_path, report)
    gate = CommonAdapterGateRecord(
        proposal_id=f"common-adapter:{candidate['candidate_id']}",
        decision="allow",
        desired_gate="offline",
        validation_delta=0.25,
        capacity_cost=0.1,
        rollback_evidence="restore common-v0",
        is_reversible=True,
        evaluation_ref=(
            f"common-adapter-evaluation:{evaluation_id}:"
            f"sha256:{_sha(report_path)}"
        ),
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(asdict(gate)), encoding="utf-8")
    return (
        candidate_path,
        gate_path,
        report_path,
        held_out_path,
        state_manifest_path,
    )


def test_publish_binds_candidate_to_external_gate_record(tmp_path) -> None:
    candidate_path, gate_path, report_path, held_out_path, _ = _candidate(
        tmp_path
    )
    output = tmp_path / "common-adapter-bundle.json"

    bundle = PIPELINE.publish_bundle(
        candidate_path=candidate_path,
        gate_path=gate_path,
        evaluation_report_path=report_path,
        held_out_path=held_out_path,
        output_path=output,
    )

    assert output.is_file()
    assert bundle.active_eligible
    assert CommonAdapterBundle.from_json(output.read_text(encoding="utf-8")) == bundle


def test_public_evidence_validator_binds_candidate_report_and_gate(tmp_path) -> None:
    candidate_path, gate_path, report_path, held_out_path, _ = _candidate(
        tmp_path
    )

    evidence = PIPELINE.validate_common_adapter_evidence(
        candidate_path=candidate_path,
        gate_path=gate_path,
        evaluation_report_path=report_path,
        held_out_path=held_out_path,
    )

    assert evidence.material.payload["candidate_id"]
    assert evidence.gate.decision == "allow"
    assert evidence.report["held_out"] is True


def test_publish_rejects_state_kv_training_order_tamper(tmp_path) -> None:
    candidate_path, gate_path, report_path, held_out_path, state_manifest_path = (
        _candidate(tmp_path)
    )
    raw = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    raw["training_order"] = "base-only-legacy"
    state_manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="training manifest digest mismatch"):
        PIPELINE.publish_bundle(
            candidate_path=candidate_path,
            gate_path=gate_path,
            evaluation_report_path=report_path,
            held_out_path=held_out_path,
            output_path=tmp_path / "bundle.json",
        )


def test_candidate_requires_bound_training_provenance(tmp_path) -> None:
    candidate_path, _, _, _, _ = _candidate(tmp_path)
    raw = json.loads(candidate_path.read_text(encoding="utf-8"))
    raw.pop("training_provenance")
    raw["candidate_id"] = PIPELINE._candidate_id(raw)
    candidate_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="training_provenance"):
        PIPELINE.load_candidate_material(candidate_path)


def test_publish_rejects_evaluation_report_digest_tamper(tmp_path) -> None:
    candidate_path, gate_path, report_path, held_out_path, _ = _candidate(
        tmp_path
    )
    report_path.write_text(
        report_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation report digest"):
        PIPELINE.publish_bundle(
            candidate_path=candidate_path,
            gate_path=gate_path,
            evaluation_report_path=report_path,
            held_out_path=held_out_path,
            output_path=tmp_path / "bundle.json",
        )


def test_evidence_rejects_observations_not_bound_to_held_out_cases(tmp_path) -> None:
    candidate_path, gate_path, report_path, held_out_path, _ = _candidate(
        tmp_path
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["observations"][0]["case_id"] = "substituted-case"
    PIPELINE._write_json(report_path, report)

    with pytest.raises(ValueError, match="ordered held-out cases"):
        PIPELINE.validate_common_adapter_evidence(
            candidate_path=candidate_path,
            gate_path=gate_path,
            evaluation_report_path=report_path,
            held_out_path=held_out_path,
        )
