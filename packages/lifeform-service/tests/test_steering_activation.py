from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from lifeform_service import VerticalSpec
from lifeform_service import cli
from lifeform_service.steering_activation import (
    build_steering_activation_canary_receipt,
    load_steering_activation_authorization,
    write_steering_activation_canary_receipt,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.steering_contracts import (
    STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION,
    STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
    STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
    STEERING_READER_ARTIFACT_SCHEMA_VERSION,
    SteeringArtifactBundle,
    SteeringExecutorArtifact,
    SteeringGateArtifact,
    SteeringReaderArtifact,
)


_C3_PREREGISTRATION_SHA256 = "a" * 64
_MODEL_WEIGHTS_SHA256 = "b" * 64
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_ACTIVATION_SOURCE_PATHS = (
    "packages/lifeform-service/src/lifeform_service/cli.py",
    "packages/lifeform-service/src/lifeform_service/steering_activation.py",
    "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
    "scripts/verify_steering_activation_canary.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _activation_source_sha256() -> dict[str, str]:
    return {
        name: _sha256(_REPOSITORY_ROOT / name)
        for name in _ACTIVATION_SOURCE_PATHS
    }


def _command_sha256(command: tuple[str, ...]) -> str:
    payload = (
        json.dumps(
            list(command),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canary_command(
    *,
    bundle_path: Path,
    manifest_path: Path,
    plan_path: Path,
    step: int,
    previous_receipt: Path | None,
) -> tuple[str, ...]:
    command = (
        "/fixture/python",
        "-m",
        "lifeform_service.cli",
        "--vertical",
        "companion",
        "--steering-artifact-bundle",
        str(bundle_path.resolve()),
        "--steering-promotion-manifest",
        str(manifest_path.resolve()),
        "--steering-activation-plan",
        str(plan_path.resolve()),
        "--steering-activation-step",
        str(step),
    )
    if previous_receipt is not None:
        command = (
            *command,
            "--steering-previous-activation-receipt",
            str(previous_receipt.resolve()),
        )
    return command


def _bundle() -> SteeringArtifactBundle:
    reader = SteeringReaderArtifact(
        schema_version=STEERING_READER_ARTIFACT_SCHEMA_VERSION,
        artifact_id="reader-v1",
        model_id="frozen-model",
        model_weights_sha256=_MODEL_WEIGHTS_SHA256,
        source_preregistration_sha256=_C3_PREREGISTRATION_SHA256,
        layer_index=20,
        residual_width=2,
        class_labels=("relationship", "task"),
        weights=((1.0, -1.0), (0.0, 0.0)),
        feature_mean=(0.0, 0.0),
        feature_scale=(1.0, 1.0),
        ridge_lambda=0.1,
        description="Frozen reader.",
    )
    executor = SteeringExecutorArtifact(
        schema_version=STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
        artifact_id="executor-v1",
        model_id=reader.model_id,
        model_weights_sha256=reader.model_weights_sha256,
        source_preregistration_sha256=_C3_PREREGISTRATION_SHA256,
        reader_artifact_id=reader.artifact_id,
        layer_index=20,
        residual_width=2,
        rank=2,
        class_labels=reader.class_labels,
        u_factors=((1.0, 0.0), (0.0, 1.0)),
        v_factors=((1.0, 0.0), (0.0, 1.0)),
        condition_codes=((1.0, 0.0), (0.0, 1.0)),
        control_norm_cap_ratio=0.25,
        free_bias_present=False,
        zero_code_strict_noop=True,
        description="Frozen conditional executor.",
    )
    sensor_off = SteeringExecutorArtifact(
        schema_version=STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
        artifact_id="executor-sensor-off-v1",
        model_id=reader.model_id,
        model_weights_sha256=reader.model_weights_sha256,
        source_preregistration_sha256=_C3_PREREGISTRATION_SHA256,
        reader_artifact_id=reader.artifact_id,
        layer_index=20,
        residual_width=2,
        rank=2,
        class_labels=reader.class_labels,
        u_factors=executor.u_factors,
        v_factors=executor.v_factors,
        condition_codes=((0.5, -0.5), (0.5, -0.5)),
        control_norm_cap_ratio=0.25,
        free_bias_present=False,
        zero_code_strict_noop=True,
        description="Frozen sensor-off control.",
    )
    gate = SteeringGateArtifact(
        schema_version=STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
        artifact_id="learned-gate-v1",
        source_preregistration_sha256=_C3_PREREGISTRATION_SHA256,
        feature_names=("belief_margin",),
        weights=((0.0, 1.0),),
        bias=(0.0, 0.0),
        policy_version=4,
        description="Frozen learned gate.",
    )
    return SteeringArtifactBundle(
        schema_version=STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION,
        bundle_id="candidate-bundle-v1",
        reader=reader,
        executor=executor,
        gate=gate,
        sensor_off_executor=sensor_off,
        description="B3 candidate bundle.",
    )


def _artifacts(
    tmp_path: Path,
) -> tuple[SteeringArtifactBundle, Path, Path, Path, str]:
    bundle = _bundle()
    bundle_path = tmp_path / "candidate.json"
    bundle_path.write_text(bundle.to_json() + "\n", encoding="utf-8")
    bundle_sha256 = _sha256(bundle_path)
    steps = [
        {
            "order": 1,
            "purpose": "activate the first authorized owner",
            "single_field_flip": {
                "field": "steering_sensor",
                "from": "shadow",
                "to": "active",
            },
            "rollout_values_after_flip": {
                "steering_sensor": "active",
                "steering_executor": "shadow",
                "steering_gate": "shadow",
                "steering_ungated_action": "blocked",
            },
        },
        {
            "order": 2,
            "purpose": "prepare the explicit gate-off arm while executor is SHADOW",
            "single_field_flip": {
                "field": "steering_ungated_action",
                "from": "blocked",
                "to": "always_on",
            },
            "rollout_values_after_flip": {
                "steering_sensor": "active",
                "steering_executor": "shadow",
                "steering_gate": "shadow",
                "steering_ungated_action": "always_on",
            },
        },
        {
            "order": 3,
            "purpose": "activate the second authorized owner",
            "single_field_flip": {
                "field": "steering_executor",
                "from": "shadow",
                "to": "active",
            },
            "rollout_values_after_flip": {
                "steering_sensor": "active",
                "steering_executor": "active",
                "steering_gate": "shadow",
                "steering_ungated_action": "always_on",
            },
        },
    ]
    rollback_steps = [
        {
            "order": 1,
            "single_field_flip": {
                "field": "steering_executor",
                "from": "active",
                "to": "shadow",
            },
            "rollout_values_after_flip": steps[1]["rollout_values_after_flip"],
        },
        {
            "order": 2,
            "single_field_flip": {
                "field": "steering_ungated_action",
                "from": "always_on",
                "to": "blocked",
            },
            "rollout_values_after_flip": steps[0]["rollout_values_after_flip"],
        },
        {
            "order": 3,
            "single_field_flip": {
                "field": "steering_sensor",
                "from": "active",
                "to": "shadow",
            },
            "rollout_values_after_flip": {
                "steering_sensor": "shadow",
                "steering_executor": "shadow",
                "steering_gate": "shadow",
                "steering_ungated_action": "blocked",
            },
        },
    ]
    plan_path = tmp_path / "activation_plan.json"
    deployment_contract = {
        "model_id": "frozen-model",
        "model_weights_sha256": _MODEL_WEIGHTS_SHA256,
        "steering_layer_index": 20,
        "activation_width": 2,
        "substrate_max_length": 768,
        "generation_max_new_tokens": 16,
        "generation_temperature": 0.0,
        "fail_on_truncation": True,
    }
    modification_gate_path = tmp_path / "modification_gate_review.json"
    _write_json(
        modification_gate_path,
        {
            "schema_version": "steering-modification-gate-review.v1",
            "preregistration_sha256": "c" * 64,
            "c3_preregistration_sha256": _C3_PREREGISTRATION_SHA256,
            "proposal_target": "substrate.steering_artifact_bundle",
            "desired_gate": "offline",
            "old_value_hash": "d" * 64,
            "new_value_hash": bundle_sha256,
            "validation_delta": 0.2,
            "capacity_cost": 0.0,
            "rollback_evidence": "b3:fixture:checkpoint-round-trip",
            "contract_integrity": 1.0,
            "rollback_resilience": 1.0,
            "fallback_reliance": 0.0,
            "audit_required": False,
            "audit_evidence_id": None,
            "decision": "allow",
            "blocking_reasons": [],
            "description": "Fixture OFFLINE gate review.",
        },
    )
    _write_json(
        plan_path,
        {
            "schema_version": "steering-activation-plan.v3",
            "eligible_prefix": ["steering_sensor", "steering_executor"],
            "steps": steps,
            "rollback_steps": rollback_steps,
            "rollback_order": [
                "steering_gate",
                "steering_executor",
                "steering_sensor",
            ],
            "candidate_bundle": {
                "path": str(bundle_path),
                "sha256": bundle_sha256,
            },
            "modification_gate": {
                "review_sha256": _sha256(modification_gate_path),
                "decision": "allow",
                "blocking_reasons": [],
            },
            "canary_receipt_policy": {
                "schema_version": "steering-activation-canary-receipt.v1",
                "step_1_previous_receipt": "forbidden",
                "step_n_previous_receipt": (
                    "required exact healthy receipt for immediately "
                    "preceding step"
                ),
                "health_endpoint": "/v1/health",
                "health_host": "127.0.0.1",
                "health_status": "ok",
                "health_vertical": "companion",
                "health_persistence_scope_sha256": None,
                "endpoint_must_be_unoccupied": True,
                "intentional_shutdown_exit_code_recorded": True,
                "exact_service_argv_recorded": True,
                "stdout_stderr_paths_and_hashes_recorded": True,
                "previous_receipt_path_and_hash_recorded": True,
                "exclusive_mps_lock": True,
                "production_default_changed": False,
            },
            "deployment_contract": deployment_contract,
            "production_default_changed": False,
            "description": "Authorization plan only.",
        },
    )
    evidence_path = tmp_path / "promotion_evidence.json"
    _write_json(
        evidence_path,
        {
            "schema_version": "steering-promotion-evidence.v1",
            "preregistration_sha256": "c" * 64,
            "c3_preregistration_sha256": _C3_PREREGISTRATION_SHA256,
            "bundle_sha256": "d" * 64,
            "candidate_gate_artifact": asdict(bundle.gate),
            "free_bias_present": False,
            "zero_code_strict_noop": True,
            "raw_text_retained": False,
            "evaluation_writeback_allowed": False,
            "production_default_changed": False,
        },
    )
    report_path = tmp_path / "promotion_report.json"
    _write_json(
        report_path,
        {
            "eligible_prefix": ["steering_sensor", "steering_executor"],
            "sensor_executor_active_authorized": True,
            "gate_active_authorized": False,
            "activation_order": [
                "steering_sensor",
                "steering_executor",
                "steering_gate",
            ],
            "rollback_order": [
                "steering_gate",
                "steering_executor",
                "steering_sensor",
            ],
            "modification_gate_decision": "allow",
            "modification_gate_reasons": [],
            "blocking_reasons": [],
        },
    )
    manifest_path = tmp_path / "artifact_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "steering-promotion-formal-manifest.v2",
            "completed": True,
            "preregistration_sha256": "c" * 64,
            "c3_preregistration_sha256": _C3_PREREGISTRATION_SHA256,
            "eligible_prefix": ["steering_sensor", "steering_executor"],
            "sensor_executor_active_authorized": True,
            "gate_active_authorized": False,
            "modification_gate_decision": "allow",
            "modification_gate_reasons": [],
            "modification_gate_audit_required": False,
            "production_default_changed": False,
            "candidate_bundle_id": bundle.bundle_id,
            "deployment_contract": deployment_contract,
            "canary_receipt_policy": {
                "schema_version": "steering-activation-canary-receipt.v1",
                "step_1_previous_receipt": "forbidden",
                "step_n_previous_receipt": (
                    "required exact healthy receipt for immediately "
                    "preceding step"
                ),
                "health_endpoint": "/v1/health",
                "health_host": "127.0.0.1",
                "health_status": "ok",
                "health_vertical": "companion",
                "health_persistence_scope_sha256": None,
                "endpoint_must_be_unoccupied": True,
                "intentional_shutdown_exit_code_recorded": True,
                "exact_service_argv_recorded": True,
                "stdout_stderr_paths_and_hashes_recorded": True,
                "previous_receipt_path_and_hash_recorded": True,
                "exclusive_mps_lock": True,
                "production_default_changed": False,
            },
            "source_sha256": _activation_source_sha256(),
            "promotion_evidence_sha256": _sha256(evidence_path),
            "modification_gate_review_sha256": _sha256(
                modification_gate_path
            ),
            "promotion_report_sha256": _sha256(report_path),
            "activation_plan_sha256": _sha256(plan_path),
            "candidate_steering_artifact_bundle_sha256": bundle_sha256,
            "blocking_reasons": [],
        },
    )
    step_1_receipt_path = tmp_path / "step-1-canary-receipt.json"
    _write_json(step_1_receipt_path, {"fixture": "prior healthy receipt"})
    stdout_path = tmp_path / "step-2-canary-receipt.stdout.log"
    stderr_path = tmp_path / "step-2-canary-receipt.stderr.log"
    stdout_path.write_bytes(b"fixture stdout\n")
    stderr_path.write_bytes(b"fixture stderr\n")
    service_command = _canary_command(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        plan_path=plan_path,
        step=2,
        previous_receipt=step_1_receipt_path,
    )
    _write_json(
        tmp_path / "step-2-canary-receipt.json",
        {
            "schema_version": "steering-activation-canary-receipt.v1",
            "completed": True,
            "completed_at": "2026-08-05T00:00:00+00:00",
            "completed_rollout_step": 2,
            "single_field_flip": steps[1]["single_field_flip"],
            "rollout_values": steps[1]["rollout_values_after_flip"],
            "manifest_sha256": _sha256(manifest_path),
            "activation_plan_sha256": _sha256(plan_path),
            "modification_gate_review_sha256": _sha256(
                modification_gate_path
            ),
            "candidate_bundle_sha256": bundle_sha256,
            "candidate_bundle_id": bundle.bundle_id,
            "eligible_prefix": ["steering_sensor", "steering_executor"],
            "previous_receipt_path": str(step_1_receipt_path.resolve()),
            "previous_receipt_sha256": _sha256(step_1_receipt_path),
            "canary_health": {
                "status": "ok",
                "session_count": 0,
                "vertical": "companion",
                "persistence_scope_sha256": None,
            },
            "service_pid": 1234,
            "service_exit_code": -15,
            "service_command": list(service_command),
            "service_command_sha256": _command_sha256(service_command),
            "stdout_log_path": str(stdout_path.resolve()),
            "stdout_sha256": _sha256(stdout_path),
            "stderr_log_path": str(stderr_path.resolve()),
            "stderr_sha256": _sha256(stderr_path),
            "intentional_shutdown_after_health_check": True,
            "production_default_changed": False,
            "description": "Fixture completed canary.",
        },
    )
    return bundle, bundle_path, manifest_path, plan_path, bundle_sha256


def test_b3_authorization_materializes_exact_sensor_executor_step(
    tmp_path: Path,
) -> None:
    bundle, _, manifest_path, plan_path, bundle_sha256 = _artifacts(tmp_path)

    authorization = load_steering_activation_authorization(
        bundle=bundle,
        bundle_sha256=bundle_sha256,
        promotion_manifest=manifest_path,
        activation_plan=plan_path,
        rollout_step=3,
        substrate_model_id="frozen-model",
        substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
        substrate_layer_indices=(20,),
        substrate_activation_width=2,
        substrate_max_length=768,
        previous_activation_receipt=(
            tmp_path / "step-2-canary-receipt.json"
        ),
    )

    rollout = authorization.rollout_config
    assert authorization.eligible_prefix == (
        "steering_sensor",
        "steering_executor",
    )
    assert rollout.steering_sensor is WiringLevel.ACTIVE
    assert rollout.steering_executor is WiringLevel.ACTIVE
    assert rollout.steering_gate is WiringLevel.SHADOW
    assert rollout.steering_ungated_action == "always_on"
    assert authorization.applied_field == "steering_executor"
    assert authorization.previous_receipt_sha256 == _sha256(
        tmp_path / "step-2-canary-receipt.json"
    )


def test_b3_authorization_rejects_rollout_jump_without_previous_receipt(
    tmp_path: Path,
) -> None:
    bundle, _, manifest_path, plan_path, bundle_sha256 = _artifacts(tmp_path)

    with pytest.raises(ValueError, match="immediately preceding healthy"):
        load_steering_activation_authorization(
            bundle=bundle,
            bundle_sha256=bundle_sha256,
            promotion_manifest=manifest_path,
            activation_plan=plan_path,
            rollout_step=3,
            substrate_model_id="frozen-model",
            substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
            substrate_layer_indices=(20,),
            substrate_activation_width=2,
            substrate_max_length=768,
        )


def test_b3_authorization_rejects_a_malformed_previous_canary_receipt(
    tmp_path: Path,
) -> None:
    bundle, _, manifest_path, plan_path, bundle_sha256 = _artifacts(tmp_path)
    receipt_path = tmp_path / "step-2-canary-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["canary_health"]["session_count"] = True
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="previous activation receipt is invalid"):
        load_steering_activation_authorization(
            bundle=bundle,
            bundle_sha256=bundle_sha256,
            promotion_manifest=manifest_path,
            activation_plan=plan_path,
            rollout_step=3,
            substrate_model_id="frozen-model",
            substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
            substrate_layer_indices=(20,),
            substrate_activation_width=2,
            substrate_max_length=768,
            previous_activation_receipt=receipt_path,
        )


def test_b3_authorization_rejects_previous_canary_log_drift(
    tmp_path: Path,
) -> None:
    bundle, _, manifest_path, plan_path, bundle_sha256 = _artifacts(tmp_path)
    receipt_path = tmp_path / "step-2-canary-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    Path(receipt["stdout_log_path"]).write_bytes(b"tampered stdout\n")

    with pytest.raises(ValueError, match="previous activation receipt is invalid"):
        load_steering_activation_authorization(
            bundle=bundle,
            bundle_sha256=bundle_sha256,
            promotion_manifest=manifest_path,
            activation_plan=plan_path,
            rollout_step=3,
            substrate_model_id="frozen-model",
            substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
            substrate_layer_indices=(20,),
            substrate_activation_width=2,
            substrate_max_length=768,
            previous_activation_receipt=receipt_path,
        )


def test_b3_canary_receipt_is_immutable_and_chains_exact_authorization(
    tmp_path: Path,
) -> None:
    bundle, _, manifest_path, plan_path, bundle_sha256 = _artifacts(tmp_path)
    authorization = load_steering_activation_authorization(
        bundle=bundle,
        bundle_sha256=bundle_sha256,
        promotion_manifest=manifest_path,
        activation_plan=plan_path,
        rollout_step=3,
        substrate_model_id="frozen-model",
        substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
        substrate_layer_indices=(20,),
        substrate_activation_width=2,
        substrate_max_length=768,
        previous_activation_receipt=(
            tmp_path / "step-2-canary-receipt.json"
        ),
    )
    receipt_path = tmp_path / "step-3-canary-receipt.json"
    stdout_path = receipt_path.with_suffix(".stdout.log")
    stderr_path = receipt_path.with_suffix(".stderr.log")
    stdout_path.write_bytes(b"step 3 stdout\n")
    stderr_path.write_bytes(b"step 3 stderr\n")
    service_command = _canary_command(
        bundle_path=tmp_path / "candidate.json",
        manifest_path=manifest_path,
        plan_path=plan_path,
        step=3,
        previous_receipt=tmp_path / "step-2-canary-receipt.json",
    )
    receipt = build_steering_activation_canary_receipt(
        authorization=authorization,
        canary_health={
            "status": "ok",
            "session_count": 0,
            "vertical": "companion",
            "persistence_scope_sha256": None,
        },
        service_pid=4321,
        service_exit_code=-15,
        service_command=service_command,
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
    )

    write_steering_activation_canary_receipt(
        path=receipt_path,
        receipt=receipt,
    )
    write_steering_activation_canary_receipt(
        path=receipt_path,
        receipt=receipt,
    )

    observed = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert observed["completed_rollout_step"] == 3
    assert observed["single_field_flip"] == {
        "field": "steering_executor",
        "from": "shadow",
        "to": "active",
    }
    assert observed["previous_receipt_sha256"] == _sha256(
        tmp_path / "step-2-canary-receipt.json"
    )
    assert observed["service_command"] == list(service_command)
    assert observed["stdout_sha256"] == _sha256(stdout_path)
    with pytest.raises(ValueError, match="immutable B3 canary receipt differs"):
        write_steering_activation_canary_receipt(
            path=receipt_path,
            receipt=observed | {"service_pid": 9999},
        )


def test_b3_authorization_rejects_unadmitted_step(tmp_path: Path) -> None:
    bundle, _, manifest_path, plan_path, bundle_sha256 = _artifacts(tmp_path)

    with pytest.raises(ValueError, match="not authorized"):
        load_steering_activation_authorization(
            bundle=bundle,
            bundle_sha256=bundle_sha256,
            promotion_manifest=manifest_path,
            activation_plan=plan_path,
            rollout_step=4,
            substrate_model_id="frozen-model",
            substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
            substrate_layer_indices=(20,),
            substrate_activation_width=2,
            substrate_max_length=768,
        )


def test_b3_authorization_rejects_candidate_bundle_hash_drift(
    tmp_path: Path,
) -> None:
    bundle, _, manifest_path, plan_path, _ = _artifacts(tmp_path)

    with pytest.raises(ValueError, match="does not authorize"):
        load_steering_activation_authorization(
            bundle=bundle,
            bundle_sha256="d" * 64,
            promotion_manifest=manifest_path,
            activation_plan=plan_path,
            rollout_step=1,
            substrate_model_id="frozen-model",
            substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
            substrate_layer_indices=(20,),
            substrate_activation_width=2,
            substrate_max_length=768,
        )


def test_b3_authorization_rejects_post_formal_active_source_drift(
    tmp_path: Path,
) -> None:
    bundle, _, manifest_path, plan_path, bundle_sha256 = _artifacts(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_sha256"][
        "packages/lifeform-service/src/lifeform_service/steering_activation.py"
    ] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="activation source drift"):
        load_steering_activation_authorization(
            bundle=bundle,
            bundle_sha256=bundle_sha256,
            promotion_manifest=manifest_path,
            activation_plan=plan_path,
            rollout_step=1,
            substrate_model_id="frozen-model",
            substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
            substrate_layer_indices=(20,),
            substrate_activation_width=2,
            substrate_max_length=768,
        )


def test_b3_authorization_rejects_consistently_hashed_blocked_modification_gate(
    tmp_path: Path,
) -> None:
    bundle, _, manifest_path, plan_path, bundle_sha256 = _artifacts(tmp_path)
    review_path = tmp_path / "modification_gate_review.json"
    report_path = tmp_path / "promotion_report.json"

    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["decision"] = "block"
    review["blocking_reasons"] = [
        "validation_delta 0.010 below required margin 0.050"
    ]
    _write_json(review_path, review)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["modification_gate_decision"] = "block"
    report["modification_gate_reasons"] = review["blocking_reasons"]
    _write_json(report_path, report)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["modification_gate"] = {
        "review_sha256": _sha256(review_path),
        "decision": "block",
        "blocking_reasons": review["blocking_reasons"],
    }
    _write_json(plan_path, plan)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["modification_gate_decision"] = "block"
    manifest["modification_gate_reasons"] = review["blocking_reasons"]
    manifest["modification_gate_review_sha256"] = _sha256(review_path)
    manifest["promotion_report_sha256"] = _sha256(report_path)
    manifest["activation_plan_sha256"] = _sha256(plan_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="does not authorize"):
        load_steering_activation_authorization(
            bundle=bundle,
            bundle_sha256=bundle_sha256,
            promotion_manifest=manifest_path,
            activation_plan=plan_path,
            rollout_step=1,
            substrate_model_id="frozen-model",
            substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
            substrate_layer_indices=(20,),
            substrate_activation_width=2,
            substrate_max_length=768,
        )


def test_b3_authorization_rejects_runtime_budget_drift(tmp_path: Path) -> None:
    bundle, _, manifest_path, plan_path, bundle_sha256 = _artifacts(tmp_path)

    with pytest.raises(ValueError, match="service substrate differs"):
        load_steering_activation_authorization(
            bundle=bundle,
            bundle_sha256=bundle_sha256,
            promotion_manifest=manifest_path,
            activation_plan=plan_path,
            rollout_step=1,
            substrate_model_id="frozen-model",
            substrate_expected_weights_sha256=_MODEL_WEIGHTS_SHA256,
            substrate_layer_indices=(20,),
            substrate_activation_width=2,
            substrate_max_length=1024,
        )


def test_cli_threads_only_the_verified_b3_rollout_to_companion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bundle_path, manifest_path, plan_path, _ = _artifacts(tmp_path)
    captured: dict[str, object] = {}

    def discover(**kwargs):
        captured.update(kwargs)
        return {
            "companion": VerticalSpec(
                name="companion",
                factory=lambda _runtime: None,  # type: ignore[arg-type,return-value]
                has_temporal_bootstrap=False,
                has_regime_bootstrap=False,
            )
        }

    monkeypatch.setattr(cli, "discover_verticals", discover)
    exit_code = cli.main(
        [
            "--list-verticals",
            "--substrate-mode",
            "hf-shared",
            "--substrate-local-files-only",
            "--substrate-model-id",
            "frozen-model",
            "--substrate-expected-weights-sha256",
            _MODEL_WEIGHTS_SHA256,
            "--substrate-layer-indices",
            "20",
            "--substrate-activation-width",
            "2",
            "--substrate-max-length",
            "768",
            "--steering-artifact-bundle",
            str(bundle_path),
            "--steering-promotion-manifest",
            str(manifest_path),
            "--steering-activation-plan",
            str(plan_path),
            "--steering-activation-step",
            "3",
            "--steering-previous-activation-receipt",
            str(tmp_path / "step-2-canary-receipt.json"),
        ]
    )

    assert exit_code == 0
    rollout = captured["companion_steering_rollout_config"]
    assert rollout.steering_sensor is WiringLevel.ACTIVE
    assert rollout.steering_executor is WiringLevel.ACTIVE
    assert rollout.steering_gate is WiringLevel.SHADOW
    assert captured["companion_steering_rollout_max_new_tokens"] == 16
    assert captured["companion_steering_rollout_temperature"] == 0.0


def test_cli_rejects_env_override_of_authorized_b3_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, bundle_path, manifest_path, plan_path, _ = _artifacts(tmp_path)
    monkeypatch.setenv("VZ_STEERING_GATE", "active")

    exit_code = cli.main(
        [
            "--list-verticals",
            "--substrate-mode",
            "hf-shared",
            "--substrate-local-files-only",
            "--substrate-model-id",
            "frozen-model",
            "--substrate-expected-weights-sha256",
            _MODEL_WEIGHTS_SHA256,
            "--substrate-layer-indices",
            "20",
            "--substrate-activation-width",
            "2",
            "--substrate-max-length",
            "768",
            "--steering-artifact-bundle",
            str(bundle_path),
            "--steering-promotion-manifest",
            str(manifest_path),
            "--steering-activation-plan",
            str(plan_path),
            "--steering-activation-step",
            "3",
        ]
    )

    assert exit_code == 1
    assert "forbids VZ_STEERING_* overrides" in capsys.readouterr().err


def test_active_steering_runtime_fails_instead_of_truncating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def build_runtime(**kwargs):
        captured.update(kwargs)
        return type(
            "Runtime",
            (),
            {"model_id": "frozen-model", "runtime_origin": "hf-local"},
        )()

    monkeypatch.setattr(
        "volvence_zero.substrate.build_transformers_runtime_with_fallback",
        build_runtime,
    )
    args = cli._build_parser().parse_args(
        [
            "--substrate-mode",
            "hf-shared",
            "--substrate-model-id",
            "frozen-model",
            "--substrate-max-length",
            "768",
            "--steering-activation-step",
            "1",
        ]
    )

    cli._build_shared_substrate(args)

    assert captured["max_length"] == 768
    assert captured["fail_on_truncation"] is True
