from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from lifeform_service import VerticalSpec
from lifeform_service import cli
from lifeform_service.steering_activation import (
    load_steering_activation_authorization,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


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
    _write_json(
        plan_path,
        {
            "schema_version": "steering-activation-plan.v2",
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
            "blocking_reasons": [],
        },
    )
    manifest_path = tmp_path / "artifact_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "steering-promotion-formal-manifest.v1",
            "completed": True,
            "preregistration_sha256": "c" * 64,
            "c3_preregistration_sha256": _C3_PREREGISTRATION_SHA256,
            "eligible_prefix": ["steering_sensor", "steering_executor"],
            "sensor_executor_active_authorized": True,
            "gate_active_authorized": False,
            "production_default_changed": False,
            "candidate_bundle_id": bundle.bundle_id,
            "deployment_contract": deployment_contract,
            "promotion_evidence_sha256": _sha256(evidence_path),
            "promotion_report_sha256": _sha256(report_path),
            "activation_plan_sha256": _sha256(plan_path),
            "candidate_steering_artifact_bundle_sha256": bundle_sha256,
            "blocking_reasons": [],
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
