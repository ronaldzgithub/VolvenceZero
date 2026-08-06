#!/usr/bin/env python3
"""B3 steering-only SHADOW -> ACTIVE promotion control plane.

Preregistration binds the C3 preregistration and every threshold before the
real-dialogue result exists.  Formal evaluation later consumes the immutable
C3 trace/report/bundle; it never reruns or substitutes an ETA-off control and
never changes production defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from lifeform_service.steering_activation import (
    steering_activation_canary_receipt_policy,
)
from volvence_zero.agent.dialogue_steering_evidence import (
    DialogueSteeringReport,
    DialogueSteeringTraceDataset,
)
from volvence_zero.agent.steering_promotion_gate import (
    STEERING_MODIFICATION_TARGET,
    STEERING_PROMOTION_ORDER,
    SteeringPromotionThresholds,
    build_steering_modification_gate_review,
    build_steering_promotion_evidence,
    evaluate_steering_promotion,
)
from volvence_zero.steering_contracts import SteeringArtifactBundle


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PREREG_SCHEMA = "steering-promotion-formal-prereg.v2"
SOURCE_FILES = (
    "packages/lifeform-expression/src/lifeform_expression/llm_synthesizer.py",
    "packages/lifeform-service/src/lifeform_service/app.py",
    "packages/lifeform-service/src/lifeform_service/cli.py",
    "packages/lifeform-service/src/lifeform_service/steering_activation.py",
    "packages/lifeform-service/src/lifeform_service/verticals.py",
    "packages/vz-contracts/src/volvence_zero/runtime/kernel.py",
    "packages/vz-contracts/src/volvence_zero/steering_contracts.py",
    "packages/vz-cognition/src/volvence_zero/credit/gate.py",
    "packages/vz-cognition/src/volvence_zero/evaluation/__init__.py",
    "packages/vz-cognition/src/volvence_zero/evaluation/types.py",
    "packages/vz-cognition/src/volvence_zero/steering_sensor.py",
    "packages/vz-runtime/src/volvence_zero/agent/dialogue_steering_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/response.py",
    "packages/vz-runtime/src/volvence_zero/agent/steering_promotion_gate.py",
    "packages/vz-runtime/src/volvence_zero/agent/session.py",
    "packages/vz-runtime/src/volvence_zero/brain.py",
    "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_contracts.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_interfaces.py",
    "packages/vz-substrate/src/volvence_zero/steering_executor.py",
    "packages/vz-temporal/src/volvence_zero/steering_gate.py",
    "scripts/run_dialogue_steering_test_plan.py",
    "scripts/run_steering_promotion_test_plan.py",
    "scripts/verify_steering_activation_canary.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: object) -> None:
    _write_immutable(path, _canonical_bytes(payload))


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _canary_receipt_policy() -> dict[str, object]:
    return steering_activation_canary_receipt_policy()


def _c3_preregistration(path: Path) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "dialogue-steering-formal-prereg.v1"
    ):
        raise ValueError("B3 requires the immutable C3 formal preregistration")
    return payload, _sha256(path)


def _configuration(args: argparse.Namespace) -> dict[str, object]:
    c3_preregistration, c3_prereg_sha = _c3_preregistration(
        args.c3_preregistration.resolve()
    )
    c3_configuration = c3_preregistration.get("run_configuration")
    if not isinstance(c3_configuration, dict):
        raise ValueError("B3 requires the C3 frozen run configuration")
    deployment_contract = {
        "model_id": c3_configuration.get("model_id"),
        "model_weights_sha256": c3_configuration.get(
            "model_weights_sha256"
        ),
        "steering_layer_index": c3_configuration.get(
            "steering_layer_index"
        ),
        "activation_width": c3_configuration.get("activation_width"),
        "substrate_max_length": c3_configuration.get("max_length"),
        "generation_max_new_tokens": c3_configuration.get(
            "runtime_max_new_tokens"
        ),
        "generation_temperature": 0.0,
        "fail_on_truncation": True,
    }
    if (
        not isinstance(deployment_contract["model_id"], str)
        or not deployment_contract["model_id"]
        or not _valid_sha256(deployment_contract["model_weights_sha256"])
        or not isinstance(deployment_contract["steering_layer_index"], int)
        or isinstance(deployment_contract["steering_layer_index"], bool)
        or deployment_contract["steering_layer_index"] < 0
        or not isinstance(deployment_contract["activation_width"], int)
        or isinstance(deployment_contract["activation_width"], bool)
        or deployment_contract["activation_width"] < 1
        or not isinstance(deployment_contract["substrate_max_length"], int)
        or isinstance(deployment_contract["substrate_max_length"], bool)
        or deployment_contract["substrate_max_length"] < 1
        or not isinstance(deployment_contract["generation_max_new_tokens"], int)
        or isinstance(deployment_contract["generation_max_new_tokens"], bool)
        or deployment_contract["generation_max_new_tokens"] < 16
    ):
        raise ValueError("B3 C3 deployment contract is incomplete")
    return {
        "schema_version": "steering-promotion-run-configuration.v2",
        "c3_preregistration_path": str(args.c3_preregistration.resolve()),
        "c3_preregistration_sha256": c3_prereg_sha,
        "c3_output": str(args.c3_output.resolve()),
        "thresholds": asdict(SteeringPromotionThresholds()),
        "bootstrap_resamples": args.bootstrap_resamples,
        "validation_axes": [
            "normalized_n_plus_one_mse",
            "n_plus_one_cosine_error",
        ],
        "gate_off_controls": ["noop", "always_on"],
        "sensor_off_control": "matched-budget-unconditional-operator",
        "promotion_order": [item.value for item in STEERING_PROMOTION_ORDER],
        "candidate_gate_selection": (
            "minimum C3 training-side selection loss; deterministic seed tie-break"
        ),
        "activation_policy": (
            "one-field-per-rollout including explicit gate-off control preparation"
        ),
        "canary_receipt_policy": _canary_receipt_policy(),
        "modification_gate": {
            "proposal_target": STEERING_MODIFICATION_TARGET,
            "desired_gate": "offline",
            "validation_delta": (
                "minimum relative improvement over informative held-out axes"
            ),
            "capacity_cost": 0.0,
            "rollback_evidence": (
                "candidate-bound steering gate checkpoint JSON round-trip"
            ),
            "audit_required": False,
            "audit_phase": (
                "OA-4 business audit content pending; phase-1 contract is explicit"
            ),
        },
        "deployment_contract": deployment_contract,
        "source_sha256": _source_hashes(),
    }


def _load_preregistration(
    path: Path, *, expected_configuration: dict[str, object]
) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PREREG_SCHEMA:
        raise ValueError("B3 preregistration schema is invalid")
    if payload.get("run_configuration") != expected_configuration:
        raise ValueError("B3 run configuration drifted from preregistration")
    return payload, _sha256(path)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / "artifact_manifest.json",
        "bundle": root / "steering_artifact_bundle.json",
        "trace": root / "dialogue_steering_trace.json.gz",
        "report": root / "report.json",
    }


def _load_c3_artifacts(
    *, root: Path, expected_preregistration_sha256: str
) -> tuple[
    dict[str, object],
    SteeringArtifactBundle,
    DialogueSteeringTraceDataset,
    DialogueSteeringReport,
    dict[str, str],
]:
    paths = _paths(root)
    missing = tuple(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"B3 C3 artifact set is incomplete: {missing!r}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "dialogue-steering-formal-manifest.v1"
        or manifest.get("completed") is not True
        or manifest.get("formal_claim_allowed") is not True
        or manifest.get("preregistration_sha256")
        != expected_preregistration_sha256
        or manifest.get("raw_text_retained") is not False
        or manifest.get("evaluation_writeback_allowed") is not False
    ):
        raise ValueError("B3 rejected the C3 formal manifest")
    hashes = {
        name: _sha256(paths[name]) for name in ("bundle", "trace", "report")
    }
    for name in hashes:
        if manifest.get(f"{name}_sha256") != hashes[name]:
            raise ValueError(f"B3 C3 {name} SHA-256 drift")
    bundle = SteeringArtifactBundle.from_json(
        paths["bundle"].read_text(encoding="utf-8")
    )
    trace = DialogueSteeringTraceDataset.from_json(
        gzip.decompress(paths["trace"].read_bytes()).decode("utf-8")
    )
    report = DialogueSteeringReport.from_json(
        paths["report"].read_text(encoding="utf-8")
    )
    if (
        trace.bundle_id != bundle.bundle_id
        or report.preregistration_sha256 != expected_preregistration_sha256
        or manifest.get("admitted") is not report.admission.admitted
    ):
        raise ValueError("B3 C3 bundle/trace/report lineage drift")
    sensor_off_executor = bundle.sensor_off_executor
    if sensor_off_executor is None:
        raise ValueError("B3 C3 bundle lacks the preregistered sensor-off control")
    if any(
        artifact.source_preregistration_sha256
        != expected_preregistration_sha256
        for artifact in (
            bundle.reader,
            bundle.executor,
            bundle.gate,
            sensor_off_executor,
            *(point.selected_gate_artifact for point in report.seed_points),
        )
    ):
        raise ValueError("B3 C3 artifact preregistration lineage drift")
    if any(
        row.reader_artifact_id != bundle.reader.artifact_id
        or row.executor_artifact_id != bundle.executor.artifact_id
        or row.sensor_off_executor_artifact_id
        != sensor_off_executor.artifact_id
        for row in (*trace.train_rows, *trace.validation_rows)
    ):
        raise ValueError("B3 C3 trace artifact ids drifted from the bundle")
    return manifest, bundle, trace, report, hashes


def _activation_plan(
    verdict,
    *,
    candidate_bundle_path: Path,
    candidate_bundle_sha256: str,
    modification_gate_review_sha256: str,
    deployment_contract: dict[str, object],
) -> dict[str, object]:
    prefix = tuple(item.value for item in verdict.eligible_prefix)
    state = {
        "steering_sensor": "shadow",
        "steering_executor": "shadow",
        "steering_gate": "shadow",
        "steering_ungated_action": "blocked",
    }
    steps: list[dict[str, object]] = []

    def append_step(*, field: str, value: str, purpose: str) -> None:
        before = state[field]
        if before == value:
            return
        state[field] = value
        steps.append(
            {
                "order": len(steps) + 1,
                "purpose": purpose,
                "single_field_flip": {
                    "field": field,
                    "from": before,
                    "to": value,
                },
                "rollout_values_after_flip": dict(state),
            }
        )

    if len(verdict.eligible_prefix) >= 1:
        append_step(
            field="steering_sensor",
            value="active",
            purpose="activate the first authorized owner",
        )
    if len(verdict.eligible_prefix) >= 2:
        append_step(
            field="steering_ungated_action",
            value="always_on",
            purpose="prepare the explicit gate-off arm while executor is SHADOW",
        )
        append_step(
            field="steering_executor",
            value="active",
            purpose="activate the second authorized owner",
        )
    if len(verdict.eligible_prefix) >= 3:
        append_step(
            field="steering_gate",
            value="active",
            purpose="activate the learned gate after sensor and executor",
        )
        append_step(
            field="steering_ungated_action",
            value="blocked",
            purpose="remove the now-inert gate-off override",
        )

    rollback_state = dict(state)
    rollback_steps = []
    for step in reversed(steps):
        flip = step["single_field_flip"]
        if not isinstance(flip, dict):  # pragma: no cover - local construction
            raise RuntimeError("activation plan flip shape drift")
        field = str(flip["field"])
        before = str(flip["from"])
        after = rollback_state[field]
        rollback_state[field] = before
        rollback_steps.append(
            {
                "order": len(rollback_steps) + 1,
                "single_field_flip": {
                    "field": field,
                    "from": after,
                    "to": before,
                },
                "rollout_values_after_flip": dict(rollback_state),
            }
        )
    return {
        "schema_version": "steering-activation-plan.v3",
        "eligible_prefix": prefix,
        "steps": steps,
        "rollback_steps": rollback_steps,
        "rollback_order": tuple(item.value for item in verdict.rollback_order),
        "candidate_bundle": {
            "path": str(candidate_bundle_path.resolve()),
            "sha256": candidate_bundle_sha256,
        },
        "modification_gate": {
            "review_sha256": modification_gate_review_sha256,
            "decision": verdict.modification_gate_decision.value,
            "blocking_reasons": verdict.modification_gate_reasons,
        },
        "canary_receipt_policy": _canary_receipt_policy(),
        "deployment_contract": deployment_contract,
        "production_default_changed": False,
        "description": (
            "Authorization plan only: apply and verify exactly one field per rollout."
        ),
    }


def _preregister(args: argparse.Namespace) -> int:
    configuration = _configuration(args)
    target = args.preregistration.resolve()
    if target.exists():
        _load_preregistration(target, expected_configuration=configuration)
    else:
        observed_c3_artifacts = tuple(
            name
            for name, path in _paths(args.c3_output.resolve()).items()
            if path.exists()
        )
        if observed_c3_artifacts:
            raise ValueError(
                "B3 preregistration must precede every C3 formal artifact; "
                f"found {observed_c3_artifacts!r}"
            )
        _write_json(
            target,
            {
                "schema_version": PREREG_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "claim": (
                    "Steering owners may leave SHADOW only through independent "
                    "real-trace, validation, gate-off, sensor-off, rollback, "
                    "latency, safety, R12, and ModificationGate.OFFLINE gates."
                ),
                "run_configuration": configuration,
                "forbidden_substitutions": (
                    "learned-active-eta-off-gate",
                    "evaluation-or-judge-writeback",
                    "missing-ablation-treated-as-pass",
                    "multi-component-default-flip",
                    "rare-heavy-artifact-without-modification-gate",
                    "rollout-step-jump-without-healthy-receipt",
                    "post-result-threshold-change",
                ),
            },
        )
    print(json.dumps({"preregistration": str(target), "sha256": _sha256(target)}, indent=2))
    return 0


def _preflight(args: argparse.Namespace) -> int:
    configuration = _configuration(args)
    prereg_exists = args.preregistration.resolve().is_file()
    if prereg_exists:
        _load_preregistration(
            args.preregistration.resolve(), expected_configuration=configuration
        )
    paths = _paths(args.c3_output.resolve())
    artifact_presence = {name: path.is_file() for name, path in paths.items()}
    c3_artifacts_valid = False
    if all(artifact_presence.values()):
        _load_c3_artifacts(
            root=args.c3_output.resolve(),
            expected_preregistration_sha256=str(
                configuration["c3_preregistration_sha256"]
            ),
        )
        c3_artifacts_valid = True
    passed = prereg_exists and c3_artifacts_valid
    result = {
        "schema_version": "steering-promotion-preflight.v1",
        "passed": passed,
        "preregistration_exists": prereg_exists,
        "c3_artifacts_present": artifact_presence,
        "c3_artifacts_valid": c3_artifacts_valid,
        "run_configuration": configuration,
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


def _completed_result(args: argparse.Namespace, *, prereg_sha: str) -> dict[str, object] | None:
    manifest_path = args.output.resolve() / "artifact_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "steering-promotion-formal-manifest.v2"
        or manifest.get("completed") is not True
        or manifest.get("preregistration_sha256") != prereg_sha
    ):
        raise ValueError("existing B3 manifest is invalid or belongs to another prereg")
    for name in (
        "promotion_evidence",
        "modification_gate_review",
        "promotion_report",
        "activation_plan",
        "candidate_steering_artifact_bundle",
    ):
        path = args.output.resolve() / f"{name}.json"
        if not path.is_file() or manifest.get(f"{name}_sha256") != _sha256(path):
            raise ValueError(f"existing B3 {name} artifact drift")
    return manifest


def _formal(args: argparse.Namespace) -> int:
    configuration = _configuration(args)
    _, prereg_sha = _load_preregistration(
        args.preregistration.resolve(), expected_configuration=configuration
    )
    completed = _completed_result(args, prereg_sha=prereg_sha)
    if completed is not None:
        print(json.dumps(completed, indent=2))
        return 0
    c3_prereg_sha = str(configuration["c3_preregistration_sha256"])
    _, bundle, dataset, report, c3_hashes = _load_c3_artifacts(
        root=args.c3_output.resolve(),
        expected_preregistration_sha256=c3_prereg_sha,
    )
    evidence = build_steering_promotion_evidence(
        dataset=dataset,
        c3_report=report,
        preregistration_sha256=prereg_sha,
        c3_report_sha256=c3_hashes["report"],
        trace_sha256=c3_hashes["trace"],
        bundle_sha256=c3_hashes["bundle"],
        thresholds=SteeringPromotionThresholds(),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    output = args.output.resolve()
    evidence_path = output / "promotion_evidence.json"
    modification_gate_path = output / "modification_gate_review.json"
    report_path = output / "promotion_report.json"
    activation_path = output / "activation_plan.json"
    candidate_bundle_path = output / "candidate_steering_artifact_bundle.json"
    gate_digest = hashlib.sha256(
        _canonical_bytes(asdict(evidence.candidate_gate_artifact))
    ).hexdigest()
    candidate_bundle = replace(
        bundle,
        bundle_id=f"{bundle.bundle_id}:c3-gate:{gate_digest[:16]}",
        gate=evidence.candidate_gate_artifact,
        description=(
            "B3-frozen candidate bundle; each ACTIVE owner remains separately "
            "gated by the ordered promotion verdict."
        ),
    )
    _write_immutable(
        candidate_bundle_path,
        (candidate_bundle.to_json() + "\n").encode("utf-8"),
    )
    candidate_bundle_sha256 = _sha256(candidate_bundle_path)
    modification_gate_review = build_steering_modification_gate_review(
        evidence=evidence,
        candidate_bundle_sha256=candidate_bundle_sha256,
    )
    verdict = evaluate_steering_promotion(
        evidence,
        modification_gate_review=modification_gate_review,
        thresholds=SteeringPromotionThresholds(),
    )
    _write_json(evidence_path, asdict(evidence))
    _write_json(modification_gate_path, asdict(modification_gate_review))
    _write_json(report_path, asdict(verdict))
    modification_gate_sha256 = _sha256(modification_gate_path)
    _write_json(
        activation_path,
        _activation_plan(
            verdict,
            candidate_bundle_path=candidate_bundle_path,
            candidate_bundle_sha256=candidate_bundle_sha256,
            modification_gate_review_sha256=modification_gate_sha256,
            deployment_contract=dict(configuration["deployment_contract"]),
        ),
    )
    manifest = {
        "schema_version": "steering-promotion-formal-manifest.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "preregistration_sha256": prereg_sha,
        "c3_preregistration_sha256": c3_prereg_sha,
        "c3_admitted": report.admission.admitted,
        "eligible_prefix": tuple(item.value for item in verdict.eligible_prefix),
        "sensor_executor_active_authorized": (
            verdict.sensor_executor_active_authorized
        ),
        "gate_active_authorized": verdict.gate_active_authorized,
        "modification_gate_decision": (
            verdict.modification_gate_decision.value
        ),
        "modification_gate_reasons": verdict.modification_gate_reasons,
        "modification_gate_audit_required": (
            modification_gate_review.audit_required
        ),
        "production_default_changed": False,
        "bundle_id": bundle.bundle_id,
        "candidate_bundle_id": candidate_bundle.bundle_id,
        "deployment_contract": configuration["deployment_contract"],
        "canary_receipt_policy": configuration["canary_receipt_policy"],
        "source_sha256": configuration["source_sha256"],
        "promotion_evidence_sha256": _sha256(evidence_path),
        "modification_gate_review_sha256": modification_gate_sha256,
        "promotion_report_sha256": _sha256(report_path),
        "activation_plan_sha256": _sha256(activation_path),
        "candidate_steering_artifact_bundle_sha256": (
            candidate_bundle_sha256
        ),
        "blocking_reasons": verdict.blocking_reasons,
    }
    _write_json(output / "artifact_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def _status(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    print(
        json.dumps(
            {
                "preregistration_exists": args.preregistration.resolve().is_file(),
                "c3_artifacts": {
                    name: path.is_file()
                    for name, path in _paths(args.c3_output.resolve()).items()
                },
                "promotion_evidence_exists": (
                    output / "promotion_evidence.json"
                ).is_file(),
                "modification_gate_review_exists": (
                    output / "modification_gate_review.json"
                ).is_file(),
                "promotion_report_exists": (
                    output / "promotion_report.json"
                ).is_file(),
                "manifest_exists": (output / "artifact_manifest.json").is_file(),
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("status", "preflight", "preregister", "formal"))
    parser.add_argument("--c3-preregistration", type=Path, required=True)
    parser.add_argument("--c3-output", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_resamples < 100:
        raise ValueError("B3 bootstrap_resamples must be at least 100")
    if args.stage == "status":
        return _status(args)
    if args.stage == "preflight":
        return _preflight(args)
    if args.stage == "preregister":
        return _preregister(args)
    return _formal(args)


if __name__ == "__main__":
    raise SystemExit(main())
