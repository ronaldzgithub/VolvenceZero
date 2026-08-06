"""Fail-closed consumption of a B3 steering activation authorization.

The promotion runner writes evidence, a verdict, a candidate artifact bundle,
and an ordered one-field rollout plan.  This module is the deployment-side
reader for those artifacts.  Loading a bundle alone never authorizes ACTIVE
wiring; the exact B3 manifest and activation plan must bind its SHA-256, and a
caller must select one concrete rollout step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel
from volvence_zero.steering_contracts import SteeringArtifactBundle


_ORDER = (
    "steering_sensor",
    "steering_executor",
    "steering_gate",
)
_INITIAL_STATE = {
    "steering_sensor": "shadow",
    "steering_executor": "shadow",
    "steering_gate": "shadow",
    "steering_ungated_action": "blocked",
}
_REQUIRED_ACTIVATION_SOURCE_PATHS = (
    "packages/lifeform-service/src/lifeform_service/cli.py",
    "packages/lifeform-service/src/lifeform_service/steering_activation.py",
    "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
    "scripts/verify_steering_activation_canary.py",
)
STEERING_ACTIVATION_CANARY_RECEIPT_SCHEMA = (
    "steering-activation-canary-receipt.v1"
)


def steering_activation_canary_receipt_policy() -> dict[str, object]:
    """Publish the single JSON policy shared by B3 and deployment."""

    return {
        "schema_version": STEERING_ACTIVATION_CANARY_RECEIPT_SCHEMA,
        "step_1_previous_receipt": "forbidden",
        "step_n_previous_receipt": (
            "required exact healthy receipt for immediately preceding step"
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
    }


_CANARY_RECEIPT_POLICY = steering_activation_canary_receipt_policy()


@dataclass(frozen=True)
class SteeringActivationAuthorization:
    manifest_path: str
    manifest_sha256: str
    activation_plan_path: str
    activation_plan_sha256: str
    modification_gate_review_path: str
    modification_gate_review_sha256: str
    candidate_bundle_sha256: str
    candidate_bundle_id: str
    eligible_prefix: tuple[str, ...]
    rollout_step: int
    applied_field: str
    applied_from: str
    applied_to: str
    previous_receipt_path: str | None
    previous_receipt_sha256: str | None
    rollout_config: FinalRolloutConfig
    substrate_max_length: int
    generation_max_new_tokens: int
    generation_temperature: float
    fail_on_truncation: bool
    description: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _service_command_sha256(command: tuple[str, ...]) -> str:
    payload = (
        json.dumps(
            list(command),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _command_option(command: tuple[str, ...], option: str) -> str | None:
    positions = tuple(
        index for index, value in enumerate(command) if value == option
    )
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"B3 canary command option is ambiguous: {option}")
    return command[positions[0] + 1]


def _repository_root() -> Path:
    source = Path(__file__).resolve()
    relative = Path(
        "packages/lifeform-service/src/lifeform_service/steering_activation.py"
    )
    for parent in source.parents:
        if (parent / relative).resolve() == source:
            return parent
    raise RuntimeError("B3 activation source is not inside a repository tree")


def _validate_activation_source_snapshot(payload: object) -> None:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("B3 activation source snapshot is missing")
    if not set(_REQUIRED_ACTIVATION_SOURCE_PATHS).issubset(payload):
        raise ValueError("B3 activation source snapshot omits the ACTIVE chain")
    repository_root = _repository_root()
    for raw_name, expected_sha256 in payload.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("B3 activation source path is invalid")
        relative = Path(raw_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("B3 activation source path escapes the repository")
        if not _valid_sha256(expected_sha256):
            raise ValueError("B3 activation source SHA-256 is invalid")
        source_path = (repository_root / relative).resolve()
        if not source_path.is_relative_to(repository_root) or not source_path.is_file():
            raise ValueError(f"B3 activation source file is missing: {raw_name}")
        if _sha256(source_path) != expected_sha256:
            raise ValueError(f"B3 activation source drift: {raw_name}")


def _steering_state(config: FinalRolloutConfig) -> dict[str, str]:
    return {
        "steering_sensor": config.steering_sensor.value,
        "steering_executor": config.steering_executor.value,
        "steering_gate": config.steering_gate.value,
        "steering_ungated_action": config.steering_ungated_action,
    }


def _config_with_steering_state(
    config: FinalRolloutConfig,
    state: dict[str, str],
) -> FinalRolloutConfig:
    if set(state) != set(_INITIAL_STATE):
        raise ValueError("steering rollout state fields drifted")
    return replace(
        config,
        steering_sensor=WiringLevel(state["steering_sensor"]),
        steering_executor=WiringLevel(state["steering_executor"]),
        steering_gate=WiringLevel(state["steering_gate"]),
        steering_ungated_action=state["steering_ungated_action"],
    )


def _validate_previous_receipt(
    *,
    path: Path,
    expected_step: int,
    expected_state: dict[str, str],
    expected_single_field_flip: dict[str, object],
    expected_manifest_sha256: str,
    expected_plan_sha256: str,
    expected_modification_gate_sha256: str,
    expected_bundle_sha256: str,
    expected_bundle_id: str,
    expected_eligible_prefix: tuple[str, ...],
    expected_manifest_path: Path,
    expected_plan_path: Path,
    expected_bundle_path: Path,
) -> tuple[str, dict[str, object]]:
    receipt_path = path.resolve()
    receipt = _load_object(
        receipt_path,
        label="B3 previous activation canary receipt",
    )
    receipt_sha256 = _sha256(receipt_path)
    health = receipt.get("canary_health")
    previous_sha = receipt.get("previous_receipt_sha256")
    session_count = health.get("session_count") if isinstance(health, dict) else None
    service_pid = receipt.get("service_pid")
    raw_command = receipt.get("service_command")
    command = (
        tuple(raw_command)
        if isinstance(raw_command, list)
        and all(isinstance(value, str) and value for value in raw_command)
        else ()
    )
    raw_previous_path = receipt.get("previous_receipt_path")
    previous_path = (
        Path(raw_previous_path).resolve()
        if isinstance(raw_previous_path, str) and raw_previous_path
        else None
    )
    raw_stdout_path = receipt.get("stdout_log_path")
    stdout_path = (
        Path(raw_stdout_path).resolve()
        if isinstance(raw_stdout_path, str) and raw_stdout_path
        else None
    )
    raw_stderr_path = receipt.get("stderr_log_path")
    stderr_path = (
        Path(raw_stderr_path).resolve()
        if isinstance(raw_stderr_path, str) and raw_stderr_path
        else None
    )
    expected_previous_shape = (
        previous_sha is None and previous_path is None
        if expected_step == 1
        else (
            _valid_sha256(previous_sha)
            and previous_path is not None
            and previous_path.is_file()
            and _sha256(previous_path) == previous_sha
        )
    )
    command_matches = bool(command) and (
        _service_command_sha256(command)
        == receipt.get("service_command_sha256")
        and _command_option(command, "--vertical") == "companion"
        and _command_option(command, "--steering-activation-step")
        == str(expected_step)
        and _command_option(command, "--steering-promotion-manifest")
        == str(expected_manifest_path.resolve())
        and _command_option(command, "--steering-activation-plan")
        == str(expected_plan_path.resolve())
        and _command_option(command, "--steering-artifact-bundle")
        == str(expected_bundle_path.resolve())
        and _command_option(
            command,
            "--steering-previous-activation-receipt",
        )
        == (str(previous_path) if previous_path is not None else None)
    )
    logs_match = (
        stdout_path is not None
        and stderr_path is not None
        and stdout_path.is_file()
        and stderr_path.is_file()
        and _sha256(stdout_path) == receipt.get("stdout_sha256")
        and _sha256(stderr_path) == receipt.get("stderr_sha256")
    )
    if (
        receipt.get("schema_version")
        != STEERING_ACTIVATION_CANARY_RECEIPT_SCHEMA
        or receipt.get("completed") is not True
        or receipt.get("completed_rollout_step") != expected_step
        or receipt.get("single_field_flip") != expected_single_field_flip
        or receipt.get("rollout_values") != expected_state
        or receipt.get("manifest_sha256") != expected_manifest_sha256
        or receipt.get("activation_plan_sha256") != expected_plan_sha256
        or receipt.get("modification_gate_review_sha256")
        != expected_modification_gate_sha256
        or receipt.get("candidate_bundle_sha256")
        != expected_bundle_sha256
        or receipt.get("candidate_bundle_id") != expected_bundle_id
        or receipt.get("eligible_prefix") != list(expected_eligible_prefix)
        or receipt.get("production_default_changed") is not False
        or receipt.get("intentional_shutdown_after_health_check") is not True
        or expected_previous_shape is not True
        or command_matches is not True
        or logs_match is not True
        or not isinstance(health, dict)
        or health.get("status") != "ok"
        or health.get("vertical") != "companion"
        or not isinstance(session_count, int)
        or isinstance(session_count, bool)
        or session_count < 0
        or health.get("persistence_scope_sha256") is not None
        or not isinstance(service_pid, int)
        or isinstance(service_pid, bool)
        or service_pid < 1
        or not isinstance(receipt.get("completed_at"), str)
        or not receipt["completed_at"].strip()
        or not _valid_sha256(receipt.get("service_command_sha256"))
        or not _valid_sha256(receipt.get("stdout_sha256"))
        or not _valid_sha256(receipt.get("stderr_sha256"))
        or not isinstance(receipt.get("service_exit_code"), int)
        or isinstance(receipt.get("service_exit_code"), bool)
    ):
        raise ValueError(
            "B3 previous activation receipt is invalid or not the "
            "immediately preceding healthy canary"
        )
    return receipt_sha256, receipt


def _expected_plan(
    eligible_prefix: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if eligible_prefix not in tuple(_ORDER[:size] for size in range(4)):
        raise ValueError("steering activation eligible_prefix is not ordered")
    state = dict(_INITIAL_STATE)
    steps: list[dict[str, object]] = []

    def append_step(*, field: str, value: str, purpose: str) -> None:
        before = state[field]
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

    if len(eligible_prefix) >= 1:
        append_step(
            field="steering_sensor",
            value="active",
            purpose="activate the first authorized owner",
        )
    if len(eligible_prefix) >= 2:
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
    if len(eligible_prefix) >= 3:
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
    rollback_steps: list[dict[str, object]] = []
    for step in reversed(steps):
        flip = step["single_field_flip"]
        if not isinstance(flip, dict):  # pragma: no cover - local construction
            raise RuntimeError("steering activation flip shape drift")
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
    return steps, rollback_steps


def load_steering_activation_authorization(
    *,
    bundle: SteeringArtifactBundle,
    bundle_sha256: str,
    promotion_manifest: Path,
    activation_plan: Path,
    rollout_step: int,
    substrate_model_id: str,
    substrate_expected_weights_sha256: str,
    substrate_layer_indices: tuple[int, ...],
    substrate_activation_width: int,
    substrate_max_length: int,
    previous_activation_receipt: Path | None = None,
    base_config: FinalRolloutConfig | None = None,
) -> SteeringActivationAuthorization:
    """Validate B3 lineage and materialize exactly one authorized rollout state."""

    if not _valid_sha256(bundle_sha256):
        raise ValueError("candidate steering bundle SHA-256 is invalid")
    manifest_path = promotion_manifest.resolve()
    plan_path = activation_plan.resolve()
    manifest = _load_object(manifest_path, label="B3 promotion manifest")
    _validate_activation_source_snapshot(manifest.get("source_sha256"))
    plan = _load_object(plan_path, label="B3 activation plan")
    plan_sha256 = _sha256(plan_path)
    manifest_sha256 = _sha256(manifest_path)
    evidence_path = manifest_path.parent / "promotion_evidence.json"
    modification_gate_path = (
        manifest_path.parent / "modification_gate_review.json"
    )
    report_path = manifest_path.parent / "promotion_report.json"
    evidence = _load_object(evidence_path, label="B3 promotion evidence")
    modification_gate = _load_object(
        modification_gate_path,
        label="B3 ModificationGate review",
    )
    report = _load_object(report_path, label="B3 promotion report")
    raw_prefix = manifest.get("eligible_prefix")
    if not isinstance(raw_prefix, list) or not all(
        isinstance(value, str) for value in raw_prefix
    ):
        raise ValueError("B3 promotion manifest eligible_prefix is invalid")
    eligible_prefix = tuple(raw_prefix)
    expected_steps, expected_rollback = _expected_plan(eligible_prefix)
    c3_preregistration_sha256 = manifest.get("c3_preregistration_sha256")
    if (
        manifest.get("schema_version")
        != "steering-promotion-formal-manifest.v2"
        or manifest.get("completed") is not True
        or manifest.get("production_default_changed") is not False
        or not _valid_sha256(manifest.get("preregistration_sha256"))
        or not _valid_sha256(c3_preregistration_sha256)
        or manifest.get("activation_plan_sha256") != plan_sha256
        or manifest.get("candidate_steering_artifact_bundle_sha256")
        != bundle_sha256
        or manifest.get("candidate_bundle_id") != bundle.bundle_id
        or manifest.get("promotion_evidence_sha256")
        != _sha256(evidence_path)
        or manifest.get("modification_gate_review_sha256")
        != _sha256(modification_gate_path)
        or manifest.get("promotion_report_sha256") != _sha256(report_path)
        or manifest.get("sensor_executor_active_authorized")
        is not (len(eligible_prefix) >= 2)
        or manifest.get("gate_active_authorized")
        is not (len(eligible_prefix) == 3)
        or manifest.get("modification_gate_decision") != "allow"
        or manifest.get("modification_gate_reasons") != []
        or manifest.get("modification_gate_audit_required") is not False
        or manifest.get("canary_receipt_policy") != _CANARY_RECEIPT_POLICY
    ):
        raise ValueError("B3 promotion manifest does not authorize this rollout")
    sensor_off = bundle.sensor_off_executor
    if sensor_off is None or any(
        artifact.source_preregistration_sha256 != c3_preregistration_sha256
        for artifact in (bundle.reader, bundle.executor, bundle.gate, sensor_off)
    ):
        raise ValueError("candidate steering bundle C3 lineage drift")
    expected_gate = json.loads(json.dumps(asdict(bundle.gate), sort_keys=True))
    if (
        evidence.get("schema_version") != "steering-promotion-evidence.v1"
        or evidence.get("preregistration_sha256")
        != manifest["preregistration_sha256"]
        or evidence.get("c3_preregistration_sha256")
        != c3_preregistration_sha256
        or evidence.get("candidate_gate_artifact") != expected_gate
        or evidence.get("free_bias_present") is not False
        or evidence.get("zero_code_strict_noop") is not True
        or evidence.get("raw_text_retained") is not False
        or evidence.get("evaluation_writeback_allowed") is not False
        or evidence.get("production_default_changed") is not False
        or modification_gate.get("schema_version")
        != "steering-modification-gate-review.v1"
        or modification_gate.get("preregistration_sha256")
        != manifest["preregistration_sha256"]
        or modification_gate.get("c3_preregistration_sha256")
        != c3_preregistration_sha256
        or modification_gate.get("proposal_target")
        != "substrate.steering_artifact_bundle"
        or modification_gate.get("desired_gate") != "offline"
        or modification_gate.get("old_value_hash")
        != evidence.get("bundle_sha256")
        or modification_gate.get("new_value_hash") != bundle_sha256
        or not isinstance(modification_gate.get("rollback_evidence"), str)
        or not modification_gate["rollback_evidence"]
        or modification_gate.get("audit_required") is not False
        or modification_gate.get("audit_evidence_id") is not None
        or modification_gate.get("decision") != "allow"
        or modification_gate.get("blocking_reasons") != []
        or report.get("eligible_prefix") != list(eligible_prefix)
        or report.get("sensor_executor_active_authorized")
        is not (len(eligible_prefix) >= 2)
        or report.get("gate_active_authorized")
        is not (len(eligible_prefix) == 3)
        or report.get("activation_order") != list(_ORDER)
        or report.get("rollback_order") != list(reversed(_ORDER))
        or report.get("modification_gate_decision") != "allow"
        or report.get("modification_gate_reasons") != []
        or report.get("blocking_reasons")
        != manifest.get("blocking_reasons")
    ):
        raise ValueError("B3 promotion evidence/report lineage drift")

    candidate = plan.get("candidate_bundle")
    plan_modification_gate = plan.get("modification_gate")
    deployment = plan.get("deployment_contract")
    if (
        plan.get("schema_version") != "steering-activation-plan.v3"
        or plan.get("eligible_prefix") != list(eligible_prefix)
        or plan.get("steps") != expected_steps
        or plan.get("rollback_steps") != expected_rollback
        or plan.get("rollback_order") != list(reversed(_ORDER))
        or plan.get("production_default_changed") is not False
        or not isinstance(candidate, dict)
        or candidate.get("sha256") != bundle_sha256
        or not isinstance(candidate.get("path"), str)
        or plan_modification_gate
        != {
            "review_sha256": _sha256(modification_gate_path),
            "decision": "allow",
            "blocking_reasons": [],
        }
        or plan.get("canary_receipt_policy") != _CANARY_RECEIPT_POLICY
        or not isinstance(deployment, dict)
        or manifest.get("deployment_contract") != deployment
    ):
        raise ValueError("B3 activation plan is invalid or drifted")
    expected_deployment_keys = {
        "model_id",
        "model_weights_sha256",
        "steering_layer_index",
        "activation_width",
        "substrate_max_length",
        "generation_max_new_tokens",
        "generation_temperature",
        "fail_on_truncation",
    }
    if (
        set(deployment) != expected_deployment_keys
        or deployment.get("model_id") != bundle.reader.model_id
        or deployment.get("model_weights_sha256")
        != bundle.reader.model_weights_sha256
        or deployment.get("steering_layer_index") != bundle.reader.layer_index
        or deployment.get("activation_width") != bundle.reader.residual_width
        or not isinstance(deployment.get("substrate_max_length"), int)
        or isinstance(deployment["substrate_max_length"], bool)
        or deployment["substrate_max_length"] < 1
        or not isinstance(deployment.get("generation_max_new_tokens"), int)
        or isinstance(deployment["generation_max_new_tokens"], bool)
        or deployment["generation_max_new_tokens"] < 16
        or deployment.get("generation_temperature") != 0.0
        or deployment.get("fail_on_truncation") is not True
    ):
        raise ValueError("B3 deployment contract is invalid or unbound")
    if (
        substrate_model_id != deployment["model_id"]
        or substrate_expected_weights_sha256
        != deployment["model_weights_sha256"]
        or deployment["steering_layer_index"] not in substrate_layer_indices
        or substrate_activation_width != deployment["activation_width"]
        or substrate_max_length != deployment["substrate_max_length"]
    ):
        raise ValueError("service substrate differs from the B3 deployment contract")
    if rollout_step < 1 or rollout_step > len(expected_steps):
        raise ValueError(
            "requested steering rollout step is not authorized by the B3 prefix"
        )
    previous_state = (
        dict(_INITIAL_STATE)
        if rollout_step == 1
        else expected_steps[rollout_step - 2]["rollout_values_after_flip"]
    )
    if not isinstance(previous_state, dict):  # pragma: no cover - construction
        raise RuntimeError("steering previous rollout state shape drift")
    previous_receipt_path: str | None = None
    previous_receipt_sha256: str | None = None
    if rollout_step == 1:
        if previous_activation_receipt is not None:
            raise ValueError("first B3 rollout step forbids a previous receipt")
    else:
        if previous_activation_receipt is None:
            raise ValueError(
                "B3 rollout step requires the immediately preceding healthy "
                "canary receipt"
            )
        previous_receipt_sha256, _ = _validate_previous_receipt(
            path=previous_activation_receipt,
            expected_step=rollout_step - 1,
            expected_state=previous_state,
            expected_single_field_flip=expected_steps[rollout_step - 2][
                "single_field_flip"
            ],
            expected_manifest_sha256=manifest_sha256,
            expected_plan_sha256=plan_sha256,
            expected_modification_gate_sha256=_sha256(
                modification_gate_path
            ),
            expected_bundle_sha256=bundle_sha256,
            expected_bundle_id=bundle.bundle_id,
            expected_eligible_prefix=eligible_prefix,
            expected_manifest_path=manifest_path,
            expected_plan_path=plan_path,
            expected_bundle_path=Path(str(candidate["path"])),
        )
        previous_receipt_path = str(previous_activation_receipt.resolve())

    config = base_config or _config_with_steering_state(
        FinalRolloutConfig(),
        previous_state,
    )
    if _steering_state(config) != previous_state:
        raise ValueError(
            "steering activation base config differs from the attested "
            "immediately preceding rollout state"
        )
    raw_flip = expected_steps[rollout_step - 1]["single_field_flip"]
    expected_state = expected_steps[rollout_step - 1][
        "rollout_values_after_flip"
    ]
    if not isinstance(raw_flip, dict) or not isinstance(expected_state, dict):
        raise RuntimeError("steering rollout plan shape drift")
    field = str(raw_flip["field"])
    before = str(raw_flip["from"])
    after = str(raw_flip["to"])
    if previous_state[field] != before:
        raise RuntimeError("steering rollout previous state/flip drift")
    applied_state = dict(previous_state)
    applied_state[field] = after
    if applied_state != expected_state:
        raise RuntimeError("steering rollout does not apply exactly one field")
    rollout_config = _config_with_steering_state(config, applied_state)
    return SteeringActivationAuthorization(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        activation_plan_path=str(plan_path),
        activation_plan_sha256=plan_sha256,
        modification_gate_review_path=str(modification_gate_path),
        modification_gate_review_sha256=_sha256(modification_gate_path),
        candidate_bundle_sha256=bundle_sha256,
        candidate_bundle_id=bundle.bundle_id,
        eligible_prefix=eligible_prefix,
        rollout_step=rollout_step,
        applied_field=field,
        applied_from=before,
        applied_to=after,
        previous_receipt_path=previous_receipt_path,
        previous_receipt_sha256=previous_receipt_sha256,
        rollout_config=rollout_config,
        substrate_max_length=int(deployment["substrate_max_length"]),
        generation_max_new_tokens=int(
            deployment["generation_max_new_tokens"]
        ),
        generation_temperature=float(deployment["generation_temperature"]),
        fail_on_truncation=bool(deployment["fail_on_truncation"]),
        description=(
            "B3-authorized steering rollout; exact candidate bundle and "
            "immediately preceding canary receipt plus single-field step "
            "verified."
        ),
    )


def build_steering_activation_canary_receipt(
    *,
    authorization: SteeringActivationAuthorization,
    canary_health: Mapping[str, object],
    service_pid: int,
    service_exit_code: int,
    service_command: tuple[str, ...],
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> dict[str, object]:
    """Build the chain link required before the next rollout step.

    The caller must obtain ``canary_health`` from the exact bounded service
    process after it becomes reachable, then intentionally stop that canary.
    This receipt records health/materialization only; it does not claim user
    value or change the production default.
    """

    if (
        canary_health.get("status") != "ok"
        or canary_health.get("vertical") != "companion"
        or not isinstance(canary_health.get("session_count"), int)
        or isinstance(canary_health["session_count"], bool)
        or canary_health["session_count"] < 0
        or canary_health.get("persistence_scope_sha256") is not None
    ):
        raise ValueError("B3 canary health payload is invalid")
    if service_pid < 1:
        raise ValueError("B3 canary service pid must be positive")
    if not isinstance(service_exit_code, int) or isinstance(
        service_exit_code,
        bool,
    ):
        raise ValueError("B3 canary service exit code must be an integer")
    if not service_command or not all(
        isinstance(value, str) and value for value in service_command
    ):
        raise ValueError("B3 canary service command is invalid")
    stdout_path = stdout_log_path.resolve()
    stderr_path = stderr_log_path.resolve()
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise FileNotFoundError("B3 canary stdout/stderr log is missing")
    return {
        "schema_version": STEERING_ACTIVATION_CANARY_RECEIPT_SCHEMA,
        "completed": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "completed_rollout_step": authorization.rollout_step,
        "single_field_flip": {
            "field": authorization.applied_field,
            "from": authorization.applied_from,
            "to": authorization.applied_to,
        },
        "rollout_values": _steering_state(authorization.rollout_config),
        "manifest_sha256": authorization.manifest_sha256,
        "activation_plan_sha256": authorization.activation_plan_sha256,
        "modification_gate_review_sha256": (
            authorization.modification_gate_review_sha256
        ),
        "candidate_bundle_sha256": authorization.candidate_bundle_sha256,
        "candidate_bundle_id": authorization.candidate_bundle_id,
        "eligible_prefix": list(authorization.eligible_prefix),
        "previous_receipt_path": authorization.previous_receipt_path,
        "previous_receipt_sha256": authorization.previous_receipt_sha256,
        "canary_health": dict(canary_health),
        "service_pid": service_pid,
        "service_exit_code": service_exit_code,
        "service_command": list(service_command),
        "service_command_sha256": _service_command_sha256(service_command),
        "stdout_log_path": str(stdout_path),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_log_path": str(stderr_path),
        "stderr_sha256": _sha256(stderr_path),
        "intentional_shutdown_after_health_check": True,
        "production_default_changed": False,
        "description": (
            "Bounded B3 service canary reached the companion health endpoint "
            "after applying exactly one attested rollout field, then stopped."
        ),
    }


def write_steering_activation_canary_receipt(
    *,
    path: Path,
    receipt: Mapping[str, object],
) -> None:
    target = path.resolve()
    payload = (
        json.dumps(
            dict(receipt),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError(f"immutable B3 canary receipt differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = (
    "STEERING_ACTIVATION_CANARY_RECEIPT_SCHEMA",
    "SteeringActivationAuthorization",
    "build_steering_activation_canary_receipt",
    "load_steering_activation_authorization",
    "steering_activation_canary_receipt_policy",
    "write_steering_activation_canary_receipt",
)
