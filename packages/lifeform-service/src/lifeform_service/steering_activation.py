"""Fail-closed consumption of a B3 steering activation authorization.

The promotion runner writes evidence, a verdict, a candidate artifact bundle,
and an ordered one-field rollout plan.  This module is the deployment-side
reader for those artifacts.  Loading a bundle alone never authorizes ACTIVE
wiring; the exact B3 manifest and activation plan must bind its SHA-256, and a
caller must select one concrete rollout step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

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


@dataclass(frozen=True)
class SteeringActivationAuthorization:
    manifest_path: str
    manifest_sha256: str
    activation_plan_path: str
    activation_plan_sha256: str
    candidate_bundle_sha256: str
    candidate_bundle_id: str
    eligible_prefix: tuple[str, ...]
    rollout_step: int
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
    base_config: FinalRolloutConfig | None = None,
) -> SteeringActivationAuthorization:
    """Validate B3 lineage and materialize exactly one authorized rollout state."""

    if not _valid_sha256(bundle_sha256):
        raise ValueError("candidate steering bundle SHA-256 is invalid")
    manifest_path = promotion_manifest.resolve()
    plan_path = activation_plan.resolve()
    manifest = _load_object(manifest_path, label="B3 promotion manifest")
    plan = _load_object(plan_path, label="B3 activation plan")
    plan_sha256 = _sha256(plan_path)
    manifest_sha256 = _sha256(manifest_path)
    evidence_path = manifest_path.parent / "promotion_evidence.json"
    report_path = manifest_path.parent / "promotion_report.json"
    evidence = _load_object(evidence_path, label="B3 promotion evidence")
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
        != "steering-promotion-formal-manifest.v1"
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
        or manifest.get("promotion_report_sha256") != _sha256(report_path)
        or manifest.get("sensor_executor_active_authorized")
        is not (len(eligible_prefix) >= 2)
        or manifest.get("gate_active_authorized")
        is not (len(eligible_prefix) == 3)
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
        or report.get("eligible_prefix") != list(eligible_prefix)
        or report.get("sensor_executor_active_authorized")
        is not (len(eligible_prefix) >= 2)
        or report.get("gate_active_authorized")
        is not (len(eligible_prefix) == 3)
        or report.get("activation_order") != list(_ORDER)
        or report.get("rollback_order") != list(reversed(_ORDER))
        or report.get("blocking_reasons")
        != manifest.get("blocking_reasons")
    ):
        raise ValueError("B3 promotion evidence/report lineage drift")

    candidate = plan.get("candidate_bundle")
    deployment = plan.get("deployment_contract")
    if (
        plan.get("schema_version") != "steering-activation-plan.v2"
        or plan.get("eligible_prefix") != list(eligible_prefix)
        or plan.get("steps") != expected_steps
        or plan.get("rollback_steps") != expected_rollback
        or plan.get("rollback_order") != list(reversed(_ORDER))
        or plan.get("production_default_changed") is not False
        or not isinstance(candidate, dict)
        or candidate.get("sha256") != bundle_sha256
        or not isinstance(candidate.get("path"), str)
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

    config = base_config or FinalRolloutConfig()
    if {
        "steering_sensor": config.steering_sensor.value,
        "steering_executor": config.steering_executor.value,
        "steering_gate": config.steering_gate.value,
        "steering_ungated_action": config.steering_ungated_action,
    } != _INITIAL_STATE:
        raise ValueError("steering activation requires the frozen SHADOW baseline")
    state = expected_steps[rollout_step - 1]["rollout_values_after_flip"]
    if not isinstance(state, dict):  # pragma: no cover - local construction
        raise RuntimeError("steering rollout state shape drift")
    rollout_config = replace(
        config,
        steering_sensor=WiringLevel(str(state["steering_sensor"])),
        steering_executor=WiringLevel(str(state["steering_executor"])),
        steering_gate=WiringLevel(str(state["steering_gate"])),
        steering_ungated_action=str(state["steering_ungated_action"]),
    )
    return SteeringActivationAuthorization(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        activation_plan_path=str(plan_path),
        activation_plan_sha256=plan_sha256,
        candidate_bundle_sha256=bundle_sha256,
        candidate_bundle_id=bundle.bundle_id,
        eligible_prefix=eligible_prefix,
        rollout_step=rollout_step,
        rollout_config=rollout_config,
        substrate_max_length=int(deployment["substrate_max_length"]),
        generation_max_new_tokens=int(
            deployment["generation_max_new_tokens"]
        ),
        generation_temperature=float(deployment["generation_temperature"]),
        fail_on_truncation=bool(deployment["fail_on_truncation"]),
        description=(
            "B3-authorized steering rollout; exact candidate bundle and "
            "single-field step verified."
        ),
    )


__all__ = (
    "SteeringActivationAuthorization",
    "load_steering_activation_authorization",
)
