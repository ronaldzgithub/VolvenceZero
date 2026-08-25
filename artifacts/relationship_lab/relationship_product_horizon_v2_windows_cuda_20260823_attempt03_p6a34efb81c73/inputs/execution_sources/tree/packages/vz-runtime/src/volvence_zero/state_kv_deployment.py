"""Fail-closed deployment gate and profile binding for State-KV."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from volvence_zero.substrate import (
    OpenWeightResidualRuntime,
    TransformersOpenWeightResidualRuntime,
)

STATE_KV_DEPLOYMENT_SCHEMA_VERSION = "state-kv-deployment-gate.v1"
STATE_KV_DEPLOYMENT_PROFILE_LABEL = "state-kv-active-v1"
STATE_KV_DEPLOYMENT_ARTIFACT_ID = (
    "8064f8b6de8ec215807619f404c84404087109076634d1ffda53112b4684e238"
)
STATE_KV_DEPLOYMENT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_TEMPORAL_SCHEMA = "state-kv-temporal-causal.v1"
_COURT_SCHEMA = "state-kv-judge-court.v1"
_GENERATION_SEED_SCHEMA = "state-kv-generation-seed-gate.v1"


class DeploymentClaimState(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class DeploymentGateState(str, Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class DeploymentClaim:
    claim: str
    state: DeploymentClaimState
    detail: str

    def as_json_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "state": self.state.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class StateKVDeploymentSafetyObservation:
    """Runtime negative controls required before the ACTIVE profile is valid."""

    profile_label: str
    artifact_id: str
    model_id: str
    prompt_sha256: str
    cold_start_baseline_equal: bool
    zero_confidence_baseline_equal: bool
    shadow_baseline_equal: bool
    revoked_baseline_equal: bool
    rollback_baseline_equal: bool
    inert_controls_report_applied_false: bool
    active_correct_applied: bool
    active_wrong_user_applied: bool
    active_users_diverge: bool
    active_replay_equal: bool
    user_cache_scopes_distinct: bool
    baseline_output_sha256: str
    correct_output_sha256: str
    wrong_user_output_sha256: str

    def as_json_dict(self) -> dict[str, object]:
        return {
            "profile_label": self.profile_label,
            "artifact_id": self.artifact_id,
            "model_id": self.model_id,
            "prompt_sha256": self.prompt_sha256,
            "cold_start_baseline_equal": self.cold_start_baseline_equal,
            "zero_confidence_baseline_equal": (
                self.zero_confidence_baseline_equal
            ),
            "shadow_baseline_equal": self.shadow_baseline_equal,
            "revoked_baseline_equal": self.revoked_baseline_equal,
            "rollback_baseline_equal": self.rollback_baseline_equal,
            "inert_controls_report_applied_false": (
                self.inert_controls_report_applied_false
            ),
            "active_correct_applied": self.active_correct_applied,
            "active_wrong_user_applied": self.active_wrong_user_applied,
            "active_users_diverge": self.active_users_diverge,
            "active_replay_equal": self.active_replay_equal,
            "user_cache_scopes_distinct": self.user_cache_scopes_distinct,
            "baseline_output_sha256": self.baseline_output_sha256,
            "correct_output_sha256": self.correct_output_sha256,
            "wrong_user_output_sha256": self.wrong_user_output_sha256,
        }


@dataclass(frozen=True)
class StateKVDeploymentReport:
    schema_version: str
    gate_state: DeploymentGateState
    profile_label: str
    artifact_id: str
    model_id: str
    claims: tuple[DeploymentClaim, ...]
    evidence_paths: tuple[tuple[str, str], ...]
    safety_observation: StateKVDeploymentSafetyObservation
    rollback: str
    notes: tuple[str, ...]

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate_state": self.gate_state.value,
            "profile_label": self.profile_label,
            "artifact_id": self.artifact_id,
            "model_id": self.model_id,
            "claims": [claim.as_json_dict() for claim in self.claims],
            "evidence_paths": dict(self.evidence_paths),
            "safety_observation": self.safety_observation.as_json_dict(),
            "rollback": self.rollback,
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_json_dict(), ensure_ascii=False, indent=2)


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _require_schema(
    payload: Mapping[str, Any],
    *,
    schema: str,
    path: Path,
) -> None:
    if payload.get("schema_version") != schema:
        raise ValueError(
            f"{path} has schema {payload.get('schema_version')!r}; "
            f"expected {schema!r}"
        )


def _claim(
    name: str,
    passed: bool,
    pass_detail: str,
    fail_detail: str,
) -> DeploymentClaim:
    return DeploymentClaim(
        claim=name,
        state=(
            DeploymentClaimState.PASS
            if passed
            else DeploymentClaimState.FAIL
        ),
        detail=pass_detail if passed else fail_detail,
    )


def validate_state_kv_deployment_runtime(
    runtime: OpenWeightResidualRuntime | None,
) -> TransformersOpenWeightResidualRuntime:
    """Require the exact frozen runtime/artifact bound by the profile."""

    if runtime is None:
        raise ValueError(
            f"{STATE_KV_DEPLOYMENT_PROFILE_LABEL} requires an explicit "
            "open-weight runtime; fallback construction is forbidden"
        )
    if not isinstance(runtime, TransformersOpenWeightResidualRuntime):
        raise TypeError(
            f"{STATE_KV_DEPLOYMENT_PROFILE_LABEL} requires "
            "TransformersOpenWeightResidualRuntime"
        )
    if runtime.model_id != STATE_KV_DEPLOYMENT_MODEL_ID:
        raise ValueError(
            f"deployment model mismatch: {runtime.model_id!r} != "
            f"{STATE_KV_DEPLOYMENT_MODEL_ID!r}"
        )
    if not runtime.is_frozen:
        raise ValueError("State-KV deployment runtime must keep base weights frozen")
    if (
        runtime.personal_conditioning_prefix_id
        != STATE_KV_DEPLOYMENT_ARTIFACT_ID
    ):
        raise ValueError(
            "State-KV deployment artifact mismatch: "
            f"{runtime.personal_conditioning_prefix_id!r} != "
            f"{STATE_KV_DEPLOYMENT_ARTIFACT_ID!r}"
        )
    return runtime


def build_state_kv_deployment_config(
    runtime: OpenWeightResidualRuntime | None,
):
    """Build the opt-in ACTIVE config after validating runtime binding."""

    validate_state_kv_deployment_runtime(runtime)
    from volvence_zero.agent.profile_registry import resolve_profile
    from volvence_zero.integration.final_wiring import FinalRolloutConfig

    profile = resolve_profile(STATE_KV_DEPLOYMENT_PROFILE_LABEL)
    flags = profile.merged_flag_overrides
    expected = {
        "personal_conditioning": "WiringLevel.ACTIVE",
        "personal_conditioning_mode": "prefix_kv",
        "personal_conditioning_prefix_artifact_id": (
            STATE_KV_DEPLOYMENT_ARTIFACT_ID
        ),
    }
    if any(flags.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"{STATE_KV_DEPLOYMENT_PROFILE_LABEL} registry binding drifted"
        )
    return profile.apply_to_config(FinalRolloutConfig())


def _panel_artifact_ids(payload: Mapping[str, Any], path: Path) -> set[str]:
    panels = payload.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError(f"{path} requires non-empty panels")
    artifact_ids: set[str] = set()
    for panel in panels:
        if not isinstance(panel, dict):
            raise ValueError(f"{path} panels must be objects")
        artifact_id = panel.get("prefix_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"{path} panel requires prefix_artifact_id")
        artifact_ids.add(artifact_id)
    return artifact_ids


def _panel_model_ids(payload: Mapping[str, Any], path: Path) -> set[str]:
    panels = payload.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError(f"{path} requires non-empty panels")
    model_ids: set[str] = set()
    for panel in panels:
        if not isinstance(panel, dict):
            raise ValueError(f"{path} panels must be objects")
        fingerprint = panel.get("substrate_fingerprint")
        if not isinstance(fingerprint, str) or "@" not in fingerprint:
            raise ValueError(
                f"{path} panel requires model@revision substrate_fingerprint"
            )
        model_ids.add(fingerprint.split("@", maxsplit=1)[0])
    return model_ids


def build_state_kv_deployment_report(
    *,
    temporal_causal_path: Path | str,
    judge_court_path: Path | str,
    generation_seed_path: Path | str,
    safety_observation: StateKVDeploymentSafetyObservation,
) -> StateKVDeploymentReport:
    """Combine upstream evidence and runtime controls into one promotion gate."""

    temporal_path = Path(temporal_causal_path).expanduser().resolve()
    court_path = Path(judge_court_path).expanduser().resolve()
    seed_path = Path(generation_seed_path).expanduser().resolve()
    temporal = _read_object(temporal_path)
    court = _read_object(court_path)
    seed = _read_object(seed_path)
    _require_schema(temporal, schema=_TEMPORAL_SCHEMA, path=temporal_path)
    _require_schema(court, schema=_COURT_SCHEMA, path=court_path)
    _require_schema(seed, schema=_GENERATION_SEED_SCHEMA, path=seed_path)

    upstream_pass = (
        temporal.get("gate_state") == "pass"
        and court.get("court_state") == "pass"
        and seed.get("gate_state") == "pass"
    )
    court_artifacts = _panel_artifact_ids(court, court_path)
    seed_artifacts = _panel_artifact_ids(seed, seed_path)
    court_models = _panel_model_ids(court, court_path)
    seed_models = _panel_model_ids(seed, seed_path)
    temporal_artifact = temporal.get("artifact_id")
    temporal_fingerprint = temporal.get("substrate_fingerprint")
    temporal_model = (
        temporal_fingerprint.split("@", maxsplit=1)[0]
        if isinstance(temporal_fingerprint, str)
        and "@" in temporal_fingerprint
        else ""
    )
    material_bound = (
        temporal_artifact == STATE_KV_DEPLOYMENT_ARTIFACT_ID
        and court_artifacts == {STATE_KV_DEPLOYMENT_ARTIFACT_ID}
        and seed_artifacts == {STATE_KV_DEPLOYMENT_ARTIFACT_ID}
        and temporal_model == STATE_KV_DEPLOYMENT_MODEL_ID
        and court_models == {STATE_KV_DEPLOYMENT_MODEL_ID}
        and seed_models == {STATE_KV_DEPLOYMENT_MODEL_ID}
        and safety_observation.artifact_id
        == STATE_KV_DEPLOYMENT_ARTIFACT_ID
        and safety_observation.model_id == STATE_KV_DEPLOYMENT_MODEL_ID
        and safety_observation.profile_label
        == STATE_KV_DEPLOYMENT_PROFILE_LABEL
    )
    inert_controls = all(
        (
            safety_observation.cold_start_baseline_equal,
            safety_observation.zero_confidence_baseline_equal,
            safety_observation.shadow_baseline_equal,
            safety_observation.inert_controls_report_applied_false,
        )
    )
    revocation = safety_observation.revoked_baseline_equal
    isolation = all(
        (
            safety_observation.active_correct_applied,
            safety_observation.active_wrong_user_applied,
            safety_observation.active_users_diverge,
            safety_observation.active_replay_equal,
            safety_observation.user_cache_scopes_distinct,
        )
    )
    rollback = safety_observation.rollback_baseline_equal
    claims = (
        _claim(
            "claim_upstream_evidence_passed",
            upstream_pass,
            "temporal causal, judge court, and generation-seed gates passed",
            "one or more upstream evidence gates did not pass",
        ),
        _claim(
            "claim_profile_artifact_binding",
            material_bound,
            "profile, frozen model, safety run, and all evidence bind one artifact/model",
            "profile/model/artifact binding differs across evidence or safety run",
        ),
        _claim(
            "claim_cold_start_and_shadow_inert",
            inert_controls,
            "cold-start, zero-confidence, and SHADOW are baseline-equivalent and unapplied",
            "an inert control changed output or reported conditioning applied",
        ),
        _claim(
            "claim_revocation_restores_baseline",
            revocation,
            "revoked conditioning is baseline-equivalent",
            "revoked conditioning did not restore baseline",
        ),
        _claim(
            "claim_cross_user_isolation",
            isolation,
            "user scopes differ, both states apply, outputs diverge, and replay is stable",
            "cross-user scope, divergence, application, or replay isolation failed",
        ),
        _claim(
            "claim_atomic_rollback",
            rollback,
            "rollback output is byte-identical to baseline",
            "rollback did not restore the baseline output",
        ),
    )
    gate_state = (
        DeploymentGateState.PASS
        if all(claim.state is DeploymentClaimState.PASS for claim in claims)
        else DeploymentGateState.FAIL
    )
    return StateKVDeploymentReport(
        schema_version=STATE_KV_DEPLOYMENT_SCHEMA_VERSION,
        gate_state=gate_state,
        profile_label=STATE_KV_DEPLOYMENT_PROFILE_LABEL,
        artifact_id=STATE_KV_DEPLOYMENT_ARTIFACT_ID,
        model_id=STATE_KV_DEPLOYMENT_MODEL_ID,
        claims=claims,
        evidence_paths=(
            ("temporal_causal", str(temporal_path)),
            ("judge_court", str(court_path)),
            ("generation_seed", str(seed_path)),
        ),
        safety_observation=safety_observation,
        rollback=(
            "Switch personal_conditioning to SHADOW or DISABLED; omit the "
            "deployment profile to return to the default residual baseline."
        ),
        notes=(
            "Passing this gate authorizes only the explicit "
            f"{STATE_KV_DEPLOYMENT_PROFILE_LABEL} profile. The repository "
            "default remains SHADOW.",
        ),
    )


__all__: Sequence[str] = (
    "DeploymentClaimState",
    "DeploymentGateState",
    "STATE_KV_DEPLOYMENT_ARTIFACT_ID",
    "STATE_KV_DEPLOYMENT_MODEL_ID",
    "STATE_KV_DEPLOYMENT_PROFILE_LABEL",
    "STATE_KV_DEPLOYMENT_SCHEMA_VERSION",
    "StateKVDeploymentReport",
    "StateKVDeploymentSafetyObservation",
    "build_state_kv_deployment_config",
    "build_state_kv_deployment_report",
    "validate_state_kv_deployment_runtime",
)
