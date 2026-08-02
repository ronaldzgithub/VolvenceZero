from __future__ import annotations

import json
from pathlib import Path

import pytest

from volvence_zero.agent.profile_registry import resolve_profile
from volvence_zero.agent.session_observation import (
    _personal_conditioning_delivery_from_config,
)
from volvence_zero.conditioning_bank_contracts import (
    ConditioningRevocationState,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.agent.dialogue import (
    DEFAULT_DIALOGUE_PROOF_CASES,
    build_standard_dialogue_runner,
)
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.runtime import WiringLevel
from volvence_zero.state_kv_deployment import (
    DeploymentGateState,
    STATE_KV_DEPLOYMENT_ARTIFACT_ID,
    STATE_KV_DEPLOYMENT_MODEL_ID,
    STATE_KV_DEPLOYMENT_PROFILE_LABEL,
    StateKVDeploymentSafetyObservation,
    build_state_kv_deployment_config,
    build_state_kv_deployment_report,
    validate_state_kv_deployment_runtime,
)
from volvence_zero.substrate import TransformersOpenWeightResidualRuntime


def _runtime(*, artifact_id: str = STATE_KV_DEPLOYMENT_ARTIFACT_ID):
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime.model_id = STATE_KV_DEPLOYMENT_MODEL_ID
    runtime.is_frozen = True
    runtime._personal_conditioning_prefix_id = artifact_id
    return runtime


def _safety(**overrides) -> StateKVDeploymentSafetyObservation:
    values = {
        "profile_label": STATE_KV_DEPLOYMENT_PROFILE_LABEL,
        "artifact_id": STATE_KV_DEPLOYMENT_ARTIFACT_ID,
        "model_id": STATE_KV_DEPLOYMENT_MODEL_ID,
        "prompt_sha256": "a" * 64,
        "cold_start_baseline_equal": True,
        "zero_confidence_baseline_equal": True,
        "shadow_baseline_equal": True,
        "revoked_baseline_equal": True,
        "rollback_baseline_equal": True,
        "inert_controls_report_applied_false": True,
        "active_correct_applied": True,
        "active_wrong_user_applied": True,
        "active_users_diverge": True,
        "active_replay_equal": True,
        "user_cache_scopes_distinct": True,
        "baseline_output_sha256": "b" * 64,
        "correct_output_sha256": "c" * 64,
        "wrong_user_output_sha256": "d" * 64,
    }
    values.update(overrides)
    return StateKVDeploymentSafetyObservation(**values)


def _write_evidence(root: Path) -> tuple[Path, Path, Path]:
    temporal = root / "temporal.json"
    court = root / "court.json"
    seed = root / "seed.json"
    temporal.write_text(
        json.dumps(
            {
                "schema_version": "state-kv-temporal-causal.v1",
                "gate_state": "pass",
                "artifact_id": STATE_KV_DEPLOYMENT_ARTIFACT_ID,
                "substrate_fingerprint": (
                    f"{STATE_KV_DEPLOYMENT_MODEL_ID}@revision"
                ),
            }
        ),
        encoding="utf-8",
    )
    panel = {
        "prefix_artifact_id": STATE_KV_DEPLOYMENT_ARTIFACT_ID,
        "substrate_fingerprint": (
            f"{STATE_KV_DEPLOYMENT_MODEL_ID}@revision"
        ),
    }
    court.write_text(
        json.dumps(
            {
                "schema_version": "state-kv-judge-court.v1",
                "court_state": "pass",
                "panels": [panel, panel],
            }
        ),
        encoding="utf-8",
    )
    seed.write_text(
        json.dumps(
            {
                "schema_version": "state-kv-generation-seed-gate.v1",
                "gate_state": "pass",
                "panels": [panel, panel, panel],
            }
        ),
        encoding="utf-8",
    )
    return temporal, court, seed


def test_deployment_profile_binds_active_prefix_and_artifact() -> None:
    profile = resolve_profile(STATE_KV_DEPLOYMENT_PROFILE_LABEL)

    assert profile.merged_flag_overrides == {
        "personal_conditioning": "WiringLevel.ACTIVE",
        "personal_conditioning_mode": "prefix_kv",
        "personal_conditioning_prefix_artifact_id": (
            STATE_KV_DEPLOYMENT_ARTIFACT_ID
        ),
    }
    config = build_state_kv_deployment_config(_runtime())
    assert config.personal_conditioning is WiringLevel.ACTIVE
    assert config.personal_conditioning_mode == "prefix_kv"
    assert (
        config.personal_conditioning_prefix_artifact_id
        == STATE_KV_DEPLOYMENT_ARTIFACT_ID
    )
    assert config.prompt_state_delivery == "text"


def test_default_profile_remains_shadow_residual() -> None:
    from volvence_zero.integration.final_wiring import FinalRolloutConfig

    config = FinalRolloutConfig()

    assert config.personal_conditioning is WiringLevel.SHADOW
    assert config.personal_conditioning_mode == "residual"
    assert config.personal_conditioning_prefix_artifact_id is None


def test_deployment_profile_applies_through_formal_config_contract() -> None:
    from volvence_zero.integration.final_wiring import FinalRolloutConfig

    config = resolve_profile(STATE_KV_DEPLOYMENT_PROFILE_LABEL).apply_to_config(
        FinalRolloutConfig()
    )

    assert (
        config.personal_conditioning_prefix_artifact_id
        == STATE_KV_DEPLOYMENT_ARTIFACT_ID
    )


def test_runner_fails_closed_when_formal_artifact_binding_mismatches() -> None:
    from volvence_zero.integration.final_wiring import FinalRolloutConfig

    with pytest.raises(ValueError, match="does not match the loaded runtime"):
        AgentSessionRunner(
            config=FinalRolloutConfig(
                personal_conditioning=WiringLevel.ACTIVE,
                personal_conditioning_mode="prefix_kv",
                personal_conditioning_prefix_artifact_id="wrong-artifact",
            ),
            default_residual_runtime=_runtime(),
        )


def test_standard_runner_accepts_only_the_bound_deployment_runtime() -> None:
    runner = build_standard_dialogue_runner(
        profile_label=STATE_KV_DEPLOYMENT_PROFILE_LABEL,
        case=DEFAULT_DIALOGUE_PROOF_CASES[0],
        residual_runtime=_runtime(),
    )

    assert runner._config.personal_conditioning is WiringLevel.ACTIVE
    assert runner._config.personal_conditioning_mode == "prefix_kv"
    runner._previous_personal_conditioning_snapshot = object()
    runner.set_personal_conditioning_revocation_state(
        ConditioningRevocationState.REVOKED
    )
    assert (
        runner.personal_conditioning_revocation_state
        is ConditioningRevocationState.REVOKED
    )
    assert runner._previous_personal_conditioning_snapshot is None


def test_deployment_runtime_rejects_missing_or_wrong_artifact() -> None:
    with pytest.raises(ValueError, match="explicit"):
        validate_state_kv_deployment_runtime(None)
    with pytest.raises(ValueError, match="artifact mismatch"):
        validate_state_kv_deployment_runtime(_runtime(artifact_id="wrong"))


def test_revocation_blocks_the_production_delivery_selector() -> None:
    snapshot = PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(
            0.7 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS
        ),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("boundary_consent", 2),),
        source_fingerprint="deployment-revocation-test",
        confidence=0.7,
        is_cold_start=False,
        description="deployment revocation test",
    )

    conditioning, statement, statement_ref, carrier = (
        _personal_conditioning_delivery_from_config(
            active_conditioning=snapshot,
            personal_conditioning_mode="prefix_kv",
            revocation_state=ConditioningRevocationState.REVOKED,
        )
    )

    assert conditioning is None
    assert statement == ""
    assert statement_ref == ""
    assert carrier == "residual"


def test_deployment_gate_passes_complete_evidence_and_controls(
    tmp_path: Path,
) -> None:
    temporal, court, seed = _write_evidence(tmp_path)

    report = build_state_kv_deployment_report(
        temporal_causal_path=temporal,
        judge_court_path=court,
        generation_seed_path=seed,
        safety_observation=_safety(),
    )

    assert report.gate_state is DeploymentGateState.PASS
    assert all(claim.state.value == "pass" for claim in report.claims)


@pytest.mark.parametrize(
    "override",
    (
        {"cold_start_baseline_equal": False},
        {"revoked_baseline_equal": False},
        {"active_replay_equal": False},
        {"user_cache_scopes_distinct": False},
        {"rollback_baseline_equal": False},
    ),
)
def test_deployment_gate_fails_each_safety_boundary(
    tmp_path: Path,
    override,
) -> None:
    temporal, court, seed = _write_evidence(tmp_path)

    report = build_state_kv_deployment_report(
        temporal_causal_path=temporal,
        judge_court_path=court,
        generation_seed_path=seed,
        safety_observation=_safety(**override),
    )

    assert report.gate_state is DeploymentGateState.FAIL


def test_deployment_gate_fails_upstream_or_artifact_drift(
    tmp_path: Path,
) -> None:
    temporal, court, seed = _write_evidence(tmp_path)
    temporal.write_text(
        json.dumps(
            {
                "schema_version": "state-kv-temporal-causal.v1",
                "gate_state": "fail",
                "artifact_id": "wrong",
                "substrate_fingerprint": "other/model@revision",
            }
        ),
        encoding="utf-8",
    )

    report = build_state_kv_deployment_report(
        temporal_causal_path=temporal,
        judge_court_path=court,
        generation_seed_path=seed,
        safety_observation=_safety(),
    )

    assert report.gate_state is DeploymentGateState.FAIL
