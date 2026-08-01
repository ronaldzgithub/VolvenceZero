from __future__ import annotations

import json

import pytest

from lifeform_service import cli
from lifeform_service.app import create_app
from lifeform_service.companion_evidence_profile import (
    GATE10_RARE_HEAVY_IMPORT,
    GATE10_RARE_HEAVY_REVIEW,
    GATE1_PE_TEMPORAL_OFF,
    GATE1_PE_TEMPORAL_ON,
    GATE4_ACTIVE_SELECTOR,
    GATE4_RANDOM_FEEDBACK,
    GATE5_MULTIFREQUENCY_CMS,
    GATE5_SINGLE_TIMESCALE,
    GATE6_CONDITIONED_META_INIT,
    GATE6_COPY_INIT,
    GATE7_NO_RL,
    GATE7_NO_SSL,
    GATE7_SSL_RL_FULL,
    GATE9_M3_SLOW_OFF,
    GATE9_M3_SLOW_ON,
    resolve_companion_evidence_profile,
    write_companion_evidence_profile_attestation,
)
from lifeform_service.verticals import _try_companion
from volvence_zero.brain import BrainConfig
from volvence_zero.runtime import WiringLevel


def test_gate1_profiles_keep_publication_and_match_non_pe_runtime() -> None:
    base = BrainConfig()
    pe_on = resolve_companion_evidence_profile(GATE1_PE_TEMPORAL_ON).apply(base)
    pe_off = resolve_companion_evidence_profile(GATE1_PE_TEMPORAL_OFF).apply(base)

    assert pe_on.external_prediction_error_drive is True
    assert pe_off.external_prediction_error_drive is False
    assert pe_on.prediction_error_readout_only is False
    assert pe_off.prediction_error_readout_only is True
    assert pe_on.primary_prediction_error_dominance_enabled is True
    assert pe_off.primary_prediction_error_dominance_enabled is False
    assert pe_on.final_rollout_config == pe_off.final_rollout_config
    assert pe_on.final_rollout_config.prediction_error_temporal_switch is WiringLevel.ACTIVE
    assert pe_on.final_rollout_config.prediction_error_runtime_modulation is WiringLevel.ACTIVE
    assert BrainConfig().final_rollout_config is None


def test_profiled_companion_factory_threads_brain_configuration() -> None:
    spec = _try_companion(evidence_profile=GATE1_PE_TEMPORAL_OFF)
    assert spec is not None
    lifeform = spec.factory(None)

    config = lifeform.brain.config
    assert config.external_prediction_error_drive is False
    assert config.prediction_error_readout_only is True
    assert config.final_rollout_config.prediction_error_runtime_modulation is WiringLevel.ACTIVE


def test_evidence_profile_cli_rejects_non_isolated_startup(capsys) -> None:
    rc = cli.main(
        [
            "--companion-evidence-profile",
            GATE1_PE_TEMPORAL_ON,
        ]
    )

    assert rc == 1
    assert "--alpha-enabled is required" in capsys.readouterr().err


def test_profile_attestation_is_immutable_and_machine_readable(tmp_path) -> None:
    profile = resolve_companion_evidence_profile(GATE1_PE_TEMPORAL_ON)
    first = write_companion_evidence_profile_attestation(
        output_dir=tmp_path,
        profile=profile,
        substrate_model_id="local/frozen-qwen",
        substrate_device="mps",
    )
    second = write_companion_evidence_profile_attestation(
        output_dir=tmp_path,
        profile=profile,
        substrate_model_id="local/frozen-qwen",
        substrate_device="mps",
    )

    assert first == second
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["profile"] == GATE1_PE_TEMPORAL_ON
    assert payload["intervention"]["prediction_error_publication"] == ("active-in-both-arms")
    assert len(payload["attestation_sha256"]) == 64


def test_remaining_gate_profiles_select_owner_level_interventions() -> None:
    base = BrainConfig()
    gate4_on = resolve_companion_evidence_profile(GATE4_ACTIVE_SELECTOR)
    gate4_control = resolve_companion_evidence_profile(GATE4_RANDOM_FEEDBACK)
    assert gate4_on.turn_trigger_kind == gate4_control.turn_trigger_kind == "apprentice"
    assert gate4_on.apply(base).apprenticeship_feedback_policy == "owner"
    assert gate4_control.apply(base).apprenticeship_feedback_policy == "random"

    gate5_on = resolve_companion_evidence_profile(GATE5_MULTIFREQUENCY_CMS).apply(base)
    gate5_control = resolve_companion_evidence_profile(GATE5_SINGLE_TIMESCALE).apply(base)
    assert (gate5_on.cms_variant, gate5_on.cms_session_cadence) == ("nested", 2)
    assert (gate5_control.cms_variant, gate5_control.cms_session_cadence) == (
        "independent",
        1,
    )
    assert gate5_on.cms_pe_features_enabled == gate5_control.cms_pe_features_enabled

    gate6_on = resolve_companion_evidence_profile(GATE6_CONDITIONED_META_INIT).apply(base)
    gate6_control = resolve_companion_evidence_profile(GATE6_COPY_INIT).apply(base)
    assert gate6_on.cms_context_conditioned_meta_init is True
    assert gate6_on.nested_context_reset_mode == "meta-init"
    assert gate6_control.cms_context_conditioned_meta_init is False
    assert gate6_control.nested_context_reset_mode == "copy-init"

    gate7_full = resolve_companion_evidence_profile(GATE7_SSL_RL_FULL).apply(base)
    gate7_no_ssl = resolve_companion_evidence_profile(GATE7_NO_SSL).apply(base)
    gate7_no_rl = resolve_companion_evidence_profile(GATE7_NO_RL).apply(base)
    assert gate7_full.joint_apply_ssl_optimization is True
    assert gate7_no_ssl.joint_apply_ssl_optimization is False
    assert gate7_full.joint_apply_policy_optimization is True
    assert gate7_no_rl.joint_apply_policy_optimization is False

    gate9_on = resolve_companion_evidence_profile(GATE9_M3_SLOW_ON).apply(base)
    gate9_off = resolve_companion_evidence_profile(GATE9_M3_SLOW_OFF).apply(base)
    assert gate9_on.final_rollout_config.temporal_ssl_m3_slow_gain == 1.0
    assert gate9_off.final_rollout_config.temporal_ssl_m3_slow_gain == 0.0

    gate10_on_profile = resolve_companion_evidence_profile(GATE10_RARE_HEAVY_IMPORT)
    gate10_off_profile = resolve_companion_evidence_profile(GATE10_RARE_HEAVY_REVIEW)
    gate10_on = gate10_on_profile.apply(base)
    gate10_off = gate10_off_profile.apply(base)
    assert gate10_on.allow_live_substrate_mutation is True
    assert gate10_off.allow_live_substrate_mutation is False
    assert gate10_on_profile.allow_single_session_live_substrate_mutation is True
    assert gate10_off_profile.allow_single_session_live_substrate_mutation is False


class _MutableEvidenceRuntime:
    supports_live_substrate_mutation = True
    model_id = "local/mutable-evidence-runtime"


def test_mutable_shared_runtime_is_only_allowed_for_gate10_single_session() -> None:
    runtime = _MutableEvidenceRuntime()
    with pytest.raises(ValueError, match="Cannot share a runtime"):
        create_app(
            substrate_runtime=runtime,
            companion_evidence_profile=GATE1_PE_TEMPORAL_ON,
            allow_evidence_single_session_mutation=True,
            max_sessions=1,
        )
    with pytest.raises(ValueError, match="Cannot share a runtime"):
        create_app(
            substrate_runtime=runtime,
            companion_evidence_profile=GATE10_RARE_HEAVY_IMPORT,
            allow_evidence_single_session_mutation=True,
            max_sessions=2,
        )
    app = create_app(
        vertical=_try_companion(evidence_profile=GATE10_RARE_HEAVY_IMPORT),
        substrate_runtime=runtime,
        companion_evidence_profile=GATE10_RARE_HEAVY_IMPORT,
        allow_evidence_single_session_mutation=True,
        max_sessions=1,
    )
    assert app["companion_evidence_profile"] == GATE10_RARE_HEAVY_IMPORT


async def test_product_http_path_publishes_typed_gate_telemetry(
    aiohttp_client,
) -> None:
    app = create_app(
        vertical=_try_companion(evidence_profile=GATE5_MULTIFREQUENCY_CMS),
        companion_evidence_profile=GATE5_MULTIFREQUENCY_CMS,
        max_sessions=1,
    )
    client = await aiohttp_client(app)
    created = await client.post("/v1/sessions", json={"session_id": "gate-suite-http"})
    assert created.status == 201

    response = await client.post(
        "/v1/sessions/gate-suite-http/turns",
        json={"user_input": "I want to revisit the plan we discussed."},
    )
    assert response.status == 200
    payload = await response.json()
    telemetry = payload["evidence_telemetry"]
    assert telemetry["cms_variant"] == "nested"
    assert telemetry["cms_atlas_replay_active"] is True
    assert telemetry["cms_pe_gate_active"] is True

    ended = await client.post(
        "/v1/sessions/gate-suite-http/end-scene",
        json={"drain_slow_loop": True},
    )
    assert ended.status == 200
    end_payload = await ended.json()
    assert isinstance(
        end_payload["evidence_telemetry"]["nested_context_reset_applied"],
        bool,
    )
