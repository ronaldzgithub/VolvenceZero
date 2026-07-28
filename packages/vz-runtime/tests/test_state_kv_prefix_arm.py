"""Arm G (prefix-KV carrier) wiring and claim parameterisation.

The prefix arm exists so the identification claims can be recomputed against a
higher-bandwidth carrier without changing anything else: same control arm, same
probes, same substrate, same claim logic. These tests pin the parts that make
that comparison honest -- the arm's settings come from the registry, the
carrier reaches the substrate, the claims can be pointed at either candidate,
and the artifact records which one was on trial.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from volvence_zero.agent.profile_registry import (
    list_builtin_profiles,
    resolve_profile,
)
from volvence_zero.agent.response import ResponseContext
from volvence_zero.agent.session_observation import (
    _personal_conditioning_delivery_from_config,
)
from volvence_zero.application.runtime import (
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.state_kv_identification import (
    ALL_ARM_LABELS,
    CONTROL_ARM_LABEL,
    DEFAULT_CANDIDATE_ARM_LABEL,
    PREFIX_ARM_LABEL,
    PREFIX_IDENTIFICATION_ARM_LABELS,
    ArmObservation,
    ClaimState,
    IdentificationTurn,
    ProbeCase,
    SubstrateEvidenceKind,
    arm_from_profile,
    build_identification_verdict,
    context_for_arm,
)
from volvence_zero.state_kv_deployment import (
    STATE_KV_DEPLOYMENT_ARTIFACT_ID,
    STATE_KV_DEPLOYMENT_PROFILE_LABEL,
)


def _assembly() -> ResponseAssemblySnapshot:
    return ResponseAssemblySnapshot(
        regime_id="steady",
        regime_name="Steady",
        abstract_action=None,
        response_mode=ResponseMode.SUPPORT,
        answer_depth_limit="high-level-only",
        citation_mode="none",
        clarification_required=False,
        refer_out_required=False,
        ordering_plan=(),
        knowledge_briefs=(),
        case_briefs=(),
        playbook_ordering=(),
        required_disclaimers=(),
        required_disclaimer_phrases=(),
        control_code=(),
        control_scale=0.0,
        max_questions=0,
        prompt_residue_summary="",
        prompt_residue_ratio=0.0,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="prefix arm test assembly",
        ordering_driver="playbook-only",
    )


def _case() -> ProbeCase:
    return ProbeCase(
        user_id="alice",
        probe_id="p0",
        user_input="我又搞砸了",
        conditioning=PersonalConditioningSnapshot(
            schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
            state_vector=tuple(
                0.6 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS
            ),
            vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
            source_versions=(("user_model", 1),),
            source_fingerprint="prefix-arm-test",
            confidence=0.72,
            is_cold_start=False,
            description="prefix arm test state",
            rendered_statement="state readout",
        ),
        assembly=_assembly(),
    )


def _base_context() -> ResponseContext:
    return ResponseContext(
        regime_id="steady",
        regime_name="Steady",
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="none",
    )


def test_prefix_arm_settings_come_from_the_registry() -> None:
    arm = arm_from_profile(PREFIX_ARM_LABEL)

    assert arm.conditioning_active is True
    assert arm.conditioning_mode == "prefix_kv"
    # Arm G is only comparable to arm A-pure if it closes the prompt carrier
    # too; otherwise claim 1 could never hold for it.
    assert arm.prompt_state_delivery == "suppressed"


def test_prefix_arm_delivers_the_snapshot_through_the_prefix_carrier() -> None:
    context = context_for_arm(
        arm=arm_from_profile(PREFIX_ARM_LABEL),
        case=_case(),
        base_context=_base_context(),
    )

    assert context.personal_conditioning is not None
    assert context.personal_conditioning_carrier == "prefix_kv"
    # The rendered statement belongs to the text arm; delivering both would
    # make the arm untestable as a latent carrier.
    assert context.personal_conditioning_statement == ""


def test_residual_arm_keeps_the_default_carrier() -> None:
    context = context_for_arm(
        arm=arm_from_profile(DEFAULT_CANDIDATE_ARM_LABEL),
        case=_case(),
        base_context=_base_context(),
    )

    assert context.personal_conditioning_carrier == "residual"


def test_session_observation_selects_prefix_carrier_for_active_prefix_mode() -> None:
    case = _case()

    conditioning, statement, statement_ref, carrier = (
        _personal_conditioning_delivery_from_config(
            active_conditioning=case.conditioning,
            personal_conditioning_mode="prefix_kv",
        )
    )

    assert conditioning is case.conditioning
    assert statement == ""
    assert statement_ref == ""
    assert carrier == "prefix_kv"


def test_session_observation_rollback_delivers_no_state_without_active_snapshot() -> None:
    conditioning, statement, statement_ref, carrier = (
        _personal_conditioning_delivery_from_config(
            active_conditioning=None,
            personal_conditioning_mode="prefix_kv",
        )
    )

    assert conditioning is None
    assert statement == ""
    assert statement_ref == ""
    assert carrier == "residual"


def test_session_observation_text_mode_requires_rendered_statement() -> None:
    case = _case()

    with pytest.raises(ValueError, match="rendered_statement"):
        _personal_conditioning_delivery_from_config(
            active_conditioning=replace(case.conditioning, rendered_statement=""),
            personal_conditioning_mode="text",
        )


def test_response_context_rejects_an_unknown_carrier() -> None:
    with pytest.raises(ValueError, match="personal_conditioning_carrier"):
        ResponseContext(
            regime_id="steady",
            regime_name="Steady",
            regime_switched=False,
            abstract_action=None,
            alert_count=0,
            temporal_switch_gate=0.0,
            temporal_is_switching=False,
            reflection_lesson_count=0,
            reflection_tension_count=0,
            reflection_writeback_applied=False,
            primary_reflection_lesson=None,
            primary_reflection_tension=None,
            joint_schedule_action="none",
            personal_conditioning_carrier="soft-prompt",
        )


def test_prefix_carrier_is_limited_to_evidence_and_bound_deployment() -> None:
    """Only the explicit evidence arm and bound deployment may select prefix-KV.

    The deployment profile must bind the promoted artifact exactly; all other
    profiles stay off the prefix carrier.
    """

    offenders = [
        label
        for label in list_builtin_profiles()
        if resolve_profile(label).merged_flag_overrides.get(
            "personal_conditioning_mode"
        )
        == "prefix_kv"
        and label not in (PREFIX_ARM_LABEL, STATE_KV_DEPLOYMENT_PROFILE_LABEL)
    ]

    assert offenders == []
    deployment = resolve_profile(STATE_KV_DEPLOYMENT_PROFILE_LABEL)
    assert (
        deployment.merged_flag_overrides[
            "personal_conditioning_prefix_artifact_id"
        ]
        == STATE_KV_DEPLOYMENT_ARTIFACT_ID
    )


def _turn(*, arm: str, user: str, text: str) -> IdentificationTurn:
    return IdentificationTurn(
        arm_label=arm,
        user_id=user,
        probe_id="p0",
        prompt_fp="fp-shared",
        prompt_state_sections=0,
        decode_fp="decode-same",
        sampling_seed=None,
        conditioning_applied=arm != CONTROL_ARM_LABEL,
        conditioning_delivered=arm != CONTROL_ARM_LABEL,
        text=text,
    )


def _observation(label: str, *, diverges: bool) -> ArmObservation:
    return ArmObservation(
        arm=arm_from_profile(label),
        turns=tuple(
            _turn(
                arm=label,
                user=user,
                text=f"{label}-{user}" if diverges else f"{label}-same",
            )
            for user in ("alice", "bob")
        ),
    )


def _verdict(candidate: str, *, prefix_diverges: bool):
    return build_identification_verdict(
        observations=[
            _observation(CONTROL_ARM_LABEL, diverges=False),
            _observation(DEFAULT_CANDIDATE_ARM_LABEL, diverges=False),
            _observation(PREFIX_ARM_LABEL, diverges=prefix_diverges),
        ],
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        substrate_fingerprint="Qwen/test@deadbeef",
        candidate_arm_label=candidate,
    )


def test_claims_follow_the_candidate_arm() -> None:
    residual = _verdict(DEFAULT_CANDIDATE_ARM_LABEL, prefix_diverges=True)
    prefix = _verdict(PREFIX_ARM_LABEL, prefix_diverges=True)

    # Same run, same turns: only the arm under test changes the verdict. The
    # residual arm produced identical text across users; arm G did not.
    assert (
        residual.claim("claim_output_divergence").state is ClaimState.FAIL
    )
    assert prefix.claim("claim_output_divergence").state is ClaimState.PASS
    assert prefix.claim("claim_prompt_identity").state is ClaimState.PASS


def test_verdict_records_which_carrier_was_on_trial() -> None:
    payload = _verdict(PREFIX_ARM_LABEL, prefix_diverges=True).as_json_dict()

    assert payload["candidate_arm"] == PREFIX_ARM_LABEL


def test_prefix_arm_still_fails_divergence_when_outputs_match() -> None:
    prefix = _verdict(PREFIX_ARM_LABEL, prefix_diverges=False)

    assert prefix.claim("claim_output_divergence").state is ClaimState.FAIL


def test_unknown_candidate_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown candidate arm"):
        _verdict("state-kv-arm-does-not-exist", prefix_diverges=True)


def test_prefix_lane_extends_the_four_arm_order() -> None:
    assert PREFIX_IDENTIFICATION_ARM_LABELS[-1] == PREFIX_ARM_LABEL
    assert PREFIX_ARM_LABEL in ALL_ARM_LABELS
