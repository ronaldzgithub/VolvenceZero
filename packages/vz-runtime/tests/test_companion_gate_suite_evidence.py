from __future__ import annotations

import hashlib
import json

import pytest

from lifeform_service.companion_evidence_profile import (
    resolve_companion_evidence_profile,
)
from volvence_zero.agent.companion_gate_suite_evidence import (
    GATE_ARM_SCHEDULES,
    evaluate_companion_gate_suite,
)
from volvence_zero.agent.seven_day_companion_evidence import (
    SEVEN_DAY_METRICS,
    SevenDayExperimentCase,
)


def _attestation(arm: str) -> dict[str, object]:
    profile = resolve_companion_evidence_profile(arm)
    payload: dict[str, object] = {
        "schema_version": "companion-evidence-runtime-profile.v1",
        "profile": arm,
        "scope": "evidence-only",
        "substrate_model_id": "frozen/sut",
        "substrate_device": "mps",
        "intervention": profile.intervention_contract(),
        "rollback": {
            "method": "restart-without---companion-evidence-profile",
            "production_default": "all-new-gates-disabled",
        },
        "cuda": None,
    }
    payload["attestation_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _continuity(*, treatment: bool) -> dict[str, float]:
    return {
        name: (
            (0.0 if treatment else 0.1)
            if name
            in {
                "boundary_violation_rate",
                "wrong_user_attribution_rate",
                "user_correction_rate",
            }
            else (0.8 if treatment else 0.6)
        )
        for name in SEVEN_DAY_METRICS
    }


def _telemetry(gate_id: int, arm: str, *, day: int) -> dict[str, object]:
    treatment = arm == GATE_ARM_SCHEDULES[gate_id][0]
    if gate_id == 4:
        return {
            "apprenticeship_active": True,
            "environment_trigger_kind": "apprentice",
            "feedback_requested": True,
        }
    if gate_id == 5:
        return {
            "cms_variant": "nested" if treatment else "independent",
            "cms_atlas_replay_active": True,
            "cms_pe_gate_active": True,
            "cms_total_observations": day,
            "cms_new_knowledge_absorption": 0.8 if treatment else 0.5,
            "cms_old_knowledge_retention": 0.9 if treatment else 0.6,
        }
    if gate_id == 7:
        no_ssl = arm == "gate7-no-ssl-v1"
        no_rl = arm == "gate7-no-rl-v1"
        early_reward = 0.1
        late_reward = 0.5 if treatment else (0.2 if no_ssl else 0.15)
        return {
            "joint_cycle_executed": True,
            "ssl_rollback_applied": no_ssl,
            "internal_rl_policy_update_applied": not no_rl,
            "internal_rl_total_reward": (early_reward if day <= 2 else late_reward),
            "runtime_replay_wiring": "active" if treatment else "disabled",
            "runtime_replay_transition_count": 1 if treatment else 0,
        }
    if gate_id == 9:
        return {
            "joint_cycle_executed": True,
            "ssl_prediction_loss": (0.8 if day <= 2 else (0.3 if treatment else 0.7)),
            "ssl_m3_slow_gain": 1.0 if treatment else 0.0,
            "ssl_m3_slow_momentum_norm": 0.2,
        }
    if gate_id == 10:
        return {
            "rare_heavy_recommended": True,
            "rare_heavy_applied": treatment,
            "rare_heavy_import_decision": ("imported" if treatment else "blocked-by-doctrine"),
            "rare_heavy_pre_import_passed": treatment,
        }
    return {}


def _run(gate_id: int, case: SevenDayExperimentCase, arm: str) -> dict[str, object]:
    treatment = arm == GATE_ARM_SCHEDULES[gate_id][0]
    days = []
    for day in range(1, 8):
        turns = []
        for exchange in range(1, 6):
            telemetry = _telemetry(gate_id, arm, day=day)
            pe = 0.4
            if gate_id in {6, 10}:
                pe = 0.8 if day <= 2 else (0.2 if treatment else 0.7)
            turns.append(
                {
                    "exchange_index": exchange,
                    "event_tags": (["boundary", "callback"] if gate_id != 4 or treatment else ["neutral"]),
                    "pe_magnitude": pe,
                    "gate_telemetry": sorted(telemetry.items()),
                }
            )
        end_telemetry = {
            "nested_context_reset_applied": gate_id == 6,
            "nested_context_reset_meta_init": gate_id == 6 and treatment,
            "nested_context_reset_copy_init": gate_id == 6 and not treatment,
            "nested_context_reset_conditioned": gate_id == 6 and treatment,
            "nested_context_reset_prototype_count": (2 if gate_id == 6 and treatment else 0),
            "nested_context_reset_context_match_score": (0.8 if gate_id == 6 and treatment else 0.0),
            "slow_to_fast_target_alignment_gain": (0.2 if gate_id == 6 and treatment else 0.0),
        }
        days.append(
            {
                "day_index": day,
                "turns": turns,
                "continuity_metrics": _continuity(treatment=treatment),
                "end_scene_gate_telemetry": sorted(end_telemetry.items()),
            }
        )
    return {
        "schema_version": "seven-day-companion-run.v1",
        "arm_label": arm,
        "scenario_id": case.scenario_id,
        "paraphrase_seed": case.paraphrase_seed,
        "process_restart_count": 6,
        "all_restarts_exact": True,
        "production_promotion_authorized": False,
        "days": days,
        "runtime_profile_attestation": _attestation(arm),
    }


@pytest.mark.parametrize("gate_id", [4, 5, 6, 7, 9, 10])
def test_gate_suite_evaluator_accepts_exact_load_bearing_matrix(
    gate_id: int,
) -> None:
    cases = (
        SevenDayExperimentCase("F1-01", 1),
        SevenDayExperimentCase("F1-02", 2),
        SevenDayExperimentCase("F2-01", 3),
    )
    arms = GATE_ARM_SCHEDULES[gate_id]
    preregistration = {
        "schema_version": "companion-gate-suite-seven-day-prereg.v1",
        "gate_id": gate_id,
        "profile_contracts": {arm: resolve_companion_evidence_profile(arm).intervention_contract() for arm in arms},
        "minimum_effects": {
            "primary_gain": 0.02,
            "continuity_gain": 0.02,
        },
    }
    runs = {(case.case_id, arm): _run(gate_id, case, arm) for case in cases for arm in arms}

    result = evaluate_companion_gate_suite(
        gate_id=gate_id,
        cases=cases,
        runs=runs,
        preregistration=preregistration,
    )

    assert result.mechanism_supported is True
    assert result.causal_supported is True
    assert result.production_promotion_authorized is False


def test_gate_suite_fails_closed_on_missing_arm() -> None:
    case = SevenDayExperimentCase("F1-01", 1)
    arm = GATE_ARM_SCHEDULES[4][0]
    preregistration = {
        "schema_version": "companion-gate-suite-seven-day-prereg.v1",
        "gate_id": 4,
        "profile_contracts": {
            name: resolve_companion_evidence_profile(name).intervention_contract() for name in GATE_ARM_SCHEDULES[4]
        },
        "minimum_effects": {
            "primary_gain": 0.02,
            "continuity_gain": 0.02,
        },
    }

    with pytest.raises(ValueError, match="matrix incomplete"):
        evaluate_companion_gate_suite(
            gate_id=4,
            cases=(case,),
            runs={(case.case_id, arm): _run(4, case, arm)},
            preregistration=preregistration,
        )
