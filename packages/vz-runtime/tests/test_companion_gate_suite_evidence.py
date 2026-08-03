from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

from lifeform_service.companion_evidence_profile import (
    resolve_companion_evidence_profile,
)
from volvence_zero.agent.companion_gate_suite_evidence import (
    GATE_ARM_SCHEDULES,
    CompanionGateSuiteHarness,
    _gate10_score,
    _gate5_score,
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
                    "user_text": (
                        f"{case.case_id} day {day} exchange {exchange}"
                    ),
                    "assistant_text": f"{arm} response {day}-{exchange}",
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


def _preregistration(
    *,
    gate_id: int,
    cases: tuple[SevenDayExperimentCase, ...],
    n_plus_one_contract: dict[str, object],
) -> dict[str, object]:
    arms = GATE_ARM_SCHEDULES[gate_id]
    return {
        "schema_version": "companion-gate-suite-seven-day-prereg.v2",
        "gate_id": gate_id,
        "formal_run": {
            "pair_count": len(cases),
            "run_count": len(cases) * len(arms),
        },
        "n_plus_one_measurement": n_plus_one_contract,
        "profile_contracts": {
            arm: resolve_companion_evidence_profile(
                arm
            ).intervention_contract()
            for arm in arms
        },
        "minimum_effects": {
            "primary_gain": 0.02,
            "n_plus_one_prediction_quality_gain": 0.02,
        },
    }


def _matched_runs(
    *,
    gate_id: int,
    cases: tuple[SevenDayExperimentCase, ...],
    attach_n_plus_one: Callable[[dict[str, object], float], None],
) -> dict[tuple[str, str], dict[str, object]]:
    arms = GATE_ARM_SCHEDULES[gate_id]
    runs = {
        (case.case_id, arm): _run(gate_id, case, arm)
        for case in cases
        for arm in arms
    }
    for (_case_id, arm), run in runs.items():
        attach_n_plus_one(run, 0.9 if arm == arms[0] else 0.5)
    return runs


@pytest.mark.parametrize("gate_id", [4, 5, 6, 7, 9, 10])
def test_gate_suite_evaluator_accepts_exact_load_bearing_matrix(
    gate_id: int,
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (
        SevenDayExperimentCase("F1-01", 1),
        SevenDayExperimentCase("F1-02", 2),
        SevenDayExperimentCase("F2-01", 3),
    )
    preregistration = _preregistration(
        gate_id=gate_id,
        cases=cases,
        n_plus_one_contract=seven_day_n_plus_one_contract,
    )
    runs = _matched_runs(
        gate_id=gate_id,
        cases=cases,
        attach_n_plus_one=attach_n_plus_one,
    )

    result = evaluate_companion_gate_suite(
        gate_id=gate_id,
        cases=cases,
        runs=runs,
        preregistration=preregistration,
    )

    assert result.mechanism_supported is True
    assert result.causal_supported is True
    assert result.production_promotion_authorized is False


def test_gate_suite_fails_closed_on_missing_arm(
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    case = SevenDayExperimentCase("F1-01", 1)
    arm = GATE_ARM_SCHEDULES[4][0]
    preregistration = _preregistration(
        gate_id=4,
        cases=(case,),
        n_plus_one_contract=seven_day_n_plus_one_contract,
    )

    with pytest.raises(ValueError, match="matrix incomplete"):
        evaluate_companion_gate_suite(
            gate_id=4,
            cases=(case,),
            runs={(case.case_id, arm): _run(4, case, arm)},
            preregistration=preregistration,
        )


def test_gate9_ignores_pre_cycle_zero_gain(
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (
        SevenDayExperimentCase("F1-01", 1),
        SevenDayExperimentCase("F1-02", 2),
    )
    runs = _matched_runs(
        gate_id=9,
        cases=cases,
        attach_n_plus_one=attach_n_plus_one,
    )
    treatment = GATE_ARM_SCHEDULES[9][0]
    first_turn = runs[(cases[0].case_id, treatment)]["days"][0]["turns"][0]
    telemetry = dict(first_turn["gate_telemetry"])
    telemetry["joint_cycle_executed"] = False
    telemetry["ssl_m3_slow_gain"] = 0.0
    first_turn["gate_telemetry"] = sorted(telemetry.items())

    result = evaluate_companion_gate_suite(
        gate_id=9,
        cases=cases,
        runs=runs,
        preregistration=_preregistration(
            gate_id=9,
            cases=cases,
            n_plus_one_contract=seven_day_n_plus_one_contract,
        ),
    )

    assert result.mechanism_supported is True


def test_gate9_off_arm_requires_observed_slow_momentum(
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (
        SevenDayExperimentCase("F1-01", 1),
        SevenDayExperimentCase("F1-02", 2),
    )
    runs = _matched_runs(
        gate_id=9,
        cases=cases,
        attach_n_plus_one=attach_n_plus_one,
    )
    off_arm = GATE_ARM_SCHEDULES[9][1]
    for case in cases:
        for day in runs[(case.case_id, off_arm)]["days"]:
            for turn in day["turns"]:
                telemetry = dict(turn["gate_telemetry"])
                telemetry["ssl_m3_slow_momentum_norm"] = 0.0
                turn["gate_telemetry"] = sorted(telemetry.items())

    result = evaluate_companion_gate_suite(
        gate_id=9,
        cases=cases,
        runs=runs,
        preregistration=_preregistration(
            gate_id=9,
            cases=cases,
            n_plus_one_contract=seven_day_n_plus_one_contract,
        ),
    )

    assert result.mechanism_supported is False
    assert result.causal_supported is False


@pytest.mark.parametrize("gate_id", [7, 9])
def test_gate7_and_gate9_smoke_fail_before_formal_without_early_cycles(
    gate_id: int,
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (SevenDayExperimentCase("F1-01", 1),)
    runs = _matched_runs(
        gate_id=gate_id,
        cases=cases,
        attach_n_plus_one=attach_n_plus_one,
    )
    for run in runs.values():
        for day in run["days"][:2]:
            for turn in day["turns"]:
                telemetry = dict(turn["gate_telemetry"])
                telemetry["joint_cycle_executed"] = False
                turn["gate_telemetry"] = sorted(telemetry.items())

    with pytest.raises(ValueError, match="early/late"):
        evaluate_companion_gate_suite(
            gate_id=gate_id,
            cases=cases,
            runs=runs,
            preregistration=_preregistration(
                gate_id=gate_id,
                cases=cases,
                n_plus_one_contract=seven_day_n_plus_one_contract,
            ),
            evidence_tier="smoke",
        )


def test_gate_suite_formal_rejects_preregistered_matrix_size_drift(
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (
        SevenDayExperimentCase("F1-01", 1),
        SevenDayExperimentCase("F1-02", 2),
    )
    preregistration = _preregistration(
        gate_id=4,
        cases=cases,
        n_plus_one_contract=seven_day_n_plus_one_contract,
    )
    preregistration["formal_run"]["pair_count"] = 1
    with pytest.raises(ValueError, match="pair count drift"):
        evaluate_companion_gate_suite(
            gate_id=4,
            cases=cases,
            runs=_matched_runs(
                gate_id=4,
                cases=cases,
                attach_n_plus_one=attach_n_plus_one,
            ),
            preregistration=preregistration,
        )


@pytest.mark.parametrize("missing_field", ["event_tags", "gate_telemetry"])
def test_gate_suite_turn_contract_fails_loudly_on_missing_fields(
    missing_field: str,
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (SevenDayExperimentCase("F1-01", 1),)
    runs = _matched_runs(
        gate_id=4,
        cases=cases,
        attach_n_plus_one=attach_n_plus_one,
    )
    treatment = GATE_ARM_SCHEDULES[4][0]
    del runs[(cases[0].case_id, treatment)]["days"][0]["turns"][0][
        missing_field
    ]
    with pytest.raises(ValueError, match=("event_tags" if missing_field == "event_tags" else "gate_telemetry")):
        evaluate_companion_gate_suite(
            gate_id=4,
            cases=cases,
            runs=runs,
            preregistration=_preregistration(
                gate_id=4,
                cases=cases,
                n_plus_one_contract=seven_day_n_plus_one_contract,
            ),
            evidence_tier="smoke",
        )


def test_gate_suite_rejects_runtime_profile_sha_drift(
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (SevenDayExperimentCase("F1-01", 1),)
    runs = _matched_runs(
        gate_id=4,
        cases=cases,
        attach_n_plus_one=attach_n_plus_one,
    )
    treatment = GATE_ARM_SCHEDULES[4][0]
    runs[(cases[0].case_id, treatment)]["runtime_profile_attestation"][
        "scope"
    ] = "tampered"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_companion_gate_suite(
            gate_id=4,
            cases=cases,
            runs=runs,
            preregistration=_preregistration(
                gate_id=4,
                cases=cases,
                n_plus_one_contract=seven_day_n_plus_one_contract,
            ),
            evidence_tier="smoke",
        )


def test_gate_suite_rejects_arm_dependent_n_plus_one_target(
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (SevenDayExperimentCase("F1-01", 1),)
    runs = _matched_runs(
        gate_id=4,
        cases=cases,
        attach_n_plus_one=attach_n_plus_one,
    )
    control = GATE_ARM_SCHEDULES[4][1]
    runs[(cases[0].case_id, control)]["n_plus_one_representation_evidence"][
        "target_lineage"
    ]["snapshot_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="target snapshot differs"):
        evaluate_companion_gate_suite(
            gate_id=4,
            cases=cases,
            runs=runs,
            preregistration=_preregistration(
                gate_id=4,
                cases=cases,
                n_plus_one_contract=seven_day_n_plus_one_contract,
            ),
            evidence_tier="smoke",
        )


def test_gate_suite_safety_regression_is_a_hard_failure(
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (
        SevenDayExperimentCase("F1-01", 1),
        SevenDayExperimentCase("F1-02", 2),
    )
    runs = _matched_runs(
        gate_id=4,
        cases=cases,
        attach_n_plus_one=attach_n_plus_one,
    )
    treatment = GATE_ARM_SCHEDULES[4][0]
    runs[(cases[0].case_id, treatment)]["days"][-1]["continuity_metrics"][
        "boundary_violation_rate"
    ] = 0.2
    result = evaluate_companion_gate_suite(
        gate_id=4,
        cases=cases,
        runs=runs,
        preregistration=_preregistration(
            gate_id=4,
            cases=cases,
            n_plus_one_contract=seven_day_n_plus_one_contract,
        ),
    )
    assert result.gates["safety-noninferior-all-controls"] is False
    assert result.causal_supported is False


def test_gate_suite_missing_secondary_continuity_metric_does_not_block_primary(
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    cases = (
        SevenDayExperimentCase("F1-01", 1),
        SevenDayExperimentCase("F1-02", 2),
    )
    runs = _matched_runs(
        gate_id=4,
        cases=cases,
        attach_n_plus_one=attach_n_plus_one,
    )
    treatment = GATE_ARM_SCHEDULES[4][0]
    runs[(cases[0].case_id, treatment)]["days"][-1][
        "continuity_metrics"
    ]["remembered_item_usefulness"] = None
    result = evaluate_companion_gate_suite(
        gate_id=4,
        cases=cases,
        runs=runs,
        preregistration=_preregistration(
            gate_id=4,
            cases=cases,
            n_plus_one_contract=seven_day_n_plus_one_contract,
        ),
    )

    assert result.causal_supported is True
    assert result.readouts[0].final_day_continuity_composite is None


def test_gate5_and_gate10_empty_windows_raise_contract_errors() -> None:
    with pytest.raises(ValueError, match="Gate 5 lacks"):
        _gate5_score(())
    with pytest.raises(ValueError, match="Gate 10 lacks"):
        _gate10_score(())


def test_gate_suite_smoke_uses_distinct_schema_and_filename(
    tmp_path: Path,
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    case = SevenDayExperimentCase("F1-01", 1)

    class Executor:
        def execute(
            self,
            *,
            case: SevenDayExperimentCase,
            arm_label: str,
            drain_slow_loop: bool,
            output_path: Path,
        ) -> dict[str, object]:
            assert drain_slow_loop is True
            run = _run(4, case, arm_label)
            attach_n_plus_one(
                run,
                0.9 if arm_label == GATE_ARM_SCHEDULES[4][0] else 0.5,
            )
            output_path.write_text("{}\n", encoding="utf-8")
            return run

    result = CompanionGateSuiteHarness(gate_id=4, executor=Executor()).run(
        cases=(case,),
        preregistration=_preregistration(
            gate_id=4,
            cases=(case,),
            n_plus_one_contract=seven_day_n_plus_one_contract,
        ),
        output_dir=tmp_path,
        evidence_tier="smoke",
    )
    assert result.schema_version == "companion-gate-suite-seven-day-smoke.v1"
    assert result.evidence_tier == "smoke"
    assert (tmp_path / "gate4_smoke_evaluation.json").is_file()
    assert not (tmp_path / "gate4_evaluation.json").exists()
