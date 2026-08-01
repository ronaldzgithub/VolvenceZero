from __future__ import annotations

import hashlib
import json
from pathlib import Path

from volvence_zero.agent.gate1_seven_day_evidence import (
    GATE1_PE_OFF_ARM,
    GATE1_PE_ON_ARM,
    evaluate_gate1_seven_day_runs,
)
from volvence_zero.agent.gate1_seven_day_preregistration import (
    build_gate1_seven_day_preregistration,
    validate_gate1_seven_day_preregistration,
)
from volvence_zero.agent.seven_day_companion_evidence import (
    SevenDayExperimentCase,
)


def _canonical_without_newline(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _profile_attestation(profile: str) -> dict[str, object]:
    enabled = profile == GATE1_PE_ON_ARM
    payload = {
        "schema_version": "companion-evidence-runtime-profile.v1",
        "profile": profile,
        "scope": "evidence-only",
        "substrate_model_id": "frozen-sut",
        "substrate_device": "cuda",
        "intervention": {
            "external_prediction_error_drive": enabled,
            "prediction_error_readout_only": not enabled,
            "primary_prediction_error_dominance_enabled": enabled,
            "prediction_error_temporal_learning_enabled": enabled,
            "prediction_error_temporal_switch": "active",
            "prediction_error_runtime_modulation": "active",
            "prediction_error_publication": "active-in-both-arms",
            "sut_generation_temperature": 0.0,
            "production_default_changed": False,
        },
        "rollback": {
            "method": "restart-without---companion-evidence-profile",
            "production_default": "all-new-gates-disabled",
        },
        "cuda": None,
    }
    payload["attestation_sha256"] = hashlib.sha256(
        _canonical_without_newline(payload)
    ).hexdigest()
    return payload


def _metrics(*, quality: float) -> dict[str, object]:
    return {
        "callback_hit_rate": quality,
        "boundary_violation_rate": 0.0,
        "wrong_user_attribution_rate": 0.0,
        "open_loop_closure_rate": quality,
        "user_correction_rate": 0.0,
        "remembered_item_usefulness": quality,
        "seven_day_trust_delta": quality - 0.5,
    }


def _run(
    *, case: SevenDayExperimentCase, arm: str
) -> dict[str, object]:
    enabled = arm == GATE1_PE_ON_ARM
    days = []
    for day_index in range(1, 8):
        if day_index <= 2:
            magnitude = 0.60
        elif day_index >= 6:
            magnitude = 0.20 if enabled else 0.50
        else:
            magnitude = 0.40
        turns = []
        for exchange_index in range(1, 6):
            bootstrap = exchange_index == 1
            turns.append(
                {
                    "exchange_index": exchange_index,
                    "user_text": "user",
                    "assistant_text": "assistant",
                    "fsm_action": None,
                    "fsm_payload": None,
                    "event_tags": [],
                    "fsm_probe_passed": None,
                    "pe_magnitude": magnitude,
                    "pe_bootstrap": bootstrap,
                    "world_temporal_prediction_error_applied": (
                        enabled and not bootstrap
                    ),
                    "self_temporal_prediction_error_applied": (
                        enabled and not bootstrap
                    ),
                }
            )
        days.append(
            {
                "day_index": day_index,
                "turns": turns,
                "continuity_metrics": _metrics(
                    quality=0.90 if enabled else 0.60
                ),
            }
        )
    return {
        "schema_version": "seven-day-companion-run.v1",
        "run_id": f"{case.case_id}:{arm}",
        "arm_label": arm,
        "scenario_id": case.scenario_id,
        "paraphrase_seed": case.paraphrase_seed,
        "process_restart_count": 6,
        "all_restarts_exact": True,
        "production_promotion_authorized": False,
        "runtime_profile_attestation": _profile_attestation(arm),
        "days": days,
    }


def test_gate1_evaluator_requires_load_bearing_path_and_product_gain() -> None:
    cases = tuple(
        SevenDayExperimentCase("F1-seven-day-test", seed)
        for seed in (1, 2, 3)
    )
    runs = {
        (case.case_id, arm): _run(case=case, arm=arm)
        for case in cases
        for arm in (GATE1_PE_ON_ARM, GATE1_PE_OFF_ARM)
    }
    preregistration = {
        "schema_version": "gate1-seven-day-companion-prereg.v1",
        "formal_run": {"run_count": 6, "pair_count": 3},
        "minimum_effects": {
            "pe_adaptation_gain": 0.02,
            "final_day_continuity_composite_gain": 0.02,
            "maximum_safety_regression": 0.0,
        },
    }

    result = evaluate_gate1_seven_day_runs(
        cases=cases,
        runs=runs,
        preregistration=preregistration,
    )

    assert result.mechanism_supported is True
    assert result.causal_supported is True
    assert result.pe_adaptation_gain_ci95[0] > 0.0
    assert result.final_day_continuity_gain_ci95[0] > 0.0
    assert result.production_promotion_authorized is False


def test_gate1_preregistration_round_trips_current_code() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    model_common = {
        "local_files_only": True,
        "frozen": True,
        "max_new_tokens": 32,
    }
    payload = build_gate1_seven_day_preregistration(
        repo_root=repo_root,
        created_at_unix_ms=1_800_000_000_000,
        execution_device="cuda",
        sut_model={
            **model_common,
            "model_id": "sut",
            "model_family": "family-a",
            "weights_sha256": "a" * 64,
        },
        simulator_model={
            **model_common,
            "model_id": "simulator",
            "model_family": "family-b",
            "weights_sha256": "b" * 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "rendering_contract": "typed deterministic rendering",
        },
    )
    validate_gate1_seven_day_preregistration(
        payload, repo_root=repo_root
    )
    source = payload["execution_source_snapshot"]
    assert source["file_count"] > 1_000
    assert len(source["tree_sha256"]) == 64
    assert "scripts/companion_test_plan_common.py" in source["roots"]
    assert "scripts/run_seven_day_companion_test_plan.py" in source["roots"]
    assert "scripts/companion_test_plan_common.py" in payload["code_manifest"]
    assert "scripts/run_seven_day_companion_test_plan.py" in payload["code_manifest"]
