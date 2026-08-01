from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from volvence_zero.agent.seven_day_companion_evidence import (
    SEVEN_DAY_ALL_ARMS,
    SevenDayCompanionAblationHarness,
    SevenDayExperimentCase,
    SevenDayRunEnvelope,
    evaluate_seven_day_ablation,
)


def _prereg() -> dict[str, object]:
    return {
        "schema_version": "seven-day-companion-simulated.v1",
        "scenario_ids": ["persona-researcher", "persona-nurse"],
        "formal_run": {"paraphrase_seeds": [1401]},
        "minimum_effects": {
            "final_day_continuity_composite_gain": 0.02,
            "callback_hit_rate_gain": 0.02,
            "cold_start_continuity_composite_gain": 0.02,
        },
    }


def _metrics(score: float) -> dict[str, object]:
    return {
        "callback_hit_rate": score,
        "boundary_violation_rate": 1.0 - score,
        "wrong_user_attribution_rate": 1.0 - score,
        "open_loop_closure_rate": score,
        "user_correction_rate": 1.0 - score,
        "remembered_item_usefulness": score,
        "seven_day_trust_delta": score * 2.0 - 1.0,
        "user_scope_hash": "scope",
        "sample_sizes": {"callback": 2},
    }


def _score(arm: str) -> float:
    return {
        "correct-user-state": 0.88,
        "stateless": 0.52,
        "swapped-user-state": 0.48,
        "shuffled-history": 0.58,
        "sleep-consolidation": 0.84,
        "no-sleep": 0.50,
    }[arm]


def _run(*, case: SevenDayExperimentCase, arm: str) -> dict[str, object]:
    score = _score(arm)
    days = []
    for day_index in range(1, 8):
        turns = []
        for exchange_index in range(1, 6):
            tags = []
            if day_index == 1 and exchange_index == 1:
                tags.append("emotion")
            if day_index == 4 and exchange_index == 1:
                tags.append("boundary")
            if day_index in (5, 7) and exchange_index == 1:
                tags.append("callback")
            turns.append(
                {
                    "exchange_index": exchange_index,
                    "user_text": (
                        f"{case.scenario_id} seed {case.paraphrase_seed} "
                        f"day {day_index} exchange {exchange_index}"
                    ),
                    "assistant_text": f"{arm} response",
                    "fsm_action": "callback_probe" if tags == ["callback"] else None,
                    "fsm_payload": (
                        "typed payload"
                        if day_index == 1 and exchange_index == 1
                        else None
                    ),
                    "event_tags": tags,
                    "fsm_probe_passed": score >= 0.8 if tags else None,
                }
            )
        policy = {
            "correct-user-state": "correct-user-state",
            "stateless": "stateless",
            "swapped-user-state": "swapped-user-state",
            "shuffled-history": "shuffled-history",
            "sleep-consolidation": "correct-user-state",
            "no-sleep": "correct-user-state",
        }[arm]
        stateful = policy != "stateless"
        restart = (
            {
                "after_day_index": day_index,
                "previous_instance_id": f"i-{day_index}",
                "next_instance_id": f"i-{day_index + 1}",
                "healthcheck_passed": True,
                "persistence_scope_unchanged": True,
                "state_intervention": {
                    "experiment_arm_label": arm,
                    "state_loading_policy": policy,
                    "after_day_index": day_index,
                    "archived_state_ref": f"archive/day-{day_index}",
                    "archived_state_sha256": "4" * 64,
                    "measurement_checkpoint_sha256": "6" * 64,
                    "next_day_source_arm": (
                        "correct-user-state" if stateful else None
                    ),
                    "next_day_source_day_index": (
                        day_index if stateful else None
                    ),
                    "next_day_loaded_state_sha256": (
                        "5" * 64 if stateful else None
                    ),
                },
            }
            if day_index < 7
            else None
        )
        days.append(
            {
                "schema_version": "seven-day-companion-day.v1",
                "run_id": f"{case.case_id}:{arm}",
                "arm_label": arm,
                "scenario_id": case.scenario_id,
                "day_index": day_index,
                "virtual_observed_at_ms": 1_800_000_000_000
                + (day_index - 1) * 86_400_000,
                "session_id": f"s-{day_index}",
                "service_instance_id": f"i-{day_index}",
                "cold_start_continuity_metrics": _metrics(score),
                "turns": turns,
                "console_probe_actions": [
                    {
                        "item_id": f"proposal:{day_index}:keep",
                        "action_id": f"action:{day_index}:keep",
                        "action": "keep",
                        "correction_kind": None,
                        "replacement_sha256": None,
                        "created_at_ms": 1_800_000_000_000
                        + (day_index - 1) * 86_400_000,
                        "status": "applied",
                    },
                    {
                        "item_id": f"proposal:{day_index}:delete",
                        "action_id": f"action:{day_index}:delete",
                        "action": "delete",
                        "correction_kind": "content_inaccurate",
                        "replacement_sha256": None,
                        "created_at_ms": 1_800_000_000_000
                        + (day_index - 1) * 86_400_000,
                        "status": "applied",
                    },
                ],
                "continuity_metrics": _metrics(score),
                "pilot_day_evidence_ref": "pilot.json",
                "pilot_day_transcript_sha256": "2" * 64,
                "end_scene_slow_loop_drained": arm != "no-sleep",
                "owner_persisted_before_restart": True,
                "restart_after_day": restart,
            }
        )
    return {
        "schema_version": "seven-day-companion-run.v1",
        "run_id": f"{case.case_id}:{arm}",
        "arm_label": arm,
        "scenario_id": case.scenario_id,
        "paraphrase_seed": case.paraphrase_seed,
        "persona_ref": case.scenario_id,
        "arc_type": "progressive_warmth",
        "user_scope_hash": "scope",
        "source_attestation": {
            "simulator_model_id": "tinyllama",
            "simulator_model_family": "llama",
            "sut_model_id": "qwen",
            "sut_model_family": "qwen",
            "model_and_adapter_fingerprint": "1" * 64,
            "consent_scope": "synthetic-no-human-subject",
            "pii_scan_artifact_sha256": "2" * 64,
            "judge_model_family": None,
        },
        "days": days,
        "event_coverage": ["boundary", "callback", "emotion"],
        "process_restart_count": 6,
        "all_restarts_exact": True,
        "simulated_longitudinal_only": True,
        "external_human_value_claim_allowed": False,
        "production_promotion_authorized": False,
    }


def _envelopes() -> list[SevenDayRunEnvelope]:
    cases = (
        SevenDayExperimentCase("persona-researcher", 1401),
        SevenDayExperimentCase("persona-nurse", 1401),
    )
    return [
        SevenDayRunEnvelope(case=case, arm_label=arm, run=_run(case=case, arm=arm))
        for case in cases
        for arm in SEVEN_DAY_ALL_ARMS
    ]


def test_two_persona_regression_covers_all_arms_and_daily_readouts() -> None:
    result = evaluate_seven_day_ablation(
        runs=_envelopes(),
        preregistration=_prereg(),
    )
    assert result.passed is True
    assert result.case_count == 2
    assert result.run_count == 12
    assert len(result.daily_readouts) == 2 * 6 * 7 * 2
    assert len(result.comparisons) == 4
    assert result.production_promotion_authorized is False
    assert result.evaluation_writeback_allowed is False


def test_exact_user_turn_matching_fails_loudly() -> None:
    runs = _envelopes()
    mutated = deepcopy(runs[1].run)
    mutated["days"][0]["turns"][0]["user_text"] = "arm-specific input"
    runs[1] = SevenDayRunEnvelope(
        case=runs[1].case,
        arm_label=runs[1].arm_label,
        run=mutated,
    )
    with pytest.raises(ValueError, match="exact user turns"):
        evaluate_seven_day_ablation(runs=runs, preregistration=_prereg())


def test_missing_owner_metric_cannot_pass_evidence_gate() -> None:
    runs = _envelopes()
    mutated = deepcopy(runs[0].run)
    mutated["days"][6]["continuity_metrics"][
        "remembered_item_usefulness"
    ] = None
    runs[0] = SevenDayRunEnvelope(
        case=runs[0].case,
        arm_label=runs[0].arm_label,
        run=mutated,
    )
    result = evaluate_seven_day_ablation(
        runs=runs,
        preregistration=_prereg(),
    )
    assert result.passed is False
    assert any(
        not passed
        for name, passed in result.gates.items()
        if "metric-coverage" in name
    )


def test_harness_freezes_arm_and_sleep_schedule(tmp_path: Path) -> None:
    calls: list[tuple[str, bool]] = []

    class Executor:
        def execute(
            self,
            *,
            case: SevenDayExperimentCase,
            arm_label: str,
            drain_slow_loop: bool,
            output_path: Path,
        ) -> dict[str, object]:
            calls.append((arm_label, drain_slow_loop))
            payload = _run(case=case, arm=arm_label)
            output_path.write_text("{}\n", encoding="utf-8")
            return payload

    result = SevenDayCompanionAblationHarness(executor=Executor()).run(
        cases=(
            SevenDayExperimentCase("persona-researcher", 1401),
            SevenDayExperimentCase("persona-nurse", 1401),
        ),
        preregistration=_prereg(),
        output_dir=tmp_path,
    )
    assert result.passed is True
    assert calls == [
        (arm, arm != "no-sleep")
        for arm in SEVEN_DAY_ALL_ARMS
        for _case in range(2)
    ]
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "daily_metrics.jsonl").is_file()
