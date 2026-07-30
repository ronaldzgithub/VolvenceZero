from __future__ import annotations

import json
from pathlib import Path

import pytest

from volvence_zero.agent.gate11_longitudinal_source import (
    GATE11_LONGITUDINAL_SOURCE_SEEDS,
    build_gate11_longitudinal_source_plans,
)
from volvence_zero.agent.gate8_longitudinal_evidence import (
    GATE8_LONGITUDINAL_REQUIRED_FILES,
    _compare_arms,
    _record_to_plan,
    verify_gate8_longitudinal_bundle,
)
from volvence_zero.agent.gate8_wake_sleep_evidence import GATE8_ARMS


def _metric(seed: int, arm: str, payoff: float):
    from volvence_zero.agent.gate8_longitudinal_evidence import (
        Gate8LongitudinalArmMetric,
    )

    callback = 1.0 if arm in {
        "sleep-consolidation",
        "memory-only-sleep",
    } else 0.0
    return Gate8LongitudinalArmMetric(
        seed=seed,
        arm=arm,
        settled_transition_count=510,
        consumer_session_count=51,
        constructor_restart_count=50,
        memory_entry_count=510 if callback else 0,
        isolated_memory_entry_count=(
            510 if arm == "policy-only-sleep" else 0
        ),
        temporal_operation_count=(
            510
            if arm in {"sleep-consolidation", "policy-only-sleep"}
            else 0
        ),
        next_session_cold_start_loss=1.0 - payoff,
        callback_commitment_consistency=callback,
        temporal_policy_alignment=payoff,
        delayed_payoff=payoff,
        owner_state_drift=0.25,
        prompt_token_increment=0,
        unique_job_count=510,
        worker_execution_count=510,
        duplicate_job_count=51,
        duplicate_job_execution_count=0,
        owner_lineage_expected_count=(
            1020
            if arm == "sleep-consolidation"
            else 510
            if arm != "no-sleep"
            else 0
        ),
        owner_lineage_observed_count=(
            1020
            if arm == "sleep-consolidation"
            else 510
            if arm != "no-sleep"
            else 0
        ),
        owner_writeback_lineage_coverage=1.0,
        turn_latency_ms=1.0,
        slow_job_latency_ms=2.0,
        turn_latency_contains_slow_job=False,
        persistence_roundtrip_exact=True,
        rollback_exact=True,
        rollback_fingerprint_before="same",
        rollback_fingerprint_after="same",
        frozen_substrate_mutation_count=0,
    )


def test_compare_arms_requires_minimum_effect_and_positive_ci() -> None:
    payoffs = {
        "sleep-consolidation": 0.80,
        "no-sleep": 0.50,
        "memory-only-sleep": 0.65,
        "policy-only-sleep": 0.70,
    }
    metrics = tuple(
        _metric(seed, arm, payoffs[arm])
        for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS
        for arm in GATE8_ARMS
    )

    aggregate, confidence, gates, status = _compare_arms(metrics)

    assert status == "longitudinal-supported"
    assert aggregate["delayed_payoff_gain_vs_no_sleep"] == pytest.approx(
        0.30
    )
    assert gates["paired_seed_ci_lower_positive"] is True
    assert (
        confidence["payoff_margin_vs_policy_only"][
            "confidence_interval_95"
        ][0]
        > 0.0
    )


def test_record_adapter_uses_structured_owner_fields() -> None:
    plan = build_gate11_longitudinal_source_plans(1201)[0]
    record = {
        "transition_id": plan.transition_id,
        "seed": plan.seed,
        "global_index": plan.global_index,
        "partition": plan.partition,
        "context_id": plan.context_id,
        "domain": plan.domain,
        "user_id": plan.user_id,
        "knowledge_key": plan.knowledge_key,
        "input": {
            "prediction_turn": plan.turn_one_text,
            "settlement_turn": plan.turn_two_text,
        },
        "prediction_error": {"magnitude": 0.4},
        "temporal_snapshot": {
            "active_abstract_action": "family-a",
            "closed_segments": [
                {
                    "abstract_action_id": "family-b",
                    "open_turn_index": 0,
                    "close_turn_index": 2,
                }
            ],
            "memory_feedback_signal": [0.1, 0.2, 0.3, 0.4],
            "controller_state": {
                "code": [0.3, 0.4, 0.5],
                "switch_gate": 0.8,
            },
        },
    }

    adapted = _record_to_plan(record)

    assert adapted.episode_id == plan.transition_id
    assert adapted.user_prior == (0.3, 0.4, 0.5, 0.8)
    assert adapted.action_family_ids == ("family-b",)
    assert plan.knowledge_key in adapted.session_two_turns[0]


def test_verify_bundle_rejects_missing_files(tmp_path: Path) -> None:
    result = verify_gate8_longitudinal_bundle(tmp_path)

    assert result["passed"] is False
    assert tuple(result["missing_files"]) == GATE8_LONGITUDINAL_REQUIRED_FILES


def test_verify_bundle_accepts_complete_development_packet(
    tmp_path: Path,
) -> None:
    for name in GATE8_LONGITUDINAL_REQUIRED_FILES:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.yaml").write_text(
        json.dumps(
            {
                "schema_version": "gate8-wake-sleep-longitudinal.v1",
                "seed_schedule": list(GATE11_LONGITUDINAL_SOURCE_SEEDS),
                "arm_schedule": list(GATE8_ARMS),
                "required_files": list(GATE8_LONGITUDINAL_REQUIRED_FILES),
                "formal_locked_run": False,
                "arm_transition_count": 120,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "promotion_verdict.json").write_text(
        json.dumps({"status": "development-supported"}),
        encoding="utf-8",
    )

    result = verify_gate8_longitudinal_bundle(tmp_path)

    assert result["passed"] is True
