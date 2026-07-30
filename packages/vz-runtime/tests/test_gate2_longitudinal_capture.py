from __future__ import annotations

import json

import pytest

from volvence_zero.agent.gate2_longitudinal_capture import (
    gate2_permutation_action_index,
    load_gate2_candidate_control_contract,
    summarize_gate2_longitudinal_seed,
)


def test_permutation_schedule_is_balanced_and_seed_shifted() -> None:
    schedules = {
        seed: tuple(
            gate2_permutation_action_index(
                seed=seed,
                global_index=index,
            )
            for index in range(510)
        )
        for seed in (1201, 1213, 1223)
    }

    for schedule in schedules.values():
        counts = tuple(schedule.count(index) for index in range(22))
        assert min(counts) == 23
        assert max(counts) == 24
    assert schedules[1201] != schedules[1213]
    assert schedules[1213] != schedules[1223]


def test_candidate_control_contract_rejects_mapping_drift(tmp_path) -> None:
    source = tmp_path / "counterfactual_outcomes.jsonl"
    rows = []
    for index in range(22):
        rows.append(
            {
                "candidate_index": index,
                "applied_control": [float(index), 0.0, 0.0],
            }
        )
    rows.append(
        {
            "candidate_index": 1,
            "applied_control": [999.0, 0.0, 0.0],
        }
    )
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="drifted across rows"):
        load_gate2_candidate_control_contract(source)


def test_seed_summary_applies_preregistered_stoploss() -> None:
    inputs = {f"t-{index}": {} for index in range(500)}
    outcomes = {
        f"t-{index}": {
            "selector_minus_permutation": 0.01,
            "selected_realized_delta": 0.03,
            "consumer_session_index": index // 10,
            "selected_action_index": index % 22,
            "selected_equals_permutation": False,
        }
        for index in range(500)
    }
    summary = summarize_gate2_longitudinal_seed(
        seed=1201,
        source_transition_count=500,
        inputs=inputs,
        outcomes=outcomes,
    )

    assert summary["complete"] is True
    assert summary["selector_minus_permutation_mean"] == pytest.approx(0.01)
    assert summary["gates"][
        "selector_minus_permutation_at_least_0_02"
    ] is False
    assert summary["single_seed_stoploss_passed"] is False
