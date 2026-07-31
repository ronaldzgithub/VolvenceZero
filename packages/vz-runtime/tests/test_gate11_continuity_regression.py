from __future__ import annotations

import json

from volvence_zero.agent.gate11_per_user_continuity_evidence import (
    GATE11_CONTINUITY_SCHEMA_VERSION,
    evaluate_gate11_continuity_regression,
)


def test_gate11_regression_requires_all_three_negative_controls(tmp_path):
    gate_names = (
        "all_arms_all_seeds_present",
        "settled_transition_count_510_per_arm_seed",
        "multiple_sessions_per_arm_seed",
        "constructor_restarts_present",
        "persistence_roundtrip_exact",
        "same_current_probe_all_arms",
        "swapped_state_target_hits_zero",
        "cross_user_read_leakage_zero",
        "cross_user_write_leakage_zero",
        "cross_user_key_collision_zero",
        "delete_exact",
        "rollback_exact",
        "correct_vs_stateless_effect",
        "correct_vs_swapped_effect",
        "correct_vs_shuffled_effect",
    )
    payload = {
        "schema_version": GATE11_CONTINUITY_SCHEMA_VERSION,
        "gates": {name: True for name in gate_names},
        "comparisons": {
            control: {
                "mean_gain": gain,
                "confidence_interval_95": [gain / 2, gain * 1.5],
            }
            for control, gain in (
                ("stateless", 0.3),
                ("swapped-user-state", 0.35),
                ("shuffled-history", 0.2),
            )
        },
    }
    (tmp_path / "ablation_results.json").write_text(json.dumps(payload))
    assert evaluate_gate11_continuity_regression(tmp_path).passed is True

    payload["gates"]["correct_vs_shuffled_effect"] = False
    (tmp_path / "ablation_results.json").write_text(json.dumps(payload))
    result = evaluate_gate11_continuity_regression(tmp_path)
    assert result.passed is False
    assert "correct_vs_shuffled_effect" in result.failed_gates
