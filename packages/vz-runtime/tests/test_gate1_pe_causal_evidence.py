from __future__ import annotations

from dataclasses import replace
import json

from volvence_zero.agent.gate1_pe_causal_evidence import (
    GATE1_PE_CAUSAL_MIN_EFFECT,
    GATE1_PE_CAUSAL_PROFILES,
    GATE1_PE_CAUSAL_REQUIRED_FILES,
    GATE1_PE_CAUSAL_SCHEMA_VERSION,
    Gate1PECausalCaseResult,
    Gate1PECausalSeedResult,
    _write_packet,
    gate1_pe_causal_heldout_scenarios,
)


def _case_result(
    *,
    seed: int,
    profile_label: str,
    scenario_id: str,
    primary_passed: bool,
) -> Gate1PECausalCaseResult:
    return Gate1PECausalCaseResult(
        seed=seed,
        profile_label=profile_label,
        scenario_id=scenario_id,
        split="open_heldout",
        primary_check_passed=primary_passed,
        open_case_passed=primary_passed,
        reasons=(),
        acceptance_checks=(
            ("late-episode-stabilization-or-improvement", primary_passed),
        ),
        delayed_improvement_observed=primary_passed,
        late_episode_stability_score=1.0 if primary_passed else 0.0,
        prediction_chain_turn_count=6,
        pe_triggered_turn_count=3,
        online_learning_turn_count=2,
        temporal_change_count=1,
        turn_count=6,
    )


def _seed_result(seed: int, *, gain: float = 0.25):
    scenario_ids = tuple(
        scenario.scenario_id
        for scenario in gate1_pe_causal_heldout_scenarios()
    )
    full_pass_count = 4
    no_pe_pass_count = full_pass_count - round(gain * 4)
    cases = tuple(
        _case_result(
            seed=seed,
            profile_label=profile,
            scenario_id=scenario_id,
            primary_passed=(
                True
                if profile == "pe-eta"
                else index < no_pe_pass_count
            ),
        )
        for profile in GATE1_PE_CAUSAL_PROFILES
        for index, scenario_id in enumerate(scenario_ids)
    )
    return Gate1PECausalSeedResult(
        seed=seed,
        profile_labels=GATE1_PE_CAUSAL_PROFILES,
        scenario_ids=scenario_ids,
        substrate_fingerprint="fixed-fingerprint",
        runtime_origin="builtin-fallback",
        full_learning_success_rate=1.0,
        no_pe_drive_learning_success_rate=no_pe_pass_count / 4,
        heldout_learning_gain=gain,
        full_open_pass_rate=1.0,
        no_pe_drive_open_pass_rate=no_pe_pass_count / 4,
        passed=gain >= GATE1_PE_CAUSAL_MIN_EFFECT,
        case_results=cases,
    )


def test_gate1_causal_heldout_registry_is_exact() -> None:
    assert tuple(
        scenario.scenario_id
        for scenario in gate1_pe_causal_heldout_scenarios()
    ) == (
        "open_repair_heldout",
        "open_clarification_heldout",
        "open_failure_loop_heldout",
        "open_goal_shift_heldout",
    )
    assert all(
        scenario.split == "open_heldout"
        for scenario in gate1_pe_causal_heldout_scenarios()
    )


def test_gate1_causal_packet_requires_three_seed_minimum_effect(
    tmp_path,
) -> None:
    results = tuple(_seed_result(seed) for seed in (101, 211, 307))

    paths = _write_packet(
        output_dir=tmp_path,
        results=results,
        requested_full_matrix=True,
    )

    assert {path.name for path in paths} == set(
        GATE1_PE_CAUSAL_REQUIRED_FILES
    )
    verdict = json.loads(
        (tmp_path / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    assert verdict["schema_version"] == GATE1_PE_CAUSAL_SCHEMA_VERSION
    assert verdict["status"] == "causal-supported"
    assert verdict["causal_status"] == "causal-supported"
    assert verdict["failed_gates"] == []


def test_gate1_causal_probe_failure_stops_claim(tmp_path) -> None:
    failed_probe = _seed_result(101, gain=0.0)

    _write_packet(
        output_dir=tmp_path,
        results=(failed_probe,),
        requested_full_matrix=False,
    )

    verdict = json.loads(
        (tmp_path / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    assert verdict["status"] == "not-supported"
    assert verdict["causal_status"] == "not-supported"
    assert (
        verdict["claim_if_not_supported"]
        == "PE is an auditable primary signal"
    )


def test_gate1_causal_diagnostic_open_pass_does_not_enter_primary_gate(
    tmp_path,
) -> None:
    result = replace(
        _seed_result(101),
        full_open_pass_rate=0.0,
        no_pe_drive_open_pass_rate=1.0,
    )

    _write_packet(
        output_dir=tmp_path,
        results=(result,),
        requested_full_matrix=False,
    )

    verdict = json.loads(
        (tmp_path / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    assert verdict["status"] == "probe-passed"
