# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""P7 tests: reproducibility verifier + cost telemetry."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from companion_bench.arc_runner import ArcRunConfig, run_arc
from companion_bench.cost import (
    CostTracker,
    TokenPricing,
    default_pricing,
)
from companion_bench.spec import AxisId, load_scenario_yaml
from companion_bench.sut_client import EchoFakeSUTClient
from companion_bench.user_simulator import DeterministicFakeUtteranceClient
from companion_bench.verifier import (
    compare_axis_scores,
    pick_verification_arc,
)


_FIXTURE = textwrap.dedent(
    """\
    scenario_id: F1-continuity-001
    family: F1
    arc_length_sessions: 2
    session_turn_range: [3, 3]
    inter_session_gap_days: [1]
    user_simulator:
      persona: "graduate student"
      goals:
        - "share a research crisis in S1"
        - "test if assistant remembers the project name in S2"
      perturbation_seed: 7
      fsm:
        - session: 1
          turn: 1
          action: establish_pattern
          payload: "I'm working on reaction-diffusion"
        - session: 2
          turn: 1
          action: callback_probe
    expected_axes:
      primary: [A3]
      hard_constraint: A6
    disqualifiers:
      - kind: fabricates_callback_to_unmentioned_detail
    public_test: true
    held_out: false
    """
)


def _spec(tmp_path: pathlib.Path):
    p = tmp_path / "f1.yaml"
    p.write_text(_FIXTURE, encoding="utf-8")
    return load_scenario_yaml(p)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_pick_verification_arc_is_deterministic() -> None:
    arcs = ["arc-a", "arc-b", "arc-c", "arc-d", "arc-e"]
    a = pick_verification_arc(submission_id="sub-1", public_arc_ids=arcs)
    b = pick_verification_arc(submission_id="sub-1", public_arc_ids=arcs)
    assert a == b
    assert a in arcs


def test_pick_verification_arc_varies_with_submission() -> None:
    arcs = ["arc-a", "arc-b", "arc-c", "arc-d", "arc-e"]
    chosen: set[str] = set()
    # Different submissions land on different arcs (probabilistically;
    # 5 arcs × 20 submissions should have at least 2 distinct picks).
    for i in range(20):
        chosen.add(pick_verification_arc(submission_id=f"sub-{i}", public_arc_ids=arcs))
    assert len(chosen) >= 2


def test_pick_verification_arc_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one public arc id"):
        pick_verification_arc(submission_id="x", public_arc_ids=[])


def test_compare_axis_scores_no_flag_when_within_threshold() -> None:
    original = {a: 75.0 for a in AxisId}
    rerun = {a: 76.0 for a in AxisId}  # 1.0 delta < 5.0 threshold
    res = compare_axis_scores(
        submission_id="x", arc_id="arc-1",
        original=original, rerun=rerun,
    )
    assert res.flagged is False
    assert res.flag_reasons == ()


def test_compare_axis_scores_flags_when_above_threshold() -> None:
    original = {a: 75.0 for a in AxisId}
    rerun = {a: 75.0 for a in AxisId}
    rerun[AxisId.A3_CONTINUITY] = 60.0  # 15-point swing on A3
    res = compare_axis_scores(
        submission_id="x", arc_id="arc-1",
        original=original, rerun=rerun,
    )
    assert res.flagged is True
    assert any("A3" in reason for reason in res.flag_reasons)


def test_compare_axis_scores_to_json_round_trip() -> None:
    original = {a: 75.0 for a in AxisId}
    rerun = {a: 75.0 for a in AxisId}
    res = compare_axis_scores(
        submission_id="x", arc_id="arc-1",
        original=original, rerun=rerun,
    )
    payload = res.to_json()
    assert payload["arc_id"] == "arc-1"
    assert "A1" in payload["original_axis_scores"]


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_tracker_no_calls_emits_zero_breakdown() -> None:
    tracker = CostTracker()
    breakdown = tracker.freeze()
    assert breakdown.total_usd == 0.0
    assert breakdown.sut_calls == 0
    assert breakdown.missing_models == ()


def test_cost_tracker_records_sut_and_perturn_and_arc() -> None:
    pricing = {
        "sysA": TokenPricing("sysA", 1.0, 2.0),
        "judgeP": TokenPricing("judgeP", 3.0, 4.0),
        "judgeA": TokenPricing("judgeA", 5.0, 6.0),
    }
    tracker = CostTracker(pricing=pricing)
    tracker.record_sut(model="sysA", prompt_tokens=1_000_000, completion_tokens=500_000)
    tracker.record_perturn_judge(model="judgeP", prompt_tokens=200_000, completion_tokens=50_000)
    tracker.record_arc_judge(model="judgeA", prompt_tokens=100_000, completion_tokens=20_000)

    breakdown = tracker.freeze()
    # SUT cost: 1.0 (1M*1.0) + 1.0 (0.5M*2.0) = 2.0
    assert breakdown.sut_usd == pytest.approx(2.0)
    # PT cost: 0.6 (0.2M*3.0) + 0.2 (0.05M*4.0) = 0.8
    assert breakdown.perturn_usd == pytest.approx(0.8)
    # Arc cost: 0.5 (0.1M*5.0) + 0.12 (0.02M*6.0) = 0.62
    assert breakdown.arc_usd == pytest.approx(0.62)
    assert breakdown.total_usd == pytest.approx(2.0 + 0.8 + 0.62)


def test_cost_tracker_flags_missing_model() -> None:
    tracker = CostTracker(pricing={})
    tracker.record_sut(model="some-unknown-model", prompt_tokens=10, completion_tokens=10)
    breakdown = tracker.freeze()
    assert breakdown.sut_usd is None
    assert breakdown.total_usd is None
    assert "some-unknown-model" in breakdown.missing_models


def test_cost_tracker_record_arc_record_aggregates_all_turns(tmp_path: pathlib.Path) -> None:
    spec = _spec(tmp_path)
    arc = run_arc(
        spec=spec, paraphrase_seed=0,
        sut_client=EchoFakeSUTClient(model="fake-sut/echo"),
        user_backend=DeterministicFakeUtteranceClient(),
        config=ArcRunConfig(submission_id="t", user_simulator_model="fake/user-sim"),
    )
    tracker = CostTracker()
    tracker.record_arc_record(arc)
    breakdown = tracker.freeze()
    expected_calls = sum(len(s.turns) for s in arc.sessions)
    assert breakdown.sut_calls == expected_calls
    # fake-sut/echo is in default pricing at 0.0; total is 0.
    assert breakdown.total_usd == 0.0


def test_default_pricing_includes_common_judge_models() -> None:
    pricing = default_pricing()
    assert "anthropic/claude-3.7-sonnet" in pricing
    assert "openai/gpt-5" in pricing
