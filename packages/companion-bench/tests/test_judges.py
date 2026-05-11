# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Unit tests for per-turn and arc judges (P5)."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from companion_bench.arc_runner import ArcRunConfig, run_arc
from companion_bench.callback_ledger import HeuristicCallbackExtractor, build_callback_ledger
from companion_bench.judge_perturn import (
    CRITERIA,
    DeterministicFakePerTurnJudge,
    LLMPerTurnJudge,
    score_arc_perturn,
)
from companion_bench.judge_arc import (
    AXIS_ORDER,
    DeterministicFakeArcJudge,
    LLMArcJudge,
    score_arc_axes,
)
from companion_bench.spec import AxisId, load_scenario_yaml
from companion_bench.sut_client import EchoFakeSUTClient
from companion_bench.user_simulator import DeterministicFakeUtteranceClient


_FIXTURE = textwrap.dedent(
    """\
    scenario_id: F2-repair-001
    family: F2
    arc_length_sessions: 3
    session_turn_range: [3, 3]
    inter_session_gap_days: [1, 7]
    user_simulator:
      persona: "young professional, anxious attachment"
      goals:
        - "establish problem in S1"
        - "withdraw in S2"
        - "test repair in S3"
      perturbation_seed: 7
      fsm:
        - session: 1
          turn: 1
          action: establish_pattern
          payload: "I told her I felt invisible at the dinner"
        - session: 2
          turn: 1
          action: withdrawal_under_handling
        - session: 3
          turn: 2
          action: repair_window
    expected_axes:
      primary: [A3]
      hard_constraint: A6
    disqualifiers:
      - kind: fabricates_callback_to_unmentioned_detail
    public_test: true
    held_out: false
    """
)


def _make_arc(tmp_path: pathlib.Path):
    p = tmp_path / "f2.yaml"
    p.write_text(_FIXTURE, encoding="utf-8")
    spec = load_scenario_yaml(p)
    arc = run_arc(
        spec=spec,
        paraphrase_seed=0,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        config=ArcRunConfig(submission_id="t", user_simulator_model="fake/user-sim"),
    )
    return spec, arc


# ---------------------------------------------------------------------------
# Per-turn judge
# ---------------------------------------------------------------------------


def test_perturn_fake_judge_returns_all_eight_criteria(tmp_path: pathlib.Path) -> None:
    _, arc = _make_arc(tmp_path)
    rubric = score_arc_perturn(arc=arc, judge=DeterministicFakePerTurnJudge())
    assert rubric.judge_model == "fake/perturn"
    for ts in rubric.turn_scores:
        assert set(ts.scores.keys()) == set(CRITERIA)
        for v in ts.scores.values():
            assert 0 <= v <= 5


def test_perturn_score_count_equals_total_assistant_turns(tmp_path: pathlib.Path) -> None:
    _, arc = _make_arc(tmp_path)
    total_turns = sum(len(s.turns) for s in arc.sessions)
    rubric = score_arc_perturn(arc=arc, judge=DeterministicFakePerTurnJudge())
    assert len(rubric.turn_scores) == total_turns


def test_perturn_judge_repeated_runs_match_fake(tmp_path: pathlib.Path) -> None:
    _, arc = _make_arc(tmp_path)
    a = score_arc_perturn(arc=arc, judge=DeterministicFakePerTurnJudge())
    b = score_arc_perturn(arc=arc, judge=DeterministicFakePerTurnJudge())
    assert [t.scores for t in a.turn_scores] == [t.scores for t in b.turn_scores]


def test_llm_perturn_judge_parses_clean_json() -> None:
    def fake_complete(prompt, *, seed, system):
        return json.dumps({k: 4 for k in CRITERIA})

    import json
    judge = LLMPerTurnJudge(client_complete=fake_complete, model="fake/llm")
    scores = judge.score(
        prior_context=[{"role": "user", "content": "..."}],
        assistant_text="response",
        session_index=1,
        turn_index=1,
    )
    assert scores == {k: 4 for k in CRITERIA}


def test_llm_perturn_judge_clamps_out_of_range() -> None:
    import json
    def fake_complete(prompt, *, seed, system):
        # Out-of-range values should be clamped, missing keys default to 0.
        body = {k: 99 for k in CRITERIA}
        body[CRITERIA[0]] = -10
        body.pop(CRITERIA[-1])  # missing
        return json.dumps(body)

    judge = LLMPerTurnJudge(client_complete=fake_complete, model="fake/llm")
    scores = judge.score(
        prior_context=[{"role": "user", "content": "..."}],
        assistant_text="x",
        session_index=1, turn_index=1,
    )
    assert scores[CRITERIA[0]] == 0  # clamped from -10
    assert scores[CRITERIA[-1]] == 0  # missing → 0
    for k in CRITERIA[1:-1]:
        assert scores[k] == 5  # clamped from 99


def test_llm_perturn_judge_recovers_from_prose_wrapped_json() -> None:
    def fake_complete(prompt, *, seed, system):
        body = {k: 3 for k in CRITERIA}
        import json
        return f"Here are the scores:\n{json.dumps(body)}\nThanks."

    judge = LLMPerTurnJudge(client_complete=fake_complete, model="fake/llm")
    scores = judge.score(
        prior_context=[],
        assistant_text="x",
        session_index=1, turn_index=1,
    )
    assert all(v == 3 for v in scores.values())


def test_llm_perturn_judge_raises_on_unparseable_output() -> None:
    def fake_complete(prompt, *, seed, system):
        return "totally not json"

    judge = LLMPerTurnJudge(client_complete=fake_complete, model="fake/llm")
    with pytest.raises(ValueError, match="parseable JSON"):
        judge.score(
            prior_context=[],
            assistant_text="x",
            session_index=1, turn_index=1,
        )


# ---------------------------------------------------------------------------
# Arc judge
# ---------------------------------------------------------------------------


def test_arc_fake_judge_returns_six_axes_in_range(tmp_path: pathlib.Path) -> None:
    _, arc = _make_arc(tmp_path)
    ledger = build_callback_ledger(arc=arc, extractor=HeuristicCallbackExtractor())
    judge = DeterministicFakeArcJudge()
    result = score_arc_axes(arc=arc, ledger=ledger, family="F2", judge=judge)
    assert set(result.scores.keys()) == set(AXIS_ORDER)
    for axis, score in result.scores.items():
        assert 0.0 <= score <= 100.0


def test_arc_fake_judge_caps_a3_when_fabrication_in_ledger(tmp_path: pathlib.Path) -> None:
    """If the ledger has fabrications, the deterministic fake caps A3 at 30."""
    _, arc = _make_arc(tmp_path)
    # Construct an artificial fabricated ledger.
    from companion_bench.callback_ledger import (
        CallbackClaim,
        CallbackLedger,
        CallbackLedgerEntry,
    )
    fabricated_entry = CallbackLedgerEntry(
        claim=CallbackClaim(
            session_index=2, turn_index=1, claim_text="fake claim", claimed_when="earlier"
        ),
        evidence_session=None,
        evidence_turn=None,
        evidence_text=None,
        matched=False,
        similarity_score=0.0,
        fabricated=True,
    )
    ledger = CallbackLedger(arc_id=arc.arc_id, entries=(fabricated_entry,))
    judge = DeterministicFakeArcJudge()
    result = score_arc_axes(arc=arc, ledger=ledger, family="F2", judge=judge)
    assert result.scores[AxisId.A3_CONTINUITY] <= 30.0


def test_llm_arc_judge_parses_clean_json(tmp_path: pathlib.Path) -> None:
    import json
    _, arc = _make_arc(tmp_path)
    ledger = build_callback_ledger(arc=arc, extractor=HeuristicCallbackExtractor())

    def fake_complete(prompt, *, seed, system):
        return json.dumps({a.value: 75.5 for a in AXIS_ORDER})

    judge = LLMArcJudge(client_complete=fake_complete, model="fake/arc-llm")
    result = score_arc_axes(arc=arc, ledger=ledger, family="F2", judge=judge)
    for axis in AXIS_ORDER:
        assert result.scores[axis] == 75.5


def test_arc_judge_to_json_round_trip(tmp_path: pathlib.Path) -> None:
    _, arc = _make_arc(tmp_path)
    ledger = build_callback_ledger(arc=arc, extractor=HeuristicCallbackExtractor())
    judge = DeterministicFakeArcJudge()
    result = score_arc_axes(arc=arc, ledger=ledger, family="F2", judge=judge)
    payload = result.to_json()
    assert "A1" in payload["scores"]
    assert "A6" in payload["scores"]
    assert payload["judge_model"] == "fake/arc"
