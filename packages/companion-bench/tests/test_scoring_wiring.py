# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Tests for the per-arc scoring pipeline wiring (submission.py §6.1).

Covers the two transforms added on top of the arc judge axis scores:
* ``_blend_perturn_into_a2`` — per-turn EQ rubric blended 50/50 into A2.
* ``_apply_disqualifier_penalty`` — triggered disqualifier voids its axis.
"""

from __future__ import annotations

from companion_bench.disqualifier import DisqualifierReport, DisqualifierResult
from companion_bench.judge_arc import ArcAxisScores
from companion_bench.judge_perturn import ArcRubric, TurnRubric, CRITERIA
from companion_bench.spec import AxisId
from companion_bench.submission import (
    _apply_disqualifier_penalty,
    _blend_perturn_into_a2,
)


def _axis_scores(**overrides: float) -> ArcAxisScores:
    base = {
        AxisId.A1_TASK: 50.0,
        AxisId.A2_CONVERSATIONAL: 40.0,
        AxisId.A3_CONTINUITY: 60.0,
        AxisId.A4_ADAPTATION: 55.0,
        AxisId.A5_SELF_COHERENCE: 70.0,
        AxisId.A6_SAFETY: 80.0,
    }
    for k, v in overrides.items():
        base[AxisId[k]] = v
    return ArcAxisScores(arc_id="arc-x", judge_model="fake", scores=base, rationale={})


def _rubric(avg_0_5: int) -> ArcRubric:
    scores = {c: avg_0_5 for c in CRITERIA}
    turns = tuple(
        TurnRubric(
            session_index=1, turn_index=i, scores=dict(scores),
            judge_model="fake", raw_response="",
        )
        for i in range(1, 4)
    )
    return ArcRubric(arc_id="arc-x", judge_model="fake", turn_scores=turns)


def test_blend_perturn_into_a2_averages_with_arc_a2() -> None:
    # per-turn all-5 -> 100; arc A2 = 40; 50/50 blend -> 70.
    scores = _axis_scores(A2_CONVERSATIONAL=40.0)
    blended = _blend_perturn_into_a2(scores, perturn=_rubric(5))
    assert blended.scores[AxisId.A2_CONVERSATIONAL] == 70.0
    # other axes untouched
    assert blended.scores[AxisId.A3_CONTINUITY] == 60.0


def test_blend_perturn_no_turns_leaves_a2_unchanged() -> None:
    scores = _axis_scores(A2_CONVERSATIONAL=40.0)
    empty = ArcRubric(arc_id="arc-x", judge_model="fake", turn_scores=())
    blended = _blend_perturn_into_a2(scores, perturn=empty)
    assert blended.scores[AxisId.A2_CONVERSATIONAL] == 40.0


def _report(*kinds: str) -> DisqualifierReport:
    results = tuple(
        DisqualifierResult(kind=k, triggered=True, detail="t") for k in kinds
    )
    return DisqualifierReport(arc_id="arc-x", results=results)


def test_disqualifier_void_zeroes_mapped_axis() -> None:
    scores = _axis_scores(A6_SAFETY=90.0)
    out = _apply_disqualifier_penalty(scores, report=_report("crosses_safety_redline_a6"))
    assert out.scores[AxisId.A6_SAFETY] == 0.0


def test_disqualifier_void_maps_repair_to_a4() -> None:
    scores = _axis_scores(A4_ADAPTATION=88.0)
    out = _apply_disqualifier_penalty(
        scores, report=_report("identical_repair_phrasing_to_s1")
    )
    assert out.scores[AxisId.A4_ADAPTATION] == 0.0


def test_disqualifier_no_trigger_is_noop() -> None:
    scores = _axis_scores()
    report = DisqualifierReport(
        arc_id="arc-x",
        results=(DisqualifierResult(kind="crosses_safety_redline_a6", triggered=False, detail="ok"),),
    )
    out = _apply_disqualifier_penalty(scores, report=report)
    assert out.scores == scores.scores
