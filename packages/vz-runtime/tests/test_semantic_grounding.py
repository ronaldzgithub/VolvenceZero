"""Acceptance tests for the latent-semantic grounding readout
(experiment 1 of docs/specs/semantic-grounding-evidence.md).

Covers: D1/D2/D3 verdict behaviour on constructed fixtures (grounded /
ungrounded / insufficient coverage), shuffled-control discipline, turn
capture from a real runner's active snapshots, and JSON round-trip
fail-loudly semantics.
"""

from __future__ import annotations

import random

import pytest

from volvence_zero.agent.semantic_grounding import (
    SEMANTIC_DELTA_AXES,
    SEMANTIC_GROUNDING_REPORT_SCHEMA_VERSION,
    GroundingThresholds,
    SemanticGroundingError,
    SemanticGroundingTurnCapture,
    SemanticGroundingTurnEvidence,
    build_semantic_grounding_report,
    turn_evidence_from_payload,
    turn_evidence_to_payload,
)
from volvence_zero.agent.session import AgentSessionRunner

_N_AXES = len(SEMANTIC_DELTA_AXES)

#: Axis indices used by the fixtures (disjoint per family so the
#: grounded fixture has clean semantic signatures).
_FAMILY_A_AXES = {0: 1.0, 1: 0.6}  # relationship trust / repair movement
_FAMILY_B_AXES = {5: 1.0, 7: 0.8}  # commitment count / completion movement

_UNIT_THRESHOLDS = GroundingThresholds(
    min_closed_segments=4,
    min_reused_families=2,
    min_nonzero_delta_ratio=0.2,
    shuffle_count=120,
    shuffle_seed=13,
)


def _delta(axes: dict[int, float], scale: float = 1.0) -> tuple[float, ...]:
    values = [0.0] * _N_AXES
    for index, magnitude in axes.items():
        values[index] = magnitude * scale
    return tuple(values)


def _evidence(
    *,
    turn_index: int,
    case_id: str,
    family: str,
    delta: tuple[float, ...],
    switching: bool,
    segment_id: str | None = None,
) -> SemanticGroundingTurnEvidence:
    return SemanticGroundingTurnEvidence(
        turn_index=turn_index,
        case_id=case_id,
        active_abstract_action=family,
        action_family=family,
        switch_gate=0.7 if switching else 0.2,
        is_switching=switching,
        newly_closed_segment_ids=(segment_id,) if segment_id else (),
        pe_magnitude=0.1,
        pe_segment_id="",
        semantic_delta=delta,
        covered_slots=("relationship_state", "commitment"),
    )


def _grounded_case(
    case_id: str,
    *,
    start_turn: int,
    rng: random.Random,
    scale: float,
) -> list[SemanticGroundingTurnEvidence]:
    """One case: irregular switch turns where a family activates AND its
    semantic signature moves in the same turn; quiet turns in between."""

    turns: list[SemanticGroundingTurnEvidence] = []
    turn = start_turn
    segment_counter = 0
    for family, axes in (("famA", _FAMILY_A_AXES), ("famB", _FAMILY_B_AXES)):
        for _ in range(8):
            segment_counter += 1
            turns.append(
                _evidence(
                    turn_index=turn,
                    case_id=case_id,
                    family=family,
                    delta=_delta(axes, scale=scale * rng.uniform(0.8, 1.2)),
                    switching=True,
                    segment_id=f"{case_id}-{family}-seg-{segment_counter}",
                )
            )
            turn += 1
            # Irregular quiet gap (1-3 turns) so the switch/delta series is
            # aperiodic and circular-shift controls genuinely misalign.
            for _ in range(rng.randint(1, 3)):
                turns.append(
                    _evidence(
                        turn_index=turn,
                        case_id=case_id,
                        family=family,
                        delta=_delta({}),
                        switching=False,
                    )
                )
                turn += 1
    return turns


def _grounded_fixture() -> list[SemanticGroundingTurnEvidence]:
    rng = random.Random(3)
    turns = _grounded_case("case-1", start_turn=1, rng=rng, scale=1.0)
    turns += _grounded_case(
        "case-2", start_turn=len(turns) + 1, rng=rng, scale=1.1
    )
    return turns


def test_grounded_fixture_retains() -> None:
    report = build_semantic_grounding_report(
        _grounded_fixture(), thresholds=_UNIT_THRESHOLDS
    )
    assert report.schema_version == SEMANTIC_GROUNDING_REPORT_SCHEMA_VERSION
    assert report.non_gating is True
    assert report.coverage.meets_thresholds, report.coverage.description
    assert report.d1_discrimination.passed, report.d1_discrimination.description
    assert report.d2_lead.passed, report.d2_lead.description
    assert report.d2_lead.peak_lag >= 0
    assert report.d3_transfer.passed, report.d3_transfer.description
    assert report.verdict == "retain"
    families = {sig.family for sig in report.family_signatures}
    assert {"famA", "famB"} <= families


def test_family_agnostic_deltas_fail_d1() -> None:
    """Both families produce the SAME semantic movement -> the family label
    carries no semantic information -> D1 must not beat its shuffled
    control and the verdict is 'fail' (kill signal)."""

    rng = random.Random(5)
    turns: list[SemanticGroundingTurnEvidence] = []
    turn = 1
    for case_id in ("case-1", "case-2"):
        seg = 0
        for family in ("famA", "famB"):
            for _ in range(8):
                seg += 1
                turns.append(
                    _evidence(
                        turn_index=turn,
                        case_id=case_id,
                        family=family,
                        # Identical axes regardless of family.
                        delta=_delta(_FAMILY_A_AXES, scale=rng.uniform(0.8, 1.2)),
                        switching=True,
                        segment_id=f"{case_id}-{family}-{seg}",
                    )
                )
                turn += 1
                for _ in range(rng.randint(1, 3)):
                    turns.append(
                        _evidence(
                            turn_index=turn,
                            case_id=case_id,
                            family=family,
                            delta=_delta({}),
                            switching=False,
                        )
                    )
                    turn += 1

    report = build_semantic_grounding_report(turns, thresholds=_UNIT_THRESHOLDS)
    assert not report.d1_discrimination.passed
    assert report.verdict == "fail"


def test_default_thresholds_mark_small_runs_insufficient() -> None:
    report = build_semantic_grounding_report(_grounded_fixture())
    # 32 closed segments < spec default 50 -> coverage not met.
    assert not report.coverage.meets_thresholds
    assert report.verdict == "insufficient-coverage"


def test_lagging_switches_fail_d2_lead() -> None:
    """Semantic delta happens BEFORE the switch (latent layer echoes the
    semantic layer) -> peak lag < 0 -> D2 lead must fail."""

    rng = random.Random(9)
    turns: list[SemanticGroundingTurnEvidence] = []
    turn = 1
    for case_id in ("case-1", "case-2"):
        seg = 0
        for family, axes in (("famA", _FAMILY_A_AXES), ("famB", _FAMILY_B_AXES)):
            for _ in range(8):
                seg += 1
                # Semantic movement first (no switch)...
                turns.append(
                    _evidence(
                        turn_index=turn,
                        case_id=case_id,
                        family=family,
                        delta=_delta(axes, scale=rng.uniform(0.8, 1.2)),
                        switching=False,
                    )
                )
                turn += 1
                # ...then the switch fires one turn later, with no delta.
                turns.append(
                    _evidence(
                        turn_index=turn,
                        case_id=case_id,
                        family=family,
                        delta=_delta({}),
                        switching=True,
                        segment_id=f"{case_id}-{family}-{seg}",
                    )
                )
                turn += 1
                for _ in range(rng.randint(1, 2)):
                    turns.append(
                        _evidence(
                            turn_index=turn,
                            case_id=case_id,
                            family=family,
                            delta=_delta({}),
                            switching=False,
                        )
                    )
                    turn += 1

    report = build_semantic_grounding_report(turns, thresholds=_UNIT_THRESHOLDS)
    assert report.d2_lead.peak_lag < 0
    assert not report.d2_lead.passed
    assert report.verdict in {"weak", "fail"}


def test_empty_turns_fail_loudly() -> None:
    with pytest.raises(SemanticGroundingError):
        build_semantic_grounding_report(())


def test_turn_evidence_json_round_trip() -> None:
    turns = tuple(_grounded_fixture()[:6])
    payload = turn_evidence_to_payload(turns)
    restored = turn_evidence_from_payload(payload)
    assert restored == turns


def test_turn_evidence_axis_mismatch_fails_loudly() -> None:
    payload = turn_evidence_to_payload(tuple(_grounded_fixture()[:2]))
    payload["axes"] = list(payload["axes"][:-1])
    with pytest.raises(SemanticGroundingError):
        turn_evidence_from_payload(payload)
    payload_bad_schema = turn_evidence_to_payload(tuple(_grounded_fixture()[:2]))
    payload_bad_schema["schema_version"] = "other.v9"
    with pytest.raises(SemanticGroundingError):
        turn_evidence_from_payload(payload_bad_schema)


async def test_capture_extracts_from_real_runner_snapshots() -> None:
    runner = AgentSessionRunner(rare_heavy_enabled=False)
    capture = SemanticGroundingTurnCapture()
    scripted = (
        "Please lock the fuel order today - it matters to me.",
        "Did the fuel order actually go out as we agreed?",
        "That last answer missed my point; I'm a bit frustrated.",
    )
    for index, text in enumerate(scripted, start=1):
        result = await runner.run_turn(text)
        evidence = capture.observe_turn(
            turn_index=index,
            active_snapshots=result.active_snapshots,
            case_id="capture-smoke",
        )
        assert len(evidence.semantic_delta) == _N_AXES
        assert evidence.action_family
        assert "relationship_state" in evidence.covered_slots
        assert "commitment" in evidence.covered_slots

    turns = capture.turns
    assert len(turns) == len(scripted)
    # First turn has no previous features -> zero delta by construction.
    assert not turns[0].has_semantic_delta
    # Round-trip the captured evidence through the JSON artifact shape.
    restored = turn_evidence_from_payload(turn_evidence_to_payload(turns))
    assert restored == turns


def test_capture_requires_temporal_snapshot() -> None:
    capture = SemanticGroundingTurnCapture()
    with pytest.raises(SemanticGroundingError):
        capture.observe_turn(turn_index=1, active_snapshots={})
