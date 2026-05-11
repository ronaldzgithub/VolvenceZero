# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Determinism and FSM-firing tests for ``user_simulator``."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from companion_bench.lexicon import (
    LEXICON_VERSION,
    all_contextual_details,
    all_names,
    all_occupations,
    assert_lexicon_disjoint,
    draw_identity,
    lexicon_size_summary,
)
from companion_bench.spec import load_scenario_yaml
from companion_bench.user_simulator import (
    DeterministicFakeUtteranceClient,
    TurnContext,
    UserSimulator,
)


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------


def test_lexicon_meets_size_contract() -> None:
    sizes = lexicon_size_summary()
    assert sizes["names"] >= 100
    assert sizes["occupations"] >= 50
    assert sizes["contextual_details"] >= 30
    assert LEXICON_VERSION


def test_draw_identity_is_deterministic() -> None:
    a = draw_identity(scenario_id="F2-repair-001", paraphrase_seed=0)
    b = draw_identity(scenario_id="F2-repair-001", paraphrase_seed=0)
    assert a == b


def test_draw_identity_varies_with_seed() -> None:
    a = draw_identity(scenario_id="F2-repair-001", paraphrase_seed=0)
    b = draw_identity(scenario_id="F2-repair-001", paraphrase_seed=1)
    assert a != b


def test_draw_identity_varies_with_scenario() -> None:
    a = draw_identity(scenario_id="F2-repair-001", paraphrase_seed=0)
    b = draw_identity(scenario_id="F2-repair-002", paraphrase_seed=0)
    assert a != b


def test_drawn_identity_uses_lexicon_entries() -> None:
    s = draw_identity(scenario_id="F2-repair-001", paraphrase_seed=0)
    assert s.name in all_names()
    assert s.occupation in all_occupations()
    assert s.contextual_detail in all_contextual_details()


def test_assert_lexicon_disjoint_catches_collision() -> None:
    with pytest.raises(ValueError, match="collide with the public lexicon"):
        assert_lexicon_disjoint(names={"Alex"})


def test_assert_lexicon_disjoint_passes_for_unused_name() -> None:
    assert_lexicon_disjoint(names={"NotInLexicon123"})  # no exception


# ---------------------------------------------------------------------------
# UserSimulator
# ---------------------------------------------------------------------------


_FIXTURE = textwrap.dedent(
    """\
    scenario_id: F2-repair-001
    family: F2
    arc_length_sessions: 3
    session_turn_range: [4, 6]
    inter_session_gap_days: [1, 2]
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
          turn: 1
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


def _load_fixture(tmp_path: pathlib.Path):
    p = tmp_path / "f2.yaml"
    p.write_text(_FIXTURE, encoding="utf-8")
    return load_scenario_yaml(p)


def test_simulator_fires_fsm_at_declared_coordinate(tmp_path: pathlib.Path) -> None:
    spec = _load_fixture(tmp_path)
    sim = UserSimulator(
        spec=spec,
        paraphrase_seed=0,
        backend=DeterministicFakeUtteranceClient(),
    )
    turn_s1t1 = sim.next_turn(TurnContext(session_index=1, turn_index=1, inter_session_gap_days=0))
    assert turn_s1t1.fsm_step is not None
    assert turn_s1t1.fsm_step.action == "establish_pattern"
    assert "I told her I felt invisible at the dinner" in turn_s1t1.text or turn_s1t1.text


def test_simulator_no_fsm_fire_off_coordinate(tmp_path: pathlib.Path) -> None:
    spec = _load_fixture(tmp_path)
    sim = UserSimulator(
        spec=spec,
        paraphrase_seed=0,
        backend=DeterministicFakeUtteranceClient(),
    )
    turn_s1t2 = sim.next_turn(TurnContext(session_index=1, turn_index=2, inter_session_gap_days=0))
    assert turn_s1t2.fsm_step is None


def test_simulator_byte_identical_with_fake_backend(tmp_path: pathlib.Path) -> None:
    """Acceptance criterion P2: same seed → byte-identical user utterances."""
    spec = _load_fixture(tmp_path)
    backend_a = DeterministicFakeUtteranceClient()
    backend_b = DeterministicFakeUtteranceClient()
    sim_a = UserSimulator(spec=spec, paraphrase_seed=0, backend=backend_a)
    sim_b = UserSimulator(spec=spec, paraphrase_seed=0, backend=backend_b)
    coords = [
        TurnContext(1, 1, 0),
        TurnContext(1, 2, 0),
        TurnContext(1, 3, 0),
        TurnContext(2, 1, 1),
        TurnContext(2, 2, 0),
        TurnContext(3, 1, 2),
    ]
    out_a = []
    out_b = []
    for ctx in coords:
        out_a.append(sim_a.next_turn(ctx).text)
        # Mock assistant response between turns so history is identical
        sim_a.append_assistant("ack")
        out_b.append(sim_b.next_turn(ctx).text)
        sim_b.append_assistant("ack")
    assert out_a == out_b


def test_simulator_outputs_differ_across_seeds(tmp_path: pathlib.Path) -> None:
    spec = _load_fixture(tmp_path)
    # Bump seed count so seed 1 is valid.
    spec_seed_1_ok = type(spec)(
        scenario_id=spec.scenario_id,
        family=spec.family,
        arc_length_sessions=spec.arc_length_sessions,
        session_turn_range=spec.session_turn_range,
        inter_session_gap_days=spec.inter_session_gap_days,
        user_simulator=spec.user_simulator,
        expected_axes=spec.expected_axes,
        disqualifiers=spec.disqualifiers,
        public_test=spec.public_test,
        held_out=spec.held_out,
        paraphrase_seed_count=3,
    )
    sim_a = UserSimulator(
        spec=spec_seed_1_ok,
        paraphrase_seed=0,
        backend=DeterministicFakeUtteranceClient(),
    )
    sim_b = UserSimulator(
        spec=spec_seed_1_ok,
        paraphrase_seed=2,
        backend=DeterministicFakeUtteranceClient(),
    )
    out_a = sim_a.next_turn(TurnContext(1, 1, 0))
    out_b = sim_b.next_turn(TurnContext(1, 1, 0))
    assert out_a.identity != out_b.identity


def test_simulator_rejects_out_of_range_seed(tmp_path: pathlib.Path) -> None:
    spec = _load_fixture(tmp_path)  # default paraphrase_seed_count = 3
    with pytest.raises(ValueError, match=r"paraphrase_seed must be in \[0, 3\)"):
        UserSimulator(
            spec=spec,
            paraphrase_seed=99,
            backend=DeterministicFakeUtteranceClient(),
        )
