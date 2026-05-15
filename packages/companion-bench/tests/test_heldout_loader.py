# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Held-out submodule loader tests."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from companion_bench.heldout_loader import (
    HeldOutMissingError,
    is_heldout_present,
    load_heldout_scenarios,
)


_HELDOUT_FIXTURE = textwrap.dedent(
    """\
    scenario_id: F2-repair-001-v01-heldout
    family: F2
    arc_length_sessions: 3
    session_turn_range: [4, 6]
    inter_session_gap_days: [1, 2]
    user_simulator:
      persona: "young professional, anxious attachment (held-out variant: different cultural background)"
      goals:
        - "establish problem in S1"
        - "withdraw in S2"
        - "test repair in S3"
      perturbation_seed: 49
      fsm:
        - session: 1
          turn: 1
          action: establish_pattern
          payload: "different-locale-I told her I felt invisible at the dinner"
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
    public_test: false
    held_out: true
    """
)

_PUBLIC_LIKE_BUT_IN_HELDOUT_DIR = textwrap.dedent(
    """\
    scenario_id: P-bad-001
    family: F2
    arc_length_sessions: 3
    session_turn_range: [4, 6]
    inter_session_gap_days: [1, 2]
    user_simulator:
      persona: "x"
      goals: ["x"]
      perturbation_seed: 1
      fsm: []
    expected_axes:
      primary: [A3]
      hard_constraint: A6
    disqualifiers: []
    public_test: true
    held_out: false
    """
)


def test_missing_dir_returns_empty_with_warning(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "companionbench-heldout" / "scenarios"
    with pytest.warns(UserWarning, match="held-out submodule not present"):
        out = load_heldout_scenarios(heldout_dir=missing, require=False)
    assert out == ()


def test_missing_dir_with_require_raises(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "companionbench-heldout" / "scenarios"
    with pytest.raises(HeldOutMissingError, match="held-out submodule not found"):
        load_heldout_scenarios(heldout_dir=missing, require=True)


def test_present_dir_loads_held_out_scenarios(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "scenarios"
    d.mkdir()
    (d / "f2-v01-heldout.yaml").write_text(_HELDOUT_FIXTURE, encoding="utf-8")
    out = load_heldout_scenarios(heldout_dir=d, require=False)
    assert len(out) == 1
    assert out[0].held_out is True


def test_public_like_in_heldout_dir_raises(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "scenarios"
    d.mkdir()
    (d / "bad.yaml").write_text(_PUBLIC_LIKE_BUT_IN_HELDOUT_DIR, encoding="utf-8")
    with pytest.raises(ValueError, match="not marked held_out=true"):
        load_heldout_scenarios(heldout_dir=d, require=False)


def test_is_heldout_present_detects_yaml_anywhere_under(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "scenarios"
    nested = d / "f2"
    nested.mkdir(parents=True)
    (nested / "x.yaml").write_text(_HELDOUT_FIXTURE, encoding="utf-8")
    assert is_heldout_present(d) is True


def test_is_heldout_present_false_for_empty_dir(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "scenarios"
    d.mkdir()
    assert is_heldout_present(d) is False
