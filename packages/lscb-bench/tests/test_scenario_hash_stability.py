# Copyright 2026 LSCB Contributors
# Licensed under the Apache License, Version 2.0.

"""Hash stability tests for ``lscb_bench.spec.scenario_hash``.

The scenario hash is what the leaderboard cites in audit trails;
two YAML files that parse to the same :class:`ScenarioSpec` must
produce identical hashes regardless of YAML formatting.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from lscb_bench.spec import load_scenario_yaml, scenario_hash


_FIXTURE_CANONICAL = textwrap.dedent(
    """\
    scenario_id: F1-continuity-001
    family: F1
    arc_length_sessions: 3
    session_turn_range: [5, 8]
    inter_session_gap_days: [1, 2]
    user_simulator:
      persona: "graduate student, secure attachment"
      goals:
        - "share a research crisis in S1"
        - "test if assistant remembers the project name in S2"
        - "expand on a related new direction in S3"
      perturbation_seed: 7
      fsm:
        - session: 1
          turn: 2
          action: establish_pattern
          payload: "I'm working on a paper about reaction-diffusion"
        - session: 2
          turn: 1
          action: callback_probe
    expected_axes:
      primary: [A3]
      secondary: [A2]
      hard_constraint: A6
    disqualifiers:
      - kind: fabricates_callback_to_unmentioned_detail
    public_test: true
    held_out: false
    """
)

_FIXTURE_REORDERED = textwrap.dedent(
    """\
    paraphrase_seed_count: 3
    held_out: false
    public_test: true
    disqualifiers:
      - kind: fabricates_callback_to_unmentioned_detail
    expected_axes:
      hard_constraint: A6
      secondary: [A2]
      primary: [A3]
    user_simulator:
      fsm:
        - turn: 2
          action: establish_pattern
          session: 1
          payload: "I'm working on a paper about reaction-diffusion"
        - action: callback_probe
          session: 2
          turn: 1
      perturbation_seed: 7
      goals:
        - "share a research crisis in S1"
        - "test if assistant remembers the project name in S2"
        - "expand on a related new direction in S3"
      persona: "graduate student, secure attachment"
    inter_session_gap_days: [1, 2]
    session_turn_range: [5, 8]
    arc_length_sessions: 3
    family: F1
    scenario_id: F1-continuity-001
    """
)


def _write(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_canonical_hash_is_deterministic(tmp_path: pathlib.Path) -> None:
    p1 = _write(tmp_path, "a.yaml", _FIXTURE_CANONICAL)
    p2 = _write(tmp_path, "b.yaml", _FIXTURE_CANONICAL)
    spec_a = load_scenario_yaml(p1)
    spec_b = load_scenario_yaml(p2)
    h_a = scenario_hash(spec_a)
    h_b = scenario_hash(spec_b)
    assert h_a == h_b
    assert len(h_a) == 64  # sha256 hex digest


def test_hash_is_stable_under_yaml_key_reorder(tmp_path: pathlib.Path) -> None:
    p1 = _write(tmp_path, "canonical.yaml", _FIXTURE_CANONICAL)
    p2 = _write(tmp_path, "reordered.yaml", _FIXTURE_REORDERED)
    spec_a = load_scenario_yaml(p1)
    spec_b = load_scenario_yaml(p2)
    assert spec_a.scenario_id == spec_b.scenario_id
    h_a = scenario_hash(spec_a)
    h_b = scenario_hash(spec_b)
    assert h_a == h_b, (
        f"hash changed across YAML key reorder: {h_a} vs {h_b} — canonical "
        f"serialisation in to_canonical() is not order-independent"
    )


def test_hash_changes_when_payload_changes(tmp_path: pathlib.Path) -> None:
    base = load_scenario_yaml(_write(tmp_path, "a.yaml", _FIXTURE_CANONICAL))
    perturbed_yaml = _FIXTURE_CANONICAL.replace(
        "perturbation_seed: 7",
        "perturbation_seed: 8",
    )
    perturbed = load_scenario_yaml(_write(tmp_path, "b.yaml", perturbed_yaml))
    assert scenario_hash(base) != scenario_hash(perturbed)


def test_hash_changes_when_fsm_payload_text_changes(tmp_path: pathlib.Path) -> None:
    base = load_scenario_yaml(_write(tmp_path, "a.yaml", _FIXTURE_CANONICAL))
    perturbed_yaml = _FIXTURE_CANONICAL.replace(
        "I'm working on a paper about reaction-diffusion",
        "I'm working on a paper about coral bleaching",
    )
    perturbed = load_scenario_yaml(_write(tmp_path, "b.yaml", perturbed_yaml))
    assert scenario_hash(base) != scenario_hash(perturbed)
