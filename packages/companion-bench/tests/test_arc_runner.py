# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""arc_runner end-to-end smoke tests with fake user backend + fake SUT."""

from __future__ import annotations

import pathlib
import textwrap

from companion_bench.arc_runner import (
    ArcRunConfig,
    run_arc,
    write_arc_record,
)
from companion_bench.spec import load_scenario_yaml
from companion_bench.sut_client import EchoFakeSUTClient
from companion_bench.user_simulator import DeterministicFakeUtteranceClient


_FIXTURE = textwrap.dedent(
    """\
    scenario_id: F2-repair-001
    family: F2
    arc_length_sessions: 3
    session_turn_range: [3, 4]
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


def _load(tmp_path: pathlib.Path):
    p = tmp_path / "f2.yaml"
    p.write_text(_FIXTURE, encoding="utf-8")
    return load_scenario_yaml(p)


def test_run_arc_produces_n_sessions_with_turns_in_range(tmp_path: pathlib.Path) -> None:
    spec = _load(tmp_path)
    sut = EchoFakeSUTClient()
    rec = run_arc(
        spec=spec,
        paraphrase_seed=0,
        sut_client=sut,
        user_backend=DeterministicFakeUtteranceClient(),
        config=ArcRunConfig(
            submission_id="test-sub",
            user_simulator_model="fake/user-sim",
        ),
    )
    assert rec.scenario_id == "F2-repair-001"
    assert len(rec.sessions) == 3
    for s in rec.sessions:
        assert spec.session_turn_range[0] <= len(s.turns) <= spec.session_turn_range[1]
        # session_id must follow `arc_id-sN` convention
        assert s.session_id.endswith(f"-s{s.session_index}")


def test_run_arc_inserts_gap_prefix_only_on_first_turn_of_non_s1(tmp_path: pathlib.Path) -> None:
    spec = _load(tmp_path)
    sut = EchoFakeSUTClient()
    rec = run_arc(
        spec=spec,
        paraphrase_seed=0,
        sut_client=sut,
        user_backend=DeterministicFakeUtteranceClient(),
        config=ArcRunConfig(
            submission_id="t",
            user_simulator_model="fake/user-sim",
        ),
    )
    s2_first_turn = rec.sessions[1].turns[0]
    assert "[It has been" in s2_first_turn.user_text
    if len(rec.sessions[1].turns) > 1:
        s2_second_turn = rec.sessions[1].turns[1]
        assert "[It has been" not in s2_second_turn.user_text


def test_run_arc_passes_metadata_session_id_per_session(tmp_path: pathlib.Path) -> None:
    spec = _load(tmp_path)
    sut = EchoFakeSUTClient()
    run_arc(
        spec=spec,
        paraphrase_seed=0,
        sut_client=sut,
        user_backend=DeterministicFakeUtteranceClient(),
        config=ArcRunConfig(
            submission_id="t",
            user_simulator_model="fake/user-sim",
        ),
    )
    seen = {call["session_id"] for call in sut.calls}
    # Three distinct session ids, all sharing the same arc_id prefix.
    assert len(seen) == 3
    arc_prefixes = {sid.rsplit("-s", 1)[0] for sid in seen}
    assert len(arc_prefixes) == 1


def test_run_arc_arc_id_stable_for_same_inputs(tmp_path: pathlib.Path) -> None:
    spec = _load(tmp_path)
    sut1 = EchoFakeSUTClient()
    sut2 = EchoFakeSUTClient()
    cfg = ArcRunConfig(submission_id="stable", user_simulator_model="fake/user-sim")
    rec1 = run_arc(
        spec=spec, paraphrase_seed=0, sut_client=sut1,
        user_backend=DeterministicFakeUtteranceClient(), config=cfg,
    )
    rec2 = run_arc(
        spec=spec, paraphrase_seed=0, sut_client=sut2,
        user_backend=DeterministicFakeUtteranceClient(), config=cfg,
    )
    assert rec1.arc_id == rec2.arc_id


def test_run_arc_arc_id_differs_for_different_seeds(tmp_path: pathlib.Path) -> None:
    spec = _load(tmp_path)
    cfg = ArcRunConfig(submission_id="diff", user_simulator_model="fake/user-sim")
    rec1 = run_arc(
        spec=spec, paraphrase_seed=0,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        config=cfg,
    )
    rec2 = run_arc(
        spec=spec, paraphrase_seed=1,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        config=cfg,
    )
    assert rec1.arc_id != rec2.arc_id


def test_run_arc_records_telemetry_headers_when_present(tmp_path: pathlib.Path) -> None:
    """The fake SUT emits ``x-fake-sut: 1`` which should be ignored
    (does not match the lifeform/lscb/bench prefixes). When the SUT is
    real, we'd see ``x-lifeform-pe-magnitude`` etc. — this test
    asserts the runner filters correctly."""
    spec = _load(tmp_path)
    sut = EchoFakeSUTClient()
    rec = run_arc(
        spec=spec, paraphrase_seed=0,
        sut_client=sut,
        user_backend=DeterministicFakeUtteranceClient(),
        config=ArcRunConfig(submission_id="t", user_simulator_model="fake/user-sim"),
    )
    for s in rec.sessions:
        for t in s.turns:
            assert "x-fake-sut" not in t.sut_telemetry


def test_write_arc_record_emits_json(tmp_path: pathlib.Path) -> None:
    spec = _load(tmp_path)
    rec = run_arc(
        spec=spec, paraphrase_seed=0,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        config=ArcRunConfig(submission_id="t", user_simulator_model="fake/user-sim"),
    )
    out_dir = tmp_path / "out"
    out_path = write_arc_record(rec, out_dir)
    assert out_path.exists()
    import json as _json
    payload = _json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "F2-repair-001"
    assert payload["scenario_hash"]
    assert len(payload["sessions"]) == 3
