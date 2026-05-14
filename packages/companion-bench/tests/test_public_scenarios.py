# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Public scenario set: structural + smoke tests (P8 acceptance gate)."""

from __future__ import annotations

import importlib.resources as res
import pathlib

import pytest

from companion_bench.arc_runner import ArcRunConfig, run_arc
from companion_bench.callback_ledger import HeuristicCallbackExtractor, build_callback_ledger
from companion_bench.disqualifier import run_disqualifiers
from companion_bench.spec import (
    FamilyId,
    ScenarioSpec,
    load_scenario_yaml,
    load_scenarios_dir,
    scenario_hash,
)
from companion_bench.sut_client import EchoFakeSUTClient
from companion_bench.user_simulator import DeterministicFakeUtteranceClient


def _scenarios_dir() -> pathlib.Path:
    pkg = res.files("companion_bench") / "scenarios" / "public"
    return pathlib.Path(str(pkg))


def _all_specs() -> tuple[ScenarioSpec, ...]:
    return load_scenarios_dir(_scenarios_dir(), include_held_out=True)


def test_public_set_size_matches_i18n_roadmap() -> None:
    """24 en (initial reviewer-curated) + 6 zh demo (debt #55 Batch 1) = 30.

    See ``docs/external/companion-bench-i18n-roadmap.md`` §3.
    """
    specs = _all_specs()
    assert len(specs) == 30, (
        f"public set must have 30 scenarios (24 en + 6 zh demo); "
        f"got {len(specs)}"
    )


def test_each_family_has_exactly_4_en_plus_1_zh() -> None:
    """Per family: 4 English (initial Wave A) + 1 Chinese (Batch 1)."""
    specs = _all_specs()
    en_counts: dict[FamilyId, int] = {f: 0 for f in FamilyId}
    zh_counts: dict[FamilyId, int] = {f: 0 for f in FamilyId}
    for s in specs:
        if s.language == "en":
            en_counts[s.family] += 1
        elif s.language == "zh":
            zh_counts[s.family] += 1
    for family in FamilyId:
        assert en_counts[family] == 4, (
            f"family {family.value} en count = {en_counts[family]}, expected 4"
        )
        assert zh_counts[family] == 1, (
            f"family {family.value} zh count = {zh_counts[family]}, expected 1 "
            f"(Batch 1 floor; Batch 2 will lift to 4)"
        )


def test_no_held_out_in_public_directory() -> None:
    specs = _all_specs()
    for s in specs:
        assert s.held_out is False, f"{s.scenario_id} marked held_out under public/"


def test_scenario_ids_unique() -> None:
    specs = _all_specs()
    ids = [s.scenario_id for s in specs]
    assert len(set(ids)) == len(ids), f"duplicate scenario ids: {ids}"


def test_scenario_hashes_unique() -> None:
    specs = _all_specs()
    hashes = [scenario_hash(s) for s in specs]
    assert len(set(hashes)) == len(hashes), "scenario hashes are not unique"


def test_scenario_id_prefix_matches_family_id() -> None:
    """``F2-repair-001`` must declare ``family: F2``."""
    specs = _all_specs()
    for s in specs:
        prefix = s.scenario_id.split("-", 1)[0]
        assert prefix == s.family.value, (
            f"{s.scenario_id} prefix {prefix!r} != family {s.family.value!r}"
        )


def test_a6_is_hard_constraint_on_every_scenario() -> None:
    """A6 (safety) must be the hard_constraint on every public scenario."""
    specs = _all_specs()
    for s in specs:
        assert (
            s.expected_axes.hard_constraint is not None
            and s.expected_axes.hard_constraint.value == "A6"
        ), f"{s.scenario_id} hard_constraint must be A6"


@pytest.mark.parametrize(
    "scenario_id",
    [s.scenario_id for s in _all_specs()],
)
def test_scenario_runs_end_to_end_with_fakes(scenario_id: str) -> None:
    """Smoke: every scenario must run to completion with fake SUT/backend."""
    specs = {s.scenario_id: s for s in _all_specs()}
    spec = specs[scenario_id]
    arc = run_arc(
        spec=spec,
        paraphrase_seed=0,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        config=ArcRunConfig(submission_id="public-smoke", user_simulator_model="fake"),
    )
    assert len(arc.sessions) == spec.arc_length_sessions
    ledger = build_callback_ledger(arc=arc, extractor=HeuristicCallbackExtractor())
    report = run_disqualifiers(arc=arc, ledger=ledger, declared=spec.disqualifiers)
    # Smoke assertion: report exists; we don't assert any specific
    # disqualifier truth value here because the EchoFakeSUTClient
    # output is not realistic enough for real disqualifier semantics.
    assert isinstance(report.results, tuple)
