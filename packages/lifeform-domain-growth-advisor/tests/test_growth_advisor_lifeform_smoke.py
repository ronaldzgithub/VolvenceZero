"""Smoke test for the growth-advisor ``Lifeform`` factory.

Exercises ``build_growth_advisor_lifeform`` end-to-end on the
default Cheng Laoshi profile and asserts:

* The factory returns a usable ``Lifeform`` whose first session is
  constructible.
* The optional ingestion envelope path produces a valid envelope
  when ``sample_excerpt`` is supplied.
* The ``scenarios_dir()`` accessor returns a directory that the
  ``lifeform-evolution`` scenario loader can read end-to-end (every
  shipped JSON file parses into a ``ScriptedScenario``).
* All seven 7-day scenario IDs are present.
"""

from __future__ import annotations

from lifeform_evolution.scenario_pack import load_scenario_pack_dir

from lifeform_core import Lifeform
from lifeform_domain_growth_advisor import (
    GrowthAdvisorLifeformBundle,
    build_cheng_laoshi_lifeform,
    build_growth_advisor_lifeform,
    cheng_laoshi_sample_excerpt,
    scenarios_dir,
)


_EXPECTED_SCENARIO_IDS = (
    "growth-advisor-s01-icebreaker",
    "growth-advisor-s02-baseline",
    "growth-advisor-s03-empathy-kb",
    "growth-advisor-s04-pain-mining",
    "growth-advisor-s05-rapport",
    "growth-advisor-s06-targeted-advice",
    "growth-advisor-s07-summary-hook",
)


def test_build_growth_advisor_lifeform_returns_bundle() -> None:
    bundle = build_growth_advisor_lifeform()
    assert isinstance(bundle, GrowthAdvisorLifeformBundle)
    assert isinstance(bundle.lifeform, Lifeform)
    assert bundle.profile.profile_id == "cheng-laoshi"
    assert bundle.ingestion_envelope is None, (
        "no sample_excerpt => no envelope"
    )


def test_build_cheng_laoshi_lifeform_is_an_alias_factory() -> None:
    bundle = build_cheng_laoshi_lifeform()
    assert isinstance(bundle, GrowthAdvisorLifeformBundle)
    assert bundle.profile.profile_id == "cheng-laoshi"
    assert isinstance(bundle.lifeform, Lifeform)


def test_lifeform_session_is_constructible() -> None:
    bundle = build_growth_advisor_lifeform()
    session = bundle.lifeform.create_session(session_id="probe-growth-advisor")
    assert session is not None


def test_sample_excerpt_path_produces_envelope() -> None:
    text = cheng_laoshi_sample_excerpt()
    assert text.strip(), "sample excerpt must not be empty"
    bundle = build_growth_advisor_lifeform(sample_excerpt=text)
    assert bundle.ingestion_envelope is not None
    envelope = bundle.ingestion_envelope
    assert envelope.envelope_id == (
        "growth-advisor-ingestion:cheng-laoshi"
    )
    assert envelope.chunks, (
        "ingestion envelope must contain at least one chunk for the "
        "smoke / evidence runs"
    )


def test_scenarios_dir_loads_seven_day_shapes() -> None:
    """Every shipped scenario JSON loads via the canonical loader.

    Validates the wheel's package-data config (``pyproject.toml``)
    is correct and that all seven day-shape files are well-formed
    ``ScriptedScenario`` payloads.
    """
    sdir = scenarios_dir()
    assert sdir.is_dir(), f"scenarios directory must exist at {sdir}"
    scenarios = load_scenario_pack_dir(sdir)
    actual_ids = {scenario.scenario_id for scenario in scenarios}
    for expected in _EXPECTED_SCENARIO_IDS:
        assert expected in actual_ids, (
            f"scenarios dir is missing {expected!r}; got {sorted(actual_ids)!r}"
        )
    assert len(scenarios) >= 7
