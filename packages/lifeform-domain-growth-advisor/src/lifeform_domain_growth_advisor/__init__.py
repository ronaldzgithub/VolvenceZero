"""Vertical: long-term private-domain growth-advisor companion.

This wheel is the LTV / private-domain operations vertical. It
converts the reviewed growth-advisor playbook into existing Volvence
Zero inputs:

* ``DomainExperiencePackage`` for knowledge / cases / playbook /
  boundaries.
* ``VitalsBootstrap`` for the advisor's drive profile.
* ``IngestionEnvelope`` for replaying livestream-channel introduction
  text + 7-day dialogue samples through the canonical turn pipeline.

It does not add a new kernel owner and does not infer behaviour from
keywords in raw user text. Behaviour differences across the 7-day
playbook reach the kernel via ``applicability_scope`` tags
(``growth_advisor:day1`` ... ``growth_advisor:day7``) and
``regime_tags`` carried by the typed compiled records.

The wheel is parallel to ``lifeform-domain-character`` /
``lifeform-domain-figure`` / ``lifeform-domain-emogpt`` /
``lifeform-domain-coding``. Per the ``PARALLEL_VERTICAL_PAIRS``
invariant in ``tests/contracts/test_import_boundaries.py``, none of
those wheels may be imported from here.
"""

from __future__ import annotations

from lifeform_domain_growth_advisor.compiler import (
    build_growth_advisor_ingestion_envelope,
    build_growth_advisor_package,
    build_growth_advisor_vitals_bootstrap,
)
from lifeform_domain_growth_advisor.lifeform_builder import (
    GrowthAdvisorLifeformBundle,
    build_cheng_laoshi_lifeform,
    build_growth_advisor_lifeform,
)
from lifeform_domain_growth_advisor.profile import (
    GrowthAdvisorBoundaryPrior,
    GrowthAdvisorDrivePrior,
    GrowthAdvisorKnowledgeSeed,
    GrowthAdvisorProfile,
    GrowthAdvisorSignatureCase,
    GrowthAdvisorStrategyPrior,
)
from lifeform_domain_growth_advisor.profiles import build_cheng_laoshi_profile
from lifeform_domain_growth_advisor.sample_excerpts import (
    GROWTH_ADVISOR_SAMPLE_TEXT,
    cheng_laoshi_sample_excerpt,
)


import pathlib


def scenarios_dir() -> pathlib.Path:
    """Return the directory containing this vertical's scripted scenarios.

    The directory ships as package data (see ``pyproject.toml``'s
    ``package-data`` section). Each ``.json`` file in it is loadable
    by ``lifeform_evolution.load_scenario_pack`` /
    ``load_scenario_pack_dir`` / ``load_scenarios``.
    """
    return pathlib.Path(__file__).resolve().parent / "scenarios"


__all__ = [
    # Schema dataclasses
    "GrowthAdvisorBoundaryPrior",
    "GrowthAdvisorDrivePrior",
    "GrowthAdvisorKnowledgeSeed",
    "GrowthAdvisorProfile",
    "GrowthAdvisorSignatureCase",
    "GrowthAdvisorStrategyPrior",
    # Compilation helpers
    "build_growth_advisor_ingestion_envelope",
    "build_growth_advisor_package",
    "build_growth_advisor_vitals_bootstrap",
    # Factory
    "GrowthAdvisorLifeformBundle",
    "build_cheng_laoshi_lifeform",
    "build_growth_advisor_lifeform",
    # Profiles
    "build_cheng_laoshi_profile",
    # Sample text
    "GROWTH_ADVISOR_SAMPLE_TEXT",
    "cheng_laoshi_sample_excerpt",
    # Package-data accessors
    "scenarios_dir",
]
