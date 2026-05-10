"""Reviewed growth-advisor profiles shipped with the wheel.

Each profile lives in its own module (``cheng_laoshi.py`` etc.) so
the contents are inspectable and version-controlled. Adding a new
profile means adding a new module + listing its builder here.

Profiles are reviewed structured artifacts: they encode the advisor's
persona core / user archetypes / playbook / boundaries / drives in
the typed :class:`GrowthAdvisorProfile` schema, NOT a free-form blob
of playbook text. Sample text itself enters the lifeform through
``build_growth_advisor_ingestion_envelope`` + the canonical ingestion
pipeline.
"""

from __future__ import annotations

from lifeform_domain_growth_advisor.profiles.cheng_laoshi import (
    build_cheng_laoshi_profile,
)


__all__ = [
    "build_cheng_laoshi_profile",
]
