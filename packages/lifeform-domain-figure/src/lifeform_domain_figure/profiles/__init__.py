"""Reviewed historical-figure profiles shipped with the wheel.

Each profile lives in its own module (``einstein.py``, ``lu_xun.py``,
...) so the contents are inspectable and version-controlled. Adding
a new profile means adding a new module + listing its builder here.

Profiles are reviewed structured artifacts: they encode a figure's
documented stances, drives, signature cases, value seeds, and
boundaries in the typed :class:`HistoricalFigureProfile` schema, NOT
verbatim primary-source text. The corpus itself enters the lifeform
through :func:`lifeform_domain_figure.build_figure_ingestion_envelope`
+ the canonical ingestion pipeline (see
``docs/specs/figure-vertical.md``).
"""

from __future__ import annotations

from lifeform_domain_figure.profiles.einstein import build_einstein_profile
from lifeform_domain_figure.profiles.lu_xun import build_lu_xun_profile


__all__ = [
    "build_einstein_profile",
    "build_lu_xun_profile",
]
