"""Dynamic figure profile loaders.

The legacy ``einstein`` / ``lu_xun`` profiles are reviewer-curated and
ship with the wheel under :mod:`lifeform_domain_figure`. They are
appropriate for figures whose corpora are public-record (philosophers,
scientists, authors with extensive published primary sources).

For figures whose corpora are NOT public record — most notably the
family-memorial product where the corpus is a private collection of
interviews, letters, and recordings — the profile is generated at
bake time from a small JSON descriptor (family-attested name +
lifespan + bio). This module provides the loaders for those dynamic
profiles.
"""

from __future__ import annotations

from lifeform_domain_figure.profiles.einstein import build_einstein_profile
from lifeform_domain_figure.profiles.family import (
    build_family_profile_from_json,
    load_family_profile_file,
)
from lifeform_domain_figure.profiles.lu_xun import build_lu_xun_profile

__all__ = (
    # Reviewer-curated public-record profiles. The package `__init__.py`
    # imports these from here; without the re-export the whole package
    # failed to import and every pytest in it was uncollectable (R5-2).
    "build_einstein_profile",
    "build_lu_xun_profile",
    "build_family_profile_from_json",
    "load_family_profile_file",
)
