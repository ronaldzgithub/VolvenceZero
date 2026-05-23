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

from lifeform_domain_figure.profiles.family import (
    build_family_profile_from_json,
    load_family_profile_file,
)

__all__ = (
    "build_family_profile_from_json",
    "load_family_profile_file",
)
