"""Reviewed character profiles shipped with the wheel.

Each profile lives in its own module (``zhang_wuji.py`` etc.) so the
contents are inspectable and version-controlled. Adding a new profile
means adding a new module + listing its builder here.

Profiles are reviewed structured artifacts: they encode a character's
drives / cases / boundaries / value seeds in the typed
:class:`CharacterSoulProfile` schema, NOT a free-form blob of novel
text. The novel text itself enters the lifeform through
``build_character_ingestion_envelope`` + the canonical ingestion
pipeline (see ``docs/specs/character-soul-bootstrap.md``).
"""

from __future__ import annotations

from lifeform_domain_character.profiles.zhang_wuji import (
    build_zhang_wuji_profile,
)


__all__ = [
    "build_zhang_wuji_profile",
]
