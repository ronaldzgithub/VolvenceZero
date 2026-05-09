"""Vertical: reviewed fictional-character bootstrap.

This package is the monorepo-local application layer for "novel character to
lifeform" work. It converts reviewed character profiles into existing Volvence
Zero inputs:

* ``DomainExperiencePackage`` for knowledge, cases, playbook, and boundaries.
* ``VitalsBootstrap`` for the character's drive profile.
* ``IngestionEnvelope`` for replaying source text through the canonical turn
  pipeline.

It does not add a new kernel owner and does not infer behavior from keywords in
novel text.
"""

from __future__ import annotations

from lifeform_domain_character.compiler import (
    build_character_ingestion_envelope,
    build_character_package,
    build_character_vitals_bootstrap,
)
from lifeform_domain_character.lifeform_builder import (
    CharacterLifeformBundle,
    build_character_lifeform,
    build_zhang_wuji_lifeform,
)
from lifeform_domain_character.profile import (
    CharacterBoundaryPrior,
    CharacterDrivePrior,
    CharacterKnowledgeSeed,
    CharacterSignatureCase,
    CharacterSoulProfile,
    CharacterStrategyPrior,
)
from lifeform_domain_character.profiles import build_zhang_wuji_profile
from lifeform_domain_character.sample_excerpts import zhang_wuji_long_arc_excerpt

__all__ = [
    "CharacterBoundaryPrior",
    "CharacterDrivePrior",
    "CharacterKnowledgeSeed",
    "CharacterLifeformBundle",
    "CharacterSignatureCase",
    "CharacterSoulProfile",
    "CharacterStrategyPrior",
    "build_character_ingestion_envelope",
    "build_character_lifeform",
    "build_character_package",
    "build_character_vitals_bootstrap",
    "build_zhang_wuji_lifeform",
    "build_zhang_wuji_profile",
    "zhang_wuji_long_arc_excerpt",
]
