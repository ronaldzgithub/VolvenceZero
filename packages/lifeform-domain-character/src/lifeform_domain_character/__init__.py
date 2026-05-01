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
from lifeform_domain_character.profile import (
    CharacterBoundaryPrior,
    CharacterDrivePrior,
    CharacterKnowledgeSeed,
    CharacterSignatureCase,
    CharacterSoulProfile,
    CharacterStrategyPrior,
)

__all__ = [
    "CharacterBoundaryPrior",
    "CharacterDrivePrior",
    "CharacterKnowledgeSeed",
    "CharacterSignatureCase",
    "CharacterSoulProfile",
    "CharacterStrategyPrior",
    "build_character_ingestion_envelope",
    "build_character_package",
    "build_character_vitals_bootstrap",
]
