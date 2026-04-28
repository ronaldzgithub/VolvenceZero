"""Vertical: relationship-aware companion (the EmoGPT archetype).

Public API:

* ``build_companion_package`` — returns the canonical ``DomainExperiencePackage``
  for this vertical.
"""

from __future__ import annotations

from lifeform_domain_emogpt.companion_pack import build_companion_package

__all__ = ("build_companion_package",)
