"""Vertical: pair-programmer engineering partner.

Public API mirrors ``lifeform-domain-emogpt`` so the service-layer
vertical registry can treat the two interchangeably:

* ``build_coding_package`` \u2014 returns the canonical
  ``DomainExperiencePackage`` for the pair-programmer archetype.
* ``build_coding_vitals_bootstrap`` \u2014 returns the vertical's drive set
  (``solution_clarity`` / ``code_freshness`` / ``direction_certainty``).
* ``scenarios_dir`` \u2014 absolute path to the JSON scenario pack shipped
  with this vertical, suitable for passing to
  ``lifeform-bench --scenarios PATH``.
* ``build_coding_lifeform`` \u2014 convenience factory returning a ready-to-run
  ``Lifeform`` with the domain pack and vitals wired in.

This vertical does NOT yet ship pre-trained ``temporal_bootstrap`` /
``regime_bootstrap`` artefacts \u2014 those would come from running
``lifeform-super-loop`` over the vertical's own scenario set. Until that
exists, ``build_coding_lifeform`` returns a calibration-free lifeform
that still works (the kernel's flat regime priors are the fallback).

The contrast with ``lifeform-domain-emogpt`` is the proof of trigger \u2461 in
``SPLIT.md``: the kernel ships zero awareness of which vertical is loaded,
and a second domain wheel needed nothing inside the kernel to land. The
two verticals share the same kernel surfaces but encode different
"what does this lifeform care about" signatures via their own data
(scenarios + drives + experience packages).
"""

from __future__ import annotations

import pathlib
from typing import Any

from lifeform_domain_coding.coding_pack import build_coding_package
from lifeform_domain_coding.coding_vitals import build_coding_vitals_bootstrap


def scenarios_dir() -> pathlib.Path:
    """Return the directory containing this vertical's scripted scenarios.

    Each ``.json`` file in it is loadable by
    ``lifeform_evolution.load_scenarios`` (single file) or
    ``lifeform_evolution.load_scenario_pack_dir`` (whole directory).
    """
    return pathlib.Path(__file__).resolve().parent / "scenarios"


def build_coding_lifeform(
    *,
    config: object | None = None,
    use_vitals_bootstrap: bool = True,
    substrate_runtime: Any = None,
) -> Any:
    """Build a Lifeform with the pair-programmer vertical fully wired in.

    Steps:

    1. Construct a ``LifeformConfig`` (default if not provided) with
       rare-heavy disabled for deterministic behaviour. If
       ``substrate_runtime`` is supplied the brain config is forced into
       ``substrate_mode="injected"`` so the brain consumes the supplied
       runtime rather than building a fresh one per session.
    2. Apply this vertical's ``DomainExperiencePackage`` so the kernel
       ships with engineering-pair knowledge / cases / playbook /
       boundary priors.
    3. Optionally wire the vertical's drive set via
       ``LifeformConfig.vitals_bootstrap`` so idle-tick decay produces
       slow-scale PE in the engineering-relevant drives.

    Returns a ``lifeform_core.Lifeform`` instance ready for
    ``create_session``.

    Args:
        config: optional ``LifeformConfig`` override.
        use_vitals_bootstrap: ship the vertical's drive set. Set False
            for ablation runs.
        substrate_runtime: optional pre-built ``OpenWeightResidualRuntime``.
            When supplied, all sessions share this runtime instance \u2014
            the path used by ``lifeform-service`` on a single GPU.
            See ``lifeform_domain_emogpt.build_companion_lifeform`` for
            the frozen-substrate invariant.
    """
    from dataclasses import replace as _replace
    from lifeform_core import Lifeform, LifeformConfig

    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    brain_overrides: dict[str, Any] = {"rare_heavy_enabled": False}
    if substrate_runtime is not None:
        brain_overrides["substrate_mode"] = "injected"
    base_config = _replace(
        base_config,
        brain_config=_replace(base_config.brain_config, **brain_overrides),
    )
    base_config = base_config.with_domain_experience(
        (build_coding_package(),)
    )

    if use_vitals_bootstrap:
        base_config = base_config.with_vitals(build_coding_vitals_bootstrap())

    return Lifeform(
        base_config,
        substrate_runtime=substrate_runtime,
    )


__all__ = (
    "build_coding_lifeform",
    "build_coding_package",
    "build_coding_vitals_bootstrap",
    "scenarios_dir",
)
