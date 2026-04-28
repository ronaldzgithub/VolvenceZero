"""Vertical: relationship-aware companion (the EmoGPT archetype).

Public API:

* ``build_companion_package`` \u2014 returns the canonical
  ``DomainExperiencePackage`` for this vertical.
* ``scenarios_dir`` \u2014 absolute path to the JSON scenario pack shipped
  with this vertical, suitable for passing to
  ``lifeform-bench --scenarios PATH``.
* ``bootstraps_dir`` \u2014 absolute path to the pre-trained bootstrap
  artifacts (temporal snapshot + regime bootstrap) for this vertical.
* ``load_companion_temporal_bootstrap`` /
  ``load_companion_regime_bootstrap`` \u2014 typed loaders for those artifacts.
* ``build_companion_lifeform`` \u2014 convenience factory returning a
  ready-to-run ``Lifeform`` with the domain pack and pre-trained
  bootstraps wired in.

Why pre-trained bootstraps live with the vertical (not the kernel):

The kernel ships flat (uniform) defaults so it stays vertical-agnostic.
Each vertical encodes its own scripted-supervision priors by running the
super loop over its own scenarios and committing the resulting artifacts
alongside its ``DomainExperiencePackage``. A new vertical adds new
scenarios + new bootstraps + (optionally) a new package; the kernel is
not touched.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Any

from volvence_zero.regime import RegimeBootstrap
from volvence_zero.temporal import MetacontrollerParameterSnapshot

from lifeform_domain_emogpt.companion_pack import build_companion_package


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def scenarios_dir() -> pathlib.Path:
    """Return the directory containing this vertical's scripted scenarios.

    The directory is shipped as package data (see ``pyproject.toml``'s
    ``package-data`` section). Each ``.json`` file in it is loadable by
    ``lifeform_evolution.load_scenario_pack`` / ``-_dir`` /
    ``load_scenarios``.
    """
    return pathlib.Path(__file__).resolve().parent / "scenarios"


def bootstraps_dir() -> pathlib.Path:
    """Return the directory containing the vertical's pre-trained bootstraps.

    Files inside the directory:

    * ``companion-temporal.snap`` \u2014 ``MetacontrollerParameterSnapshot``
      produced by ``lifeform-super-loop`` on this vertical's scenarios.
    * ``companion-regime.bs`` \u2014 ``RegimeBootstrap`` produced by the same
      super-loop run.

    Both are pickle envelopes with magic-byte headers, identical in shape
    to ``lifeform_evolution.save_snapshot`` and ``save_regime_bootstrap``.
    The loaders below validate the headers so accidentally swapping a
    different pickle file fails loudly rather than silently.
    """
    return pathlib.Path(__file__).resolve().parent / "bootstraps"


# ---------------------------------------------------------------------------
# Pickle envelope readers (kept inline so this wheel does NOT need a
# dependency on lifeform-evolution; that would be a layering inversion).
# ---------------------------------------------------------------------------


_TEMPORAL_MAGIC = b"VZ-METASNAP\x00"
_TEMPORAL_SCHEMA_VERSION = "vz-metasnap.v1"
_REGIME_MAGIC = b"VZ-REGIMEBS\x00"
_REGIME_SCHEMA_VERSION = "vz-regimebs.v1"


def _read_envelope(
    *, path: pathlib.Path, magic: bytes, schema_version: str
) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Vertical artifact not found: {path}")
    with path.open("rb") as handle:
        header = handle.read(len(magic))
        if header != magic:
            raise ValueError(
                f"File at {path} is not a Volvence Zero artifact "
                f"(missing magic header)."
            )
        artifact = pickle.load(handle)
    artifact_version = getattr(artifact, "schema_version", None)
    if artifact_version != schema_version:
        raise ValueError(
            f"Artifact at {path} has schema_version={artifact_version!r}; "
            f"this build only loads {schema_version!r}."
        )
    return artifact


def load_companion_temporal_bootstrap() -> MetacontrollerParameterSnapshot:
    """Load the vertical's pre-trained metacontroller snapshot.

    Raises ``FileNotFoundError`` if the artifact is missing (the package
    was installed without its bootstrap data, or the path was hand-deleted).
    """
    artifact = _read_envelope(
        path=bootstraps_dir() / "companion-temporal.snap",
        magic=_TEMPORAL_MAGIC,
        schema_version=_TEMPORAL_SCHEMA_VERSION,
    )
    return artifact.snapshot


def load_companion_regime_bootstrap() -> RegimeBootstrap:
    """Load the vertical's pre-trained regime selection-weights bootstrap."""
    artifact = _read_envelope(
        path=bootstraps_dir() / "companion-regime.bs",
        magic=_REGIME_MAGIC,
        schema_version=_REGIME_SCHEMA_VERSION,
    )
    return artifact.bootstrap


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_companion_lifeform(
    *,
    config: object | None = None,
    use_temporal_bootstrap: bool = True,
    use_regime_bootstrap: bool = True,
) -> Any:
    """Build a Lifeform with the companion vertical fully wired in.

    Steps:

    1. Construct a ``LifeformConfig`` (default if not provided) with
       rare-heavy disabled for deterministic behaviour.
    2. Apply this vertical's ``DomainExperiencePackage`` so the kernel
       ships with companion knowledge / cases / playbook / boundary
       priors.
    3. Optionally inject the vertical's pre-trained
       ``temporal_bootstrap`` and ``regime_bootstrap`` so a fresh session
       starts from the supervised-calibration basin instead of the flat
       defaults.

    Returns a ``lifeform_core.Lifeform`` instance ready for
    ``create_session``.

    The flags exist so a product can opt into bootstrap-free behaviour
    (for ablation runs / clean baselines / when scenarios drift from
    what the bootstraps were trained on).
    """
    from dataclasses import replace as _replace
    from lifeform_core import Lifeform, LifeformConfig

    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    base_config = _replace(
        base_config,
        brain_config=_replace(
            base_config.brain_config, rare_heavy_enabled=False
        ),
    )
    base_config = base_config.with_domain_experience(
        (build_companion_package(),)
    )

    temporal_bootstrap = (
        load_companion_temporal_bootstrap() if use_temporal_bootstrap else None
    )
    regime_bootstrap = (
        load_companion_regime_bootstrap() if use_regime_bootstrap else None
    )

    return Lifeform(
        base_config,
        temporal_bootstrap=temporal_bootstrap,
        regime_bootstrap=regime_bootstrap,
    )


__all__ = (
    "bootstraps_dir",
    "build_companion_lifeform",
    "build_companion_package",
    "load_companion_regime_bootstrap",
    "load_companion_temporal_bootstrap",
    "scenarios_dir",
)
