"""Vertical: pair-programmer engineering partner.

Public API mirrors ``lifeform-domain-emogpt`` so the service-layer
vertical registry and the evolution loops can treat the two
interchangeably:

* ``build_coding_package`` \u2014 returns the canonical
  ``DomainExperiencePackage`` for the pair-programmer archetype.
* ``build_coding_vitals_bootstrap`` \u2014 returns the vertical's drive set
  (``solution_clarity`` / ``code_freshness`` / ``direction_certainty``).
* ``scenarios_dir`` \u2014 absolute path to the JSON scenario pack shipped
  with this vertical, suitable for passing to
  ``lifeform-bench --scenarios PATH``.
* ``bootstraps_dir`` \u2014 absolute path to the pre-trained bootstrap
  artifacts (``coding-temporal.snap`` + ``coding-regime.bs``) when
  shipped. Empty by default; produced by ``lifeform-super-loop --vertical
  coding --save-temporal ... --save-regime ...``.
* ``load_coding_temporal_bootstrap`` /
  ``load_coding_regime_bootstrap`` \u2014 typed loaders that fail loudly on
  schema-version drift, identical envelope shape to the companion
  vertical's loaders.
* ``build_coding_lifeform`` \u2014 convenience factory returning a
  ready-to-run ``Lifeform`` with the domain pack, vitals, and (when
  artifacts are shipped) pre-trained bootstraps wired in.

The contrast with ``lifeform-domain-emogpt`` is the proof of trigger \u2461 in
``SPLIT.md``: the kernel ships zero awareness of which vertical is loaded,
and a second domain wheel needed nothing inside the kernel to land. Both
verticals now also share the same training pipeline:
``lifeform-super-loop --vertical {companion,coding}`` produces a temporal
snapshot + regime bootstrap that the corresponding ``build_*_lifeform``
loads automatically.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Any

from volvence_zero.regime import RegimeBootstrap
from volvence_zero.temporal import MetacontrollerParameterSnapshot

from lifeform_domain_coding.coding_affordances import (
    CODING_AFFORDANCE_DESCRIPTORS,
    CONSENT_FILESYSTEM_READ,
    SandboxPathError,
    build_coding_affordance_backends,
    build_coding_affordance_invoker,
    build_coding_affordance_registry,
    resolve_sandbox_path,
)
from lifeform_domain_coding.coding_pack import build_coding_package
from lifeform_domain_coding.coding_vitals import build_coding_vitals_bootstrap


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def scenarios_dir() -> pathlib.Path:
    """Return the directory containing this vertical's scripted scenarios."""
    return pathlib.Path(__file__).resolve().parent / "scenarios"


def bootstraps_dir() -> pathlib.Path:
    """Return the directory containing the vertical's pre-trained bootstraps.

    Files inside the directory:

    * ``coding-temporal.snap`` \u2014 ``MetacontrollerParameterSnapshot``
      produced by ``lifeform-super-loop --vertical coding`` over this
      vertical's scenarios.
    * ``coding-regime.bs`` \u2014 ``RegimeBootstrap`` produced by the same run.

    Both use pickle envelopes with magic-byte headers identical in shape
    to the companion vertical's. Missing files are not an error \u2014 the
    factory falls back to flat kernel defaults until they are produced.
    """
    return pathlib.Path(__file__).resolve().parent / "bootstraps"


# ---------------------------------------------------------------------------
# Pickle envelope readers
# ---------------------------------------------------------------------------


_TEMPORAL_MAGIC = b"VZ-METASNAP\x00"
_TEMPORAL_SCHEMA_VERSION = "vz-metasnap.v1"
_REGIME_MAGIC = b"VZ-REGIMEBS\x00"
_REGIME_SCHEMA_VERSION = "vz-regimebs.v1"


def _read_envelope(
    *, path: pathlib.Path, magic: bytes, schema_version: str
) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Coding vertical artifact not found: {path}")
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


def load_coding_temporal_bootstrap() -> MetacontrollerParameterSnapshot:
    """Load the coding vertical's pre-trained metacontroller snapshot."""
    artifact = _read_envelope(
        path=bootstraps_dir() / "coding-temporal.snap",
        magic=_TEMPORAL_MAGIC,
        schema_version=_TEMPORAL_SCHEMA_VERSION,
    )
    return artifact.snapshot


def load_coding_regime_bootstrap() -> RegimeBootstrap:
    """Load the coding vertical's pre-trained regime bootstrap."""
    artifact = _read_envelope(
        path=bootstraps_dir() / "coding-regime.bs",
        magic=_REGIME_MAGIC,
        schema_version=_REGIME_SCHEMA_VERSION,
    )
    return artifact.bootstrap


def has_coding_temporal_bootstrap() -> bool:
    return (bootstraps_dir() / "coding-temporal.snap").is_file()


def has_coding_regime_bootstrap() -> bool:
    return (bootstraps_dir() / "coding-regime.bs").is_file()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_coding_lifeform(
    *,
    config: object | None = None,
    use_vitals_bootstrap: bool = True,
    use_temporal_bootstrap: bool = True,
    use_regime_bootstrap: bool = True,
    substrate_runtime: Any = None,
) -> Any:
    """Build a Lifeform with the pair-programmer vertical fully wired in.

    Steps:

    1. Construct a ``LifeformConfig`` (default if not provided) with
       rare-heavy disabled for deterministic behaviour. If
       ``substrate_runtime`` is supplied the brain config is forced into
       ``substrate_mode="injected"`` so sessions share that runtime.
    2. Apply this vertical's ``DomainExperiencePackage`` so the kernel
       ships with engineering-pair knowledge / cases / playbook /
       boundary priors.
    3. Wire the vertical's drive set via
       ``LifeformConfig.vitals_bootstrap`` so idle ticks produce
       engineering-flavoured slow-scale PE.
    4. Inject the vertical's pre-trained ``temporal_bootstrap`` /
       ``regime_bootstrap`` when the artifacts are present on disk;
       silently fall back to flat kernel defaults otherwise. Set the
       corresponding ``use_*_bootstrap`` flags to False for ablation runs
       even when artefacts exist.
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

    temporal_bootstrap: MetacontrollerParameterSnapshot | None = None
    if use_temporal_bootstrap and has_coding_temporal_bootstrap():
        temporal_bootstrap = load_coding_temporal_bootstrap()

    regime_bootstrap: RegimeBootstrap | None = None
    if use_regime_bootstrap and has_coding_regime_bootstrap():
        regime_bootstrap = load_coding_regime_bootstrap()

    return Lifeform(
        base_config,
        substrate_runtime=substrate_runtime,
        temporal_bootstrap=temporal_bootstrap,
        regime_bootstrap=regime_bootstrap,
    )


__all__ = (
    "bootstraps_dir",
    "build_coding_affordance_backends",
    "build_coding_affordance_invoker",
    "build_coding_affordance_registry",
    "build_coding_lifeform",
    "build_coding_package",
    "build_coding_vitals_bootstrap",
    "CODING_AFFORDANCE_DESCRIPTORS",
    "CONSENT_FILESYSTEM_READ",
    "has_coding_regime_bootstrap",
    "has_coding_temporal_bootstrap",
    "load_coding_regime_bootstrap",
    "load_coding_temporal_bootstrap",
    "resolve_sandbox_path",
    "SandboxPathError",
    "scenarios_dir",
)
