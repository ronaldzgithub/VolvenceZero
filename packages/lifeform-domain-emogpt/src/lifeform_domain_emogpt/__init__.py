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
from volvence_zero.substrate import SemanticFeatureSurfaceSubstrateAdapter
from volvence_zero.temporal import MetacontrollerParameterSnapshot

from lifeform_domain_emogpt.companion_pack import build_companion_package
from lifeform_domain_emogpt.companion_vitals import build_companion_vitals_bootstrap


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
    use_vitals_bootstrap: bool = True,
    substrate_runtime: Any = None,
    semantic_proposal_runtime: Any = None,
    memory_store: Any = None,
    response_synthesizer: Any = None,
) -> Any:
    """Build a Lifeform with the companion vertical fully wired in.

    Steps:

    1. Construct a ``LifeformConfig`` (default if not provided) with
       rare-heavy disabled for deterministic behaviour. If
       ``substrate_runtime`` is supplied the brain config is forced into
       ``substrate_mode="injected"`` so the brain consumes the supplied
       runtime rather than building a fresh one per session \u2014 this is
       the multi-session-shares-one-Qwen path.
    2. Apply this vertical's ``DomainExperiencePackage`` so the kernel
       ships with companion knowledge / cases / playbook / boundary
       priors.
    3. Optionally inject the vertical's pre-trained
       ``temporal_bootstrap`` and ``regime_bootstrap`` so a fresh session
       starts from the supervised-calibration basin instead of the flat
       defaults.

    Returns a ``lifeform_core.Lifeform`` instance ready for
    ``create_session``.

    Args:
        config: optional ``LifeformConfig`` override.
        use_temporal_bootstrap: load the vertical's pre-trained
            metacontroller snapshot. Set False for ablation runs.
        use_regime_bootstrap: load the vertical's pre-trained regime
            selection-weights bootstrap. Set False for ablation runs.
        substrate_runtime: optional pre-built ``OpenWeightResidualRuntime``.
            When supplied, the resulting ``Lifeform``'s sessions all share
            this one runtime instance. Required for multi-tenant services
            on a single GPU \u2014 otherwise every session would load Qwen
            weights independently. The runtime MUST have
            ``allow_live_substrate_mutation=False`` (the default) so
            sharing does not corrupt one session's weights from another's
            updates; this is enforced fail-loud at the service layer.
    """
    from dataclasses import replace as _replace
    from lifeform_core import Lifeform, LifeformConfig

    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    brain_overrides: dict[str, Any] = {"rare_heavy_enabled": False}
    if substrate_runtime is not None:
        # Force the brain config into ``injected`` mode so the supplied
        # runtime is used as-is rather than being treated as a synthetic
        # fallback (in synthetic mode the injected runtime is used too,
        # but injected mode makes the intent explicit and would also
        # raise loudly if substrate_runtime is somehow None elsewhere).
        brain_overrides["substrate_mode"] = "injected"
    base_config = _replace(
        base_config,
        brain_config=_replace(base_config.brain_config, **brain_overrides),
    )
    base_config = base_config.with_domain_experience(
        (build_companion_package(),)
    )

    if use_vitals_bootstrap:
        base_config = base_config.with_vitals(build_companion_vitals_bootstrap())

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
        substrate_runtime=substrate_runtime,
        substrate_adapter_factory=(
            None
            if substrate_runtime is not None
            else lambda user_input, turn_index: SemanticFeatureSurfaceSubstrateAdapter(
                source_text=user_input,
                model_id=f"companion-semantic-surface:{turn_index}",
                fallback_active=1.0,
            )
        ),
        response_synthesizer=response_synthesizer,
        semantic_proposal_runtime=semantic_proposal_runtime,
        memory_store=memory_store,
    )


def __getattr__(name: str) -> Any:
    """Lazy re-exports for the real-substrate helper.

    Importing ``volvence_zero.substrate`` pulls torch + transformers
    transitively. We don't want every consumer of
    ``lifeform_domain_emogpt`` to pay that startup cost; the
    real-substrate helper imports lazily from ``real_substrate.py``
    only when the caller actually asks for it.
    """
    if name in {
        "CompanionLifeformBundle",
        "DEFAULT_REAL_MODEL_ID",
        "DEFAULT_REAL_MODEL_SOURCE",
        "build_companion_lifeform_with_real_substrate",
    }:
        from lifeform_domain_emogpt import real_substrate

        return getattr(real_substrate, name)
    raise AttributeError(
        f"module 'lifeform_domain_emogpt' has no attribute {name!r}"
    )


__all__ = (
    "bootstraps_dir",
    "build_companion_lifeform",
    "build_companion_lifeform_with_real_substrate",
    "build_companion_package",
    "build_companion_vitals_bootstrap",
    "CompanionLifeformBundle",
    "DEFAULT_REAL_MODEL_ID",
    "DEFAULT_REAL_MODEL_SOURCE",
    "load_companion_regime_bootstrap",
    "load_companion_temporal_bootstrap",
    "scenarios_dir",
)
