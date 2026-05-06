"""Vertical registry \u2014 maps vertical name to a Lifeform factory.

The service layer must not import any specific ``lifeform-domain-*``
wheel directly in route code (that would couple the service to a single
vertical). Instead we keep the import locally here, so when more verticals
are added (``lifeform-domain-coding``, ``lifeform-domain-customer-service``,
\u2026) only this module changes.

A vertical entry is a small dataclass with:

* ``name`` \u2014 stable string identifier shipped in API responses.
* ``factory`` \u2014 callable that takes an optional shared
  ``OpenWeightResidualRuntime`` and returns a ``Lifeform``. The service
  builds the runtime exactly once at startup and passes the same instance
  to every ``factory()`` call so all sessions share one model on one GPU.
  Pass ``None`` for the synthetic / per-session-runtime fallback.
* ``has_temporal_bootstrap`` / ``has_regime_bootstrap`` \u2014 advertised
  capability flags so ``GET /v1/info`` can tell the client whether the
  vertical ships pre-trained calibration.
* ``bootstraps_dir`` / ``scenarios_dir`` \u2014 paths shown in ``/v1/info``
  for diagnostics; ``None`` when the vertical does not ship them.

If the importing wheel is missing (e.g. someone ran ``pip install
lifeform-service`` without ``lifeform-domain-emogpt``), the registry
catches the ImportError and skips that vertical rather than failing
service start. This keeps a future ``lifeform-service`` install useful
even with vertical-only optional deps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lifeform_core import Lifeform

if TYPE_CHECKING:
    from volvence_zero.memory import IdentityProvider
    from volvence_zero.substrate import OpenWeightResidualRuntime


VerticalFactory = Callable[["OpenWeightResidualRuntime | None"], Lifeform]
AlphaVerticalFactory = Callable[
    ["OpenWeightResidualRuntime | None", "IdentityProvider", str | None],
    Lifeform,
]


@dataclass(frozen=True)
class VerticalSpec:
    name: str
    factory: VerticalFactory
    has_temporal_bootstrap: bool
    has_regime_bootstrap: bool
    bootstraps_dir: str | None = None
    scenarios_dir: str | None = None
    alpha_factory: AlphaVerticalFactory | None = None


def _expression_synthesizer_for_runtime(
    runtime: "OpenWeightResidualRuntime | None",
    *,
    repair_alpha_enabled: bool = False,
):
    """Use the LLM expression path only when a real/shared runtime exists."""

    if runtime is None:
        from lifeform_expression import GroundedResponseSynthesizer, PromptPlanner

        return GroundedResponseSynthesizer(
            planner=PromptPlanner(repair_alpha_enabled=repair_alpha_enabled)
        )
    from lifeform_expression import LifeformLLMResponseSynthesizer, PromptPlanner

    return LifeformLLMResponseSynthesizer(
        runtime=runtime,
        planner=PromptPlanner(repair_alpha_enabled=repair_alpha_enabled),
    )


def _try_companion() -> VerticalSpec | None:
    try:
        from lifeform_domain_emogpt import (
            bootstraps_dir,
            build_companion_lifeform,
            scenarios_dir,
        )
    except ImportError:
        return None
    bdir = bootstraps_dir()
    sdir = scenarios_dir()
    def alpha_factory(runtime, identity_provider, memory_scope_root_dir):
        from lifeform_core import LifeformConfig
        from volvence_zero.brain import BrainConfig

        config = LifeformConfig(
            brain_config=BrainConfig(memory_scope_root_dir=memory_scope_root_dir)
        )
        return build_companion_lifeform(
            config=config,
            substrate_runtime=runtime,
            identity_provider=identity_provider,
            response_synthesizer=_expression_synthesizer_for_runtime(
                runtime,
                repair_alpha_enabled=True,
            ),
        )

    return VerticalSpec(
        name="companion",
        factory=lambda runtime: build_companion_lifeform(
            substrate_runtime=runtime,
            response_synthesizer=_expression_synthesizer_for_runtime(runtime),
        ),
        has_temporal_bootstrap=(bdir / "companion-temporal.snap").is_file(),
        has_regime_bootstrap=(bdir / "companion-regime.bs").is_file(),
        bootstraps_dir=str(bdir) if bdir.is_dir() else None,
        scenarios_dir=str(sdir) if sdir.is_dir() else None,
        alpha_factory=alpha_factory,
    )


def _try_uncalibrated_companion() -> VerticalSpec | None:
    """Bare companion vertical without the pre-trained bootstraps.

    Always reachable as ``companion-cold`` for ablation runs, even when
    the bootstrap files are intentionally missing.
    """
    try:
        from lifeform_domain_emogpt import build_companion_lifeform
    except ImportError:
        return None
    def alpha_factory(runtime, identity_provider, memory_scope_root_dir):
        from lifeform_core import LifeformConfig
        from volvence_zero.brain import BrainConfig

        config = LifeformConfig(
            brain_config=BrainConfig(memory_scope_root_dir=memory_scope_root_dir)
        )
        return build_companion_lifeform(
            config=config,
            use_temporal_bootstrap=False,
            use_regime_bootstrap=False,
            substrate_runtime=runtime,
            identity_provider=identity_provider,
            response_synthesizer=_expression_synthesizer_for_runtime(
                runtime,
                repair_alpha_enabled=True,
            ),
        )

    return VerticalSpec(
        name="companion-cold",
        factory=lambda runtime: build_companion_lifeform(
            use_temporal_bootstrap=False,
            use_regime_bootstrap=False,
            substrate_runtime=runtime,
            response_synthesizer=_expression_synthesizer_for_runtime(runtime),
        ),
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        alpha_factory=alpha_factory,
    )


def _try_coding() -> VerticalSpec | None:
    """Pair-programmer engineering-partner vertical.

    Pre-trained bootstraps may or may not ship: the factory checks at
    construction time and only loads them when present. Verticals
    without artifacts still produce a fully-functional Lifeform with
    the kernel's flat regime priors as the fallback.
    """
    try:
        from lifeform_domain_coding import (
            bootstraps_dir,
            build_coding_lifeform,
            has_coding_regime_bootstrap,
            has_coding_temporal_bootstrap,
            scenarios_dir,
        )
    except ImportError:
        return None
    bdir = bootstraps_dir()
    sdir = scenarios_dir()
    return VerticalSpec(
        name="coding",
        factory=lambda runtime: build_coding_lifeform(substrate_runtime=runtime),
        has_temporal_bootstrap=has_coding_temporal_bootstrap(),
        has_regime_bootstrap=has_coding_regime_bootstrap(),
        bootstraps_dir=str(bdir) if bdir.is_dir() else None,
        scenarios_dir=str(sdir) if sdir.is_dir() else None,
    )


_BUILDERS = (_try_companion, _try_uncalibrated_companion, _try_coding)


def discover_verticals() -> dict[str, VerticalSpec]:
    """Return every vertical that successfully imports in this environment."""
    out: dict[str, VerticalSpec] = {}
    for builder in _BUILDERS:
        spec = builder()
        if spec is not None:
            out[spec.name] = spec
    return out


def default_vertical_name() -> str:
    """First vertical that loaded successfully wins as default.

    In normal installs that is ``companion``. In a kernel-only install
    (lifeform-service without lifeform-domain-emogpt) we raise rather than
    silently fall through to no default.
    """
    discovered = discover_verticals()
    if not discovered:
        raise RuntimeError(
            "No lifeform-domain-* wheel is installed alongside lifeform-service. "
            "Install lifeform-domain-emogpt (or another vertical) before serving."
        )
    return next(iter(discovered.keys()))
