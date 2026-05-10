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


def _try_growth_advisor() -> VerticalSpec | None:
    """Long-term private-domain growth-advisor vertical (LTV path).

    Compiles the shipped Cheng Laoshi reference profile into a
    Lifeform; the vertical does not yet ship pre-trained
    metacontroller / regime bootstraps, so a fresh session starts
    from the kernel's flat defaults plus the reviewed profile-derived
    seeds (knowledge / cases / playbook / boundaries / drives).
    """
    try:
        from lifeform_domain_growth_advisor import (
            build_growth_advisor_lifeform,
            scenarios_dir,
        )
    except ImportError:
        return None

    sdir = scenarios_dir()
    return VerticalSpec(
        name="growth_advisor",
        factory=lambda runtime: build_growth_advisor_lifeform(
            substrate_runtime=runtime,
            response_synthesizer=_expression_synthesizer_for_runtime(runtime),
        ).lifeform,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        scenarios_dir=str(sdir) if sdir.is_dir() else None,
    )


def _build_llm_semantic_runtime_from_runtime(runtime):
    """Wrap a TransformersOpenWeightResidualRuntime's underlying HF
    model + tokenizer in an :class:`LLMSemanticProposalRuntime`.

    This is the bridge that activates the ToM, common-ground and
    semantic-state owners (commitment / boundary / goal-value) on
    the LLM expression path. Without it those owners stay in their
    fail-closed empty state even when a real Qwen runtime is wired
    behind the response synthesizer.

    The runtime's HF objects are private (``_model`` / ``_tokenizer``);
    we mirror the access pattern from
    ``lifeform_domain_emogpt.real_substrate.build_companion_lifeform_with_real_substrate``
    which constructs the same provider for the companion vertical.
    Falls back to ``None`` (no LLM proposal runtime) when the runtime
    does not expose HF internals — e.g. the builtin synthetic
    fallback runtime, which would only emit theatre against random
    weights.
    """

    if runtime is None:
        return None
    model = getattr(runtime, "_model", None)
    tokenizer = getattr(runtime, "_tokenizer", None)
    device = getattr(runtime, "_device", "cpu")
    if model is None or tokenizer is None:
        return None
    from volvence_zero.semantic_state.llm_runtime import (
        LLMSemanticProposalRuntime,
    )
    from volvence_zero.substrate.text_generation import (
        HFTextGenerationProvider,
    )

    provider = HFTextGenerationProvider(
        model=model, tokenizer=tokenizer, device=device
    )
    return LLMSemanticProposalRuntime(provider=provider)


def _try_zhang_wuji() -> VerticalSpec | None:
    """张无忌 character vertical.

    Compiles the shipped reference profile into a Lifeform; when a
    real substrate runtime is wired the LLM expression synthesizer
    and the LLM semantic-proposal runtime are auto-attached so ToM
    / common-ground / commitment / boundary / goal-value owners run
    against the live Qwen weights.

    Optional env-controlled rebirth: when ``ZHANG_WUJI_TEMPLATE_PATH``
    is set to an existing JSON template file, the factory routes
    through :func:`give_birth` instead of :func:`build_zhang_wuji_lifeform`,
    so the served lifeform inherits the saved drive levels, memory
    checkpoint, and (optionally) evolved profile from a prior
    experiential-replay run.
    """
    try:
        from lifeform_domain_character import (
            build_zhang_wuji_lifeform,
            give_birth,
        )
    except ImportError:
        return None

    import os
    import pathlib

    def _template_path() -> pathlib.Path | None:
        raw = os.environ.get("ZHANG_WUJI_TEMPLATE_PATH", "").strip()
        if not raw:
            return None
        path = pathlib.Path(raw)
        return path if path.is_file() else None

    def factory(runtime):
        synthesizer = _expression_synthesizer_for_runtime(runtime)
        semantic_runtime = _build_llm_semantic_runtime_from_runtime(runtime)
        template_path = _template_path()
        if template_path is not None:
            bundle = give_birth(
                template_path,
                substrate_runtime=runtime,
                response_synthesizer=synthesizer,
                semantic_proposal_runtime=semantic_runtime,
            )
            return bundle.lifeform
        return build_zhang_wuji_lifeform(
            substrate_runtime=runtime,
            response_synthesizer=synthesizer,
            semantic_proposal_runtime=semantic_runtime,
        ).lifeform

    def alpha_factory(runtime, identity_provider, memory_scope_root_dir):
        from lifeform_core import LifeformConfig
        from volvence_zero.brain import BrainConfig

        synthesizer = _expression_synthesizer_for_runtime(
            runtime, repair_alpha_enabled=True
        )
        semantic_runtime = _build_llm_semantic_runtime_from_runtime(runtime)
        config = LifeformConfig(
            brain_config=BrainConfig(
                memory_scope_root_dir=memory_scope_root_dir
            )
        )
        template_path = _template_path()
        if template_path is not None:
            # Alpha mode + saved template: keep the template's
            # profile / drives / evolved profile as the "trained"
            # base, but skip the template's frozen memory checkpoint
            # so the alpha kernel can build a per-user
            # filesystem-scoped store. Each user's chat then
            # accumulates ON TOP of the saved drives instead of
            # inheriting another user's lived memories.
            bundle = give_birth(
                template_path,
                config=config,
                substrate_runtime=runtime,
                response_synthesizer=synthesizer,
                semantic_proposal_runtime=semantic_runtime,
                identity_provider=identity_provider,
                skip_memory_restore=True,
            )
            return bundle.lifeform
        return build_zhang_wuji_lifeform(
            config=config,
            substrate_runtime=runtime,
            response_synthesizer=synthesizer,
            semantic_proposal_runtime=semantic_runtime,
            identity_provider=identity_provider,
        ).lifeform

    return VerticalSpec(
        name="zhang_wuji",
        factory=factory,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        alpha_factory=alpha_factory,
    )


def _try_einstein() -> VerticalSpec | None:
    """Real-person figure vertical: Albert Einstein (synthetic placeholder corpus).

    Builds an Einstein lifeform that combines the existing
    domain-experience compilation path (knowledge / cases / playbook /
    boundaries) with the figure-vertical's runtime artifact bundle
    (retrieval index, coverage map, style prior). The bundle is
    attached to the LLM expression synthesizer via
    :meth:`LifeformLLMResponseSynthesizer.with_figure_bundle` so the
    L1 / L3 / L4 enforcement layers (style injector, grounded
    decoder, scope refuser) can consume it on each turn.

    Steering (F5) and persona LoRA (F6) are not yet wired by default
    — those packets will populate ``bundle.steering`` /
    ``bundle.lora`` and the synthesizer side will pick them up
    through the same hook without further changes here.
    """

    try:
        from lifeform_core import LifeformConfig
        from lifeform_domain_figure import build_einstein_lifeform
    except ImportError:
        return None

    def _attach_bundle(synthesizer, bundle):
        attach = getattr(synthesizer, "with_figure_bundle", None)
        if callable(attach):
            return attach(bundle)
        return synthesizer

    def factory(runtime):
        base_synthesizer = _expression_synthesizer_for_runtime(runtime)
        # Build with a placeholder synthesizer first so we can read
        # the resulting artifact_bundle, then re-bind the same
        # synthesizer with the bundle attached. The lifeform's
        # response_synthesizer is mutable via Lifeform.with_*
        # only at construction; we bind once here and pass it in.
        # Steps: build bundle (no synthesizer wired yet), bind
        # synthesizer to bundle, then construct lifeform.
        bundle = build_einstein_lifeform(
            substrate_runtime=runtime,
        )
        bound_synthesizer = _attach_bundle(
            base_synthesizer, bundle.artifact_bundle
        )
        rebound = build_einstein_lifeform(
            substrate_runtime=runtime,
            response_synthesizer=bound_synthesizer,
        )
        return rebound.lifeform

    def alpha_factory(runtime, identity_provider, memory_scope_root_dir):
        from volvence_zero.brain import BrainConfig

        config = LifeformConfig(
            brain_config=BrainConfig(
                memory_scope_root_dir=memory_scope_root_dir
            )
        )
        base_synthesizer = _expression_synthesizer_for_runtime(
            runtime, repair_alpha_enabled=True
        )
        bundle = build_einstein_lifeform(
            config=config,
            substrate_runtime=runtime,
            identity_provider=identity_provider,
        )
        bound_synthesizer = _attach_bundle(
            base_synthesizer, bundle.artifact_bundle
        )
        rebound = build_einstein_lifeform(
            config=config,
            substrate_runtime=runtime,
            response_synthesizer=bound_synthesizer,
            identity_provider=identity_provider,
        )
        return rebound.lifeform

    return VerticalSpec(
        name="einstein",
        factory=factory,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        alpha_factory=alpha_factory,
    )


_BUILDERS = (
    _try_companion,
    _try_uncalibrated_companion,
    _try_coding,
    _try_zhang_wuji,
    _try_einstein,
    _try_growth_advisor,
)


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
