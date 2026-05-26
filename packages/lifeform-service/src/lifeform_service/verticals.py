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

from lifeform_service.templates import VerticalTemplateAdapter

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
    # Template I/O is opt-in per vertical. When set, the chat-browser
    # ``GET /v1/templates`` lists this vertical's templates,
    # ``POST /v1/sessions`` accepts ``template_id`` to spawn from one,
    # and ``POST /v1/sessions/{id}/save-as-template`` is enabled. A
    # vertical that leaves ``template_adapter=None`` keeps the legacy
    # factory-only behavior (no template selector in UI). See
    # :mod:`lifeform_service.templates` for the Protocol contract.
    template_adapter: VerticalTemplateAdapter | None = None
    # Default sub-directory name (relative to the service-level
    # templates root) that the adapter scans. ``None`` means use the
    # vertical's own ``name`` as the sub-dir; a vertical that wants a
    # different layout sets this explicitly. Absolute paths are NOT
    # accepted here — the service-level root is the policy boundary.
    template_subdir: str | None = None


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

    Alpha mode (``alpha_factory``): closed-alpha runs go through
    :class:`SessionManager`'s ``bind_session_legacy_alias`` path so
    ``scope_key == user_id`` (1-layer) — same contract as the
    companion / zhang_wuji / einstein verticals on this lane. The
    forward-looking 2-layer scope path (``scope_key == "alpha:end_user_id"``,
    debt #69 P2 growth-advisor admin endpoint) lives outside
    :class:`SessionManager` and is independent of this factory;
    callers that wire that path must not also mint sessions for the
    same end_user through this 1-layer alpha_factory without an
    explicit migration, otherwise scoped memory would split across
    two keys for the same human user.

    LLM semantic-proposal runtime + per-end-user archetype classifier
    (debt #66 ACTIVE wiring) are *not* attached here: the semantic-
    proposal runtime is opt-in at the lifeform_builder level, and
    archetype classifier needs its own LLM client/config. Both stay
    consistent across the dev ``factory`` and the alpha ``alpha_factory``
    paths so the only thing the alpha path adds is per-user identity
    + scoped memory root — nothing surprising leaks in.
    """
    try:
        from lifeform_domain_growth_advisor import (
            build_growth_advisor_lifeform,
            scenarios_dir,
        )
    except ImportError:
        return None

    def alpha_factory(runtime, identity_provider, memory_scope_root_dir):
        from lifeform_core import LifeformConfig
        from volvence_zero.brain import BrainConfig

        config = LifeformConfig(
            brain_config=BrainConfig(memory_scope_root_dir=memory_scope_root_dir)
        )
        return build_growth_advisor_lifeform(
            config=config,
            substrate_runtime=runtime,
            identity_provider=identity_provider,
            response_synthesizer=_expression_synthesizer_for_runtime(
                runtime, repair_alpha_enabled=True
            ),
        ).lifeform

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
        alpha_factory=alpha_factory,
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

    Template surface: this vertical registers a
    :class:`CharacterTemplateAdapter` so the chat-browser UI can
    list / load / save :class:`LifeformTemplate` JSON files. When a
    user picks "no template" the adapter's
    ``build_default_session_context`` builds the base 张无忌 profile;
    when they pick a template id the adapter routes through
    :func:`give_birth`. Save-as-Template captures the current
    session's drives + memory back into a new template. The legacy
    ``ZHANG_WUJI_TEMPLATE_PATH`` env var is still honored as a
    one-shot startup default (useful for ``start_browser_chat_zhang_wuji.*``
    convenience wrappers).
    """
    try:
        from lifeform_domain_character import (
            build_character_template_adapter,
            build_zhang_wuji_lifeform,
            give_birth,
        )
    except ImportError:
        return None

    import os
    import pathlib

    def _legacy_env_template_path() -> pathlib.Path | None:
        raw = os.environ.get("ZHANG_WUJI_TEMPLATE_PATH", "").strip()
        if not raw:
            return None
        path = pathlib.Path(raw)
        return path if path.is_file() else None

    def factory(runtime):
        synthesizer = _expression_synthesizer_for_runtime(runtime)
        semantic_runtime = _build_llm_semantic_runtime_from_runtime(runtime)
        template_path = _legacy_env_template_path()
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
        template_path = _legacy_env_template_path()
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

    template_adapter = build_character_template_adapter(
        response_synthesizer_factory=_expression_synthesizer_for_runtime,
        semantic_proposal_runtime_factory=_build_llm_semantic_runtime_from_runtime,
    )

    return VerticalSpec(
        name="zhang_wuji",
        factory=factory,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        alpha_factory=alpha_factory,
        template_adapter=template_adapter,
        template_subdir="zhang_wuji",
    )


def _attach_figure_bundle(synthesizer, bundle):
    """Bind ``bundle`` onto ``synthesizer`` via the duck-typed hook.

    Returns the bound synthesizer when the hook is callable, else the
    original (kernel fallback / GroundedResponseSynthesizer carries no
    figure_bundle path; L1/L3/L4 enforcement only runs on the LLM
    synthesizer).
    """

    attach = getattr(synthesizer, "with_figure_bundle", None)
    if callable(attach):
        return attach(bundle)
    return synthesizer


def _resolve_einstein_artifact_bundle(synthetic_fallback):
    """Pick the artifact bundle to attach to the synthesizer.

    Reads :func:`lifeform_service.einstein_resolver.resolve_einstein_bundle`
    (disk-backed Wave K artefact path); falls back to
    ``synthetic_fallback`` (typically the bundle produced by a fresh
    ``build_einstein_lifeform`` call) when no disk bundle is
    available AND ``EINSTEIN_REQUIRE_REAL_BUNDLE`` is unset.

    The resolver itself fail-louds on schema / integrity violations;
    this helper only widens the result into the
    ``(bundle, bundle_id, source)`` triple the verticals layer logs.
    """

    from lifeform_service.einstein_resolver import resolve_einstein_bundle

    resolution = resolve_einstein_bundle()
    if resolution.bundle is None:
        return synthetic_fallback, "", "synthetic"
    return resolution.bundle, resolution.bundle_id, resolution.source


def _register_einstein_persona_lora_if_present(bundle) -> str | None:
    """Register the disk bundle's LoRA artefact into the default pool.

    No-op when ``bundle.lora`` is ``None`` (the resolver picked a
    base / curated bundle without an F6 artefact). Returns the pool
    record id on success so the startup log can surface it.

    The pool is process-wide, so registration persists across all
    three Einstein verticals in the same process. ``einstein-bundle``
    is **not** insulated from this registration today (see module
    docstring's "Known limitations" note); the demo-time mitigation
    is that synthetic LoRA backends produce a zero-delta forward
    (debt #40), so ``einstein-bundle`` and ``einstein-full`` behave
    identically until #41 lands a real PEFT artefact.
    """

    from lifeform_service.figure_bundle_store import register_bundle_persona_lora

    return register_bundle_persona_lora(bundle)


def _build_einstein_lifeform_with_bundle(
    *,
    runtime,
    use_alpha: bool,
    identity_provider=None,
    memory_scope_root_dir=None,
    attach_bundle: bool,
    register_lora: bool,
    condition_label: str,
):
    """Construct an Einstein :class:`Lifeform` for a given condition.

    Three call shapes correspond to the three ablation conditions
    in :mod:`lifeform_domain_figure.verification.persona.runtime_conditions`:

    * ``attach_bundle=False, register_lora=False`` -> RAW (no L1/L3/L4
      enforcement on the LLM synthesizer).
    * ``attach_bundle=True,  register_lora=False`` -> BUNDLE
      (L1/L3/L4 active; persona-LoRA auto-activate falls through).
    * ``attach_bundle=True,  register_lora=True``  -> BUNDLE_LORA
      (LoRA artefact registered in the default pool; synthesizer's
      auto-activate path triggers on each turn).

    The lifeform's ``domain_package`` + ``vitals_bootstrap`` always
    come from the reviewed Einstein profile -- those are not what we
    are ablating. We are ablating the L1/L3/L4 layer + the LoRA hook,
    which live on the response synthesizer.
    """

    from lifeform_core import LifeformConfig
    from lifeform_domain_figure import build_einstein_lifeform

    config: LifeformConfig | None = None
    if use_alpha:
        from volvence_zero.brain import BrainConfig

        config = LifeformConfig(
            brain_config=BrainConfig(
                memory_scope_root_dir=memory_scope_root_dir
            )
        )

    seed_bundle = build_einstein_lifeform(
        config=config,
        substrate_runtime=runtime,
        identity_provider=identity_provider if use_alpha else None,
    )

    if attach_bundle:
        artifact, bundle_id, source = _resolve_einstein_artifact_bundle(
            synthetic_fallback=seed_bundle.artifact_bundle
        )
        base_synthesizer = _expression_synthesizer_for_runtime(
            runtime, repair_alpha_enabled=use_alpha
        )
        bound_synthesizer = _attach_figure_bundle(base_synthesizer, artifact)
        lora_record_id: str | None = None
        if register_lora:
            lora_record_id = _register_einstein_persona_lora_if_present(
                artifact
            )
        rebound = build_einstein_lifeform(
            config=config,
            substrate_runtime=runtime,
            response_synthesizer=bound_synthesizer,
            identity_provider=identity_provider if use_alpha else None,
        )
        _log_einstein_vertical_ready(
            condition_label=condition_label,
            bundle_id=bundle_id,
            source=source,
            lora_record_id=lora_record_id,
            lora_present_in_bundle=getattr(artifact, "lora", None) is not None,
        )
        return rebound.lifeform

    base_synthesizer = _expression_synthesizer_for_runtime(
        runtime, repair_alpha_enabled=use_alpha
    )
    rebound = build_einstein_lifeform(
        config=config,
        substrate_runtime=runtime,
        response_synthesizer=base_synthesizer,
        identity_provider=identity_provider if use_alpha else None,
    )
    _log_einstein_vertical_ready(
        condition_label=condition_label,
        bundle_id="",
        source="raw",
        lora_record_id=None,
        lora_present_in_bundle=False,
    )
    return rebound.lifeform


def _log_einstein_vertical_ready(
    *,
    condition_label: str,
    bundle_id: str,
    source: str,
    lora_record_id: str | None,
    lora_present_in_bundle: bool,
) -> None:
    """Single-line stdout breadcrumb so operators can audit the wiring."""

    import sys

    lora_label = (
        f"lora_record={lora_record_id}"
        if lora_record_id
        else (
            "lora_artifact=in-bundle (not registered)"
            if lora_present_in_bundle
            else "lora=absent"
        )
    )
    print(
        f"[verticals] einstein condition={condition_label} "
        f"bundle_id={bundle_id or '-'} source={source} {lora_label}",
        file=sys.stderr,
        flush=True,
    )


def _try_einstein_raw() -> VerticalSpec | None:
    """RAW ablation arm: pure base LLM, no figure_bundle on the synthesizer.

    Matches ``PersonaCondition.RAW`` in
    :mod:`lifeform_domain_figure.verification.persona.runtime_conditions`:
    no L1 (style hint), no L3 (grounded decoder), no L4 (scope refusal).
    The lifeform still carries the reviewed Einstein profile (domain
    package + vitals + drives) -- without that the kernel has no
    figure identity at all; the ablation is on the expression-time
    enforcement, not on the kernel.
    """

    try:
        from lifeform_domain_figure import build_einstein_lifeform  # noqa: F401
    except ImportError:
        return None

    def factory(runtime):
        return _build_einstein_lifeform_with_bundle(
            runtime=runtime,
            use_alpha=False,
            attach_bundle=False,
            register_lora=False,
            condition_label="raw",
        )

    def alpha_factory(runtime, identity_provider, memory_scope_root_dir):
        return _build_einstein_lifeform_with_bundle(
            runtime=runtime,
            use_alpha=True,
            identity_provider=identity_provider,
            memory_scope_root_dir=memory_scope_root_dir,
            attach_bundle=False,
            register_lora=False,
            condition_label="raw",
        )

    return VerticalSpec(
        name="einstein-raw",
        factory=factory,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        alpha_factory=alpha_factory,
    )


def _try_einstein_bundle() -> VerticalSpec | None:
    """BUNDLE ablation arm: L1/L3/L4 enforcement on, no persona-LoRA.

    Matches ``PersonaCondition.BUNDLE``: the synthesizer carries the
    figure bundle (disk-backed Wave K artefact when available, else
    the synthetic fallback). Persona-LoRA registration is skipped on
    this factory; if a sibling ``einstein-full`` factory has already
    registered a LoRA into the process-wide pool, the synthesizer's
    auto-activate hook will pick it up (this is the documented
    cross-vertical pool-sharing limitation today; once debt #41 lands
    a real PEFT artefact, a future ``persona_lora_enabled`` flag on
    the synthesizer will let the three verticals coexist with true
    forward-level isolation).
    """

    try:
        from lifeform_domain_figure import build_einstein_lifeform  # noqa: F401
    except ImportError:
        return None

    def factory(runtime):
        return _build_einstein_lifeform_with_bundle(
            runtime=runtime,
            use_alpha=False,
            attach_bundle=True,
            register_lora=False,
            condition_label="bundle",
        )

    def alpha_factory(runtime, identity_provider, memory_scope_root_dir):
        return _build_einstein_lifeform_with_bundle(
            runtime=runtime,
            use_alpha=True,
            identity_provider=identity_provider,
            memory_scope_root_dir=memory_scope_root_dir,
            attach_bundle=True,
            register_lora=False,
            condition_label="bundle",
        )

    return VerticalSpec(
        name="einstein-bundle",
        factory=factory,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        alpha_factory=alpha_factory,
    )


def _try_einstein_full() -> VerticalSpec | None:
    """BUNDLE_LORA ablation arm: L1/L3/L4 + persona-LoRA active.

    Matches ``PersonaCondition.BUNDLE_LORA``: the disk bundle's
    ``lora`` slot (when populated by an F6 bake) is registered into
    the process-wide :class:`PersonaLoRAPool`, so the synthesizer's
    ``_maybe_activate_persona_lora`` hook fires on each turn.

    Known limitation (today): when the resolver hands back a bundle
    whose ``lora`` slot was filled by the synthetic backend, the
    LoRA delta is zeroed by the substrate's LayerNorm before any
    head logits change (known-debts #40), so the forward output is
    byte-equivalent to the BUNDLE condition. Once debt #41 lands a
    real PEFT-on-Qwen LoRA, the same wiring will surface a real
    behavioural delta with no code change here.
    """

    try:
        from lifeform_domain_figure import build_einstein_lifeform  # noqa: F401
    except ImportError:
        return None

    def factory(runtime):
        return _build_einstein_lifeform_with_bundle(
            runtime=runtime,
            use_alpha=False,
            attach_bundle=True,
            register_lora=True,
            condition_label="bundle_lora",
        )

    def alpha_factory(runtime, identity_provider, memory_scope_root_dir):
        return _build_einstein_lifeform_with_bundle(
            runtime=runtime,
            use_alpha=True,
            identity_provider=identity_provider,
            memory_scope_root_dir=memory_scope_root_dir,
            attach_bundle=True,
            register_lora=True,
            condition_label="bundle_lora",
        )

    return VerticalSpec(
        name="einstein-full",
        factory=factory,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        alpha_factory=alpha_factory,
    )


def _try_einstein() -> VerticalSpec | None:
    """Backward-compatibility alias for ``einstein-bundle``.

    The original ``einstein`` vertical (Wave F4.2) shipped a single
    factory that carried the figure_bundle. We now split that into
    three ablation arms (``einstein-raw`` / ``einstein-bundle`` /
    ``einstein-full``) matching the verification harness's
    ``PersonaCondition``; this shim keeps the legacy ``einstein``
    name resolving to the closest-equivalent (BUNDLE) arm so
    existing ``VERTICAL=einstein`` invocations and saved templates
    do not break.
    """

    spec = _try_einstein_bundle()
    if spec is None:
        return None
    return VerticalSpec(
        name="einstein",
        factory=spec.factory,
        has_temporal_bootstrap=spec.has_temporal_bootstrap,
        has_regime_bootstrap=spec.has_regime_bootstrap,
        bootstraps_dir=spec.bootstraps_dir,
        scenarios_dir=spec.scenarios_dir,
        alpha_factory=spec.alpha_factory,
        template_adapter=spec.template_adapter,
        template_subdir=spec.template_subdir,
    )


def _try_novel_worlds_character() -> VerticalSpec | None:
    """novel-worlds character vertical.

    Generic CharacterTemplateAdapter binding for the novel-worlds app.
    Distinct from `_try_zhang_wuji` because:
      - template_subdir = "novel-worlds" so DLaaS scans
        /data/novel-bundles/novel-worlds/ (the path the
        apps/novel-worlds/workers/bake-worker writes to);
      - default factory falls back to the upstream zhang_wuji profile
        when no template_id is supplied, since the adapter requires a
        base profile but novel-worlds requests always include
        template_id;
      - runtime_template_id "novel-worlds.character.v0" is what the
        BFF passes on wake / adoption calls.
    """
    try:
        from lifeform_domain_character import (
            build_character_template_adapter,
            build_zhang_wuji_lifeform,
        )
    except ImportError:
        return None

    def factory(runtime):
        synthesizer = _expression_synthesizer_for_runtime(runtime)
        semantic_runtime = _build_llm_semantic_runtime_from_runtime(runtime)
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
        return build_zhang_wuji_lifeform(
            config=config,
            substrate_runtime=runtime,
            response_synthesizer=synthesizer,
            semantic_proposal_runtime=semantic_runtime,
            identity_provider=identity_provider,
        ).lifeform

    template_adapter = build_character_template_adapter(
        response_synthesizer_factory=_expression_synthesizer_for_runtime,
        semantic_proposal_runtime_factory=_build_llm_semantic_runtime_from_runtime,
    )

    return VerticalSpec(
        name="novel-worlds-character",
        factory=factory,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        alpha_factory=alpha_factory,
        template_adapter=template_adapter,
        template_subdir="novel-worlds",
    )


_BUILDERS = (
    _try_companion,
    _try_uncalibrated_companion,
    _try_coding,
    _try_zhang_wuji,
    _try_novel_worlds_character,
    _try_einstein,
    _try_einstein_raw,
    _try_einstein_bundle,
    _try_einstein_full,
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
