"""Convenience facade for building a Lifeform from a GrowthAdvisorProfile.

Mirrors :func:`lifeform_domain_character.build_character_lifeform` and
:func:`lifeform_domain_emogpt.build_companion_lifeform`: takes a
profile, compiles it through the three pure builders in
:mod:`lifeform_domain_growth_advisor.compiler`, and returns a
:class:`lifeform_core.Lifeform` ready for ``create_session`` with the
package + vitals attached.

Why a facade exists:

A consumer that just wants to spin up "Cheng Laoshi as a lifeform"
should not have to know which builder produces which artifact, or
how :class:`LifeformConfig` is composed. The facade encodes the
canonical wiring once.

Why the ingestion envelope is RETURNED rather than auto-drained:

Ingestion through the canonical path is intrinsically a per-session
operation. The facade builds a :class:`Lifeform`, which is a per-
process factory; it has no session to drain into yet. So we return
``GrowthAdvisorLifeformBundle(lifeform, profile, ingestion_envelope)``
and let the caller decide when to drain — typically right after
``create_session`` and before the first user turn.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _replace
from typing import Any

from lifeform_core import Lifeform, LifeformConfig
from lifeform_ingestion import IngestionComplianceProfile, IngestionEnvelope
from volvence_zero.memory import MemoryStore

from lifeform_domain_growth_advisor.compiler import (
    build_growth_advisor_ingestion_envelope,
    build_growth_advisor_package,
    build_growth_advisor_vitals_bootstrap,
)
from lifeform_domain_growth_advisor.profile import GrowthAdvisorProfile


@dataclass(frozen=True)
class GrowthAdvisorLifeformBundle:
    """Bundle returned by :func:`build_growth_advisor_lifeform`.

    The caller drains the optional ingestion envelope into a session
    before the first user turn (typical pattern: build, create_session,
    drain envelope, then run user turns).
    """

    lifeform: Lifeform
    profile: GrowthAdvisorProfile
    ingestion_envelope: IngestionEnvelope | None


def build_growth_advisor_lifeform(
    profile: GrowthAdvisorProfile | None = None,
    *,
    config: LifeformConfig | None = None,
    sample_excerpt: str | None = None,
    sample_uploader: str = "lifeform-domain-growth-advisor",
    sample_max_chunk_chars: int = 1024,
    sample_compliance_profile: IngestionComplianceProfile = (
        IngestionComplianceProfile.FORCED
    ),
    use_vitals_bootstrap: bool = True,
    memory_store: MemoryStore | None = None,
    substrate_runtime: Any = None,
    substrate_adapter_factory: Any = None,
    response_synthesizer: Any = None,
    semantic_proposal_runtime: Any = None,
    identity_provider: Any = None,
    rare_heavy_enabled: bool = False,
) -> GrowthAdvisorLifeformBundle:
    """Construct a Lifeform from a reviewed GrowthAdvisorProfile.

    Steps:

    1. Compile profile into a :class:`DomainExperiencePackage` and
       attach via ``LifeformConfig.with_domain_experience``.
    2. If ``use_vitals_bootstrap`` is true (default), compile the
       drive priors into a :class:`VitalsBootstrap` and attach via
       ``LifeformConfig.with_vitals``.
    3. Construct the :class:`Lifeform`. If ``memory_store`` is
       provided it is forwarded so multiple sessions / multiple
       lifeforms can share the same long-term memory.
    4. If ``sample_excerpt`` is provided, build an
       :class:`IngestionEnvelope` (source_kind=CORPUS) keyed to this
       profile and return it on the bundle for the caller to drain.

    Args:
        profile: Reviewed growth-advisor profile. Defaults to the
            shipped Cheng Laoshi reference profile when ``None``.
        config: Optional :class:`LifeformConfig` override. Defaults to
            a fresh ``LifeformConfig()`` with rare-heavy disabled.
        sample_excerpt: Optional plain-text excerpt to wrap as an
            ingestion envelope. Defaults to ``None`` (no envelope).
        sample_uploader: Operator id for envelope provenance.
        sample_max_chunk_chars: Chunk size for the ingestion envelope.
        sample_compliance_profile: Vitals override semantics for
            ingestion (default ``FORCED`` — operator-supplied content
            does not generate vitals resistance).
        use_vitals_bootstrap: When true, attach the advisor's vitals
            bootstrap. Set false for ablation runs.
        memory_store: Optional shared :class:`MemoryStore` for cross-
            session continuity.
        substrate_runtime: Optional pre-built substrate runtime to
            inject. When supplied, the brain is forced into
            ``substrate_mode='injected'``.
        substrate_adapter_factory: Optional callable for constructing
            per-turn substrate adapters when no substrate runtime is
            injected.
        response_synthesizer: Optional LLM expression synthesizer.
        semantic_proposal_runtime: Optional LLM-driven semantic
            proposal runtime.
        identity_provider: Optional :class:`UserIdentity` provider for
            cross-session memory scoping.
        rare_heavy_enabled: Forwarded to the brain config; defaults to
            False (deterministic behaviour for tests).

    Returns:
        A :class:`GrowthAdvisorLifeformBundle` carrying the
        constructed lifeform, the profile (for traceability), and an
        optional ingestion envelope.
    """
    if profile is None:
        from lifeform_domain_growth_advisor.profiles import (
            build_cheng_laoshi_profile,
        )

        profile = build_cheng_laoshi_profile()

    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    brain_overrides: dict[str, Any] = {"rare_heavy_enabled": rare_heavy_enabled}
    if substrate_runtime is not None:
        brain_overrides["substrate_mode"] = "injected"
    base_config = _replace(
        base_config,
        brain_config=_replace(base_config.brain_config, **brain_overrides),
    )
    package = build_growth_advisor_package(profile)
    base_config = base_config.with_domain_experience((package,))
    if use_vitals_bootstrap:
        base_config = base_config.with_vitals(
            build_growth_advisor_vitals_bootstrap(profile)
        )
    lifeform_kwargs: dict[str, Any] = {"memory_store": memory_store}
    if substrate_runtime is not None:
        lifeform_kwargs["substrate_runtime"] = substrate_runtime
    if substrate_adapter_factory is not None:
        lifeform_kwargs["substrate_adapter_factory"] = substrate_adapter_factory
    if response_synthesizer is not None:
        lifeform_kwargs["response_synthesizer"] = response_synthesizer
    if semantic_proposal_runtime is not None:
        lifeform_kwargs["semantic_proposal_runtime"] = semantic_proposal_runtime
    if identity_provider is not None:
        lifeform_kwargs["identity_provider"] = identity_provider
    lifeform = Lifeform(base_config, **lifeform_kwargs)
    envelope: IngestionEnvelope | None = None
    if sample_excerpt is not None and sample_excerpt.strip():
        envelope = build_growth_advisor_ingestion_envelope(
            profile,
            sample_excerpt,
            uploader=sample_uploader,
            max_chunk_chars=sample_max_chunk_chars,
            compliance_profile=sample_compliance_profile,
        )
    return GrowthAdvisorLifeformBundle(
        lifeform=lifeform,
        profile=profile,
        ingestion_envelope=envelope,
    )


def build_cheng_laoshi_lifeform(
    *,
    config: LifeformConfig | None = None,
    sample_excerpt: str | None = None,
    use_vitals_bootstrap: bool = True,
    memory_store: MemoryStore | None = None,
    substrate_runtime: Any = None,
    substrate_adapter_factory: Any = None,
    response_synthesizer: Any = None,
    semantic_proposal_runtime: Any = None,
    identity_provider: Any = None,
) -> GrowthAdvisorLifeformBundle:
    """Convenience: ``build_growth_advisor_lifeform`` pre-bound to 谌老师.

    The shipped reference profile lives in
    :mod:`lifeform_domain_growth_advisor.profiles.cheng_laoshi`. Useful
    for demos / smoke tests that do not need to construct their own
    profile.

    All kernel-side wiring kwargs are forwarded to
    :func:`build_growth_advisor_lifeform`.
    """
    from lifeform_domain_growth_advisor.profiles import (
        build_cheng_laoshi_profile,
    )

    return build_growth_advisor_lifeform(
        build_cheng_laoshi_profile(),
        config=config,
        sample_excerpt=sample_excerpt,
        use_vitals_bootstrap=use_vitals_bootstrap,
        memory_store=memory_store,
        substrate_runtime=substrate_runtime,
        substrate_adapter_factory=substrate_adapter_factory,
        response_synthesizer=response_synthesizer,
        semantic_proposal_runtime=semantic_proposal_runtime,
        identity_provider=identity_provider,
    )


__all__ = [
    "GrowthAdvisorLifeformBundle",
    "build_cheng_laoshi_lifeform",
    "build_growth_advisor_lifeform",
]
