"""Convenience facade for building a Lifeform from a CharacterSoulProfile.

Mirrors :func:`lifeform_domain_emogpt.build_companion_lifeform` but for
the reviewed-character vertical: takes a profile, compiles it through
the three pure builders, and returns a :class:`lifeform_core.Lifeform`
with the package + vitals applied.

Why a facade exists:

A consumer that just wants to spin up a "Zhang Wuji as a lifeform"
should not have to know which builder produces which artifact, or
how :class:`LifeformConfig` is composed. The facade encodes the
canonical wiring — domain experience package + vitals bootstrap +
optional ingestion envelope — once, in one place.

Why the ingestion envelope is RETURNED rather than auto-drained:

Ingestion through the canonical path
(``LifeformSession.run_turn(..., trigger_kind=INGESTION)``) is
intrinsically a per-session operation. The facade builds a
:class:`Lifeform`, which is a per-process factory; it has no session
to drain into yet. So we return ``(Lifeform, IngestionEnvelope |
None)`` and let the caller decide when to drain — typically right
after ``create_session`` and before the first user turn.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _replace
from typing import Any

from lifeform_core import Lifeform, LifeformConfig
from lifeform_ingestion import IngestionComplianceProfile, IngestionEnvelope
from volvence_zero.memory import MemoryStore

from lifeform_domain_character.compiler import (
    build_character_ingestion_envelope,
    build_character_package,
    build_character_vitals_bootstrap,
)
from lifeform_domain_character.profile import CharacterSoulProfile


@dataclass(frozen=True)
class CharacterLifeformBundle:
    """Bundle of everything needed to start running a character lifeform.

    Returned by :func:`build_character_lifeform`. The caller drains
    the optional ingestion envelope into a session before the first
    user turn (typical pattern: build, create_session, drain envelope,
    then run user turns).
    """

    lifeform: Lifeform
    profile: CharacterSoulProfile
    ingestion_envelope: IngestionEnvelope | None


def build_character_lifeform(
    profile: CharacterSoulProfile,
    *,
    config: LifeformConfig | None = None,
    novel_excerpt: str | None = None,
    novel_uploader: str = "lifeform-domain-character",
    novel_max_chunk_chars: int = 1024,
    novel_compliance_profile: IngestionComplianceProfile = (
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
) -> CharacterLifeformBundle:
    """Construct a Lifeform from a reviewed CharacterSoulProfile.

    Steps:

    1. Compile profile into a :class:`DomainExperiencePackage` and
       attach it via ``LifeformConfig.with_domain_experience``.
    2. If ``use_vitals_bootstrap`` is true (default), compile the
       drive priors into a :class:`VitalsBootstrap` and attach via
       ``LifeformConfig.with_vitals``.
    3. Construct the :class:`Lifeform`. If ``memory_store`` is
       provided it is forwarded so multiple sessions / multiple
       lifeforms can share the same long-term memory (the canonical
       Tier 3 evidence pattern).
    4. If ``novel_excerpt`` is provided, build an
       :class:`IngestionEnvelope` (source_kind=BOOK) keyed to this
       profile and return it on the bundle for the caller to drain.

    Args:
        profile: Reviewed character profile.
        config: Optional :class:`LifeformConfig` override. Defaults
            to a fresh ``LifeformConfig()`` with rare-heavy disabled.
        novel_excerpt: Optional plain-text excerpt to wrap as an
            ingestion envelope. ``None`` means no envelope is built;
            the bundle's ``ingestion_envelope`` is then ``None``.
        novel_uploader: Operator id for envelope provenance.
        novel_max_chunk_chars: Chunk size for the ingestion envelope
            (default 1024; smaller than ``lifeform_ingestion``'s
            default 2048 so a moderate excerpt produces enough chunks
            to exercise the multi-chunk path).
        novel_compliance_profile: Vitals override semantics for
            ingestion (default ``FORCED`` — operator-supplied content
            does not generate vitals resistance).
        use_vitals_bootstrap: When true, attach the character's
            vitals bootstrap. Set false for ablation runs.
        memory_store: Optional shared :class:`MemoryStore` for
            cross-session continuity. When ``None``, the lifeform's
            sessions get fresh per-session stores (default behaviour).

    Returns:
        A :class:`CharacterLifeformBundle` carrying the constructed
        lifeform, the profile (for traceability), and an optional
        ingestion envelope.
    """
    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    # Service paths (alpha mode + real substrate) need
    # ``substrate_mode='injected'`` so the brain consumes the supplied
    # runtime as-is. We mirror ``build_companion_lifeform``'s pattern.
    brain_overrides: dict[str, Any] = {"rare_heavy_enabled": rare_heavy_enabled}
    if substrate_runtime is not None:
        brain_overrides["substrate_mode"] = "injected"
    base_config = _replace(
        base_config,
        brain_config=_replace(base_config.brain_config, **brain_overrides),
    )
    package = build_character_package(profile)
    base_config = base_config.with_domain_experience((package,))
    if use_vitals_bootstrap:
        base_config = base_config.with_vitals(
            build_character_vitals_bootstrap(profile)
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
    if novel_excerpt is not None and novel_excerpt.strip():
        envelope = build_character_ingestion_envelope(
            profile,
            novel_excerpt,
            uploader=novel_uploader,
            max_chunk_chars=novel_max_chunk_chars,
            compliance_profile=novel_compliance_profile,
        )
    return CharacterLifeformBundle(
        lifeform=lifeform,
        profile=profile,
        ingestion_envelope=envelope,
    )


def build_zhang_wuji_lifeform(
    *,
    config: LifeformConfig | None = None,
    novel_excerpt: str | None = None,
    use_vitals_bootstrap: bool = True,
    memory_store: MemoryStore | None = None,
    substrate_runtime: Any = None,
    substrate_adapter_factory: Any = None,
    response_synthesizer: Any = None,
    semantic_proposal_runtime: Any = None,
    identity_provider: Any = None,
) -> CharacterLifeformBundle:
    """Convenience: ``build_character_lifeform`` pre-bound to 张无忌.

    The shipped reference profile lives in
    ``lifeform_domain_character.profiles.zhang_wuji``. Useful for
    demos / smoke tests that do not need to construct their own
    profile.

    All kernel-side wiring kwargs (``substrate_runtime``,
    ``response_synthesizer``, ``semantic_proposal_runtime``,
    ``identity_provider`` …) are forwarded to
    :func:`build_character_lifeform` so the browser-chat / service
    paths can hand a real Qwen runtime + LLM expression layer + LLM
    semantic proposal runtime in a single call.
    """
    from lifeform_domain_character.profiles import build_zhang_wuji_profile

    return build_character_lifeform(
        build_zhang_wuji_profile(),
        config=config,
        novel_excerpt=novel_excerpt,
        use_vitals_bootstrap=use_vitals_bootstrap,
        memory_store=memory_store,
        substrate_runtime=substrate_runtime,
        substrate_adapter_factory=substrate_adapter_factory,
        response_synthesizer=response_synthesizer,
        semantic_proposal_runtime=semantic_proposal_runtime,
        identity_provider=identity_provider,
    )


__all__ = [
    "CharacterLifeformBundle",
    "build_character_lifeform",
    "build_zhang_wuji_lifeform",
]
