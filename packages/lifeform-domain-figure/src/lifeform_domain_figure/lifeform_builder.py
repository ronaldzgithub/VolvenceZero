"""Convenience facade for building a Lifeform from a HistoricalFigureProfile.

Mirrors :func:`lifeform_domain_character.build_character_lifeform` but
for the real-person figure vertical: takes a profile + corpus
envelopes, compiles them through the F1 / F2 builders, and returns a
:class:`lifeform_core.Lifeform` configured with the figure-vertical
domain package + vitals bootstrap, with the
:class:`FigureArtifactBundle` available for runtime injection into
the LLM expression synthesizer.

Why a facade exists:

A consumer that just wants to spin up an "Einstein lifeform" should
not have to know which builder produces which artifact, or how
:class:`LifeformConfig` is composed. The facade encodes the
canonical wiring — domain experience package + vitals bootstrap +
figure artifact bundle — once, in one place.

Why we expose the artifact bundle:

The L1 / L3 / L4 enforcement layers in ``lifeform-expression``
(``StylePriorInjector`` / ``GroundedDecoder`` / ``ScopeRefuser``)
consume the bundle through duck-typed Protocols. The bundle is
attached to the LLM synthesizer at session-construction time; this
facade returns the bundle alongside the lifeform so the caller can
manage attachment without re-deriving anything.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _replace
from typing import Any

from lifeform_core import Lifeform, LifeformConfig
from lifeform_ingestion import IngestionEnvelope
from volvence_zero.memory import MemoryStore

from lifeform_domain_figure.compiler import (
    FigureBundleInputs,
    build_figure_artifact_bundle,
)
from lifeform_domain_figure.envelope_builder import FigureCorpusSourceBundle
from lifeform_domain_figure.figure_artifact import FigureArtifactBundle
from lifeform_domain_figure.profile import HistoricalFigureProfile


@dataclass(frozen=True)
class FigureLifeformBundle:
    """Bundle of everything needed to start running a figure lifeform.

    Returned by :func:`build_figure_lifeform`. The caller drains the
    optional ingestion envelopes into a session before the first
    user turn (typical pattern: build, create_session, drain
    envelopes, then run user turns) and / or attaches the artifact
    bundle to the LLM synthesizer via
    :meth:`LifeformLLMResponseSynthesizer.with_figure_bundle`.
    """

    lifeform: Lifeform
    profile: HistoricalFigureProfile
    artifact_bundle: FigureArtifactBundle
    ingestion_envelopes: tuple[IngestionEnvelope, ...]


def build_figure_lifeform(
    profile: HistoricalFigureProfile,
    corpus_bundle: FigureCorpusSourceBundle,
    *,
    config: LifeformConfig | None = None,
    use_vitals_bootstrap: bool = True,
    memory_store: MemoryStore | None = None,
    substrate_runtime: Any = None,
    substrate_adapter_factory: Any = None,
    response_synthesizer: Any = None,
    semantic_proposal_runtime: Any = None,
    identity_provider: Any = None,
    rare_heavy_enabled: bool = False,
    time_window_id: str | None = None,
    extra_style_terms: tuple[str, ...] = (),
    uploader: str = "lifeform-domain-figure:builder",
) -> FigureLifeformBundle:
    """Construct a Lifeform from a reviewed HistoricalFigureProfile.

    Steps:

    1. Build a typed ``IngestionEnvelopeSet`` from the corpus bundle.
    2. Compile the profile + envelopes into a
       :class:`FigureArtifactBundle` (retrieval index, coverage map,
       style prior, plus the compiled domain package and vitals
       bootstrap).
    3. Attach ``domain_package`` to the :class:`LifeformConfig` and,
       if ``use_vitals_bootstrap`` is true, the vitals bootstrap as
       well.
    4. Construct the :class:`Lifeform` with the standard kernel-side
       wiring (substrate runtime, response synthesizer, etc.).
    5. Return the lifeform + the immutable artifact bundle so the
       caller can attach the bundle to the LLM synthesizer at
       session creation time.
    """

    from lifeform_domain_figure.envelope_builder import (
        build_figure_ingestion_envelope,
    )

    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    brain_overrides: dict[str, Any] = {"rare_heavy_enabled": rare_heavy_enabled}
    if substrate_runtime is not None:
        brain_overrides["substrate_mode"] = "injected"
    base_config = _replace(
        base_config,
        brain_config=_replace(base_config.brain_config, **brain_overrides),
    )
    envelope_set = build_figure_ingestion_envelope(
        corpus_bundle, uploader=uploader
    )
    inputs = FigureBundleInputs(
        profile=profile,
        envelopes=envelope_set.envelopes,
        time_window_id=time_window_id,
        extra_style_terms=extra_style_terms,
    )
    artifact_bundle = build_figure_artifact_bundle(inputs)
    base_config = base_config.with_domain_experience(
        (artifact_bundle.domain_package,)
    )
    if use_vitals_bootstrap:
        base_config = base_config.with_vitals(artifact_bundle.vitals_bootstrap)
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
    return FigureLifeformBundle(
        lifeform=lifeform,
        profile=artifact_bundle.profile,
        artifact_bundle=artifact_bundle,
        ingestion_envelopes=envelope_set.envelopes,
    )


def build_figure_lifeform_from_bundle(
    artifact_bundle: FigureArtifactBundle,
    *,
    config: LifeformConfig | None = None,
    use_vitals_bootstrap: bool = True,
    memory_store: MemoryStore | None = None,
    substrate_runtime: Any = None,
    substrate_adapter_factory: Any = None,
    response_synthesizer: Any = None,
    semantic_proposal_runtime: Any = None,
    identity_provider: Any = None,
) -> FigureLifeformBundle:
    """Wake a Lifeform from an already-baked :class:`FigureArtifactBundle`.

    This is the generic ``figure-full <slug>`` factory (D25): given a
    bundle that the bake pipeline already compiled + persisted to disk
    (resolved by ``slug`` via ``bundle_io.load_figure_bundle`` or the
    service-layer figure resolver), construct the runnable Lifeform with
    **no per-figure code**. The bundle already carries the figure's
    ``domain_package`` (L4/playbook/boundary owners), ``vitals_bootstrap``
    (drive shape) and ``profile``; we attach those to a
    :class:`LifeformConfig` and build the kernel exactly as
    :func:`build_figure_lifeform` does, but without re-ingesting or
    re-compiling the corpus (the bundle's integrity hash already pins
    what the corpus produced — R15).

    Returned ``FigureLifeformBundle.ingestion_envelopes`` is empty: a
    waked-from-bundle lifeform has no fresh envelopes to drain (the
    corpus lives behind the bundle's retrieval index, which the
    expression-layer L3 decoder reads through ``artifact_bundle``).

    Args mirror :func:`build_figure_lifeform`; the caller attaches the
    returned ``artifact_bundle`` to the LLM synthesizer via
    ``with_figure_bundle`` for the L1/L3/L4 enforcement path.
    """

    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    brain_overrides: dict[str, Any] = {"rare_heavy_enabled": False}
    if substrate_runtime is not None:
        brain_overrides["substrate_mode"] = "injected"
    base_config = _replace(
        base_config,
        brain_config=_replace(base_config.brain_config, **brain_overrides),
    )
    base_config = base_config.with_domain_experience(
        (artifact_bundle.domain_package,)
    )
    if use_vitals_bootstrap:
        base_config = base_config.with_vitals(artifact_bundle.vitals_bootstrap)
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
    return FigureLifeformBundle(
        lifeform=lifeform,
        profile=artifact_bundle.profile,
        artifact_bundle=artifact_bundle,
        ingestion_envelopes=(),
    )


def build_einstein_lifeform(
    *,
    config: LifeformConfig | None = None,
    use_vitals_bootstrap: bool = True,
    memory_store: MemoryStore | None = None,
    substrate_runtime: Any = None,
    substrate_adapter_factory: Any = None,
    response_synthesizer: Any = None,
    semantic_proposal_runtime: Any = None,
    identity_provider: Any = None,
    time_window_id: str | None = None,
) -> FigureLifeformBundle:
    """Convenience: ``build_figure_lifeform`` pre-bound to Einstein."""

    from lifeform_domain_figure.profiles import build_einstein_profile
    from lifeform_domain_figure.sample_corpus import synthetic_einstein_corpus

    profile = build_einstein_profile()
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    corpus_bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    return build_figure_lifeform(
        profile,
        corpus_bundle,
        config=config,
        use_vitals_bootstrap=use_vitals_bootstrap,
        memory_store=memory_store,
        substrate_runtime=substrate_runtime,
        substrate_adapter_factory=substrate_adapter_factory,
        response_synthesizer=response_synthesizer,
        semantic_proposal_runtime=semantic_proposal_runtime,
        identity_provider=identity_provider,
        time_window_id=time_window_id,
    )


__all__ = [
    "FigureLifeformBundle",
    "build_einstein_lifeform",
    "build_figure_lifeform",
    "build_figure_lifeform_from_bundle",
]
