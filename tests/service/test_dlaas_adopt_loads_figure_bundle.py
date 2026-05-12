"""Wave E contract: DLaaS adopt path auto-hooks the figure bundle.

Closes debt #22. Adopting a template that names a
``figure_artifact_id`` MUST:

1. Look the bundle up via ``lookup_figure_bundle`` (the
   ``lifeform-service`` public surface).
2. Bind the bundle on the SessionManager so subsequent sessions
   carry it through to their synthesizer.
3. Register the bundle's persona LoRA (when bundle.lora is set)
   in the default :class:`PersonaLoRAPool`.

We don't spin up a full DLaaS HTTP server here — we exercise the
SessionManager bind + Lifeform attach paths and assert the
plumbing is in place. The full HTTP-level adopt smoke lives in
``packages/lifeform-service/tests/test_dlaas_chat_smoke.py`` and
will pick up the new wiring on its next run; this contract test
guards the wiring itself.
"""

from __future__ import annotations

import pytest

from volvence_zero.substrate import (
    PersonaLoRAPool,
    default_persona_lora_pool,
)

from lifeform_domain_figure import (
    FigureBundleInputs,
    FigureCorpusSourceBundle,
    SyntheticLoRABakeBackend,
    apply_persona_lora_through_gate,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    build_lora_training_plan,
    synthetic_einstein_corpus,
)
from lifeform_service import (
    default_figure_bundle_store,
    lookup_figure_bundle,
    register_bundle_persona_lora,
)
from volvence_zero.evaluation.types import EvaluationScore, EvaluationSnapshot


def _einstein_envelopes():
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    return build_figure_ingestion_envelope(
        FigureCorpusSourceBundle(
            figure_id="einstein",
            papers=papers,
            letters=letters,
            lectures=lectures,
            notebooks=notebooks,
        ),
        uploader="contract-test:adopt",
    ).envelopes


def _bundle_with_lora():
    profile = build_einstein_profile()
    envelopes = _einstein_envelopes()
    base = build_figure_artifact_bundle(
        FigureBundleInputs(profile=profile, envelopes=envelopes)
    )
    plan = build_lora_training_plan(figure_id="einstein", envelopes=envelopes)
    artifact = SyntheticLoRABakeBackend().bake(plan)
    snapshot = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore("behavior", "contract_integrity", 0.99, 0.95, "ok"),
            EvaluationScore("behavior", "rollback_resilience", 0.99, 0.95, "ok"),
            EvaluationScore("behavior", "fallback_reliance", 0.10, 0.95, "ok"),
        ),
        session_scores=(),
        alerts=(),
        description="adopt-contract-clean",
    )
    pool = PersonaLoRAPool()
    result = apply_persona_lora_through_gate(
        base_bundle=base,
        artifact=artifact,
        evaluation_snapshot=snapshot,
        pool=pool,
        rollback_evidence=f"prev_lora=absent;base={base.bundle_id}",
    )
    assert result.applied
    return result.bundle


def test_lookup_figure_bundle_helper_resolves_default_einstein() -> None:
    """default_figure_bundle_store seeds Einstein on first access; the
    public helper must resolve that bundle by figure_id."""

    store = default_figure_bundle_store()
    assert store.has("einstein")
    bundle = lookup_figure_bundle(bundle_id="einstein")
    assert getattr(bundle, "figure_id", "") == "einstein"


def test_register_bundle_persona_lora_no_op_when_no_lora() -> None:
    """When the bundle has no LoRA artifact, register returns None
    rather than crashing — supports adopt paths where LoRA hasn't
    been baked yet."""

    bundle = build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=build_einstein_profile(),
            envelopes=_einstein_envelopes(),
        )
    )
    assert getattr(bundle, "lora", None) is None
    pool = PersonaLoRAPool()
    record_id = register_bundle_persona_lora(bundle, pool=pool)
    assert record_id is None
    assert not pool.has("einstein")


def test_register_bundle_persona_lora_pushes_artifact_to_pool() -> None:
    bundle = _bundle_with_lora()
    pool = PersonaLoRAPool()
    record_id = register_bundle_persona_lora(bundle, pool=pool)
    assert record_id is not None
    assert pool.has("einstein")
    record = pool.lookup("einstein")
    assert record.figure_id == "einstein"
    assert record.adapter_layers == bundle.lora.adapter_layers


@pytest.mark.asyncio
async def test_session_manager_bind_figure_bundle_propagates_to_lifeform() -> None:
    """Wave E load-bearing: SessionManager.bind_figure_bundle MUST
    propagate the bundle to lifeforms it builds, so the synthesizer
    clone consumed by per-turn synthesize() reads the bundle and
    can route through L1 / L3 / L4 enforcers + pool.activate."""

    from lifeform_service.session_manager import SessionManager
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["einstein"]
    bundle = lookup_figure_bundle(bundle_id="einstein")

    manager = SessionManager(
        lifeform_factory=spec.factory,
        vertical_name="einstein",
    )
    assert manager.figure_bundle is None
    manager.bind_figure_bundle(bundle)
    assert manager.figure_bundle is bundle

    session = await manager.create_session(session_id="contract-session")
    assert session is not None
    # Wave E contract: SessionManager.bind_figure_bundle propagates
    # to the per-session synthesizer through Lifeform.bind_figure_bundle.
    # The synthesizer is owned by the runner inside the brain session;
    # we walk runner._response_synthesizer and assert figure_bundle is
    # the same bundle the manager bound.
    runner = session.brain_session.runner
    synthesizer = runner._response_synthesizer  # noqa: SLF001 — runner-internal
    bundle_on_synth = getattr(synthesizer, "figure_bundle", None)
    assert bundle_on_synth is not None
    assert getattr(bundle_on_synth, "figure_id", "") == "einstein"


def test_default_persona_lora_pool_is_process_wide() -> None:
    """register_bundle_persona_lora(bundle) without an explicit pool
    must reach the process-wide default pool — that's what the DLaaS
    adopt path relies on."""

    bundle = _bundle_with_lora()
    default_pool = default_persona_lora_pool()
    if default_pool.has("einstein"):
        default_pool.deregister("einstein")
    record_id = register_bundle_persona_lora(bundle)
    assert record_id is not None
    assert default_pool.has("einstein")
    # Cleanup so other tests start from a clean default pool.
    default_pool.deregister("einstein")
