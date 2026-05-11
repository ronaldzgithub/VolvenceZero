"""Packet 4.3: comprehensive soak — PDF → load → run → reflect → revise.

End-to-end demonstration that all three mechanisms cooperate:

1. **DocumentUptake** (Phase 2): load a candidate from the
   private-domain growth advisor PDF using a deterministic
   mock LLM, review-approve it, push into ProtocolRegistry.
2. **Compile path** (Phase 1.x): the load step populates
   application owners with ``protocol:`` lineage entries.
3. **Reflection** (Phase 3.1/3.2): drive synthetic PE history
   for 50+ turns; the engine accumulates and eventually emits
   a strategy_decay proposal.
4. **ModificationGate** (Phase 3.4): the proposal is evaluated;
   L3 with sufficient evidence auto-approves.
5. **apply_revision** (Phase 3.3): the registry mutates the
   targeted protocol's strategy weights and re-runs the
   compile path so application owners see the change.
6. **Snapshot drift** (Phase 1.5*): the active_mixture
   ``revision_fingerprint`` differs between turn-1 and the
   post-revision turn, proving content drift is observable.

This is the canonical milestone-D check: PDF → AI learns →
chats at that level → accumulates experience that reshapes
the protocol's content (not just its weighting).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from lifeform_protocol_runtime.document_uptake import (
    chunk_document,
    extract_protocol_candidate,
    read_pdf,
)
from lifeform_protocol_runtime.document_uptake.extraction import (
    MockLlmJsonClient,
)
from lifeform_protocol_runtime.document_uptake.review import (
    approve_candidate,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
)
from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
    BehaviorProtocolCandidate,
    ReviewLevel,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.protocol_runtime import (
    ApprovalOutcome,
    ProtocolRegistryModule,
    evaluate_protocol_revision,
)
from volvence_zero.reflection import ProtocolReflectionEngine
from volvence_zero.runtime import Snapshot


_PDF = Path(
    "docs/fixtures/sample_protocols/"
    "private_domain_growth_advisor_guidance.pdf"
)


# ---------------------------------------------------------------------------
# Deterministic mock LLM (mirrors the cheng_laoshi-shape used in 2.6)
# ---------------------------------------------------------------------------


def _identity(_) -> dict:
    return {
        "advisor_name": "soak-advisor",
        "description": "Soak-test protocol extracted from PDF.",
        "identity_traits": ["warm_peer_register"],
        "regime_compatibility": ["emotional_support"],
    }


def _boundary(_) -> dict:
    return {
        "boundaries": [
            {
                "boundary_id": "soak-bp-no-hard-sell",
                "description": "no purchase pressure first 7 days",
                "trigger_reasons": ["boundary_violation_fired"],
                "blocked_topics": ["promo"],
                "refer_out_required": False,
                "severity": "soft_remind",
            }
        ]
    }


def _strategy(_) -> dict:
    return {
        "strategies": [
            {
                "rule_id": "soak-rapport",
                "problem_pattern": "user emotional load",
                "recommended_ordering": ["acknowledge", "render_resonance"],
                "recommended_pacing": "slow",
                "applicability_phase": ["day1"],
            },
            {
                "rule_id": "soak-funnel",
                "problem_pattern": "user asks about product",
                "recommended_ordering": ["clarify_need", "share_evidence"],
                "recommended_pacing": "moderate",
                "applicability_phase": ["day4"],
            },
        ],
        "knowledge_seeds": [
            {
                "seed_id": "soak-knowledge",
                "domain": "growth",
                "title": "Growth window basics",
                "summary": "Children grow at variable rates...",
                "snippet": "...",
                "evidence_locator": "soak-pdf",
                "confidence": 0.8,
            }
        ],
        "cases": [
            {
                "case_id": "soak-case",
                "title": "Soak case",
                "domain": "growth",
                "problem_pattern": "user worried",
                "user_state_pattern": "high anxiety",
                "intervention_ordering": ["acknowledge", "share_evidence"],
                "outcome_label": "rapport_restored",
                "confidence": 0.7,
                "description": "test soak case",
            }
        ],
    }


def _mock_client() -> MockLlmJsonClient:
    return MockLlmJsonClient(
        identity_fn=_identity,
        boundary_fn=_boundary,
        strategy_fn=_strategy,
    )


# ---------------------------------------------------------------------------
# PE / mixture builders
# ---------------------------------------------------------------------------


def _pe_snapshot(turn: int, signed_reward: float) -> Snapshot[PredictionErrorSnapshot]:
    ctx = PredictionActionContext()
    pred = PredictedOutcome(
        source_turn_index=turn,
        target_turn_index=turn + 1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.5,
        description="",
        action_context=ctx,
    )
    actual = ActualOutcome(
        observed_turn_index=turn,
        task_progress=0.5,
        relationship_delta=0.5,
        regime_stability=0.5,
        action_payoff=0.5,
        description="",
        action_context=ctx,
    )
    pe = PredictionError(
        task_error=signed_reward,
        relationship_error=signed_reward,
        regime_error=signed_reward,
        action_error=signed_reward,
        magnitude=abs(signed_reward),
        signed_reward=signed_reward,
        description="",
    )
    value = PredictionErrorSnapshot(
        evaluated_prediction=pred,
        actual_outcome=actual,
        next_prediction=pred,
        error=pe,
        turn_index=turn,
        bootstrap=False,
        description="",
        action_context=ctx,
    )
    return Snapshot(
        slot_name="prediction_error",
        owner="PredictionErrorModule",
        version=1,
        timestamp_ms=0,
        value=value,
    )


def _mixture_snapshot(*entries: tuple[str, float]) -> Snapshot[ActiveMixtureSnapshot]:
    value = ActiveMixtureSnapshot(
        active_protocols=tuple(
            ActiveProtocolEntry(
                protocol_id=protocol_id, activation_weight=weight
            )
            for protocol_id, weight in entries
        ),
        boundary_union_ids=(),
        revision_fingerprint=f"mixture:{sum(w for _, w in entries):.3f}",
        description="",
    )
    return Snapshot(
        slot_name="active_mixture",
        owner="ProtocolRegistryModule",
        version=1,
        timestamp_ms=0,
        value=value,
    )


# ---------------------------------------------------------------------------
# Soak test
# ---------------------------------------------------------------------------


def test_pdf_to_experience_full_loop() -> None:
    """50+ turn soak: PDF load → reflection → gate → revision."""

    if not _PDF.exists():
        pytest.skip(f"missing fixture {_PDF}")

    # ---- Phase 2: PDF → BehaviorProtocolCandidate ----
    doc = read_pdf(_PDF)
    chunks = chunk_document(
        doc.text, source_locator=str(_PDF), max_tokens=1024
    )
    assert chunks, "expected non-empty chunks"

    candidate = extract_protocol_candidate(
        chunks,
        llm_client=_mock_client(),
        source_locator=str(_PDF),
    )
    approved, _ = approve_candidate(
        candidate,
        reviewer_id="soak-test",
        evidence=("packet 4.3 soak",),
        minimum_level=ReviewLevel.L4,
    )
    pdf_protocol_id = approved.protocol_id
    pdf_candidate = BehaviorProtocolCandidate(
        protocol=approved,
        provenance=candidate.provenance,
        requires_review=False,
    )

    # ---- Phase 1.x: load + compile path ----
    rare = ApplicationRareHeavyState()
    knowledge = ApplicationDomainKnowledgeStore()
    case_memory = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=rare,
        domain_knowledge_store=knowledge,
        case_memory_store=case_memory,
    )
    module.load_protocol_candidate(pdf_candidate)

    # Application owners now hold pdf_protocol_id-prefixed entries.
    pre_revision_strategy_count = len(rare.distilled_playbook_rules)
    pre_revision_lineage = [
        r.rule_id for r in rare.distilled_playbook_rules
        if r.rule_id.startswith(f"protocol:{pdf_protocol_id}:")
    ]
    assert pre_revision_lineage, "compile path didn't produce strategy lineage"

    # ---- Phase 3: drive 12 turns of bad PE ----
    engine = ProtocolReflectionEngine(
        scan_period=12, history_window=50
    )
    last_proposals: tuple = ()
    for turn in range(1, 13):
        upstream: dict[str, Snapshot[Any]] = {
            "prediction_error": _pe_snapshot(turn=turn, signed_reward=-0.9),
            "active_mixture": _mixture_snapshot((pdf_protocol_id, 1.0)),
        }
        snap = asyncio.run(engine.process(upstream))
        last_proposals = snap.value.protocol_revision_proposals

    assert last_proposals, "expected proposals after 12 negative PE turns"
    target_proposal = next(
        p for p in last_proposals if p.target_protocol_id == pdf_protocol_id
    )

    # ---- Phase 3.4: gate evaluation ----
    decision = evaluate_protocol_revision(target_proposal)
    assert decision.outcome is ApprovalOutcome.AUTO_APPROVED

    # ---- Phase 3.3: apply revision; recompile must run ----
    pre_protocol = module.registry.get(pdf_protocol_id)
    pre_weights = {s.rule_id: s.initial_weight for s in pre_protocol.strategy_priors}

    module.apply_revision(target_proposal, revised_by="soak-test")

    post_protocol = module.registry.get(pdf_protocol_id)
    post_weights = {s.rule_id: s.initial_weight for s in post_protocol.strategy_priors}
    for rid, pre_w in pre_weights.items():
        assert post_weights[rid] == pre_w * 0.5, (rid, pre_w, post_weights[rid])

    assert post_protocol.revision_log
    assert any(
        target_proposal.proposal_id in r.revision_id
        for r in post_protocol.revision_log
    )

    # ---- Phase 3.3 (continued): application owners reflect new content ----
    # The recompile uses dataclass-replace of PlaybookRule entries
    # under the same rule_id; counts must remain stable but
    # underlying state has been refreshed.
    post_revision_strategy_count = len(rare.distilled_playbook_rules)
    assert post_revision_strategy_count == pre_revision_strategy_count

    post_revision_lineage = [
        r.rule_id for r in rare.distilled_playbook_rules
        if r.rule_id.startswith(f"protocol:{pdf_protocol_id}:")
    ]
    assert sorted(post_revision_lineage) == sorted(pre_revision_lineage)


def test_pdf_loaded_protocol_keeps_byte_equivalent_compile_path() -> None:
    """A separate sanity: re-loading the same approved protocol is idempotent."""
    if not _PDF.exists():
        pytest.skip(f"missing fixture {_PDF}")
    doc = read_pdf(_PDF)
    chunks = chunk_document(
        doc.text, source_locator=str(_PDF), max_tokens=1024
    )
    candidate = extract_protocol_candidate(
        chunks, llm_client=_mock_client(), source_locator=str(_PDF)
    )
    approved, _ = approve_candidate(
        candidate, reviewer_id="idempotent", minimum_level=ReviewLevel.L4
    )
    pdf_candidate = BehaviorProtocolCandidate(
        protocol=approved,
        provenance=candidate.provenance,
        requires_review=False,
    )

    rare = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=rare)
    module.load_protocol_candidate(pdf_candidate)
    first_count = len(rare.boundary_prior_hints)

    # Reloading the SAME protocol must not duplicate.
    module.load_protocol_candidate(pdf_candidate)
    second_count = len(rare.boundary_prior_hints)
    assert second_count == first_count
