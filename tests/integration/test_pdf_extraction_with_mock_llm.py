"""Packet 2.5: end-to-end PDF → BehaviorProtocol with a mock LLM.

CI-friendly version of ``test_pdf_to_protocol_e2e.py`` that
runs deterministically without an API key. Uses
:class:`MockLlmJsonClient` with chunk-aware canned responses so
the test exercises the full flow including chunking, multi-call
LLM iteration, candidate construction, review, and registry load.
"""

from __future__ import annotations

from pathlib import Path

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
    BehaviorProtocolCandidate,
    ReviewLevel,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule


_PDF_FIXTURE = Path(
    "docs/fixtures/sample_protocols/"
    "private_domain_growth_advisor_guidance.pdf"
)


# ---------------------------------------------------------------------------
# Chunk-aware mock responses (simulate a sensible extractor)
# ---------------------------------------------------------------------------


def _identity_for_chunk(_user_prompt: str) -> dict:
    """Identity is global; same on every chunk."""
    return {
        "advisor_name": "growth-advisor",
        "description": "Private-domain growth advisor extracted from the PDF.",
        "identity_traits": ["warm_peer_register", "long_horizon"],
        "regime_compatibility": ["emotional_support"],
    }


def _boundary_for_chunk(user_prompt: str) -> dict:
    """Return at least one boundary; vary by chunk so dedup is exercised."""
    # Emit a different boundary on roughly every chunk so the
    # accumulator exercises append + dedup.
    if "private" in user_prompt.lower() or "domain" in user_prompt.lower():
        return {
            "boundaries": [
                {
                    "boundary_id": "no-hard-sell",
                    "description": "Do not push purchases in first 7 days",
                    "trigger_reasons": ["boundary_violation_fired"],
                    "blocked_topics": ["promo", "discount"],
                    "refer_out_required": False,
                    "severity": "soft_remind",
                }
            ]
        }
    return {
        "boundaries": [
            {
                "boundary_id": "no-overclaim",
                "description": "Avoid medical / efficacy claims without disclaimer",
                "trigger_reasons": ["boundary_violation_fired"],
                "blocked_topics": ["cure", "guarantee"],
                "refer_out_required": True,
                "severity": "hard_block",
            }
        ]
    }


def _strategy_for_chunk(_user_prompt: str) -> dict:
    """Return strategy + knowledge + case content."""
    return {
        "strategies": [
            {
                "rule_id": "rapport-empathy",
                "problem_pattern": "User shares emotional load",
                "recommended_ordering": [
                    "acknowledge_pressure",
                    "render_resonance",
                ],
                "recommended_pacing": "slow",
                "avoid_patterns": ["solution_pitch"],
                "applicability_phase": ["day1"],
            }
        ],
        "knowledge_seeds": [
            {
                "seed_id": "growth-window",
                "domain": "child_growth",
                "title": "Growth windows 3-12",
                "summary": "Children aged 3-12 have variable growth rates...",
                "snippet": "...",
                "evidence_locator": "private_domain_guidance.pdf",
                "confidence": 0.8,
                "topic_tags": ["growth"],
                "jurisdiction_tags": ["cn"],
            }
        ],
        "cases": [
            {
                "case_id": "case-empathy-pivot",
                "title": "Empathy pivot day 3",
                "transcript_summary": "Mom said overwhelmed; pivoted from KB to empathy.",
                "lesson": "Empathy first when emotional load high",
                "domain": "private_domain_growth",
                "problem_pattern": "User feels overwhelmed",
                "user_state_pattern": "high emotional load",
                "intervention_ordering": ["acknowledge", "render_resonance"],
                "outcome_label": "rapport_restored",
                "confidence": 0.75,
            }
        ],
    }


def _make_mock_client() -> MockLlmJsonClient:
    return MockLlmJsonClient(
        identity_fn=_identity_for_chunk,
        boundary_fn=_boundary_for_chunk,
        strategy_fn=_strategy_for_chunk,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_pipeline_with_mock_llm() -> None:
    """End-to-end PDF → candidate → review → registry → application owners."""
    if not _PDF_FIXTURE.exists():
        pytest.skip(f"missing fixture {_PDF_FIXTURE}")

    doc = read_pdf(_PDF_FIXTURE)
    chunks = chunk_document(
        doc.text, source_locator=str(_PDF_FIXTURE), max_tokens=1024
    )
    assert chunks, "expected non-empty chunks"

    candidate = extract_protocol_candidate(
        chunks,
        llm_client=_make_mock_client(),
        source_locator=str(_PDF_FIXTURE),
    )

    # Candidate shape
    assert isinstance(candidate, BehaviorProtocolCandidate)
    assert candidate.requires_review is True
    assert len(candidate.protocol.boundary_contracts) >= 1
    assert len(candidate.protocol.strategy_priors) >= 1
    assert len(candidate.protocol.knowledge_seeds) >= 1
    assert len(candidate.protocol.signature_cases) >= 1

    # Review path
    approved, approval = approve_candidate(
        candidate,
        reviewer_id="ops-admin",
        evidence=("packet 2.5 mock e2e",),
        minimum_level=ReviewLevel.L4,
    )
    assert approval.review_level_required is ReviewLevel.L4
    approved_candidate = BehaviorProtocolCandidate(
        protocol=approved,
        provenance=candidate.provenance,
        requires_review=False,
    )

    # Load path
    rare = ApplicationRareHeavyState()
    knowledge = ApplicationDomainKnowledgeStore()
    case_memory = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=rare,
        domain_knowledge_store=knowledge,
        case_memory_store=case_memory,
    )
    module.load_protocol_candidate(approved_candidate)

    # Application owners now hold protocol-prefixed lineage
    assert any(
        h.hint_id.startswith("protocol:")
        for h in rare.boundary_prior_hints
    )
    assert any(
        r.rule_id.startswith("protocol:")
        for r in rare.distilled_playbook_rules
    )
    assert any(
        r.record_id.startswith("protocol:")
        for r in knowledge.records
    )
    assert any(
        r.case_id.startswith("protocol:")
        for r in case_memory.records
    )
