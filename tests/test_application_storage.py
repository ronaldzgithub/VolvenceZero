from __future__ import annotations

import asyncio
import tempfile

from volvence_zero.application import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    build_filesystem_persistence_backend,
)
from volvence_zero.application.runtime import EvidenceStrength, KnowledgeSourceType
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


def test_application_storage_round_trip_restores_domain_and_case_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        domain_backend = build_filesystem_persistence_backend(base_dir=f"{tmpdir}/domain")
        case_backend = build_filesystem_persistence_backend(base_dir=f"{tmpdir}/case")
        domain_store = ApplicationDomainKnowledgeStore(
            records=(
                DomainKnowledgeRecord(
                    record_id="knowledge:test:1",
                    domain="career_decision",
                    topic_tags=("career",),
                    jurisdiction_tags=("general",),
                    source_type="reviewed-article",
                    title="Career guidance",
                    locator="seed",
                    summary="Use a structured decision frame.",
                    snippet="Structured trade-off framing helps.",
                    freshness_label="current",
                    confidence=0.77,
                    evidence_strength="medium",
                ),
            ),
            persistence_backend=domain_backend,
        )
        case_store = ApplicationCaseMemoryStore(
            records=(
                CaseMemoryRecord(
                    case_id="case:test:1",
                    domain="structured_decision_patterns",
                    problem_pattern="structured-decision-overwhelm",
                    user_state_pattern="needs-structure",
                    risk_markers=("risk-medium",),
                    track_tags=("world",),
                    regime_tags=("problem_solving",),
                    intervention_ordering=("narrow_scope", "smallest_next_step"),
                    outcome_label="stable",
                    delayed_signal_count=3,
                    escalation_observed=False,
                    repair_observed=False,
                    confidence=0.74,
                    relevance_score=0.81,
                    description="Stored case memory example.",
                ),
            ),
            persistence_backend=case_backend,
        )

        assert domain_store.save_to_backend() is True
        assert case_store.save_to_backend() is True

        restored_domain_store = ApplicationDomainKnowledgeStore(persistence_backend=domain_backend)
        restored_case_store = ApplicationCaseMemoryStore(persistence_backend=case_backend)

        assert restored_domain_store.load_from_backend() is True
        assert restored_case_store.load_from_backend() is True
        assert restored_domain_store.records[0].title == "Career guidance"
        assert restored_case_store.records[0].problem_pattern == "structured-decision-overwhelm"
        assert restored_case_store.records[0].continuum_band_id is None


def test_final_wiring_prefers_real_application_stores_over_mock_fallbacks():
    domain_store = ApplicationDomainKnowledgeStore(
        records=(
            DomainKnowledgeRecord(
                record_id="knowledge:real-store:1",
                domain="career_decision",
                topic_tags=("career", "tradeoff"),
                jurisdiction_tags=("general",),
                source_type="reviewed-article",
                title="Real store career guidance",
                locator="real-store",
                summary="Persisted store says to compare trade-offs before choosing.",
                snippet="Compare trade-offs before choosing.",
                freshness_label="current",
                confidence=0.83,
                evidence_strength="high",
            ),
        )
    )
    case_store = ApplicationCaseMemoryStore(
        records=(
            CaseMemoryRecord(
                case_id="case:real-store:1",
                domain="structured_decision_patterns",
                problem_pattern="structured-decision-overwhelm",
                user_state_pattern="needs-structure",
                risk_markers=("risk-medium",),
                track_tags=("world",),
                regime_tags=("problem_solving",),
                intervention_ordering=("narrow_scope", "option_compare", "smallest_next_step"),
                outcome_label="improved",
                delayed_signal_count=4,
                escalation_observed=False,
                repair_observed=False,
                confidence=0.8,
                relevance_score=0.88,
                description="Persisted case memory from real application store.",
            ),
        )
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(case_memory=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="application-store-model",
                feature_surface=(
                    FeatureSignal(name="career_decision_signal", values=(0.62,), source="adapter"),
                    FeatureSignal(name="offer_tradeoff_signal", values=(0.71,), source="adapter"),
                ),
            ),
            domain_knowledge_store=domain_store,
            case_memory_store=case_store,
            session_id="application-store-session",
            wave_id="application-store-wave",
        )
    )

    domain_knowledge = result.active_snapshots["domain_knowledge"].value
    case_memory = result.active_snapshots["case_memory"].value

    assert domain_knowledge.hits[0].hit_id == "knowledge:real-store:1"
    assert domain_knowledge.hits[0].evidence_strength is EvidenceStrength.HIGH
    assert domain_knowledge.hits[0].citations[0].source_type is KnowledgeSourceType.REVIEWED_ARTICLE
    assert case_memory.hits[0].case_id == "case:real-store:1"
    assert case_memory.hits[0].problem_pattern == "structured-decision-overwhelm"
    assert case_memory.hits[0].continuum_location is not None


def test_final_wiring_uses_semantic_store_matching_without_keyword_overlap():
    domain_store = ApplicationDomainKnowledgeStore(
        records=(
            DomainKnowledgeRecord(
                record_id="knowledge:semantic-career:1",
                domain="career_decision",
                topic_tags=("career", "tradeoff"),
                jurisdiction_tags=("general",),
                source_type="reviewed-article",
                title="Career trade-off framing",
                locator="semantic-store",
                summary="Compare role trade-offs and choose the smallest next move.",
                snippet="Trade-off framing beats vague life-plan advice.",
                freshness_label="current",
                confidence=0.82,
                evidence_strength="high",
            ),
        )
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="semantic-store-model",
                feature_surface=(
                    FeatureSignal(name="semantic_task_pull", values=(0.78,), source="adapter"),
                    FeatureSignal(name="semantic_directive_pull", values=(0.61,), source="adapter"),
                ),
            ),
            domain_knowledge_store=domain_store,
            session_id="semantic-store-session",
            wave_id="semantic-store-wave",
        )
    )

    domain_knowledge = result.active_snapshots["domain_knowledge"].value
    assert domain_knowledge.hits[0].hit_id == "knowledge:semantic-career:1"
