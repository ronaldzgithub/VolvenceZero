from __future__ import annotations

import pytest

import volvence_zero
from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
    PlaybookRule,
)
from volvence_zero.brain import Brain, BrainConfig


def _package() -> DomainExperiencePackage:
    return DomainExperiencePackage(
        manifest=DomainExperienceManifest(
            package_id="core-boundary-support-v0",
            version="0.1.0",
            display_name="Core boundary support package",
            domain_ids=("emotional_support_basics", "support_and_decision_patterns"),
            target_contexts=("chat",),
            evidence_level="test-fixture",
            owner="tests",
            description="Package fixture for stable Brain API tests.",
        ),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="knowledge:core-boundary:support",
                domain="emotional_support_basics",
                topic_tags=("support", "stabilization"),
                jurisdiction_tags=("general",),
                source_type="internal-guide",
                title="Support-first decision guidance",
                locator="core-boundary-fixture",
                summary="Stabilize emotional pressure before narrowing options.",
                snippet="Acknowledge pressure, then clarify one next step.",
                freshness_label="test-current",
                confidence=0.84,
                evidence_strength="high",
            ),
        ),
        case_records=(
            CaseMemoryRecord(
                case_id="case:core-boundary:support-decision",
                domain="support_and_decision_patterns",
                problem_pattern="support-before-decision",
                user_state_pattern="emotionally-overloaded",
                risk_markers=("risk-medium",),
                track_tags=("self", "world"),
                regime_tags=("emotional_support",),
                intervention_ordering=("acknowledge_pressure", "clarify_values", "smallest_next_step"),
                outcome_label="stable",
                delayed_signal_count=2,
                escalation_observed=False,
                repair_observed=True,
                confidence=0.79,
                relevance_score=0.87,
                description="Core boundary package case seed.",
            ),
        ),
        playbook_rules=(
            PlaybookRule(
                rule_id="playbook:core-boundary:support-decision",
                problem_pattern="support-before-decision",
                recommended_regime="emotional_support",
                recommended_ordering=("acknowledge_pressure", "clarify_values", "smallest_next_step"),
                recommended_pacing="support-first",
                avoid_patterns=("premature-solutioning",),
                knowledge_weight_hint=0.42,
                experience_weight_hint=0.70,
                applicability_scope=("risk-medium", "emotional_support"),
                confidence=0.80,
                description="Support before narrowing decisions.",
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="boundary:core-boundary:support",
                regime_id=None,
                trigger_reasons=("citation-required", "risk-medium"),
                answer_depth_limit_hint="support-first",
                clarification_required=True,
                refer_out_required=False,
                blocked_topics=("definitive-life-decision",),
                required_disclaimers=("support-not-diagnosis",),
                confidence=0.76,
                description="Keep support package guidance bounded.",
            ),
        ),
    )


def test_core_import_exposes_narrow_brain_api_without_model_weights() -> None:
    assert volvence_zero.Brain is Brain
    assert volvence_zero.BrainConfig is BrainConfig


def test_brain_default_uses_synthetic_substrate_and_runs_turn() -> None:
    brain = Brain(BrainConfig(rare_heavy_enabled=False))
    session = brain.create_session(session_id="core-package-session")

    result = session.run_turn("I feel stuck and need help deciding.")

    assert result.response.text
    assert result.active_snapshots["substrate"].value.description
    assert session.runner._default_residual_runtime.runtime_origin == "synthetic-open-weight"


def test_brain_loads_domain_experience_package_through_stable_api() -> None:
    package = _package()
    brain = Brain(
        BrainConfig(
            domain_experience_packages=(package,),
            rare_heavy_enabled=False,
        )
    )
    session = brain.create_session(session_id="core-package-domain-session")

    result = session.run_turn("I am overwhelmed and need help with a decision.")

    case_memory = result.active_snapshots["case_memory"].value
    assert any(
        record.record_id == "knowledge:core-boundary:support"
        for record in session.runner._domain_knowledge_store.records
    )
    assert any(hit.case_id == "case:core-boundary:support-decision" for hit in case_memory.hits)


def test_brain_injected_mode_requires_runtime() -> None:
    brain = Brain(BrainConfig(substrate_mode="injected"))

    with pytest.raises(ValueError, match="requires substrate_runtime"):
        brain.create_session()
