from __future__ import annotations

import asyncio
import tempfile

from volvence_zero.application import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    ApplicationRareHeavyState,
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainExperienceEvaluationScenario,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
    KnowledgeReviewStatus,
    KnowledgeSourceKind,
    PlaybookRule,
    ReviewedKnowledgeCandidate,
    apply_domain_experience_packages,
    build_filesystem_persistence_backend,
    compile_domain_experience_package,
    validate_domain_experience_package,
)
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


def _support_package(*, package_id: str = "emotional_decision_support_v0") -> DomainExperiencePackage:
    knowledge = DomainKnowledgeRecord(
        record_id=f"knowledge:{package_id}:support-basics",
        domain="emotional_support_basics",
        topic_tags=("support", "decision", "stabilization"),
        jurisdiction_tags=("general",),
        source_type="internal-guide",
        title="Emotional decision support basics",
        locator=f"{package_id}:seed",
        summary="Stabilize emotional load before narrowing decision options.",
        snippet="Acknowledge the felt pressure, then move into a small next-step frame.",
        freshness_label="seed-current",
        confidence=0.82,
        evidence_strength="high",
    )
    case = CaseMemoryRecord(
        case_id=f"case:{package_id}:overwhelm-choice",
        domain="support_and_decision_patterns",
        problem_pattern="emotionally-loaded-decision",
        user_state_pattern="overwhelmed-but-seeking-clarity",
        risk_markers=("risk-medium",),
        track_tags=("self", "world"),
        regime_tags=("emotional_support", "guided_exploration"),
        intervention_ordering=("acknowledge_load", "separate_feeling_from_choice", "smallest_next_step"),
        outcome_label="stable",
        delayed_signal_count=2,
        escalation_observed=False,
        repair_observed=True,
        confidence=0.78,
        relevance_score=0.86,
        description="Seed case for emotional support before decision narrowing.",
    )
    playbook = PlaybookRule(
        rule_id=f"playbook:{package_id}:emotionally-loaded-decision",
        problem_pattern="emotionally-loaded-decision",
        recommended_regime="emotional_support",
        recommended_ordering=("acknowledge_load", "clarify_values", "smallest_next_step"),
        recommended_pacing="support-first",
        avoid_patterns=("premature-solutioning",),
        knowledge_weight_hint=0.38,
        experience_weight_hint=0.72,
        applicability_scope=("risk-medium", "emotional_support"),
        confidence=0.80,
        description="Prefer support-first pacing before decision structure.",
    )
    boundary = BoundaryPriorHint(
        hint_id=f"boundary:{package_id}:safety",
        regime_id=None,
        trigger_reasons=("citation-required", "risk-medium", "emotionally-loaded-decision"),
        answer_depth_limit_hint="support-first",
        clarification_required=True,
        refer_out_required=False,
        blocked_topics=("definitive-life-decision",),
        required_disclaimers=("support-not-diagnosis",),
        confidence=0.76,
        description="Keep emotional decision support bounded and clarify context.",
    )
    evaluation = DomainExperienceEvaluationScenario(
        scenario_id=f"eval:{package_id}:overwhelm-choice",
        domain="emotional_support_basics",
        prompt="I feel overwhelmed and need help deciding what to do next.",
        expected_capabilities=("support-first", "decision-clarification", "bounded-advice"),
        risk_markers=("risk-medium",),
        description="Checks support-first decision clarification.",
    )
    return DomainExperiencePackage(
        manifest=DomainExperienceManifest(
            package_id=package_id,
            version="0.1.0",
            display_name="Emotional decision support",
            domain_ids=("emotional_support_basics", "support_and_decision_patterns"),
            target_contexts=("chat", "decision-support"),
            evidence_level="seed-fixture",
            owner="tests",
            description="Generic emotional decision support seed package.",
        ),
        knowledge_records=(knowledge,),
        case_records=(case,),
        playbook_rules=(playbook,),
        boundary_hints=(boundary,),
        evaluation_scenarios=(evaluation,),
        domain_template_biases=(("emotional_support_basics", 0.66),),
    )


def test_domain_experience_validation_rejects_invalid_package() -> None:
    package = _support_package()
    invalid_knowledge = DomainKnowledgeRecord(
        record_id="knowledge:invalid",
        domain="emotional_support_basics",
        topic_tags=("support",),
        jurisdiction_tags=("general",),
        source_type="unsupported-source",
        title="Invalid source",
        locator="invalid",
        summary="Invalid.",
        snippet="Invalid.",
        freshness_label="invalid",
        confidence=0.8,
        evidence_strength="unsupported-strength",
    )
    rejected_candidate = ReviewedKnowledgeCandidate(
        candidate_id="candidate:rejected",
        source_kind=KnowledgeSourceKind.CONVERSATION,
        review_status=KnowledgeReviewStatus.REJECTED,
        record=package.knowledge_records[0],
        source_candidate_ids=("candidate:source",),
        review_note="Rejected source.",
        confidence=0.4,
    )
    invalid_case = CaseMemoryRecord(
        case_id=package.case_records[0].case_id,
        domain="support_and_decision_patterns",
        problem_pattern="emotionally-loaded-decision",
        user_state_pattern="overwhelmed",
        risk_markers=("risk-unknown",),
        track_tags=("invalid-track",),
        regime_tags=("emotional_support",),
        intervention_ordering=(),
        outcome_label="stable",
        delayed_signal_count=1,
        escalation_observed=False,
        repair_observed=False,
        confidence=1.2,
        relevance_score=0.8,
        description="Invalid case.",
    )
    invalid = DomainExperiencePackage(
        manifest=package.manifest,
        knowledge_records=(invalid_knowledge,),
        case_records=(package.case_records[0], invalid_case),
        playbook_rules=package.playbook_rules,
        boundary_hints=(),
        reviewed_knowledge_candidates=(rejected_candidate,),
    )

    report = validate_domain_experience_package(invalid)

    assert report.valid is False
    issue_text = " ".join(issue.description for issue in report.issues)
    assert "boundary hint" in issue_text
    assert "Duplicate value" in issue_text
    assert "Risk marker risk-unknown" in issue_text
    assert "Knowledge source type" in issue_text
    assert "Reviewed package knowledge must be approved" in issue_text


def test_domain_experience_compiles_and_applies_to_existing_application_owners() -> None:
    package = _support_package()
    compiled = compile_domain_experience_package(package)
    domain_store = ApplicationDomainKnowledgeStore(records=())
    case_store = ApplicationCaseMemoryStore()
    rare_state = ApplicationRareHeavyState()

    report = apply_domain_experience_packages(
        packages=(package,),
        domain_knowledge_store=domain_store,
        case_memory_store=case_store,
        application_rare_heavy_state=rare_state,
    )

    assert compiled.application_prior_update.domain_knowledge_updates[0].record.record_id == package.knowledge_records[0].record_id
    assert compiled.application_prior_update.case_memory_updates[0].record.case_id == package.case_records[0].case_id
    assert report.applied_knowledge_count == 1
    assert domain_store.records[0].record_id == package.knowledge_records[0].record_id
    assert case_store.records[0].case_id == package.case_records[0].case_id
    assert rare_state.distilled_playbook_rules[0].rule_id == package.playbook_rules[0].rule_id
    assert rare_state.boundary_prior_hints[0].hint_id == package.boundary_hints[0].hint_id


def test_domain_experience_application_can_persist_package_records() -> None:
    package = _support_package()
    with tempfile.TemporaryDirectory() as tmpdir:
        domain_backend = build_filesystem_persistence_backend(base_dir=f"{tmpdir}/domain")
        case_backend = build_filesystem_persistence_backend(base_dir=f"{tmpdir}/case")
        domain_store = ApplicationDomainKnowledgeStore(records=(), persistence_backend=domain_backend)
        case_store = ApplicationCaseMemoryStore(persistence_backend=case_backend)

        apply_domain_experience_packages(
            packages=(package,),
            domain_knowledge_store=domain_store,
            case_memory_store=case_store,
            application_rare_heavy_state=ApplicationRareHeavyState(),
            persist=True,
        )

        restored_domain_store = ApplicationDomainKnowledgeStore(records=(), persistence_backend=domain_backend)
        restored_case_store = ApplicationCaseMemoryStore(persistence_backend=case_backend)
        assert restored_domain_store.load_from_backend() is True
        assert restored_case_store.load_from_backend() is True
        assert restored_domain_store.records[0].record_id == package.knowledge_records[0].record_id
        assert restored_case_store.records[0].case_id == package.case_records[0].case_id


def test_domain_experience_final_wiring_uses_package_records() -> None:
    package = _support_package()

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(
                case_memory=WiringLevel.ACTIVE,
                strategy_playbook=WiringLevel.ACTIVE,
            ),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="domain-experience-model",
                feature_surface=(
                    FeatureSignal(name="support_signal", values=(0.86,), source="test"),
                    FeatureSignal(name="decision_signal", values=(0.72,), source="test"),
                ),
            ),
            user_input="I feel overwhelmed and need help deciding what to do next.",
            domain_experience_packages=(package,),
            session_id="domain-experience-session",
            wave_id="domain-experience-wave",
        )
    )

    knowledge = result.active_snapshots["domain_knowledge"].value
    case_memory = result.active_snapshots["case_memory"].value
    playbook = result.active_snapshots["strategy_playbook"].value
    boundary = result.active_snapshots["boundary_policy"].value

    assert any(hit.hit_id == package.knowledge_records[0].record_id for hit in knowledge.hits)
    assert any(hit.case_id == package.case_records[0].case_id for hit in case_memory.hits)
    assert any(rule.rule_id == package.playbook_rules[0].rule_id for rule in playbook.matched_rules)
    assert "boundary-prior-consumed" in boundary.trigger_reasons
