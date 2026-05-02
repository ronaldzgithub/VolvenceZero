from __future__ import annotations

import json

import pytest

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
    PlaybookRule,
)
from volvence_zero.brain import Brain, BrainConfig
from volvence_zero.semantic_state import (
    SEMANTIC_OWNER_SLOTS,
    load_semantic_json_schema,
    load_semantic_prompt_template,
)


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
    # Top-level `import volvence_zero` is no longer a regular package — it is a
    # PEP 420 namespace package contributed to by every vz-* wheel. Stable API
    # is the explicit submodule path; convenience top-level re-exports are
    # intentionally NOT provided so that `pip install vz-contracts` alone does
    # not fail with `AttributeError: module has no attribute 'Brain'`.
    from volvence_zero.brain import Brain as ImportedBrain, BrainConfig as ImportedBrainConfig

    assert ImportedBrain is Brain
    assert ImportedBrainConfig is BrainConfig


def test_core_package_includes_semantic_state_runtime_resources() -> None:
    prompt = load_semantic_prompt_template()
    schema = json.loads(load_semantic_json_schema())

    assert "typed semantic-state proposals" in prompt
    assert schema["required"] == ["proposals", "runtime_id", "schema_version", "description"]
    assert schema["properties"]["proposals"]["items"]["properties"]["target_slot"]["type"] == "string"
    assert SEMANTIC_OWNER_SLOTS == (
        "plan_intent",
        "commitment",
        "open_loop",
        "user_model",
        "execution_result",
        "belief_assumption",
        "relationship_state",
        "goal_value",
        "boundary_consent",
    )


def test_brain_default_uses_synthetic_substrate_and_runs_turn() -> None:
    brain = Brain(BrainConfig(rare_heavy_enabled=False))
    session = brain.create_session(session_id="core-package-session")

    result = session.run_turn("I feel stuck and need help deciding.")

    assert result.response.text
    assert result.active_snapshots["substrate"].value.description
    assert session.runner._default_residual_runtime.runtime_origin == "synthetic-open-weight"


def test_brain_session_accepts_external_semantic_events() -> None:
    brain = Brain(BrainConfig(rare_heavy_enabled=False))
    session = brain.create_session(session_id="core-package-semantic-adapter")

    queued = session.submit_tool_result(
        event_id="tool:core:1",
        tool_name="local-tool",
        action_id="action:core",
        status="succeeded",
        summary="Local tool completed",
        detail="The local tool produced an artifact.",
        artifact_refs=("artifact:core",),
        plan_ref="core-package-plan",
    )
    pending_event = session.runner._pending_semantic_events[0]
    result = session.run_turn("Continue from the external tool result.")

    assert queued == ("tool:core:1",)
    assert "environment_outcome:tool:core:1:outcome" in pending_event.artifact_refs
    assert result.active_snapshots["execution_result"].value.completed_actions
    assert result.active_snapshots["plan_intent"].value.plan_revision_count >= 1


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
