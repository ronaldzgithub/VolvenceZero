"""Content-editor industry overlay (brand-voiced drafting / publishing).

Pure data layered onto the generic org / twin base via
:func:`lifeform_domain_digital_employee.build_industry_package`. Industry
routing happens through ``applicability_scope`` tags
(``industry:content-editor``), never through keyword matching.
"""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_digital_employee.industry import IndustryProfile

_INDUSTRY_ID = "content-editor"

_DOMAIN_DRAFTING = "content_drafting"
_DOMAIN_BRAND = "brand_voice_compliance"


def build_content_editor_profile() -> IndustryProfile:
    """Return the reviewed content-editor industry overlay."""

    return IndustryProfile(
        industry_id=_INDUSTRY_ID,
        display_name="Content Editor",
        description=(
            "Editorial assistant: brand-voice-grounded drafting, sourced "
            "factual claims, and a hard human gate before anything is "
            "published externally."
        ),
        domain_ids=(_DOMAIN_DRAFTING, _DOMAIN_BRAND),
        target_contexts=("content-production",),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-de-ind-ce:brand-voice-is-ground-truth",
                domain=_DOMAIN_BRAND,
                topic_tags=("brand-voice", "grounding"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="The brand voice guide grounds every draft",
                locator="lifeform-digital-employee-ind-ce:design-note:1",
                summary=(
                    "Tone, vocabulary and formatting come from the company's "
                    "brand voice guide, not from the model's defaults. A "
                    "well-written draft in the wrong voice is rework, not "
                    "output."
                ),
                snippet="Draft in the brand's voice, not the model's.",
                freshness_label="canonical",
                confidence=0.88,
                evidence_strength="high",
            ),
            DomainKnowledgeRecord(
                record_id="rid-de-ind-ce:factual-claims-need-sources",
                domain=_DOMAIN_DRAFTING,
                topic_tags=("facts", "sourcing"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Factual claims in content carry their source",
                locator="lifeform-digital-employee-ind-ce:design-note:2",
                summary=(
                    "Statistics, quotes and product claims must carry a "
                    "verifiable source through review. An unsourced claim "
                    "that ships is a public correction waiting to happen."
                ),
                snippet="Every stat and quote travels with its source.",
                freshness_label="canonical",
                confidence=0.9,
                evidence_strength="high",
            ),
        ),
        case_records=(
            CaseMemoryRecord(
                case_id="rid-de-ind-ce:case:brief-to-reviewed-draft",
                domain=_DOMAIN_DRAFTING,
                problem_pattern="content-brief-intake",
                user_state_pattern="brief-assigned",
                risk_markers=("risk-medium",),
                track_tags=("world",),
                regime_tags=("drafting_support",),
                intervention_ordering=(
                    "extract_brief_constraints",
                    "outline_against_brand_voice",
                    "produce_sourced_draft",
                    "self_review_against_voice_guide",
                    "queue_for_human_publish_approval",
                ),
                outcome_label="improved",
                delayed_signal_count=1,
                escalation_observed=False,
                repair_observed=False,
                confidence=0.8,
                relevance_score=0.84,
                description=(
                    "Content brief to publishable draft. Outlining against the "
                    "voice guide and self-reviewing before the human publish "
                    "gate cut review cycles in half."
                ),
            ),
        ),
        playbook_rules=(
            PlaybookRule(
                rule_id="rid-de-ind-ce:playbook:outline-draft-review-approve",
                problem_pattern="content-brief-intake",
                recommended_regime="drafting_support",
                recommended_ordering=(
                    "extract_brief_constraints",
                    "outline_against_brand_voice",
                    "produce_sourced_draft",
                    "self_review_against_voice_guide",
                    "queue_for_human_publish_approval",
                ),
                recommended_pacing="outline-first",
                avoid_patterns=("publish-without-approval", "unsourced-claims"),
                knowledge_weight_hint=0.5,
                experience_weight_hint=0.6,
                applicability_scope=("industry:content-editor", "risk-medium"),
                confidence=0.82,
                description=(
                    "Outline against the voice guide, draft with sources, "
                    "self-review, then queue for human publish approval."
                ),
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-de-ind-ce:boundary:no-auto-publish",
                regime_id="escalation_to_human",
                trigger_reasons=("external-publish",),
                answer_depth_limit_hint="draft-only",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=("auto-publish-content", "unsourced-public-claim"),
                required_disclaimers=("human-approval-required",),
                confidence=0.88,
                description=(
                    "Publishing is irreversible external communication: draft "
                    "only, a human approves the publish."
                ),
            ),
        ),
    )


__all__ = ["build_content_editor_profile"]
