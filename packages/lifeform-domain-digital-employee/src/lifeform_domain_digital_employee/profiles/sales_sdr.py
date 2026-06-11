"""Sales-SDR industry overlay (outbound prospecting digital employee).

Pure data: extra knowledge / case / playbook / boundary records layered
onto the generic org / twin base by
:func:`lifeform_domain_digital_employee.build_industry_package`. Industry
routing happens through ``applicability_scope`` tags (``industry:sales-sdr``)
consumed by the existing playbook / boundary owners — never through
matching keywords in user text.
"""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_digital_employee.industry import IndustryProfile

_INDUSTRY_ID = "sales-sdr"

_DOMAIN_PROSPECTING = "sdr_prospecting"
_DOMAIN_OUTREACH = "sdr_outreach_quality"


def build_sales_sdr_profile() -> IndustryProfile:
    """Return the reviewed sales-SDR industry overlay."""

    return IndustryProfile(
        industry_id=_INDUSTRY_ID,
        display_name="Sales SDR",
        description=(
            "Outbound prospecting assistant: research-first lead handling, "
            "personalised outreach drafts, and a hard human gate before any "
            "external send."
        ),
        domain_ids=(_DOMAIN_PROSPECTING, _DOMAIN_OUTREACH),
        target_contexts=("sales-outbound",),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-de-ind-sdr:research-before-outreach",
                domain=_DOMAIN_PROSPECTING,
                topic_tags=("research", "personalisation"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Research the prospect before drafting outreach",
                locator="lifeform-digital-employee-ind-sdr:design-note:1",
                summary=(
                    "A personalised message grounded in the prospect's actual "
                    "context outperforms volume. Research company, role and "
                    "recent signals before drafting; a generic blast burns the "
                    "domain's reputation."
                ),
                snippet="Research first; one grounded message beats ten generic ones.",
                freshness_label="canonical",
                confidence=0.86,
                evidence_strength="medium",
            ),
            DomainKnowledgeRecord(
                record_id="rid-de-ind-sdr:claims-must-be-sourced",
                domain=_DOMAIN_OUTREACH,
                topic_tags=("claims", "compliance"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Product claims in outreach must trace to approved material",
                locator="lifeform-digital-employee-ind-sdr:design-note:2",
                summary=(
                    "Every product or pricing claim in an outbound message must "
                    "trace to approved sales material. Inventing a capability "
                    "to win a reply is a compliance failure, not a sales win."
                ),
                snippet="No claim without an approved source behind it.",
                freshness_label="canonical",
                confidence=0.9,
                evidence_strength="high",
            ),
        ),
        case_records=(
            CaseMemoryRecord(
                case_id="rid-de-ind-sdr:case:qualify-then-draft",
                domain=_DOMAIN_PROSPECTING,
                problem_pattern="new-lead-intake",
                user_state_pattern="lead-list-assigned",
                risk_markers=("risk-low",),
                track_tags=("world",),
                regime_tags=("task_execution",),
                intervention_ordering=(
                    "research_prospect_context",
                    "qualify_against_icp",
                    "draft_personalised_outreach",
                    "queue_for_human_send_approval",
                ),
                outcome_label="improved",
                delayed_signal_count=1,
                escalation_observed=False,
                repair_observed=False,
                confidence=0.8,
                relevance_score=0.84,
                description=(
                    "New lead batch. Researching and qualifying against the ICP "
                    "before drafting kept reply rates up and spam complaints at "
                    "zero; every send went through the human approval queue."
                ),
            ),
        ),
        playbook_rules=(
            PlaybookRule(
                rule_id="rid-de-ind-sdr:playbook:research-qualify-draft-approve",
                problem_pattern="new-lead-intake",
                recommended_regime="task_execution",
                recommended_ordering=(
                    "research_prospect_context",
                    "qualify_against_icp",
                    "draft_personalised_outreach",
                    "queue_for_human_send_approval",
                ),
                recommended_pacing="research-first",
                avoid_patterns=("generic-blast", "send-without-approval"),
                knowledge_weight_hint=0.5,
                experience_weight_hint=0.6,
                applicability_scope=("industry:sales-sdr", "risk-medium"),
                confidence=0.82,
                description=(
                    "Research, qualify and draft; external sends always queue "
                    "for human approval."
                ),
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-de-ind-sdr:boundary:no-auto-send",
                regime_id="escalation_to_human",
                trigger_reasons=("external-outreach-send",),
                answer_depth_limit_hint="draft-only",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=("auto-send-outreach", "fabricated-product-claim"),
                required_disclaimers=("human-approval-required",),
                confidence=0.88,
                description=(
                    "Outbound messages to prospects are external publications: "
                    "draft only, human approves the send."
                ),
            ),
        ),
    )


__all__ = ["build_sales_sdr_profile"]
