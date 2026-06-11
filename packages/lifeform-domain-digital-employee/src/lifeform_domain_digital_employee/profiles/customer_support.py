"""Customer-support industry overlay (ticket triage / resolution).

Pure data layered onto the generic org / twin base via
:func:`lifeform_domain_digital_employee.build_industry_package`. Industry
routing happens through ``applicability_scope`` tags
(``industry:customer-support``), never through keyword matching.
"""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_digital_employee.industry import IndustryProfile

_INDUSTRY_ID = "customer-support"

_DOMAIN_TRIAGE = "support_ticket_triage"
_DOMAIN_RESOLUTION = "support_resolution"


def build_customer_support_profile() -> IndustryProfile:
    """Return the reviewed customer-support industry overlay."""

    return IndustryProfile(
        industry_id=_INDUSTRY_ID,
        display_name="Customer Support",
        description=(
            "Ticket triage and resolution assistant: acknowledge-first "
            "responses, policy-bounded promises, and refund / credit actions "
            "escalated to a human."
        ),
        domain_ids=(_DOMAIN_TRIAGE, _DOMAIN_RESOLUTION),
        target_contexts=("customer-support",),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-de-ind-cs:acknowledge-before-resolving",
                domain=_DOMAIN_RESOLUTION,
                topic_tags=("tone", "acknowledgement"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Acknowledge the customer's situation before fixing it",
                locator="lifeform-digital-employee-ind-cs:design-note:1",
                summary=(
                    "A correct answer delivered coldly still reads as a bad "
                    "support experience. Acknowledge what the customer is "
                    "dealing with, then resolve; the order matters."
                ),
                snippet="Acknowledge first, then resolve.",
                freshness_label="canonical",
                confidence=0.86,
                evidence_strength="medium",
            ),
            DomainKnowledgeRecord(
                record_id="rid-de-ind-cs:promises-bounded-by-policy",
                domain=_DOMAIN_RESOLUTION,
                topic_tags=("policy", "promises"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Never promise beyond documented support policy",
                locator="lifeform-digital-employee-ind-cs:design-note:2",
                summary=(
                    "Refund amounts, SLA commitments and exceptions are bounded "
                    "by documented policy. Promising beyond policy to defuse an "
                    "angry customer creates a worse second conversation."
                ),
                snippet="Policy bounds every promise; exceptions go to a human.",
                freshness_label="canonical",
                confidence=0.9,
                evidence_strength="high",
            ),
        ),
        case_records=(
            CaseMemoryRecord(
                case_id="rid-de-ind-cs:case:angry-refund-request",
                domain=_DOMAIN_RESOLUTION,
                problem_pattern="refund-or-credit-request",
                user_state_pattern="frustrated-customer",
                risk_markers=("risk-high",),
                track_tags=("world",),
                regime_tags=("escalation_to_human",),
                intervention_ordering=(
                    "acknowledge_frustration",
                    "verify_policy_coverage",
                    "prepare_refund_recommendation",
                    "hand_to_human_for_approval",
                ),
                outcome_label="stable",
                delayed_signal_count=0,
                escalation_observed=True,
                repair_observed=False,
                confidence=0.84,
                relevance_score=0.86,
                description=(
                    "Frustrated customer demanded a refund. Acknowledging first, "
                    "verifying policy and handing the actual credit action to a "
                    "human kept both the customer and the ledger intact."
                ),
            ),
        ),
        playbook_rules=(
            PlaybookRule(
                rule_id="rid-de-ind-cs:playbook:triage-acknowledge-resolve",
                problem_pattern="inbound-support-ticket",
                recommended_regime="task_execution",
                recommended_ordering=(
                    "classify_severity_and_topic",
                    "acknowledge_customer_situation",
                    "check_known_issues",
                    "respond_with_concrete_steps",
                    "schedule_follow_up",
                ),
                recommended_pacing="acknowledge-first",
                avoid_patterns=("cold-canned-reply", "promise-beyond-policy"),
                knowledge_weight_hint=0.5,
                experience_weight_hint=0.6,
                applicability_scope=("industry:customer-support", "risk-low"),
                confidence=0.82,
                description=(
                    "Triage, acknowledge, then resolve with concrete steps and "
                    "a follow-up."
                ),
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-de-ind-cs:boundary:refunds-need-human",
                regime_id="escalation_to_human",
                trigger_reasons=("refund-or-credit-request",),
                answer_depth_limit_hint="recommendation-only",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=("auto-issue-refund", "policy-exception-grant"),
                required_disclaimers=("human-approval-required",),
                confidence=0.88,
                description=(
                    "Refunds, credits and policy exceptions spend company money: "
                    "recommend only, a human approves the action."
                ),
            ),
        ),
    )


__all__ = ["build_customer_support_profile"]
