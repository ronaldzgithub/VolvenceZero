"""Admin-assistant industry overlay (scheduling / minutes / travel process).

Pure data layered onto the generic org / twin base via
:func:`lifeform_domain_digital_employee.build_industry_package`. Industry
routing happens through ``applicability_scope`` tags
(``industry:admin-assistant``), never through keyword matching.

Manifesto red line: this role assists with *process* (collecting
constraints, preparing checklists, drafting notices) and never renders a
financial judgement. Expense / reimbursement questions beyond process
reminders are refused with a referral — see the boundary hint below,
which complements (and never replaces) the base finance-tax refusal.
"""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_digital_employee.industry import IndustryProfile

_INDUSTRY_ID = "admin-assistant"

_DOMAIN_COORDINATION = "admin_coordination"
_DOMAIN_PROCESS = "admin_process_support"


def build_admin_assistant_profile() -> IndustryProfile:
    """Return the reviewed admin-assistant industry overlay."""

    return IndustryProfile(
        industry_id=_INDUSTRY_ID,
        display_name="Admin Assistant",
        description=(
            "Administrative assistant: constraint-first meeting scheduling, "
            "decision-and-owner meeting minutes, travel and reimbursement "
            "process support strictly limited to process reminders — any "
            "financial judgement is refused and referred to the finance "
            "owner or a licensed professional."
        ),
        domain_ids=(_DOMAIN_COORDINATION, _DOMAIN_PROCESS),
        target_contexts=("office-administration",),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-de-ind-aa:collect-constraints-before-scheduling",
                domain=_DOMAIN_COORDINATION,
                topic_tags=("scheduling", "coordination"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Collect constraints before proposing meeting slots",
                locator="lifeform-digital-employee-ind-aa:design-note:1",
                summary=(
                    "Scheduling starts from constraints: required attendees' "
                    "availability, time zones, room or link availability. "
                    "Propose two or three candidate slots for the organiser "
                    "to confirm, then send one consolidated invite — invite "
                    "churn is a coordination failure."
                ),
                snippet="Constraints first, candidate slots second, one invite.",
                freshness_label="canonical",
                confidence=0.86,
                evidence_strength="medium",
            ),
            DomainKnowledgeRecord(
                record_id="rid-de-ind-aa:minutes-capture-decisions-and-owners",
                domain=_DOMAIN_PROCESS,
                topic_tags=("minutes", "accountability"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Minutes capture decisions, owners and deadlines",
                locator="lifeform-digital-employee-ind-aa:design-note:2",
                summary=(
                    "Meeting minutes are an accountability artefact: each "
                    "decision is recorded with its owner and deadline, open "
                    "questions are listed separately, and the draft goes to "
                    "the chair for confirmation before distribution."
                ),
                snippet="A decision without an owner and a deadline is a wish.",
                freshness_label="canonical",
                confidence=0.86,
                evidence_strength="medium",
            ),
        ),
        case_records=(
            CaseMemoryRecord(
                case_id="rid-de-ind-aa:case:travel-request-to-approval-packet",
                domain=_DOMAIN_PROCESS,
                problem_pattern="travel-arrangement-request",
                user_state_pattern="trip-dates-confirmed",
                risk_markers=("risk-medium",),
                track_tags=("world",),
                regime_tags=("task_execution",),
                intervention_ordering=(
                    "confirm_dates_and_travel_policy_band",
                    "collect_compliant_flight_and_hotel_options",
                    "prepare_cost_comparison_packet",
                    "queue_booking_and_payment_for_human_approval",
                ),
                outcome_label="improved",
                delayed_signal_count=1,
                escalation_observed=False,
                repair_observed=False,
                confidence=0.8,
                relevance_score=0.84,
                description=(
                    "Travel request. Confirming dates and the policy band "
                    "first, then preparing a compliant options packet with a "
                    "cost comparison, let the approver decide in one pass; "
                    "booking and payment stayed with the authorised human."
                ),
            ),
        ),
        playbook_rules=(
            PlaybookRule(
                rule_id="rid-de-ind-aa:playbook:constraints-options-packet-approve",
                problem_pattern="travel-arrangement-request",
                recommended_regime="task_execution",
                recommended_ordering=(
                    "confirm_dates_and_travel_policy_band",
                    "collect_compliant_flight_and_hotel_options",
                    "prepare_cost_comparison_packet",
                    "queue_booking_and_payment_for_human_approval",
                ),
                recommended_pacing="constraints-first",
                avoid_patterns=(
                    "book-or-pay-without-approval",
                    "judge-reimbursement-eligibility",
                ),
                knowledge_weight_hint=0.5,
                experience_weight_hint=0.6,
                applicability_scope=("industry:admin-assistant", "risk-medium"),
                confidence=0.82,
                description=(
                    "Confirm constraints, assemble compliant options into an "
                    "approval packet; booking, payment and any eligibility "
                    "judgement stay with the human."
                ),
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-de-ind-aa:boundary:process-help-not-finance-judgement",
                regime_id="escalation_to_human",
                trigger_reasons=("reimbursement-eligibility-judgement",),
                answer_depth_limit_hint="process-checklist-only",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=(
                    "reimbursement-approval-decision",
                    "expense-deductibility-advice",
                    "tax-treatment-advice",
                ),
                required_disclaimers=(
                    "process-reminder-only",
                    "refer-to-licensed-professional",
                ),
                confidence=0.9,
                description=(
                    "Reimbursement help is process support only: required "
                    "receipts, form fields, submission deadlines. Whether an "
                    "expense is reimbursable, deductible or tax-favourable is "
                    "refused and referred to the finance owner or a licensed "
                    "professional — the assistant never renders the judgement."
                ),
            ),
        ),
    )


__all__ = ["build_admin_assistant_profile"]
