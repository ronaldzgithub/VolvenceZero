"""Recruiting-HR industry overlay (sourcing / screening / scheduling).

Pure data layered onto the generic org / twin base via
:func:`lifeform_domain_digital_employee.build_industry_package`. Industry
routing happens through ``applicability_scope`` tags
(``industry:recruiting-hr``), never through keyword matching.
"""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_digital_employee.industry import IndustryProfile

_INDUSTRY_ID = "recruiting-hr"

_DOMAIN_SCREENING = "hr_candidate_screening"
_DOMAIN_COORDINATION = "hr_interview_coordination"


def build_recruiting_hr_profile() -> IndustryProfile:
    """Return the reviewed recruiting-HR industry overlay."""

    return IndustryProfile(
        industry_id=_INDUSTRY_ID,
        display_name="Recruiting HR",
        description=(
            "Recruiting assistant: criteria-confirmed JD parsing and resume "
            "screening, interview scheduling, candidate communication drafts, "
            "with hiring decisions and compensation always human-owned and a "
            "hard refusal on discriminatory screening."
        ),
        domain_ids=(_DOMAIN_SCREENING, _DOMAIN_COORDINATION),
        target_contexts=("recruiting-pipeline",),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-de-ind-hr:confirm-criteria-before-screening",
                domain=_DOMAIN_SCREENING,
                topic_tags=("screening", "criteria"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Confirm screening criteria with the hiring manager first",
                locator="lifeform-digital-employee-ind-hr:design-note:1",
                summary=(
                    "Resume screening starts only after the hiring manager "
                    "confirms the criteria: hard requirements, nice-to-haves "
                    "and red lines extracted from the JD. Screening against "
                    "guessed criteria silently rejects good candidates and "
                    "cannot be audited."
                ),
                snippet="No screening before the criteria are confirmed.",
                freshness_label="canonical",
                confidence=0.88,
                evidence_strength="high",
            ),
            DomainKnowledgeRecord(
                record_id="rid-de-ind-hr:every-verdict-carries-reasons",
                domain=_DOMAIN_SCREENING,
                topic_tags=("auditability", "fairness"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Every screening verdict records its job-related reasons",
                locator="lifeform-digital-employee-ind-hr:design-note:2",
                summary=(
                    "Each pass / hold / reject verdict records which confirmed "
                    "job-related criterion drove it. Uncertain resumes are "
                    "marked for human review, never silently dropped; a "
                    "verdict without a recorded reason is not a verdict."
                ),
                snippet="A verdict without a job-related reason is invalid.",
                freshness_label="canonical",
                confidence=0.9,
                evidence_strength="high",
            ),
        ),
        case_records=(
            CaseMemoryRecord(
                case_id="rid-de-ind-hr:case:batch-screen-to-shortlist",
                domain=_DOMAIN_SCREENING,
                problem_pattern="resume-batch-intake",
                user_state_pattern="requisition-open",
                risk_markers=("risk-medium",),
                track_tags=("world",),
                regime_tags=("task_execution",),
                intervention_ordering=(
                    "confirm_criteria_with_hiring_manager",
                    "screen_with_recorded_reasons",
                    "flag_uncertain_for_human_review",
                    "draft_candidate_communications",
                    "queue_shortlist_for_human_decision",
                ),
                outcome_label="improved",
                delayed_signal_count=1,
                escalation_observed=False,
                repair_observed=False,
                confidence=0.8,
                relevance_score=0.84,
                description=(
                    "New resume batch. Confirming criteria first and recording "
                    "a job-related reason per verdict produced an auditable "
                    "shortlist; uncertain resumes went to human review and the "
                    "final decision stayed with the hiring manager."
                ),
            ),
        ),
        playbook_rules=(
            PlaybookRule(
                rule_id="rid-de-ind-hr:playbook:criteria-screen-schedule-decide",
                problem_pattern="resume-batch-intake",
                recommended_regime="task_execution",
                recommended_ordering=(
                    "confirm_criteria_with_hiring_manager",
                    "screen_with_recorded_reasons",
                    "flag_uncertain_for_human_review",
                    "propose_interview_slots",
                    "queue_shortlist_for_human_decision",
                ),
                recommended_pacing="criteria-first",
                avoid_patterns=(
                    "screen-on-unconfirmed-criteria",
                    "silent-rejection-without-reason",
                ),
                knowledge_weight_hint=0.5,
                experience_weight_hint=0.6,
                applicability_scope=("industry:recruiting-hr", "risk-medium"),
                confidence=0.82,
                description=(
                    "Confirm criteria, screen with recorded reasons, propose "
                    "interview slots; hire / reject decisions queue for the "
                    "hiring manager."
                ),
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-de-ind-hr:boundary:no-discriminatory-screening",
                regime_id="escalation_to_human",
                trigger_reasons=("screening-criteria-includes-protected-attribute",),
                answer_depth_limit_hint="refuse-and-restate-job-related-criteria",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=(
                    "protected-attribute-screening",
                    "discriminatory-filter-request",
                ),
                required_disclaimers=("job-related-criteria-only",),
                confidence=0.9,
                description=(
                    "Screening must use job-related criteria only. A request "
                    "to filter on protected attributes (gender, age, origin, "
                    "marital status, …) is refused and escalated to a human "
                    "with the policy stated, never quietly applied."
                ),
            ),
            BoundaryPriorHint(
                hint_id="rid-de-ind-hr:boundary:offer-and-comp-need-human",
                regime_id="escalation_to_human",
                trigger_reasons=("hiring-decision-or-compensation",),
                answer_depth_limit_hint="recommendation-only",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=(
                    "auto-send-offer-or-rejection",
                    "compensation-commitment",
                ),
                required_disclaimers=("human-approval-required",),
                confidence=0.88,
                description=(
                    "Offers, rejections and any compensation number are human "
                    "decisions: the assistant prepares the shortlist and "
                    "drafts, a human decides and sends."
                ),
            ),
        ),
    )


__all__ = ["build_recruiting_hr_profile"]
