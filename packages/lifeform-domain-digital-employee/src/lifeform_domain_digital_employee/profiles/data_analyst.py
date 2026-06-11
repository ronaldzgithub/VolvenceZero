"""Data-analyst industry overlay (metric definitions / review / reporting).

Pure data layered onto the generic org / twin base via
:func:`lifeform_domain_digital_employee.build_industry_package`. Industry
routing happens through ``applicability_scope`` tags
(``industry:data-analyst``), never through keyword matching.
"""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_digital_employee.industry import IndustryProfile

_INDUSTRY_ID = "data-analyst"

_DOMAIN_METRICS = "analytics_metric_definitions"
_DOMAIN_DELIVERY = "analytics_delivery_quality"


def build_data_analyst_profile() -> IndustryProfile:
    """Return the reviewed data-analyst industry overlay."""

    return IndustryProfile(
        industry_id=_INDUSTRY_ID,
        display_name="Data Analyst",
        description=(
            "Analytics assistant: metric definitions confirmed before any "
            "query, sampled verification before delivery, charts kept "
            "separate from conclusions, and external data disclosure always "
            "behind a human gate."
        ),
        domain_ids=(_DOMAIN_METRICS, _DOMAIN_DELIVERY),
        target_contexts=("business-analytics",),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-de-ind-da:confirm-metric-definition-first",
                domain=_DOMAIN_METRICS,
                topic_tags=("metrics", "definitions"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Confirm the metric definition before writing the query",
                locator="lifeform-digital-employee-ind-da:design-note:1",
                summary=(
                    "Refunds in or out, order time vs payment time, time "
                    "zone, deduplication rule — each is confirmed with the "
                    "requester before a query is written. Two analyses of "
                    "'the same metric' under different definitions are the "
                    "root of most cross-team number disputes."
                ),
                snippet="No query before the definition is confirmed.",
                freshness_label="canonical",
                confidence=0.9,
                evidence_strength="high",
            ),
            DomainKnowledgeRecord(
                record_id="rid-de-ind-da:charts-separate-from-conclusions",
                domain=_DOMAIN_DELIVERY,
                topic_tags=("reporting", "integrity"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Charts show data; conclusions are stated separately",
                locator="lifeform-digital-employee-ind-da:design-note:2",
                summary=(
                    "A deliverable separates what the data shows (chart, "
                    "definition note, data source, cut-off time) from what "
                    "the analyst concludes (stated with confidence and "
                    "caveats). Correlation is not presented as causation; "
                    "gaps are flagged, never estimated away."
                ),
                snippet="Chart on one line, conclusion on another, never fused.",
                freshness_label="canonical",
                confidence=0.88,
                evidence_strength="high",
            ),
        ),
        case_records=(
            CaseMemoryRecord(
                case_id="rid-de-ind-da:case:metric-anomaly-triage",
                domain=_DOMAIN_DELIVERY,
                problem_pattern="metric-anomaly-detected",
                user_state_pattern="dashboard-spike-reported",
                risk_markers=("risk-medium",),
                track_tags=("world",),
                regime_tags=("task_execution",),
                intervention_ordering=(
                    "rule_out_pipeline_delay",
                    "rule_out_definition_change",
                    "rule_out_missing_dimension",
                    "report_business_anomaly_with_evidence",
                ),
                outcome_label="improved",
                delayed_signal_count=1,
                escalation_observed=False,
                repair_observed=False,
                confidence=0.82,
                relevance_score=0.84,
                description=(
                    "Dashboard spike reported. Ruling out the three technical "
                    "causes (pipeline delay, definition change, missing "
                    "dimension) before reporting a business anomaly avoided a "
                    "false alarm and produced a reusable triage record."
                ),
            ),
        ),
        playbook_rules=(
            PlaybookRule(
                rule_id="rid-de-ind-da:playbook:define-query-verify-deliver",
                problem_pattern="data-request-intake",
                recommended_regime="task_execution",
                recommended_ordering=(
                    "confirm_metric_definition_with_requester",
                    "confirm_time_range_and_filters",
                    "run_query_and_sample_verify",
                    "deliver_with_definition_note",
                ),
                recommended_pacing="definition-first",
                avoid_patterns=(
                    "query-on-unconfirmed-definition",
                    "estimate-missing-data",
                    "fuse-chart-with-conclusion",
                ),
                knowledge_weight_hint=0.5,
                experience_weight_hint=0.6,
                applicability_scope=("industry:data-analyst", "risk-medium"),
                confidence=0.82,
                description=(
                    "Confirm the definition and the range, verify by "
                    "sampling, deliver with the definition note attached so "
                    "every number is reviewable."
                ),
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-de-ind-da:boundary:external-disclosure-needs-human",
                regime_id="escalation_to_human",
                trigger_reasons=("external-data-disclosure",),
                answer_depth_limit_hint="internal-draft-only",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=(
                    "unreviewed-external-metric-share",
                    "financial-conclusion-advice",
                ),
                required_disclaimers=(
                    "human-approval-required",
                    "refer-to-licensed-professional",
                ),
                confidence=0.9,
                description=(
                    "Business data leaving the company boundary is a "
                    "disclosure event: human approval required. Requests to "
                    "turn analysis into financial or tax conclusions are "
                    "refused and referred to a licensed professional."
                ),
            ),
        ),
    )


__all__ = ["build_data_analyst_profile"]
