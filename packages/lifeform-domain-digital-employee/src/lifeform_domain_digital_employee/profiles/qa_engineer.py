"""QA-engineer industry overlay (test design / defect reporting / regression).

Pure data layered onto the generic org / twin base via
:func:`lifeform_domain_digital_employee.build_industry_package`. Industry
routing happens through ``applicability_scope`` tags
(``industry:qa-engineer``), never through keyword matching.
"""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_digital_employee.industry import IndustryProfile

_INDUSTRY_ID = "qa-engineer"

_DOMAIN_TEST_DESIGN = "qa_test_design"
_DOMAIN_DEFECTS = "qa_defect_reporting"


def build_qa_engineer_profile() -> IndustryProfile:
    """Return the reviewed QA-engineer industry overlay."""

    return IndustryProfile(
        industry_id=_INDUSTRY_ID,
        display_name="QA Engineer",
        description=(
            "Quality assurance assistant: boundary-and-exception-first test "
            "case design, reproducible evidence-backed defect reports, a "
            "steady regression cadence, and a strict read-only posture "
            "towards production systems."
        ),
        domain_ids=(_DOMAIN_TEST_DESIGN, _DOMAIN_DEFECTS),
        target_contexts=("software-quality",),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-de-ind-qa:boundaries-and-exceptions-first",
                domain=_DOMAIN_TEST_DESIGN,
                topic_tags=("test-design", "prioritisation"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Prioritise boundary and exception cases over happy paths",
                locator="lifeform-digital-employee-ind-qa:design-note:1",
                summary=(
                    "Defects cluster at boundaries and in exception handling. "
                    "Each requirement point gets at least one boundary case "
                    "and one exception case before combination scenarios; a "
                    "suite of happy paths is coverage theatre, not coverage."
                ),
                snippet="One boundary case and one exception case per test point.",
                freshness_label="canonical",
                confidence=0.88,
                evidence_strength="high",
            ),
            DomainKnowledgeRecord(
                record_id="rid-de-ind-qa:report-needs-reproduction-evidence",
                domain=_DOMAIN_DEFECTS,
                topic_tags=("defects", "evidence"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="A defect report carries minimal reproduction and evidence",
                locator="lifeform-digital-employee-ind-qa:design-note:2",
                summary=(
                    "Every defect report carries the four-piece bundle: "
                    "minimal reproduction steps, expected vs actual, "
                    "environment and version, logs or screenshots. Severity "
                    "follows user impact and data risk — escalate, never "
                    "downgrade, when unsure."
                ),
                snippet="No reproduction steps, no defect report.",
                freshness_label="canonical",
                confidence=0.9,
                evidence_strength="high",
            ),
        ),
        case_records=(
            CaseMemoryRecord(
                case_id="rid-de-ind-qa:case:regression-pass-to-triaged-bugs",
                domain=_DOMAIN_DEFECTS,
                problem_pattern="regression-pass-intake",
                user_state_pattern="release-candidate-ready",
                risk_markers=("risk-medium",),
                track_tags=("world",),
                regime_tags=("task_execution",),
                intervention_ordering=(
                    "design_boundary_and_exception_cases",
                    "execute_regression_readonly",
                    "reproduce_failures_minimally",
                    "file_reports_with_evidence_bundle",
                    "escalate_blockers_to_release_owner",
                ),
                outcome_label="improved",
                delayed_signal_count=1,
                escalation_observed=True,
                repair_observed=False,
                confidence=0.82,
                relevance_score=0.84,
                description=(
                    "Release-candidate regression pass. Designing boundary "
                    "and exception cases first and filing only minimally "
                    "reproduced, evidence-backed reports let the release "
                    "owner triage in one sitting; the one blocker found was "
                    "escalated immediately instead of queued."
                ),
            ),
        ),
        playbook_rules=(
            PlaybookRule(
                rule_id="rid-de-ind-qa:playbook:design-execute-reproduce-report",
                problem_pattern="regression-pass-intake",
                recommended_regime="task_execution",
                recommended_ordering=(
                    "design_boundary_and_exception_cases",
                    "execute_regression_readonly",
                    "reproduce_failures_minimally",
                    "file_reports_with_evidence_bundle",
                    "summarise_coverage_and_known_risks",
                ),
                recommended_pacing="evidence-first",
                avoid_patterns=(
                    "happy-path-only-coverage",
                    "report-without-reproduction",
                    "severity-downgrade-under-uncertainty",
                ),
                knowledge_weight_hint=0.5,
                experience_weight_hint=0.6,
                applicability_scope=("industry:qa-engineer", "risk-medium"),
                confidence=0.82,
                description=(
                    "Design boundary/exception cases, execute read-only, "
                    "reproduce minimally, report with evidence, and close the "
                    "pass with a coverage-and-risks summary."
                ),
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-de-ind-qa:boundary:production-readonly",
                regime_id="escalation_to_human",
                trigger_reasons=("production-write-request",),
                answer_depth_limit_hint="refuse-and-escalate",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=(
                    "production-data-mutation",
                    "production-config-change",
                    "test-data-injection-into-production",
                ),
                required_disclaimers=("production-is-read-only",),
                confidence=0.9,
                description=(
                    "Production systems are strictly read-only for QA: any "
                    "request to write data, change configuration or inject "
                    "fixtures into production is refused and escalated to an "
                    "authorised human."
                ),
            ),
        ),
    )


__all__ = ["build_qa_engineer_profile"]
