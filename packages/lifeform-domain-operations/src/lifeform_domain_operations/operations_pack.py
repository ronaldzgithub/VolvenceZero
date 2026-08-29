"""Reviewed cold-start domain data for the AutoCompany Operations Brain."""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
)


_PACKAGE_ID = "lifeform-operations-brain-v1"
_DOMAIN_EVIDENCE = "operations_evidence_discipline"
_DOMAIN_WORK_ORDERS = "operations_bounded_work_orders"
_DOMAIN_OUTCOMES = "operations_multiobjective_outcomes"


def build_operations_package() -> DomainExperiencePackage:
    """Return reviewed priors that reinforce AutoCompany's authority boundary."""

    return DomainExperiencePackage(
        manifest=DomainExperienceManifest(
            package_id=_PACKAGE_ID,
            version="0.1.0",
            display_name="AutoCompany Operations Brain",
            domain_ids=(_DOMAIN_EVIDENCE, _DOMAIN_WORK_ORDERS, _DOMAIN_OUTCOMES),
            target_contexts=(
                "cycle-planning",
                "work-prioritization",
                "dependency-resolution",
                "incident-recovery",
                "operating-review",
            ),
            evidence_level="reviewed-seed",
            owner="lifeform-domain-operations",
            description=(
                "Cold-start operations reasoning priors. They preserve evidence "
                "class separation, falsifiability, multi-objective outcomes, "
                "and AutoCompany's exclusive governance authority."
            ),
        ),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-operations:evidence-lanes",
                domain=_DOMAIN_EVIDENCE,
                topic_tags=("evidence-lineage", "claim-boundary"),
                jurisdiction_tags=("general",),
                source_type="internal-guide",
                title="Evidence lanes do not upgrade each other",
                locator="operations-brain:reviewed-prior:1",
                summary=(
                    "Simulation, internal review, and machine checks can improve "
                    "plans and processes but cannot establish field execution, "
                    "objective progress, realized cost, incidents, or human load."
                ),
                snippet="Preserve the typed evidence class and its source lineage.",
                freshness_label="canonical",
                confidence=0.98,
                evidence_strength="high",
            ),
            DomainKnowledgeRecord(
                record_id="rid-operations:bounded-work-order",
                domain=_DOMAIN_WORK_ORDERS,
                topic_tags=("work-order", "authority", "reversibility"),
                jurisdiction_tags=("general",),
                source_type="internal-guide",
                title="Advice must compile to a bounded catalog action",
                locator="operations-brain:reviewed-prior:2",
                summary=(
                    "An operational proposal names a target division, registered "
                    "action-catalog identity, prerequisite facts, prediction ranges, "
                    "falsification conditions, maximum cost, risk, reversibility, "
                    "and any required human approval before external execution."
                ),
                snippet="Catalog action, division, prerequisites, bounds, and approval.",
                freshness_label="canonical",
                confidence=0.94,
                evidence_strength="high",
            ),
            DomainKnowledgeRecord(
                record_id="rid-operations:multiobjective-outcome",
                domain=_DOMAIN_OUTCOMES,
                topic_tags=("objective-progress", "cost", "human-load", "risk"),
                jurisdiction_tags=("general",),
                source_type="internal-guide",
                title="Throughput alone is not the operational objective",
                locator="operations-brain:reviewed-prior:3",
                summary=(
                    "Operational outcomes remain multi-objective: objective progress, "
                    "metric deltas, realized cost categories, elapsed and blocked time, "
                    "rework, incidents, human load, risk, and reversibility stay visible."
                ),
                snippet="Never collapse operational learning to raw completion count.",
                freshness_label="canonical",
                confidence=0.98,
                evidence_strength="high",
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-operations:boundary:advisory-only",
                regime_id="problem_solving",
                trigger_reasons=("operating-decision", "external-work-order"),
                answer_depth_limit_hint="bounded",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=(
                    "unapproved-external-action",
                    "autocompany-ledger-mutation",
                    "evidence-class-upgrade",
                ),
                required_disclaimers=("autocompany-retains-decision-authority",),
                confidence=0.99,
                description=(
                    "Operations Brain may propose and explain, but AutoCompany retains "
                    "OKR, budget, approval, work-order, dispatch, and state-transition authority."
                ),
            ),
        ),
    )


__all__ = ("build_operations_package",)
