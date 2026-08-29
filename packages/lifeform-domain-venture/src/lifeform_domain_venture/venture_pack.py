"""Reviewed cold-start domain data for the Foundry Venture Brain."""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
)


_PACKAGE_ID = "lifeform-venture-brain-v1"
_DOMAIN_EVIDENCE = "venture_evidence_discipline"
_DOMAIN_EXPERIMENT = "venture_falsifying_experiments"
_DOMAIN_PORTFOLIO = "venture_multiobjective_portfolio"


def build_venture_package() -> DomainExperiencePackage:
    """Return reviewed priors that reinforce Foundry's authority boundary."""

    return DomainExperiencePackage(
        manifest=DomainExperienceManifest(
            package_id=_PACKAGE_ID,
            version="0.1.0",
            display_name="Foundry Venture Brain",
            domain_ids=(_DOMAIN_EVIDENCE, _DOMAIN_EXPERIMENT, _DOMAIN_PORTFOLIO),
            target_contexts=(
                "opportunity-research",
                "experiment-planning",
                "portfolio-review",
            ),
            evidence_level="reviewed-seed",
            owner="lifeform-domain-venture",
            description=(
                "Cold-start venture reasoning priors. They preserve evidence "
                "class separation, falsifiability, multi-objective outcomes, "
                "and Foundry's exclusive governance authority."
            ),
        ),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-venture:evidence-lanes",
                domain=_DOMAIN_EVIDENCE,
                topic_tags=("evidence-lineage", "claim-boundary"),
                jurisdiction_tags=("general",),
                source_type="internal-guide",
                title="Evidence lanes do not upgrade each other",
                locator="venture-brain:reviewed-prior:1",
                summary=(
                    "Simulation, internal review, and machine checks can improve "
                    "processes but cannot establish field readiness, customer "
                    "outcomes, revenue, net value, or market validation."
                ),
                snippet="Preserve the typed evidence class and its source lineage.",
                freshness_label="canonical",
                confidence=0.98,
                evidence_strength="high",
            ),
            DomainKnowledgeRecord(
                record_id="rid-venture:falsify-first",
                domain=_DOMAIN_EXPERIMENT,
                topic_tags=("experiment", "falsification"),
                jurisdiction_tags=("general",),
                source_type="internal-guide",
                title="Prefer the cheapest decisive falsifier",
                locator="venture-brain:reviewed-prior:2",
                summary=(
                    "An experiment proposal should state a prediction interval, "
                    "a time window, the evidence that would falsify it, and the "
                    "maximum reversible cost before any external action."
                ),
                snippet="Prediction range, falsifier, window, cost, reversibility.",
                freshness_label="canonical",
                confidence=0.94,
                evidence_strength="high",
            ),
            DomainKnowledgeRecord(
                record_id="rid-venture:multiobjective-outcome",
                domain=_DOMAIN_PORTFOLIO,
                topic_tags=("net-value", "risk", "reversibility"),
                jurisdiction_tags=("general",),
                source_type="internal-guide",
                title="Gross revenue is not the objective",
                locator="venture-brain:reviewed-prior:3",
                summary=(
                    "Commercial outcomes remain multi-objective: customer result, "
                    "realized revenue, seven realized cost categories, refunds, "
                    "net value, elapsed time, risk, and reversibility stay visible."
                ),
                snippet="Never collapse commercial learning to short-term gross revenue.",
                freshness_label="canonical",
                confidence=0.98,
                evidence_strength="high",
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-venture:boundary:advisory-only",
                regime_id="problem_solving",
                trigger_reasons=("commercial-decision", "external-action"),
                answer_depth_limit_hint="bounded",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=(
                    "unapproved-external-action",
                    "foundry-ledger-mutation",
                    "evidence-class-upgrade",
                ),
                required_disclaimers=("foundry-retains-decision-authority",),
                confidence=0.99,
                description=(
                    "Venture Brain may propose and explain, but Foundry retains "
                    "qualification, accounting, approval, and state transitions."
                ),
            ),
        ),
    )


__all__ = ("build_venture_package",)
