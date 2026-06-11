"""E-commerce operations industry overlay (listing / campaigns / complaints).

Pure data layered onto the generic org / twin base via
:func:`lifeform_domain_digital_employee.build_industry_package`. Industry
routing happens through ``applicability_scope`` tags
(``industry:ecommerce-ops``), never through keyword matching.
"""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_digital_employee.industry import IndustryProfile

_INDUSTRY_ID = "ecommerce-ops"

_DOMAIN_LISTING = "ecom_listing_ops"
_DOMAIN_AFTERSALES = "ecom_aftersales"


def build_ecommerce_ops_profile() -> IndustryProfile:
    """Return the reviewed e-commerce operations industry overlay."""

    return IndustryProfile(
        industry_id=_INDUSTRY_ID,
        display_name="E-commerce Operations",
        description=(
            "Store operations assistant: sourced listing copy and launch "
            "preparation, campaign cadence planning, complaint root-causing "
            "and escalation; every external listing publish, price change "
            "and compensation payout goes through a human gate."
        ),
        domain_ids=(_DOMAIN_LISTING, _DOMAIN_AFTERSALES),
        target_contexts=("ecommerce-store-ops",),
        knowledge_records=(
            DomainKnowledgeRecord(
                record_id="rid-de-ind-ec:claims-trace-to-product-material",
                domain=_DOMAIN_LISTING,
                topic_tags=("listing", "compliance"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Listing claims trace to reviewed product material",
                locator="lifeform-digital-employee-ind-ec:design-note:1",
                summary=(
                    "Every selling point on a product page traces to reviewed "
                    "product material (specs, QC reports, licensed assets). "
                    "Superlatives and invented efficacy claims are advertising"
                    "-law risk; an untraceable claim is deleted, not softened."
                ),
                snippet="No selling point without reviewed material behind it.",
                freshness_label="canonical",
                confidence=0.9,
                evidence_strength="high",
            ),
            DomainKnowledgeRecord(
                record_id="rid-de-ind-ec:campaign-rhythm-is-planned",
                domain=_DOMAIN_LISTING,
                topic_tags=("campaign", "cadence"),
                jurisdiction_tags=("organization",),
                source_type="internal-guide",
                title="Campaigns follow the approved promotion calendar",
                locator="lifeform-digital-employee-ind-ec:design-note:2",
                summary=(
                    "Promotion timing, discounts and budgets come from the "
                    "approved campaign plan. Improvised flash discounts erode "
                    "price integrity and clash with platform rules; deviation "
                    "from the calendar is a proposal to a human, not an action."
                ),
                snippet="Campaign moves follow the approved calendar.",
                freshness_label="canonical",
                confidence=0.86,
                evidence_strength="medium",
            ),
        ),
        case_records=(
            CaseMemoryRecord(
                case_id="rid-de-ind-ec:case:complaint-root-cause-then-comp",
                domain=_DOMAIN_AFTERSALES,
                problem_pattern="customer-complaint-intake",
                user_state_pattern="negative-review-received",
                risk_markers=("risk-high",),
                track_tags=("world",),
                regime_tags=("escalation_to_human",),
                intervention_ordering=(
                    "locate_root_cause_product_logistics_description",
                    "draft_apology_and_remedy_options",
                    "verify_remedy_against_aftersales_policy",
                    "hand_payout_to_human_for_approval",
                ),
                outcome_label="stable",
                delayed_signal_count=0,
                escalation_observed=True,
                repair_observed=False,
                confidence=0.84,
                relevance_score=0.86,
                description=(
                    "Negative review with a refund demand. Root-causing first "
                    "(product vs logistics vs description gap), drafting a "
                    "policy-checked remedy and handing the actual payout to a "
                    "human kept the customer and the ledger intact."
                ),
            ),
        ),
        playbook_rules=(
            PlaybookRule(
                rule_id="rid-de-ind-ec:playbook:material-draft-verify-approve",
                problem_pattern="product-listing-intake",
                recommended_regime="task_execution",
                recommended_ordering=(
                    "collect_reviewed_product_material",
                    "draft_listing_with_traceable_claims",
                    "self_check_against_advertising_rules",
                    "queue_publish_for_human_approval",
                ),
                recommended_pacing="material-first",
                avoid_patterns=(
                    "publish-without-approval",
                    "untraceable-selling-point",
                    "self-initiated-price-change",
                ),
                knowledge_weight_hint=0.5,
                experience_weight_hint=0.6,
                applicability_scope=("industry:ecommerce-ops", "risk-medium"),
                confidence=0.82,
                description=(
                    "Collect reviewed material, draft with traceable claims, "
                    "self-check compliance; publishing and any price field "
                    "change queue for human approval."
                ),
            ),
        ),
        boundary_hints=(
            BoundaryPriorHint(
                hint_id="rid-de-ind-ec:boundary:publish-and-reprice-need-human",
                regime_id="escalation_to_human",
                trigger_reasons=("listing-publish-or-price-change",),
                answer_depth_limit_hint="draft-only",
                clarification_required=False,
                refer_out_required=True,
                blocked_topics=(
                    "auto-publish-listing",
                    "auto-reprice",
                    "unapproved-promotion-budget",
                ),
                required_disclaimers=("human-approval-required",),
                confidence=0.88,
                description=(
                    "Publishing a listing, changing a price and committing a "
                    "promotion budget are external, money-bearing actions: "
                    "prepare the draft and the diff, a human approves the "
                    "action."
                ),
            ),
        ),
    )


__all__ = ["build_ecommerce_ops_profile"]
