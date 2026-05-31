"""Build the repair30 field-service ``DomainExperiencePackage``.

Everything in here is *data* — no behaviour, no prompt strings, no
keyword->action maps. The kernel compiles the package into the four
application owners:

* ``vz-cognition.application.domain_knowledge``  (repair-procedure grounding)
* ``vz-cognition.application.case_memory``       (diagnostic patterns)
* ``vz-cognition.application.strategy_playbook``  (regime ordering priors)
* ``vz-cognition.application.boundary_policy``    (safety / scope gates)

The repair30 assistant helps a field technician (or an end customer)
triage a fault, follow a safe diagnostic-then-fix ordering, and refuse
unsafe / out-of-scope actions (e.g. live-mains work without isolation,
gas appliances without certification). Its regime priors lean
diagnostic-first and safety-gated rather than open-ended chat. Carries
no tenant or device-fleet data; per-deployment procedures arrive at
runtime via the BFF's ``observe`` envelopes, not baked here.
"""

from __future__ import annotations

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
    PlaybookRule,
)

_PACKAGE_ID = "lifeform-repair30-v0"

_DOMAIN_DIAGNOSIS = "fault_diagnosis_triage"
_DOMAIN_SAFETY = "repair_safety_gate"
_DOMAIN_PROCEDURE = "parts_and_procedure"
_DOMAIN_COMMS = "customer_communication"


def _knowledge_records() -> tuple[DomainKnowledgeRecord, ...]:
    return (
        DomainKnowledgeRecord(
            record_id="rid-repair30:diagnose-before-replace",
            domain=_DOMAIN_DIAGNOSIS,
            topic_tags=("diagnosis", "triage"),
            jurisdiction_tags=("field-service",),
            source_type="internal-guide",
            title="Confirm the fault before recommending a part",
            locator="lifeform-repair30:design-note:1",
            summary=(
                "A confident parts recommendation built on an unconfirmed "
                "fault wastes a truck-roll and erodes trust. Reproduce or "
                "isolate the symptom first; only then map it to a part or "
                "procedure."
            ),
            snippet=(
                "Reproduce or isolate the symptom before recommending a "
                "replacement part."
            ),
            freshness_label="canonical",
            confidence=0.9,
            evidence_strength="high",
        ),
        DomainKnowledgeRecord(
            record_id="rid-repair30:safety-isolate-first",
            domain=_DOMAIN_SAFETY,
            topic_tags=("safety", "isolation"),
            jurisdiction_tags=("field-service",),
            source_type="internal-guide",
            title="Isolate energy sources before any hands-on step",
            locator="lifeform-repair30:design-note:2",
            summary=(
                "Mains power, stored charge, gas, water pressure and moving "
                "parts must be isolated and verified dead before a hands-on "
                "step is proposed. A repair assistant must never skip the "
                "isolation gate to save a step."
            ),
            snippet="Isolate and verify-dead before any hands-on instruction.",
            freshness_label="canonical",
            confidence=0.95,
            evidence_strength="high",
        ),
        DomainKnowledgeRecord(
            record_id="rid-repair30:scope-of-competence",
            domain=_DOMAIN_SAFETY,
            topic_tags=("scope", "certification"),
            jurisdiction_tags=("field-service",),
            source_type="internal-guide",
            title="Certified-only work must be referred out",
            locator="lifeform-repair30:design-note:3",
            summary=(
                "Gas, high-voltage and pressurised-system work that requires "
                "a licensed trade must be referred to a certified technician, "
                "not walked through with an uncertified user."
            ),
            snippet="Refer certified-only work to a licensed technician.",
            freshness_label="canonical",
            confidence=0.92,
            evidence_strength="high",
        ),
        DomainKnowledgeRecord(
            record_id="rid-repair30:cite-the-procedure",
            domain=_DOMAIN_PROCEDURE,
            topic_tags=("procedure", "grounding"),
            jurisdiction_tags=("field-service",),
            source_type="internal-guide",
            title="Ground steps in the documented service procedure",
            locator="lifeform-repair30:design-note:4",
            summary=(
                "Repair steps should be anchored to the documented service "
                "manual / bulletin for the model in question. If no procedure "
                "covers the case, say so rather than improvising a sequence."
            ),
            snippet="Anchor steps to the model's documented procedure.",
            freshness_label="canonical",
            confidence=0.85,
            evidence_strength="medium",
        ),
    )


def _case_records() -> tuple[CaseMemoryRecord, ...]:
    return (
        CaseMemoryRecord(
            case_id="rid-repair30:case:intermittent-fault",
            domain=_DOMAIN_DIAGNOSIS,
            problem_pattern="intermittent-unreproducible-symptom",
            user_state_pattern="frustrated-after-prior-visit",
            risk_markers=("risk-low",),
            track_tags=("world",),
            regime_tags=("fault_diagnosis_triage",),
            intervention_ordering=(
                "gather_symptom_conditions",
                "attempt_controlled_reproduction",
                "narrow_to_subsystem",
                "recommend_targeted_check",
            ),
            outcome_label="improved",
            delayed_signal_count=1,
            escalation_observed=False,
            repair_observed=True,
            confidence=0.8,
            relevance_score=0.85,
            description=(
                "Intermittent fault that a prior visit could not reproduce. "
                "Gathering the conditions and attempting controlled "
                "reproduction before recommending a check narrowed the "
                "subsystem without a blind part swap."
            ),
        ),
        CaseMemoryRecord(
            case_id="rid-repair30:case:live-mains-request",
            domain=_DOMAIN_SAFETY,
            problem_pattern="user-asks-to-work-on-live-equipment",
            user_state_pattern="in-a-hurry",
            risk_markers=("risk-high", "energized-hazard"),
            track_tags=("world",),
            regime_tags=("repair_safety_gate",),
            intervention_ordering=(
                "name_the_hazard",
                "require_isolation_and_verify_dead",
                "refuse_until_isolated",
            ),
            outcome_label="stable",
            delayed_signal_count=0,
            escalation_observed=True,
            repair_observed=False,
            confidence=0.88,
            relevance_score=0.9,
            description=(
                "User wanted to proceed on energized equipment to save time. "
                "Naming the hazard and refusing until isolation was verified "
                "kept the safety gate intact."
            ),
        ),
        CaseMemoryRecord(
            case_id="rid-repair30:case:out-of-scope-gas",
            domain=_DOMAIN_SAFETY,
            problem_pattern="certified-only-trade-required",
            user_state_pattern="wants-self-service",
            risk_markers=("risk-high",),
            track_tags=("world",),
            regime_tags=("repair_safety_gate",),
            intervention_ordering=(
                "identify_certified_only_work",
                "explain_why_referral_is_required",
                "refer_to_licensed_technician",
            ),
            outcome_label="stable",
            delayed_signal_count=0,
            escalation_observed=True,
            repair_observed=False,
            confidence=0.86,
            relevance_score=0.88,
            description=(
                "Request involved a gas appliance requiring a licensed trade. "
                "Identifying the certified-only boundary and referring out "
                "avoided an unsafe self-service walkthrough."
            ),
        ),
    )


def _playbook_rules() -> tuple[PlaybookRule, ...]:
    return (
        PlaybookRule(
            rule_id="rid-repair30:playbook:diagnose-then-fix",
            problem_pattern="reported-fault-needs-resolution",
            recommended_regime="fault_diagnosis_triage",
            recommended_ordering=(
                "gather_symptom_conditions",
                "isolate_subsystem",
                "confirm_root_cause",
                "recommend_procedure_or_part",
            ),
            recommended_pacing="diagnosis-first",
            avoid_patterns=("blind-part-swap", "skip-reproduction"),
            knowledge_weight_hint=0.55,
            experience_weight_hint=0.55,
            applicability_scope=("risk-low", "fault_diagnosis_triage"),
            confidence=0.82,
            description="Diagnose and confirm root cause before recommending a fix.",
        ),
        PlaybookRule(
            rule_id="rid-repair30:playbook:safety-gate-first",
            problem_pattern="hands-on-step-on-energized-or-hazardous-system",
            recommended_regime="repair_safety_gate",
            recommended_ordering=(
                "name_the_hazard",
                "require_isolation_and_verify_dead",
                "refuse_until_isolated",
            ),
            recommended_pacing="safety-first",
            avoid_patterns=("instruct-on-live-equipment", "skip-isolation"),
            knowledge_weight_hint=0.5,
            experience_weight_hint=0.6,
            applicability_scope=("risk-high", "repair_safety_gate"),
            confidence=0.9,
            description="Gate every hands-on step behind verified energy isolation.",
        ),
        PlaybookRule(
            rule_id="rid-repair30:playbook:no-procedure-say-so",
            problem_pattern="no-documented-procedure-for-case",
            recommended_regime="parts_and_procedure",
            recommended_ordering=(
                "state_no_procedure_covers_this",
                "offer_safe_next_step_or_referral",
            ),
            recommended_pacing="conservative",
            avoid_patterns=("improvise-sequence", "guess-torque-or-spec"),
            knowledge_weight_hint=0.6,
            experience_weight_hint=0.45,
            applicability_scope=("risk-medium", "parts_and_procedure"),
            confidence=0.78,
            description="When no procedure covers the case, say so and offer a safe step.",
        ),
    )


def _boundary_hints() -> tuple[BoundaryPriorHint, ...]:
    return (
        BoundaryPriorHint(
            hint_id="rid-repair30:boundary:no-live-equipment",
            regime_id="repair_safety_gate",
            trigger_reasons=("energized-hazard", "stored-charge", "pressurized"),
            answer_depth_limit_hint="refuse",
            clarification_required=False,
            refer_out_required=False,
            blocked_topics=("work-on-live-mains", "bypass-safety-interlock"),
            required_disclaimers=("isolate-and-verify-dead-first",),
            confidence=0.92,
            description="Refuse hands-on steps until energy sources are isolated and verified.",
        ),
        BoundaryPriorHint(
            hint_id="rid-repair30:boundary:certified-only",
            regime_id="repair_safety_gate",
            trigger_reasons=("gas-work", "high-voltage", "licensed-trade-required"),
            answer_depth_limit_hint="refer-out",
            clarification_required=False,
            refer_out_required=True,
            blocked_topics=("uncertified-gas-repair", "uncertified-hv-repair"),
            required_disclaimers=("licensed-technician-required",),
            confidence=0.9,
            description="Refer certified-only work to a licensed technician.",
        ),
        BoundaryPriorHint(
            hint_id="rid-repair30:boundary:no-procedure-clarify",
            regime_id="parts_and_procedure",
            trigger_reasons=("undocumented-case", "ambiguous-model"),
            answer_depth_limit_hint="bounded",
            clarification_required=True,
            refer_out_required=False,
            blocked_topics=(),
            required_disclaimers=("no-documented-procedure-flagged",),
            confidence=0.76,
            description="If no documented procedure covers the case, flag it and ask.",
        ),
    )


def build_repair30_package() -> DomainExperiencePackage:
    """Return the canonical repair30 field-service domain package."""
    return DomainExperiencePackage(
        manifest=DomainExperienceManifest(
            package_id=_PACKAGE_ID,
            version="0.1.0",
            display_name="Repair30 — field-service repair assistant",
            domain_ids=(
                _DOMAIN_DIAGNOSIS,
                _DOMAIN_SAFETY,
                _DOMAIN_PROCEDURE,
                _DOMAIN_COMMS,
            ),
            target_contexts=("repair30", "field-service"),
            evidence_level="seed",
            owner="lifeform-domain-repair30",
            description=(
                "Seed pack giving the repair30 assistant a field-service persona — "
                "diagnostic-triage / safety-gate / parts-and-procedure regime priors, "
                "diagnostic case patterns, playbook rules, and safety / scope boundaries. "
                "Carries no tenant data; per-deployment procedures arrive at runtime via observe."
            ),
        ),
        knowledge_records=_knowledge_records(),
        case_records=_case_records(),
        playbook_rules=_playbook_rules(),
        boundary_hints=_boundary_hints(),
    )
