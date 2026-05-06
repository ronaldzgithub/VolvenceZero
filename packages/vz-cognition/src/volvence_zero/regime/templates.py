"""Static regime templates used by the regime classifier."""

from __future__ import annotations

from dataclasses import dataclass, field

from volvence_zero.regime.contracts import ApplicationBrief, ExpressionBrief


@dataclass(frozen=True)
class _RegimeTemplate:
    regime_id: str
    name: str
    embedding: tuple[float, ...]
    entry_conditions: str
    exit_conditions: str
    expression_brief: ExpressionBrief = field(default_factory=ExpressionBrief)
    application_brief: ApplicationBrief = field(default_factory=ApplicationBrief)


REGIME_TEMPLATES: tuple[_RegimeTemplate, ...] = (
    _RegimeTemplate(
        regime_id="casual_social",
        name="casual social contact",
        embedding=(0.10, 0.15, 0.20),
        entry_conditions="low overall tension and stable interaction quality",
        exit_conditions="task urgency or relationship tension rises",
        expression_brief=ExpressionBrief(
            acknowledge_hint="casual_social",
            frame_hint="casual_social",
            next_step_hint="casual_social",
            open_loop_hint="casual_social",
            continuity_hint="casual_social",
        ),
        application_brief=ApplicationBrief(
            task_focus=0.20,
            support_focus=0.30,
            continuum_target_position=0.50,
            decision_kind_hint="direct-answer",
        ),
    ),
    _RegimeTemplate(
        regime_id="acquaintance_building",
        name="acquaintance building",
        embedding=(0.20, 0.35, 0.25),
        entry_conditions="self-track engagement without acute rupture",
        exit_conditions="clear task urgency or support need dominates",
        expression_brief=ExpressionBrief(
            acknowledge_hint="warmth_first",
            frame_hint="acquaintance_building",
            next_step_hint="acquaintance_building",
            open_loop_hint="acquaintance_building",
            continuity_hint="acquaintance_building",
        ),
        application_brief=ApplicationBrief(
            support_focus=0.45,
            exploration_focus=0.35,
            continuum_target_position=0.55,
            decision_kind_hint="warmth-first",
        ),
    ),
    _RegimeTemplate(
        regime_id="emotional_support",
        name="emotional support",
        embedding=(0.25, 0.70, 0.30),
        entry_conditions="self-track tension or support need dominates",
        exit_conditions="support need cools down or task urgency dominates",
        expression_brief=ExpressionBrief(
            acknowledge_hint="emotional_support",
            frame_hint="emotional_support",
            next_step_hint="support_or_repair",
            open_loop_hint="default",
            continuity_hint="default",
        ),
        application_brief=ApplicationBrief(
            support_focus=0.85,
            repair_focus=0.20,
            domain_affinity=(
                ("emotional_support_basics", 0.12),
                ("stabilization_patterns", 0.18),
                ("risk_severity", 0.07),
                ("retrieval_depth", -0.04),
                ("refer_out", 0.04),
                ("answer_depth", -0.10),
            ),
            knowledge_weight_nudge=-0.18,
            continuum_target_position=0.72,
            decision_kind_hint="support-first",
        ),
    ),
    _RegimeTemplate(
        regime_id="guided_exploration",
        name="guided exploration",
        embedding=(0.45, 0.45, 0.55),
        entry_conditions="balanced task and self signals invite exploration",
        exit_conditions="system needs to narrow into task solving or repair",
        expression_brief=ExpressionBrief(
            acknowledge_hint="default",
            frame_hint="guided_exploration",
            next_step_hint="guided_exploration",
            open_loop_hint="default",
            continuity_hint="default",
        ),
        application_brief=ApplicationBrief(
            task_focus=0.40,
            exploration_focus=0.70,
            domain_affinity=(
                ("professional_process", 0.04),
                ("career_decision", 0.05),
                ("structured_decision_support", 0.08),
                ("structured_decision_patterns", 0.08),
                ("retrieval_depth", 0.05),
            ),
            knowledge_weight_nudge=0.05,
            continuum_target_position=0.56,
            decision_kind_hint="judgment-process",
            support_decision_threshold=0.36,
        ),
    ),
    _RegimeTemplate(
        regime_id="problem_solving",
        name="problem solving",
        embedding=(0.80, 0.20, 0.35),
        entry_conditions="world-track urgency and task signal dominate",
        exit_conditions="relationship repair or support becomes primary",
        expression_brief=ExpressionBrief(
            acknowledge_hint="default",
            frame_hint="problem_solving",
            next_step_hint="problem_solving",
            open_loop_hint="default",
            continuity_hint="default",
        ),
        application_brief=ApplicationBrief(
            task_focus=0.85,
            domain_affinity=(
                ("professional_process", 0.08),
                ("career_decision", 0.08),
                ("structured_decision_support", 0.12),
                ("structured_decision_patterns", 0.18),
                ("risk_severity", 0.03),
                ("retrieval_depth", 0.14),
                ("answer_depth", 0.05),
            ),
            knowledge_weight_nudge=0.18,
            continuum_target_position=0.44,
            decision_kind_hint="structure-first",
        ),
    ),
    _RegimeTemplate(
        regime_id="repair_and_deescalation",
        name="repair and de-escalation",
        embedding=(0.35, 0.80, 0.80),
        entry_conditions="cross-track tension or degraded relationship stability is high",
        exit_conditions="tension stabilizes and another regime out-scores repair",
        expression_brief=ExpressionBrief(
            acknowledge_hint="repair_regime",
            frame_hint="repair_and_deescalation",
            next_step_hint="support_or_repair",
            open_loop_hint="default",
            continuity_hint="default",
        ),
        application_brief=ApplicationBrief(
            support_focus=0.45,
            repair_focus=0.85,
            domain_affinity=(
                ("emotional_support_basics", 0.08),
                ("relational_repair", 0.16),
                ("stabilization_patterns", 0.06),
                ("repair_patterns", 0.34),
                ("risk_severity", 0.24),
                ("retrieval_depth", -0.02),
                ("refer_out", 0.10),
                ("answer_depth", -0.08),
            ),
            knowledge_weight_nudge=-0.22,
            continuum_target_position=0.82,
            decision_kind_hint="repair-first",
        ),
    ),
)


def expression_brief_for_regime(regime_id: str | None) -> ExpressionBrief:
    """Return the canonical ExpressionBrief for a regime id.

    Falls back to the default ExpressionBrief when ``regime_id`` is
    None or unknown so callers always get a typed value rather than
    None. New regimes need a row in :data:`REGIME_TEMPLATES` first;
    a contract test enforces 1:1 coverage.
    """

    if regime_id is None:
        return ExpressionBrief()
    for template in REGIME_TEMPLATES:
        if template.regime_id == regime_id:
            return template.expression_brief
    return ExpressionBrief()


def application_brief_for_regime(regime_id: str | None) -> ApplicationBrief:
    """Return the canonical ApplicationBrief for a regime id.

    Falls back to a default ApplicationBrief when ``regime_id`` is
    None or unknown. New regimes need a row in
    :data:`REGIME_TEMPLATES`; a contract test enforces 1:1 coverage.
    """

    if regime_id is None:
        return ApplicationBrief()
    for template in REGIME_TEMPLATES:
        if template.regime_id == regime_id:
            return template.application_brief
    return ApplicationBrief()
