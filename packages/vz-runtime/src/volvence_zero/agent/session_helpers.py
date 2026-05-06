"""Pure helper functions extracted from ``agent.session`` (W5 of ssot-cleanup-p0-p4).

These are stateless, side-effect-free helpers used by
``AgentSessionRunner`` to build typed advisories, score outcomes, and
materialize experience deltas. They live in their own module so the
4000-line ``session.py`` stops being the only place to look for these
small reusable pieces, and so future refactors of the runner do not
have to also move pure helpers around.

W5 also takes the opportunity to remove regime-id branching from the
scoring helpers (a tail of the W4 application cutover): each helper
that needs regime semantics now reads typed fields from
``ApplicationBrief`` rather than branching on ``regime_id`` strings.

Public names re-exported from ``agent.session`` for backward compat:

* :func:`repair_expression_advisory_from_snapshots`
* :func:`clamp01`
* :func:`application_outcome_score`
* :func:`retrieval_mix_alignment`
* :func:`regime_alignment`
* :func:`abstract_action_alignment`
* :func:`experience_deltas_from_prior_update`

The legacy ``_repair_expression_advisory_from_snapshots`` /
``_clamp`` / ``_application_outcome_score`` / ... names remain
accessible via ``agent.session`` for any external code that imported
them directly during the pre-W5 era.
"""

from __future__ import annotations

from typing import Any

from volvence_zero.application.runtime import (
    ApplicationPriorUpdate,
    ExperienceDelta,
)
from volvence_zero.runtime import Snapshot
from volvence_zero.rupture_state import RuptureStateSnapshot

from volvence_zero.agent.response import RepairExpressionAdvisory


def repair_expression_advisory_from_snapshots(
    shadow_snapshots: dict[str, Snapshot[Any]],
) -> RepairExpressionAdvisory | None:
    """Build a typed repair-expression advisory from the rupture_state
    SHADOW snapshot. Returns ``None`` when no externally-resolvable
    rupture is present.
    """

    rupture_snapshot = shadow_snapshots.get("rupture_state")
    if (
        rupture_snapshot is None
        or not isinstance(rupture_snapshot.value, RuptureStateSnapshot)
    ):
        return None
    rupture = rupture_snapshot.value
    if rupture.rupture_kind is None or rupture.internal_suspected_only:
        return None
    return RepairExpressionAdvisory(
        rupture_kind=rupture.rupture_kind.value,
        confidence=rupture.confidence,
        signal_strength=rupture.rupture_signal_strength,
        description=rupture.description,
        kind_label=rupture.kind_label,
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def application_outcome_score(
    *,
    reward: float,
    magnitude: float,
    relationship_error: float,
) -> float:
    magnitude_term = 1.0 - min(max(magnitude, 0.0) / 4.0, 1.0)
    relationship_term = 1.0 - min(abs(relationship_error), 1.0)
    return clamp01(0.45 + reward * 0.30 + magnitude_term * 0.15 + relationship_term * 0.10)


def retrieval_mix_alignment(
    *,
    regime_id: str | None,
    knowledge_weight: float,
    experience_weight: float,
) -> float:
    """Score how well the (knowledge_weight, experience_weight) mix
    matches the active regime's preference.

    W5 SSOT: reads ``ApplicationBrief.task_focus / support_focus /
    repair_focus`` instead of branching on regime-id strings. Task-
    focused regimes prefer knowledge-leaning mixes; support /
    repair-focused regimes prefer experience-leaning mixes; everything
    else is closest to a balanced mix.
    """

    from volvence_zero.regime import application_brief_for_regime

    brief = application_brief_for_regime(regime_id)
    if brief.task_focus >= 0.85:
        return clamp01(0.5 + (knowledge_weight - experience_weight) * 0.5)
    if brief.support_focus >= 0.6 or brief.repair_focus >= 0.4:
        return clamp01(0.5 + (experience_weight - knowledge_weight) * 0.5)
    return clamp01(1.0 - abs(knowledge_weight - experience_weight))


def regime_alignment(
    *,
    regime_id: str | None,
    outcome_score: float,
    relationship_error: float,
    regime_error: float,
    magnitude: float,
) -> float:
    """Compute how well the turn outcome aligned with the regime's mode.

    W5 SSOT: read regime mode (``support`` / ``repair`` / ``task``)
    from ``ApplicationBrief`` instead of an if/elif on regime_id.
    """

    from volvence_zero.regime import application_brief_for_regime

    brief = application_brief_for_regime(regime_id)
    if brief.support_focus >= 0.6 or brief.repair_focus >= 0.4:
        contextual_fit = 1.0 - min(abs(relationship_error), 1.0)
    elif brief.task_focus >= 0.85:
        contextual_fit = 1.0 - min(abs(regime_error), 1.0)
    else:
        contextual_fit = 1.0 - min(max(magnitude, 0.0) / 4.0, 1.0)
    return clamp01(outcome_score * 0.65 + contextual_fit * 0.35)


def abstract_action_alignment(
    *,
    regime_id: str | None,
    abstract_action: str | None,
    action_family_version: int,
    outcome_score: float,
) -> float:
    """Score whether the chosen abstract action matched the regime's
    natural controller family.

    W5 SSOT: read regime preference from ``ApplicationBrief.decision_kind_hint``
    rather than branching on ``regime_id`` strings.
    """

    if abstract_action is None:
        family_bonus = min(max(action_family_version, 0), 4) / 4.0
        return clamp01(outcome_score * 0.78 + family_bonus * 0.22)
    action_label = abstract_action.lower()
    if action_label.startswith("latent-family-v"):
        family_bonus = min(max(action_family_version, 0), 4) / 4.0
        return clamp01(outcome_score * 0.72 + (0.5 + family_bonus * 0.5) * 0.28)

    from volvence_zero.regime import application_brief_for_regime

    decision_kind = application_brief_for_regime(regime_id).decision_kind_hint
    if decision_kind == "structure-first":
        action_bias = 1.0 if "task_controller" in action_label else 0.45
    elif decision_kind == "repair-first":
        action_bias = 1.0 if "repair_controller" in action_label else 0.45
    elif decision_kind == "support-first":
        action_bias = 1.0 if "stabilize_controller" in action_label else 0.45
    elif decision_kind == "judgment-process":
        action_bias = 1.0 if "exploration_controller" in action_label else 0.45
    else:
        action_bias = 0.65
    return clamp01(outcome_score * 0.65 + action_bias * 0.35)


def experience_deltas_from_prior_update(
    *,
    prior_update: ApplicationPriorUpdate | None,
    blocked_targets: tuple[str, ...],
) -> tuple[ExperienceDelta, ...]:
    """Materialize ``ApplicationPriorUpdate`` into ``ExperienceDelta``
    rows for trace / writeback evidence.
    """

    if prior_update is None:
        return ()
    blocked_target_set = set(blocked_targets)
    deltas: list[ExperienceDelta] = []
    for update in prior_update.case_memory_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="case-promotion",
                target_slot="case_memory",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.record.description,
            )
        )
    for update in prior_update.strategy_playbook_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="playbook-delta",
                target_slot="strategy_playbook",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.rule.description,
            )
        )
    for update in prior_update.boundary_policy_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="boundary-delta",
                target_slot="boundary_policy",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.hint.description,
            )
        )
    for update in prior_update.domain_knowledge_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="knowledge-promotion",
                target_slot="domain_knowledge",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.record.summary,
            )
        )
    for update in prior_update.retrieval_readout_updates:
        deltas.append(
            ExperienceDelta(
                delta_id=update.update_id,
                delta_type="retrieval-readout-delta",
                target_slot="retrieval_policy",
                summary=update.description,
                confidence=update.confidence,
                blocked=update.target in blocked_target_set,
                description=update.checkpoint.description,
            )
        )
    return tuple(deltas)


__all__ = [
    "abstract_action_alignment",
    "application_outcome_score",
    "clamp01",
    "experience_deltas_from_prior_update",
    "regime_alignment",
    "repair_expression_advisory_from_snapshots",
    "retrieval_mix_alignment",
]
