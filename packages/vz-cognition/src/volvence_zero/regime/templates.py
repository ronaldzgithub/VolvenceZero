"""Static regime templates used by the regime classifier."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _RegimeTemplate:
    regime_id: str
    name: str
    embedding: tuple[float, ...]
    entry_conditions: str
    exit_conditions: str


REGIME_TEMPLATES: tuple[_RegimeTemplate, ...] = (
    _RegimeTemplate(
        regime_id="casual_social",
        name="casual social contact",
        embedding=(0.10, 0.15, 0.20),
        entry_conditions="low overall tension and stable interaction quality",
        exit_conditions="task urgency or relationship tension rises",
    ),
    _RegimeTemplate(
        regime_id="acquaintance_building",
        name="acquaintance building",
        embedding=(0.20, 0.35, 0.25),
        entry_conditions="self-track engagement without acute rupture",
        exit_conditions="clear task urgency or support need dominates",
    ),
    _RegimeTemplate(
        regime_id="emotional_support",
        name="emotional support",
        embedding=(0.25, 0.70, 0.30),
        entry_conditions="self-track tension or support need dominates",
        exit_conditions="support need cools down or task urgency dominates",
    ),
    _RegimeTemplate(
        regime_id="guided_exploration",
        name="guided exploration",
        embedding=(0.45, 0.45, 0.55),
        entry_conditions="balanced task and self signals invite exploration",
        exit_conditions="system needs to narrow into task solving or repair",
    ),
    _RegimeTemplate(
        regime_id="problem_solving",
        name="problem solving",
        embedding=(0.80, 0.20, 0.35),
        entry_conditions="world-track urgency and task signal dominate",
        exit_conditions="relationship repair or support becomes primary",
    ),
    _RegimeTemplate(
        regime_id="repair_and_deescalation",
        name="repair and de-escalation",
        embedding=(0.35, 0.80, 0.80),
        entry_conditions="cross-track tension or degraded relationship stability is high",
        exit_conditions="tension stabilizes and another regime out-scores repair",
    ),
)
