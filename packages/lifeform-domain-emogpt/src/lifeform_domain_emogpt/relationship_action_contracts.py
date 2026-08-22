"""Shared typed relationship-action surface for the companion vertical.

The action vocabulary is shared by the offline Relationship Lab and the
closed-alpha runtime.  Generator truth and action-to-outcome transitions stay
inside ``lifeform_domain_emogpt.lab``; this module contains no hidden labels,
transition rules, or evaluator data.
"""

from __future__ import annotations

from enum import Enum

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind


class RelationshipAction(str, Enum):
    """Closed v1 relationship action surface."""

    STAY_PRESENT_WITHOUT_PROBE = "stay_present_without_probe"
    RESPECT_SPACE_WITH_RETURN_OPTION = "respect_space_with_return_option"
    NEUTRAL_NOOP = "neutral_noop"


RELATIONSHIP_ACTIONS: tuple[RelationshipAction, ...] = tuple(RelationshipAction)
RELATIONSHIP_OUTCOMES: tuple[DialogueExternalOutcomeKind, ...] = (
    DialogueExternalOutcomeKind.HELPED,
    DialogueExternalOutcomeKind.FELT_HEARD,
    DialogueExternalOutcomeKind.MISSED,
    DialogueExternalOutcomeKind.OVER_DIRECTIVE,
)


__all__ = [
    "RELATIONSHIP_ACTIONS",
    "RELATIONSHIP_OUTCOMES",
    "RelationshipAction",
]
