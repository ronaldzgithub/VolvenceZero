"""Reflection-driven UX hint mapping (Wave 1 part B).

Translates the typed ``ReflectionLessonId`` / ``ReflectionTensionId``
identifiers published by the reflection writeback owner into the short
human-readable lines the lifeform's response synthesizer can include
in a reply. Living here, in lifeform-expression, makes the layer
boundary explicit:

* ``vz-cognition.reflection`` owns the *what* (which lessons /
  tensions fired).
* ``lifeform-expression`` owns the *how* (how those firings sound to
  the user).

The kernel ``ResponseSynthesizer`` (in ``vz-runtime.agent.response``)
no longer renders this text directly. Tests / fallback paths that
used the old in-kernel ``lesson_hint_map`` should call
``reflection_lesson_hint(...)`` from here.

Adding a new lesson / tension requires:

1. adding the enum member in
   ``volvence_zero.reflection.writeback``;
2. adding a corresponding string here;
3. updating ``docs/specs/expression-layer.md``.

A contract test (``tests/test_rationale_tags.py`` /
``tests/test_reflection_hints.py``) enforces 1:1 coverage so a missing
hint surfaces immediately rather than silently falling back to the
generic line.
"""

from __future__ import annotations

from typing import Mapping

from volvence_zero.reflection import ReflectionLessonId, ReflectionTensionId


_LESSON_HINTS: Mapping[ReflectionLessonId, str] = {
    ReflectionLessonId.PROMOTE_HIGH_SIGNAL_MEMORIES: (
        "I want to keep hold of the strongest signals rather than restart from scratch."
    ),
    ReflectionLessonId.REINFORCE_RECENT_HIGH_CREDIT_BELIEFS: (
        "I want to lean on the parts of the interaction that have already proven stable."
    ),
    ReflectionLessonId.ADJUST_TRACK_PRIORITY_FROM_SESSION_FEEDBACK: (
        "I want to rebalance between the task and the relationship instead of "
        "over-favoring one side."
    ),
    ReflectionLessonId.REBALANCE_TEMPORAL_PRIOR_TOWARD_MEMORY: (
        "I want to use continuity and recalled context more strongly in how I respond."
    ),
    ReflectionLessonId.REBALANCE_TEMPORAL_PRIOR_TOWARD_REFLECTION: (
        "I want the slower reflective layer to shape this reply more directly."
    ),
    ReflectionLessonId.REBALANCE_TEMPORAL_PRIOR_TOWARD_RESIDUAL: (
        "I want to stay closer to the immediate task signal while keeping the "
        "frame coherent."
    ),
    ReflectionLessonId.INCREASE_CONTROLLER_PERSISTENCE_FOR_CONTINUITY: (
        "I want to keep the same internal frame steady for longer instead of "
        "jumping too quickly."
    ),
    ReflectionLessonId.REDUCE_CONTROLLER_PERSISTENCE_FOR_FASTER_RECOVERY: (
        "I want to stay flexible enough to recover if the current frame is not "
        "the right one."
    ),
    ReflectionLessonId.ALLOW_CONTROLLER_SWITCH_WHEN_CONTEXT_SHIFTS: (
        "I am allowing myself to change internal stance because the context has shifted."
    ),
    ReflectionLessonId.HOLD_CONTROLLER_BEFORE_SWITCHING: (
        "I want to hold the current stance a bit longer before I switch tracks."
    ),
    ReflectionLessonId.RESPECT_METACONTROLLER_RUNTIME_GUARD: (
        "I am keeping the response conservative because an internal guard was triggered."
    ),
    ReflectionLessonId.KEEP_CONTROLLER_GUARD_SIGNAL_IN_BACKGROUND: (
        "I am keeping a background check on the controller while still moving forward."
    ),
    ReflectionLessonId.REVIEW_TENSION_BEFORE_AUTO_WRITEBACK: (
        "I do not want to smooth over the remaining tension too quickly."
    ),
    ReflectionLessonId.RELATIONSHIP_STRATEGY_MISMATCH: (
        "I want to revisit the relational strategy that just missed."
    ),
    ReflectionLessonId.TASK_FRAMING_INADEQUATE: (
        "I want to reframe the task; the previous framing did not land."
    ),
    ReflectionLessonId.ABSTRACT_ACTION_INSTABILITY: (
        "I want to steady the abstract action instead of switching it again."
    ),
    ReflectionLessonId.REGIME_SELECTION_MISMATCH: (
        "I want to reconsider which regime fits this turn rather than push through."
    ),
    ReflectionLessonId.RESTRUCTURE_ACTION_FAMILY_BANK: (
        "I am letting the slower loop reorganise how my abstract actions group together."
    ),
}


_TENSION_HINTS: Mapping[ReflectionTensionId, str] = {
    ReflectionTensionId.CROSS_TRACK_TENSION_HIGH: (
        "There is a strong mismatch between task pressure and relational stability "
        "right now."
    ),
    ReflectionTensionId.CROSS_TRACK_ALIGNMENT_DRIFT: (
        "I can feel some drift between the task frame and the relational frame."
    ),
    ReflectionTensionId.SELF_TRACK_PRESSURE_DOMINANT: (
        "The relational or emotional side currently needs more weight than the task side."
    ),
    ReflectionTensionId.WORLD_TRACK_PRESSURE_DOMINANT: (
        "The task side is currently pressing harder than the relational side."
    ),
    ReflectionTensionId.RELATIONSHIP_STABILITY_SOFT_DROP: (
        "I do not want to assume continuity is fully stable yet."
    ),
    ReflectionTensionId.WARMTH_SIGNAL_THIN: (
        "I want to add more warmth instead of sounding mechanically efficient."
    ),
    ReflectionTensionId.TASK_SIGNAL_DIFFUSE: (
        "The task signal is still a bit diffuse, so I should narrow it carefully."
    ),
    ReflectionTensionId.PE_RELATIONSHIP_MISMATCH: (
        "There is a relationship-level prediction gap I do not want to paper over."
    ),
    ReflectionTensionId.PE_TASK_MISMATCH: (
        "There is a task-level prediction gap I want to take seriously."
    ),
    ReflectionTensionId.PE_ACTION_INSTABILITY: (
        "My internal action choice has been unstable; I will move more deliberately."
    ),
    ReflectionTensionId.PE_REGIME_INSTABILITY: (
        "The interaction frame itself is wobbly; I do not want to compound that."
    ),
}


_GENERIC_LESSON_HINT = (
    "I am letting the slower reflective layer shape the reply rather than "
    "treating it as a no-op."
)
_GENERIC_TENSION_HINT = (
    "I want to keep an eye on the tensions that are still open rather than "
    "collapsing too quickly."
)


def reflection_lesson_hint(lesson_id: str | None) -> str:
    """Return the UX hint for a reflection lesson id.

    Falls back to a generic line if the id is unknown so the renderer
    never crashes on a fresh-from-future-version snapshot. Schema drift
    is detected by the contract test, not by silent fallback.
    """

    if lesson_id is None:
        return ""
    try:
        key = ReflectionLessonId(lesson_id)
    except ValueError:
        return _GENERIC_LESSON_HINT
    return _LESSON_HINTS.get(key, _GENERIC_LESSON_HINT)


def reflection_tension_hint(tension_id: str | None) -> str:
    """Return the UX hint for a reflection tension id."""

    if tension_id is None:
        return ""
    try:
        key = ReflectionTensionId(tension_id)
    except ValueError:
        return _GENERIC_TENSION_HINT
    return _TENSION_HINTS.get(key, _GENERIC_TENSION_HINT)


def lesson_hint_map() -> Mapping[ReflectionLessonId, str]:
    """Read-only view of the lesson hint map.

    Useful for tests that want to assert 1:1 coverage between the
    enum and the hint map.
    """

    return dict(_LESSON_HINTS)


def tension_hint_map() -> Mapping[ReflectionTensionId, str]:
    """Read-only view of the tension hint map."""

    return dict(_TENSION_HINTS)


__all__ = [
    "lesson_hint_map",
    "tension_hint_map",
    "reflection_lesson_hint",
    "reflection_tension_hint",
]
