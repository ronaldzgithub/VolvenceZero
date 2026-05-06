"""Wave 1 part B: every reflection lesson / tension id has a hint.

This is the SSOT contract test for the lesson / tension enum: each
member of ``ReflectionLessonId`` / ``ReflectionTensionId`` published
by reflection writeback must have a corresponding UX hint in
``lifeform_expression.reflection_hints``.

If a new enum member lands without a hint, this test fails before
the consumer silently falls back to a generic line.
"""

from __future__ import annotations

import pytest

from lifeform_expression import (
    lesson_hint_map,
    reflection_lesson_hint,
    reflection_tension_hint,
    tension_hint_map,
)
from volvence_zero.reflection import (
    ReflectionLessonId,
    ReflectionTensionId,
)


def test_every_lesson_id_has_a_hint() -> None:
    hints = lesson_hint_map()
    missing = [member for member in ReflectionLessonId if member not in hints]
    assert not missing, (
        "Reflection lesson ids missing a hint in "
        "lifeform_expression.reflection_hints: "
        + ", ".join(member.name for member in missing)
    )


def test_every_tension_id_has_a_hint() -> None:
    hints = tension_hint_map()
    missing = [member for member in ReflectionTensionId if member not in hints]
    assert not missing, (
        "Reflection tension ids missing a hint in "
        "lifeform_expression.reflection_hints: "
        + ", ".join(member.name for member in missing)
    )


@pytest.mark.parametrize("member", list(ReflectionLessonId))
def test_lesson_hint_lookup_by_string(member: ReflectionLessonId) -> None:
    """Lookup by the enum value (the string id reflection actually
    publishes in ``ReflectionSnapshot.lessons_extracted``) must
    return the canonical hint.
    """

    assert reflection_lesson_hint(member.value), (
        f"empty hint for lesson id {member.value!r}"
    )


@pytest.mark.parametrize("member", list(ReflectionTensionId))
def test_tension_hint_lookup_by_string(member: ReflectionTensionId) -> None:
    assert reflection_tension_hint(member.value), (
        f"empty hint for tension id {member.value!r}"
    )


def test_unknown_lesson_id_falls_back_to_generic() -> None:
    hint = reflection_lesson_hint("not-a-real-lesson-id")
    assert hint, "unknown lesson id should fall back to generic line"


def test_unknown_tension_id_falls_back_to_generic() -> None:
    hint = reflection_tension_hint("not-a-real-tension-id")
    assert hint, "unknown tension id should fall back to generic line"


def test_none_lesson_id_returns_empty() -> None:
    assert reflection_lesson_hint(None) == ""


def test_none_tension_id_returns_empty() -> None:
    assert reflection_tension_hint(None) == ""
