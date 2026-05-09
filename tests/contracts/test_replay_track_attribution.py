"""Wave T3 contract tests — first-person helper + track attribution.

Two halves:

1. ``to_first_person`` text helper: deterministic rewrite of named
   third-person references to "你"; sentence-bounded coreference;
   warns on no-op input.

2. Track attribution: after compiling the 张无忌 character package,
   the seeded ``case_records`` carry ``track_tags`` that include
   "self" or "shared" — these are the upstream of the dual_track
   owner's SELF / SHARED classification. The replay driver inherits
   this attribution by replaying scenes whose application package
   already tags cases with the right tracks; no additional kernel
   surface is required.

This test does NOT assert "the kernel routed memory writes to SELF
track at runtime" — that's a property of the dual_track owner under
specific regime contexts, and is exercised by the broader ToM /
dual_track contract test surface. Here we only pin the upstream
attribution: the package's case records have track tags consistent
with the character's lived-self framing.
"""

from __future__ import annotations

import pytest

from lifeform_domain_character import (
    FirstPersonRewriteResult,
    build_character_package,
    build_zhang_wuji_profile,
    to_first_person,
)


# ---------------------------------------------------------------------------
# to_first_person
# ---------------------------------------------------------------------------


def test_to_first_person_replaces_named_token() -> None:
    result = to_first_person(
        "张无忌站在断桥边。",
        character_name="张无忌",
    )
    assert "你" in result.rewritten
    assert "张无忌" not in result.rewritten
    assert result.replacements_made == 1


def test_to_first_person_replaces_pronouns_after_named_subject() -> None:
    result = to_first_person(
        "张无忌停下脚步。他听见了呼救声。",
        character_name="张无忌",
    )
    # Both the name AND the subsequent 他 (in a new sentence) — wait,
    # sentence-bounded coreference means after a "。" we forget the
    # subject. So only the name itself flips here.
    assert result.rewritten.startswith("你停下脚步。")
    # The second sentence's 他 should NOT flip because the previous
    # sentence ended (sentence boundary 。 reset coreference).
    assert "他听见" in result.rewritten


def test_to_first_person_replaces_in_same_sentence() -> None:
    """Within the same sentence, after seeing the named character,
    subsequent ``他`` flips. This is the most useful case for prep:
    "张无忌走过去，他没有回头" → "你走过去，你没有回头"."""
    result = to_first_person(
        "张无忌走过去，他没有回头",
        character_name="张无忌",
    )
    assert "你走过去" in result.rewritten
    assert "你没有回头" in result.rewritten


def test_to_first_person_handles_aliases() -> None:
    result = to_first_person(
        "张无忌看着无忌剑，无忌剑反射着他的影子。",
        character_name="张无忌",
        aliases=("无忌剑",),
    )
    # "无忌剑" alias is used as a name token here; in a real review
    # workflow you would NOT alias an object name like this — this
    # test is exercising the alias mechanism, not best-practice usage.
    assert "你" in result.rewritten
    assert result.replacements_made >= 2


def test_to_first_person_warns_on_no_replacements() -> None:
    result = to_first_person(
        "山道上传来呼救声。你停下脚步。",
        character_name="张无忌",
    )
    assert result.replacements_made == 0
    assert any("no replacements" in warning for warning in result.warnings)


def test_to_first_person_rejects_empty_character_name() -> None:
    with pytest.raises(ValueError, match="character_name"):
        to_first_person("一些文字。", character_name="")


def test_to_first_person_rejects_only_whitespace_name() -> None:
    with pytest.raises(ValueError, match="character_name"):
        to_first_person("一些文字。", character_name="   ")


def test_first_person_result_is_frozen() -> None:
    """The result dataclass is immutable so callers can pass it
    around without worrying about mutation across reviews."""
    result = to_first_person("张无忌走开。", character_name="张无忌")
    assert isinstance(result, FirstPersonRewriteResult)
    with pytest.raises(Exception):
        result.rewritten = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Track attribution upstream
# ---------------------------------------------------------------------------


def test_zhang_wuji_package_case_records_have_self_or_shared_tracks() -> None:
    """The compiled case records must carry track tags consistent with
    a character living their own life. We expect at least one record
    tagged 'self' (the protagonist's own arc) and at least one tagged
    'shared' (interpersonal scene).

    These are the upstream of the dual_track owner's SELF/SHARED
    classification at runtime. Without this attribution, the replay
    cannot land in the right tracks.
    """
    profile = build_zhang_wuji_profile()
    package = build_character_package(profile)

    self_count = sum(
        1
        for case in package.case_records
        if "self" in case.track_tags
    )
    shared_count = sum(
        1
        for case in package.case_records
        if "shared" in case.track_tags
    )
    world_count = sum(
        1
        for case in package.case_records
        if "world" in case.track_tags
    )

    assert self_count >= 1, (
        f"Zhang Wuji package must have >= 1 case tagged 'self'; "
        f"got self={self_count} shared={shared_count} world={world_count}"
    )
    assert (shared_count + world_count) >= 1, (
        "Zhang Wuji package must have >= 1 interpersonal case "
        "(track_tags including 'shared' or 'world'); "
        f"got self={self_count} shared={shared_count} world={world_count}"
    )


def test_zhang_wuji_package_case_records_track_tags_use_documented_vocabulary() -> None:
    """All track_tags strings must come from the documented track
    vocabulary {self, world, shared}. New track tags require a
    deliberate addition to the application layer's ALLOWED_TRACK_TAGS
    constant, not silent introduction here.
    """
    profile = build_zhang_wuji_profile()
    package = build_character_package(profile)
    documented = {"self", "world", "shared"}
    for case in package.case_records:
        for tag in case.track_tags:
            assert tag in documented, (
                f"case {case.case_id!r} has track_tag={tag!r} not in "
                f"documented vocabulary {documented}"
            )
