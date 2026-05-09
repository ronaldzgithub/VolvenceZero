"""Text helper: rewrite third-person narration into second-person.

This is a **pure text utility**, not a behaviour-driving module. It
helps reviewers prepare ``NarrativeScene.setting`` strings that read
as if happening to the lifeform ("you stand at the bridge") rather
than as 3rd-person prose ("Zhang Wuji stood at the bridge"). The
ExperientialReplayDriver does NOT call this at runtime — by the
time a scene reaches the driver it must already be reviewer-written
in 1st-person framing. The helper is for offline content prep.

Why this lives in lifeform-domain-character (not vz-*):

It is a content-prep tool tied to the Reviewed CharacterSoulProfile
workflow. The kernel never sees this code path. It uses only stdlib
(no LLM) so it is safe to run in CI / local dev without any model
dependency.

The helper is intentionally small and conservative:

* Replace exact ``character_name`` and ``alias`` tokens with "你".
* Replace third-person pronoun ``他`` / ``她`` with "你" only when
  the previous sentence subject was the named character.
* Output is best-effort; the reviewer is expected to read the result
  and fix anything wrong before adding to a NarrativeArc.

What this helper deliberately does NOT do:

* Free-text classification of "who's speaking".
* Heuristic guesses about coreference outside the named-token /
  recent-subject window.
* Any LLM call.

If a reviewer needs richer rewriting they should run their own
LLM-assisted rewrite and re-review the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FirstPersonRewriteResult:
    """Result of :func:`to_first_person`. Reviewers inspect ``warnings``
    before accepting the rewrite.
    """

    rewritten: str
    replacements_made: int
    warnings: tuple[str, ...]


# Sentence boundary characters used to decide when to forget
# coreference state. Keeps the helper deterministic across paragraph /
# sentence boundaries.
_SENTENCE_BOUNDARIES: frozenset[str] = frozenset({"。", "！", "？", ".", "!", "?", "\n"})

# Conservative third-person pronoun set we are willing to flip when
# the recent subject was the named character. We do NOT flip ``它``
# (object pronoun) or possessive forms here.
_THIRD_PERSON_PRONOUNS: frozenset[str] = frozenset({"他", "她"})


def to_first_person(
    text: str,
    *,
    character_name: str,
    aliases: tuple[str, ...] = (),
) -> FirstPersonRewriteResult:
    """Rewrite third-person narration about a named character into 2nd-person.

    Algorithm:

    1. Replace every literal occurrence of ``character_name`` and each
       alias with "你".
    2. Track sentence boundaries; within a sentence whose subject was
       just the character (i.e. the sentence's first replacement was
       a name token), replace subsequent ``他`` / ``她`` pronouns with
       "你" as well.
    3. Emit a warning if no replacements were made (input may already
       be 1st-person, or character name didn't appear).

    Args:
        text: The third-person paragraph to rewrite.
        character_name: Canonical name token to replace
            (e.g. "张无忌").
        aliases: Additional name tokens (e.g. ("无忌", "教主"));
            order does not matter, longer match strings should ideally
            come first to avoid sub-token replacements but the helper
            sorts them by descending length internally.

    Returns:
        A :class:`FirstPersonRewriteResult`. The caller is expected
        to read ``rewritten`` and reviewer-edit anything that looks
        wrong before adding the result to a ``NarrativeScene``.
    """
    if not character_name.strip():
        raise ValueError("to_first_person: character_name must be non-empty")
    tokens = sorted(
        {character_name, *aliases},
        key=lambda token: len(token),
        reverse=True,
    )
    tokens = [token for token in tokens if token.strip()]
    if not tokens:
        raise ValueError("to_first_person: must supply at least one name token")
    output: list[str] = []
    replacements = 0
    warnings: list[str] = []
    sentence_subject_was_character = False
    cursor = 0
    while cursor < len(text):
        # Try matching any of the name tokens at the current cursor.
        matched_token = None
        for token in tokens:
            if text.startswith(token, cursor):
                matched_token = token
                break
        if matched_token is not None:
            output.append("你")
            replacements += 1
            sentence_subject_was_character = True
            cursor += len(matched_token)
            continue
        # Possible third-person pronoun.
        ch = text[cursor]
        if ch in _THIRD_PERSON_PRONOUNS and sentence_subject_was_character:
            output.append("你")
            replacements += 1
            cursor += 1
            continue
        # Sentence boundary: forget the coreference state.
        if ch in _SENTENCE_BOUNDARIES:
            sentence_subject_was_character = False
        output.append(ch)
        cursor += 1
    rewritten = "".join(output)
    if replacements == 0:
        warnings.append(
            "to_first_person: no replacements were made; the input may "
            "already be in first/second person, or the character_name "
            "did not appear in the text."
        )
    return FirstPersonRewriteResult(
        rewritten=rewritten,
        replacements_made=replacements,
        warnings=tuple(warnings),
    )


__all__ = [
    "FirstPersonRewriteResult",
    "to_first_person",
]
