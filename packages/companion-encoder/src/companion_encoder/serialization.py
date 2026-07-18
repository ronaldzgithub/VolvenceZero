# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Deterministic trajectory -> text serialization.

The encoder consumes a *prefix* of an ``InteractionTrajectory``: every
turn up to and including a label anchor, rendered as plain text. The
rendering is a pure function of the trajectory content (no timestamps,
no random ids), so the same trajectory always tokenizes identically —
a precondition for reproducible training and for the leakage audit
(release gate G3), which compares serialized training inputs against
held-out material by content.

Format (one line per element)::

    [session 0 | gap 0d]
    user: ...
    assistant: ...
    [session 1 | gap 21d]
    user: ...

Session boundaries and gap days are first-class text because the
relationship phases the encoder predicts (dormant, re_engaged, ...) are
defined over multi-session structure, not single-session sentiment.
"""

from __future__ import annotations

from companion_standard import InteractionTrajectory, RelationshipStateLabel


def session_header(session_index: int, gap_days_before: int) -> str:
    return f"[session {session_index} | gap {gap_days_before}d]"


def render_prefix(
    trajectory: InteractionTrajectory,
    *,
    session_index: int,
    turn_index: int,
) -> str:
    """Render all turns up to and including anchor ``(session_index, turn_index)``.

    Fail-loud on out-of-range anchors (mirrors the standard's own label
    validation) so a mis-anchored example never silently trains on the
    wrong prefix.
    """
    if session_index >= len(trajectory.sessions):
        raise ValueError(
            f"invalid_anchor: session {session_index} out of range "
            f"(trajectory has {len(trajectory.sessions)} sessions)"
        )
    anchor_session = trajectory.sessions[session_index]
    if turn_index >= len(anchor_session.turns):
        raise ValueError(
            f"invalid_anchor: turn {turn_index} out of range in session "
            f"{session_index} ({len(anchor_session.turns)} turns)"
        )

    lines: list[str] = []
    for session in trajectory.sessions[: session_index + 1]:
        lines.append(session_header(session.session_index, session.gap_days_before))
        last_turn = (
            turn_index
            if session.session_index == session_index
            else len(session.turns) - 1
        )
        for turn in session.turns[: last_turn + 1]:
            lines.append(f"{turn.role.value}: {turn.text}")
    return "\n".join(lines)


def render_label_prefix(
    trajectory: InteractionTrajectory, label: RelationshipStateLabel
) -> str:
    return render_prefix(
        trajectory,
        session_index=label.session_index,
        turn_index=label.turn_index,
    )


def render_full(trajectory: InteractionTrajectory) -> str:
    """Render the whole trajectory (used for trajectory-level embeddings)."""
    last_session = trajectory.sessions[-1]
    return render_prefix(
        trajectory,
        session_index=last_session.session_index,
        turn_index=len(last_session.turns) - 1,
    )


__all__ = [
    "render_full",
    "render_label_prefix",
    "render_prefix",
    "session_header",
]
