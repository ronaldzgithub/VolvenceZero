"""Contract tests for the AAC commitment lifecycle (Gap 7).

These tests enforce the invariants that ``docs/specs/aac-lifecycle.md``
spells out and that the rest of the system relies on:

1. **Operation \u2192 lifecycle table is exhaustive.** Every
   ``SemanticProposalOperation`` maps to a lifecycle transition and a
   follow-up policy (outcome may be ``None`` but the operation must still
   appear in the outcome map so we can't silently lose a state).
2. **Outcome-requires-evidence.** A ``CommitmentLifecycleEntry`` carrying
   a typed ``last_outcome`` MUST have non-empty ``last_outcome_evidence``
   AND a non-negative ``last_outcome_at_turn``. Missing either is a
   contract violation \u2014 reflection writeback / audit downstream rely
   on every outcome being anchored.
3. **No keyword-driven alignment extraction.** Grep the semantic_state
   package for patterns like ``if "agree" in user_text`` / ``if "\u597d\u7684"
   in user_input``: those are forbidden by red line A
   (``.cursor/rules/no-keyword-matching-hacks.mdc``). Alignment MUST come
   from a typed ``SemanticProposal`` (REVISE / COMPLETE / BLOCK op), not
   from regex over raw user text.
4. **Legal transition boundaries.** We do not hand-encode a full NxN
   state machine here \u2014 the transition table is the source of truth \u2014
   but we do verify that illegal constructions (e.g. outcome without
   evidence) are rejected by the dataclass ``__post_init__``.

This file is a *contract* test: fast, deterministic, no LLM calls.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from volvence_zero.semantic_state import (
    AdvocacyState,
    AlignmentState,
    CommitmentLifecycleEntry,
    CommitmentOutcomeKind,
    FollowupPolicy,
    SemanticProposalOperation,
    commitment_followup_policy_for_operation,
    commitment_lifecycle_for_operation,
    commitment_outcome_for_operation,
)


# ---------------------------------------------------------------------------
# Operation coverage: every enum value must appear in every lifecycle map
# ---------------------------------------------------------------------------


def test_every_operation_has_lifecycle_mapping() -> None:
    """Every ``SemanticProposalOperation`` must be mapped to an (advocacy,
    alignment) pair by ``commitment_lifecycle_for_operation`` (possibly
    inheriting from ``previous``). If a new op is added to the enum and
    no mapping is registered, this test fails loudly.
    """
    for op in SemanticProposalOperation:
        advocacy, alignment = commitment_lifecycle_for_operation(op)
        assert isinstance(advocacy, AdvocacyState), (
            f"Operation {op.value} returned non-AdvocacyState: {advocacy!r}"
        )
        assert isinstance(alignment, AlignmentState), (
            f"Operation {op.value} returned non-AlignmentState: {alignment!r}"
        )


def test_every_operation_has_followup_policy() -> None:
    for op in SemanticProposalOperation:
        policy = commitment_followup_policy_for_operation(op)
        assert isinstance(policy, FollowupPolicy), (
            f"Operation {op.value} returned non-FollowupPolicy: {policy!r}"
        )


def test_every_operation_has_outcome_or_explicit_none() -> None:
    """Outcome mapping may return ``None`` (e.g. OBSERVE is not an
    outcome), but it must be an *explicit* decision \u2014 we assert the
    helper returns either a ``CommitmentOutcomeKind`` or ``None``, not
    raises. This stops silent attribute errors if a new op skipped the
    map.
    """
    for op in SemanticProposalOperation:
        outcome = commitment_outcome_for_operation(op)
        assert outcome is None or isinstance(outcome, CommitmentOutcomeKind), (
            f"Operation {op.value} returned non-outcome / non-None: {outcome!r}"
        )


# ---------------------------------------------------------------------------
# BLOCK is an alignment REJECT \u2014 the high-PE path
# ---------------------------------------------------------------------------


def test_block_operation_produces_reject_alignment() -> None:
    _, alignment = commitment_lifecycle_for_operation(SemanticProposalOperation.BLOCK)
    assert alignment is AlignmentState.REJECT


def test_block_operation_produces_defer_only_followup_policy() -> None:
    assert (
        commitment_followup_policy_for_operation(SemanticProposalOperation.BLOCK)
        is FollowupPolicy.DEFER_ONLY
    )


def test_complete_operation_produces_completed_outcome() -> None:
    assert (
        commitment_outcome_for_operation(SemanticProposalOperation.COMPLETE)
        is CommitmentOutcomeKind.COMPLETED
    )


def test_block_operation_produces_rejected_outcome() -> None:
    assert (
        commitment_outcome_for_operation(SemanticProposalOperation.BLOCK)
        is CommitmentOutcomeKind.REJECTED
    )


# ---------------------------------------------------------------------------
# Outcome-requires-evidence invariant
# ---------------------------------------------------------------------------


def test_outcome_without_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="last_outcome_evidence"):
        CommitmentLifecycleEntry(
            record_id="rec-1",
            advocacy_state=AdvocacyState.PROPOSED,
            alignment_state=AlignmentState.REJECT,
            last_outcome=CommitmentOutcomeKind.REJECTED,
            last_outcome_evidence="",
            last_outcome_at_turn=3,
        )


def test_outcome_without_turn_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="last_outcome_at_turn"):
        CommitmentLifecycleEntry(
            record_id="rec-1",
            advocacy_state=AdvocacyState.PROPOSED,
            alignment_state=AlignmentState.AGREE,
            last_outcome=CommitmentOutcomeKind.COMPLETED,
            last_outcome_evidence="user confirmed plan",
            last_outcome_at_turn=-1,
        )


def test_outcome_with_evidence_and_turn_is_accepted() -> None:
    entry = CommitmentLifecycleEntry(
        record_id="rec-2",
        advocacy_state=AdvocacyState.PROPOSED,
        alignment_state=AlignmentState.AGREE,
        followup_policy=FollowupPolicy.GENTLE_CHECKIN,
        last_outcome=CommitmentOutcomeKind.COMPLETED,
        last_outcome_evidence="user said yes, proceeding.",
        last_outcome_at_turn=4,
    )
    assert entry.last_outcome is CommitmentOutcomeKind.COMPLETED


def test_no_outcome_entry_is_accepted_with_defaults() -> None:
    """Fresh commitment records (no outcome yet) must construct cleanly."""
    entry = CommitmentLifecycleEntry(
        record_id="rec-3",
        advocacy_state=AdvocacyState.NOT_READY,
        alignment_state=AlignmentState.UNKNOWN,
    )
    assert entry.last_outcome is None
    assert entry.last_outcome_evidence == ""
    assert entry.last_outcome_at_turn == -1
    assert entry.followup_policy is FollowupPolicy.GENTLE_CHECKIN


# ---------------------------------------------------------------------------
# Red line A: no keyword-driven alignment extraction
# ---------------------------------------------------------------------------


_KEYWORD_EXTRACTION_PATTERNS = [
    # Direct keyword-in-user_text gates
    re.compile(r'if\s+["\'](?:agree|reject|modify|yes|no|\u540c\u610f|\u62d2\u7edd|\u597d\u7684)["\']\s+in\s+user_input'),
    re.compile(r'if\s+["\'](?:agree|reject|modify|yes|no|\u540c\u610f|\u62d2\u7edd|\u597d\u7684)["\']\s+in\s+user_text'),
    # LLM-keyword fallback that maps to alignment_state
    re.compile(r'alignment_state\s*=\s*["\']agree["\'].*#\s*keyword', re.IGNORECASE),
]


def _package_src_dirs() -> list[pathlib.Path]:
    """Return the source tree roots we want to lint for keyword extraction.

    We scope to ``packages/vz-cognition/src`` (where the commitment owner
    lives) and ``packages/lifeform-expression/src`` (where any alignment
    inference *might* leak in). ``packages/lifeform-core`` is allowed to
    read ``.value`` strings via ``_enum_value`` \u2014 we don't lint it here
    because those are enum value decodes, not alignment inference.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    return [
        repo_root / "packages" / "vz-cognition" / "src",
        repo_root / "packages" / "lifeform-expression" / "src",
    ]


def test_no_keyword_driven_alignment_extraction() -> None:
    offending: list[str] = []
    for src_root in _package_src_dirs():
        if not src_root.is_dir():
            continue
        for py_file in src_root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            # Strip this test file itself if it were ever accidentally
            # placed under those roots (not the case today, but defensive).
            if py_file.resolve() == pathlib.Path(__file__).resolve():
                continue
            for pattern in _KEYWORD_EXTRACTION_PATTERNS:
                for match in pattern.finditer(text):
                    offending.append(f"{py_file}: {match.group(0)!r}")
    assert not offending, (
        "Found keyword-driven alignment extraction in:\n"
        + "\n".join(offending)
        + "\nAlignment MUST come from typed SemanticProposal operations "
        "(REVISE / COMPLETE / BLOCK), not regex over user text. See "
        ".cursor/rules/no-keyword-matching-hacks.mdc."
    )


# ---------------------------------------------------------------------------
# FollowupPolicy is a typed enum, not a free-form string
# ---------------------------------------------------------------------------


def test_followup_policy_values_are_exhaustive() -> None:
    """If someone adds a FollowupPolicy value, this test reminds them to
    extend ``_COMMITMENT_FOLLOWUP_POLICY_TRANSITIONS`` and the follow-up
    manager's dispatch. Fails loudly on missed values.
    """
    known = {FollowupPolicy.GENTLE_CHECKIN, FollowupPolicy.DEFER_ONLY}
    assert set(FollowupPolicy) == known, (
        "FollowupPolicy set changed; update "
        "_COMMITMENT_FOLLOWUP_POLICY_TRANSITIONS and "
        "FollowupManager.ingest_commitment_lifecycle to cover the new "
        "value, then update this test."
    )


def test_commitment_outcome_kind_values_are_exhaustive() -> None:
    known = {
        CommitmentOutcomeKind.PROGRESSED,
        CommitmentOutcomeKind.COMPLETED,
        CommitmentOutcomeKind.STALLED,
        CommitmentOutcomeKind.REJECTED,
        CommitmentOutcomeKind.FOLLOWUP_NO_RESPONSE,
    }
    assert set(CommitmentOutcomeKind) == known, (
        "CommitmentOutcomeKind set changed; update "
        "_COMMITMENT_OUTCOME_TRANSITIONS and reflection writeback to "
        "cover the new value, then update this test."
    )
