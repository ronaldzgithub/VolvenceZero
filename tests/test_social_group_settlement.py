"""G1 (CP-18) contract tests: group-level PE settlement learning loop.

The R20 group owner now closes its own prediction loop, mirroring the
ToM / common-ground settlement contract:

* GROUP_COMMITMENT_DURABILITY predictions are parked in the session
  ``SocialRecordStore`` and settled next turn against the observed
  joint state (typed-summary semantic similarity — never keywords),
* settled outcomes update a bounded learned per-group durability
  score ([0,1], 0.5 uninformed prior) that becomes the confidence of
  future durability predictions (broken commitments lower it, kept
  ones raise it),
* settled errors are published on ``GroupSnapshot.settled_errors`` and
  forwarded by ``SocialPredictionErrorModule`` without downstream
  reconstruction (SSOT),
* without a record store the owner stays a stateless scaffold.
"""

from __future__ import annotations

import asyncio
from typing import Any

from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.social import GroupModule, SocialRecordStore
from volvence_zero.social.identity import SocialPredictionErrorModule
from volvence_zero.social.record_store import GROUP_DURABILITY_PRIOR
from volvence_zero.social_cognition import (
    GroupIdentity,
    GroupSnapshot,
    SocialPredictionKind,
    SocialPredictionOutcome,
)


def _equality_similarity(left: str, right: str) -> float:
    """Deterministic similarity for settlement tests (same pattern as
    the ToM settlement tests): identical typed summaries confirm,
    anything else disconfirms."""

    return 1.0 if left == right else 0.0


def _store() -> SocialRecordStore:
    return SocialRecordStore(similarity=_equality_similarity)


_GROUP = GroupIdentity(
    group_id="group:launch",
    member_ids=("alice", "bob", "carol"),
    display_name="Launch group",
    confidence=0.9,
    evidence=("host membership",),
)


def _run_turn(
    *,
    store: SocialRecordStore | None,
    turn_index: int,
    joint_commitments: tuple[str, ...] = ("commitment:ship",),
    joint_attention: tuple[str, ...] = ("launch-plan",),
    group_regime_id: str | None = "problem_solving",
) -> GroupSnapshot:
    module = GroupModule(
        groups=(_GROUP,),
        active_group_id=_GROUP.group_id,
        joint_attention=joint_attention,
        joint_commitments=joint_commitments,
        group_regime_id=group_regime_id,
        record_store=store,
        turn_index=turn_index,
        wiring_level=WiringLevel.ACTIVE,
    )
    return asyncio.run(module.process({})).value


def test_stable_joint_state_confirms_and_raises_durability() -> None:
    store = _store()

    first = _run_turn(store=store, turn_index=1)
    assert first.settled_errors == ()
    assert first.group_durability_score == GROUP_DURABILITY_PRIOR
    assert len(store.pending_group_predictions) == 1

    # Turn 2: identical joint state -> prior prediction CONFIRMED, the
    # learned durability score moves above the prior, and the NEXT
    # prediction carries the higher confidence.
    second = _run_turn(store=store, turn_index=2)
    assert len(second.settled_errors) == 1
    error = second.settled_errors[0]
    assert error.kind is SocialPredictionKind.GROUP_COMMITMENT_DURABILITY
    assert error.outcome is SocialPredictionOutcome.CONFIRMED
    assert error.owner == "GroupModule"
    assert error.scope_id == _GROUP.group_id
    assert second.group_durability_score > GROUP_DURABILITY_PRIOR
    assert second.active_predictions[0].confidence == (
        second.group_durability_score
    )


def test_broken_joint_state_disconfirms_and_lowers_durability() -> None:
    store = _store()
    _run_turn(store=store, turn_index=1)

    # Turn 2: commitments dropped, regime flipped -> the durability
    # prediction is DISCONFIRMED and the learned score falls below the
    # prior (broken commitments are stronger evidence than kept ones).
    second = _run_turn(
        store=store,
        turn_index=2,
        joint_commitments=(),
        joint_attention=("retro", "postmortem", "incident-review"),
        group_regime_id="relational_repair",
    )
    assert len(second.settled_errors) == 1
    assert second.settled_errors[0].outcome is SocialPredictionOutcome.DISCONFIRMED
    assert second.group_durability_score < GROUP_DURABILITY_PRIOR


def test_prediction_confidence_is_bounded_by_group_confidence() -> None:
    store = _store()
    # Drive the durability score up beyond the group-identity
    # confidence bound; the published confidence must stay clamped.
    for turn in range(1, 8):
        snapshot = _run_turn(store=store, turn_index=turn)
    assert store.group_durability_for(_GROUP.group_id) > _GROUP.confidence
    assert snapshot.active_predictions[0].confidence == _GROUP.confidence


def test_stateless_owner_without_store_keeps_scaffold_behavior() -> None:
    first = _run_turn(store=None, turn_index=1)
    second = _run_turn(store=None, turn_index=2)
    assert first.settled_errors == ()
    assert second.settled_errors == ()
    assert second.group_durability_score == GROUP_DURABILITY_PRIOR


def test_reissued_prediction_refreshes_pending_without_duplication() -> None:
    store = _store()
    _run_turn(store=store, turn_index=1)
    _run_turn(store=store, turn_index=2)
    # Settlement consumed the old entry; exactly one refreshed entry
    # (this turn's re-issued prediction) remains parked.
    assert len(store.pending_group_predictions) == 1
    assert store.pending_group_predictions[0].issued_turn == 2


def _groups_envelope(value: GroupSnapshot) -> dict[str, Snapshot[Any]]:
    return {
        "groups": Snapshot(
            slot_name="groups",
            owner="GroupModule",
            version=1,
            timestamp_ms=1,
            value=value,
        )
    }


def test_error_module_forwards_group_settled_errors() -> None:
    store = _store()
    _run_turn(store=store, turn_index=1)
    second = _run_turn(store=store, turn_index=2)
    assert second.settled_errors

    module = SocialPredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    assert "groups" in module.dependencies
    forwarded = asyncio.run(module.process(_groups_envelope(second))).value
    forwarded_ids = {error.error_id for error in forwarded.errors}
    assert {error.error_id for error in second.settled_errors} <= forwarded_ids
