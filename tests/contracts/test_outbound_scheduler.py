"""W3-B: ``dlaas-platform-ops`` OutboundScheduler contract tests.

The scheduler is a pure decision producer (mirrors ``handoff_trigger``
in shape) — these tests exercise it against a fake brain session that
only exposes ``runner.active_snapshots()`` so the suite never spins
up the kernel. The point is to lock down:

1. Cadence rules (``min_gap_seconds`` / ``max_followups`` /
   ``min_age_turns`` / unresolved-tension block / eligible funnel
   stages) all gate independently and report typed reasons.
2. The ``OutboundFollowupRequest`` produced when ``should_send`` is
   True carries the snapshot view that authorised it (so the
   dispatch path can audit-trail "this proactive turn fired because
   relationship_state was at funnel_stage=nurturing, age=5 turns").
3. The scheduler reads only the W2-A enriched snapshot fields
   (``funnel_stage``, ``cumulative_trust_level``,
   ``relationship_age_turns``, ``relational_tensions``) — never raw
   user-facing chat text and never kernel internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dlaas_platform_ops import (
    OutboundCadenceConfig,
    OutboundDecision,
    OutboundFollowupRequest,
    OutboundLedger,
    OutboundScheduler,
    evaluate_outbound,
)


# ---------------------------------------------------------------------------
# Fakes mirroring the platform's snapshot read path
# ---------------------------------------------------------------------------


@dataclass
class _FakeRelationshipState:
    """Mock of the W2-A enriched ``RelationshipStateSnapshot`` view.

    Only the fields the scheduler consumes are populated. Keeping
    this typed (instead of a dict) ensures the test fails loudly if
    the scheduler ever reaches for a field outside the agreed
    contract surface.
    """

    funnel_stage: str = "nurturing"
    cumulative_trust_level: float = 0.45
    relationship_age_turns: int = 5
    relational_tensions: tuple[Any, ...] = field(default_factory=tuple)


@dataclass
class _FakeSnapshotEnvelope:
    value: Any


@dataclass
class _FakeRunner:
    snapshots: dict[str, Any] = field(default_factory=dict)

    def active_snapshots(self) -> dict[str, Any]:
        return self.snapshots


@dataclass
class _FakeBrainSession:
    runner: _FakeRunner = field(default_factory=_FakeRunner)


class _FakeSession:
    """Minimal stand-in: only exposes ``_brain_session.runner``.

    Mirrors the surface the scheduler reaches for in production
    (which is identical to ``handoff_trigger``'s read path).
    """

    def __init__(self, relationship: _FakeRelationshipState | None) -> None:
        runner = _FakeRunner()
        if relationship is not None:
            runner.snapshots["relationship_state"] = _FakeSnapshotEnvelope(
                value=relationship
            )
        self._brain_session = _FakeBrainSession(runner=runner)


def _build_session(
    *,
    funnel_stage: str = "nurturing",
    cumulative_trust_level: float = 0.45,
    relationship_age_turns: int = 5,
    tension_count: int = 0,
) -> _FakeSession:
    state = _FakeRelationshipState(
        funnel_stage=funnel_stage,
        cumulative_trust_level=cumulative_trust_level,
        relationship_age_turns=relationship_age_turns,
        relational_tensions=tuple(object() for _ in range(tension_count)),
    )
    return _FakeSession(state)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_evaluate_outbound_returns_request_when_all_conditions_pass() -> None:
    session = _build_session()
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=OutboundCadenceConfig(),
        ledger=OutboundLedger(),
        now_ts_ms=1_000_000,
        followup_brief="Day 5 outbound message",
        followup_evidence_ref="scheduler:nurturing:5",
    )
    assert decision.should_send is True
    assert decision.reason == "ok"
    assert isinstance(decision.request, OutboundFollowupRequest)
    request = decision.request
    assert request.ai_id == "ai_growth"
    assert request.end_user_ref == "user_mom"
    assert request.funnel_stage_at_schedule == "nurturing"
    assert request.relationship_age_turns_at_schedule == 5
    assert request.cumulative_trust_at_schedule == pytest.approx(0.45)
    assert request.followup_brief == "Day 5 outbound message"
    assert request.followup_evidence_ref == "scheduler:nurturing:5"
    assert request.scheduled_at_ts_ms == 1_000_000


def test_outbound_request_renders_dispatch_compatible_extras() -> None:
    """The request's ``to_envelope_extra`` matches the dispatcher contract.

    The dispatcher reads ``followup_brief`` and
    ``followup_evidence_ref`` from ``structured_context`` — keeping
    these aligned is the W3-B handshake invariant. If the dispatcher
    ever renames a key, this test fails loud and the rename has to
    be coordinated.
    """
    request = OutboundFollowupRequest(
        ai_id="ai_growth",
        end_user_ref="user_mom",
        funnel_stage_at_schedule="nurturing",
        relationship_age_turns_at_schedule=5,
        cumulative_trust_at_schedule=0.45,
        followup_brief="Day 5 outbound message",
        followup_evidence_ref="scheduler:nurturing:5",
        scheduled_at_ts_ms=1_000_000,
    )
    extra = request.to_envelope_extra()
    assert extra["followup_brief"] == "Day 5 outbound message"
    assert extra["followup_evidence_ref"] == "scheduler:nurturing:5"


# ---------------------------------------------------------------------------
# Cadence gating
# ---------------------------------------------------------------------------


def test_no_snapshot_blocks_with_typed_reason() -> None:
    session = _FakeSession(None)
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=OutboundCadenceConfig(),
        ledger=OutboundLedger(),
        now_ts_ms=1_000_000,
        followup_brief="x",
    )
    assert decision.should_send is False
    assert decision.reason == "no_snapshot"
    assert decision.request is None


def test_max_followups_reached_blocks_further_outbound() -> None:
    session = _build_session()
    cadence = OutboundCadenceConfig(max_followups=1)
    ledger = OutboundLedger()
    request = OutboundFollowupRequest(
        ai_id="ai_growth",
        end_user_ref="user_mom",
        funnel_stage_at_schedule="nurturing",
        relationship_age_turns_at_schedule=5,
        cumulative_trust_at_schedule=0.45,
        followup_brief="prior",
        followup_evidence_ref="prior",
        scheduled_at_ts_ms=900_000,
    )
    ledger.record(
        ai_id="ai_growth",
        end_user_ref="user_mom",
        submitted_at_ts_ms=900_000,
        request=request,
    )
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=cadence,
        ledger=ledger,
        now_ts_ms=10_000_000_000,  # arbitrarily far in the future
        followup_brief="next",
    )
    assert decision.should_send is False
    assert decision.reason == "max_followups_reached"
    assert decision.trigger_details["sent_count"] == 1
    assert decision.trigger_details["max_followups"] == 1


def test_min_gap_seconds_blocks_when_recent_send_exists() -> None:
    session = _build_session()
    cadence = OutboundCadenceConfig(min_gap_seconds=3600)  # 1 hour
    ledger = OutboundLedger()
    prior = OutboundFollowupRequest(
        ai_id="ai_growth",
        end_user_ref="user_mom",
        funnel_stage_at_schedule="nurturing",
        relationship_age_turns_at_schedule=5,
        cumulative_trust_at_schedule=0.45,
        followup_brief="prior",
        followup_evidence_ref="prior",
        scheduled_at_ts_ms=1_000_000_000,
    )
    ledger.record(
        ai_id="ai_growth",
        end_user_ref="user_mom",
        submitted_at_ts_ms=1_000_000_000,
        request=prior,
    )
    # 30 minutes later -> still inside the 1h gap.
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=cadence,
        ledger=ledger,
        now_ts_ms=1_000_000_000 + 30 * 60 * 1000,
        followup_brief="next",
    )
    assert decision.should_send is False
    assert decision.reason == "below_min_gap"


def test_min_age_turns_blocks_premature_outbound() -> None:
    session = _build_session(relationship_age_turns=0)
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=OutboundCadenceConfig(min_age_turns=2),
        ledger=OutboundLedger(),
        now_ts_ms=1_000_000,
        followup_brief="x",
    )
    assert decision.should_send is False
    assert decision.reason == "below_min_age"


def test_unresolved_tension_blocks_when_configured() -> None:
    session = _build_session(tension_count=1)
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=OutboundCadenceConfig(block_on_unresolved_tension=True),
        ledger=OutboundLedger(),
        now_ts_ms=1_000_000,
        followup_brief="x",
    )
    assert decision.should_send is False
    assert decision.reason == "unresolved_tension"


def test_unresolved_tension_does_not_block_when_disabled() -> None:
    session = _build_session(tension_count=1)
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=OutboundCadenceConfig(block_on_unresolved_tension=False),
        ledger=OutboundLedger(),
        now_ts_ms=1_000_000,
        followup_brief="x",
    )
    assert decision.should_send is True
    assert decision.reason == "ok"


def test_funnel_stage_outside_eligible_set_blocks() -> None:
    session = _build_session(funnel_stage="prospecting")
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=OutboundCadenceConfig(),  # default does NOT include prospecting
        ledger=OutboundLedger(),
        now_ts_ms=1_000_000,
        followup_brief="x",
    )
    assert decision.should_send is False
    assert decision.reason == "stage_not_eligible"


def test_empty_followup_brief_blocks_at_edge() -> None:
    session = _build_session()
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=OutboundCadenceConfig(),
        ledger=OutboundLedger(),
        now_ts_ms=1_000_000,
        followup_brief="   ",
    )
    assert decision.should_send is False
    assert decision.reason == "empty_followup_brief"


# ---------------------------------------------------------------------------
# OutboundScheduler facade
# ---------------------------------------------------------------------------


def test_outbound_scheduler_facade_records_submission() -> None:
    """The facade chains evaluate -> record cleanly.

    A call to ``record_submission`` should make the next ``evaluate``
    refuse to fire again until the cadence window opens, even with
    no time advance (max_followups=1 case).
    """
    session = _build_session()
    scheduler = OutboundScheduler(
        cadence=OutboundCadenceConfig(max_followups=1)
    )
    first = scheduler.evaluate(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        now_ts_ms=1_000_000,
        followup_brief="first",
    )
    assert first.should_send is True
    scheduler.record_submission(request=first.request)

    second = scheduler.evaluate(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        now_ts_ms=2_000_000,
        followup_brief="second",
    )
    assert second.should_send is False
    assert second.reason == "max_followups_reached"


def test_outbound_decision_to_json_round_trips() -> None:
    """``OutboundDecision.to_json`` is the SSE / audit log payload.

    Lock the keys so a downstream consumer (admin SSE / ledger
    persistence) can rely on the shape staying stable.
    """
    session = _build_session()
    decision = evaluate_outbound(
        session=session,
        ai_id="ai_growth",
        end_user_ref="user_mom",
        cadence=OutboundCadenceConfig(),
        ledger=OutboundLedger(),
        now_ts_ms=1_000_000,
        followup_brief="x",
    )
    payload = decision.to_json()
    assert payload["should_send"] is True
    assert payload["reason"] == "ok"
    assert payload["request"]["ai_id"] == "ai_growth"
    assert payload["request"]["funnel_stage_at_schedule"] == "nurturing"
    assert payload["trigger_details"]["funnel_stage"] == "nurturing"


# ---------------------------------------------------------------------------
# Cadence config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expected_msg_fragment",
    [
        ({"min_gap_seconds": -1}, "min_gap_seconds"),
        ({"max_followups": -1}, "max_followups"),
        ({"min_age_turns": -1}, "min_age_turns"),
        ({"eligible_funnel_stages": ()}, "eligible_funnel_stages"),
    ],
)
def test_cadence_config_rejects_invalid_inputs(
    kwargs: dict, expected_msg_fragment: str
) -> None:
    with pytest.raises(ValueError, match=expected_msg_fragment):
        OutboundCadenceConfig(**kwargs)
