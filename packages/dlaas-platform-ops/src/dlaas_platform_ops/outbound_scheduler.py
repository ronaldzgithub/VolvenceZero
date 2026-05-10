"""Outbound followup scheduler (W3-B).

LTV / private-domain operations verticals (e.g.
``lifeform-domain-growth-advisor``) need to *proactively* reach out
to a user across multiple days — not because the user said
something to react to, but because the relationship-state snapshot
indicates the next planned step in a multi-day playbook is due.

This module is a **pure decision producer**, mirroring the design of
:func:`dlaas_platform_ops.handoff_trigger.evaluate_session`:

* It reads the kernel's ``relationship_state`` snapshot through the
  public ``BrainSession.runner.active_snapshots()`` surface.
* It applies a typed cadence config (per ai_id × end_user_ref) to
  the funnel stage / trust / age signals from the snapshot.
* It returns a :class:`OutboundDecision` describing whether (and
  why) a followup should be sent, plus the typed
  :class:`OutboundFollowupRequest` that callers can dispatch as
  ``interaction_type=command`` /
  ``command_name=initiate_proactive_followup``.

The module deliberately does NOT:

* Schedule wall-clock time itself (no asyncio.sleep / cron loop).
  Time evaluation is the caller's job — they decide when to call
  :func:`evaluate_outbound`. This keeps the kernel-tier R8
  invariant clean and makes the policy itself trivial to test.
* Mutate any kernel state — only the public snapshot is read.
* Inspect raw user-facing chat text — only typed snapshot fields
  (``funnel_stage``, ``cumulative_trust_level``,
  ``relationship_age_turns``, ``relational_tensions`` count,
  ``boundary_consent.external_action_blocked``).
* Persist anything to disk — the in-memory ledger is per-process.
  A future packet can promote it to a registry-backed store; the
  decision-function shape stays the same.

Why the scheduler lives in ``dlaas-platform-ops`` and not in the
vertical wheel: outbound delivery is a platform-tier concern (rate
limits, auditability, operator override). The vertical supplies the
*template* (what to say); the platform decides *whether* to send.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace as _replace
from typing import Any


# ---------------------------------------------------------------------------
# Cadence configuration
# ---------------------------------------------------------------------------


# Default cooldown values are chosen to match the LTV growth-advisor
# vertical's 7-day playbook: at most one outbound message per 24h, no
# more than seven per relationship in the first two weeks. Operators
# can override these per ai_id × end_user_ref via :class:`OutboundCadenceConfig`.
_DEFAULT_MIN_GAP_SECONDS: int = 24 * 60 * 60
_DEFAULT_MAX_FOLLOWUPS: int = 7
_DEFAULT_MIN_AGE_TURNS_FOR_FOLLOWUP: int = 1


@dataclass(frozen=True)
class OutboundCadenceConfig:
    """Per (ai_id, end_user_ref) cadence policy.

    Attributes:
        min_gap_seconds: minimum wall-clock gap between two outbound
            messages to the same user. Default 24h.
        max_followups: hard upper bound on the number of outbound
            messages this policy will authorise across the
            relationship's lifetime. Default 7 (matches the 7-day
            growth-advisor playbook).
        eligible_funnel_stages: tuple of ``relationship_state.funnel_stage``
            labels at which an outbound followup is allowed. Default
            covers the early-to-mid relationship range; converting /
            repurchasing stages are excluded by default because the
            playbook there switches to inbound-only behaviour.
        min_age_turns: minimum ``relationship_age_turns`` before any
            followup is allowed. Default 1 (no Day-0 outbound).
        block_on_unresolved_tension: if True, no outbound is sent
            while ``relational_tensions`` is non-empty. Default True;
            forces repair-first.
    """

    min_gap_seconds: int = _DEFAULT_MIN_GAP_SECONDS
    max_followups: int = _DEFAULT_MAX_FOLLOWUPS
    eligible_funnel_stages: tuple[str, ...] = (
        "discovery",
        "nurturing",
        "recommending",
    )
    min_age_turns: int = _DEFAULT_MIN_AGE_TURNS_FOR_FOLLOWUP
    block_on_unresolved_tension: bool = True

    def __post_init__(self) -> None:
        if self.min_gap_seconds < 0:
            raise ValueError(
                f"OutboundCadenceConfig.min_gap_seconds must be non-negative, "
                f"got {self.min_gap_seconds!r}"
            )
        if self.max_followups < 0:
            raise ValueError(
                f"OutboundCadenceConfig.max_followups must be non-negative, "
                f"got {self.max_followups!r}"
            )
        if self.min_age_turns < 0:
            raise ValueError(
                f"OutboundCadenceConfig.min_age_turns must be non-negative, "
                f"got {self.min_age_turns!r}"
            )
        if not self.eligible_funnel_stages:
            raise ValueError(
                "OutboundCadenceConfig.eligible_funnel_stages must be non-empty"
            )


# ---------------------------------------------------------------------------
# Decision + request shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutboundFollowupRequest:
    """Typed proactive-followup request, ready for DLaaS dispatch.

    The request mirrors the wire shape that
    ``interaction_type=command`` expects:

    * ``human_brief="initiate_proactive_followup"`` (the typed command
      name; not free-form text the lifeform parses).
    * ``structured_context.followup_brief`` carries the message the
      lifeform should send, *generated by the vertical's playbook
      template*, not inferred from chat text.
    * ``structured_context.followup_evidence_ref`` lets the audit log
      tie the outbound back to the snapshot that authorised it.
    """

    ai_id: str
    end_user_ref: str
    funnel_stage_at_schedule: str
    relationship_age_turns_at_schedule: int
    cumulative_trust_at_schedule: float
    followup_brief: str
    followup_evidence_ref: str
    scheduled_at_ts_ms: int

    def __post_init__(self) -> None:
        if not self.ai_id.strip():
            raise ValueError("OutboundFollowupRequest.ai_id must be non-empty")
        if not self.end_user_ref.strip():
            raise ValueError(
                "OutboundFollowupRequest.end_user_ref must be non-empty"
            )
        if not self.followup_brief.strip():
            raise ValueError(
                "OutboundFollowupRequest.followup_brief must be non-empty"
            )
        if self.relationship_age_turns_at_schedule < 0:
            raise ValueError(
                "OutboundFollowupRequest.relationship_age_turns_at_schedule "
                "must be non-negative"
            )
        if self.scheduled_at_ts_ms < 0:
            raise ValueError(
                "OutboundFollowupRequest.scheduled_at_ts_ms must be non-negative"
            )

    def to_envelope_extra(self) -> dict[str, Any]:
        """Render the typed extras the dispatcher reads from
        ``structured_context``.

        Mirrors the contract documented on
        :class:`dlaas_platform_contracts.dispatch_vocab.CommandName`
        for ``INITIATE_PROACTIVE_FOLLOWUP`` (W3-B).
        """
        return {
            "followup_brief": self.followup_brief,
            "followup_evidence_ref": self.followup_evidence_ref,
        }


@dataclass(frozen=True)
class OutboundDecision:
    """Outcome of evaluating outbound cadence against a snapshot view.

    Stays parallel to :class:`HandoffDecision` so the calling
    pattern is familiar: a typed reason string for telemetry and an
    optional typed request that the caller can dispatch.
    """

    should_send: bool
    reason: str
    request: OutboundFollowupRequest | None = None
    trigger_details: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "should_send": self.should_send,
            "reason": self.reason,
            "request": (
                {
                    "ai_id": self.request.ai_id,
                    "end_user_ref": self.request.end_user_ref,
                    "funnel_stage_at_schedule": (
                        self.request.funnel_stage_at_schedule
                    ),
                    "relationship_age_turns_at_schedule": (
                        self.request.relationship_age_turns_at_schedule
                    ),
                    "cumulative_trust_at_schedule": (
                        self.request.cumulative_trust_at_schedule
                    ),
                    "followup_brief": self.request.followup_brief,
                    "followup_evidence_ref": self.request.followup_evidence_ref,
                    "scheduled_at_ts_ms": self.request.scheduled_at_ts_ms,
                }
                if self.request is not None
                else None
            ),
            "trigger_details": dict(self.trigger_details),
        }


# ---------------------------------------------------------------------------
# In-memory ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OutboundLedgerEntry:
    """One submitted outbound request, recorded for cadence checks."""

    ai_id: str
    end_user_ref: str
    submitted_at_ts_ms: int
    request: OutboundFollowupRequest


class OutboundLedger:
    """Per-process record of submitted outbound followups.

    Single-process v0 store. The decision function reads it to honour
    ``min_gap_seconds`` / ``max_followups``. Promoting this to a
    registry-backed store is a separate packet — the decision shape
    stays the same.
    """

    def __init__(self) -> None:
        self._entries: list[_OutboundLedgerEntry] = []

    def record(
        self,
        *,
        ai_id: str,
        end_user_ref: str,
        submitted_at_ts_ms: int,
        request: OutboundFollowupRequest,
    ) -> None:
        if not ai_id.strip() or not end_user_ref.strip():
            raise ValueError(
                "OutboundLedger.record requires non-empty ai_id and end_user_ref"
            )
        if submitted_at_ts_ms < 0:
            raise ValueError(
                "OutboundLedger.record submitted_at_ts_ms must be non-negative"
            )
        self._entries.append(
            _OutboundLedgerEntry(
                ai_id=ai_id,
                end_user_ref=end_user_ref,
                submitted_at_ts_ms=submitted_at_ts_ms,
                request=request,
            )
        )

    def entries_for(
        self, *, ai_id: str, end_user_ref: str
    ) -> tuple[_OutboundLedgerEntry, ...]:
        return tuple(
            entry
            for entry in self._entries
            if entry.ai_id == ai_id and entry.end_user_ref == end_user_ref
        )

    def last_entry_for(
        self, *, ai_id: str, end_user_ref: str
    ) -> _OutboundLedgerEntry | None:
        bucket = self.entries_for(ai_id=ai_id, end_user_ref=end_user_ref)
        if not bucket:
            return None
        return max(bucket, key=lambda entry: entry.submitted_at_ts_ms)

    def count_for(self, *, ai_id: str, end_user_ref: str) -> int:
        return len(self.entries_for(ai_id=ai_id, end_user_ref=end_user_ref))

    def clear(self) -> None:
        """Reset the ledger. Test-only helper."""
        self._entries.clear()


# ---------------------------------------------------------------------------
# Snapshot reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RelationshipSnapshotView:
    """Subset of relationship_state fields the scheduler consumes."""

    funnel_stage: str
    cumulative_trust_level: float
    relationship_age_turns: int
    unresolved_tension_count: int


def _read_relationship_snapshot(session: Any) -> _RelationshipSnapshotView | None:
    """Read the W2-A enriched ``relationship_state`` snapshot.

    Mirrors :func:`handoff_trigger._read_rupture_snapshot`: never
    imports kernel internals; pulls fields through the public
    ``runner.active_snapshots()`` mapping by name.
    """
    brain_session = getattr(session, "_brain_session", None)
    if brain_session is None:
        return None
    runner = getattr(brain_session, "runner", None)
    if runner is None:
        return None
    try:
        snapshots = runner.active_snapshots()
    except AttributeError:
        return None
    if not isinstance(snapshots, Mapping):
        return None
    snap = snapshots.get("relationship_state")
    if snap is None:
        return None
    value = getattr(snap, "value", snap)
    funnel_stage = getattr(value, "funnel_stage", "unknown")
    cumulative = getattr(value, "cumulative_trust_level", None)
    age_turns = getattr(value, "relationship_age_turns", None)
    tensions = getattr(value, "relational_tensions", ())
    if cumulative is None or age_turns is None:
        # The W2-A enriched fields default to non-None values when the
        # owner is wired ACTIVE; missing here means the owner is not
        # publishing a new-shape snapshot yet (e.g. a stale fixture).
        # Return None so the caller falls through to "below_threshold".
        return None
    try:
        unresolved = len(tensions)
    except TypeError:
        unresolved = 0
    return _RelationshipSnapshotView(
        funnel_stage=str(funnel_stage),
        cumulative_trust_level=float(cumulative),
        relationship_age_turns=int(age_turns),
        unresolved_tension_count=int(unresolved),
    )


# ---------------------------------------------------------------------------
# Decision function
# ---------------------------------------------------------------------------


def evaluate_outbound(
    *,
    session: Any,
    ai_id: str,
    end_user_ref: str,
    cadence: OutboundCadenceConfig,
    ledger: OutboundLedger,
    now_ts_ms: int,
    followup_brief: str,
    followup_evidence_ref: str = "",
) -> OutboundDecision:
    """Decide whether to send an outbound followup right now.

    The function is intentionally side-effect free. The caller is
    responsible for:

    1. Calling this at whatever cadence makes sense for their
       deployment (cron / asyncio loop / event-driven).
    2. If ``decision.should_send`` is True, dispatching the request
       through the DLaaS ``interaction_type=command`` /
       ``command_name=initiate_proactive_followup`` path AND
       calling :meth:`OutboundLedger.record` once delivery
       succeeds. The ledger is the cadence book — without recording,
       the next call would re-fire immediately.

    Decision rules (in order; first hit wins):

    * Relationship snapshot unavailable → ``no_snapshot``.
    * Cadence config blocks because of ``max_followups`` reached →
      ``max_followups_reached``.
    * Cadence config blocks because of ``min_gap_seconds`` since last
      outbound → ``below_min_gap``.
    * Cadence config blocks because of ``min_age_turns`` not reached
      → ``below_min_age``.
    * Cadence config blocks because of unresolved tensions and
      ``block_on_unresolved_tension`` is True → ``unresolved_tension``.
    * Funnel stage not in ``eligible_funnel_stages`` → ``stage_not_eligible``.
    * Followup brief empty → ``empty_followup_brief``.
    * Otherwise → ``ok`` and request is emitted.
    """
    if not followup_brief.strip():
        return OutboundDecision(
            should_send=False,
            reason="empty_followup_brief",
            trigger_details={"note": "vertical did not supply a followup_brief"},
        )

    view = _read_relationship_snapshot(session)
    if view is None:
        return OutboundDecision(
            should_send=False,
            reason="no_snapshot",
            trigger_details={
                "note": (
                    "relationship_state snapshot unavailable or pre-W2-A "
                    "shape; cannot evaluate cadence safely"
                )
            },
        )

    sent_count = ledger.count_for(ai_id=ai_id, end_user_ref=end_user_ref)
    if sent_count >= cadence.max_followups:
        return OutboundDecision(
            should_send=False,
            reason="max_followups_reached",
            trigger_details={
                "sent_count": sent_count,
                "max_followups": cadence.max_followups,
            },
        )

    last_entry = ledger.last_entry_for(ai_id=ai_id, end_user_ref=end_user_ref)
    if last_entry is not None:
        elapsed_seconds = (now_ts_ms - last_entry.submitted_at_ts_ms) / 1000.0
        if elapsed_seconds < cadence.min_gap_seconds:
            return OutboundDecision(
                should_send=False,
                reason="below_min_gap",
                trigger_details={
                    "elapsed_seconds": elapsed_seconds,
                    "min_gap_seconds": cadence.min_gap_seconds,
                    "last_submitted_at_ts_ms": last_entry.submitted_at_ts_ms,
                },
            )

    if view.relationship_age_turns < cadence.min_age_turns:
        return OutboundDecision(
            should_send=False,
            reason="below_min_age",
            trigger_details={
                "relationship_age_turns": view.relationship_age_turns,
                "min_age_turns": cadence.min_age_turns,
            },
        )

    if (
        cadence.block_on_unresolved_tension
        and view.unresolved_tension_count > 0
    ):
        return OutboundDecision(
            should_send=False,
            reason="unresolved_tension",
            trigger_details={
                "unresolved_tension_count": view.unresolved_tension_count,
            },
        )

    if view.funnel_stage not in cadence.eligible_funnel_stages:
        return OutboundDecision(
            should_send=False,
            reason="stage_not_eligible",
            trigger_details={
                "funnel_stage": view.funnel_stage,
                "eligible_funnel_stages": list(cadence.eligible_funnel_stages),
            },
        )

    request = OutboundFollowupRequest(
        ai_id=ai_id,
        end_user_ref=end_user_ref,
        funnel_stage_at_schedule=view.funnel_stage,
        relationship_age_turns_at_schedule=view.relationship_age_turns,
        cumulative_trust_at_schedule=view.cumulative_trust_level,
        followup_brief=followup_brief,
        followup_evidence_ref=followup_evidence_ref,
        scheduled_at_ts_ms=now_ts_ms,
    )
    return OutboundDecision(
        should_send=True,
        reason="ok",
        request=request,
        trigger_details={
            "funnel_stage": view.funnel_stage,
            "relationship_age_turns": view.relationship_age_turns,
            "cumulative_trust_level": view.cumulative_trust_level,
        },
    )


# ---------------------------------------------------------------------------
# Convenience: simple scheduler facade
# ---------------------------------------------------------------------------


class OutboundScheduler:
    """Thin convenience wrapper bundling cadence config + ledger.

    For callers that want one object per ai_id / end_user_ref pair.
    The decision logic still lives in :func:`evaluate_outbound`; this
    class just keeps the cadence config + ledger in one place so the
    caller does not have to thread them through every call.
    """

    def __init__(
        self,
        *,
        cadence: OutboundCadenceConfig | None = None,
        ledger: OutboundLedger | None = None,
    ) -> None:
        self._cadence = cadence or OutboundCadenceConfig()
        self._ledger = ledger or OutboundLedger()

    @property
    def cadence(self) -> OutboundCadenceConfig:
        return self._cadence

    @property
    def ledger(self) -> OutboundLedger:
        return self._ledger

    def with_cadence(self, cadence: OutboundCadenceConfig) -> "OutboundScheduler":
        """Return a new scheduler sharing the ledger but using ``cadence``."""
        return OutboundScheduler(cadence=cadence, ledger=self._ledger)

    def evaluate(
        self,
        *,
        session: Any,
        ai_id: str,
        end_user_ref: str,
        now_ts_ms: int,
        followup_brief: str,
        followup_evidence_ref: str = "",
    ) -> OutboundDecision:
        return evaluate_outbound(
            session=session,
            ai_id=ai_id,
            end_user_ref=end_user_ref,
            cadence=self._cadence,
            ledger=self._ledger,
            now_ts_ms=now_ts_ms,
            followup_brief=followup_brief,
            followup_evidence_ref=followup_evidence_ref,
        )

    def record_submission(
        self,
        *,
        request: OutboundFollowupRequest,
        submitted_at_ts_ms: int | None = None,
    ) -> None:
        ts = (
            submitted_at_ts_ms
            if submitted_at_ts_ms is not None
            else request.scheduled_at_ts_ms
        )
        self._ledger.record(
            ai_id=request.ai_id,
            end_user_ref=request.end_user_ref,
            submitted_at_ts_ms=ts,
            request=request,
        )


__all__ = [
    "OutboundCadenceConfig",
    "OutboundDecision",
    "OutboundFollowupRequest",
    "OutboundLedger",
    "OutboundScheduler",
    "evaluate_outbound",
]
