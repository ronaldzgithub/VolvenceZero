"""Bounded credit-driven bank confidence feedback (State KV P5-c).

P5-b made credit records carry which conditioning banks were live for the
rated action (``CreditRecord.conditioning_bank_set``). This module closes
the loop on the owner side: a conditioning bank owner consumes the credit
snapshot from upstream -- the sanctioned snapshot channel, no direct module
calls -- and maintains a small bounded online-fast state that nudges its
published confidence up when actions taken under this bank keep earning
positive PE-derived credit, and down when they keep earning negative credit.

Ownership: the state lives with the conditioning owner (the module holds a
``BankCreditFeedbackState`` and is the only writer). Credit stays a readout
producer; it never reaches into conditioning. The adjustment is a bounded
delta on top of the evidence-derived base confidence, never a replacement
for it, and both are published separately so the drift is auditable and a
rollback to zero is checkable from the snapshot alone.

Both current bank owners (Personal, Relationship) share this one update
rule so the two banks cannot drift apart in how they interpret the same
credit stream.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.credit import CreditSnapshot

# EMA smoothing for the per-turn attributed-credit signal. Small enough
# that one surprising turn cannot swing the bank, large enough that a
# consistent sign accumulates within a session (online-fast timescale).
CREDIT_FEEDBACK_ALPHA = 0.2

# Gain from smoothed credit to confidence delta, and the hard cap on the
# delta itself. The cap is the safety envelope: credit can modulate how
# strongly a bank asserts itself, but can never fabricate confidence for
# an unevidenced bank or fully silence an evidenced one on its own.
CREDIT_FEEDBACK_GAIN = 0.3
CREDIT_FEEDBACK_DELTA_CAP = 0.15


@dataclass
class BankCreditFeedbackState:
    """Owner-held bounded state for one bank's credit feedback.

    Mutable on purpose: this is online-fast controller state, not a
    snapshot. It must be held by a session-lived owner (the runner passes
    it into the per-turn module instance) so the EMA survives module
    reconstruction across turns. ``last_consumed_timestamp_ms`` guards
    against re-counting the same records out of the credit snapshot's
    rolling window on subsequent turns.
    """

    ema: float = 0.0
    last_consumed_timestamp_ms: int = -1
    consumed_record_count: int = 0

    def reset(self) -> None:
        self.ema = 0.0
        self.last_consumed_timestamp_ms = -1
        self.consumed_record_count = 0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def consume_bank_credit_feedback(
    state: BankCreditFeedbackState,
    *,
    bank_name: str,
    credit_snapshot: CreditSnapshot | None,
) -> float:
    """Fold newly attributed credit into the bank's bounded delta.

    Only records whose typed ``conditioning_bank_set`` names this bank are
    consumed -- that is the P5-b attribution contract, no prose parsing.
    Records already consumed (timestamp at or before the watermark) are
    skipped so the rolling owner-published action-lineage window is not
    double-counted.
    A turn with no attributed records leaves the EMA untouched: absence of
    attribution means the bank was not live, which is a routing fact, not
    evidence about the bank's quality.

    Returns the current bounded confidence delta. With ``credit_snapshot``
    of ``None`` (credit module SHADOW/DISABLED or not yet published) the
    prior state simply persists.
    """

    if credit_snapshot is not None:
        records = (
            credit_snapshot.recent_action_lineage_credits
            or credit_snapshot.recent_credits
        )
        fresh = [
            record
            for record in records
            if bank_name in record.conditioning_bank_set
            and record.timestamp_ms > state.last_consumed_timestamp_ms
        ]
        if fresh:
            signal = sum(
                _clamp(record.credit_value, -1.0, 1.0) for record in fresh
            ) / len(fresh)
            state.ema = (
                (1.0 - CREDIT_FEEDBACK_ALPHA) * state.ema
                + CREDIT_FEEDBACK_ALPHA * signal
            )
            state.last_consumed_timestamp_ms = max(
                record.timestamp_ms for record in fresh
            )
            state.consumed_record_count += len(fresh)
    return _clamp(
        CREDIT_FEEDBACK_GAIN * state.ema,
        -CREDIT_FEEDBACK_DELTA_CAP,
        CREDIT_FEEDBACK_DELTA_CAP,
    )


__all__ = [
    "CREDIT_FEEDBACK_ALPHA",
    "CREDIT_FEEDBACK_DELTA_CAP",
    "CREDIT_FEEDBACK_GAIN",
    "BankCreditFeedbackState",
    "consume_bank_credit_feedback",
]
