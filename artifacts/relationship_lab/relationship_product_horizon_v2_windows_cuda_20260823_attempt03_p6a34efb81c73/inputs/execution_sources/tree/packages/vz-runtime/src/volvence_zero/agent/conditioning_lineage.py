"""Per-turn conditioning lineage: which banks actually shaped this action.

This is the right-hand side of the external-outcome attribution join. An
outcome reported later carries ``(session_scope, action_turn_index)``; the
lineage recorded here is what turns that pair into "these banks, at these
versions, produced that response".

The single rule that makes the attribution meaningful: **only banks that
actually influenced the output are recorded**. A bank that was published but
gated off -- SHADOW wiring, cold start, revoked consent, zero confidence --
did not shape anything, and recording it would assign credit for an outcome
to state that had no causal path to it. That is exactly the false-positive
the negative controls in the ablation plan are designed to catch, so it must
not be manufactured by the lineage writer itself.

Delivery form does not matter for attribution: a bank read as a residual
bias and a bank rendered into the system prompt both influenced the
response, so both are recorded. Which form was used is recoverable from the
turn's audit tags, not from the bank set.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

from volvence_zero.conditioning_bank_contracts import (
    ConditioningBankSnapshot,
    ConditioningLineageRef,
)
from volvence_zero.dialogue_trace import (
    ConditioningLineage,
    DialogueExternalOutcomeEvidence,
    DialogueTraceSnapshot,
)


def bank_fingerprint(bank: ConditioningBankSnapshot) -> str:
    """Stable short digest identifying this bank's exact state version.

    Derived from the contract's ``fingerprint_parts`` so the lineage digest
    and the KV cache key agree by construction: if two turns share a
    fingerprint they were conditioned on identical bank state, and if a
    consent version or revocation changed, both the cache entry and the
    lineage entry change together.
    """

    payload = "\x1f".join(bank.fingerprint_parts).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def build_conditioning_lineage(
    *,
    session_scope: str,
    banks: Sequence[ConditioningBankSnapshot],
    state_encoder_version: str = "",
    prefix_generator_version: str = "",
    router_version: str = "",
    router_scores: tuple[tuple[str, float], ...] = (),
    shadow_router_version: str = "",
    shadow_router_scores: tuple[tuple[str, float], ...] = (),
) -> ConditioningLineage | None:
    """Record the banks that influenced this turn, or ``None`` if none did.

    Returning ``None`` rather than an empty lineage is deliberate: it keeps
    "no bank was live" distinguishable from "lineage recording was not
    wired", which matters when auditing whether a null result came from an
    inert system or from a missing instrumentation path.
    """

    if not session_scope:
        return None
    influencing = [bank for bank in banks if bank.is_injectable]
    if not influencing:
        return None
    # Sorted so the same bank set always produces the same recorded order;
    # an unstable order would make two identical turns look different to any
    # downstream grouping or cache key built from this field.
    influencing.sort(key=lambda bank: bank.bank_type.value)
    return ConditioningLineage(
        session_scope=session_scope,
        selected_bank_set=tuple(bank.bank_type.value for bank in influencing),
        bank_fingerprints=tuple(
            (bank.bank_type.value, bank_fingerprint(bank)) for bank in influencing
        ),
        state_encoder_version=state_encoder_version,
        prefix_generator_version=prefix_generator_version,
        router_version=router_version,
        router_scores=router_scores,
        shadow_router_version=shadow_router_version,
        shadow_router_scores=shadow_router_scores,
    )


def build_conditioning_lineage_ref(
    *,
    session_scope: str,
    banks: Sequence[ConditioningBankSnapshot],
    state_encoder_version: str = "",
    prefix_generator_version: str = "",
    router_version: str = "",
    carrier: str = "",
    delivery_phase: str = "substrate-capture",
) -> ConditioningLineageRef | None:
    """Public snapshot-safe lineage reference for conditioned model surfaces."""

    if not session_scope:
        return None
    influencing = [bank for bank in banks if bank.is_injectable]
    if not influencing:
        return None
    influencing.sort(key=lambda bank: bank.bank_type.value)
    return ConditioningLineageRef(
        session_scope=session_scope,
        selected_bank_set=tuple(bank.bank_type.value for bank in influencing),
        bank_fingerprints=tuple(
            (bank.bank_type.value, bank_fingerprint(bank)) for bank in influencing
        ),
        state_encoder_version=state_encoder_version,
        prefix_generator_version=prefix_generator_version,
        router_version=router_version,
        carrier=carrier,
        delivery_phase=delivery_phase,
        description=(
            "State KV conditioning lineage for public substrate/temporal "
            "snapshot propagation."
        ),
    )


def resolve_conditioning_lineage_for_outcome(
    *,
    evidence: DialogueExternalOutcomeEvidence,
    trace_snapshot: DialogueTraceSnapshot,
) -> ConditioningLineage | None:
    """Join an external outcome back to the bank set that shaped its action.

    This is the attribution join the two contracts were built for: the
    outcome carries ``(session_scope, action_turn_index)``, the dialogue
    trace carries per-turn ``conditioning_lineage``, and this function is
    the single place the two halves meet.

    Returns ``None`` in exactly three documented cases, each of which means
    "counted but not attributable" rather than an error:

    * the evidence is not attributable (missing session scope or declared
      action turn) -- guessing a turn would credit the wrong bank set;
    * no trace exists for the declared turn (trimmed from the bounded
      store, or never recorded) -- falling back to a nearby turn would
      misattribute, so the honest answer is "cannot resolve";
    * the turn exists but recorded no lineage -- no bank influenced that
      action, which is a meaningful negative result for credit assignment.

    A session-scope mismatch between the evidence and the located turn's
    lineage raises instead of returning ``None``: the caller joined
    evidence against the wrong session's trace snapshot, and silently
    reporting "unattributable" would hide that wiring bug.
    """

    if not evidence.is_attributable:
        return None
    target = next(
        (
            trace
            for trace in reversed(trace_snapshot.traces)
            if trace.turn_index == evidence.action_turn_index
        ),
        None,
    )
    if target is None or target.conditioning_lineage is None:
        return None
    lineage = target.conditioning_lineage
    if lineage.session_scope != evidence.session_scope:
        raise ValueError(
            "External outcome attribution joined across sessions: evidence "
            f"session_scope={evidence.session_scope!r} does not match the "
            f"lineage session_scope={lineage.session_scope!r} recorded for "
            f"turn {evidence.action_turn_index}. The caller resolved against "
            "the wrong session's dialogue trace snapshot."
        )
    return lineage


__all__ = [
    "bank_fingerprint",
    "build_conditioning_lineage",
    "build_conditioning_lineage_ref",
    "resolve_conditioning_lineage_for_outcome",
]
