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

from volvence_zero.conditioning_bank_contracts import ConditioningBankSnapshot
from volvence_zero.dialogue_trace import ConditioningLineage


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
    )


__all__ = ["bank_fingerprint", "build_conditioning_lineage"]
