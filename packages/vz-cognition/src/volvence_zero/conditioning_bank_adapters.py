"""Adapters from per-bank owner snapshots to the generic bank contract.

The generic ``ConditioningBankSnapshot`` is what State KV routing, cache
keying and attribution consume. Individual owners keep publishing their own
typed snapshot -- ``PersonalConditioningSnapshot`` v1 is frozen and stays --
and an adapter projects it into the generic shape at the point of use.

Why an adapter rather than migrating the owner: the v1 snapshot is already
consumed by the substrate residual path, the renderer, and the runtime
evidence map. Rewriting the owner to publish the generic type would change
all three at once with no rollback, whereas an adapter lets the generic path
be wired, exercised, and reverted independently of the v1 path.

Scope is supplied by the caller, not the owner. A cognition owner has no
business knowing tenant or session identity -- that is runtime's -- so the
runtime passes the scope in. This keeps the owner free of session state
while still producing a bank that cannot be cached across users.
"""

from __future__ import annotations

from volvence_zero.conditioning_bank_contracts import (
    CONDITIONING_BANK_SCHEMA_VERSION,
    ConditioningBankReadout,
    ConditioningBankSnapshot,
    ConditioningBankType,
    ConditioningRevocationState,
    ConditioningScope,
    expected_bank_type,
)
from volvence_zero.personal_conditioning_contracts import PersonalConditioningSnapshot

PERSONAL_CONDITIONING_SLOT = "personal_conditioning"

# Fallback when the v1 snapshot carries no boundary_consent source. v1 always
# does in practice (it is a declared dependency), but the generic contract
# requires a non-negative value and silently substituting a plausible version
# would make an ungated bank look consent-gated.
_UNGATED_CONSENT_VERSION = 0


def personal_conditioning_to_bank(
    *,
    snapshot: PersonalConditioningSnapshot,
    scope: ConditioningScope,
    freshness: float = 1.0,
    revocation_state: ConditioningRevocationState = ConditioningRevocationState.ACTIVE,
    event_time_ms: int = 0,
    effective_time_ms: int = 0,
) -> ConditioningBankSnapshot:
    """Project the frozen v1 personal snapshot onto the PERSONAL bank.

    ``freshness`` is the caller's decay readout: the owner republishes every
    turn, so from the owner's point of view a snapshot is always fresh, but
    the *evidence* behind it may be old. Only the runtime knows how long ago
    the underlying owners actually changed, so it supplies the value.

    A ``REVOKED`` state zeroes the readout, confidence and rendered statement
    here rather than expecting every consumer to remember to check. The
    generic contract enforces that invariant, so a revoked bank that still
    carried values would raise instead of quietly influencing generation.
    """

    bank_type = expected_bank_type(PERSONAL_CONDITIONING_SLOT)
    if bank_type is not ConditioningBankType.PERSONAL:  # pragma: no cover - guard
        raise ValueError(
            "personal_conditioning slot must map to the PERSONAL bank; "
            f"registry resolved {bank_type!r}."
        )

    consent_version = next(
        (
            version
            for slot, version in snapshot.source_versions
            if slot == "boundary_consent"
        ),
        _UNGATED_CONSENT_VERSION,
    )

    revoked = revocation_state is ConditioningRevocationState.REVOKED
    inert = revoked or snapshot.is_cold_start
    readout = (
        tuple(0.0 for _ in snapshot.vector_labels)
        if inert
        else tuple(float(value) for value in snapshot.state_vector)
    )

    return ConditioningBankSnapshot(
        schema_version=CONDITIONING_BANK_SCHEMA_VERSION,
        bank_type=ConditioningBankType.PERSONAL,
        scope=scope,
        readout=readout,
        readout_labels=tuple(snapshot.vector_labels),
        source_versions=tuple(snapshot.source_versions),
        source_fingerprint=snapshot.source_fingerprint,
        confidence=0.0 if inert else float(snapshot.confidence),
        freshness=float(freshness),
        consent_version=int(consent_version),
        provenance=f"owner:PersonalConditioningModule/{snapshot.schema_version}",
        revocation_state=revocation_state,
        is_cold_start=snapshot.is_cold_start,
        description=snapshot.description,
        event_time_ms=int(event_time_ms),
        effective_time_ms=int(effective_time_ms),
        rendered_statement="" if inert else snapshot.rendered_statement,
    )


def bank_readout_to_bank(
    *,
    readout: ConditioningBankReadout,
    slot_name: str,
    scope: ConditioningScope,
    freshness: float = 1.0,
    revocation_state: ConditioningRevocationState = ConditioningRevocationState.ACTIVE,
    event_time_ms: int = 0,
    effective_time_ms: int = 0,
) -> ConditioningBankSnapshot:
    """Project a generic scope-free bank readout onto its scoped bank.

    The runtime half of the P4-a split: the owner published everything it
    can know (typed coordinates, lineage, confidence, rendered statement);
    this adapter supplies what only the runtime knows (scope, freshness,
    revocation, event times). ``slot_name`` is the slot the readout was
    consumed from, so a readout wired to the wrong slot fails loudly here
    instead of producing a mislabelled cache key.

    Revocation and cold start zero the readout, confidence and rendered
    statement exactly as the personal adapter does; the scoped contract
    enforces the invariant either way.
    """

    slot_bank_type = expected_bank_type(slot_name)
    if readout.bank_type is not slot_bank_type:
        raise ValueError(
            f"conditioning bank readout published on slot {slot_name!r} must "
            f"carry bank_type {slot_bank_type!r}, got {readout.bank_type!r}."
        )

    consent_version = next(
        (
            version
            for slot, version in readout.source_versions
            if slot == "boundary_consent"
        ),
        _UNGATED_CONSENT_VERSION,
    )

    revoked = revocation_state is ConditioningRevocationState.REVOKED
    inert = revoked or readout.is_cold_start
    values = (
        tuple(0.0 for _ in readout.readout_labels)
        if inert
        else tuple(float(value) for value in readout.readout)
    )

    return ConditioningBankSnapshot(
        schema_version=CONDITIONING_BANK_SCHEMA_VERSION,
        bank_type=readout.bank_type,
        scope=scope,
        readout=values,
        readout_labels=tuple(readout.readout_labels),
        source_versions=tuple(readout.source_versions),
        source_fingerprint=readout.source_fingerprint,
        confidence=0.0 if inert else float(readout.confidence),
        freshness=float(freshness),
        consent_version=int(consent_version),
        provenance=readout.provenance,
        revocation_state=revocation_state,
        is_cold_start=readout.is_cold_start,
        description=readout.description,
        event_time_ms=int(event_time_ms),
        effective_time_ms=int(effective_time_ms),
        rendered_statement="" if inert else readout.rendered_statement,
    )


__all__ = [
    "PERSONAL_CONDITIONING_SLOT",
    "bank_readout_to_bank",
    "personal_conditioning_to_bank",
]
