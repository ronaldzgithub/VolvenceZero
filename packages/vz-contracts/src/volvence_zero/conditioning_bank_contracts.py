"""Generic conditioning-bank contract shared by every State KV bank.

``PersonalConditioningSnapshot`` (``personal_conditioning_contracts``) is
the frozen v1 shape for a single bank. State KV needs the *same* auditable
posture for five more banks (relationship, object/social, environment,
world/domain, task) without minting a bespoke contract per bank, so this
module defines one generic value type plus the scope and bank-type
vocabularies.

Design constraints this shape has to satisfy, and why:

- **One slot, one owner, one bank.** The kernel's ``SlotRegistry`` rejects a
  second owner per slot, but because every bank shares this one class the
  kernel's ``isinstance`` check degenerates to "is some bank". Per-bank
  correctness is therefore the owner's job: each owner module must assert
  that ``bank_type`` matches the slot it publishes on. ``expected_bank_type``
  exists for exactly that cross-check.
- **Typed scope.** v1 carries no scope at all, so nothing in the repo can
  express "which user/session/object this conditioning belongs to". State KV
  cannot key a KV cache, enforce cross-tenant isolation, or join an external
  outcome back to the banks that were live without it.
- **Latent, never factual.** The readout is a bounded numeric vector over
  declared labels. Names, amounts, dates, quotes, and any other precise fact
  stay on the auditable context channel. A bank never carries raw dialogue,
  profile prose, or memory text.
- **Revocation is a first-class state**, not an absence. A revoked bank still
  publishes (so the audit trail shows the revocation), but with a zeroed
  readout, and consumers must refuse to inject it.

Two fields duplicate information available elsewhere, deliberately:

- ``freshness`` is the owner's decay readout in [0, 1]. The authoritative
  ordering signal is still the outer ``Snapshot.timestamp_ms``/``version``;
  freshness exists because a State KV cache needs a staleness signal that
  moves *continuously* while the fingerprint only changes on republication.
  Without it, a bank that has not been updated for a week is indistinguishable
  from one published a second ago.
- ``consent_version`` is also recoverable from ``source_versions``. It is
  promoted to a field because it is a mandatory component of the KV cache key,
  and reconstructing it by searching a tuple at cache-key time is both slower
  and easier to get silently wrong. The invariant below keeps the two in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

CONDITIONING_BANK_SCHEMA_VERSION = "conditioning-bank.v1"

# Slot name that each bank type is required to publish on. The owner passes
# its own slot to ``expected_bank_type`` so a copy-paste error between banks
# fails loudly at publication instead of producing a mislabelled cache key.
CONDITIONING_BANK_SLOTS: dict[str, str] = {
    "personal_conditioning": "PERSONAL",
    "relationship_conditioning": "RELATIONSHIP",
    "object_social_conditioning": "OBJECT_SOCIAL",
    "environment_conditioning": "ENVIRONMENT",
    "world_domain_conditioning": "WORLD_DOMAIN",
    "task_conditioning": "TASK",
}


class ConditioningBankType(Enum):
    """Which state bank a snapshot belongs to.

    Ordered as the design's rollout sequence: personal and relationship
    first (they have existing upstream owners), then object/task, then
    environment/world. Adding a member is a contract change: the bank must
    have a registered owner and a slot entry before it appears here.
    """

    PERSONAL = "personal"
    RELATIONSHIP = "relationship"
    OBJECT_SOCIAL = "object_social"
    ENVIRONMENT = "environment"
    WORLD_DOMAIN = "world_domain"
    TASK = "task"


class ConditioningRevocationState(Enum):
    """Whether this bank may still influence generation.

    ``REVOKED`` is published rather than withheld so the audit trail records
    that a revocation happened on this turn. Consumers must treat it exactly
    like a cold start: no injection, no rendering, no cache reuse.
    """

    ACTIVE = "active"
    REVOKED = "revoked"


@dataclass(frozen=True)
class ConditioningScope:
    """Isolation boundary a conditioning bank belongs to.

    Every field is a caller-supplied opaque identifier; this layer never
    parses them. ``tenant_scope`` and ``user_scope`` are mandatory because
    cross-tenant and cross-user KV reuse must be impossible to express, not
    merely discouraged. The remaining three are optional: a bank may be
    session-wide (no object), or object-scoped without a task.

    ``session_scope`` is the join key for external-outcome attribution. It
    must be stable across a context reset within one session -- otherwise an
    outcome reported after a context boundary cannot be traced back to the
    banks that were live when the action was taken.
    """

    tenant_scope: str
    user_scope: str
    session_scope: str = ""
    object_scope: str = ""
    task_scope: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_scope:
            raise ValueError("ConditioningScope tenant_scope must be non-empty.")
        if not self.user_scope:
            raise ValueError("ConditioningScope user_scope must be non-empty.")

    @property
    def cache_key_parts(self) -> tuple[str, ...]:
        """Scope components in the fixed order used to build a KV cache key.

        Returned as a tuple rather than a joined string so callers cannot
        accidentally create a collision between, say, user ``"a:b"`` with no
        session and user ``"a"`` with session ``"b"``.
        """

        return (
            self.tenant_scope,
            self.user_scope,
            self.session_scope,
            self.object_scope,
            self.task_scope,
        )


def expected_bank_type(slot_name: str) -> ConditioningBankType:
    """Return the bank type a given slot is required to publish.

    Owners call this in ``__post_init__``/``process`` to assert their snapshot
    matches their slot. Raises for an unregistered slot so a new bank cannot
    reach runtime without a contract entry.
    """

    try:
        member = CONDITIONING_BANK_SLOTS[slot_name]
    except KeyError:
        raise ValueError(
            f"{slot_name!r} is not a registered conditioning-bank slot; "
            f"known slots: {sorted(CONDITIONING_BANK_SLOTS)}."
        ) from None
    return ConditioningBankType[member]


@dataclass(frozen=True)
class ConditioningBankSnapshot:
    """One auditable, bounded, revocable state bank.

    The readout carries only typed owner values over declared labels, so a
    substrate consumer that reads it cannot become a second semantic owner:
    it sees numbers and a schema version, never business meaning.

    ``rendered_statement`` is the owner's natural-language form of this same
    readout, used by the text-delivery arm and the distillation teacher. It
    must be derived exclusively from the labelled coordinates, confidence and
    coverage -- never from semantic records or raw text -- so that the text
    and latent delivery paths carry identical information.
    """

    schema_version: str
    bank_type: ConditioningBankType
    scope: ConditioningScope
    readout: tuple[float, ...]
    readout_labels: tuple[str, ...]
    source_versions: tuple[tuple[str, int], ...]
    source_fingerprint: str
    confidence: float
    freshness: float
    consent_version: int
    provenance: str
    revocation_state: ConditioningRevocationState
    is_cold_start: bool
    description: str
    event_time_ms: int = 0
    effective_time_ms: int = 0
    rendered_statement: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != CONDITIONING_BANK_SCHEMA_VERSION:
            raise ValueError(
                "ConditioningBankSnapshot schema_version must be "
                f"{CONDITIONING_BANK_SCHEMA_VERSION!r}."
            )
        if not self.readout_labels:
            raise ValueError(
                "ConditioningBankSnapshot readout_labels must be non-empty: an "
                "unlabelled readout cannot be audited or rendered."
            )
        if len(set(self.readout_labels)) != len(self.readout_labels):
            raise ValueError(
                "ConditioningBankSnapshot readout_labels must be unique."
            )
        if len(self.readout) != len(self.readout_labels):
            raise ValueError(
                "ConditioningBankSnapshot readout length must match readout_labels."
            )
        if any(not 0.0 <= value <= 1.0 for value in self.readout):
            raise ValueError(
                "ConditioningBankSnapshot readout values must be in [0, 1]."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "ConditioningBankSnapshot confidence must be in [0, 1]."
            )
        if not 0.0 <= self.freshness <= 1.0:
            raise ValueError(
                "ConditioningBankSnapshot freshness must be in [0, 1]."
            )
        if self.consent_version < 0:
            raise ValueError(
                "ConditioningBankSnapshot consent_version must be non-negative; "
                "use 0 for a bank that has not yet been consent-gated."
            )
        if not self.source_fingerprint:
            raise ValueError(
                "ConditioningBankSnapshot source_fingerprint must be non-empty."
            )
        if not self.provenance:
            raise ValueError(
                "ConditioningBankSnapshot provenance must be non-empty: an "
                "unattributable bank cannot be reviewed or revoked."
            )
        if not self.description:
            raise ValueError(
                "ConditioningBankSnapshot description must be non-empty "
                "(DATA_CONTRACT: whoever owns the data describes it)."
            )
        if self.event_time_ms < 0 or self.effective_time_ms < 0:
            raise ValueError(
                "ConditioningBankSnapshot times must be non-negative epoch ms."
            )
        source_slots = tuple(slot for slot, _ in self.source_versions)
        if len(set(source_slots)) != len(source_slots):
            raise ValueError(
                "ConditioningBankSnapshot source_versions must name each slot once."
            )
        if any(version < 0 for _, version in self.source_versions):
            raise ValueError(
                "ConditioningBankSnapshot source_versions must be non-negative."
            )
        # consent_version is promoted out of source_versions for cache-key
        # construction; if the bank declares a consent source the two must
        # agree, otherwise the cache key would silently disagree with lineage.
        for slot, version in self.source_versions:
            if slot == "boundary_consent" and version != self.consent_version:
                raise ValueError(
                    "ConditioningBankSnapshot consent_version must equal the "
                    "boundary_consent entry in source_versions "
                    f"({version} != {self.consent_version})."
                )
        # A bank with no evidence and a bank whose consent was withdrawn must
        # both be inert. Allowing a non-zero readout in either state is the
        # exact failure mode -- silent influence -- the audit chain exists to
        # prevent, so it is rejected at construction rather than at the hook.
        if self.is_cold_start and (
            self.confidence != 0.0 or any(value != 0.0 for value in self.readout)
        ):
            raise ValueError(
                "Cold-start conditioning bank must have zero confidence and an "
                "all-zero readout."
            )
        if self.is_cold_start and self.rendered_statement:
            raise ValueError(
                "Cold-start conditioning bank must not carry a rendered "
                "statement: there is no evidence to state."
            )
        if self.revocation_state is ConditioningRevocationState.REVOKED:
            if self.confidence != 0.0 or any(value != 0.0 for value in self.readout):
                raise ValueError(
                    "Revoked conditioning bank must publish a zeroed readout "
                    "and zero confidence."
                )
            if self.rendered_statement:
                raise ValueError(
                    "Revoked conditioning bank must not carry a rendered statement."
                )

    @property
    def is_injectable(self) -> bool:
        """Whether a consumer may let this bank influence generation.

        The single predicate every consumer must gate on. Cold start, a
        revoked consent, and a zero-confidence readout are all inert for the
        same reason -- there is nothing evidenced to inject -- so they are
        collapsed here rather than re-derived at each call site.
        """

        return (
            not self.is_cold_start
            and self.revocation_state is ConditioningRevocationState.ACTIVE
            and self.confidence > 0.0
        )

    @property
    def fingerprint_parts(self) -> tuple[str, ...]:
        """Bank-identity components for a State KV cache key.

        Excludes ``confidence``/``freshness``: those scale the injection but
        do not change the generated K/V content, so including them would
        needlessly shatter the cache. Includes ``revocation_state`` so a
        revoked bank can never reuse the entry cached while it was active.
        """

        return (
            self.schema_version,
            self.bank_type.value,
            self.source_fingerprint,
            str(self.consent_version),
            self.revocation_state.value,
        )


__all__ = [
    "CONDITIONING_BANK_SCHEMA_VERSION",
    "CONDITIONING_BANK_SLOTS",
    "ConditioningBankSnapshot",
    "ConditioningBankType",
    "ConditioningRevocationState",
    "ConditioningScope",
    "expected_bank_type",
]
