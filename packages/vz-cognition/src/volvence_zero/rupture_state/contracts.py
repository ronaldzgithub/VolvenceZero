"""Rupture-state contract types.

Pure data module. Defines the closed ``RuptureKind`` evidence-bucket
vocabulary, the typed ``RuptureEvidenceSource`` enum, the per-source
contributing-signal record, and the published snapshot. All types are
frozen dataclasses so snapshots remain immutable.

Adding a new ``RuptureKind`` is gated on adding a new typed
:class:`RuptureEvidenceSource` that can produce it; the contract test
enforces that every kind has at least one non-PE source able to emit it
(see ``tests/contracts/test_rupture_state_contract.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind


class RuptureKind(str, Enum):
    """Closed evidence-bucket vocabulary for rupture kinds.

    These are *not* emotion labels. Each value is produced only when a
    typed signal source fires; see
    :data:`EXTERNAL_OUTCOME_TO_RUPTURE_KIND` and
    :data:`RUPTURE_KIND_SEVERITY`.
    """

    MISREAD = "misread"
    OVER_DIRECTIVE = "over_directive"
    PUSHED_TOO_FAST = "pushed_too_fast"
    COLD = "cold"
    UNSAFE = "unsafe"
    ABANDONED = "abandoned"


class RuptureEvidenceSource(str, Enum):
    """Typed evidence source for rupture detection.

    ``INTERNAL_PE`` is the only source that alone cannot resolve a
    ``rupture_kind``: when only internal PE fires, the owner publishes
    ``internal_suspected_only=True`` and ``rupture_kind=None``.

    ``LLM_PROPOSAL`` is defined for contract stability; runtime intake
    of LLM-sourced outcome evidence is gated by a ``BrainConfig`` flag
    disabled by default in v0.
    """

    INTERNAL_PE = "internal_pe"
    BEHAVIORAL_TRACE = "behavioral_trace"
    SELF_CHECK_ASSEMBLY = "self_check_assembly"
    EXTERNAL_USER = "external_user"
    ENVIRONMENT = "environment"
    LLM_PROPOSAL = "llm_proposal"


# Closed 1:1 structural mapping from DialogueExternalOutcomeKind -> RuptureKind.
# Kinds missing from this table (HELPED, FELT_HEARD, DECISION_CLEARER) do not
# produce rupture evidence by themselves; they are positive / neutral
# external outcomes and are handled by PE / regime as positive signals.
# COLD is not in this table because it requires a compositional trigger
# (MISSED + relationship_pressure), handled inside the owner aggregator.
EXTERNAL_OUTCOME_TO_RUPTURE_KIND: dict[DialogueExternalOutcomeKind, RuptureKind] = {
    DialogueExternalOutcomeKind.MISSED: RuptureKind.MISREAD,
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: RuptureKind.OVER_DIRECTIVE,
    DialogueExternalOutcomeKind.COME_BACK: RuptureKind.PUSHED_TOO_FAST,
    DialogueExternalOutcomeKind.UNSAFE: RuptureKind.UNSAFE,
    DialogueExternalOutcomeKind.ABANDONED: RuptureKind.ABANDONED,
}


# Severity ordering for disambiguation when multiple kinds could apply in a
# single turn. Highest severity wins. This is a documented static ordering,
# not a heuristic; learned prioritisation is post-v0.
RUPTURE_KIND_SEVERITY: dict[RuptureKind, int] = {
    RuptureKind.UNSAFE: 6,
    RuptureKind.ABANDONED: 5,
    RuptureKind.OVER_DIRECTIVE: 4,
    RuptureKind.MISREAD: 3,
    RuptureKind.PUSHED_TOO_FAST: 2,
    RuptureKind.COLD: 1,
}


# W3 of ssot-cleanup-p0-p4: human-readable phrase per RuptureKind. The
# rupture_state owner publishes ``kind_label`` from this single source so
# the lifeform expression layer does not maintain a duplicate dict.
RUPTURE_KIND_LABEL: dict[RuptureKind, str] = {
    RuptureKind.MISREAD: "a misread",
    RuptureKind.OVER_DIRECTIVE: "over-directive",
    RuptureKind.PUSHED_TOO_FAST: "pushed too fast",
    RuptureKind.COLD: "cold or not heard",
    RuptureKind.UNSAFE: "unsafe",
    RuptureKind.ABANDONED: "abandoned",
}


def rupture_kind_label(kind: RuptureKind | None) -> str:
    """Return the canonical human-readable phrase for a RuptureKind.

    Returns an empty string when ``kind`` is None. Adding a new kind
    requires landing a member in ``RUPTURE_KIND_LABEL`` (a contract
    test enforces 1:1 coverage in
    ``tests/contracts/test_rupture_state_contract.py``).
    """

    if kind is None:
        return ""
    return RUPTURE_KIND_LABEL[kind]


@dataclass(frozen=True)
class RuptureContributingSignal:
    """Single typed signal contribution to this turn's rupture readout.

    Consumers can audit which sources fired and with what strength. The
    ``signal_strength`` is the per-source normalised contribution in
    ``[0, 1]``; ``confidence`` is the per-source confidence.
    """

    source: RuptureEvidenceSource
    signal_strength: float
    confidence: float
    kind_hint: RuptureKind | None
    detail: str


@dataclass(frozen=True)
class RuptureStateSnapshot:
    """Per-turn rupture-state readout.

    Invariants (enforced by the owner):

    * ``rupture_kind is None`` iff no non-PE source contributed.
    * ``internal_suspected_only`` is True iff only ``INTERNAL_PE`` fired.
    * ``evidence_sources`` lists every source present in
      ``contributing_signals`` (de-duplicated, order-preserving).
    * ``rupture_signal_strength`` and ``confidence`` are both in ``[0, 1]``.
    * ``kind_label`` is the owner-published human-readable phrase for
      ``rupture_kind`` (empty string when ``rupture_kind is None``);
      consumers MUST read this field instead of maintaining their own
      RuptureKind -> string dict (W3 SSOT cleanup).
    """

    rupture_signal_strength: float
    rupture_kind: RuptureKind | None
    confidence: float
    internal_suspected_only: bool
    evidence_sources: tuple[RuptureEvidenceSource, ...]
    contributing_signals: tuple[RuptureContributingSignal, ...]
    description: str
    kind_label: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.rupture_signal_strength <= 1.0:
            raise ValueError(
                "rupture_signal_strength must be in [0, 1], "
                f"got {self.rupture_signal_strength!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        non_pe_sources = tuple(
            src
            for src in self.evidence_sources
            if src is not RuptureEvidenceSource.INTERNAL_PE
        )
        if self.rupture_kind is not None and not non_pe_sources:
            raise ValueError(
                "rupture_kind can only be resolved when at least one non-PE "
                "typed source contributes."
            )
        only_pe = (
            len(self.evidence_sources) == 1
            and self.evidence_sources[0] is RuptureEvidenceSource.INTERNAL_PE
        )
        if self.internal_suspected_only and not only_pe:
            raise ValueError(
                "internal_suspected_only=True requires exactly one source, INTERNAL_PE."
            )
        if not self.internal_suspected_only and only_pe:
            raise ValueError(
                "internal_suspected_only must be True when only INTERNAL_PE is present."
            )
        signal_sources = tuple(signal.source for signal in self.contributing_signals)
        if set(signal_sources) != set(self.evidence_sources):
            raise ValueError(
                "contributing_signals sources must match evidence_sources set."
            )
        # W3 SSOT: kind_label is always derived from rupture_kind via the
        # canonical map so consumers cannot drift the label by hand.
        canonical = rupture_kind_label(self.rupture_kind)
        if self.kind_label != canonical:
            object.__setattr__(self, "kind_label", canonical)


def _bootstrap_rupture_snapshot() -> RuptureStateSnapshot:
    """Build the canonical empty (no-rupture) snapshot for bootstrap turns."""

    return RuptureStateSnapshot(
        rupture_signal_strength=0.0,
        rupture_kind=None,
        confidence=0.0,
        internal_suspected_only=False,
        evidence_sources=(),
        contributing_signals=(),
        description="No rupture evidence this turn.",
    )


__all__ = [
    "EXTERNAL_OUTCOME_TO_RUPTURE_KIND",
    "RUPTURE_KIND_LABEL",
    "RUPTURE_KIND_SEVERITY",
    "RuptureContributingSignal",
    "RuptureEvidenceSource",
    "RuptureKind",
    "RuptureStateSnapshot",
    "_bootstrap_rupture_snapshot",
    "rupture_kind_label",
]
