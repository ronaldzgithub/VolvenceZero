"""Rupture-and-repair owner (R-PE / R7 / R8 / R11 / R15).

Publishes a frozen :class:`RuptureStateSnapshot` at ``WiringLevel.SHADOW``
describing whether the relationship appears to have ruptured, what kind
the rupture is (from a closed evidence-bucket vocabulary), and what typed
sources contributed.

The owner reads upstream snapshots only. It does not classify text. It
does not own relationship, regime, or PE state; it is a readout that
reflection can aggregate.

See :doc:`docs/specs/rupture-and-repair` for the full contract.
"""

from __future__ import annotations

from volvence_zero.rupture_state.contracts import (
    EXTERNAL_OUTCOME_TO_RUPTURE_KIND,
    RUPTURE_KIND_SEVERITY,
    RuptureContributingSignal,
    RuptureEvidenceSource,
    RuptureKind,
    RuptureStateSnapshot,
)
from volvence_zero.rupture_state.detection import (
    behavioral_signal,
    external_user_signal,
    llm_proposal_signal,
    pe_spike_signal,
    self_check_signal,
)
from volvence_zero.rupture_state.owner import RuptureStateModule

__all__ = [
    "EXTERNAL_OUTCOME_TO_RUPTURE_KIND",
    "RUPTURE_KIND_SEVERITY",
    "RuptureContributingSignal",
    "RuptureEvidenceSource",
    "RuptureKind",
    "RuptureStateModule",
    "RuptureStateSnapshot",
    "behavioral_signal",
    "external_user_signal",
    "llm_proposal_signal",
    "pe_spike_signal",
    "self_check_signal",
]
