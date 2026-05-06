"""Interlocutor module - 12-axis readout + SHADOW owner.

This package was a single :file:`__init__.py` until Wave 2 of
``ssot-cleanup-p0-p4`` split it into:

* :mod:`.contracts` - 12-axis :class:`InterlocutorState`,
  :class:`InterlocutorReadoutContext`,
  :class:`InterlocutorStateSnapshot`, and the
  :class:`InterlocutorThresholds` SSOT for zone classification.
* :mod:`.readout` - pure 12-axis readout function and the
  duck-typed builder that translates kernel snapshots into the
  feature bundle.
* :mod:`.owner` - :class:`InterlocutorStateModule`, the SHADOW
  owner registered in the runtime so consumers read one snapshot.

The public API stays compatible with pre-W2 imports:

>>> from volvence_zero.interlocutor import (
...     InterlocutorState,
...     readout_interlocutor_state,
...     build_interlocutor_readout_context_from_snapshots,
... )

See ``docs/specs/interlocutor-state.md`` for the contract.
"""

from __future__ import annotations

from volvence_zero.interlocutor.contracts import (
    InterlocutorReadoutContext,
    InterlocutorState,
    InterlocutorStateSnapshot,
    InterlocutorThresholds,
    compute_zones,
    with_zones,
)
from volvence_zero.interlocutor.owner import InterlocutorStateModule
from volvence_zero.interlocutor.readout import (
    build_interlocutor_readout_context_from_snapshots,
    readout_interlocutor_state,
)


__all__ = [
    "InterlocutorReadoutContext",
    "InterlocutorState",
    "InterlocutorStateModule",
    "InterlocutorStateSnapshot",
    "InterlocutorThresholds",
    "build_interlocutor_readout_context_from_snapshots",
    "compute_zones",
    "readout_interlocutor_state",
    "with_zones",
]
