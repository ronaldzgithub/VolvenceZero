"""Owner for the ``dialogue_external_outcome`` snapshot slot.

This is the single legal snapshot channel through which external dialogue
outcome evidence enters the kernel. It has three invariants:

1. The module never touches ``PredictionErrorModule`` /
   ``RegimeModule`` internal state directly. Consumers declare
   ``dialogue_external_outcome`` as a dependency and each owner decides
   how to consume it inside its own ``process(...)`` (R8).
2. LLM-sourced evidence is rejected unless the BrainConfig flag
   ``allow_llm_outcome_proposals`` is true. In v0 the flag is off by
   default; the module raises on any attempt to submit LLM evidence
   without the flag.
3. Each turn publishes a snapshot reflecting **only** evidence accepted
   up to the current turn; older evidence from previous turns is not
   re-published (consumers that want historical evidence read the
   dialogue trace store / reflection snapshot).

The submit API (``append_evidence`` here, plus the thin adapter on
``LifeformSession`` / ``BrainSession``) is the path any M2+ work must
flow through. The M1 scope of this module is:

* establish the slot and the typed value_type so ``RuptureStateModule``
  and downstream owners can declare the dependency;
* accept ``append_evidence`` calls and publish them;
* the ``LifeformSession.submit_dialogue_outcome`` adapter is still M2
  work — in M1 this module is reachable via ``append_evidence`` only.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


class DialogueExternalOutcomeModule(RuntimeModule[DialogueExternalOutcomeSnapshot]):
    """Owner of the ``dialogue_external_outcome`` snapshot slot."""

    slot_name: ClassVar[str] = "dialogue_external_outcome"
    owner: ClassVar[str] = "DialogueExternalOutcomeModule"
    value_type: ClassVar[type[Any]] = DialogueExternalOutcomeSnapshot
    dependencies: ClassVar[tuple[str, ...]] = ()
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        allow_llm_outcome_proposals: bool = False,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._allow_llm_outcome_proposals = bool(allow_llm_outcome_proposals)
        self._current_turn_index = 0
        self._pending_entries: list[DialogueExternalOutcomeEvidence] = []

    @property
    def allow_llm_outcome_proposals(self) -> bool:
        return self._allow_llm_outcome_proposals

    def set_turn_index(self, turn_index: int) -> None:
        """Advance the owner's view of the current turn index.

        Called by the session runner before each turn so any evidence
        appended this turn carries a correct ``turn_index`` context on
        the published snapshot.
        """

        if turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        self._current_turn_index = int(turn_index)

    def append_evidence(
        self,
        evidence: DialogueExternalOutcomeEvidence,
    ) -> None:
        """Accept one typed external-outcome evidence entry.

        Raises ``ValueError`` if the evidence source is
        ``LLM_PROPOSAL`` and ``allow_llm_outcome_proposals`` is false
        (Risk 2 mitigation in the v0 scope lock).
        """

        if (
            evidence.source is DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL
            and not self._allow_llm_outcome_proposals
        ):
            raise ValueError(
                "DialogueExternalOutcomeEvidence with source=LLM_PROPOSAL is not "
                "accepted unless BrainConfig.allow_llm_outcome_proposals is True. "
                "See docs/specs/rupture-and-repair.md Risk 2 mitigation."
            )
        self._pending_entries.append(evidence)

    def pending_entry_count(self) -> int:
        return len(self._pending_entries)

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[DialogueExternalOutcomeSnapshot]:
        _ = upstream  # no dependencies
        entries = tuple(self._pending_entries)
        # Evidence is consumed for this turn's snapshot; the owner does
        # not retain a ledger across turns (reflection and the dialogue
        # trace store own that job).
        self._pending_entries = []
        description = (
            f"DialogueExternalOutcome turn={self._current_turn_index} "
            f"entries={len(entries)} "
            f"llm_enabled={self._allow_llm_outcome_proposals}"
        )
        snapshot_value = DialogueExternalOutcomeSnapshot(
            turn_index=self._current_turn_index,
            entries=entries,
            description=description,
        )
        return self.publish(snapshot_value)


__all__ = ["DialogueExternalOutcomeModule"]
