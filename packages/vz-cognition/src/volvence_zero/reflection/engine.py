"""Background-slow protocol reflection (packets 3.1 / 3.2).

The :class:`ProtocolReflectionEngine` runs at the
``background-slow`` timescale and proposes
``ProtocolRevisionProposal``s that the R10 ModificationGate
later evaluates and applies.

Invariants:

* SHADOW by default. Promotion to ACTIVE happens after at least
  one consumer (``ProtocolRegistry.apply_revision``) is wired and
  the gate has matched-control evidence.
* Reads only from the propagate snapshot map (``prediction_error``
  + ``active_mixture``). Holds an internal rolling window of
  observed history; never mutates upstream owners.
* Runs the rules collection every ``scan_period`` turns;
  intervening turns re-publish the previous snapshot (or an
  empty one on cold start) so consumers see a stable surface.

Packet 3.1 lands the skeleton (history collection + empty-
proposals output). Packet 3.2 wires the actual rule set.
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar, Mapping

from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ProtocolReflectionSnapshot,
    ProtocolRegistrySnapshot,
    ProtocolRevisionProposal,
)
from volvence_zero.prediction import PredictionErrorSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


_DEFAULT_HISTORY_WINDOW = 100
_DEFAULT_SCAN_PERIOD = 10


class ProtocolReflectionEngine(RuntimeModule[ProtocolReflectionSnapshot]):
    """SHADOW owner of the ``protocol_reflection`` slot.

    Internal state (across turns):

    * ``_pe_history``: ``deque[PredictionErrorSnapshot]`` of the
      last ``history_window`` non-bootstrap PE snapshots.
    * ``_active_mixture_history``: ``deque[ActiveMixtureSnapshot]``
      paired one-for-one with ``_pe_history`` (so attribution
      tooling can find which protocols were active when each
      PE fired).
    * ``_turns_since_last_scan``: counter incremented every
      turn. When it reaches ``scan_period`` the rule set runs
      and the counter resets.
    * ``_last_proposals``: cache of last cycle's proposals so
      intervening turns can re-publish without re-running rules.
    """

    slot_name: ClassVar[str] = "protocol_reflection"
    owner: ClassVar[str] = "ProtocolReflectionEngine"
    value_type: ClassVar[type[Any]] = ProtocolReflectionSnapshot
    # Packet 6.2: domain_knowledge / case_memory added so the
    # archival rules can read hit_ids per turn without breaking
    # the cognition→application import-boundary rule (snapshots
    # are read duck-typedly via getattr, no concrete type import).
    # Packet 9.2: protocol_registry added so archival rules can
    # see which knowledge / case lineage IDs each loaded protocol
    # owns (compile-time lineage prefix).
    dependencies: ClassVar[tuple[str, ...]] = (
        "prediction_error",
        "active_mixture",
        "domain_knowledge",
        "case_memory",
        "protocol_registry",
    )
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        history_window: int = _DEFAULT_HISTORY_WINDOW,
        scan_period: int = _DEFAULT_SCAN_PERIOD,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        if history_window < 1:
            raise ValueError(
                "history_window must be >= 1; got "
                f"{history_window!r}"
            )
        if scan_period < 1:
            raise ValueError(
                f"scan_period must be >= 1; got {scan_period!r}"
            )
        self._history_window = history_window
        self._scan_period = scan_period
        self._pe_history: deque[PredictionErrorSnapshot] = deque(
            maxlen=history_window
        )
        self._active_mixture_history: deque[ActiveMixtureSnapshot] = deque(
            maxlen=history_window
        )
        # Packet 6.2: per-turn observed hit_id sets for archival rules.
        # Duck-typed reads from domain_knowledge.hits[*].hit_id and
        # case_memory.hits[*].case_id; rules use these to detect
        # "this protocol's seed X never gets retrieved" patterns.
        self._knowledge_hit_history: deque[tuple[str, ...]] = deque(
            maxlen=history_window
        )
        self._case_hit_history: deque[tuple[str, ...]] = deque(
            maxlen=history_window
        )
        # Packet 9.2: latest registry snapshot — used by archival
        # rules to map lineage IDs back to owning protocol IDs.
        self._latest_registry_snapshot: ProtocolRegistrySnapshot | None = None
        self._turns_since_last_scan: int = 0
        self._last_pe_turn_index: int | None = None
        self._last_proposals: tuple[ProtocolRevisionProposal, ...] = ()

    @property
    def pe_history(self) -> tuple[PredictionErrorSnapshot, ...]:
        """Read-only view of buffered PE history (most-recent last)."""
        return tuple(self._pe_history)

    @property
    def active_mixture_history(self) -> tuple[ActiveMixtureSnapshot, ...]:
        """Read-only view of buffered active_mixture history."""
        return tuple(self._active_mixture_history)

    @property
    def turns_since_last_scan(self) -> int:
        return self._turns_since_last_scan

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ProtocolReflectionSnapshot]:
        # Collect history from this turn (SHADOW-tolerant: missing
        # upstream just means we don't append anything).
        self._ingest_pe(upstream)
        self._ingest_active_mixture(upstream)
        self._ingest_knowledge_hits(upstream)
        self._ingest_case_hits(upstream)
        self._ingest_registry(upstream)

        self._turns_since_last_scan += 1

        if self._turns_since_last_scan >= self._scan_period:
            self._last_proposals = self._run_rules()
            self._turns_since_last_scan = 0
            description = (
                f"protocol_reflection scan executed; "
                f"history_size={len(self._pe_history)} "
                f"proposals={len(self._last_proposals)}"
            )
        else:
            description = (
                f"protocol_reflection awaiting next scan; "
                f"turns_until_scan="
                f"{self._scan_period - self._turns_since_last_scan} "
                f"history_size={len(self._pe_history)}"
            )

        return self.publish(
            ProtocolReflectionSnapshot(
                protocol_revision_proposals=self._last_proposals,
                observation_window_turns=len(self._pe_history),
                turns_since_last_scan=self._turns_since_last_scan,
                description=description,
            )
        )

    # ------------------------------------------------------------------
    # Upstream readers
    # ------------------------------------------------------------------

    def _ingest_pe(self, upstream: Mapping[str, Snapshot[Any]]) -> None:
        snapshot = upstream.get("prediction_error")
        if snapshot is None:
            return
        value = snapshot.value
        if not isinstance(value, PredictionErrorSnapshot):
            return
        if value.bootstrap:
            return
        if (
            self._last_pe_turn_index is not None
            and value.turn_index <= self._last_pe_turn_index
        ):
            return
        self._last_pe_turn_index = value.turn_index
        self._pe_history.append(value)

    def _ingest_active_mixture(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> None:
        snapshot = upstream.get("active_mixture")
        if snapshot is None:
            return
        value = snapshot.value
        if not isinstance(value, ActiveMixtureSnapshot):
            return
        self._active_mixture_history.append(value)

    def _ingest_knowledge_hits(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> None:
        """Duck-typed read of domain_knowledge.hits[*].hit_id.

        Reads via getattr to keep cognition tier free of vz-application
        type imports. Missing snapshot / shape mismatch → empty tuple
        appended (means "no hits this turn", not "skip turn").
        """

        snapshot = upstream.get("domain_knowledge")
        if snapshot is None:
            self._knowledge_hit_history.append(())
            return
        value = snapshot.value
        hits = getattr(value, "hits", None)
        if hits is None:
            self._knowledge_hit_history.append(())
            return
        hit_ids = tuple(
            getattr(h, "hit_id", "")
            for h in hits
            if getattr(h, "hit_id", "")
        )
        self._knowledge_hit_history.append(hit_ids)

    def _ingest_case_hits(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> None:
        """Duck-typed read of case_memory.hits[*].case_id."""

        snapshot = upstream.get("case_memory")
        if snapshot is None:
            self._case_hit_history.append(())
            return
        value = snapshot.value
        hits = getattr(value, "hits", None)
        if hits is None:
            self._case_hit_history.append(())
            return
        case_ids = tuple(
            getattr(h, "case_id", "")
            for h in hits
            if getattr(h, "case_id", "")
        )
        self._case_hit_history.append(case_ids)

    def _ingest_registry(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> None:
        snapshot = upstream.get("protocol_registry")
        if snapshot is None:
            return
        value = snapshot.value
        if not isinstance(value, ProtocolRegistrySnapshot):
            return
        self._latest_registry_snapshot = value

    # ------------------------------------------------------------------
    # Rules dispatch (filled in by packet 3.2)
    # ------------------------------------------------------------------

    def _run_rules(self) -> tuple[ProtocolRevisionProposal, ...]:
        """Run all reflection rules and return their merged proposals."""

        from volvence_zero.reflection.protocol_revision_rules import (
            run_all_protocol_revision_rules,
        )

        # Packet 9.2: build per-protocol → lineage_ids maps from the
        # latest registry snapshot so archival rules can attribute
        # absent hits to a specific (protocol, seed/case) pair.
        knowledge_lineage_by_protocol: dict[str, tuple[str, ...]] = {}
        case_lineage_by_protocol: dict[str, tuple[str, ...]] = {}
        if self._latest_registry_snapshot is not None:
            for entry in self._latest_registry_snapshot.entries:
                knowledge_lineage_by_protocol[entry.protocol_id] = (
                    entry.knowledge_lineage_ids
                )
                case_lineage_by_protocol[entry.protocol_id] = (
                    entry.case_lineage_ids
                )

        return run_all_protocol_revision_rules(
            pe_history=tuple(self._pe_history),
            active_mixture_history=tuple(self._active_mixture_history),
            knowledge_hit_history=tuple(self._knowledge_hit_history),
            case_hit_history=tuple(self._case_hit_history),
            knowledge_lineage_by_protocol=knowledge_lineage_by_protocol,
            case_lineage_by_protocol=case_lineage_by_protocol,
        )

    @property
    def knowledge_hit_history(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._knowledge_hit_history)

    @property
    def case_hit_history(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._case_hit_history)


__all__ = ["ProtocolReflectionEngine"]
