"""``RuptureStateModule`` — the v0 SHADOW owner.

Reads four upstream snapshots (``prediction_error``,
``relationship_state``, ``response_assembly``,
``dialogue_external_outcome``), runs the pure per-source detection
functions, and publishes a :class:`RuptureStateSnapshot`.

The aggregator is intentionally transparent and non-learned:

* ``rupture_signal_strength`` = max non-PE strength if any non-PE source
  fired, else the PE strength; clamped to ``[0, 1]``.
* ``confidence`` = mean of contributing-source confidences.
* ``internal_suspected_only = True`` iff only ``INTERNAL_PE`` fired.
* ``rupture_kind`` = the severity-highest kind suggested by non-PE
  sources, with the compositional ``COLD`` rule applied when a MISREAD
  external signal co-occurs with elevated behavioral repair-pressure
  even without a direct match in the 1:1 table.

Any code change that introduces a new ``RuptureKind`` must also
register the typed source that produces it; the contract test catches
accidental additions.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from volvence_zero.dialogue_trace import DialogueExternalOutcomeSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.semantic_state.contracts import RelationshipStateSnapshot

from volvence_zero.rupture_state.contracts import (
    RUPTURE_KIND_SEVERITY,
    RuptureContributingSignal,
    RuptureEvidenceSource,
    RuptureKind,
    RuptureStateSnapshot,
    _bootstrap_rupture_snapshot,
)
from volvence_zero.rupture_state.detection import (
    behavioral_signal,
    external_user_signal,
    llm_proposal_signal,
    pe_spike_signal,
    self_check_signal,
)


_COLD_COMPOSITION_REPAIR_PRESSURE_THRESHOLD = 0.5


class RuptureStateModule(RuntimeModule[RuptureStateSnapshot]):
    """v0 SHADOW rupture-state owner (F1 / F2)."""

    slot_name: ClassVar[str] = "rupture_state"
    owner: ClassVar[str] = "RuptureStateModule"
    value_type: ClassVar[type[Any]] = RuptureStateSnapshot
    dependencies: ClassVar[tuple[str, ...]] = (
        "prediction_error",
        "relationship_state",
        "response_assembly",
        "dialogue_external_outcome",
    )
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        allow_llm_proposals: bool = False,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._allow_llm_proposals = bool(allow_llm_proposals)

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[RuptureStateSnapshot]:
        prediction_error_value = _get_value(upstream, "prediction_error")
        relationship_state_value = _get_value(upstream, "relationship_state")
        if not isinstance(relationship_state_value, RelationshipStateSnapshot):
            relationship_state_value = None
        response_assembly_value = _get_value(upstream, "response_assembly")
        external_outcome_value = _get_value(upstream, "dialogue_external_outcome")
        if not isinstance(external_outcome_value, DialogueExternalOutcomeSnapshot):
            external_outcome_value = None

        pe_signals = pe_spike_signal(prediction_error_value)
        behavioral_signals = behavioral_signal(relationship_state_value)
        self_check_signals = self_check_signal(response_assembly_value)
        external_signals = external_user_signal(external_outcome_value)
        llm_signals = llm_proposal_signal(
            external_outcome_value, allow_llm_proposals=self._allow_llm_proposals
        )

        contributing: tuple[RuptureContributingSignal, ...] = (
            pe_signals
            + behavioral_signals
            + self_check_signals
            + external_signals
            + llm_signals
        )

        if not contributing:
            return self.publish(_bootstrap_rupture_snapshot())

        snapshot = self._aggregate(
            contributing=contributing,
            relationship_state_value=relationship_state_value,
        )
        return self.publish(snapshot)

    def _aggregate(
        self,
        *,
        contributing: tuple[RuptureContributingSignal, ...],
        relationship_state_value: RelationshipStateSnapshot | None,
    ) -> RuptureStateSnapshot:
        sources_in_order: list[RuptureEvidenceSource] = []
        seen: set[RuptureEvidenceSource] = set()
        for signal in contributing:
            if signal.source in seen:
                continue
            seen.add(signal.source)
            sources_in_order.append(signal.source)
        evidence_sources = tuple(sources_in_order)

        non_pe_signals = tuple(
            signal
            for signal in contributing
            if signal.source is not RuptureEvidenceSource.INTERNAL_PE
        )
        pe_only = not non_pe_signals

        if pe_only:
            strength = max(signal.signal_strength for signal in contributing)
        else:
            strength = max(signal.signal_strength for signal in non_pe_signals)
        rupture_signal_strength = min(1.0, max(0.0, strength))

        confidence = sum(signal.confidence for signal in contributing) / len(
            contributing
        )
        confidence = min(1.0, max(0.0, confidence))

        kind = self._resolve_kind(
            non_pe_signals=non_pe_signals,
            relationship_state_value=relationship_state_value,
        )
        if pe_only:
            # PE alone cannot name a kind; it only raises internal suspicion.
            kind = None

        description = self._describe(
            kind=kind,
            strength=rupture_signal_strength,
            confidence=confidence,
            sources=evidence_sources,
            internal_suspected_only=pe_only,
        )

        return RuptureStateSnapshot(
            rupture_signal_strength=round(rupture_signal_strength, 4),
            rupture_kind=kind,
            confidence=round(confidence, 4),
            internal_suspected_only=pe_only,
            evidence_sources=evidence_sources,
            contributing_signals=contributing,
            description=description,
        )

    def _resolve_kind(
        self,
        *,
        non_pe_signals: tuple[RuptureContributingSignal, ...],
        relationship_state_value: RelationshipStateSnapshot | None,
    ) -> RuptureKind | None:
        if not non_pe_signals:
            return None
        candidates: list[RuptureKind] = []
        misread_seen = False
        for signal in non_pe_signals:
            if signal.kind_hint is None:
                continue
            candidates.append(signal.kind_hint)
            if signal.kind_hint is RuptureKind.MISREAD:
                misread_seen = True
        # Compositional COLD rule: MISREAD-external + elevated
        # behavioral repair pressure = cold. This is the sole compositional
        # rule in v0; it is encoded here once, in a named owner, as
        # documented in the spec.
        if misread_seen and relationship_state_value is not None:
            # ``RelationshipStateSnapshot`` is a typed dataclass; access
            # the field directly so a future schema rename / removal
            # fails loudly rather than silently producing degraded
            # cold-composition behaviour (W3 SSOT typed-access fix).
            repair_pressure = float(relationship_state_value.repair_pressure)
            if repair_pressure >= _COLD_COMPOSITION_REPAIR_PRESSURE_THRESHOLD:
                candidates.append(RuptureKind.COLD)
        if not candidates:
            return None
        return max(candidates, key=lambda kind: RUPTURE_KIND_SEVERITY.get(kind, 0))

    def _describe(
        self,
        *,
        kind: RuptureKind | None,
        strength: float,
        confidence: float,
        sources: tuple[RuptureEvidenceSource, ...],
        internal_suspected_only: bool,
    ) -> str:
        source_labels = ",".join(src.value for src in sources) or "none"
        if kind is None:
            if internal_suspected_only:
                return (
                    f"Internally suspected rupture (PE only) strength={strength:.2f} "
                    f"confidence={confidence:.2f} sources={source_labels}; "
                    "no externally-confirmed kind."
                )
            return (
                f"Rupture signals present but no kind resolved strength={strength:.2f} "
                f"confidence={confidence:.2f} sources={source_labels}."
            )
        return (
            f"Rupture kind={kind.value} strength={strength:.2f} "
            f"confidence={confidence:.2f} sources={source_labels}."
        )


def _get_value(upstream: Mapping[str, Snapshot[Any]], slot: str) -> object | None:
    snapshot = upstream.get(slot)
    if snapshot is None:
        return None
    return snapshot.value


__all__ = ["RuptureStateModule"]
