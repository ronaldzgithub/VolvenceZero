"""Pure per-source rupture-detection functions.

Each function reads a single upstream snapshot (or typed evidence tuple)
and returns zero or more :class:`RuptureContributingSignal` entries. None
of these functions parses free text; they only read structured snapshot
fields produced by their typed source.

Rules:

* PE spike is the only source that is alone not sufficient to resolve a
  ``rupture_kind``. The owner aggregator marks any resulting snapshot as
  ``internal_suspected_only=True`` when nothing else fires.
* Behavioral source reads semantic readouts on
  :class:`RelationshipStateSnapshot` (e.g. ``unresolved_tension_count``);
  no text heuristics.
* Self-check source depends on a structural diagnostic flag on
  response-assembly that does not exist in v0; this function is a stub
  that always returns ``()`` until that flag is published (post-v0).
* External-user source reads
  :class:`DialogueExternalOutcomeSnapshot.entries` and maps typed kinds
  through the closed 1:1 :data:`EXTERNAL_OUTCOME_TO_RUPTURE_KIND` table
  defined in :mod:`volvence_zero.rupture_state.contracts`.
* LLM proposal source is stubbed OFF: it returns ``()`` unconditionally
  in v0 unless the caller explicitly passes ``allow_llm_proposals=True``,
  and even then the evidence carries low confidence.
"""

from __future__ import annotations

from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.rupture_state.contracts import (
    EXTERNAL_OUTCOME_TO_RUPTURE_KIND,
    RuptureContributingSignal,
    RuptureEvidenceSource,
    RuptureKind,
)


# v0 thresholds are documented constants, not learned parameters. They
# exist only to normalise raw source intensity into the bounded [0, 1]
# signal strength required by RuptureStateSnapshot. Learned weighting is
# post-v0 (see docs/specs/rupture-and-repair.md).
_PE_SPIKE_MAGNITUDE_THRESHOLD = 0.35
_PE_SPIKE_RELATIONSHIP_THRESHOLD = 0.35
_BEHAVIORAL_TENSION_THRESHOLD = 1
_BEHAVIORAL_REPAIR_PRESSURE_THRESHOLD = 0.4
_LLM_PROPOSAL_MAX_CONFIDENCE = 0.4


def _clamp_unit(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def pe_spike_signal(
    prediction_error_snapshot: object | None,
) -> tuple[RuptureContributingSignal, ...]:
    """Internal PE spike: reads relationship / action / regime errors.

    PE is only one input — alone it can only put the owner into the
    ``internal_suspected_only=True`` state, not resolve a rupture_kind.
    """

    if prediction_error_snapshot is None:
        return ()
    bootstrap = bool(getattr(prediction_error_snapshot, "bootstrap", True))
    if bootstrap:
        return ()
    error = getattr(prediction_error_snapshot, "error", None)
    if error is None:
        return ()
    relationship_error = abs(float(getattr(error, "relationship_error", 0.0)))
    action_error = abs(float(getattr(error, "action_error", 0.0)))
    regime_error = abs(float(getattr(error, "regime_error", 0.0)))
    magnitude = abs(float(getattr(error, "magnitude", 0.0)))
    if (
        magnitude < _PE_SPIKE_MAGNITUDE_THRESHOLD
        and relationship_error < _PE_SPIKE_RELATIONSHIP_THRESHOLD
    ):
        return ()
    # Signal strength is the maximum of the three error axes, clamped to
    # [0, 1]. Confidence mirrors this (PE is a low-confidence source for
    # rupture because it is an internal surprise signal, not an external
    # confirmation).
    raw = max(relationship_error, action_error, regime_error)
    strength = _clamp_unit(raw)
    return (
        RuptureContributingSignal(
            source=RuptureEvidenceSource.INTERNAL_PE,
            signal_strength=strength,
            confidence=min(strength, 0.5),
            kind_hint=None,
            detail=(
                f"pe.magnitude={magnitude:.2f} "
                f"relationship_error={relationship_error:.2f} "
                f"action_error={action_error:.2f} "
                f"regime_error={regime_error:.2f}"
            ),
        ),
    )


def behavioral_signal(
    relationship_state_snapshot: object | None,
) -> tuple[RuptureContributingSignal, ...]:
    """Behavioral / relationship readout source.

    Reads structured relationship readouts
    (``unresolved_tension_count``, ``repair_pressure``,
    ``stabilization_need``) from the relationship-state snapshot. Does
    not parse text.
    """

    if relationship_state_snapshot is None:
        return ()
    unresolved = int(getattr(relationship_state_snapshot, "unresolved_tension_count", 0))
    repair_pressure = float(getattr(relationship_state_snapshot, "repair_pressure", 0.0))
    stabilization_need = float(
        getattr(relationship_state_snapshot, "stabilization_need", 0.0)
    )
    if (
        unresolved < _BEHAVIORAL_TENSION_THRESHOLD
        and repair_pressure < _BEHAVIORAL_REPAIR_PRESSURE_THRESHOLD
        and stabilization_need < _BEHAVIORAL_REPAIR_PRESSURE_THRESHOLD
    ):
        return ()
    strength = _clamp_unit(
        max(
            min(unresolved / 4.0, 1.0),
            repair_pressure,
            stabilization_need,
        )
    )
    # Behavioral evidence confidence is modest; it is a pattern-based,
    # not an externally-confirmed, source. It can contribute to the
    # aggregate but cannot alone resolve a rupture_kind without an
    # external source because it does not name what went wrong.
    return (
        RuptureContributingSignal(
            source=RuptureEvidenceSource.BEHAVIORAL_TRACE,
            signal_strength=strength,
            confidence=min(strength, 0.6),
            kind_hint=None,
            detail=(
                f"unresolved_tension_count={unresolved} "
                f"repair_pressure={repair_pressure:.2f} "
                f"stabilization_need={stabilization_need:.2f}"
            ),
        ),
    )


def self_check_signal(
    response_assembly_snapshot: object | None,
) -> tuple[RuptureContributingSignal, ...]:
    """Self-check / structural-mismatch source (v0 stub).

    Response-assembly does not publish a structural mismatch diagnostic
    yet. This function returns ``()`` unconditionally in v0 so that
    adding the diagnostic later is a no-op contract change rather than a
    new runtime dependency. Callers still pass the snapshot to keep the
    call site stable.
    """

    _ = response_assembly_snapshot
    return ()


def external_user_signal(
    external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None,
) -> tuple[RuptureContributingSignal, ...]:
    """External user / environment / human-review outcome source.

    Reads :class:`DialogueExternalOutcomeSnapshot.entries` and emits one
    contributing signal per entry whose typed kind maps through the
    closed :data:`EXTERNAL_OUTCOME_TO_RUPTURE_KIND` table. Entries whose
    source is ``LLM_PROPOSAL`` are NOT consumed here; that source is
    handled by :func:`llm_proposal_signal` so it can be independently
    gated by the BrainConfig flag.
    """

    if external_outcome_snapshot is None:
        return ()
    signals: list[RuptureContributingSignal] = []
    for entry in external_outcome_snapshot.entries:
        if entry.source is DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL:
            continue
        kind_hint = EXTERNAL_OUTCOME_TO_RUPTURE_KIND.get(entry.kind)
        if kind_hint is None:
            # HELPED / FELT_HEARD / DECISION_CLEARER etc. are positive or
            # neutral; they do not contribute to rupture evidence.
            continue
        source = (
            RuptureEvidenceSource.ENVIRONMENT
            if entry.source is DialogueExternalOutcomeEvidenceSource.ENVIRONMENT
            else RuptureEvidenceSource.EXTERNAL_USER
        )
        signals.append(
            RuptureContributingSignal(
                source=source,
                signal_strength=_clamp_unit(entry.confidence),
                confidence=_clamp_unit(entry.confidence),
                kind_hint=kind_hint,
                detail=(
                    f"external_outcome={entry.kind.value} "
                    f"source={entry.source.value} "
                    f"evidence_ref={entry.evidence_ref}"
                ),
            )
        )
    return tuple(signals)


def llm_proposal_signal(
    external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None,
    *,
    allow_llm_proposals: bool = False,
) -> tuple[RuptureContributingSignal, ...]:
    """LLM-proposal source (feature-flag gated OFF in v0).

    When the caller passes ``allow_llm_proposals=False`` (the v0
    default), this function returns ``()`` regardless of what the
    snapshot carries. When explicitly enabled, LLM-sourced evidence
    contributes at *low* confidence (clamped to
    ``_LLM_PROPOSAL_MAX_CONFIDENCE``), preserving the doc's rule that
    LLM must never set reward or become authoritative.
    """

    if not allow_llm_proposals:
        return ()
    if external_outcome_snapshot is None:
        return ()
    signals: list[RuptureContributingSignal] = []
    for entry in external_outcome_snapshot.entries:
        if entry.source is not DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL:
            continue
        kind_hint = EXTERNAL_OUTCOME_TO_RUPTURE_KIND.get(entry.kind)
        if kind_hint is None:
            continue
        clamped_conf = min(
            _clamp_unit(entry.confidence), _LLM_PROPOSAL_MAX_CONFIDENCE
        )
        signals.append(
            RuptureContributingSignal(
                source=RuptureEvidenceSource.LLM_PROPOSAL,
                signal_strength=clamped_conf,
                confidence=clamped_conf,
                kind_hint=kind_hint,
                detail=(
                    f"llm_proposal external_outcome={entry.kind.value} "
                    f"evidence_ref={entry.evidence_ref}"
                ),
            )
        )
    return tuple(signals)


def _non_pe_kind_hints(
    signals: tuple[RuptureContributingSignal, ...],
) -> tuple[tuple[RuptureKind, RuptureContributingSignal], ...]:
    out: list[tuple[RuptureKind, RuptureContributingSignal]] = []
    for signal in signals:
        if signal.source is RuptureEvidenceSource.INTERNAL_PE:
            continue
        if signal.kind_hint is None:
            continue
        out.append((signal.kind_hint, signal))
    return tuple(out)


__all__ = [
    "behavioral_signal",
    "external_user_signal",
    "llm_proposal_signal",
    "pe_spike_signal",
    "self_check_signal",
    "_non_pe_kind_hints",
]
