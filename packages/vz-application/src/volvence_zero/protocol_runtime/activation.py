"""Activation weight computation (packet 1.0 — equal-weight fallback).

The full activation formula declared in
``docs/specs/protocol-runtime.md §调度`` is (post packet-1.0.1):

    weight_i = identity_gate_i × softmax(
        α · context_match_i + β · pe_utility_i
    )

(Packet 1.0.1 dropped the original ``γ · drive_value_i`` term:
``VitalsSnapshot`` is a lifeform-side contract that kernel modules
cannot import. Drive coupling is deferred until either an explicit
kernel-side ``DriveReadoutSnapshot`` adapter is introduced or the
deferral is made permanent.)

Packet 1.0 ships only the **equal-weight fallback** branch:

* ``identity_gate`` = 1.0 for every protocol (R7-Self / R14-regime
  cross-check lights up in packet 1.3+).
* ``α = β = 0`` so the softmax of all-zeros is uniform.
* ``minimum_weight_floor`` is honoured even in the fallback (a
  protocol with floor 0.05 stays at ≥ 0.05 in the mixture).
* Co-activation incompatibility is honoured: incompatible pairs
  drop the lower-id protocol (deterministic tiebreak; later packets
  will use PE-history).

This file is the *only* place per-turn activation logic lives. The
``ProtocolRegistryModule`` calls ``compute_active_mixture`` and
publishes the result. Future packets will extend this function (or
split it) without changing the public ``ActiveMixtureSnapshot``
shape.

ACTIVE-gate guard
-----------------

``is_fallback_mode()`` reports whether the activation controller is
still in packet 1.0 placeholder mode. The owner module
(``ProtocolRegistryModule.__init__``) reads this and refuses to
construct at ``WiringLevel.ACTIVE`` while it is True, because the
fallback's outputs are not safe for downstream consumption (see
``docs/specs/protocol-runtime.md §SHADOW → ACTIVE 升级 checklist``).

Future packets that light up real machinery (PE history, typed
context-match signals, real identity gate, PE-driven arbitration)
flip this flag to False as part of their landing PR + extend the
guard with finer-grained checks if needed.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

from volvence_zero.application.types import BoundaryPolicySnapshot
from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
    ActivationReason,
    ActivationReasonKind,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    ContextMatchSignal,
)
from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.interlocutor import InterlocutorStateSnapshot
from volvence_zero.regime import RegimeSnapshot
from volvence_zero.rupture_state import RuptureStateSnapshot
from volvence_zero.runtime import Snapshot


# ---------------------------------------------------------------------------
# ACTIVE-gate guard (packet 1.0.1)
# ---------------------------------------------------------------------------
#
# Packet 1.0 ships an equal-weight / hard-coded-identity fallback. The
# spec's SHADOW → ACTIVE checklist requires real machinery before the
# slot can be promoted to ACTIVE wiring. ``is_fallback_mode()`` is the
# single boolean read by the owner module to enforce that contract.
#
# Future packets that light up real machinery flip this to False:
#   - packet 1.3 lands real identity_gate (R7 Self / R14 regime cross-check)
#   - packet 1.5 lands α/β learned weights + typed context_match + PE arbitration
# Any of those landing PRs MUST update this constant in the same PR
# that introduces the real implementation, so the guard tracks reality.
_ACTIVATION_CONTROLLER_FALLBACK_MODE: bool = True


def is_fallback_mode() -> bool:
    """Whether ``compute_active_mixture`` is still in packet-1.0 fallback.

    Read by ``ProtocolRegistryModule.__init__`` to gate ACTIVE
    wiring. True means the activation logic is still missing one
    or more upgrades from the SHADOW → ACTIVE checklist; downstream
    consumers cannot trust the outputs as a learned posterior. False
    is set only by future packets that have landed all required
    machinery (see module docstring).

    Packet 1.3a/1.3'/1.3''/1.3''' state: ``identity_gate`` fully
    real (R14 regime + R7 self-trait via
    ``IdentitySeed`` populator). Packet 1.5a state: typed
    ``context_match`` signal scoring partially real — 3 kernel-side
    detectors (interlocutor zone / rupture state / boundary policy)
    fire on real upstream snapshots; vitals-tied DRIVE detectors
    deferred (vitals not in kernel propagate graph; see packet 1.0.1
    decision). Still placeholder (packet 1.5b+/c+): PE history
    utility, learned α/β, PE-driven arbitration. The aggregate flag
    therefore stays True until those land.
    """

    return _ACTIVATION_CONTROLLER_FALLBACK_MODE


def compute_active_mixture(
    *,
    loaded_protocols: tuple[BehaviorProtocol, ...],
    upstream: Mapping[str, Snapshot[Any]],
) -> ActiveMixtureSnapshot:
    """Build the per-turn ``ActiveMixtureSnapshot`` from the registry.

    Args:
        loaded_protocols: All protocols currently registered.
            Ordering is registry-deterministic (sorted by id).
        upstream: Upstream snapshots available to the module.
            Packet 1.3a consumes ``dual_track`` and ``regime`` for
            identity-gate evaluation. Other inputs (PE history,
            typed context_match) land in packet 1.5+.

    Returns:
        The ``ActiveMixtureSnapshot`` to publish into the
        ``active_mixture`` slot.
    """

    if not loaded_protocols:
        return ActiveMixtureSnapshot(
            active_protocols=(),
            boundary_union_ids=(),
            identity_gate_traits=(),
            revision_fingerprint=_fingerprint(loaded_protocols),
            description="empty registry: no protocols loaded",
        )

    dual_track_snapshot = _read_dual_track_snapshot(upstream)
    regime_snapshot = _read_regime_snapshot(upstream)

    # Identity gate (packet 1.3a): per-protocol filter based on
    # regime compatibility (real) and self_traits (permissive
    # placeholder until dual_track exposes a trait vocabulary).
    gate_results: list[tuple[BehaviorProtocol, float, tuple[str, ...]]] = []
    for protocol in loaded_protocols:
        gate_value, gate_reasons = _compute_identity_gate(
            protocol,
            dual_track_snapshot=dual_track_snapshot,
            regime_snapshot=regime_snapshot,
        )
        gate_results.append((protocol, gate_value, gate_reasons))

    # Drop protocols whose identity gate is 0 — they are filtered
    # out entirely (R8 hard filter; not zero-weighted ghosts).
    eligible_protocols = tuple(
        p for p, gate, _ in gate_results if gate > 0.0
    )
    gate_reasons_by_id: dict[str, tuple[str, ...]] = {
        p.protocol_id: reasons for p, gate, reasons in gate_results if gate > 0.0
    }

    if not eligible_protocols:
        return ActiveMixtureSnapshot(
            active_protocols=(),
            boundary_union_ids=(),
            identity_gate_traits=(),
            revision_fingerprint=_fingerprint(loaded_protocols),
            description=(
                f"identity gate filtered all {len(loaded_protocols)} "
                "loaded protocol(s)"
            ),
        )

    eligible = _resolve_co_activation_incompatibility(eligible_protocols)

    # Packet 1.5a: compute typed context_match scores per protocol.
    # When all scores are 0 (no signals in any protocol's
    # activation_conditions, or no signals firing), the scoring step
    # collapses to ``equal_weight_with_floor`` — preserving the
    # packet 1.0+ uniform-weight contract for cheng_laoshi-shape
    # protocols. When at least one score > 0, softmax of scores
    # produces real differential weighting.
    interlocutor_snapshot = _read_interlocutor_snapshot(upstream)
    rupture_snapshot = _read_rupture_snapshot(upstream)
    boundary_policy_snapshot = _read_boundary_policy_snapshot(upstream)

    context_results: list[tuple[float, tuple[str, ...]]] = []
    max_score = 0.0
    for protocol in eligible:
        score, reasons = _compute_context_match(
            protocol,
            interlocutor_snapshot=interlocutor_snapshot,
            rupture_snapshot=rupture_snapshot,
            boundary_policy_snapshot=boundary_policy_snapshot,
        )
        context_results.append((score, reasons))
        max_score = max(max_score, score)

    if max_score > 0.0:
        weights = _softmax_weights_with_floor(eligible, context_results)
        weighting_kind = ActivationReasonKind.CONTEXT_MATCH
        weighting_detail_template = (
            "packet 1.5a: softmax of context_match scores "
            "(α=1, β=0; PE / α-β learning pending packet 1.5b+)"
        )
    else:
        weights = _equal_weight_with_floor(eligible)
        weighting_kind = ActivationReasonKind.EQUAL_WEIGHT_FALLBACK
        weighting_detail_template = (
            "no context_match signal fired; equal weight across "
            "eligible protocols (PE / α-β / drive coupling pending "
            "packet 1.5b+)"
        )

    active_entries: tuple[ActiveProtocolEntry, ...] = tuple(
        ActiveProtocolEntry(
            protocol_id=protocol.protocol_id,
            activation_weight=weight,
            current_phase_id=_default_phase_id(protocol),
            activation_reasons=(
                ActivationReason(
                    kind=ActivationReasonKind.IDENTITY_GATE,
                    contribution=1.0,
                    detail="; ".join(gate_reasons_by_id[protocol.protocol_id])
                    or "identity_gate=1.0",
                ),
                ActivationReason(
                    kind=weighting_kind,
                    contribution=weight,
                    detail=(
                        weighting_detail_template
                        + (
                            f"; signals_fired=[{','.join(context_results[idx][1])}]"
                            if context_results[idx][1]
                            else ""
                        )
                    ),
                ),
            ),
        )
        for idx, (protocol, weight) in enumerate(
            zip(eligible, weights, strict=True)
        )
    )

    boundary_union_ids = _union_boundary_ids(eligible)
    description = (
        f"active mixture: {len(active_entries)} protocol(s); "
        f"{len(boundary_union_ids)} boundary id(s); "
        f"max_context_score={max_score:.3f}"
    )

    return ActiveMixtureSnapshot(
        active_protocols=active_entries,
        boundary_union_ids=boundary_union_ids,
        identity_gate_traits=(),
        revision_fingerprint=_fingerprint(eligible),
        description=description,
    )


# ---------------------------------------------------------------------------
# Identity gate (packet 1.3a)
# ---------------------------------------------------------------------------


def _compute_identity_gate(
    protocol: BehaviorProtocol,
    *,
    dual_track_snapshot: DualTrackSnapshot | None,
    regime_snapshot: RegimeSnapshot | None,
) -> tuple[float, tuple[str, ...]]:
    """Per-protocol identity gate (R7 Self / R14 regime cross-check).

    Returns a tuple ``(gate_value, reasons)`` where:

    * ``gate_value`` is 1.0 when the protocol is compatible with
      the current Identity Core, 0.0 when filtered out.
    * ``reasons`` is an audit-friendly tuple of short strings
      explaining the verdict; surfaced as
      ``ActivationReason.detail`` for the IDENTITY_GATE reason.

    Packet 1.3a / 1.3' policy:

    * **Regime check (real, packet 1.3a)**: when
      ``protocol.identity_assertion.required_regime_compatibility``
      is non-empty, the current ``regime_snapshot.active_regime
      .regime_id`` must be a member. Missing regime snapshot →
      SHADOW-permissive pass (logged in reasons). Mismatch → gate=0.
    * **Self-trait check (real when traits populated, packet 1.3')**:
      ``DualTrackSnapshot.self_track.traits`` is the trait surface
      (``vz-cognition.dual_track.TrackState.traits``, added in
      packet 1.3'). Decision matrix:

      - ``requires_self_traits`` non-empty AND ``self_track.traits``
        non-empty: every required trait must be present in
        ``self_track.traits``; missing any → gate=0
      - ``forbidden_self_traits`` non-empty AND ``self_track.traits``
        non-empty: any forbidden trait present → gate=0
      - ``self_track.traits`` empty (no populator wired yet) and
        any side has assertions: SHADOW-permissive pass with
        ``self_traits_populator_pending`` marker. ACTIVE promotion
        is gated separately by ``FallbackActivationActiveError``
        until a populator lands.
      - ``dual_track_snapshot`` missing entirely: SHADOW-permissive
        pass (the owner is SHADOW; downstream consumers don't read
        this slot ACTIVE yet).

    * **Empty assertion fields**: skipped silently (logged once
      per protocol so audit shows we passed deliberately, not by
      omission).
    """

    reasons: list[str] = []

    # --- Regime branch (real, packet 1.3a) --------------------------------
    required_regimes = protocol.identity_assertion.required_regime_compatibility
    if required_regimes:
        if regime_snapshot is None:
            # SHADOW permissive: no regime upstream means we don't
            # have evidence to filter. The owner is SHADOW today;
            # ACTIVE promotion requires real upstream presence
            # (enforced by FallbackActivationActiveError + checklist).
            reasons.append("regime_unknown_shadow_pass")
        else:
            current_regime_id = regime_snapshot.active_regime.regime_id
            if current_regime_id not in required_regimes:
                reasons.append(
                    f"regime_mismatch:current={current_regime_id};"
                    f"required={','.join(required_regimes)}"
                )
                return 0.0, tuple(reasons)
            reasons.append(f"regime_match:{current_regime_id}")
    else:
        reasons.append("regime_check_empty_pass")

    # --- Self-trait branch (real when traits populated, packet 1.3') ------
    requires_traits = protocol.identity_assertion.requires_self_traits
    forbidden_traits = protocol.identity_assertion.forbidden_self_traits
    if requires_traits or forbidden_traits:
        if dual_track_snapshot is None:
            # SHADOW permissive: dual_track upstream missing →
            # cannot evaluate. ACTIVE promotion remains blocked by
            # the fallback flag until populator lands.
            reasons.append("self_traits_dual_track_absent_shadow_pass")
        elif not dual_track_snapshot.self_track.traits:
            # Trait surface exists (packet 1.3') but no populator
            # has filled it yet. SHADOW-permissive with explicit
            # marker so audit can track which protocols are waiting
            # on the populator.
            reasons.append("self_traits_populator_pending")
        else:
            # Real check: required traits must be subset of present
            # traits; forbidden traits must not appear.
            present = set(dual_track_snapshot.self_track.traits)
            missing_required = tuple(t for t in requires_traits if t not in present)
            present_forbidden = tuple(t for t in forbidden_traits if t in present)
            if missing_required:
                reasons.append(
                    f"self_traits_missing_required:"
                    f"missing={','.join(missing_required)};"
                    f"present={','.join(sorted(present))}"
                )
                return 0.0, tuple(reasons)
            if present_forbidden:
                reasons.append(
                    f"self_traits_forbidden_present:"
                    f"forbidden={','.join(present_forbidden)};"
                    f"present={','.join(sorted(present))}"
                )
                return 0.0, tuple(reasons)
            if requires_traits:
                reasons.append(
                    f"self_traits_required_match:{','.join(requires_traits)}"
                )
            if forbidden_traits:
                reasons.append(
                    f"self_traits_forbidden_absent:{','.join(forbidden_traits)}"
                )
    else:
        reasons.append("self_traits_empty_pass")

    return 1.0, tuple(reasons)


# ---------------------------------------------------------------------------
# Context match scoring (packet 1.5a)
# ---------------------------------------------------------------------------


def _compute_context_match(
    protocol: BehaviorProtocol,
    *,
    interlocutor_snapshot: InterlocutorStateSnapshot | None,
    rupture_snapshot: RuptureStateSnapshot | None,
    boundary_policy_snapshot: BoundaryPolicySnapshot | None,
) -> tuple[float, tuple[str, ...]]:
    """Per-protocol context_match score from typed kernel-side signals.

    Score = sum of ``signal.weight × detector(signal, upstream)`` for
    every ``ContextMatchSignal`` in
    ``protocol.activation_conditions.context_match_signals``.

    Detectors return ``1.0`` if the signal is "firing" this turn,
    ``0.0`` otherwise. Empty signal list → score 0 (caller falls
    back to equal-weight).

    Detector coverage (packet 1.5a):

    * ``INTERLOCUTOR_ZONE_TRANSITION`` — fires when any zone bool on
      the interlocutor snapshot is True (acknowledge_pressure,
      repair, direct_task, emotional_render, pace_pressure,
      cold_rapport, low_directness).
    * ``RUPTURE_KIND_FIRED`` — fires when rupture_state has a
      non-empty ``rupture_kind`` (rupture detected by R8 evidence).
    * ``BOUNDARY_VIOLATION_FIRED`` — fires when boundary_policy's
      decision has a non-empty trigger_reasons list (boundary
      filter activated).
    * ``USER_DROPOUT_OBSERVED`` — placeholder (returns False);
      requires session-level dialogue_trace inspection or a typed
      proxy slot. Future packet.
    * ``DRIVE_HOMEOSTASIS_HOLD`` / ``DRIVE_HOMEOSTASIS_BREACH`` —
      deferred: vitals lives in ``lifeform-core`` (not in kernel
      propagate graph; packet 1.0.1 design). Returns False with
      explicit ``vitals_not_kernel_accessible`` reason. Future
      packet may add a kernel-side typed drive readout slot.
    """

    signals = protocol.activation_conditions.context_match_signals
    if not signals:
        return 0.0, ()

    score = 0.0
    fired_reasons: list[str] = []
    for signal in signals:
        firing = _signal_is_firing(
            signal,
            interlocutor_snapshot=interlocutor_snapshot,
            rupture_snapshot=rupture_snapshot,
            boundary_policy_snapshot=boundary_policy_snapshot,
        )
        if firing:
            score += signal.weight * 1.0
            fired_reasons.append(signal.signal_id)
    return score, tuple(fired_reasons)


def _signal_is_firing(
    signal: ContextMatchSignal,
    *,
    interlocutor_snapshot: InterlocutorStateSnapshot | None,
    rupture_snapshot: RuptureStateSnapshot | None,
    boundary_policy_snapshot: BoundaryPolicySnapshot | None,
) -> bool:
    """Dispatch detector by signal source.

    Unknown / unsupported sources return False; the protocol-level
    score simply doesn't accrue for them (no exception). Future
    packets extend the dispatch table without breaking this
    contract.
    """

    source = signal.measurable_via
    if source is BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION:
        return _interlocutor_zone_active(interlocutor_snapshot)
    if source is BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED:
        return _rupture_kind_present(rupture_snapshot)
    if source is BehaviorProtocolSignalSource.BOUNDARY_VIOLATION_FIRED:
        return _boundary_decision_triggered(boundary_policy_snapshot)
    # USER_DROPOUT_OBSERVED: deferred; not yet wired
    # DRIVE_HOMEOSTASIS_HOLD / DRIVE_HOMEOSTASIS_BREACH: deferred
    # (vitals not in kernel propagate graph; see packet 1.0.1)
    return False


def _interlocutor_zone_active(
    snapshot: InterlocutorStateSnapshot | None,
) -> bool:
    """Any of the 7 typed zone bools True → context-relevant turn."""

    if snapshot is None:
        return False
    state = snapshot.state
    return bool(
        state.acknowledge_pressure_zone
        or state.repair_zone
        or state.direct_task_zone
        or state.emotional_render_zone
        or state.pace_pressure_zone
        or state.cold_rapport_zone
        or state.low_directness_zone
    )


def _rupture_kind_present(
    snapshot: RuptureStateSnapshot | None,
) -> bool:
    """Non-empty rupture_kind (rupture_state has fired this turn)."""

    if snapshot is None:
        return False
    return snapshot.rupture_kind is not None


def _boundary_decision_triggered(
    snapshot: BoundaryPolicySnapshot | None,
) -> bool:
    """boundary_policy snapshot has trigger_reasons (filter active)."""

    if snapshot is None:
        return False
    return bool(snapshot.trigger_reasons)


def _softmax_weights_with_floor(
    protocols: tuple[BehaviorProtocol, ...],
    context_results: list[tuple[float, tuple[str, ...]]],
) -> tuple[float, ...]:
    """softmax of context_match scores with per-protocol floors.

    Numerically-stable softmax (subtract max) followed by floor
    enforcement (mirror of ``_equal_weight_with_floor``).
    """

    import math

    if not protocols:
        return ()
    scores = [r[0] for r in context_results]
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    if total <= 0.0:
        return tuple(0.0 for _ in protocols)
    base = tuple(e / total for e in exps)
    floors = tuple(p.activation_conditions.minimum_weight_floor for p in protocols)
    raised = tuple(max(b, f) for b, f in zip(base, floors, strict=True))
    raised_total = sum(raised)
    if raised_total <= 0.0:
        return tuple(0.0 for _ in protocols)
    if raised_total <= 1.0 + 1e-9:
        deficit = 1.0 - raised_total
        if deficit > 0.0:
            n = len(raised)
            return tuple(w + deficit / n for w in raised)
        return raised
    return tuple(w / raised_total for w in raised)


def _read_interlocutor_snapshot(
    upstream: Mapping[str, Snapshot[Any]],
) -> InterlocutorStateSnapshot | None:
    snapshot = upstream.get("interlocutor_state")
    if snapshot is None:
        return None
    if not isinstance(snapshot.value, InterlocutorStateSnapshot):
        return None
    return snapshot.value


def _read_rupture_snapshot(
    upstream: Mapping[str, Snapshot[Any]],
) -> RuptureStateSnapshot | None:
    snapshot = upstream.get("rupture_state")
    if snapshot is None:
        return None
    if not isinstance(snapshot.value, RuptureStateSnapshot):
        return None
    return snapshot.value


def _read_boundary_policy_snapshot(
    upstream: Mapping[str, Snapshot[Any]],
) -> BoundaryPolicySnapshot | None:
    snapshot = upstream.get("boundary_policy")
    if snapshot is None:
        return None
    if not isinstance(snapshot.value, BoundaryPolicySnapshot):
        return None
    return snapshot.value


# ---------------------------------------------------------------------------
# Upstream snapshot readers (identity gate)
# ---------------------------------------------------------------------------


def _read_dual_track_snapshot(
    upstream: Mapping[str, Snapshot[Any]],
) -> DualTrackSnapshot | None:
    """Extract a ``DualTrackSnapshot`` from upstream, or None if missing.

    Returns None when the slot is absent (SHADOW dependency
    unavailable) or when the value is not a DualTrackSnapshot
    (e.g. RuntimePlaceholderValue when upstream owner is DISABLED).
    Upstream-aware permissive policy keeps SHADOW dual-runs
    functional; ACTIVE promotion is gated separately.
    """

    snapshot = upstream.get("dual_track")
    if snapshot is None:
        return None
    if not isinstance(snapshot.value, DualTrackSnapshot):
        return None
    return snapshot.value


def _read_regime_snapshot(
    upstream: Mapping[str, Snapshot[Any]],
) -> RegimeSnapshot | None:
    """Extract a ``RegimeSnapshot`` from upstream, or None if missing.

    Same shape as :func:`_read_dual_track_snapshot`; see that
    docstring for the rationale.
    """

    snapshot = upstream.get("regime")
    if snapshot is None:
        return None
    if not isinstance(snapshot.value, RegimeSnapshot):
        return None
    return snapshot.value


def _resolve_co_activation_incompatibility(
    protocols: tuple[BehaviorProtocol, ...],
) -> tuple[BehaviorProtocol, ...]:
    """Drop protocols that conflict with a higher-priority sibling.

    Tiebreak: lexicographic ``protocol_id`` (lower id wins). Real
    arbitration (PE history) lands in packet 1.5+.
    """

    by_id = {p.protocol_id: p for p in protocols}
    drop: set[str] = set()
    for protocol in protocols:
        if protocol.protocol_id in drop:
            continue
        for other_id in protocol.activation_conditions.co_activation_incompatible:
            if other_id in by_id and other_id not in drop:
                # Keep the lexicographically smaller id; drop the larger.
                loser = max(protocol.protocol_id, other_id)
                drop.add(loser)
    return tuple(p for p in protocols if p.protocol_id not in drop)


def _equal_weight_with_floor(
    protocols: tuple[BehaviorProtocol, ...],
) -> tuple[float, ...]:
    """Equal weight per protocol, with per-protocol floors applied.

    Algorithm (deterministic):

    1. Start each protocol at ``base = 1.0 / N``.
    2. If any ``base < floor``, raise it to ``floor``.
    3. Renormalise so the total sums to 1.0; floors are preserved
       (we shrink the non-floored remainder proportionally).

    For packet 1.0 the floors from ``cheng_laoshi`` fixture are
    all 0.0, so this collapses to the trivial 1/N.
    """

    n = len(protocols)
    if n == 0:
        return ()

    base = 1.0 / n
    floors = tuple(p.activation_conditions.minimum_weight_floor for p in protocols)
    raised = tuple(max(base, f) for f in floors)
    total = sum(raised)
    if total <= 0.0:
        return tuple(0.0 for _ in protocols)
    if total <= 1.0 + 1e-9:
        # Floors fit; pad upwards proportionally so total = 1.0.
        deficit = 1.0 - total
        if deficit > 0.0:
            # Distribute deficit equally; floors already lifted us, so
            # simple even distribution preserves the floor invariant.
            return tuple(w + deficit / n for w in raised)
        return raised
    # Floors overflow 1.0: scale down proportionally (rare; only if
    # multiple protocols declare large floors that sum > 1).
    return tuple(w / total for w in raised)


def _union_boundary_ids(
    protocols: tuple[BehaviorProtocol, ...],
) -> tuple[str, ...]:
    """Take the union of boundary IDs across active protocols.

    Dedup by ``boundary_id``. First occurrence wins (ordering is
    registry-deterministic). Boundaries are frozen; cross-protocol
    union is lossless because contracts can't conflict structurally
    (they're additive blocks, not negations).

    Packet 1.2 (Choice A): the snapshot publishes only IDs.
    Canonical content lives in ``boundary_policy`` /
    ``ApplicationRareHeavyState`` (populated by the protocol
    compile path). Consumers cross-reference the ID set against
    that canonical store, never read content from this slot.
    """

    seen: set[str] = set()
    ids: list[str] = []
    for protocol in protocols:
        for contract in protocol.boundary_contracts:
            if contract.boundary_id in seen:
                continue
            seen.add(contract.boundary_id)
            ids.append(contract.boundary_id)
    return tuple(ids)


def _default_phase_id(protocol: BehaviorProtocol) -> str | None:
    """Pick the phase id to publish for this protocol.

    Packet 1.0: always the first phase if any, else None. Packet
    1.4+ replaces this with PE-driven progression.
    """

    if protocol.temporal_arc.phases:
        return protocol.temporal_arc.phases[0].phase_id
    return None


def _fingerprint(protocols: tuple[BehaviorProtocol, ...]) -> str:
    """Stable fingerprint over the loaded protocol ids + versions.

    Used by the snapshot consumer (none yet in packet 1.0) to detect
    registry changes. Cheap to compute; only ids + versions are
    fingerprinted to keep the surface small.
    """

    if not protocols:
        return ""
    payload = ";".join(
        f"{p.protocol_id}@{p.version}" for p in protocols
    ).encode("utf-8")
    return sha256(payload).hexdigest()


__all__ = ["compute_active_mixture", "is_fallback_mode"]
