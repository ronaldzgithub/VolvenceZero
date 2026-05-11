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

from volvence_zero.application.types import (
    BoundaryPolicySnapshot,
    RetrievalPolicySnapshot,
)
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
from volvence_zero.rupture_state import RuptureKind, RuptureStateSnapshot
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
_ACTIVATION_CONTROLLER_FALLBACK_MODE: bool = False


def is_fallback_mode() -> bool:
    """Whether ``compute_active_mixture`` is still in packet-1.0 fallback.

    Read by ``ProtocolRegistryModule.__init__`` to gate ACTIVE
    wiring. True means the activation logic is still missing one
    or more upgrades from the SHADOW → ACTIVE checklist; downstream
    consumers cannot trust the outputs as a learned posterior. False
    is set only by future packets that have landed all required
    machinery (see module docstring).

    History (in landing order):

    * Packet 1.3 series: ``identity_gate`` fully real.
    * Packet 1.5a: typed ``context_match`` partial (3 detectors).
    * Packet 1.5b: ``pe_utility`` rolling EMA real.
    * Packet 1.5c-i: PE-history-driven incompatibility arbitration.
    * Packet 1.5c-iii: α/β online learning (REINFORCE-style proxy).
    * **Packet 1.5a': flag flipped to False.** All ACTIVE-blocking
      checklist items now closed:

      * ``USER_DROPOUT_OBSERVED`` detector real (reads
        ``rupture_state.rupture_kind == ABANDONED``).
      * ``REGIME_TRANSITION_RECENT`` signal source added + detector
        wired (reads ``regime.turns_in_current_regime``).
      * ``RETRIEVAL_HITS_PRESENT`` signal source added + detector
        wired (reads ``retrieval_policy.knowledge_domains``).

    Remaining deferrals (NOT blockers, do not flip flag back):

    * R8-clean ``ProtocolPerformanceModule`` split deferred —
      ``_pe_utility`` and ``_alpha`` / ``_beta`` still co-located
      with the registry (packet 1.5c-ii). This is a refactor and
      cannot regress behaviour.
    * DRIVE detectors permanently deferred (vitals layering,
      packet 1.0.1). The DRIVE enum members remain in the schema
      and detectors return False; protocols can still declare
      them but they don't fire kernel-side.

    With the flag False, ``ProtocolRegistryModule`` may be wired
    at ``WiringLevel.ACTIVE``. Existing tests that previously
    asserted the FallbackActivationActiveError fail-loud on
    ACTIVE construction now assert the inverse — see
    ``tests/contracts/test_protocol_runtime_active_gate_guard.py``.
    """

    return _ACTIVATION_CONTROLLER_FALLBACK_MODE


def compute_active_mixture(
    *,
    loaded_protocols: tuple[BehaviorProtocol, ...],
    upstream: Mapping[str, Snapshot[Any]],
    pe_utility_by_id: Mapping[str, float] | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    audit_context_scores: dict[str, float] | None = None,
    phase_by_id: Mapping[str, str] | None = None,
) -> ActiveMixtureSnapshot:
    """Build the per-turn ``ActiveMixtureSnapshot`` from the registry.

    Args:
        loaded_protocols: All protocols currently registered.
            Ordering is registry-deterministic (sorted by id).
        upstream: Upstream snapshots available to the module.
            Packet 1.3a consumes ``dual_track`` and ``regime`` for
            identity-gate evaluation. Packet 1.5a consumes
            ``interlocutor_state`` / ``rupture_state`` /
            ``boundary_policy`` for typed context_match scoring.
            Packet 1.5b consumes ``prediction_error`` indirectly via
            ``pe_utility_by_id`` (see below).
        pe_utility_by_id: Per-protocol rolling PE utility provided
            by the owner (``ProtocolRegistryModule``). Each entry
            is a clamped EMA of attributed signed_reward in
            ``[-1, 1]``. ``None`` (or missing key) → 0.0
            contribution for that protocol. The owner is
            responsible for accumulation; this function only does
            the per-turn read-out into the activation formula.
        alpha: Coefficient on ``context_match_i`` in the raw_score.
            Packet 1.5c-iii: owner-side online learning makes this
            evolve based on signed_reward × range(context_match)
            across active protocols. Default ``1.0`` for backward
            compatibility with packets 1.0–1.5b callers (and
            tests / external callers that don't have α/β state).
        beta: Coefficient on ``pe_utility_i``. Symmetric to alpha;
            evolved by the owner from ``signed_reward ×
            range(pe_utility)`` across active protocols.
        audit_context_scores: Optional output dict; when supplied
            the function populates it with
            ``{protocol_id: context_match_score}`` for every
            ELIGIBLE protocol (post identity-gate, post co-act
            arbitration). The owner uses this snapshot in the
            next turn's α/β learning step (the values are needed
            to compute ``range(context_match)``). When ``None``
            the function does not touch any side state, preserving
            the legacy pure-function shape.

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

    # Packet 1.5c-i: PE-history-driven incompatibility tiebreak.
    # ``pe_utility_by_id`` is the same mapping fed into the softmax
    # below; passing it here means a protocol with stronger past
    # performance wins the A ↔ B drop decision instead of the
    # arbitrary lex order. Cold-start (empty EMA) and protocols with
    # equal pe_utility fall back to lex, preserving determinism.
    eligible = _resolve_co_activation_incompatibility(
        eligible_protocols,
        pe_utility_by_id=pe_utility_by_id,
    )

    # Packet 1.5a / 1.5a': compute typed context_match scores per
    # protocol. When all scores are 0 (no signals in any
    # protocol's activation_conditions, or no signals firing), the
    # scoring step collapses to ``equal_weight_with_floor`` —
    # preserving the packet 1.0+ uniform-weight contract for
    # cheng_laoshi-shape protocols. When at least one score > 0,
    # softmax of scores produces real differential weighting.
    interlocutor_snapshot = _read_interlocutor_snapshot(upstream)
    rupture_snapshot = _read_rupture_snapshot(upstream)
    boundary_policy_snapshot = _read_boundary_policy_snapshot(upstream)
    retrieval_policy_snapshot = _read_retrieval_policy_snapshot(upstream)
    # Packet 7.0: optionally read commitment snapshot for COMMITMENT_*
    # detectors. SHADOW-tolerant — duck-typed read.
    commitment_snap = upstream.get("commitment")
    commitment_state = commitment_snap.value if commitment_snap is not None else None

    context_results: list[tuple[float, tuple[str, ...]]] = []
    max_context_score = 0.0
    for protocol in eligible:
        score, reasons = _compute_context_match(
            protocol,
            interlocutor_snapshot=interlocutor_snapshot,
            rupture_snapshot=rupture_snapshot,
            boundary_policy_snapshot=boundary_policy_snapshot,
            regime_snapshot=regime_snapshot,
            retrieval_policy_snapshot=retrieval_policy_snapshot,
            commitment_state=commitment_state,
        )
        context_results.append((score, reasons))
        max_context_score = max(max_context_score, score)

    # Packet 1.5b: per-protocol rolling PE utility from owner-side
    # accounting (signed_reward EMA). 0.0 by default → packet 1.5a
    # behaviour preserved when the owner doesn't supply pe history
    # (e.g. tests calling ``compute_active_mixture`` directly).
    pe_lookup: Mapping[str, float] = pe_utility_by_id or {}
    pe_utilities: list[float] = [
        float(pe_lookup.get(p.protocol_id, 0.0)) for p in eligible
    ]
    max_abs_pe = max((abs(u) for u in pe_utilities), default=0.0)

    # Packet 1.5c-iii audit: when caller (owner) supplies a dict,
    # snapshot the per-eligible-protocol context_match score so it
    # can compute ``range(context_match)`` for the next turn's α
    # learning step. Done before the raw_score combination so the
    # captured value is the bare context_match, not α·context_match.
    if audit_context_scores is not None:
        for protocol, (score, _reasons) in zip(
            eligible, context_results, strict=True
        ):
            audit_context_scores[protocol.protocol_id] = score

    # Packet 1.5c-iii: raw_score uses owner-supplied α / β.
    # Default α=β=1.0 reproduces packet 1.5b behaviour. The owner
    # learns α / β online from PE feedback (REINFORCE-style proxy
    # gradient over range(signal)) — see ``ProtocolRegistryModule._update_alpha_beta``.
    raw_scores = [
        alpha * context_results[i][0] + beta * pe_utilities[i]
        for i in range(len(eligible))
    ]
    has_signal = max_context_score > 0.0 or max_abs_pe > 0.0

    if has_signal:
        weights = _softmax_weights_with_floor(
            eligible, [(s, ()) for s in raw_scores]
        )
        weighting_kind = ActivationReasonKind.CONTEXT_MATCH
        weighting_detail_template = (
            f"packet 1.5c-iii: softmax(α·context_match + β·pe_utility), "
            f"α={alpha:.3f}, β={beta:.3f} (PE-driven incompatibility "
            f"arbitration via packet 1.5c-i)"
        )
    else:
        weights = _equal_weight_with_floor(eligible)
        weighting_kind = ActivationReasonKind.EQUAL_WEIGHT_FALLBACK
        weighting_detail_template = (
            "no context_match signal fired and no PE history; equal "
            "weight across eligible protocols "
            f"(current α={alpha:.3f}, β={beta:.3f}; drive coupling "
            "permanently deferred per packet 1.0.1)"
        )

    active_entries: tuple[ActiveProtocolEntry, ...] = tuple(
        ActiveProtocolEntry(
            protocol_id=protocol.protocol_id,
            activation_weight=weight,
            current_phase_id=_resolve_phase_id(
                protocol, phase_by_id=phase_by_id
            ),
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
                        + (
                            f"; pe_utility={pe_utilities[idx]:+.3f}"
                            if pe_utilities[idx] != 0.0
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
        f"max_context_score={max_context_score:.3f}; "
        f"max_abs_pe_utility={max_abs_pe:.3f}"
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
    regime_snapshot: RegimeSnapshot | None = None,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None = None,
    commitment_state: Any | None = None,
) -> tuple[float, tuple[str, ...]]:
    """Per-protocol context_match score from typed kernel-side signals.

    Score = sum of ``signal.weight × detector(signal, upstream)`` for
    every ``ContextMatchSignal`` in
    ``protocol.activation_conditions.context_match_signals``.

    Detectors return ``1.0`` if the signal is "firing" this turn,
    ``0.0`` otherwise. Empty signal list → score 0 (caller falls
    back to equal-weight).

    Detector coverage (packet 1.5a' — full closure of condition 3):

    * ``INTERLOCUTOR_ZONE_TRANSITION`` — fires when any zone bool on
      the interlocutor snapshot is True.
    * ``RUPTURE_KIND_FIRED`` — fires when rupture_state has a
      non-empty ``rupture_kind``.
    * ``BOUNDARY_VIOLATION_FIRED`` — fires when boundary_policy
      decision has non-empty ``trigger_reasons``.
    * ``USER_DROPOUT_OBSERVED`` (1.5a') — fires when
      ``rupture_state.rupture_kind == ABANDONED`` (the canonical
      typed evidence for user disengagement; mirrors
      ``EXTERNAL_OUTCOME_TO_RUPTURE_KIND[ABANDONED]``).
    * ``REGIME_TRANSITION_RECENT`` (1.5a') — fires when
      ``regime.turns_in_current_regime <= 1`` (active regime
      just transitioned this turn).
    * ``RETRIEVAL_HITS_PRESENT`` (1.5a') — fires when
      ``retrieval_policy.knowledge_domains`` is non-empty (the
      retrieval policy is requesting any domain lookup).
    * ``DRIVE_HOMEOSTASIS_HOLD`` / ``DRIVE_HOMEOSTASIS_BREACH`` —
      permanently deferred: vitals lives in ``lifeform-core`` (not
      in kernel propagate graph; packet 1.0.1 design). Returns
      False; protocols may still declare them but they don't
      contribute kernel-side.
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
            regime_snapshot=regime_snapshot,
            retrieval_policy_snapshot=retrieval_policy_snapshot,
            commitment_state=commitment_state,
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
    regime_snapshot: RegimeSnapshot | None = None,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None = None,
    commitment_state: Any | None = None,
) -> bool:
    interlocutor_state = interlocutor_snapshot
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
    if source is BehaviorProtocolSignalSource.USER_DROPOUT_OBSERVED:
        return _user_dropout_observed(rupture_snapshot)
    if source is BehaviorProtocolSignalSource.REGIME_TRANSITION_RECENT:
        return _regime_transition_recent(regime_snapshot)
    if source is BehaviorProtocolSignalSource.USER_REPLY_LATENCY:
        # Proxy: low engagement_intensity AND low directness combined
        # → user is replying slowly / hesitantly. Without a direct
        # latency signal we use these as the closest available proxy.
        if interlocutor_state is None:
            return False
        s = interlocutor_state.state
        return bool(
            s.engagement_intensity < 0.35 and s.directness < 0.45
        )
    if source is BehaviorProtocolSignalSource.USER_REPLY_LENGTH:
        # Proxy: high engagement_intensity AND high self_disclosure
        # → user is offering longer / richer reply.
        if interlocutor_state is None:
            return False
        s = interlocutor_state.state
        return bool(
            s.engagement_intensity >= 0.7 and s.self_disclosure_level >= 0.5
        )
    if source is BehaviorProtocolSignalSource.USER_INITIATIVE_QUESTION:
        # Proxy: high engagement + high directness + low resistance
        # → user is leaning in and asking back.
        if interlocutor_state is None:
            return False
        s = interlocutor_state.state
        return bool(
            s.engagement_intensity >= 0.7
            and s.directness >= 0.6
            and s.resistance_level < 0.4
        )
    if source is BehaviorProtocolSignalSource.COMMITMENT_FULFILLED:
        if commitment_state is None:
            return False
        return bool(getattr(commitment_state, "honored_commitment_refs", ()))
    if source is BehaviorProtocolSignalSource.COMMITMENT_BROKEN:
        if commitment_state is None:
            return False
        return bool(getattr(commitment_state, "at_risk_commitments", ()))
    if source is BehaviorProtocolSignalSource.RETRIEVAL_HITS_PRESENT:
        return _retrieval_hits_present(retrieval_policy_snapshot)
    # DRIVE_HOMEOSTASIS_HOLD / DRIVE_HOMEOSTASIS_BREACH: permanently
    # deferred (vitals not in kernel propagate graph; see packet
    # 1.0.1 design decision).
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


def _user_dropout_observed(
    snapshot: RuptureStateSnapshot | None,
) -> bool:
    """``rupture_state.rupture_kind == ABANDONED`` → user dropped out.

    Packet 1.5a' uses the existing rupture_state typed evidence as
    the canonical signal. ``RuptureKind.ABANDONED`` is produced by
    ``EXTERNAL_OUTCOME_TO_RUPTURE_KIND`` mapping
    ``DialogueExternalOutcomeKind.ABANDONED`` (the upstream
    rupture-state owner already does the typed inference). No new
    snapshot read needed.
    """

    if snapshot is None:
        return False
    return snapshot.rupture_kind is RuptureKind.ABANDONED


def _regime_transition_recent(
    snapshot: RegimeSnapshot | None,
) -> bool:
    """Active regime just transitioned this turn (turns_in_current_regime ≤ 1).

    The canonical "post-transition" floor is 1 — the new regime's
    first turn after the switch — but we accept 0 too as a
    cold-start tolerance (e.g. fresh session before any update).
    """

    if snapshot is None:
        return False
    return int(snapshot.turns_in_current_regime) <= 1


def _retrieval_hits_present(
    snapshot: RetrievalPolicySnapshot | None,
) -> bool:
    """Retrieval policy is requesting any knowledge domain lookup.

    "Requesting" = ``knowledge_domains`` is non-empty. Whether
    the actual domain_knowledge owner returns hits this turn is
    a downstream concern; from the protocol's perspective the
    relevance signal is "the retrieval layer thinks this is a
    knowledge-grounded turn".
    """

    if snapshot is None:
        return False
    return bool(snapshot.knowledge_domains)


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


def _read_retrieval_policy_snapshot(
    upstream: Mapping[str, Snapshot[Any]],
) -> RetrievalPolicySnapshot | None:
    snapshot = upstream.get("retrieval_policy")
    if snapshot is None:
        return None
    if not isinstance(snapshot.value, RetrievalPolicySnapshot):
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
    pe_utility_by_id: Mapping[str, float] | None = None,
) -> tuple[BehaviorProtocol, ...]:
    """Drop protocols that conflict with a higher-priority sibling.

    Packet 1.5c-i: PE-history-driven arbitration. Given an
    ``A ↔ B`` incompatibility (one declared on either side), keep
    the protocol with the **higher** rolling ``pe_utility``
    (signed_reward EMA from packet 1.5b). Ties resolve
    lexicographically (lower ``protocol_id`` wins) to keep the
    output deterministic — including the cold-start case where
    every protocol's pe_utility is 0.

    The tiebreak ladder is therefore:

    1. ``pe_utility_a > pe_utility_b`` → keep A
    2. ``pe_utility_a < pe_utility_b`` → keep B
    3. Equal pe_utility → lexicographic ``protocol_id``

    ``pe_utility_by_id`` defaults to an empty mapping; missing
    entries read as 0.0. So tests / external callers can omit the
    argument and get the legacy lex behaviour for free, and a
    cold-start owner with empty EMA dict produces the same output
    as packet 1.5a.

    Note: the iteration order matters for asymmetric declarations.
    The function consumes ``protocols`` in the registry-deterministic
    sort (lex by id), so the **declaration owner** is whichever
    side comes first in lex order. For an A ↔ B pair where only
    A declares B incompatible, A is encountered first; we then
    look up both A's and B's pe_utility to decide which to keep.
    For an A ↔ B pair where both declare the other incompatible,
    we hit the same ladder twice but the result is identical.
    """

    pe_lookup: Mapping[str, float] = pe_utility_by_id or {}
    by_id = {p.protocol_id: p for p in protocols}
    drop: set[str] = set()
    for protocol in protocols:
        if protocol.protocol_id in drop:
            continue
        for other_id in protocol.activation_conditions.co_activation_incompatible:
            if other_id not in by_id or other_id in drop:
                continue
            self_pe = float(pe_lookup.get(protocol.protocol_id, 0.0))
            other_pe = float(pe_lookup.get(other_id, 0.0))
            if self_pe > other_pe:
                # Self has better PE history → drop the other.
                loser = other_id
            elif self_pe < other_pe:
                # Other has better PE history → drop self.
                loser = protocol.protocol_id
            else:
                # Equal pe_utility (incl. cold-start 0/0) → lex
                # tiebreak: keep the smaller id.
                loser = max(protocol.protocol_id, other_id)
            drop.add(loser)
            if loser == protocol.protocol_id:
                # We just dropped the one we were iterating; stop
                # enumerating its other incompatibilities (it's gone).
                break
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


def _resolve_phase_id(
    protocol: BehaviorProtocol,
    *,
    phase_by_id: Mapping[str, str] | None,
) -> str | None:
    """Resolve the current phase id for this protocol.

    Order of preference:

    1. If ``phase_by_id`` is supplied (owner injects it from
       ``protocol_phase`` upstream snapshot, packet 5.0+) and
       it has an entry for this protocol, use that — this is
       the PE-driven phase pointer.
    2. Otherwise fall back to the first declared phase
       (cheng_laoshi default-shape backwards compat for tests
       that don't wire ProtocolPhaseModule).
    3. If neither, return ``None``.
    """

    if phase_by_id is not None:
        phase_id = phase_by_id.get(protocol.protocol_id)
        if phase_id is not None:
            return phase_id
    return _default_phase_id(protocol)


def _default_phase_id(protocol: BehaviorProtocol) -> str | None:
    """Pick the phase id to publish for this protocol.

    Packet 1.0: always the first phase if any, else None. Packet
    5.0+ uses ``_resolve_phase_id`` which falls back to this
    when no ``protocol_phase`` snapshot is available.
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
