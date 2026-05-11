"""Context match scoring (packets 1.5a + 1.5a'): typed signals + softmax weighting.

Asserts the contract for ``activation._compute_context_match`` and
its integration into ``compute_active_mixture``:

* **Empty signal collapse**: when every eligible protocol has empty
  ``context_match_signals``, ``compute_active_mixture`` falls back
  to equal-weight (preserves cheng_laoshi shape across the packet
  boundary).
* **Single-protocol detector firing**: each of the 6 kernel-side
  detectors (interlocutor zone / rupture / boundary / user dropout
  / regime transition / retrieval hits) maps a real upstream
  snapshot to a non-zero context_match score.
* **Differential weighting**: when one protocol's signals fire and
  another's do not, softmax produces strictly differential
  ``activation_weight``s and the activation reason marker switches
  to ``CONTEXT_MATCH``.
* **Permanently deferred detectors**: ``DRIVE_HOMEOSTASIS_*``
  always score 0 — vitals not in kernel propagate graph (packet
  1.0.1 design decision).
* **cheng_laoshi behaviour preserved**: the canonical fixture path
  yields exactly the same snapshot shape as before — ``activation_conditions``
  is empty so no detector ever fires for it.

These tests run on synthetic ``BehaviorProtocol`` fixtures so they
do not depend on lifeform-specific signal vocabulary. Packet
1.5a' added 3 new detectors (USER_DROPOUT_OBSERVED activated,
REGIME_TRANSITION_RECENT new, RETRIEVAL_HITS_PRESENT new).
"""

from __future__ import annotations

from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.types import (
    BoundaryDecision,
    BoundaryPolicySnapshot,
    KnowledgeDepth,
    ProfessionalScope,
    RetrievalPolicySnapshot,
    RiskBand,
)
from volvence_zero.regime.contracts import RegimeIdentity, RegimeSnapshot
from volvence_zero.behavior_protocol import (
    ActivationConditions,
    ActivationReasonKind,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    ContextMatchSignal,
)
from volvence_zero.interlocutor.contracts import (
    InterlocutorState,
    InterlocutorStateSnapshot,
    with_zones,
)
from volvence_zero.protocol_runtime import compute_active_mixture
from volvence_zero.protocol_runtime.activation import _compute_context_match
from volvence_zero.rupture_state.contracts import (
    RuptureContributingSignal,
    RuptureEvidenceSource,
    RuptureKind,
    RuptureStateSnapshot,
    _bootstrap_rupture_snapshot,
)
from volvence_zero.runtime import Snapshot


# ---------------------------------------------------------------------------
# Helpers — synthetic protocols
# ---------------------------------------------------------------------------


def _cheng_laoshi_protocol() -> BehaviorProtocol:
    return growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )


def _retag_signals(
    protocol: BehaviorProtocol,
    *signals: ContextMatchSignal,
) -> BehaviorProtocol:
    """Replace ``activation_conditions.context_match_signals`` only."""

    new_conditions = ActivationConditions(
        context_match_signals=tuple(signals),
        co_activation_compatible=protocol.activation_conditions.co_activation_compatible,
        co_activation_incompatible=protocol.activation_conditions.co_activation_incompatible,
        minimum_weight_floor=protocol.activation_conditions.minimum_weight_floor,
    )
    return _replace(protocol, activation_conditions=new_conditions)


def _retag_id(protocol: BehaviorProtocol, new_id: str) -> BehaviorProtocol:
    return _replace(protocol, protocol_id=new_id)


# ---------------------------------------------------------------------------
# Helpers — synthetic upstream snapshots
# ---------------------------------------------------------------------------


def _make_interlocutor_snapshot(
    *, fire_acknowledge_pressure: bool = False
) -> Snapshot[InterlocutorStateSnapshot]:
    """Build an ``InterlocutorStateSnapshot``.

    When ``fire_acknowledge_pressure=True``, axis values are tuned
    above the canonical thresholds so
    ``acknowledge_pressure_zone`` resolves True. Default produces a
    cold-start neutral state (all zones False).
    """

    if fire_acknowledge_pressure:
        state = InterlocutorState(
            emotional_weight=0.80,
            resistance_level=0.55,
            trust_signal=-0.20,
            readout_confidence=0.85,
            rationale="test: high emotional + resistance + negative trust",
        )
    else:
        # Tuned to leave every zone False while readout_confidence is
        # above ``min_confidence`` (so the snapshot is "real"). Keeping
        # resistance below 0.30, trust above 0.05, pace below 0.65, etc.
        state = InterlocutorState(
            engagement_intensity=0.20,
            self_disclosure_level=0.30,
            task_focus_level=0.40,
            emotional_weight=0.30,
            cognitive_engagement=0.40,
            resistance_level=0.20,
            openness_to_guidance=0.50,
            directness=0.55,
            trust_signal=0.30,
            stability=0.55,
            rapport_warmth=0.55,
            pace_pressure=0.40,
            readout_confidence=0.85,
            rationale="test: neutral zones",
        )
    state = with_zones(state)
    snapshot = InterlocutorStateSnapshot(
        state=state, description="test interlocutor"
    )
    return Snapshot(
        slot_name="interlocutor_state",
        owner="InterlocutorReadoutModule",
        version=1,
        timestamp_ms=0,
        value=snapshot,
    )


def _make_rupture_snapshot(
    *, fire: bool = False
) -> Snapshot[RuptureStateSnapshot]:
    if not fire:
        snapshot = _bootstrap_rupture_snapshot()
    else:
        snapshot = RuptureStateSnapshot(
            rupture_signal_strength=0.7,
            rupture_kind=RuptureKind.MISREAD,
            confidence=0.75,
            internal_suspected_only=False,
            evidence_sources=(RuptureEvidenceSource.EXTERNAL_USER,),
            contributing_signals=(
                RuptureContributingSignal(
                    source=RuptureEvidenceSource.EXTERNAL_USER,
                    signal_strength=0.7,
                    confidence=0.75,
                    kind_hint=RuptureKind.MISREAD,
                    detail="test misread",
                ),
            ),
            description="test rupture",
        )
    return Snapshot(
        slot_name="rupture_state",
        owner="RuptureStateModule",
        version=1,
        timestamp_ms=0,
        value=snapshot,
    )


def _make_boundary_policy_snapshot(
    *, fire: bool = False
) -> Snapshot[BoundaryPolicySnapshot]:
    decision = BoundaryDecision(
        decision_id="boundary:test",
        risk_band=RiskBand.MEDIUM,
        professional_scope=ProfessionalScope.GENERAL_SUPPORT,
        answer_depth_limit="support-first",
        citation_required=False,
        clarification_required=False,
        refer_out_required=False,
        blocked_topics=(),
        required_disclaimers=(),
        description="test boundary decision",
    )
    if fire:
        snapshot = BoundaryPolicySnapshot(
            active_decision=decision,
            trigger_reasons=("medical_advice_flag",),
            description="test boundary triggered",
        )
    else:
        snapshot = BoundaryPolicySnapshot(
            active_decision=decision,
            trigger_reasons=(),
            description="test boundary clear",
        )
    return Snapshot(
        slot_name="boundary_policy",
        owner="BoundaryPolicyModule",
        version=1,
        timestamp_ms=0,
        value=snapshot,
    )


# ---------------------------------------------------------------------------
# _compute_context_match: empty signals
# ---------------------------------------------------------------------------


def test_empty_context_match_signals_score_zero() -> None:
    protocol = _cheng_laoshi_protocol()
    assert protocol.activation_conditions.context_match_signals == ()

    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    )

    assert score == 0.0
    assert reasons == ()


# ---------------------------------------------------------------------------
# _compute_context_match: single-source detectors
# ---------------------------------------------------------------------------


def test_interlocutor_zone_signal_fires_on_active_zone() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="ack_pressure_match",
            measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
            weight=1.5,
            description="match when interlocutor under acknowledge pressure",
        ),
    )

    interlocutor = _make_interlocutor_snapshot(fire_acknowledge_pressure=True)
    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=interlocutor.value,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    )

    assert score == 1.5
    assert reasons == ("ack_pressure_match",)


def test_interlocutor_zone_signal_does_not_fire_on_neutral_zone() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="ack_pressure_match",
            measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
            weight=1.5,
        ),
    )

    interlocutor = _make_interlocutor_snapshot(fire_acknowledge_pressure=False)
    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=interlocutor.value,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    )

    assert score == 0.0
    assert reasons == ()


def test_rupture_signal_fires_on_resolved_rupture_kind() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="rupture_match",
            measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
            weight=2.0,
        ),
    )

    rupture = _make_rupture_snapshot(fire=True)
    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=rupture.value,
        boundary_policy_snapshot=None,
    )

    assert score == 2.0
    assert reasons == ("rupture_match",)


def test_rupture_signal_does_not_fire_on_no_rupture() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="rupture_match",
            measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
            weight=2.0,
        ),
    )

    rupture = _make_rupture_snapshot(fire=False)
    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=rupture.value,
        boundary_policy_snapshot=None,
    )

    assert score == 0.0
    assert reasons == ()


def test_boundary_signal_fires_on_triggered_decision() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="boundary_match",
            measurable_via=BehaviorProtocolSignalSource.BOUNDARY_VIOLATION_FIRED,
            weight=0.75,
        ),
    )

    boundary = _make_boundary_policy_snapshot(fire=True)
    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=boundary.value,
    )

    assert score == 0.75
    assert reasons == ("boundary_match",)


def test_boundary_signal_does_not_fire_on_allow() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="boundary_match",
            measurable_via=BehaviorProtocolSignalSource.BOUNDARY_VIOLATION_FIRED,
            weight=0.75,
        ),
    )

    boundary = _make_boundary_policy_snapshot(fire=False)
    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=boundary.value,
    )

    assert score == 0.0
    assert reasons == ()


# ---------------------------------------------------------------------------
# _compute_context_match: deferred detectors (vitals + user_dropout)
# ---------------------------------------------------------------------------


def test_drive_signals_score_zero_deferred() -> None:
    """Vitals not in kernel propagate graph (packet 1.0.1)."""
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="drive_hold",
            measurable_via=BehaviorProtocolSignalSource.DRIVE_HOMEOSTASIS_HOLD,
            weight=1.0,
        ),
        ContextMatchSignal(
            signal_id="drive_breach",
            measurable_via=BehaviorProtocolSignalSource.DRIVE_HOMEOSTASIS_BREACH,
            weight=1.0,
        ),
    )

    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    )

    assert score == 0.0
    assert reasons == ()


# ---------------------------------------------------------------------------
# _compute_context_match: USER_DROPOUT_OBSERVED detector (packet 1.5a')
# ---------------------------------------------------------------------------


def _make_abandoned_rupture_snapshot() -> Snapshot[RuptureStateSnapshot]:
    """Build a rupture snapshot with ``rupture_kind == ABANDONED``."""

    snapshot = RuptureStateSnapshot(
        rupture_signal_strength=0.85,
        rupture_kind=RuptureKind.ABANDONED,
        confidence=0.9,
        internal_suspected_only=False,
        evidence_sources=(RuptureEvidenceSource.EXTERNAL_USER,),
        contributing_signals=(
            RuptureContributingSignal(
                source=RuptureEvidenceSource.EXTERNAL_USER,
                signal_strength=0.85,
                confidence=0.9,
                kind_hint=RuptureKind.ABANDONED,
                detail="test abandoned",
            ),
        ),
        description="test abandoned",
    )
    return Snapshot(
        slot_name="rupture_state",
        owner="RuptureStateModule",
        version=1,
        timestamp_ms=0,
        value=snapshot,
    )


def test_user_dropout_signal_fires_on_abandoned_rupture() -> None:
    """Packet 1.5a': USER_DROPOUT_OBSERVED reads rupture_state.ABANDONED."""
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="dropout_match",
            measurable_via=BehaviorProtocolSignalSource.USER_DROPOUT_OBSERVED,
            weight=1.5,
        ),
    )

    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=_make_abandoned_rupture_snapshot().value,
        boundary_policy_snapshot=None,
    )

    assert score == 1.5
    assert reasons == ("dropout_match",)


def test_user_dropout_signal_does_not_fire_on_other_rupture_kinds() -> None:
    """RuptureKind.MISREAD shouldn't fire USER_DROPOUT (only ABANDONED does)."""
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="dropout_match",
            measurable_via=BehaviorProtocolSignalSource.USER_DROPOUT_OBSERVED,
            weight=1.0,
        ),
    )

    misread_rupture = _make_rupture_snapshot(fire=True)
    # Sanity: the existing helper produces MISREAD, not ABANDONED.
    assert misread_rupture.value.rupture_kind is RuptureKind.MISREAD
    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=misread_rupture.value,
        boundary_policy_snapshot=None,
    )

    assert score == 0.0
    assert reasons == ()


def test_user_dropout_signal_does_not_fire_without_rupture() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="dropout_match",
            measurable_via=BehaviorProtocolSignalSource.USER_DROPOUT_OBSERVED,
            weight=1.0,
        ),
    )

    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    )

    assert score == 0.0
    assert reasons == ()


# ---------------------------------------------------------------------------
# _compute_context_match: REGIME_TRANSITION_RECENT detector (packet 1.5a')
# ---------------------------------------------------------------------------


def _make_regime_snapshot(
    *, turns_in_current_regime: int
) -> Snapshot[RegimeSnapshot]:
    active = RegimeIdentity(
        regime_id="test_regime",
        name="test",
        embedding=(0.0,),
        entry_conditions="",
        exit_conditions="",
        historical_effectiveness=0.5,
    )
    snapshot_value = RegimeSnapshot(
        active_regime=active,
        previous_regime=None,
        switch_reason="test",
        candidate_regimes=(),
        turns_in_current_regime=turns_in_current_regime,
        description=f"turns={turns_in_current_regime}",
    )
    return Snapshot(
        slot_name="regime",
        owner="RegimeModule",
        version=1,
        timestamp_ms=0,
        value=snapshot_value,
    )


def test_regime_transition_signal_fires_on_recent_transition() -> None:
    """turns_in_current_regime <= 1 → fires."""
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="regime_match",
            measurable_via=BehaviorProtocolSignalSource.REGIME_TRANSITION_RECENT,
            weight=2.0,
        ),
    )

    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
        regime_snapshot=_make_regime_snapshot(
            turns_in_current_regime=1
        ).value,
    )

    assert score == 2.0
    assert reasons == ("regime_match",)


def test_regime_transition_signal_fires_at_turn_zero() -> None:
    """turns_in_current_regime == 0 cold-start tolerance → fires."""
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="regime_match",
            measurable_via=BehaviorProtocolSignalSource.REGIME_TRANSITION_RECENT,
            weight=1.0,
        ),
    )

    score, _ = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
        regime_snapshot=_make_regime_snapshot(
            turns_in_current_regime=0
        ).value,
    )

    assert score == 1.0


def test_regime_transition_signal_does_not_fire_for_settled_regime() -> None:
    """turns_in_current_regime > 1 → no fire (settled regime)."""
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="regime_match",
            measurable_via=BehaviorProtocolSignalSource.REGIME_TRANSITION_RECENT,
            weight=1.0,
        ),
    )

    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
        regime_snapshot=_make_regime_snapshot(
            turns_in_current_regime=5
        ).value,
    )

    assert score == 0.0
    assert reasons == ()


def test_regime_transition_signal_does_not_fire_when_regime_missing() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="regime_match",
            measurable_via=BehaviorProtocolSignalSource.REGIME_TRANSITION_RECENT,
            weight=1.0,
        ),
    )

    score, _ = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
        regime_snapshot=None,
    )

    assert score == 0.0


# ---------------------------------------------------------------------------
# _compute_context_match: RETRIEVAL_HITS_PRESENT detector (packet 1.5a')
# ---------------------------------------------------------------------------


def _make_retrieval_snapshot(
    *, knowledge_domains: tuple[str, ...]
) -> Snapshot[RetrievalPolicySnapshot]:
    snapshot_value = RetrievalPolicySnapshot(
        knowledge_domains=knowledge_domains,
        experience_domains=(),
        knowledge_weight=0.5,
        experience_weight=0.5,
        world_weight=0.5,
        self_weight=0.5,
        retrieval_depth=KnowledgeDepth.LIGHT,
        citation_required=False,
        jurisdiction_required=False,
        risk_band=RiskBand.LOW,
        regime_id=None,
        abstract_action=None,
        intent_description="test",
        description="test retrieval",
    )
    return Snapshot(
        slot_name="retrieval_policy",
        owner="RetrievalPolicyModule",
        version=1,
        timestamp_ms=0,
        value=snapshot_value,
    )


def test_retrieval_signal_fires_when_knowledge_domains_present() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="retrieval_match",
            measurable_via=BehaviorProtocolSignalSource.RETRIEVAL_HITS_PRESENT,
            weight=1.5,
        ),
    )

    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
        retrieval_policy_snapshot=_make_retrieval_snapshot(
            knowledge_domains=("private_domain_growth",)
        ).value,
    )

    assert score == 1.5
    assert reasons == ("retrieval_match",)


def test_retrieval_signal_does_not_fire_with_empty_knowledge_domains() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="retrieval_match",
            measurable_via=BehaviorProtocolSignalSource.RETRIEVAL_HITS_PRESENT,
            weight=1.5,
        ),
    )

    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
        retrieval_policy_snapshot=_make_retrieval_snapshot(
            knowledge_domains=()
        ).value,
    )

    assert score == 0.0
    assert reasons == ()


def test_retrieval_signal_does_not_fire_when_retrieval_missing() -> None:
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="retrieval_match",
            measurable_via=BehaviorProtocolSignalSource.RETRIEVAL_HITS_PRESENT,
            weight=1.0,
        ),
    )

    score, _ = _compute_context_match(
        protocol,
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
        retrieval_policy_snapshot=None,
    )

    assert score == 0.0


# ---------------------------------------------------------------------------
# _compute_context_match: signal aggregation
# ---------------------------------------------------------------------------


def test_multi_signal_aggregates_only_firing_weights() -> None:
    """Score is sum of weights of ONLY firing signals."""
    protocol = _retag_signals(
        _cheng_laoshi_protocol(),
        ContextMatchSignal(
            signal_id="ack_pressure",
            measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
            weight=1.5,
        ),
        ContextMatchSignal(
            signal_id="rupture_active",
            measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
            weight=2.0,
        ),
        ContextMatchSignal(
            signal_id="boundary_active",
            measurable_via=BehaviorProtocolSignalSource.BOUNDARY_VIOLATION_FIRED,
            weight=0.5,
        ),
    )

    interlocutor = _make_interlocutor_snapshot(fire_acknowledge_pressure=True)
    rupture = _make_rupture_snapshot(fire=False)  # not firing
    boundary = _make_boundary_policy_snapshot(fire=True)

    score, reasons = _compute_context_match(
        protocol,
        interlocutor_snapshot=interlocutor.value,
        rupture_snapshot=rupture.value,
        boundary_policy_snapshot=boundary.value,
    )

    assert score == 1.5 + 0.5
    assert reasons == ("ack_pressure", "boundary_active")


# ---------------------------------------------------------------------------
# compute_active_mixture: empty signals → equal-weight fallback preserved
# ---------------------------------------------------------------------------


def test_empty_signals_collapse_to_equal_weight_fallback() -> None:
    """Multi-protocol mixture with no signals → uniform weight + EQUAL_WEIGHT_FALLBACK marker."""
    p1 = _cheng_laoshi_protocol()
    p2 = _retag_id(_cheng_laoshi_protocol(), "synthetic.peer")

    snapshot = compute_active_mixture(
        loaded_protocols=(p1, p2),
        upstream={},
    )

    assert len(snapshot.active_protocols) == 2
    weights = [entry.activation_weight for entry in snapshot.active_protocols]
    assert weights[0] == weights[1]
    for entry in snapshot.active_protocols:
        kinds = [reason.kind for reason in entry.activation_reasons]
        assert ActivationReasonKind.EQUAL_WEIGHT_FALLBACK in kinds, (
            f"expected EQUAL_WEIGHT_FALLBACK marker (no signals); got "
            f"{kinds}"
        )


def test_cheng_laoshi_alone_unchanged_under_packet_1_5a() -> None:
    """Singleton cheng_laoshi mixture stays at weight 1.0, no CONTEXT_MATCH marker."""
    protocol = _cheng_laoshi_protocol()
    snapshot = compute_active_mixture(
        loaded_protocols=(protocol,),
        upstream={},
    )

    assert len(snapshot.active_protocols) == 1
    entry = snapshot.active_protocols[0]
    assert entry.activation_weight == 1.0
    kinds = [reason.kind for reason in entry.activation_reasons]
    assert ActivationReasonKind.EQUAL_WEIGHT_FALLBACK in kinds
    assert ActivationReasonKind.CONTEXT_MATCH not in kinds


# ---------------------------------------------------------------------------
# compute_active_mixture: differential weighting via softmax
# ---------------------------------------------------------------------------


def test_one_protocol_signal_fires_softmax_produces_differential_weights() -> None:
    """When p1 has a firing signal and p2 doesn't, softmax → p1 > p2."""
    base = _cheng_laoshi_protocol()
    p1 = _retag_signals(
        _retag_id(base, "synthetic.firing_protocol"),
        ContextMatchSignal(
            signal_id="ack_match",
            measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
            weight=2.0,
        ),
    )
    p2 = _retag_id(base, "synthetic.silent_protocol")

    interlocutor = _make_interlocutor_snapshot(fire_acknowledge_pressure=True)
    snapshot = compute_active_mixture(
        loaded_protocols=(p1, p2),
        upstream={"interlocutor_state": interlocutor},
    )

    weights_by_id = {
        entry.protocol_id: entry.activation_weight
        for entry in snapshot.active_protocols
    }
    assert weights_by_id["synthetic.firing_protocol"] > weights_by_id[
        "synthetic.silent_protocol"
    ], (
        f"expected firing protocol to dominate softmax; got {weights_by_id}"
    )

    firing_entry = next(
        e for e in snapshot.active_protocols
        if e.protocol_id == "synthetic.firing_protocol"
    )
    kinds = [reason.kind for reason in firing_entry.activation_reasons]
    assert ActivationReasonKind.CONTEXT_MATCH in kinds, (
        f"expected CONTEXT_MATCH marker (signal fired); got {kinds}"
    )


def test_signals_fired_appear_in_activation_reason_detail() -> None:
    p1 = _retag_signals(
        _retag_id(_cheng_laoshi_protocol(), "synthetic.audit"),
        ContextMatchSignal(
            signal_id="my_signal",
            measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
            weight=1.0,
        ),
    )

    rupture = _make_rupture_snapshot(fire=True)
    snapshot = compute_active_mixture(
        loaded_protocols=(p1,),
        upstream={"rupture_state": rupture},
    )

    entry = snapshot.active_protocols[0]
    detail = next(
        r.detail
        for r in entry.activation_reasons
        if r.kind is ActivationReasonKind.CONTEXT_MATCH
    )
    assert "signals_fired=[my_signal]" in detail, detail


# ---------------------------------------------------------------------------
# compute_active_mixture: cheng_laoshi e2e behaviour preservation
# ---------------------------------------------------------------------------


def test_cheng_laoshi_e2e_behaviour_preserved_across_packet_1_5a() -> None:
    """Canonical cheng_laoshi fixture path produces identical snapshot.

    Pre-packet-1.5a: 1 active protocol, weight=1.0, no signals fire.
    Post-packet-1.5a: identical (signals empty → score=0 → equal-weight
    fallback path). Pinning this guarantees no regression on the
    real lifeform.
    """

    protocol = _cheng_laoshi_protocol()
    snapshot = compute_active_mixture(
        loaded_protocols=(protocol,),
        upstream={
            "interlocutor_state": _make_interlocutor_snapshot(
                fire_acknowledge_pressure=True
            ),
            "rupture_state": _make_rupture_snapshot(fire=True),
            "boundary_policy": _make_boundary_policy_snapshot(fire=True),
        },
    )

    assert len(snapshot.active_protocols) == 1
    entry = snapshot.active_protocols[0]
    assert entry.protocol_id == protocol.protocol_id
    assert entry.activation_weight == 1.0
    assert snapshot.boundary_union_ids == tuple(
        b.boundary_id for b in protocol.boundary_contracts
    ), snapshot.boundary_union_ids
