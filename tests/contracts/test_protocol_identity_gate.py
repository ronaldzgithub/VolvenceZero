"""Identity gate (packet 1.3a): real regime check + permissive self-trait placeholder.

Asserts the contract for ``activation._compute_identity_gate`` and
its integration into ``compute_active_mixture``:

* **Regime branch (real)**:
    - empty ``required_regime_compatibility`` → pass (cheng_laoshi
      shape; backwards-compatible with packet 1.0/1.2)
    - non-empty + matching ``active_regime.regime_id`` → pass
    - non-empty + non-matching → filter out (gate=0)
    - non-empty + missing regime snapshot → SHADOW-permissive pass
* **Self-trait branch (placeholder)**:
    - non-empty ``requires_self_traits`` / ``forbidden_self_traits``
      → permissive pass with deferral marker in
      ``ActivationReason.detail``
* **Audit surface**:
    - ``ActivationReason.detail`` carries human-readable reasons so
      operations can debug "why didn't my protocol activate?"
* **Behaviour preservation**:
    - cheng_laoshi (empty ``required_regime_compatibility``) is
      unchanged before/after packet 1.3a — same active_protocols,
      same boundary_union_ids.
"""

from __future__ import annotations

from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActivationReasonKind,
    BehaviorProtocol,
    IdentityAssertion,
)
from volvence_zero.protocol_runtime import compute_active_mixture
from volvence_zero.protocol_runtime.activation import _compute_identity_gate
from volvence_zero.regime.contracts import RegimeIdentity, RegimeSnapshot
from volvence_zero.runtime import Snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regime_snapshot(regime_id: str) -> Snapshot[RegimeSnapshot]:
    """Construct a minimal valid ``RegimeSnapshot`` published as a Snapshot.

    Only ``active_regime.regime_id`` is consulted by the identity
    gate; other fields use the lightest valid defaults.
    """

    active_regime = RegimeIdentity(
        regime_id=regime_id,
        name=regime_id,
        embedding=(0.0,),
        entry_conditions="test-entry",
        exit_conditions="test-exit",
        historical_effectiveness=0.5,
    )
    snapshot_value = RegimeSnapshot(
        active_regime=active_regime,
        previous_regime=None,
        switch_reason="test",
        candidate_regimes=(),
        turns_in_current_regime=1,
        description=f"test regime {regime_id!r}",
    )
    return Snapshot(
        slot_name="regime",
        owner="RegimeModule",
        version=1,
        timestamp_ms=0,
        value=snapshot_value,
    )


def _cheng_laoshi_protocol() -> BehaviorProtocol:
    """Cheng laoshi protocol via the canonical packet 1.0+ fixture path."""

    return growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())


def _retag_identity(
    protocol: BehaviorProtocol,
    *,
    requires_self_traits: tuple[str, ...] = (),
    forbidden_self_traits: tuple[str, ...] = (),
    required_regime_compatibility: tuple[str, ...] = (),
) -> BehaviorProtocol:
    """Replace the identity assertion on a protocol for test variation."""

    new_identity = IdentityAssertion(
        requires_self_traits=requires_self_traits,
        forbidden_self_traits=forbidden_self_traits,
        required_regime_compatibility=required_regime_compatibility,
    )
    return _replace(protocol, identity_assertion=new_identity)


# ---------------------------------------------------------------------------
# _compute_identity_gate: regime branch
# ---------------------------------------------------------------------------


def test_empty_required_regime_passes_permissively() -> None:
    """Cheng laoshi shape: empty ``required_regime_compatibility``
    means the protocol is regime-agnostic; gate must pass.
    """

    bp = _cheng_laoshi_protocol()
    assert bp.identity_assertion.required_regime_compatibility == ()
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=None, regime_snapshot=None
    )
    assert gate == 1.0
    assert "regime_check_empty_pass" in reasons


def test_matching_regime_passes() -> None:
    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        required_regime_compatibility=("emotional_support",),
    )
    snapshot = _make_regime_snapshot("emotional_support")
    gate, reasons = _compute_identity_gate(
        bp,
        dual_track_snapshot=None,
        regime_snapshot=snapshot.value,
    )
    assert gate == 1.0
    assert any(r.startswith("regime_match:emotional_support") for r in reasons), reasons


def test_mismatching_regime_filters_out() -> None:
    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        required_regime_compatibility=("emotional_support",),
    )
    snapshot = _make_regime_snapshot("casual_social")
    gate, reasons = _compute_identity_gate(
        bp,
        dual_track_snapshot=None,
        regime_snapshot=snapshot.value,
    )
    assert gate == 0.0
    assert any(r.startswith("regime_mismatch:") for r in reasons), reasons


def test_missing_regime_snapshot_shadow_permissive_pass() -> None:
    """When the protocol has a non-empty required_regime_compatibility
    but no regime snapshot is present (SHADOW dependency unavailable),
    we pass permissively. ACTIVE wiring is gated separately
    (FallbackActivationActiveError + checklist condition 5).
    """

    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        required_regime_compatibility=("emotional_support",),
    )
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=None, regime_snapshot=None
    )
    assert gate == 1.0
    assert "regime_unknown_shadow_pass" in reasons


# ---------------------------------------------------------------------------
# _compute_identity_gate: self-trait placeholder branch
# ---------------------------------------------------------------------------


def test_non_empty_required_self_traits_dual_track_absent_shadow_pass() -> None:
    """Packet 1.3' partial: when dual_track snapshot is missing
    entirely (test fixture / SHADOW dependency unavailable), the
    self-trait check passes with a typed marker so audit can spot
    which protocols are deferred on missing upstream.
    """

    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        requires_self_traits=("warm_peer_register", "long_horizon"),
    )
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=None, regime_snapshot=None
    )
    assert gate == 1.0
    assert "self_traits_dual_track_absent_shadow_pass" in reasons


def test_non_empty_forbidden_self_traits_dual_track_absent_shadow_pass() -> None:
    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        forbidden_self_traits=("high_pressure_sales",),
    )
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=None, regime_snapshot=None
    )
    assert gate == 1.0
    assert "self_traits_dual_track_absent_shadow_pass" in reasons


def test_cheng_laoshi_default_dual_track_absent_pass() -> None:
    """cheng_laoshi has non-empty requires_self_traits — the test
    fixture passes dual_track_snapshot=None so the dual-track-absent
    SHADOW marker applies.
    """

    bp = _cheng_laoshi_protocol()
    assert bp.identity_assertion.requires_self_traits == (
        "warm_peer_register",
        "long_horizon",
    )
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=None, regime_snapshot=None
    )
    assert gate == 1.0
    assert "self_traits_dual_track_absent_shadow_pass" in reasons


def test_truly_empty_identity_yields_empty_pass_marker() -> None:
    bp = _retag_identity(_cheng_laoshi_protocol())  # all empty
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=None, regime_snapshot=None
    )
    assert gate == 1.0
    assert "self_traits_empty_pass" in reasons


# ---------------------------------------------------------------------------
# compute_active_mixture integration (filtering + reasons surface)
# ---------------------------------------------------------------------------


def test_compute_active_mixture_filters_incompatible_regime_protocol() -> None:
    """A loaded protocol whose required regime != current regime is
    DROPPED from ``active_protocols`` (R8 hard filter). Other
    eligible protocols continue at full weight.
    """

    bp_compatible = _retag_identity(
        _cheng_laoshi_protocol(),
        required_regime_compatibility=("emotional_support",),
    )
    bp_compatible = _replace(bp_compatible, protocol_id="test:compatible")

    bp_incompatible = _retag_identity(
        _cheng_laoshi_protocol(),
        required_regime_compatibility=("acquaintance_building",),
    )
    bp_incompatible = _replace(bp_incompatible, protocol_id="test:incompatible")

    upstream = {"regime": _make_regime_snapshot("emotional_support")}
    snapshot = compute_active_mixture(
        loaded_protocols=(bp_compatible, bp_incompatible),
        upstream=upstream,
    )
    assert isinstance(snapshot, ActiveMixtureSnapshot)
    active_ids = {entry.protocol_id for entry in snapshot.active_protocols}
    assert active_ids == {"test:compatible"}, active_ids


def test_compute_active_mixture_filters_all_when_none_match() -> None:
    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        required_regime_compatibility=("emotional_support",),
    )
    upstream = {"regime": _make_regime_snapshot("casual_social")}
    snapshot = compute_active_mixture(
        loaded_protocols=(bp,),
        upstream=upstream,
    )
    assert snapshot.active_protocols == ()
    assert snapshot.boundary_union_ids == ()
    assert "filtered all" in snapshot.description


def test_identity_gate_reasons_appear_in_activation_reasons() -> None:
    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        required_regime_compatibility=("emotional_support",),
    )
    upstream = {"regime": _make_regime_snapshot("emotional_support")}
    snapshot = compute_active_mixture(
        loaded_protocols=(bp,),
        upstream=upstream,
    )
    assert len(snapshot.active_protocols) == 1
    entry = snapshot.active_protocols[0]
    identity_reason = next(
        r for r in entry.activation_reasons
        if r.kind is ActivationReasonKind.IDENTITY_GATE
    )
    assert "regime_match:emotional_support" in identity_reason.detail


# ---------------------------------------------------------------------------
# Backwards compatibility: cheng_laoshi unchanged
# ---------------------------------------------------------------------------


def test_cheng_laoshi_active_mixture_unchanged_by_packet_1_3a() -> None:
    """Cheng laoshi has empty ``required_regime_compatibility``;
    after packet 1.3a the snapshot must still contain it as the sole
    active protocol with weight=1.0 + same 4 boundaries.

    This is the regression gate for the matched-control invariant
    on the existing fixture.
    """

    bp = _cheng_laoshi_protocol()
    # No regime upstream; cheng_laoshi must still pass (regime
    # branch is empty, self-traits branch is permissive deferred).
    snapshot = compute_active_mixture(
        loaded_protocols=(bp,),
        upstream={},
    )
    assert len(snapshot.active_protocols) == 1
    assert snapshot.active_protocols[0].protocol_id == bp.protocol_id
    assert snapshot.active_protocols[0].activation_weight == 1.0
    assert len(snapshot.boundary_union_ids) == 4


def test_cheng_laoshi_passes_with_any_regime_present() -> None:
    """Same as above but with a synthetic regime snapshot in
    upstream — cheng_laoshi shouldn't care because its
    ``required_regime_compatibility`` is empty.
    """

    bp = _cheng_laoshi_protocol()
    for regime_id in ("emotional_support", "casual_social", "anything"):
        snapshot = compute_active_mixture(
            loaded_protocols=(bp,),
            upstream={"regime": _make_regime_snapshot(regime_id)},
        )
        assert len(snapshot.active_protocols) == 1, regime_id
        assert snapshot.active_protocols[0].protocol_id == bp.protocol_id, regime_id


# ---------------------------------------------------------------------------
# Packet 1.3': self_traits real check when dual_track.self_track.traits populated
# ---------------------------------------------------------------------------
#
# These tests synthesize a ``DualTrackSnapshot`` with a populated
# ``self_track.traits`` tuple, exercising the real check path.
# Production population (deriving traits from semantic owners /
# memory consolidation / persona seeds) is **deferred** to a future
# packet; until then production code passes empty traits and falls
# back to ``self_traits_populator_pending`` (covered below).


from volvence_zero.dual_track import DualTrackSnapshot, TrackState  # noqa: E402
from volvence_zero.memory import Track  # noqa: E402


def _make_dual_track_snapshot(
    *,
    self_traits: tuple[str, ...] = (),
) -> DualTrackSnapshot:
    """Synthetic ``DualTrackSnapshot`` for identity-gate tests.

    Only ``self_track.traits`` is consulted by the identity gate.
    All other fields use neutral defaults; ``world_track`` mirrors
    ``self_track`` shape but is never read by the gate.
    """

    self_state = TrackState(
        track=Track.SELF,
        active_goals=(),
        recent_credits=(),
        controller_code=(0.0, 0.0, 0.0),
        tension_level=0.0,
        traits=self_traits,
    )
    world_state = TrackState(
        track=Track.WORLD,
        active_goals=(),
        recent_credits=(),
        controller_code=(0.0, 0.0, 0.0),
        tension_level=0.0,
    )
    return DualTrackSnapshot(
        world_track=world_state,
        self_track=self_state,
        cross_track_tension=0.0,
        description="test dual_track",
    )


def test_self_traits_required_subset_present_passes() -> None:
    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        requires_self_traits=("warm_peer_register", "long_horizon"),
    )
    dt = _make_dual_track_snapshot(
        self_traits=("warm_peer_register", "long_horizon", "mom_register"),
    )
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=dt, regime_snapshot=None
    )
    assert gate == 1.0
    assert any(r.startswith("self_traits_required_match:") for r in reasons), reasons


def test_self_traits_required_missing_filters_out() -> None:
    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        requires_self_traits=("warm_peer_register", "long_horizon"),
    )
    # Missing 'long_horizon'
    dt = _make_dual_track_snapshot(self_traits=("warm_peer_register",))
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=dt, regime_snapshot=None
    )
    assert gate == 0.0
    assert any(r.startswith("self_traits_missing_required:") for r in reasons), reasons


def test_self_traits_forbidden_absent_passes() -> None:
    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        forbidden_self_traits=("high_pressure_sales",),
    )
    dt = _make_dual_track_snapshot(self_traits=("warm_peer_register",))
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=dt, regime_snapshot=None
    )
    assert gate == 1.0
    assert any(r.startswith("self_traits_forbidden_absent:") for r in reasons), reasons


def test_self_traits_forbidden_present_filters_out() -> None:
    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        forbidden_self_traits=("high_pressure_sales",),
    )
    dt = _make_dual_track_snapshot(
        self_traits=("warm_peer_register", "high_pressure_sales"),
    )
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=dt, regime_snapshot=None
    )
    assert gate == 0.0
    assert any(r.startswith("self_traits_forbidden_present:") for r in reasons), reasons


def test_self_traits_required_and_forbidden_combined_pass() -> None:
    """Both branches active and consistent — required present,
    forbidden absent.
    """

    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        requires_self_traits=("warm_peer_register",),
        forbidden_self_traits=("high_pressure_sales",),
    )
    dt = _make_dual_track_snapshot(
        self_traits=("warm_peer_register", "long_horizon"),
    )
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=dt, regime_snapshot=None
    )
    assert gate == 1.0
    # Both markers should appear
    assert any(r.startswith("self_traits_required_match:") for r in reasons), reasons
    assert any(r.startswith("self_traits_forbidden_absent:") for r in reasons), reasons


def test_self_traits_empty_traits_field_falls_back_to_populator_pending() -> None:
    """When dual_track snapshot exists but ``self_track.traits`` is
    empty (the default before any populator is wired), the gate
    falls back to a typed deferral marker. ACTIVE promotion stays
    blocked by the fallback flag.
    """

    bp = _retag_identity(
        _cheng_laoshi_protocol(),
        requires_self_traits=("warm_peer_register",),
    )
    dt = _make_dual_track_snapshot(self_traits=())
    gate, reasons = _compute_identity_gate(
        bp, dual_track_snapshot=dt, regime_snapshot=None
    )
    assert gate == 1.0
    assert "self_traits_populator_pending" in reasons


def test_self_traits_filter_integrates_into_compute_active_mixture() -> None:
    """E2E filter via ``compute_active_mixture``: a protocol whose
    ``requires_self_traits`` is not satisfied is dropped from
    ``active_protocols``.
    """

    bp_compatible = _retag_identity(
        _cheng_laoshi_protocol(),
        requires_self_traits=("warm_peer_register",),
    )
    bp_compatible = _replace(bp_compatible, protocol_id="test:compat")

    bp_incompatible = _retag_identity(
        _cheng_laoshi_protocol(),
        requires_self_traits=("aggressive_sales",),
    )
    bp_incompatible = _replace(bp_incompatible, protocol_id="test:incompat")

    dt = _make_dual_track_snapshot(self_traits=("warm_peer_register",))
    snapshot = compute_active_mixture(
        loaded_protocols=(bp_compatible, bp_incompatible),
        upstream={
            "dual_track": Snapshot(
                slot_name="dual_track",
                owner="DualTrackModule",
                version=1,
                timestamp_ms=0,
                value=dt,
            ),
        },
    )
    active_ids = {entry.protocol_id for entry in snapshot.active_protocols}
    assert active_ids == {"test:compat"}, active_ids


# ---------------------------------------------------------------------------
# Packet 1.3'': end-to-end through run_final_wiring_turn with real seed
# ---------------------------------------------------------------------------


def test_cheng_laoshi_self_traits_pass_via_growth_advisor_seed_e2e() -> None:
    """End-to-end gate via run_final_wiring_turn:

    1. Vertical builder produces an IdentitySeed with cheng_laoshi
       traits.
    2. ``run_final_wiring_turn`` threads the seed to DualTrackModule.
    3. DualTrackModule publishes self_track.traits in the snapshot.
    4. ProtocolRegistryModule reads the snapshot via its
       ``("dual_track", "regime")`` upstream dependency and runs the
       real self_traits subset check.
    5. cheng_laoshi protocol's ``requires_self_traits`` matches the
       seed's traits → identity gate passes → protocol stays active
       in the published mixture (instead of being filtered).

    This is the proof that packet 1.3'' actually closes the
    self_traits machinery end-to-end with cheng_laoshi.
    """

    import asyncio

    from lifeform_domain_growth_advisor import (
        build_cheng_laoshi_profile,
        build_growth_advisor_identity_seed,
        growth_advisor_profile_to_behavior_protocol,
    )
    from volvence_zero.behavior_protocol import ActiveMixtureSnapshot
    from volvence_zero.integration import (
        FinalRolloutConfig,
        run_final_wiring_turn,
    )
    from volvence_zero.protocol_runtime import ProtocolRegistryModule
    from volvence_zero.runtime import WiringLevel
    from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter

    profile = build_cheng_laoshi_profile()
    seed = build_growth_advisor_identity_seed(profile)
    bp = growth_advisor_profile_to_behavior_protocol(profile)

    # cheng_laoshi requires both 'warm_peer_register' and 'long_horizon'
    assert bp.identity_assertion.requires_self_traits == (
        "warm_peer_register",
        "long_horizon",
    )
    # seed provides both
    assert "warm_peer_register" in seed.traits
    assert "long_horizon" in seed.traits

    module = ProtocolRegistryModule()
    module.load_protocol(bp)

    config = FinalRolloutConfig(dual_track=WiringLevel.ACTIVE)
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=PlaceholderSubstrateAdapter(
                model_id="cheng-laoshi-identity-e2e",
            ),
            session_id="cheng-laoshi-identity-e2e-session",
            wave_id="cheng-laoshi-identity-e2e-wave",
            identity_seed=seed,
            protocol_registry_module=module,
        )
    )
    active_mixture = result.shadow_snapshots["active_mixture"].value
    assert isinstance(active_mixture, ActiveMixtureSnapshot)
    # cheng_laoshi protocol IS in the active mixture
    active_ids = {entry.protocol_id for entry in active_mixture.active_protocols}
    assert bp.protocol_id in active_ids, active_ids
    # And the IDENTITY_GATE reason in the entry mentions the
    # required-match (real check fired, not populator-pending)
    entry = next(
        e for e in active_mixture.active_protocols if e.protocol_id == bp.protocol_id
    )
    identity_reasons = " ; ".join(
        r.detail for r in entry.activation_reasons
    )
    assert "self_traits_required_match" in identity_reasons, identity_reasons


def test_cheng_laoshi_self_traits_filter_when_seed_lacks_required_e2e() -> None:
    """Inverse e2e: a malicious or drifting protocol whose
    ``requires_self_traits`` includes a trait the lifeform doesn't
    have should be filtered out (gate=0).

    Demonstrates that packet 1.3'' actually filters in production
    code paths, not just in synthetic ``compute_active_mixture``
    fixtures.
    """

    import asyncio
    from dataclasses import replace as _replace

    from lifeform_domain_growth_advisor import (
        build_cheng_laoshi_profile,
        build_growth_advisor_identity_seed,
        growth_advisor_profile_to_behavior_protocol,
    )
    from volvence_zero.behavior_protocol import (
        ActiveMixtureSnapshot,
        IdentityAssertion,
    )
    from volvence_zero.integration import (
        FinalRolloutConfig,
        run_final_wiring_turn,
    )
    from volvence_zero.protocol_runtime import ProtocolRegistryModule
    from volvence_zero.runtime import WiringLevel
    from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter

    profile = build_cheng_laoshi_profile()
    seed = build_growth_advisor_identity_seed(profile)
    bp_legit = growth_advisor_profile_to_behavior_protocol(profile)

    # Construct a hostile protocol requiring a trait the lifeform doesn't have
    bp_hostile = _replace(
        bp_legit,
        protocol_id="test:hostile",
        identity_assertion=IdentityAssertion(
            requires_self_traits=("aggressive_sales",),
            forbidden_self_traits=(),
            required_regime_compatibility=(),
        ),
    )

    module = ProtocolRegistryModule()
    module.load_protocol(bp_legit)
    module.load_protocol(bp_hostile)

    config = FinalRolloutConfig(dual_track=WiringLevel.ACTIVE)
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=PlaceholderSubstrateAdapter(
                model_id="cheng-laoshi-hostile-e2e",
            ),
            session_id="cheng-laoshi-hostile-e2e-session",
            wave_id="cheng-laoshi-hostile-e2e-wave",
            identity_seed=seed,
            protocol_registry_module=module,
        )
    )
    active_mixture = result.shadow_snapshots["active_mixture"].value
    assert isinstance(active_mixture, ActiveMixtureSnapshot)

    active_ids = {entry.protocol_id for entry in active_mixture.active_protocols}
    # Legit protocol PASSED the gate
    assert bp_legit.protocol_id in active_ids, active_ids
    # Hostile protocol was FILTERED OUT (gate=0)
    assert "test:hostile" not in active_ids, active_ids
