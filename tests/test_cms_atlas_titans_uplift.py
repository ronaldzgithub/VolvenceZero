"""Tests for the CMS ATLAS / Titans update rule uplift.

See ``docs/specs/cms-atlas-titans-uplift.md`` for the full spec. The
acceptance ladder defined there is encoded as the test cases below
(unit + contract level; SHADOW smoke / paper-suite tiers run in their
own benchmark harnesses).
"""

from __future__ import annotations

from volvence_zero.learned_update import (
    LEARNED_UPDATE_BASE_FEATURE_DIM,
    LEARNED_UPDATE_PE_AWARE_FEATURE_DIM,
    LEARNED_UPDATE_PE_FEATURE_DIM,
    LearnedUpdateRule,
)
from volvence_zero.memory.cms import CMSBandMLP, CMSMemoryCore
from volvence_zero.memory.store import build_default_memory_store
from volvence_zero.prediction.error import (
    ActualOutcome,
    PredictedOutcome,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.substrate import FeatureSignal, SubstrateSnapshot, SurfaceKind


def _make_substrate(value: float) -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="cms-uplift-test",
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(0.5,),
        feature_surface=(
            FeatureSignal(name="signal", values=(value, value * 0.5), source="test", layer_hint=0),
        ),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="synthetic substrate for uplift tests",
    )


def _make_pe_snapshot(*, relationship: float = 0.0, task: float = 0.0) -> PredictionErrorSnapshot:
    predicted = PredictedOutcome(
        source_turn_index=0,
        target_turn_index=1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.0,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.5,
        description="prediction",
    )
    actual = ActualOutcome(
        observed_turn_index=1,
        task_progress=0.5,
        relationship_delta=0.0,
        regime_stability=0.5,
        action_payoff=0.5,
        description="actual",
    )
    error = PredictionError(
        task_error=task,
        relationship_error=relationship,
        regime_error=0.0,
        action_error=0.0,
        magnitude=max(abs(task), abs(relationship)),
        signed_reward=0.0,
        description="synthetic-pe",
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=predicted,
        actual_outcome=actual,
        next_prediction=predicted,
        error=error,
        turn_index=1,
        bootstrap=False,
        description="synthetic-pe-snapshot",
    )


# -----------------------------------------------------------------------
# §3.3 / acceptance ladder #2 — band MLP replay K=1 bit-equality
# -----------------------------------------------------------------------


def test_band_mlp_update_with_replay_k1_equals_update() -> None:
    target = (0.2, 0.4, 0.6, 0.8)
    legacy = CMSBandMLP(d_in=4, d_hidden=6, learning_rate=0.1, momentum_beta=0.9)
    replay = CMSBandMLP(d_in=4, d_hidden=6, learning_rate=0.1, momentum_beta=0.9)

    legacy.update(target=target, lr_scale=0.7, momentum_gate=0.5)
    replay.update_with_replay(
        targets=(target,),
        weights=(1.0,),
        lr_scale=0.7,
        momentum_gate=0.5,
    )

    assert legacy.export_params() == replay.export_params()


def test_band_mlp_update_with_replay_normalizes_weights() -> None:
    target = (0.2, 0.4, 0.6, 0.8)
    averaged = CMSBandMLP(d_in=4, d_hidden=6, learning_rate=0.1, momentum_beta=0.9)
    weighted = CMSBandMLP(d_in=4, d_hidden=6, learning_rate=0.1, momentum_beta=0.9)

    # Two identical targets with equal raw weights must collapse to a single
    # update on the same target (normalized weights = 0.5, 0.5; weighted
    # average = target).
    averaged.update(target=target, lr_scale=0.7, momentum_gate=0.5)
    weighted.update_with_replay(
        targets=(target, target),
        weights=(2.5, 2.5),
        lr_scale=0.7,
        momentum_gate=0.5,
    )

    assert averaged.export_params() == weighted.export_params()


def test_band_mlp_update_with_replay_zero_weight_total_is_noop() -> None:
    target = (0.2, 0.4, 0.6, 0.8)
    mlp = CMSBandMLP(d_in=4, d_hidden=6, learning_rate=0.1, momentum_beta=0.9)
    before = mlp.export_params()
    mlp.update_with_replay(
        targets=(target,),
        weights=(0.0,),
        lr_scale=0.7,
        momentum_gate=0.5,
    )
    assert mlp.export_params() == before


# -----------------------------------------------------------------------
# §5 / acceptance ladder #3 — legacy state zero-pads PE columns
# -----------------------------------------------------------------------


def test_cms_restore_legacy_state_zeros_pe_columns_when_uplift_enabled() -> None:
    """When CMS is constructed with ``pe_features_enabled=True`` and we
    restore a legacy (feature_version=1) rule state, the trailing PE
    columns of the rule's input_projection must be reset to zero so the
    PE-driven path starts clean (instead of inheriting weights trained
    for the old modulo-extended features). See spec §5."""
    canonical = CMSMemoryCore(mode="mlp", d_in=4, d_hidden=8)
    for index in range(3):
        canonical.observe_substrate(
            substrate_snapshot=_make_substrate(0.1 + index * 0.05),
            timestamp_ms=index,
        )
    legacy_checkpoint = canonical.export_state()
    assert legacy_checkpoint.update_rule_state is not None
    assert legacy_checkpoint.update_rule_state.feature_version == 1

    uplift = CMSMemoryCore(
        mode="mlp",
        d_in=4,
        d_hidden=8,
        pe_features_enabled=True,
    )
    uplift.restore_state(legacy_checkpoint)
    rule_state = uplift.snapshot().update_rule_state
    assert rule_state is not None
    assert rule_state.feature_dim == LEARNED_UPDATE_PE_AWARE_FEATURE_DIM
    for row in rule_state.input_projection:
        assert row[LEARNED_UPDATE_BASE_FEATURE_DIM:] == tuple(
            0.0 for _ in range(LEARNED_UPDATE_PE_FEATURE_DIM)
        )


def test_cms_restore_legacy_state_keeps_weights_when_uplift_disabled() -> None:
    """When CMS is constructed with the canonical (uplift-off) flags and
    we restore a legacy state, the rule's input_projection must be
    untouched. This is the bit-equal regression contract for the
    canonical path."""
    canonical = CMSMemoryCore(mode="mlp", d_in=4, d_hidden=8)
    for index in range(3):
        canonical.observe_substrate(
            substrate_snapshot=_make_substrate(0.1 + index * 0.05),
            timestamp_ms=index,
        )
    legacy_checkpoint = canonical.export_state()
    saved_input_projection = legacy_checkpoint.update_rule_state.input_projection

    restored = CMSMemoryCore(mode="mlp", d_in=4, d_hidden=8)
    restored.restore_state(legacy_checkpoint)
    assert (
        restored.snapshot().update_rule_state.input_projection
        == saved_input_projection
    )


def test_learned_update_rule_metacontroller_keeps_legacy_modulo_restore() -> None:
    """A non-CMS rule (e.g. the metacontroller with arbitrary feature_dim)
    must keep the legacy modulo padding behavior on restore so it is not
    silently affected by the uplift."""
    rule = LearnedUpdateRule(rule_id="metacontroller-update", feature_dim=5, hidden_dim=4)
    decision = rule.decide(target_id="t", features=(0.1, 0.2, 0.3, 0.4, 0.5))
    rule.learn(features=(0.1, 0.2, 0.3, 0.4, 0.5), decision=decision, improvement=0.1, stability=0.6)
    saved = rule.export_state()
    assert saved.feature_version == 1
    restored = LearnedUpdateRule(rule_id="metacontroller-update", feature_dim=5, hidden_dim=4)
    restored.restore_state(saved)
    assert restored.export_state().input_projection == saved.input_projection


# -----------------------------------------------------------------------
# §6 / acceptance ladder #4 — uplift fields default for canonical
# -----------------------------------------------------------------------


def test_cms_state_uplift_fields_default_for_canonical() -> None:
    cms = CMSMemoryCore(mode="mlp", d_in=4, d_hidden=8)
    snap = cms.snapshot()
    assert snap.atlas_replay_active is False
    assert snap.titans_pe_gate_active is False
    assert snap.replay_window_sizes == (
        ("online-fast", 1),
        ("session-medium", 1),
        ("background-slow", 1),
    )
    zero_pe = tuple(0.0 for _ in range(LEARNED_UPDATE_PE_FEATURE_DIM))
    for band in (snap.online_fast, snap.session_medium, snap.background_slow):
        assert band.replay_window_size == 0
        # Before any observation runs, no decision has fired yet for these
        # core bands. The summary stays at the zero-init contract value.
        assert band.pe_feature_summary == zero_pe


def test_cms_state_uplift_fields_active_when_flags_on() -> None:
    cms = CMSMemoryCore(
        mode="mlp",
        d_in=4,
        d_hidden=8,
        pe_features_enabled=True,
        replay_window_sizes={"online-fast": 8, "session-medium": 4, "background-slow": 2},
    )
    cms.observe_substrate(
        substrate_snapshot=_make_substrate(0.4),
        timestamp_ms=1,
        prediction_error=_make_pe_snapshot(relationship=0.6),
    )
    snap = cms.snapshot()
    assert snap.atlas_replay_active is True
    assert snap.titans_pe_gate_active is True
    assert dict(snap.replay_window_sizes) == {
        "online-fast": 8,
        "session-medium": 4,
        "background-slow": 2,
    }
    online_band = snap.online_fast
    assert online_band.replay_window_size >= 1
    # PE feature summary should reflect the relationship error magnitude
    # (index 1 in our PE feature layout).
    assert online_band.pe_feature_summary[1] > 0.5


# -----------------------------------------------------------------------
# §5.3 / acceptance ladder #1 — disabled flags reproduce legacy behavior
# -----------------------------------------------------------------------


def test_cms_uplift_disabled_path_is_bit_equal_to_legacy() -> None:
    """Two CMS instances initialized identically with both uplift flags
    off must produce identical band states across a sequence of
    observations. This locks in the canonical regression contract."""
    cms_a = CMSMemoryCore(mode="mlp", d_in=4, d_hidden=8)
    cms_b = CMSMemoryCore(mode="mlp", d_in=4, d_hidden=8)
    pe = _make_pe_snapshot(relationship=0.5, task=0.4)
    for index in range(8):
        substrate = _make_substrate(0.1 + index * 0.07)
        # ``cms_a`` ignores PE; ``cms_b`` is told about a PE that should be
        # ignored because pe_features_enabled is False.
        cms_a.observe_substrate(substrate_snapshot=substrate, timestamp_ms=index)
        cms_b.observe_substrate(
            substrate_snapshot=substrate,
            timestamp_ms=index,
            prediction_error=pe,
        )
    snap_a = cms_a.snapshot()
    snap_b = cms_b.snapshot()
    assert snap_a.online_fast.vector == snap_b.online_fast.vector
    assert snap_a.session_medium.vector == snap_b.session_medium.vector
    assert snap_a.background_slow.vector == snap_b.background_slow.vector


# -----------------------------------------------------------------------
# Mechanism 2 — Titans gate modulates decision under PE
# -----------------------------------------------------------------------


def test_cms_uplift_pe_gate_modulates_features() -> None:
    """With PE features enabled, observing a substrate together with a
    high-relationship-error PE should populate ``pe_feature_summary`` for
    the online band, while disabling the flag must keep it at zero."""
    cms_off = CMSMemoryCore(mode="mlp", d_in=4, d_hidden=8)
    cms_on = CMSMemoryCore(
        mode="mlp",
        d_in=4,
        d_hidden=8,
        pe_features_enabled=True,
    )
    pe = _make_pe_snapshot(relationship=0.8, task=0.2)
    substrate = _make_substrate(0.4)

    cms_off.observe_substrate(
        substrate_snapshot=substrate,
        timestamp_ms=1,
        prediction_error=pe,
    )
    cms_on.observe_substrate(
        substrate_snapshot=substrate,
        timestamp_ms=1,
        prediction_error=pe,
    )

    summary_off = cms_off.snapshot().online_fast.pe_feature_summary
    summary_on = cms_on.snapshot().online_fast.pe_feature_summary
    assert summary_off == (0.0, 0.0, 0.0, 0.0)
    assert summary_on == (0.2, 0.8, 0.0, 0.0)


# -----------------------------------------------------------------------
# Acceptance ladder #5 — MemoryStore observe_substrate PE optional
# -----------------------------------------------------------------------


def test_memory_store_observe_substrate_pe_optional() -> None:
    """Pre-uplift call signature (no prediction_error kwarg) still works."""
    store = build_default_memory_store(latent_dim=4)
    store.observe_substrate(
        substrate_snapshot=_make_substrate(0.3),
        timestamp_ms=1,
    )
    # Sanity: snapshot is constructible.
    snap = store.snapshot(
        retrieved_entries=(),
        suppressed_cross_scope_entries=(),
        active_subject_scope=(),
        social_pe_signals=(),
    )
    assert snap.cms_state is not None
    assert snap.cms_state.titans_pe_gate_active is True
    assert snap.cms_state.atlas_replay_active is True


def test_memory_store_default_uplift_can_be_explicitly_rolled_back() -> None:
    """The uplift is ACTIVE by default, but the rollback flags recover the
    pre-uplift CMS path for A/B and emergency revert."""
    store = build_default_memory_store(
        latent_dim=4,
        cms_pe_features_enabled=False,
        cms_replay_window_size=None,
    )
    store.observe_substrate(
        substrate_snapshot=_make_substrate(0.3),
        timestamp_ms=1,
        prediction_error=_make_pe_snapshot(relationship=0.7),
    )
    snap = store.snapshot(
        retrieved_entries=(),
        suppressed_cross_scope_entries=(),
        active_subject_scope=(),
        social_pe_signals=(),
    )
    assert snap.cms_state is not None
    assert snap.cms_state.titans_pe_gate_active is False
    assert snap.cms_state.atlas_replay_active is False
    assert snap.cms_state.online_fast.pe_feature_summary == (0.0, 0.0, 0.0, 0.0)


def test_memory_store_observe_substrate_pe_forwarded() -> None:
    store = build_default_memory_store(
        latent_dim=4,
        cms_pe_features_enabled=True,
        cms_replay_window_size=8,
    )
    pe = _make_pe_snapshot(relationship=0.7)
    store.observe_substrate(
        substrate_snapshot=_make_substrate(0.4),
        timestamp_ms=1,
        prediction_error=pe,
    )
    snap = store.snapshot(
        retrieved_entries=(),
        suppressed_cross_scope_entries=(),
        active_subject_scope=(),
        social_pe_signals=(),
    )
    cms_state = snap.cms_state
    assert cms_state is not None
    assert cms_state.titans_pe_gate_active is True
    assert cms_state.atlas_replay_active is True
    assert cms_state.online_fast.pe_feature_summary[1] > 0.5


# -----------------------------------------------------------------------
# Acceptance ladder #6 — replay window grows under repeated observations
# -----------------------------------------------------------------------


def test_cms_uplift_replay_window_grows_with_observations() -> None:
    cms = CMSMemoryCore(
        mode="mlp",
        d_in=4,
        d_hidden=8,
        pe_features_enabled=False,
        replay_window_sizes={"online-fast": 8, "session-medium": 4, "background-slow": 2},
    )
    for index in range(5):
        cms.observe_substrate(
            substrate_snapshot=_make_substrate(0.1 + index * 0.05),
            timestamp_ms=index,
        )
    online_band = cms.snapshot().online_fast
    # 5 observations into a K=8 deque -> window size = 5.
    assert online_band.replay_window_size == 5

    for index in range(5, 12):
        cms.observe_substrate(
            substrate_snapshot=_make_substrate(0.1 + index * 0.05),
            timestamp_ms=index,
        )
    online_band = cms.snapshot().online_fast
    # 12 observations into a K=8 deque -> window saturates at 8.
    assert online_band.replay_window_size == 8


# -----------------------------------------------------------------------
# Acceptance contract — nested variant survives uplift
# -----------------------------------------------------------------------


def test_cms_uplift_nested_variant_meta_targets_still_evolve() -> None:
    cms = CMSMemoryCore(
        mode="mlp",
        d_in=4,
        d_hidden=8,
        variant="nested",
        pe_features_enabled=True,
        replay_window_sizes={"online-fast": 4, "session-medium": 3, "background-slow": 2},
    )
    pe = _make_pe_snapshot(relationship=0.4, task=0.3)
    for index in range(20):
        cms.observe_substrate(
            substrate_snapshot=_make_substrate(0.1 + (index % 4) * 0.05),
            timestamp_ms=index,
            prediction_error=pe,
        )
    online_target, session_target = cms.nested_reset_targets() or ((), ())
    # Meta-learning should have moved the targets away from the zero
    # initialization in at least one component.
    assert any(value != 0.0 for value in online_target)
    assert any(value != 0.0 for value in session_target)


# -----------------------------------------------------------------------
# Checkpoint roundtrip — uplift fields survive export/restore
# -----------------------------------------------------------------------


def test_cms_uplift_checkpoint_roundtrip_preserves_uplift_fields() -> None:
    cms = CMSMemoryCore(
        mode="mlp",
        d_in=4,
        d_hidden=8,
        pe_features_enabled=True,
        replay_window_sizes={"online-fast": 8, "session-medium": 4, "background-slow": 2},
    )
    cms.observe_substrate(
        substrate_snapshot=_make_substrate(0.4),
        timestamp_ms=1,
        prediction_error=_make_pe_snapshot(relationship=0.6),
    )
    state = cms.export_state()
    assert state.atlas_replay_active is True
    assert state.titans_pe_gate_active is True
    assert dict(state.replay_window_sizes) == {
        "online-fast": 8,
        "session-medium": 4,
        "background-slow": 2,
    }
    assert state.update_rule_state is not None
    assert state.update_rule_state.feature_version == 2

    cms_restored = CMSMemoryCore(
        mode="mlp",
        d_in=4,
        d_hidden=8,
        pe_features_enabled=True,
        replay_window_sizes={"online-fast": 8, "session-medium": 4, "background-slow": 2},
    )
    cms_restored.restore_state(state)
    snap = cms_restored.snapshot()
    assert snap.online_fast.vector == cms.snapshot().online_fast.vector
