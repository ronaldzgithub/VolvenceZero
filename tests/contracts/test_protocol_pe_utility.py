"""PE utility (packet 1.5b): rolling EMA of signed_reward attributed to last-turn mixture.

Asserts the contract for ``ProtocolRegistryModule._update_pe_history``
and its integration into ``compute_active_mixture`` (β·pe_utility
term):

* **Cold start**: turn 1 has empty ``_last_active_weights`` → no
  attribution, ``pe_utility`` stays empty.
* **EMA update**: turn t reads ``signed_reward``, multiplies by
  last-turn weight, mixes with η = 0.25 into the per-protocol EMA.
* **Inactive decay**: protocols not in last turn's mixture decay
  toward 0 (multiplied by 1-η).
* **Bootstrap skip**: ``PredictionErrorSnapshot.bootstrap=True``
  short-circuits attribution (placeholder PE, not actionable).
* **Turn-index dedup**: same ``turn_index`` from a replay /
  retry doesn't double-attribute.
* **Differential weighting**: after several turns of consistent
  positive PE, the favoured protocol's softmax weight rises;
  consistent negative PE pulls it down (relative to peers).
* **cheng_laoshi singleton**: a single-protocol mixture stays at
  weight=1.0 regardless of pe_utility (softmax of a single
  raw_score collapses to 1.0).
* **SHADOW-tolerant**: missing ``prediction_error`` upstream →
  no-op, no error raised.

These tests exercise the owner directly (no kernel runtime),
mirroring the packet 1.5a test surface. They do NOT depend on
context_match signals firing — pe_utility is the only weighting
input.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    ActivationReasonKind,
    BehaviorProtocol,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule
from volvence_zero.runtime import Snapshot, WiringLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cheng_laoshi_protocol() -> BehaviorProtocol:
    return growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )


def _retag_id(protocol: BehaviorProtocol, new_id: str) -> BehaviorProtocol:
    return _replace(protocol, protocol_id=new_id)


def _make_pe_snapshot(
    *,
    signed_reward: float,
    turn_index: int,
    bootstrap: bool = False,
) -> Snapshot[PredictionErrorSnapshot]:
    """Construct a Snapshot[PredictionErrorSnapshot] for tests.

    Only ``signed_reward`` / ``turn_index`` / ``bootstrap`` are
    consulted by ``_update_pe_history``; other fields use the
    lightest valid defaults.
    """

    action_context = PredictionActionContext(
        segment_id=f"seg-{turn_index}",
        abstract_action_id="test_action",
        regime_id="test_regime",
    )
    actual = ActualOutcome(
        observed_turn_index=turn_index,
        task_progress=0.5,
        relationship_delta=0.5,
        regime_stability=0.5,
        action_payoff=0.5,
        description="test actual",
        action_context=action_context,
    )
    next_prediction = PredictedOutcome(
        source_turn_index=turn_index,
        target_turn_index=turn_index + 1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.5,
        description="test next",
        action_context=action_context,
    )
    error = PredictionError(
        task_error=signed_reward,
        relationship_error=signed_reward,
        regime_error=signed_reward,
        action_error=signed_reward,
        magnitude=abs(signed_reward),
        signed_reward=signed_reward,
        description=f"test pe signed_reward={signed_reward}",
    )
    pe_value = PredictionErrorSnapshot(
        evaluated_prediction=None if bootstrap else next_prediction,
        actual_outcome=actual,
        next_prediction=next_prediction,
        error=error,
        turn_index=turn_index,
        bootstrap=bootstrap,
        description="test pe-snapshot",
        action_context=action_context,
    )
    return Snapshot(
        slot_name="prediction_error",
        owner="PredictionErrorModule",
        version=1,
        timestamp_ms=turn_index * 1000,
        value=pe_value,
    )


def _build_module_with(
    protocols: tuple[BehaviorProtocol, ...],
) -> ProtocolRegistryModule:
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    for protocol in protocols:
        module.load_protocol(protocol)
    return module


def _run_turn(
    module: ProtocolRegistryModule,
    upstream: dict,
):
    return asyncio.run(module.process(upstream))


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


def test_cold_start_first_turn_has_empty_pe_utility() -> None:
    """Turn 1: no last_active_weights yet → no attribution."""
    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))

    snapshot = _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.7, turn_index=1)},
    )

    assert module.pe_utility == {}, module.pe_utility
    # The mixture is published but pe_utility didn't influence it.
    assert len(snapshot.value.active_protocols) == 1
    assert snapshot.value.active_protocols[0].activation_weight == 1.0


def test_missing_prediction_error_is_shadow_tolerant() -> None:
    """No prediction_error in upstream → no-op, no error."""
    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))

    snapshot = _run_turn(module, {})

    assert module.pe_utility == {}
    assert len(snapshot.value.active_protocols) == 1


# ---------------------------------------------------------------------------
# EMA update on turn 2+
# ---------------------------------------------------------------------------


def test_pe_utility_attributes_to_last_turn_active_protocols() -> None:
    """Turn 1 caches weights; turn 2 attributes signed_reward × last_weight."""
    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))

    # Turn 1: published weight=1.0 (single protocol). signed_reward
    # not yet attributed because last_active_weights empty pre-turn-1.
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.8, turn_index=1)},
    )
    assert module.pe_utility == {}

    # Turn 2: now last_active_weights[p] = 1.0 from turn 1.
    # Δ = 0.8 × 1.0 = 0.8; ema ← 0 + 0.25·0.8 = 0.2.
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.8, turn_index=2)},
    )

    assert p.protocol_id in module.pe_utility
    assert abs(module.pe_utility[p.protocol_id] - 0.2) < 1e-9


def test_pe_utility_ema_accumulates_across_turns() -> None:
    """Multiple positive turns push pe_utility upward toward signed_reward."""
    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))

    # 5 turns of signed_reward = 1.0 with weight=1.0 → ema converges
    # toward 1.0. Each step: ema ← (1-η)·ema + η·1.0
    # = 0, 0.25, 0.4375, 0.578, 0.683 (after attributions on turns 2..5)
    expected = 0.0
    eta = 0.25
    for turn in range(1, 6):
        _run_turn(
            module,
            {"prediction_error": _make_pe_snapshot(signed_reward=1.0, turn_index=turn)},
        )
        if turn >= 2:
            # Attribution on turn t uses turn (t-1)'s cached weight = 1.0.
            expected = (1.0 - eta) * expected + eta * 1.0

    assert abs(module.pe_utility[p.protocol_id] - expected) < 1e-9
    assert module.pe_utility[p.protocol_id] > 0.6


def test_pe_utility_responds_to_negative_signed_reward() -> None:
    """Negative signed_reward pulls pe_utility below zero."""
    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))

    for turn in range(1, 4):
        _run_turn(
            module,
            {"prediction_error": _make_pe_snapshot(signed_reward=-1.0, turn_index=turn)},
        )

    assert module.pe_utility[p.protocol_id] < -0.3


# ---------------------------------------------------------------------------
# Bootstrap skip + turn dedup
# ---------------------------------------------------------------------------


def test_bootstrap_pe_skips_attribution() -> None:
    """Bootstrap PE is placeholder; skip to avoid baking it into EMA."""
    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))

    # Turn 1 (regular) → cache weight=1.0
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.5, turn_index=1)},
    )

    # Turn 2 (bootstrap) → should NOT update EMA even though
    # last_active_weights is non-empty.
    _run_turn(
        module,
        {
            "prediction_error": _make_pe_snapshot(
                signed_reward=0.99, turn_index=2, bootstrap=True
            )
        },
    )

    # pe_utility stays at 0 (turn 1 didn't update because no prior
    # mixture cached; turn 2 was bootstrap).
    assert module.pe_utility == {}


def test_duplicate_turn_index_does_not_double_attribute() -> None:
    """Same turn_index PE arriving twice doesn't double-count."""
    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))

    # Turn 1 prepares weights cache.
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.4, turn_index=1)},
    )

    # Turn 2 attributes once.
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.4, turn_index=2)},
    )
    after_first = module.pe_utility[p.protocol_id]

    # Send turn 2 again (duplicate); must be no-op.
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.4, turn_index=2)},
    )
    after_dup = module.pe_utility[p.protocol_id]

    assert abs(after_first - after_dup) < 1e-12, (
        f"duplicate turn_index attributed twice: {after_first} → {after_dup}"
    )


# ---------------------------------------------------------------------------
# Differential weighting: protocol with positive history wins softmax
# ---------------------------------------------------------------------------


def test_protocol_with_positive_pe_history_wins_softmax() -> None:
    """After several positive PE turns, the rewarded protocol dominates."""
    base = _cheng_laoshi_protocol()
    p_winner = _retag_id(base, "synthetic.winner")
    p_loser = _retag_id(base, "synthetic.loser")
    module = _build_module_with((p_winner, p_loser))

    # Manually inject a positive history for p_winner (skip the
    # multi-turn rollout dance; we trust the EMA update path is
    # tested above).
    module._pe_utility[p_winner.protocol_id] = 0.6
    module._pe_utility[p_loser.protocol_id] = -0.6

    snapshot = _run_turn(module, {})

    weights = {
        e.protocol_id: e.activation_weight
        for e in snapshot.value.active_protocols
    }
    assert weights["synthetic.winner"] > weights["synthetic.loser"], weights
    # CONTEXT_MATCH marker because pe_utility != 0 → has_signal.
    winner_entry = next(
        e for e in snapshot.value.active_protocols
        if e.protocol_id == "synthetic.winner"
    )
    kinds = [r.kind for r in winner_entry.activation_reasons]
    assert ActivationReasonKind.CONTEXT_MATCH in kinds


def test_pe_utility_appears_in_activation_reason_detail() -> None:
    """``pe_utility=±0.xxx`` is logged in the audit detail when non-zero."""
    base = _cheng_laoshi_protocol()
    p1 = _retag_id(base, "synthetic.audit")
    module = _build_module_with((p1,))

    module._pe_utility[p1.protocol_id] = 0.42
    snapshot = _run_turn(module, {})

    entry = snapshot.value.active_protocols[0]
    detail = next(
        r.detail for r in entry.activation_reasons
        if r.kind is ActivationReasonKind.CONTEXT_MATCH
    )
    assert "pe_utility=+0.420" in detail, detail


# ---------------------------------------------------------------------------
# cheng_laoshi singleton: weight stable at 1.0
# ---------------------------------------------------------------------------


def test_cheng_laoshi_singleton_weight_unchanged_under_pe_history() -> None:
    """A single-protocol mixture has weight=1.0 even with non-zero pe_utility.

    Softmax of a single raw_score is always (1.0,) regardless of
    the score value. cheng_laoshi user-facing behaviour stays
    byte-equivalent across packet 1.5b (the audit detail line
    flips to CONTEXT_MATCH when pe_utility != 0, but the weight
    value is identical).
    """

    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))
    module._pe_utility[p.protocol_id] = 0.9

    snapshot = _run_turn(module, {})

    assert len(snapshot.value.active_protocols) == 1
    assert snapshot.value.active_protocols[0].activation_weight == 1.0


def test_cheng_laoshi_default_path_under_packet_1_5b() -> None:
    """No PE history + no signals → EQUAL_WEIGHT_FALLBACK preserved."""
    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))

    snapshot = _run_turn(module, {})  # No PE, no signals

    assert len(snapshot.value.active_protocols) == 1
    entry = snapshot.value.active_protocols[0]
    assert entry.activation_weight == 1.0
    kinds = [r.kind for r in entry.activation_reasons]
    assert ActivationReasonKind.EQUAL_WEIGHT_FALLBACK in kinds
    assert ActivationReasonKind.CONTEXT_MATCH not in kinds


# ---------------------------------------------------------------------------
# Inactive decay
# ---------------------------------------------------------------------------


def test_inactive_protocol_pe_utility_decays_toward_zero() -> None:
    """A protocol no longer in last_active_weights decays each PE turn."""
    base = _cheng_laoshi_protocol()
    p_active = _retag_id(base, "synthetic.active")
    p_resting = _retag_id(base, "synthetic.resting")
    module = _build_module_with((p_active, p_resting))

    # Pre-seed both with non-zero history.
    module._pe_utility[p_active.protocol_id] = 0.8
    module._pe_utility[p_resting.protocol_id] = 0.8
    # Pretend last turn only p_active was active.
    module._last_active_weights = {p_active.protocol_id: 1.0}

    eta = 0.25
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=10)},
    )

    # p_active: decays 0.8 → 0.6, then adds eta·(0×1.0)=0 → 0.6.
    assert abs(module.pe_utility[p_active.protocol_id] - 0.6) < 1e-9
    # p_resting: decays 0.8 → 0.6 (no contribution).
    assert abs(module.pe_utility[p_resting.protocol_id] - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------


def test_pe_utility_clamps_at_one() -> None:
    """Repeated extreme positive PE saturates at the clamp boundary."""
    p = _cheng_laoshi_protocol()
    module = _build_module_with((p,))

    # Inject near-max history then run more positive turns.
    module._pe_utility[p.protocol_id] = 0.95
    module._last_active_weights = {p.protocol_id: 1.0}

    for turn in range(20, 30):
        _run_turn(
            module,
            {"prediction_error": _make_pe_snapshot(signed_reward=1.0, turn_index=turn)},
        )

    assert module.pe_utility[p.protocol_id] <= 1.0
    assert module.pe_utility[p.protocol_id] >= 0.95
