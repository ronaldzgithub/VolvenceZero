"""Per-strategy weight learning (``protocol-online-learning-active``
packet, sub-packet B1).

Pre-this-packet semantics:

* ``StrategyPrior.initial_weight`` / ``pe_decay_rate`` /
  ``pe_reinforce_rate`` / ``minimum_weight_floor`` were dropped at
  compile time
  (``compile_protocol_to_application_artifacts`` docstring on
  ``compiler.py`` explicitly notes "PE-revision metadata is consumed
  by the future activation controller (packet 1.5+);
  StrategyPlaybookModule doesn't read it"). The fields existed but
  their PE consumers had not been built.

Post-this-packet semantics:

* ``ProtocolRegistryModule._strategy_weights`` is a per-protocol,
  per-rule dict initialised on ``load_protocol`` from each prior's
  ``initial_weight`` (clamped at ``minimum_weight_floor``).
* On every non-bootstrap, non-cold-start PE turn,
  ``_update_strategy_weights`` multiplicatively
  reinforces (positive ``signed_reward``) or decays (negative
  ``signed_reward``) every rule of every protocol that had
  non-zero ``_last_active_weights[pid]``, scaled by the prior's
  rate × the protocol's last-turn weight × |signed_reward|.
* Each weight is clamped to ``minimum_weight_floor`` from below
  and ``_STRATEGY_WEIGHT_MAX`` from above.
* ``ActiveMixtureSnapshot.strategy_weights`` exposes the table
  for downstream observability and (in B2 follow-up) consumption
  by the planner / strategy playbook.
* On ``unload_protocol``, the weight + reward state for that
  protocol is dropped.

What this test asserts:

1. ``load_protocol`` initialises every prior's weight to its
   ``initial_weight`` (or ``minimum_weight_floor`` when higher).
2. Positive PE reinforces weights of active protocols' rules.
3. Negative PE decays weights of active protocols' rules.
4. Weights are floored at ``minimum_weight_floor`` even after
   an extreme decay step.
5. The published ``ActiveMixtureSnapshot.strategy_weights``
   contains one entry per (protocol_id, rule_id) loaded.
6. Reloading a protocol with the same id preserves learned
   weights (idempotent re-seed).
7. Unloading a protocol drops its weight rows from the snapshot.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    StrategyPrior,
    StrategyWeightEntry,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule
from volvence_zero.protocol_runtime.owner import _STRATEGY_WEIGHT_MAX
from volvence_zero.runtime import Snapshot, WiringLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cheng_laoshi_protocol() -> BehaviorProtocol:
    return growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )


def _patch_strategy_priors_with_pe_rates(
    protocol: BehaviorProtocol,
    *,
    pe_reinforce_rate: float,
    pe_decay_rate: float,
    minimum_weight_floor: float = 0.0,
    initial_weight: float = 1.0,
) -> BehaviorProtocol:
    """Return a clone of ``protocol`` whose strategy_priors carry
    non-trivial PE rates so per-rule learning becomes observable."""

    new_priors = tuple(
        _replace(
            prior,
            initial_weight=initial_weight,
            minimum_weight_floor=minimum_weight_floor,
            pe_reinforce_rate=pe_reinforce_rate,
            pe_decay_rate=pe_decay_rate,
        )
        for prior in protocol.strategy_priors
    )
    return _replace(protocol, strategy_priors=new_priors)


def _make_pe_snapshot(
    *,
    signed_reward: float,
    turn_index: int,
    bootstrap: bool = False,
) -> Snapshot[PredictionErrorSnapshot]:
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
        description="test pe",
    )
    pe_value = PredictionErrorSnapshot(
        evaluated_prediction=None if bootstrap else next_prediction,
        actual_outcome=actual,
        next_prediction=next_prediction,
        error=error,
        turn_index=turn_index,
        bootstrap=bootstrap,
        description="test pe-snap",
        action_context=action_context,
    )
    return Snapshot(
        slot_name="prediction_error",
        owner="PredictionErrorModule",
        version=1,
        timestamp_ms=turn_index * 1000,
        value=pe_value,
    )


def _build_module_with_protocol(protocol: BehaviorProtocol) -> ProtocolRegistryModule:
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)
    return module


def _run_turn(module: ProtocolRegistryModule, upstream: dict) -> None:
    asyncio.run(module.process(upstream))


# ---------------------------------------------------------------------------
# 1. Initialisation from initial_weight
# ---------------------------------------------------------------------------


def test_load_protocol_seeds_strategy_weights_from_initial_weight() -> None:
    base = _cheng_laoshi_protocol()
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.0,
        pe_decay_rate=0.0,
        initial_weight=0.7,
        minimum_weight_floor=0.1,
    )
    module = _build_module_with_protocol(patched)

    weights = module._strategy_weights[patched.protocol_id]  # noqa: SLF001
    assert set(weights.keys()) == {p.rule_id for p in patched.strategy_priors}
    for value in weights.values():
        assert value == 0.7


def test_load_protocol_clamps_initial_weight_to_floor() -> None:
    base = _cheng_laoshi_protocol()
    # Floor higher than initial → seeded weight should be the floor.
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.0,
        pe_decay_rate=0.0,
        initial_weight=0.3,
        minimum_weight_floor=0.3,  # equal is fine; raise via reseed below
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id
    # Now manually mutate floor higher and re-load to assert the
    # clamp branch fires (re-seed preserves learned weights but
    # raises them to the floor).
    higher_floor = _replace(
        patched,
        strategy_priors=tuple(
            _replace(p, minimum_weight_floor=0.5, initial_weight=0.5)
            for p in patched.strategy_priors
        ),
    )
    module.load_protocol(higher_floor)
    for value in module._strategy_weights[pid].values():  # noqa: SLF001
        assert value >= 0.5


# ---------------------------------------------------------------------------
# 2. PE reinforcement / decay
# ---------------------------------------------------------------------------


def test_positive_pe_reinforces_strategy_weights() -> None:
    base = _cheng_laoshi_protocol()
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.5,
        pe_decay_rate=0.5,
        initial_weight=1.0,
        minimum_weight_floor=0.1,
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id
    initial = dict(module._strategy_weights[pid])  # noqa: SLF001

    # Turn 1: cache _last_active_weights with the singleton mixture.
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)},
    )
    # Turn 2: positive PE → reinforce.
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.8, turn_index=2)},
    )

    after = module._strategy_weights[pid]  # noqa: SLF001
    for rid, before in initial.items():
        assert after[rid] > before, (rid, before, after[rid])


def test_negative_pe_decays_strategy_weights() -> None:
    base = _cheng_laoshi_protocol()
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.5,
        pe_decay_rate=0.5,
        initial_weight=1.0,
        minimum_weight_floor=0.0,
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id
    initial = dict(module._strategy_weights[pid])  # noqa: SLF001

    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)},
    )
    _run_turn(
        module,
        {
            "prediction_error": _make_pe_snapshot(
                signed_reward=-0.8, turn_index=2
            )
        },
    )

    after = module._strategy_weights[pid]  # noqa: SLF001
    for rid, before in initial.items():
        assert after[rid] < before, (rid, before, after[rid])


def test_minimum_weight_floor_clamps_decay() -> None:
    base = _cheng_laoshi_protocol()
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.0,
        pe_decay_rate=0.99,
        initial_weight=0.5,
        minimum_weight_floor=0.4,
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id

    # Cold start cache.
    _run_turn(
        module,
        {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)},
    )
    # Run many strong-negative-PE turns; weights MUST never drop
    # below the floor.
    for turn in range(2, 12):
        _run_turn(
            module,
            {
                "prediction_error": _make_pe_snapshot(
                    signed_reward=-1.0, turn_index=turn
                )
            },
        )
        for value in module._strategy_weights[pid].values():  # noqa: SLF001
            assert value >= 0.4, value


def test_zero_pe_rate_means_no_movement() -> None:
    base = _cheng_laoshi_protocol()
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.0,
        pe_decay_rate=0.0,
        initial_weight=0.6,
        minimum_weight_floor=0.0,
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id
    initial = dict(module._strategy_weights[pid])  # noqa: SLF001

    for turn in range(1, 6):
        _run_turn(
            module,
            {
                "prediction_error": _make_pe_snapshot(
                    signed_reward=0.9 if turn % 2 == 0 else -0.9,
                    turn_index=turn,
                )
            },
        )

    after = module._strategy_weights[pid]  # noqa: SLF001
    for rid, before in initial.items():
        assert after[rid] == before, (rid, before, after[rid])


# ---------------------------------------------------------------------------
# 3. Snapshot exposure
# ---------------------------------------------------------------------------


def test_snapshot_strategy_weights_field_populated() -> None:
    base = _cheng_laoshi_protocol()
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.3,
        pe_decay_rate=0.3,
        initial_weight=0.5,
        minimum_weight_floor=0.1,
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id

    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
        )
    )
    snapshot = asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.7, turn_index=2)}
        )
    )

    entries = snapshot.value.strategy_weights
    assert isinstance(entries, tuple)
    assert all(isinstance(e, StrategyWeightEntry) for e in entries)
    rule_ids_in_protocol = {p.rule_id for p in patched.strategy_priors}
    rule_ids_in_snapshot = {e.rule_id for e in entries if e.protocol_id == pid}
    assert rule_ids_in_snapshot == rule_ids_in_protocol


def test_snapshot_strategy_weights_records_last_signed_reward() -> None:
    base = _cheng_laoshi_protocol()
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.3,
        pe_decay_rate=0.3,
        initial_weight=0.5,
        minimum_weight_floor=0.0,
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id

    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
        )
    )
    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.6, turn_index=2)}
        )
    )
    snapshot = asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=-0.4, turn_index=3)}
        )
    )

    for entry in snapshot.value.strategy_weights:
        if entry.protocol_id == pid:
            # Last update used signed_reward=-0.4.
            assert entry.last_signed_reward == -0.4


# ---------------------------------------------------------------------------
# 4. Re-load preserves learned state; unload drops it
# ---------------------------------------------------------------------------


def test_reloading_protocol_preserves_learned_weights() -> None:
    base = _cheng_laoshi_protocol()
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.5,
        pe_decay_rate=0.5,
        initial_weight=0.6,
        minimum_weight_floor=0.0,
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id

    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
        )
    )
    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.9, turn_index=2)}
        )
    )
    learned = dict(module._strategy_weights[pid])  # noqa: SLF001
    assert all(v > 0.6 for v in learned.values())

    # Re-load same protocol id → weights MUST be preserved.
    module.load_protocol(patched)
    after_reload = module._strategy_weights[pid]  # noqa: SLF001
    for rid, learned_value in learned.items():
        assert after_reload[rid] == learned_value


def test_unloading_protocol_drops_strategy_weights() -> None:
    base = _cheng_laoshi_protocol()
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=0.0,
        pe_decay_rate=0.0,
        initial_weight=0.5,
        minimum_weight_floor=0.0,
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id
    assert pid in module._strategy_weights  # noqa: SLF001

    module.unload_protocol(pid)
    assert pid not in module._strategy_weights  # noqa: SLF001
    assert pid not in module._last_strategy_reward  # noqa: SLF001


# ---------------------------------------------------------------------------
# 5. Hard upper clamp (defense in depth)
# ---------------------------------------------------------------------------


def test_strategy_weight_max_clamps_runaway_reinforce() -> None:
    """Even with extreme rates, the per-rule weight stops at
    ``_STRATEGY_WEIGHT_MAX`` so a misconfigured protocol can't
    cause numerical blowup downstream."""

    base = _cheng_laoshi_protocol()
    # Pick the largest legal rates and weights to drive saturation
    # quickly. ``initial_weight`` is bounded to [0, 1] at the
    # contract layer.
    patched = _patch_strategy_priors_with_pe_rates(
        base,
        pe_reinforce_rate=1.0,
        pe_decay_rate=1.0,
        initial_weight=1.0,
        minimum_weight_floor=0.0,
    )
    module = _build_module_with_protocol(patched)
    pid = patched.protocol_id

    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
        )
    )
    for turn in range(2, 200):
        asyncio.run(
            module.process(
                {
                    "prediction_error": _make_pe_snapshot(
                        signed_reward=2.0, turn_index=turn
                    )
                }
            )
        )

    for value in module._strategy_weights[pid].values():  # noqa: SLF001
        assert value <= _STRATEGY_WEIGHT_MAX
