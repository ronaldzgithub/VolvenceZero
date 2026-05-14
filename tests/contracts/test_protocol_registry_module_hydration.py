"""Cross-session persistence of ``ProtocolRegistryModule`` learning state
(``protocol-online-learning-active`` packet, sub-packet C).

What this test asserts:

1. ``export_persistence_snapshot()`` round-trips through
   ``hydrate_from_persistence(...)`` byte-stably.
2. Owner-name mismatch / unknown schema_version / missing payload
   keys raise the typed hydration errors (fail-loud, no silent
   fallback).
3. Hydration intersects ``strategy_weights`` with currently-loaded
   protocols' rule ids (rules dropped between sessions are pruned;
   rules added stay at their seeded ``initial_weight``).
4. End-to-end across two ``LifeformSession`` instances (same
   ``user_scope`` / same ``MEMORY_SCOPE_ROOT_DIR``): α / β /
   per-protocol ``_pe_utility`` learnt in session A AND persisted
   via ``persist_owners()`` are restored in session B's
   newly-built stable ``ProtocolRegistryModule`` during
   ``AgentSessionRunner.__init__``.
5. Per-user isolation: a session for user A's learnt state must
   NOT bleed into user B's session under the same root.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as _replace
from pathlib import Path

import pytest

from lifeform_core import Lifeform
from lifeform_core.lifeform import LifeformConfig
from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import BehaviorProtocol
from volvence_zero.brain import BrainConfig
from volvence_zero.memory import StaticIdentityProvider, UserIdentity
from volvence_zero.owner_hydration import (
    HydrationOwnerMismatchError,
    HydrationPayloadInvalidError,
    HydrationVersionMismatchError,
    OwnerPersistenceSnapshot,
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


def _patch_with_pe_rates(
    protocol: BehaviorProtocol,
    *,
    pe_reinforce_rate: float = 0.5,
    pe_decay_rate: float = 0.5,
    initial_weight: float = 1.0,
    minimum_weight_floor: float = 0.0,
) -> BehaviorProtocol:
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


def _build_lifeform_for_user(
    *,
    scope_root: Path,
    user_id: str,
    seed_protocols: tuple[BehaviorProtocol, ...] = (),
) -> Lifeform:
    identity = UserIdentity(user_id=user_id, scope_key=user_id)
    brain_cfg = BrainConfig(
        memory_scope_root_dir=str(scope_root),
        owner_hydration_wiring=WiringLevel.ACTIVE,
    )
    return Lifeform(
        LifeformConfig(
            brain_config=brain_cfg,
            seed_protocols=seed_protocols,
        ),
        identity_provider=StaticIdentityProvider(identity=identity),
    )


# ---------------------------------------------------------------------------
# 1. Round-trip stability + fail-loud cases
# ---------------------------------------------------------------------------


def test_export_then_hydrate_round_trip_is_identity() -> None:
    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)
    # Drive some learning to make the snapshot non-trivial.
    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
        )
    )
    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.7, turn_index=2)}
        )
    )

    snapshot1 = module.export_persistence_snapshot()
    # Hydrate a fresh module loaded with the same protocol.
    fresh = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    fresh.load_protocol(protocol)
    fresh.hydrate_from_persistence(snapshot1)

    snapshot2 = fresh.export_persistence_snapshot()
    assert snapshot2.owner_name == snapshot1.owner_name
    assert snapshot2.schema_version == snapshot1.schema_version
    assert snapshot2.payload == snapshot1.payload


def test_hydrate_rejects_wrong_owner_name() -> None:
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    bad = OwnerPersistenceSnapshot(
        owner_name="some_other_owner",
        schema_version=1,
        payload={
            "alpha": 1.0,
            "beta": 1.0,
            "pe_utility": {},
            "strategy_weights": {},
            "last_strategy_reward": {},
            "last_pe_turn_index": None,
        },
    )
    with pytest.raises(HydrationOwnerMismatchError):
        module.hydrate_from_persistence(bad)


def test_hydrate_rejects_unknown_schema_version() -> None:
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    bad = OwnerPersistenceSnapshot(
        owner_name="protocol_registry",
        schema_version=999,
        payload={
            "alpha": 1.0,
            "beta": 1.0,
            "pe_utility": {},
            "strategy_weights": {},
            "last_strategy_reward": {},
            "last_pe_turn_index": None,
        },
    )
    with pytest.raises(HydrationVersionMismatchError):
        module.hydrate_from_persistence(bad)


def test_hydrate_rejects_missing_required_payload_key() -> None:
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    bad = OwnerPersistenceSnapshot(
        owner_name="protocol_registry",
        schema_version=1,
        # Missing ``last_pe_turn_index``.
        payload={
            "alpha": 1.0,
            "beta": 1.0,
            "pe_utility": {},
            "strategy_weights": {},
            "last_strategy_reward": {},
        },
    )
    with pytest.raises(HydrationPayloadInvalidError):
        module.hydrate_from_persistence(bad)


# ---------------------------------------------------------------------------
# 2. Strategy-weight intersection with current protocol's rule ids
# ---------------------------------------------------------------------------


def test_hydrate_drops_strategy_weights_for_unknown_rule_ids() -> None:
    """A snapshot may carry rule ids that no longer exist on the
    current protocol (revision dropped them). Hydrate MUST keep
    only intersected ids."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)

    real_rule_ids = {p.rule_id for p in protocol.strategy_priors}
    # Construct a snapshot with one extra fake rule id.
    fake_id = "rule:does-not-exist"
    snapshot = OwnerPersistenceSnapshot(
        owner_name="protocol_registry",
        schema_version=1,
        payload={
            "alpha": 1.5,
            "beta": 1.2,
            "pe_utility": {protocol.protocol_id: 0.4},
            "strategy_weights": {
                protocol.protocol_id: {
                    **{rid: 1.5 for rid in real_rule_ids},
                    fake_id: 99.9,
                }
            },
            "last_strategy_reward": {protocol.protocol_id: {}},
            "last_pe_turn_index": 7,
        },
    )

    module.hydrate_from_persistence(snapshot)
    rule_table = module._strategy_weights[protocol.protocol_id]  # noqa: SLF001
    assert fake_id not in rule_table
    assert set(rule_table.keys()) == real_rule_ids
    for value in rule_table.values():
        assert value == 1.5


def test_hydrate_seeds_initial_weight_for_rule_ids_missing_from_snapshot() -> None:
    """A protocol may carry rules that were not in the persisted
    snapshot (added between sessions). Hydrate MUST leave them at
    their seeded ``initial_weight`` rather than dropping them."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)

    real_rule_ids = sorted(p.rule_id for p in protocol.strategy_priors)
    # Persisted snapshot only has the FIRST rule.
    head, *tail = real_rule_ids
    snapshot = OwnerPersistenceSnapshot(
        owner_name="protocol_registry",
        schema_version=1,
        payload={
            "alpha": 1.0,
            "beta": 1.0,
            "pe_utility": {},
            "strategy_weights": {
                protocol.protocol_id: {head: 0.25},
            },
            "last_strategy_reward": {},
            "last_pe_turn_index": None,
        },
    )

    module.hydrate_from_persistence(snapshot)
    table = module._strategy_weights[protocol.protocol_id]  # noqa: SLF001
    assert table[head] == 0.25
    for rid in tail:
        # initial_weight defaulted to 1.0 in _patch_with_pe_rates.
        assert table[rid] == 1.0


# ---------------------------------------------------------------------------
# 3. End-to-end across two LifeformSession boundaries
# ---------------------------------------------------------------------------


def test_alpha_and_pe_utility_persist_across_lifeform_sessions(
    tmp_path: Path,
) -> None:
    """Real two-session e2e: session A learns via stable
    ProtocolRegistryModule; persist_owners writes to the shared
    backend; session B (same user, same scope_root) reads it back
    on AgentSessionRunner.__init__."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    life_a = _build_lifeform_for_user(
        scope_root=tmp_path,
        user_id="alice",
        seed_protocols=(protocol,),
    )
    session_a = life_a.create_session(session_id="alice-protocol-1")
    runner_a = session_a.brain_session.runner
    module_a = runner_a._protocol_registry_module  # noqa: SLF001

    # Drive some learning by manually feeding PE through the
    # owner. Two-protocol setups are needed for α/β to actually
    # move; for this test we only assert _pe_utility persistence
    # (single-protocol updates that field every turn).
    asyncio.run(
        module_a.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
        )
    )
    asyncio.run(
        module_a.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.9, turn_index=2)}
        )
    )
    pe_utility_a = dict(module_a._pe_utility)  # noqa: SLF001
    assert pe_utility_a, "expected non-empty _pe_utility after PE turn"
    # Manually move alpha/beta out of 1.0 so the round-trip has
    # something to assert. (The real path moves these only with
    # multi-protocol differential signal; single-protocol setups
    # never trigger the α/β update by design.)
    module_a._alpha = 1.7  # noqa: SLF001
    module_a._beta = 0.6  # noqa: SLF001
    strategy_weights_a = {
        pid: dict(rules)
        for pid, rules in module_a._strategy_weights.items()  # noqa: SLF001
    }

    # Persist + tear down session A.
    persisted = session_a.persist_owners()
    assert "protocol_registry" in persisted
    del session_a
    del life_a

    # Session B: brand-new Lifeform, same user + scope_root + seed.
    life_b = _build_lifeform_for_user(
        scope_root=tmp_path,
        user_id="alice",
        seed_protocols=(protocol,),
    )
    session_b = life_b.create_session(session_id="alice-protocol-2")
    module_b = (
        session_b.brain_session.runner._protocol_registry_module  # noqa: SLF001
    )
    assert module_b.alpha == 1.7
    assert module_b.beta == 0.6
    assert dict(module_b._pe_utility) == pe_utility_a  # noqa: SLF001
    assert {
        pid: dict(rules)
        for pid, rules in module_b._strategy_weights.items()  # noqa: SLF001
    } == strategy_weights_a


def test_per_user_isolation_does_not_leak_learning_state(
    tmp_path: Path,
) -> None:
    """User alice's persisted learning state MUST NOT hydrate into
    user bob's brand-new session under the same scope root."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    # Alice learns + persists.
    life_alice = _build_lifeform_for_user(
        scope_root=tmp_path,
        user_id="alice",
        seed_protocols=(protocol,),
    )
    session_alice = life_alice.create_session(session_id="alice-1")
    module_alice = (
        session_alice.brain_session.runner._protocol_registry_module  # noqa: SLF001
    )
    module_alice._alpha = 2.5  # noqa: SLF001
    module_alice._pe_utility[protocol.protocol_id] = 0.42  # noqa: SLF001
    session_alice.persist_owners()
    del session_alice
    del life_alice

    # Bob: new lifeform, same scope root, different user.
    life_bob = _build_lifeform_for_user(
        scope_root=tmp_path,
        user_id="bob",
        seed_protocols=(protocol,),
    )
    session_bob = life_bob.create_session(session_id="bob-1")
    module_bob = (
        session_bob.brain_session.runner._protocol_registry_module  # noqa: SLF001
    )
    # Bob starts fresh: α=1.0, _pe_utility empty.
    assert module_bob.alpha == 1.0
    assert module_bob._pe_utility == {}  # noqa: SLF001
