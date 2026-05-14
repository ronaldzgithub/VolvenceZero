"""Contract: ``AgentSessionRunner`` MUST hold a stable
``ProtocolRegistryModule`` whose ``_alpha`` / ``_beta`` /
``_pe_utility`` learning state survives across every turn of the
session.

Pre-this-packet semantics (broken):

* ``AgentSessionRunner.run_turn`` called ``run_final_wiring_turn``
  WITHOUT passing ``protocol_registry_module=...``. Inside
  ``build_final_runtime_modules`` the default-construction branch
  fired every turn and built a fresh empty
  ``ProtocolRegistryModule``, throwing away ``_alpha`` / ``_beta``
  / ``_pe_utility`` from the previous turn. The α / β learning
  logic was already implemented inside the module's ``process``
  method (packet 1.5c-iii) but the wiring meant it could never
  accumulate signal across turns.

Post-this-packet semantics (load-bearing):

* ``AgentSessionRunner.__init__`` constructs ONE stable
  ``ProtocolRegistryModule`` (with the application stores
  injected so ``load_protocol`` auto-applies hint / rule /
  knowledge / case to owners) and ``run_turn`` threads it
  through ``run_final_wiring_turn(..., protocol_registry_module=...)``
  so every turn re-uses the SAME owner instance. The module's
  ``_alpha`` / ``_beta`` / ``_pe_utility`` therefore accumulate
  PE-driven signal over the session lifetime.

* The seed protocols flow ``LifeformConfig.seed_protocols`` →
  ``Lifeform`` → ``Brain`` → ``AgentSessionRunner.__init__``
  → ``ProtocolRegistryModule.load_protocol(p)`` for each
  protocol. Once loaded, the protocol participates in α / β
  PE-driven mixing AND its compiled boundary / strategy /
  knowledge / case artifacts land in the application owner
  stores in the same call (no separate
  ``apply_domain_experience_packages`` step needed).

What this test asserts:

1. ``LifeformConfig`` round-trips ``seed_protocols`` through
   ``with_seed_protocols`` (additive).
2. ``Lifeform`` constructed with ``LifeformConfig.seed_protocols``
   produces a session whose ``runner._protocol_registry_module``
   has the seeded protocol(s) loaded.
3. The seeded protocol's compiled artifacts land in the kernel
   session's application owner stores (proves
   ``load_protocol`` auto-apply ran).
4. ``runner._protocol_registry_module`` is the SAME instance
   across multiple property reads (identity stable, not
   re-constructed per access).
5. Direct ``module.process`` invocations on the runner-held
   instance produce α / β / ``_pe_utility`` updates that survive
   to the next ``process`` call (the load-bearing
   "PE accumulates across turns" property, exercised on the
   same instance the kernel session would use).
6. ``Lifeform.with_seed_protocols`` is additive (does not
   clobber existing seed protocols on the config).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as _replace

from lifeform_core import Lifeform, LifeformConfig
from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    ContextMatchSignal,
)
from volvence_zero.interlocutor.contracts import (
    InterlocutorState,
    InterlocutorStateSnapshot,
    with_zones,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.runtime import Snapshot


# ---------------------------------------------------------------------------
# Helpers (mirror tests/contracts/test_protocol_alpha_beta_learning.py so
# the upstream snapshots use the same canonical shape)
# ---------------------------------------------------------------------------


def _cheng_laoshi_protocol() -> BehaviorProtocol:
    return growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )


def _retag(
    protocol: BehaviorProtocol,
    *,
    new_id: str,
    signals: tuple[ContextMatchSignal, ...] = (),
) -> BehaviorProtocol:
    new_conditions = ActivationConditions(
        context_match_signals=signals,
        co_activation_compatible=protocol.activation_conditions.co_activation_compatible,
        co_activation_incompatible=protocol.activation_conditions.co_activation_incompatible,
        minimum_weight_floor=protocol.activation_conditions.minimum_weight_floor,
    )
    return _replace(
        protocol,
        protocol_id=new_id,
        activation_conditions=new_conditions,
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


def _make_active_interlocutor() -> Snapshot[InterlocutorStateSnapshot]:
    state = InterlocutorState(
        emotional_weight=0.80,
        resistance_level=0.55,
        trust_signal=-0.20,
        readout_confidence=0.85,
        rationale="active",
    )
    state = with_zones(state)
    return Snapshot(
        slot_name="interlocutor_state",
        owner="InterlocutorReadoutModule",
        version=1,
        timestamp_ms=0,
        value=InterlocutorStateSnapshot(
            state=state, description="active fixture"
        ),
    )


# ---------------------------------------------------------------------------
# 1. LifeformConfig.seed_protocols round-trip
# ---------------------------------------------------------------------------


def test_lifeform_config_with_seed_protocols_is_additive() -> None:
    config = LifeformConfig()
    assert config.seed_protocols == ()

    p1 = _retag(_cheng_laoshi_protocol(), new_id="p1")
    p2 = _retag(_cheng_laoshi_protocol(), new_id="p2")

    after_one = config.with_seed_protocols((p1,))
    assert after_one.seed_protocols == (p1,)

    after_two = after_one.with_seed_protocols((p2,))
    assert after_two.seed_protocols == (p1, p2)


def test_lifeform_with_seed_protocols_returns_clone_with_extra_protocols() -> None:
    p1 = _retag(_cheng_laoshi_protocol(), new_id="p1")
    p2 = _retag(_cheng_laoshi_protocol(), new_id="p2")

    base = Lifeform(LifeformConfig())
    assert base.config.seed_protocols == ()

    rebuilt = base.with_seed_protocols((p1, p2))
    assert rebuilt is not base
    assert rebuilt.config.seed_protocols == (p1, p2)


# ---------------------------------------------------------------------------
# 2. AgentSessionRunner.__init__ constructs stable module + load_protocol
# ---------------------------------------------------------------------------


def test_runner_holds_stable_protocol_registry_module_when_no_seed() -> None:
    """With ``seed_protocols=()`` the runner still owns ONE stable
    module (empty registry); each ``run_turn`` will re-use it.
    Identity stability matters even in the empty case because
    ``_alpha`` / ``_beta`` start at 1.0 and would otherwise reset
    each turn."""

    life = Lifeform(LifeformConfig())
    session = life.create_session(session_id="contract-empty-seed")
    runner = session.brain_session.runner
    module = runner._protocol_registry_module  # noqa: SLF001
    assert module is not None
    assert module.registry.loaded() == ()
    assert module.alpha == 1.0
    assert module.beta == 1.0
    # Same instance on a second access (identity stable).
    assert runner._protocol_registry_module is module  # noqa: SLF001


def test_runner_loads_seed_protocol_into_stable_module() -> None:
    """seed_protocols flow LifeformConfig → AgentSessionRunner →
    stable ProtocolRegistryModule.load_protocol."""

    seed = _cheng_laoshi_protocol()
    life = Lifeform(LifeformConfig().with_seed_protocols((seed,)))
    session = life.create_session(session_id="contract-cheng-laoshi-seed")
    runner = session.brain_session.runner
    module = runner._protocol_registry_module  # noqa: SLF001
    loaded = module.registry.loaded()
    assert len(loaded) == 1
    assert loaded[0].protocol_id == seed.protocol_id


def test_seed_protocol_compiled_artifacts_land_in_application_stores() -> None:
    """``load_protocol`` with stores injected MUST also auto-apply the
    compiled hint / rule / knowledge / case to the owner stores.
    This is the load-bearing replacement for the SessionManager's
    old ``with_domain_experience(...)`` injection path."""

    seed = _cheng_laoshi_protocol()
    life = Lifeform(LifeformConfig().with_seed_protocols((seed,)))
    session = life.create_session(session_id="contract-stores")
    runner = session.brain_session.runner

    rare_heavy_state = runner._application_rare_heavy_state  # noqa: SLF001
    assert any(
        seed.protocol_id in hint.hint_id
        for hint in rare_heavy_state.boundary_prior_hints
    )
    assert any(
        seed.protocol_id in rule.rule_id
        for rule in rare_heavy_state.distilled_playbook_rules
    )


# ---------------------------------------------------------------------------
# 3. PE accumulation across turns on the SAME stable instance
# ---------------------------------------------------------------------------


def test_pe_history_accumulates_across_process_calls_on_runner_module() -> None:
    """The runner-held module's ``_pe_utility`` MUST update after each
    ``process(upstream)`` call.

    Pre-fix the runner held no module reference and ``_pe_utility``
    was reset per turn. With the new wiring every ``process`` lands
    on the same dict and accumulates.
    """

    seed = _cheng_laoshi_protocol()
    life = Lifeform(LifeformConfig().with_seed_protocols((seed,)))
    session = life.create_session(session_id="contract-pe-accum")
    runner = session.brain_session.runner
    module = runner._protocol_registry_module  # noqa: SLF001

    # Initial: empty pe_utility.
    assert module._pe_utility == {}  # noqa: SLF001

    # Turn 1: cache active weights so turn 2's PE has something to
    # attribute to (the module's _update_pe_history needs
    # _last_active_weights to be non-empty to credit anyone).
    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.7, turn_index=1)}
        )
    )
    # Turn 2: PE arrives with a non-empty _last_active_weights cache,
    # so _pe_utility[seed.protocol_id] gets a non-zero update.
    asyncio.run(
        module.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.7, turn_index=2)}
        )
    )

    pe_after = dict(module._pe_utility)  # noqa: SLF001
    assert seed.protocol_id in pe_after, pe_after
    # Some non-zero EMA should have accumulated (positive PE × any
    # active weight on a singleton mixture).
    assert pe_after[seed.protocol_id] != 0.0


def test_alpha_rises_across_turns_on_runner_held_module() -> None:
    """End-to-end identity check: feed a 2-protocol differential
    setup through the SAME runner-held module and watch α move
    above 1.0. Pre-fix the module was rebuilt per turn → no
    accumulation possible. Post-fix it is the same instance, so
    the existing ``_update_alpha_beta`` logic actually fires."""

    base = _cheng_laoshi_protocol()
    p_a = _retag(
        base,
        new_id="alpha",
        signals=(
            ContextMatchSignal(
                signal_id="ack_pressure",
                measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
                weight=2.0,
            ),
        ),
    )
    p_b = _retag(base, new_id="beta")
    life = Lifeform(LifeformConfig().with_seed_protocols((p_a, p_b)))
    session = life.create_session(session_id="contract-alpha-up")
    runner = session.brain_session.runner
    module = runner._protocol_registry_module  # noqa: SLF001

    upstream = {
        "interlocutor_state": _make_active_interlocutor(),
        "prediction_error": _make_pe_snapshot(signed_reward=0.8, turn_index=1),
    }
    asyncio.run(module.process(upstream))
    asyncio.run(
        module.process(
            {
                "interlocutor_state": _make_active_interlocutor(),
                "prediction_error": _make_pe_snapshot(
                    signed_reward=0.8, turn_index=2
                ),
            }
        )
    )

    assert module.alpha > 1.0, module.alpha


# ---------------------------------------------------------------------------
# 4. SessionManager → seed_protocols flow (uptake-side wiring smoke)
# ---------------------------------------------------------------------------


def test_session_manager_threads_uptake_protocols_to_runner_registry() -> None:
    """Service-approved protocol → SessionManager.create_session →
    runner._protocol_registry_module.registry.loaded() contains it."""

    from lifeform_protocol_runtime import inject_protocol_from_payload
    from lifeform_service.protocol_uptake import (
        ProtocolUptakeConfig,
        ProtocolUptakeService,
    )
    from lifeform_service.session_manager import SessionManager
    from lifeform_service.vertical_registry import VerticalRegistry
    from lifeform_service.verticals import VerticalSpec

    spec_payload = {
        "protocol_id": "test-uptake:wiring-bot",
        "advisor_name": "wiring-bot",
        "description": "Test wiring bot.",
        "boundaries": [
            {
                "boundary_id": "bd:wiring",
                "description": "x",
                "trigger_reasons": ["t"],
                "severity": "soft_remind",
            }
        ],
        "strategies": [
            {
                "rule_id": "rule:wiring",
                "problem_pattern": "p",
                "recommended_ordering": ["s"],
            }
        ],
    }
    candidate = inject_protocol_from_payload(
        spec_payload,
        request_id="req-wiring-test",
        extractor_id="contract-test",
    )
    pid = candidate.protocol.protocol_id

    service = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=None,
        ),
    )

    async def _setup() -> None:
        await service.submit_candidate(candidate, note="contract")
        await service.approve_pending(pid, reviewer_id="reviewer")

    asyncio.run(_setup())

    test_life = Lifeform(LifeformConfig())
    spec = VerticalSpec(
        name="contract-wiring-vertical",
        factory=lambda _runtime: test_life,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )
    manager = SessionManager(
        vertical_registry=VerticalRegistry.single(spec, alpha_enabled=False),
        protocol_uptake_service=service,
    )

    async def _go():
        return await manager.create_session(
            session_id="contract-uptake-wiring"
        )

    session = asyncio.run(_go())
    runner = session.brain_session.runner
    loaded_ids = {p.protocol_id for p in runner._protocol_registry_module.registry.loaded()}  # noqa: SLF001
    assert pid in loaded_ids
