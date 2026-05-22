"""End-to-end proof that PE actually mutates a seeded protocol's weights
across turns of a runner-held ``ProtocolRegistryModule``.

This is the **online-learning** half of the protocol-effective proof.
``test_protocol_approved_affects_chat_e2e.py`` proves "protocol gets
loaded and its artefacts reach the application owners on turn 1."
This file proves the next thing the operator cares about: "after a
few turns of PE pressure, the seeded protocol's strategy weights
actually evolve — the AI is **learning** from this protocol's
performance, not just statically applying it."

Pre-this-packet coverage
------------------------

* ``tests/contracts/test_protocol_strategy_weight_learning.py`` —
  module-level PE-driven weight evolution; constructed in
  isolation with ``ProtocolRegistryModule(wiring_level=SHADOW)``,
  not the runner-held module produced by ``with_seed_protocols``.
* ``tests/contracts/test_protocol_alpha_beta_session_wiring.py``::
  ``test_alpha_rises_across_turns_on_runner_held_module`` — same
  invariant but for the **α / β** layer (per-protocol selection),
  not the **per-rule strategy weights** the playbook reads.

What this file adds
-------------------

* Builds the protocol with **non-zero** ``pe_reinforce_rate`` /
  ``pe_decay_rate`` (the API-injection adapter drops these fields
  on purpose — the production payload schema doesn't expose them
  yet — so we go through ``service.submit_candidate`` +
  ``approve_pending`` with a hand-built candidate; that's the
  same path the HTTP route would take if the adapter forwarded
  the fields).
* Goes through ``SessionManager.create_session`` via HTTP, so the
  production seed-injection seam is exercised end-to-end.
* Reaches the **runner-held** ``ProtocolRegistryModule`` from
  the in-process session and observes weight evolution in
  ``ActiveMixtureSnapshot.strategy_weights`` across consecutive
  ``module.process`` calls.

Why we feed PE via ``module.process`` rather than ``run_turn``
--------------------------------------------------------------

``PredictionErrorModule`` builds PE from internal signal
detectors over the conversation — there is **no operator-facing
knob** at the HTTP layer to make a turn produce a specific
``signed_reward``. Driving PE through real ``run_turn`` calls
would make the test depend on the detector implementation
details and be flaky. The same shortcut is taken by every other
PE-learning test in the repo
(``test_protocol_strategy_weight_learning.py``, etc.).

The seam that this DOES exercise — that those module-level tests
don't — is "the runner-held module is the one that learns." If
this test fails it tells us that ``with_seed_protocols`` produced
a module that is somehow disconnected from the wiring that
publishes ``active_mixture``, or that PE updates are silently
discarded on the runner instance.
"""

from __future__ import annotations

import asyncio
import datetime as _dt

import pytest
from aiohttp import web

from lifeform_service.app import create_app
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from lifeform_service.verticals import discover_verticals
from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolCandidate,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    FailureSignal,
    IdentityAssertion,
    ProtocolProvenance,
    ProtocolSourceKind,
    ReviewStatus,
    StrategyPrior,
    SuccessSignal,
    TemporalArc,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.runtime import Snapshot


# Two distinct rule ids so we can watch one rise and one fall.
_PID = "e2e:pe-learning-bot"
_RULE_FAST = "rule:e2e:fast-learner"
_RULE_SLOW = "rule:e2e:slow-learner"


def _build_protocol() -> BehaviorProtocol:
    """A 2-rule protocol with deliberately asymmetric PE rates.

    ``fast-learner`` has a high reinforce and decay rate so any PE
    moves it sharply. ``slow-learner`` is nearly frozen. After
    positive PE the weights diverge, after negative PE they
    converge back. This makes the ordering assertion in
    ``test_strategy_weights_diverge_under_positive_pe`` robust to
    floating-point noise.
    """

    return BehaviorProtocol(
        protocol_id=_PID,
        version="0.1.0",
        advisor_name="E2E PE Learner",
        description="Two-rule protocol for PE evolution e2e test",
        source_kind=ProtocolSourceKind.API_INJECTION,
        source_locator=f"api-injection://{_PID}",
        identity_assertion=IdentityAssertion(),
        boundary_contracts=(
            BoundaryContract(
                boundary_id="bd:e2e:soft",
                description="placeholder",
                trigger_reasons=("any context",),
                severity=BoundarySeverity.SOFT_REMIND,
            ),
        ),
        activation_conditions=ActivationConditions(),
        strategy_priors=(
            StrategyPrior(
                rule_id=_RULE_FAST,
                problem_pattern="generic",
                recommended_ordering=("step-a",),
                recommended_pacing="moderate",
                initial_weight=1.0,
                pe_reinforce_rate=0.5,
                pe_decay_rate=0.5,
                minimum_weight_floor=0.05,
            ),
            StrategyPrior(
                rule_id=_RULE_SLOW,
                problem_pattern="generic",
                recommended_ordering=("step-a",),
                recommended_pacing="moderate",
                initial_weight=1.0,
                pe_reinforce_rate=0.01,
                pe_decay_rate=0.01,
                minimum_weight_floor=0.05,
            ),
        ),
        temporal_arc=TemporalArc(),
        success_signals=(
            SuccessSignal(
                signal_id="ss:e2e:engaged",
                description="placeholder",
                measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
            ),
        ),
        failure_signals=(
            FailureSignal(
                signal_id="fs:e2e:rupture",
                description="placeholder",
                measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
            ),
        ),
        review_status=ReviewStatus.DRAFT,
        legacy_fixture=False,
    )


def _candidate_for(protocol: BehaviorProtocol) -> BehaviorProtocolCandidate:
    return BehaviorProtocolCandidate(
        protocol=protocol,
        provenance=ProtocolProvenance(
            source_kind=ProtocolSourceKind.API_INJECTION,
            source_locator=protocol.source_locator,
            extracted_at_iso=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            extractor_id="tests/e2e/pe-learning",
            confidence=1.0,
        ),
        requires_review=True,
    )


def _make_pe_snapshot(
    *,
    signed_reward: float,
    turn_index: int,
    bootstrap: bool = False,
) -> Snapshot[PredictionErrorSnapshot]:
    """Inline copy of ``test_protocol_strategy_weight_learning._make_pe_snapshot``.

    Kept local to avoid cross-test import coupling — if that helper's
    construction details ever change the contract test will fail
    first; we just need a stable PE shape that
    ``_update_strategy_weights`` accepts."""

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
        description="e2e actual",
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
        description="e2e next",
        action_context=action_context,
    )
    error = PredictionError(
        task_error=signed_reward,
        relationship_error=signed_reward,
        regime_error=signed_reward,
        action_error=signed_reward,
        magnitude=abs(signed_reward),
        signed_reward=signed_reward,
        description="e2e pe",
    )
    pe_value = PredictionErrorSnapshot(
        evaluated_prediction=None if bootstrap else next_prediction,
        actual_outcome=actual,
        next_prediction=next_prediction,
        error=error,
        turn_index=turn_index,
        bootstrap=bootstrap,
        description="e2e pe-snap",
        action_context=action_context,
    )
    return Snapshot(
        slot_name="prediction_error",
        owner="PredictionErrorModule",
        version=1,
        timestamp_ms=turn_index * 1000,
        value=pe_value,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


async def _build_app_with_seeded_protocol() -> tuple[web.Application, BehaviorProtocol]:
    verticals = discover_verticals()
    if "companion" not in verticals:
        pytest.skip("companion vertical not installed; cannot run e2e")
    protocol = _build_protocol()
    uptake = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=lambda: None,
        ),
        persistence=None,
    )
    await uptake.submit_candidate(_candidate_for(protocol))
    await uptake.approve_pending(_PID, reviewer_id="test")
    app = create_app(
        verticals=verticals,
        default_vertical="companion",
        max_sessions=8,
        idle_eviction_seconds=None,
        protocol_uptake_service=uptake,
    )
    return app, protocol


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def seeded_client(aiohttp_client):
    app, _ = await _build_app_with_seeded_protocol()
    return await aiohttp_client(app)


async def _create_session(client) -> str:
    resp = await client.post("/v1/sessions", json={})
    assert resp.status == 201, await resp.text()
    return (await resp.json())["session_id"]


async def _runner_module(client, sid: str):
    manager = client.app["session_manager"]
    session = await manager.get_session(sid)
    runner = session.brain_session.runner
    module = runner._protocol_registry_module  # noqa: SLF001
    assert module is not None, (
        "runner has no _protocol_registry_module — the with_seed_protocols "
        "wiring is broken."
    )
    return session, module


def _weights_of(session) -> dict[str, float]:
    """Read the per-rule weights for our protocol off the latest
    ``active_mixture`` snapshot.

    Empty dict when the snapshot is missing or the protocol isn't
    represented (which is itself a failure the assertion message
    will surface)."""

    snap = session.latest_active_snapshots.get("active_mixture")
    if snap is None:
        return {}
    return {
        w.rule_id: w.weight
        for w in snap.value.strategy_weights
        if w.protocol_id == _PID
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_runner_module_seeds_initial_weights_from_protocol(seeded_client) -> None:
    """Before any PE pressure, both rules must show their declared
    ``initial_weight``. This is the baseline — without it the PE
    deltas downstream would not be interpretable."""

    sid = await _create_session(seeded_client)
    session, module = await _runner_module(seeded_client, sid)

    seeded = dict(module._strategy_weights[_PID])  # noqa: SLF001
    assert set(seeded.keys()) == {_RULE_FAST, _RULE_SLOW}
    # Both seeded at initial_weight=1.0.
    assert pytest.approx(seeded[_RULE_FAST]) == 1.0
    assert pytest.approx(seeded[_RULE_SLOW]) == 1.0


async def test_positive_pe_reinforces_weights_on_runner_held_module(
    seeded_client,
) -> None:
    """The load-bearing assertion: PE on the **runner-held** module
    actually mutates the strategy weight table the playbook reads.

    Two turns:

    * Turn 1 = bootstrap PE; this caches ``_last_active_weights``
      so turn-2 PE has someone to credit (singleton mixture
      auto-credits the only loaded protocol).
    * Turn 2 = positive PE; ``_update_strategy_weights`` multiplies
      every active rule's weight by ``(1 + reinforce_rate × |reward|)``.

    Because ``fast-learner`` has 50× the reinforce rate of
    ``slow-learner``, both rise but fast rises far more — making
    the ordering observable in the published snapshot."""

    sid = await _create_session(seeded_client)
    session, module = await _runner_module(seeded_client, sid)
    seeded = dict(module._strategy_weights[_PID])  # noqa: SLF001

    # Turn 1 — bootstrap.
    await module.process(
        {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
    )
    # Turn 2 — positive PE.
    await module.process(
        {"prediction_error": _make_pe_snapshot(signed_reward=0.8, turn_index=2)}
    )

    after = dict(module._strategy_weights[_PID])  # noqa: SLF001
    assert after[_RULE_FAST] > seeded[_RULE_FAST], (
        f"fast-learner weight did not rise under positive PE; "
        f"{seeded[_RULE_FAST]} -> {after[_RULE_FAST]}"
    )
    assert after[_RULE_SLOW] > seeded[_RULE_SLOW], (
        f"slow-learner weight did not rise; "
        f"{seeded[_RULE_SLOW]} -> {after[_RULE_SLOW]}"
    )
    # The differential is the qualitative point — both rise, but
    # fast-learner climbs noticeably more (any constant multiplier
    # > 1 would pass; we use 2× as a robust safety margin).
    fast_delta = after[_RULE_FAST] - seeded[_RULE_FAST]
    slow_delta = after[_RULE_SLOW] - seeded[_RULE_SLOW]
    assert fast_delta > 2.0 * slow_delta, (
        f"fast-learner delta ({fast_delta}) should be >> slow delta "
        f"({slow_delta}); PE-rate differentiation isn't propagating "
        f"from StrategyPrior to runtime weight updates."
    )


async def test_negative_pe_decays_weights_on_runner_held_module(
    seeded_client,
) -> None:
    """Symmetric to the positive-PE test: negative reward must decay
    the fast-learner's weight more sharply than the slow-learner's.

    Verifying the decay direction is what proves PE-driven learning
    is genuinely two-sided on the production runner, not just an
    'accumulator that ratchets up'."""

    sid = await _create_session(seeded_client)
    session, module = await _runner_module(seeded_client, sid)
    seeded = dict(module._strategy_weights[_PID])  # noqa: SLF001

    await module.process(
        {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
    )
    await module.process(
        {"prediction_error": _make_pe_snapshot(signed_reward=-0.8, turn_index=2)}
    )

    after = dict(module._strategy_weights[_PID])  # noqa: SLF001
    assert after[_RULE_FAST] < seeded[_RULE_FAST], (
        f"fast-learner weight did not decay under negative PE; "
        f"{seeded[_RULE_FAST]} -> {after[_RULE_FAST]}"
    )
    assert after[_RULE_FAST] < after[_RULE_SLOW], (
        f"after negative PE the fast-learner should sit BELOW the "
        f"slow-learner (steeper decay rate); saw fast={after[_RULE_FAST]} "
        f"slow={after[_RULE_SLOW]}"
    )


async def test_weight_evolution_is_published_in_active_mixture_snapshot(
    seeded_client,
) -> None:
    """The published snapshot — what downstream consumers actually
    read — must reflect the same evolution we observe in the
    internal ``_strategy_weights`` table.

    If this test passes but the previous one does too, we know the
    contract is end-to-end: PE changes the internal weight AND the
    publication of that weight to downstream consumers."""

    sid = await _create_session(seeded_client)
    session, module = await _runner_module(seeded_client, sid)

    # Drive PE.
    await module.process(
        {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
    )
    await module.process(
        {"prediction_error": _make_pe_snapshot(signed_reward=0.8, turn_index=2)}
    )

    # The module's own publish path is invoked by .process; the
    # snapshot it returns IS the one consumers see. We don't need
    # to wait for a full kernel turn — the contract is that the
    # snapshot output of .process matches the internal state.
    final_snap = await module.process(
        {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=3)}
    )
    weights_by_rule = {
        w.rule_id: w
        for w in final_snap.value.strategy_weights
        if w.protocol_id == _PID
    }
    assert set(weights_by_rule.keys()) == {_RULE_FAST, _RULE_SLOW}, (
        f"published strategy_weights missing rules; saw {weights_by_rule.keys()}"
    )
    # Internal state must agree with the publication.
    internal = module._strategy_weights[_PID]  # noqa: SLF001
    for rid, entry in weights_by_rule.items():
        assert pytest.approx(entry.weight) == internal[rid], (
            f"published weight for {rid} ({entry.weight}) disagrees with "
            f"internal table ({internal[rid]}) — SSOT broken between "
            f"ProtocolRegistryModule and its own snapshot publisher."
        )
    # Every published weight entry must carry the canonical
    # downstream join key so StrategyPlaybookModule can re-rank
    # PlaybookRule entries by ``compiled_rule_id``.
    for rid, entry in weights_by_rule.items():
        assert entry.compiled_rule_id == (
            f"protocol:{_PID}:playbook:{rid}"
        ), (
            f"compiled_rule_id wrong on published entry: {entry.compiled_rule_id!r}"
        )


async def test_runner_module_is_stable_across_real_run_turn_calls(
    seeded_client,
) -> None:
    """Sanity guard against the pre-2024 regression where the
    runner's protocol module was rebuilt every turn (losing
    learning).

    After two **real** HTTP turns, the runner instance + module
    instance + ``_strategy_weights[_PID]`` table identity must be
    preserved. If this fails, every other test in this file is
    measuring transient state."""

    sid = await _create_session(seeded_client)
    _, module_t0 = await _runner_module(seeded_client, sid)
    weights_t0_id = id(module_t0._strategy_weights[_PID])  # noqa: SLF001

    await seeded_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "Hi there, first message."},
    )
    await seeded_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "Second message, please."},
    )

    _, module_t2 = await _runner_module(seeded_client, sid)
    assert module_t0 is module_t2, (
        "runner-held ProtocolRegistryModule was replaced between turns; "
        "learning would be reset every turn."
    )
    assert id(module_t2._strategy_weights[_PID]) == weights_t0_id  # noqa: SLF001
