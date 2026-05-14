"""Sub-packet F2 — ``ProtocolRegistryModule._strategy_weights`` MUST stay
in sync with the currently-loaded protocol's ``strategy_priors``
across the revision lifecycle, with snapshot + persistence
defensive filtering as belt-and-braces.

Pre-this-packet (``protocol-online-learning-active``) semantics:

* ``apply_revision`` mutated ``BehaviorProtocol.strategy_priors``
  in place (e.g. ``ADD_STRATEGY``) but never called
  ``_reseed_strategy_weights_for_protocol``. New rule_ids were
  lazy-seeded by ``_update_strategy_weights`` on the next PE turn,
  but rule_ids the revision *removed* (the in-place mutation rules
  ``DEACTIVATE`` / ``WEIGHT_DECAY`` don't drop entries, but a
  ``load_protocol`` re-load with a smaller ``strategy_priors`` set
  would) leaked into ``_strategy_weights`` until next session boot
  picked them up via the hydrate-time intersection.
* ``_build_strategy_weight_entries`` and
  ``export_persistence_snapshot`` echoed every entry in
  ``_strategy_weights`` regardless of whether the rule_id still
  appeared on a loaded protocol's ``strategy_priors``.

Post-this-packet (``protocol-online-learning-followups``,
sub-packet F2):

* ``apply_revision`` calls
  ``_reseed_strategy_weights_for_protocol(revised)`` immediately
  after the registry mutation. Same idempotent merge as
  ``load_protocol``: keep learnt weights for unchanged rule_ids,
  seed new rule_ids at their initial_weight, prune dropped
  rule_ids.
* ``_build_strategy_weight_entries`` and
  ``export_persistence_snapshot`` both intersect their output
  with the currently-loaded protocols' ``strategy_priors`` rule_id
  set. If something stale survives the reseed, neither downstream
  consumers nor cross-session persistence see it.

What this test asserts:

1. ``apply_revision`` with ``ADD_STRATEGY`` extends
   ``_strategy_weights[pid]`` to include the new rule_id at its
   ``initial_weight`` (no PE turn required to seed it).
2. ``apply_revision`` with ``DEACTIVATE`` keeps the existing
   rule_id but doesn't accidentally drop other learnt weights
   (idempotent reseed).
3. ``load_protocol`` re-load with a smaller ``strategy_priors``
   set prunes the dropped rule_id from ``_strategy_weights``.
4. Even if a stale rule_id is force-injected into
   ``_strategy_weights`` (simulating a bug elsewhere),
   ``_build_strategy_weight_entries`` does NOT publish it.
5. Same force-injected stale rule_id is NOT exported by
   ``export_persistence_snapshot``.
6. End-to-end: load → learn → load_protocol with reduced priors
   (revision-equivalent prune) → export → hydrate in session B →
   only surviving rule_ids carry learnt weights.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as _replace
from pathlib import Path

from lifeform_core import Lifeform
from lifeform_core.lifeform import LifeformConfig
from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    ProposalEvidence,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewLevel,
)
from volvence_zero.brain import BrainConfig
from volvence_zero.memory import StaticIdentityProvider, UserIdentity
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


def _make_pe_snapshot(
    *, signed_reward: float, turn_index: int
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
        evaluated_prediction=next_prediction,
        actual_outcome=actual,
        next_prediction=next_prediction,
        error=error,
        turn_index=turn_index,
        bootstrap=False,
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


def _add_strategy_proposal(
    *,
    target_protocol_id: str,
    new_rule_id: str,
    problem_pattern: str,
) -> ProtocolRevisionProposal:
    return ProtocolRevisionProposal(
        proposal_id=f"proposal-add-{new_rule_id}",
        target_protocol_id=target_protocol_id,
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=new_rule_id,
        change_kind=ProtocolRevisionChangeKind.ADD_STRATEGY,
        evidence=ProposalEvidence(
            observation_window_turns=10,
            pe_signature="test",
            summary="contract-test ADD_STRATEGY",
        ),
        proposed_payload={
            "rule_id": new_rule_id,
            "problem_pattern": problem_pattern,
            "recommended_ordering": ["greet", "ask context"],
            "recommended_pacing": "moderate",
        },
        required_review_level=ReviewLevel.L1,
    )


def _deactivate_proposal(
    *,
    target_protocol_id: str,
    rule_id: str,
) -> ProtocolRevisionProposal:
    return ProtocolRevisionProposal(
        proposal_id=f"proposal-deactivate-{rule_id}",
        target_protocol_id=target_protocol_id,
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=rule_id,
        change_kind=ProtocolRevisionChangeKind.DEACTIVATE,
        evidence=ProposalEvidence(
            observation_window_turns=10,
            pe_signature="test",
            summary="contract-test DEACTIVATE",
        ),
        proposed_payload=None,
        required_review_level=ReviewLevel.L1,
    )


# ---------------------------------------------------------------------------
# 1. apply_revision ADD_STRATEGY → reseed extends _strategy_weights
# ---------------------------------------------------------------------------


def test_apply_revision_add_strategy_seeds_new_rule_weight() -> None:
    """``ADD_STRATEGY`` revision adds a new ``StrategyPrior`` to
    ``protocol.strategy_priors``; the F2.1 reseed must surface it
    in ``_strategy_weights[pid]`` at its ``initial_weight``."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)
    pid = protocol.protocol_id
    pre_count = len(module._strategy_weights[pid])  # noqa: SLF001
    assert "added-rule-1" not in module._strategy_weights[pid]  # noqa: SLF001

    proposal = _add_strategy_proposal(
        target_protocol_id=pid,
        new_rule_id="added-rule-1",
        problem_pattern="freshly-added-pattern",
    )
    module.apply_revision(proposal)

    post_table = module._strategy_weights[pid]  # noqa: SLF001
    assert "added-rule-1" in post_table
    assert len(post_table) == pre_count + 1
    # New rule starts at the StrategyPrior's default initial_weight
    # (1.0 because _build_strategy_from_payload uses StrategyPrior
    # field defaults).
    assert post_table["added-rule-1"] == 1.0


# ---------------------------------------------------------------------------
# 2. apply_revision DEACTIVATE preserves existing learnt weights
# ---------------------------------------------------------------------------


def test_apply_revision_deactivate_preserves_other_learnt_weights() -> None:
    """``DEACTIVATE`` doesn't drop strategy_priors, just sets
    initial_weight=0. The reseed MUST keep all rule_ids and
    preserve any non-target weights that were already learnt."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)
    pid = protocol.protocol_id

    # Drive some learning so weights move off 1.0.
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
    learnt_pre = dict(module._strategy_weights[pid])  # noqa: SLF001
    target_rule = next(iter(learnt_pre.keys()))
    other_rules = {rid for rid in learnt_pre if rid != target_rule}
    assert other_rules, "test fixture must have ≥2 strategy priors"

    proposal = _deactivate_proposal(
        target_protocol_id=pid,
        rule_id=target_rule,
    )
    module.apply_revision(proposal)

    learnt_post = module._strategy_weights[pid]  # noqa: SLF001
    # Same rule_ids — DEACTIVATE doesn't drop entries.
    assert set(learnt_post.keys()) == set(learnt_pre.keys())
    # The non-target rules' learnt weights survived the reseed
    # (reseed preserves existing weight when rule_id still in protocol).
    for rid in other_rules:
        assert learnt_post[rid] == learnt_pre[rid]


# ---------------------------------------------------------------------------
# 3. load_protocol re-load with reduced priors prunes dropped rule_id
# ---------------------------------------------------------------------------


def test_reload_protocol_with_dropped_strategy_prunes_weight_table() -> None:
    """When a fresh ``load_protocol`` arrives carrying a smaller
    ``strategy_priors`` tuple (the closest in-tree analogue to a
    revision that drops a rule), ``_strategy_weights`` MUST drop
    the disappeared rule_id."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)
    pid = protocol.protocol_id

    rule_to_drop = protocol.strategy_priors[0].rule_id
    assert rule_to_drop in module._strategy_weights[pid]  # noqa: SLF001

    reduced = _replace(
        protocol,
        strategy_priors=tuple(
            p for p in protocol.strategy_priors if p.rule_id != rule_to_drop
        ),
    )
    module.load_protocol(reduced)
    assert rule_to_drop not in module._strategy_weights[pid]  # noqa: SLF001


# ---------------------------------------------------------------------------
# 4. Defensive filter in _build_strategy_weight_entries
# ---------------------------------------------------------------------------


def test_build_strategy_weight_entries_filters_out_stale_rule_ids() -> None:
    """Force-inject a stale rule_id into ``_strategy_weights`` to
    simulate a bug that bypassed the reseed. The published
    snapshot MUST NOT contain that rule_id."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)
    pid = protocol.protocol_id

    module._strategy_weights[pid]["totally-stale-rule"] = 99.0  # noqa: SLF001
    module._last_strategy_reward.setdefault(pid, {})["totally-stale-rule"] = 0.5  # noqa: SLF001

    snap = asyncio.run(module.process({}))
    rule_ids = {entry.rule_id for entry in snap.value.strategy_weights}
    assert "totally-stale-rule" not in rule_ids


# ---------------------------------------------------------------------------
# 5. Defensive filter in export_persistence_snapshot
# ---------------------------------------------------------------------------


def test_export_persistence_snapshot_filters_out_stale_rule_ids() -> None:
    """Same stale-injection scenario; the cross-session persistence
    payload MUST NOT carry the stale rule_id either."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)
    pid = protocol.protocol_id

    module._strategy_weights[pid]["totally-stale-rule"] = 99.0  # noqa: SLF001
    module._last_strategy_reward.setdefault(pid, {})["totally-stale-rule"] = 0.5  # noqa: SLF001

    payload = module.export_persistence_snapshot().payload
    weights_for_pid = payload["strategy_weights"].get(pid, {})
    assert "totally-stale-rule" not in weights_for_pid
    rewards_for_pid = payload["last_strategy_reward"].get(pid, {})
    assert "totally-stale-rule" not in rewards_for_pid


def test_export_persistence_snapshot_drops_pid_for_unloaded_protocol() -> None:
    """If the entire protocol is gone (unload between learning and
    persistence), its rows MUST NOT appear in the persistence
    payload at all."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    module.load_protocol(protocol)
    pid = protocol.protocol_id
    assert pid in module._strategy_weights  # noqa: SLF001

    module.unload_protocol(pid)
    payload = module.export_persistence_snapshot().payload
    assert pid not in payload["strategy_weights"]
    assert pid not in payload["last_strategy_reward"]


# ---------------------------------------------------------------------------
# 6. End-to-end across LifeformSession boundary with reduced priors
# ---------------------------------------------------------------------------


def test_cross_session_hydrate_does_not_resurrect_pruned_rule_weight(
    tmp_path: Path,
) -> None:
    """Session A learns weights, then ``load_protocol`` re-loads with
    a reduced ``strategy_priors``; persist; session B (same scope
    root + reduced protocol) hydrates and MUST NOT resurrect the
    pruned rule_id."""

    protocol = _patch_with_pe_rates(_cheng_laoshi_protocol())
    rule_to_drop = protocol.strategy_priors[0].rule_id

    identity = UserIdentity(user_id="alice", scope_key="alice")
    brain_cfg = BrainConfig(
        memory_scope_root_dir=str(tmp_path),
        owner_hydration_wiring=WiringLevel.ACTIVE,
    )
    life_a = Lifeform(
        LifeformConfig(brain_config=brain_cfg, seed_protocols=(protocol,)),
        identity_provider=StaticIdentityProvider(identity=identity),
    )
    session_a = life_a.create_session(session_id="alice-revision-1")
    runner_a = session_a.brain_session.runner
    module_a = runner_a._protocol_registry_module  # noqa: SLF001
    pid = protocol.protocol_id

    # Drive learning.
    asyncio.run(
        module_a.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.0, turn_index=1)}
        )
    )
    asyncio.run(
        module_a.process(
            {"prediction_error": _make_pe_snapshot(signed_reward=0.6, turn_index=2)}
        )
    )
    assert rule_to_drop in module_a._strategy_weights[pid]  # noqa: SLF001

    # Simulate revision by re-loading with reduced priors.
    reduced = _replace(
        protocol,
        strategy_priors=tuple(
            p for p in protocol.strategy_priors if p.rule_id != rule_to_drop
        ),
    )
    module_a.load_protocol(reduced)
    assert rule_to_drop not in module_a._strategy_weights[pid]  # noqa: SLF001

    # Persist + tear down.
    persisted = session_a.persist_owners()
    assert "protocol_registry" in persisted
    del session_a
    del life_a

    # Session B: same user, same scope, REDUCED protocol seed.
    life_b = Lifeform(
        LifeformConfig(brain_config=brain_cfg, seed_protocols=(reduced,)),
        identity_provider=StaticIdentityProvider(identity=identity),
    )
    session_b = life_b.create_session(session_id="alice-revision-2")
    module_b = session_b.brain_session.runner._protocol_registry_module  # noqa: SLF001
    assert rule_to_drop not in module_b._strategy_weights[pid]  # noqa: SLF001
