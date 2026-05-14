"""B2 closeout — strategy_weights MUST drive ``StrategyPlaybookSnapshot.matched_rules``
ordering (``protocol-online-learning-followups`` packet, sub-packet F1).

Pre-this-packet semantics:

* ``StrategyPlaybookModule.process`` declared ``active_mixture`` as a
  dependency (packet 5.1) but only read it for the consumer-audit
  readiness gate. The published ``StrategyPlaybookSnapshot.matched_rules``
  carried whatever order the rule-construction loop happened to
  produce — protocol-driven rules that the upstream
  ``ProtocolRegistryModule`` had been continuously learning weights
  for had no influence on which rule a downstream
  ``_response_ordering_plan`` would pick (it always took
  ``matched_rules[0].recommended_ordering``).

Post-this-packet semantics:

* ``ProtocolRegistryModule._build_strategy_weight_entries`` fills
  every ``StrategyWeightEntry.compiled_rule_id`` with the namespaced
  id (``protocol:{protocol_id}:playbook:{raw_rule_id}``) produced by
  ``compile_protocol_to_application_artifacts``. This is byte-identical
  to the corresponding ``PlaybookRule.rule_id`` written into the
  ``ApplicationRareHeavyState.distilled_playbook_rules``.
* ``StrategyPlaybookModule.process`` builds a
  ``compiled_rule_id → weight`` map from
  ``upstream["active_mixture"].value.strategy_weights`` and stable-sorts
  the published ``matched_rules`` by descending weight (default 1.0
  for any rule absent from the map).
* Stable sort + uniform-default-weight ⇒ pre-F1 byte-equivalence
  preserved when no protocol is loaded OR all weights are equal.

What this test asserts:

1. ``compiled_rule_id`` produced by the owner is byte-identical to
   the corresponding ``PlaybookRule.rule_id`` produced by the
   compile path (contract pin so future namespace drift breaks
   visibly).
2. With no ``active_mixture`` upstream (or empty
   ``strategy_weights``), ``matched_rules`` order is preserved.
3. Two protocols, one rule each (different problem_pattern),
   differential weights → ``matched_rules[0]`` is the higher-weight
   protocol's rule.
4. Two rules of the same protocol (different problem_pattern),
   differential weights → high-weight rule moves to position 0.
5. ``compiled_rule_id`` empty on a snapshot entry → that entry is
   ignored (default weight applies).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as _replace

from volvence_zero.application.modules.strategy_playbook import (
    StrategyPlaybookModule,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.types import (
    CaseMemorySnapshot,
    PlaybookRule,
    StrategyPlaybookSnapshot,
)
from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
    StrategyWeightEntry,
)
from volvence_zero.dual_track import DualTrackSnapshot, TrackState
from volvence_zero.memory import Track
from volvence_zero.regime.contracts import RegimeIdentity, RegimeSnapshot
from volvence_zero.protocol_runtime import ProtocolRegistryModule
from volvence_zero.protocol_runtime.compiler import (
    _protocol_rule_id,
    compile_protocol_to_application_artifacts,
)
from volvence_zero.runtime import Snapshot

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)


# ---------------------------------------------------------------------------
# Synthetic upstream snapshot helpers
# ---------------------------------------------------------------------------


def _regime_snapshot(regime_id: str = "test_regime") -> Snapshot[RegimeSnapshot]:
    identity = RegimeIdentity(
        regime_id=regime_id,
        name=regime_id,
        embedding=(0.0,) * 8,
        entry_conditions="test",
        exit_conditions="test",
        historical_effectiveness=0.5,
    )
    return Snapshot(
        slot_name="regime",
        owner="RegimeModule",
        version=1,
        timestamp_ms=0,
        value=RegimeSnapshot(
            active_regime=identity,
            previous_regime=None,
            switch_reason="",
            candidate_regimes=((regime_id, 1.0),),
            turns_in_current_regime=1,
            description="test regime",
        ),
    )


def _dual_track_snapshot() -> Snapshot[DualTrackSnapshot]:
    track = TrackState(
        track=Track.WORLD,
        active_goals=(),
        recent_credits=(),
        controller_code=(0.0,) * 4,
        tension_level=0.0,
    )
    self_track = _replace(track, track=Track.SELF)
    return Snapshot(
        slot_name="dual_track",
        owner="DualTrackModule",
        version=1,
        timestamp_ms=0,
        value=DualTrackSnapshot(
            world_track=track,
            self_track=self_track,
            cross_track_tension=0.0,
            description="test dual track",
        ),
    )


def _case_memory_snapshot(
    *, active_problem_patterns: tuple[str, ...]
) -> Snapshot[CaseMemorySnapshot]:
    return Snapshot(
        slot_name="case_memory",
        owner="CaseMemoryModule",
        version=1,
        timestamp_ms=0,
        value=CaseMemorySnapshot(
            retrieval_policy_id="test",
            hits=(),
            active_problem_patterns=active_problem_patterns,
            active_risk_markers=(),
            description="test case memory",
        ),
    )


def _active_mixture_snapshot(
    *,
    strategy_weights: tuple[StrategyWeightEntry, ...] = (),
    protocol_ids: tuple[str, ...] = (),
) -> Snapshot[ActiveMixtureSnapshot]:
    active_protocols = tuple(
        ActiveProtocolEntry(
            protocol_id=pid,
            activation_weight=1.0 / max(len(protocol_ids), 1),
        )
        for pid in protocol_ids
    )
    return Snapshot(
        slot_name="active_mixture",
        owner="ProtocolRegistryModule",
        version=1,
        timestamp_ms=0,
        value=ActiveMixtureSnapshot(
            active_protocols=active_protocols,
            boundary_union_ids=(),
            description="test active mixture",
            strategy_weights=strategy_weights,
        ),
    )


def _build_upstream(
    *,
    active_problem_patterns: tuple[str, ...],
    active_mixture: Snapshot[ActiveMixtureSnapshot] | None = None,
) -> dict:
    upstream: dict = {
        "case_memory": _case_memory_snapshot(
            active_problem_patterns=active_problem_patterns
        ),
        "regime": _regime_snapshot(),
        "dual_track": _dual_track_snapshot(),
    }
    if active_mixture is not None:
        upstream["active_mixture"] = active_mixture
    return upstream


def _make_synthetic_protocol(
    *,
    protocol_id: str,
    rule_specs: tuple[tuple[str, str], ...],
):
    """Return a BehaviorProtocol clone of cheng_laoshi with its
    strategy_priors replaced by ``rule_specs`` ([(rule_id, problem_pattern)]).

    Reuses the cheng_laoshi fixture for all the other required
    fields (PE signals / drives / boundaries) which the playbook
    module doesn't read."""

    base = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    # Take one canonical prior as a template, replicate with new
    # ids/patterns. Patching only rule_id + problem_pattern keeps
    # everything else (initial_weight=1.0, pe_rates, ordering, ...)
    # at the source defaults.
    template = base.strategy_priors[0]
    new_priors = tuple(
        _replace(template, rule_id=rid, problem_pattern=pp)
        for rid, pp in rule_specs
    )
    return _replace(base, protocol_id=protocol_id, strategy_priors=new_priors)


def _populate_state_from_protocol(
    *, protocols: tuple,
) -> tuple[ApplicationRareHeavyState, ProtocolRegistryModule]:
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    for protocol in protocols:
        module.load_protocol(protocol)
    return state, module


# ---------------------------------------------------------------------------
# 1. compiled_rule_id contract pin
# ---------------------------------------------------------------------------


def test_compiled_rule_id_byte_equal_to_compiler_output() -> None:
    """``StrategyWeightEntry.compiled_rule_id`` filled by the owner
    MUST equal the corresponding ``PlaybookRule.rule_id`` produced
    by the compile path. Future drift in either side breaks this.
    """

    protocol = _make_synthetic_protocol(
        protocol_id="ptrA",
        rule_specs=(("raw_a", "pattern_a"), ("raw_b", "pattern_b")),
    )
    artifacts = compile_protocol_to_application_artifacts(protocol)
    expected_rule_ids = {rule.rule_id for rule in artifacts.playbook_rules}

    state, module = _populate_state_from_protocol(protocols=(protocol,))
    asyncio.run(module.process({}))
    snapshot = module.publish.__self__._registry  # noqa: SLF001 - sanity only
    assert snapshot is not None  # registry exists

    # Round-trip through process() to populate the snapshot.
    snap = asyncio.run(module.process({}))
    entries = snap.value.strategy_weights
    actual_compiled_ids = {entry.compiled_rule_id for entry in entries}
    assert actual_compiled_ids == expected_rule_ids
    # Helper used by the owner is the same one the compiler uses.
    for entry in entries:
        assert entry.compiled_rule_id == _protocol_rule_id(
            entry.protocol_id, entry.rule_id
        )


# ---------------------------------------------------------------------------
# 2. Empty / missing active_mixture → matched_rules order preserved
# ---------------------------------------------------------------------------


def test_no_active_mixture_preserves_matched_rules_order() -> None:
    """Without ``active_mixture`` upstream, the playbook module MUST
    publish rules in their construction order (pre-F1 byte-equivalent).
    """

    protocol = _make_synthetic_protocol(
        protocol_id="ptrA",
        rule_specs=(("raw_a", "pattern_a"), ("raw_b", "pattern_b")),
    )
    state, _module = _populate_state_from_protocol(protocols=(protocol,))
    pb = StrategyPlaybookModule(rare_heavy_state=state)
    upstream = _build_upstream(
        active_problem_patterns=("pattern_a", "pattern_b"),
        active_mixture=None,
    )
    snap = asyncio.run(pb.process(upstream))
    rule_ids = [rule.rule_id for rule in snap.value.matched_rules]
    # Construction order: pattern_a first then pattern_b.
    assert rule_ids[0].endswith(":raw_a")
    assert rule_ids[1].endswith(":raw_b")


def test_empty_strategy_weights_preserves_order() -> None:
    """``active_mixture`` present but empty ``strategy_weights`` →
    every rule defaults to weight 1.0 → stable sort keeps original
    order."""

    protocol = _make_synthetic_protocol(
        protocol_id="ptrA",
        rule_specs=(("raw_a", "pattern_a"), ("raw_b", "pattern_b")),
    )
    state, _module = _populate_state_from_protocol(protocols=(protocol,))
    pb = StrategyPlaybookModule(rare_heavy_state=state)
    upstream = _build_upstream(
        active_problem_patterns=("pattern_a", "pattern_b"),
        active_mixture=_active_mixture_snapshot(strategy_weights=()),
    )
    snap = asyncio.run(pb.process(upstream))
    rule_ids = [rule.rule_id for rule in snap.value.matched_rules]
    assert rule_ids[0].endswith(":raw_a")
    assert rule_ids[1].endswith(":raw_b")


# ---------------------------------------------------------------------------
# 3. Differential weights drive ranking
# ---------------------------------------------------------------------------


def test_high_weight_rule_moves_to_position_zero_two_protocols() -> None:
    """Two protocols, one rule each (different patterns), high-weight
    protocol's rule MUST be at ``matched_rules[0]``."""

    protocol_a = _make_synthetic_protocol(
        protocol_id="ptrA",
        rule_specs=(("raw_a", "pattern_a"),),
    )
    protocol_b = _make_synthetic_protocol(
        protocol_id="ptrB",
        rule_specs=(("raw_b", "pattern_b"),),
    )
    state, _module = _populate_state_from_protocol(
        protocols=(protocol_a, protocol_b)
    )
    pb = StrategyPlaybookModule(rare_heavy_state=state)

    weights = (
        StrategyWeightEntry(
            protocol_id="ptrA",
            rule_id="raw_a",
            weight=0.5,
            compiled_rule_id=_protocol_rule_id("ptrA", "raw_a"),
        ),
        StrategyWeightEntry(
            protocol_id="ptrB",
            rule_id="raw_b",
            weight=2.5,
            compiled_rule_id=_protocol_rule_id("ptrB", "raw_b"),
        ),
    )
    upstream = _build_upstream(
        active_problem_patterns=("pattern_a", "pattern_b"),
        active_mixture=_active_mixture_snapshot(strategy_weights=weights),
    )
    snap = asyncio.run(pb.process(upstream))
    rule_ids = [rule.rule_id for rule in snap.value.matched_rules]
    assert rule_ids[0] == _protocol_rule_id("ptrB", "raw_b")
    assert rule_ids[1] == _protocol_rule_id("ptrA", "raw_a")


def test_high_weight_rule_moves_to_position_zero_same_protocol() -> None:
    """Two rules in one protocol (different patterns), high-weight
    rule MUST be at position 0 even though insertion order put the
    low-weight one first."""

    protocol = _make_synthetic_protocol(
        protocol_id="ptrA",
        rule_specs=(("raw_low", "pattern_low"), ("raw_high", "pattern_high")),
    )
    state, _module = _populate_state_from_protocol(protocols=(protocol,))
    pb = StrategyPlaybookModule(rare_heavy_state=state)

    weights = (
        StrategyWeightEntry(
            protocol_id="ptrA",
            rule_id="raw_low",
            weight=0.1,
            compiled_rule_id=_protocol_rule_id("ptrA", "raw_low"),
        ),
        StrategyWeightEntry(
            protocol_id="ptrA",
            rule_id="raw_high",
            weight=5.0,
            compiled_rule_id=_protocol_rule_id("ptrA", "raw_high"),
        ),
    )
    upstream = _build_upstream(
        active_problem_patterns=("pattern_low", "pattern_high"),
        active_mixture=_active_mixture_snapshot(strategy_weights=weights),
    )
    snap = asyncio.run(pb.process(upstream))
    rule_ids = [rule.rule_id for rule in snap.value.matched_rules]
    assert rule_ids[0] == _protocol_rule_id("ptrA", "raw_high")
    assert rule_ids[1] == _protocol_rule_id("ptrA", "raw_low")


# ---------------------------------------------------------------------------
# 4. compiled_rule_id="" entries are ignored
# ---------------------------------------------------------------------------


def test_entry_with_empty_compiled_rule_id_is_ignored() -> None:
    """A weight entry with empty ``compiled_rule_id`` (legacy /
    hand-built fixture) MUST NOT participate in ranking. The rule
    falls back to default weight 1.0; ranking uses only entries
    with a non-empty join key."""

    protocol = _make_synthetic_protocol(
        protocol_id="ptrA",
        rule_specs=(("raw_a", "pattern_a"), ("raw_b", "pattern_b")),
    )
    state, _module = _populate_state_from_protocol(protocols=(protocol,))
    pb = StrategyPlaybookModule(rare_heavy_state=state)

    weights = (
        # raw_a entry has no compiled_rule_id → ignored.
        StrategyWeightEntry(
            protocol_id="ptrA",
            rule_id="raw_a",
            weight=99.0,
            compiled_rule_id="",
        ),
        # raw_b is properly tagged with a high weight.
        StrategyWeightEntry(
            protocol_id="ptrA",
            rule_id="raw_b",
            weight=2.0,
            compiled_rule_id=_protocol_rule_id("ptrA", "raw_b"),
        ),
    )
    upstream = _build_upstream(
        active_problem_patterns=("pattern_a", "pattern_b"),
        active_mixture=_active_mixture_snapshot(strategy_weights=weights),
    )
    snap = asyncio.run(pb.process(upstream))
    rule_ids = [rule.rule_id for rule in snap.value.matched_rules]
    # raw_b (weight=2.0) ranks before raw_a (default 1.0); the
    # nominal 99.0 on the empty-compiled_rule_id entry must be
    # ignored.
    assert rule_ids[0] == _protocol_rule_id("ptrA", "raw_b")
    assert rule_ids[1] == _protocol_rule_id("ptrA", "raw_a")


# ---------------------------------------------------------------------------
# 5. Returned snapshot type sanity
# ---------------------------------------------------------------------------


def test_published_snapshot_is_strategy_playbook_snapshot() -> None:
    protocol = _make_synthetic_protocol(
        protocol_id="ptrA",
        rule_specs=(("raw_a", "pattern_a"),),
    )
    state, _module = _populate_state_from_protocol(protocols=(protocol,))
    pb = StrategyPlaybookModule(rare_heavy_state=state)
    snap = asyncio.run(
        pb.process(
            _build_upstream(
                active_problem_patterns=("pattern_a",),
                active_mixture=None,
            )
        )
    )
    assert isinstance(snap.value, StrategyPlaybookSnapshot)
    assert snap.value.matched_rules
    assert isinstance(snap.value.matched_rules[0], PlaybookRule)
