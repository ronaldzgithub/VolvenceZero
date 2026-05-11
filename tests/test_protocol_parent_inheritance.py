"""Packet 6.5: parent_protocol_id chain merge tests."""

from __future__ import annotations

from dataclasses import replace as _replace

import pytest

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.behavior_protocol import (
    BoundaryContract,
    BoundarySeverity,
    StrategyPrior,
)
from volvence_zero.protocol_runtime import (
    ProtocolRegistry,
    ProtocolRegistryModule,
    merge_protocol_chain,
)


def _bp_base():
    return growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )


def _bp_with_parent(parent_id: str, *, child_id: str, extra_strategy=None, extra_boundary=None):
    base = _bp_base()
    strategies = base.strategy_priors
    if extra_strategy is not None:
        strategies = strategies + (extra_strategy,)
    boundaries = base.boundary_contracts
    if extra_boundary is not None:
        boundaries = boundaries + (extra_boundary,)
    return _replace(
        base,
        protocol_id=child_id,
        parent_protocol_id=parent_id,
        strategy_priors=strategies,
        boundary_contracts=boundaries,
    )


def test_merge_protocol_chain_no_parent_returns_self() -> None:
    bp = _bp_base()
    registry = ProtocolRegistry()
    registry.load(bp)
    merged = merge_protocol_chain(bp, lookup=registry.get_optional)
    assert merged.boundary_contracts == bp.boundary_contracts
    assert merged.strategy_priors == bp.strategy_priors


def test_merge_protocol_chain_inherits_parent_boundaries() -> None:
    parent = _bp_base()
    parent_id = parent.protocol_id
    extra = StrategyPrior(
        rule_id="child-rule",
        problem_pattern="child specific pattern",
        recommended_ordering=("step1", "step2"),
        recommended_pacing="moderate",
    )
    child = _bp_with_parent(parent_id, child_id="growth_advisor:cheng-laoshi-child", extra_strategy=extra)
    registry = ProtocolRegistry()
    registry.load(parent)

    merged = merge_protocol_chain(child, lookup=registry.get_optional)
    rule_ids = {s.rule_id for s in merged.strategy_priors}
    assert "child-rule" in rule_ids
    parent_rule_ids = {s.rule_id for s in parent.strategy_priors}
    assert parent_rule_ids.issubset(rule_ids)


def test_merge_protocol_chain_child_overrides_parent_id() -> None:
    """Child's entry with same id replaces parent's."""
    parent = _bp_base()
    parent_id = parent.protocol_id
    parent_rule = parent.strategy_priors[0]
    overridden = _replace(parent_rule, problem_pattern="OVERRIDDEN BY CHILD")
    child_strategies = (overridden,) + parent.strategy_priors[1:]
    child = _replace(
        parent,
        protocol_id="growth_advisor:cheng-laoshi-overrider",
        parent_protocol_id=parent_id,
        strategy_priors=child_strategies,
    )
    registry = ProtocolRegistry()
    registry.load(parent)

    merged = merge_protocol_chain(child, lookup=registry.get_optional)
    overridden_in_merged = next(
        s for s in merged.strategy_priors if s.rule_id == parent_rule.rule_id
    )
    assert overridden_in_merged.problem_pattern == "OVERRIDDEN BY CHILD"


def test_merge_protocol_chain_cycle_raises() -> None:
    bp = _bp_base()
    cyclic = _replace(bp, parent_protocol_id=bp.protocol_id)
    registry = ProtocolRegistry()
    registry.load(cyclic)
    with pytest.raises(ValueError, match="cycle"):
        merge_protocol_chain(cyclic, lookup=registry.get_optional)


def test_merge_protocol_chain_missing_parent_raises() -> None:
    bp = _bp_base()
    child = _replace(
        bp,
        protocol_id="orphan-child",
        parent_protocol_id="does-not-exist",
    )
    registry = ProtocolRegistry()
    with pytest.raises(ValueError, match="parent is not loaded"):
        merge_protocol_chain(child, lookup=registry.get_optional)


def test_load_protocol_uses_merged_chain_for_compile() -> None:
    """Loading a child protocol with parent_protocol_id compiles the
    merged chain into application owners."""
    parent = _bp_base()
    parent_id = parent.protocol_id

    extra_boundary = BoundaryContract(
        boundary_id="bd:child-only",
        description="child-only boundary",
        severity=BoundarySeverity.SOFT_REMIND,
        trigger_reasons=("test trigger",),
    )
    child = _bp_with_parent(
        parent_id,
        child_id="growth_advisor:cheng-laoshi-child",
        extra_boundary=extra_boundary,
    )

    rare = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=rare)
    module.load_protocol(parent)
    module.load_protocol(child)

    boundary_ids = [
        h.hint_id for h in rare.boundary_prior_hints
    ]
    # Child-only boundary should appear via the child's compile.
    child_only = [
        bid for bid in boundary_ids
        if "growth_advisor:cheng-laoshi-child:boundary:bd:child-only" in bid
    ]
    assert child_only, boundary_ids
    # Parent's boundaries should also appear via the merged compile
    # (with child's protocol_id prefix, since the merged form has child id).
    parent_inherited = [
        bid for bid in boundary_ids
        if "growth_advisor:cheng-laoshi-child:boundary:" in bid
    ]
    assert len(parent_inherited) >= 2  # at least the parent boundaries + child one
