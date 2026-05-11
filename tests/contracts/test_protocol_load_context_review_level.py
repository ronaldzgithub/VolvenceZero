"""Packet 6.6: LoadContext review_level enforcement contract tests."""

from __future__ import annotations

from dataclasses import replace as _replace

import pytest

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    BoundaryContract,
    BoundarySeverity,
    IdentityAssertion,
    ReviewLevel,
)
from volvence_zero.protocol_runtime import LoadContext, ProtocolRegistryModule


def _bp_with_severity(severity: BoundarySeverity):
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    if not bp.boundary_contracts:
        return bp
    new_first = _replace(bp.boundary_contracts[0], severity=severity)
    new_boundaries = (new_first,) + bp.boundary_contracts[1:]
    return _replace(bp, boundary_contracts=new_boundaries)


def test_load_without_context_uses_legacy_trust_path() -> None:
    """No load_context → legacy load (trust caller, no enforcement)."""
    module = ProtocolRegistryModule()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    assert module.registry.get(bp.protocol_id) is not None


def test_load_with_l1_context_blocked_when_protocol_needs_higher() -> None:
    """cheng_laoshi has HARD_BLOCK boundaries → required L4."""
    module = ProtocolRegistryModule()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    ctx = LoadContext(reviewer_id="r1", reviewer_level=ReviewLevel.L1)
    with pytest.raises(PermissionError, match="reviewer_level="):
        module.load_protocol(bp, load_context=ctx)


def test_load_with_l4_context_passes_for_full_protocol() -> None:
    module = ProtocolRegistryModule()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    ctx = LoadContext(reviewer_id="r1", reviewer_level=ReviewLevel.L4)
    module.load_protocol(bp, load_context=ctx)
    assert module.registry.get(bp.protocol_id) is not None


def test_l3_context_passes_for_identity_only_protocol() -> None:
    """Protocol with identity_assertion but no HARD_BLOCK → L3 sufficient."""
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    # Strip HARD_BLOCK / ESCALATE_HUMAN boundaries; keep SOFT_REMIND only.
    softened = tuple(
        _replace(b, severity=BoundarySeverity.SOFT_REMIND)
        for b in bp.boundary_contracts
    )
    bp_no_hardblock = _replace(bp, boundary_contracts=softened)
    module = ProtocolRegistryModule()
    ctx = LoadContext(reviewer_id="r1", reviewer_level=ReviewLevel.L3)
    module.load_protocol(bp_no_hardblock, load_context=ctx)
    assert module.registry.get(bp_no_hardblock.protocol_id) is not None


def test_hard_block_severity_requires_l4() -> None:
    """Protocol with HARD_BLOCK boundary needs L4 reviewer."""
    module = ProtocolRegistryModule()
    bp = _bp_with_severity(BoundarySeverity.HARD_BLOCK)
    ctx_l3 = LoadContext(reviewer_id="r1", reviewer_level=ReviewLevel.L3)
    with pytest.raises(PermissionError, match="reviewer_level=l4"):
        module.load_protocol(bp, load_context=ctx_l3)
    # L4 should pass.
    module2 = ProtocolRegistryModule()
    ctx_l4 = LoadContext(reviewer_id="r2", reviewer_level=ReviewLevel.L4)
    module2.load_protocol(bp, load_context=ctx_l4)
    assert module2.registry.get(bp.protocol_id) is not None


def test_low_risk_protocol_allows_l1() -> None:
    """Protocol with no identity / no boundaries → L1 passes."""
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    bp_low_risk = _replace(
        bp,
        identity_assertion=IdentityAssertion(),
        boundary_contracts=(),
    )
    module = ProtocolRegistryModule()
    ctx = LoadContext(reviewer_id="r1", reviewer_level=ReviewLevel.L1)
    module.load_protocol(bp_low_risk, load_context=ctx)
    assert module.registry.get(bp_low_risk.protocol_id) is not None
