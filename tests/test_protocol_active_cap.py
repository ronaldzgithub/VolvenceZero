"""Packet 6.7: ACTIVE_PROTOCOL_HARD_CAP enforcement tests."""

from __future__ import annotations

from dataclasses import replace as _replace

import pytest

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import ReviewStatus
from volvence_zero.protocol_runtime import (
    ProtocolLimitExceededError,
    ProtocolRegistry,
)


def _bp_with_id(suffix: str, status: ReviewStatus = ReviewStatus.ACTIVE):
    base = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    return _replace(
        base,
        protocol_id=f"growth_advisor:cap-test-{suffix}",
        review_status=status,
    )


def test_default_cap_is_eight() -> None:
    assert ProtocolRegistry.ACTIVE_PROTOCOL_HARD_CAP == 8


def test_loading_up_to_cap_succeeds() -> None:
    registry = ProtocolRegistry()
    for i in range(ProtocolRegistry.ACTIVE_PROTOCOL_HARD_CAP):
        registry.load(_bp_with_id(str(i)))
    assert len(registry.loaded()) == ProtocolRegistry.ACTIVE_PROTOCOL_HARD_CAP


def test_loading_over_cap_raises() -> None:
    registry = ProtocolRegistry()
    for i in range(ProtocolRegistry.ACTIVE_PROTOCOL_HARD_CAP):
        registry.load(_bp_with_id(str(i)))
    with pytest.raises(ProtocolLimitExceededError, match="hard cap"):
        registry.load(_bp_with_id("overflow"))


def test_draft_does_not_count_toward_cap() -> None:
    """DRAFT protocols don't take up cap slots."""
    registry = ProtocolRegistry()
    for i in range(ProtocolRegistry.ACTIVE_PROTOCOL_HARD_CAP):
        registry.load(_bp_with_id(str(i)))
    # A DRAFT additional load should succeed.
    registry.load(_bp_with_id("draft", status=ReviewStatus.DRAFT))
    assert (
        len(registry.loaded_all()) == ProtocolRegistry.ACTIVE_PROTOCOL_HARD_CAP + 1
    )


def test_retired_does_not_count_toward_cap() -> None:
    registry = ProtocolRegistry()
    for i in range(ProtocolRegistry.ACTIVE_PROTOCOL_HARD_CAP):
        registry.load(_bp_with_id(str(i)))
    # Retire one to make room.
    registry.mark_status(
        f"growth_advisor:cap-test-0", ReviewStatus.RETIRED
    )
    # Now loading another active should succeed.
    registry.load(_bp_with_id("after-retire"))


def test_overwriting_existing_protocol_id_does_not_exceed_cap() -> None:
    registry = ProtocolRegistry()
    for i in range(ProtocolRegistry.ACTIVE_PROTOCOL_HARD_CAP):
        registry.load(_bp_with_id(str(i)))
    # Reload first id with a new version — should NOT raise.
    bp = _bp_with_id("0")
    bp_v2 = _replace(bp, version="0.2.0")
    registry.load(bp_v2)
    assert registry.get(bp.protocol_id).version == "0.2.0"
