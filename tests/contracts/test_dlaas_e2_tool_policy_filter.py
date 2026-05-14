"""Contract test: E2 DLaaS tool policy filter (debt #16).

Validates:

1. ``AffordanceRegistry.set_contract_policy(contract_id, ...)`` accepts
   a whitelist of allowed affordance names.
2. ``list_for_session(contract_id=...)`` filters descriptors per the
   policy: only names in the whitelist are returned, preserving
   registration order.
3. ``contract_id=None`` returns the legacy full list (back-compat).
4. Unknown ``contract_id`` (no policy registered) returns the legacy
   full list (default-allow until a policy is pushed).
5. Empty contract_id rejected at set_contract_policy.

Refs:

* docs/known-debts.md #16
"""

from __future__ import annotations

import pytest

from lifeform_affordance.registry import AffordanceRegistry
from volvence_zero.affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceSafety,
)


def _descriptor(name: str) -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=AffordanceKind.TOOL,
        version="1.0.0",
        display_name=f"Test {name}",
        description=(
            f"Test affordance {name}. This is a deliberately long "
            "description used to satisfy any minimum character "
            "requirements defined by AffordanceDescriptor.__post_init__."
        ),
        when_to_use=(
            "Use this affordance whenever the test scenario calls "
            "for it; this string is intentionally verbose so that "
            "the descriptor passes any when_to_use length checks."
        ),
        when_not_to_use=(
            "Do not use this affordance outside the test fixture; "
            "this string is intentionally long enough to satisfy "
            "any minimum-length validators in the descriptor."
        ),
        parameters_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
        safety_model=AffordanceSafety(),
        affordance_tags=("test",),
    )


def _seeded_registry() -> AffordanceRegistry:
    registry = AffordanceRegistry()
    for name in ("read_file", "write_file", "list_dir", "search_web"):
        registry.register(_descriptor(name))
    return registry


def test_set_contract_policy_whitelist_filters_session_view() -> None:
    registry = _seeded_registry()
    registry.set_contract_policy(
        contract_id="ctr-001",
        allowed_affordance_names=("read_file", "list_dir"),
    )
    visible = registry.list_for_session(contract_id="ctr-001")
    visible_names = [d.name for d in visible]
    # Whitelist filtering kicks in.
    assert visible_names == ["read_file", "list_dir"]


def test_set_contract_policy_preserves_registration_order() -> None:
    registry = _seeded_registry()
    registry.set_contract_policy(
        contract_id="ctr-001",
        allowed_affordance_names=("search_web", "read_file"),  # whitelist out-of-order
    )
    visible = registry.list_for_session(contract_id="ctr-001")
    # Registration order is preserved (read_file before search_web).
    assert [d.name for d in visible] == ["read_file", "search_web"]


def test_no_contract_policy_returns_full_list_back_compat() -> None:
    """Default-allow back-compat: contracts without a policy see all tools."""
    registry = _seeded_registry()
    visible = registry.list_for_session(contract_id="ctr-no-policy")
    assert len(visible) == 4


def test_contract_id_none_returns_full_list() -> None:
    """``contract_id=None`` is legacy callers — keep their behaviour intact."""
    registry = _seeded_registry()
    visible = registry.list_for_session(contract_id=None)
    assert len(visible) == 4


def test_set_contract_policy_rejects_empty_contract_id() -> None:
    registry = _seeded_registry()
    with pytest.raises(ValueError, match="contract_id must be non-empty"):
        registry.set_contract_policy(
            contract_id="",
            allowed_affordance_names=("read_file",),
        )


def test_unknown_whitelist_names_silently_skipped() -> None:
    """Stale policy w/ deregistered tool names doesn't crash."""
    registry = _seeded_registry()
    registry.set_contract_policy(
        contract_id="ctr-001",
        allowed_affordance_names=("read_file", "removed_tool", "list_dir"),
    )
    visible = registry.list_for_session(contract_id="ctr-001")
    assert [d.name for d in visible] == ["read_file", "list_dir"]


def test_has_contract_policy_predicate() -> None:
    registry = _seeded_registry()
    assert registry.has_contract_policy("ctr-001") is False
    registry.set_contract_policy(
        contract_id="ctr-001",
        allowed_affordance_names=(),
    )
    assert registry.has_contract_policy("ctr-001") is True


def test_empty_whitelist_blocks_everything() -> None:
    """Empty whitelist = explicit lockdown (no tools available)."""
    registry = _seeded_registry()
    registry.set_contract_policy(
        contract_id="ctr-locked",
        allowed_affordance_names=(),
    )
    assert registry.list_for_session(contract_id="ctr-locked") == ()
