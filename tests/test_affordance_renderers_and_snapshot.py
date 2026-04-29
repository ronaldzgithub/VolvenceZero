"""Tests for the 4 renderers + AffordanceSnapshot (Gap 1 slice 1)."""

from __future__ import annotations

import json

import pytest

from lifeform_affordance import (
    AffordanceCandidate,
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceRegistry,
    AffordanceSafety,
    AffordanceSnapshot,
    build_neutral_snapshot,
    render_catalog_json,
    render_compact_list,
    render_markdown,
    render_openai_tools,
)


_HINT = (
    "This is a hint long enough to clear the 50-character minimum; "
    "production descriptors carry much richer copy."
)


def _tool(
    name: str,
    *,
    tags: tuple[str, ...] = (),
    excluded: bool = False,
    kind: AffordanceKind = AffordanceKind.TOOL,
) -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=kind,
        version="0.1.0",
        display_name=name.replace("_", " ").title(),
        description=f"Test descriptor {name}.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " (negative).",
        parameters_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        output_schema={"type": "object"},
        cost_model=AffordanceCost(
            latency_class=AffordanceLatencyClass.FAST,
            monetary_class=AffordanceMonetaryClass.FREE,
        ),
        safety_model=AffordanceSafety(),
        affordance_tags=tags,
        examples=(f"{name}(q='example')",),
        excluded_from_runtime_selection=excluded,
    )


def _populated_registry() -> AffordanceRegistry:
    registry = AffordanceRegistry()
    registry.register_all(
        [
            _tool("read_file", tags=("read",)),
            _tool("grep", tags=("search",)),
            _tool("admin_reset", excluded=True),
            _tool("clarify", kind=AffordanceKind.ACTION),
        ]
    )
    return registry


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


def test_render_markdown_contains_each_non_excluded_affordance() -> None:
    registry = _populated_registry()
    md = render_markdown(registry.all_descriptors())
    assert "## read_file (tool, v0.1.0)" in md
    assert "## grep (tool, v0.1.0)" in md
    assert "## clarify (action, v0.1.0)" in md
    # Excluded by default.
    assert "admin_reset" not in md


def test_render_markdown_orders_deterministically() -> None:
    registry = _populated_registry()
    md_first = render_markdown(registry.all_descriptors())
    md_second = render_markdown(registry.all_descriptors())
    assert md_first == md_second


def test_render_markdown_mentions_when_to_use_and_not_to_use() -> None:
    registry = _populated_registry()
    md = render_markdown(registry.all_descriptors())
    assert "**When to use**" in md
    assert "**When NOT to use**" in md


def test_render_markdown_shows_placeholder_for_empty_catalog() -> None:
    empty = AffordanceRegistry()
    md = render_markdown(empty.all_descriptors())
    assert "no affordances" in md.lower()


def test_render_markdown_includes_excluded_when_flag_false() -> None:
    registry = _populated_registry()
    md = render_markdown(
        registry.all_descriptors(),
        exclude_excluded_from_runtime_selection=False,
    )
    assert "admin_reset" in md


# ---------------------------------------------------------------------------
# render_openai_tools
# ---------------------------------------------------------------------------


def test_render_openai_tools_returns_only_tools_by_default() -> None:
    registry = _populated_registry()
    tools = render_openai_tools(registry.all_descriptors())
    names = [t["function"]["name"] for t in tools]
    # Tools only: excludes the action "clarify" and excluded admin_reset.
    assert "read_file" in names
    assert "grep" in names
    assert "clarify" not in names
    assert "admin_reset" not in names


def test_render_openai_tools_shape_matches_openai_contract() -> None:
    registry = _populated_registry()
    tools = render_openai_tools(registry.all_descriptors())
    for tool in tools:
        assert tool["type"] == "function"
        func = tool["function"]
        assert isinstance(func["name"], str)
        assert "description" in func
        assert func["parameters"]["type"] == "object"
        # Description carries both when_to_use and when_not_to_use.
        assert "When to use:" in func["description"]
        assert "When NOT to use:" in func["description"]


def test_render_openai_tools_can_include_all_kinds_when_requested() -> None:
    registry = _populated_registry()
    tools = render_openai_tools(
        registry.all_descriptors(),
        only_kinds=frozenset(AffordanceKind),
    )
    names = {t["function"]["name"] for t in tools}
    assert "clarify" in names
    # admin_reset is still excluded via excluded_from_runtime_selection.
    assert "admin_reset" not in names


# ---------------------------------------------------------------------------
# render_catalog_json
# ---------------------------------------------------------------------------


def test_render_catalog_json_returns_every_descriptor_by_default() -> None:
    registry = _populated_registry()
    catalog = render_catalog_json(registry.all_descriptors())
    assert catalog["count"] == 4
    names = {a["name"] for a in catalog["affordances"]}
    assert names == {"read_file", "grep", "admin_reset", "clarify"}


def test_render_catalog_json_is_json_serializable() -> None:
    registry = _populated_registry()
    catalog = render_catalog_json(registry.all_descriptors())
    # Round-trip through JSON.
    round_tripped = json.loads(json.dumps(catalog))
    assert round_tripped["count"] == 4


def test_render_catalog_json_exposes_full_safety_envelope() -> None:
    registry = AffordanceRegistry()
    registry.register(
        AffordanceDescriptor(
            name="danger_tool",
            kind=AffordanceKind.TOOL,
            version="0.1.0",
            display_name="Dangerous tool",
            description="test",
            when_to_use=_HINT,
            when_not_to_use=_HINT + " neg",
            parameters_schema={"type": "object"},
            output_schema={"type": "object"},
            cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.SLOW),
            safety_model=AffordanceSafety(
                requires_user_confirmation=True,
                irreversible=True,
                requires_consent_grant=("filesystem_write",),
                blocked_in_regimes=("casual_social",),
                audit_required=True,
            ),
        )
    )
    catalog = render_catalog_json(registry.all_descriptors())
    danger = catalog["affordances"][0]
    safety = danger["safety_model"]
    assert safety["requires_user_confirmation"] is True
    assert safety["irreversible"] is True
    assert safety["requires_consent_grant"] == ["filesystem_write"]
    assert safety["blocked_in_regimes"] == ["casual_social"]
    assert safety["audit_required"] is True


def test_render_catalog_json_can_exclude_runtime_hidden_when_requested() -> None:
    registry = _populated_registry()
    catalog = render_catalog_json(
        registry.all_descriptors(),
        include_excluded_from_runtime_selection=False,
    )
    names = {a["name"] for a in catalog["affordances"]}
    assert "admin_reset" not in names
    assert catalog["count"] == 3


# ---------------------------------------------------------------------------
# render_compact_list
# ---------------------------------------------------------------------------


def test_render_compact_list_returns_name_display_pairs() -> None:
    registry = _populated_registry()
    compact = render_compact_list(registry.all_descriptors())
    # excluded admin_reset filtered by default.
    assert compact == (
        ("read_file", "Read File"),
        ("grep", "Grep"),
        ("clarify", "Clarify"),
    )


def test_render_compact_list_is_deterministic() -> None:
    registry = _populated_registry()
    assert render_compact_list(registry.all_descriptors()) == render_compact_list(
        registry.all_descriptors()
    )


# ---------------------------------------------------------------------------
# AffordanceSnapshot / build_neutral_snapshot
# ---------------------------------------------------------------------------


def test_neutral_snapshot_contains_every_non_excluded_descriptor() -> None:
    registry = _populated_registry()
    snapshot = build_neutral_snapshot(registry)
    names = {c.descriptor_name for c in snapshot.candidates_for_turn}
    assert names == {"read_file", "grep", "clarify"}
    # Every candidate scored neutrally.
    assert all(c.score == 0.5 for c in snapshot.candidates_for_turn)
    assert snapshot.selected is None


def test_neutral_snapshot_respects_excluded_flag() -> None:
    registry = _populated_registry()
    snapshot = build_neutral_snapshot(
        registry, include_excluded_from_runtime_selection=True
    )
    names = {c.descriptor_name for c in snapshot.candidates_for_turn}
    assert "admin_reset" in names


def test_candidate_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="score"):
        AffordanceCandidate(
            descriptor_name="x",
            score=1.5,
            rationale="",
            expected_cost=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
        )


def test_candidate_rejects_empty_descriptor_name() -> None:
    with pytest.raises(ValueError, match="descriptor_name"):
        AffordanceCandidate(
            descriptor_name="  ",
            score=0.5,
            rationale="",
            expected_cost=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
        )


def test_candidate_is_blocked_tracks_blocked_reason() -> None:
    permitted = AffordanceCandidate(
        descriptor_name="read_file",
        score=0.5,
        rationale="",
        expected_cost=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
    )
    blocked = AffordanceCandidate(
        descriptor_name="danger",
        score=0.0,
        rationale="",
        expected_cost=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
        blocked_reason="boundary:filesystem_write not granted",
    )
    assert permitted.is_blocked is False
    assert blocked.is_blocked is True


def test_snapshot_rejects_candidate_not_in_available() -> None:
    with pytest.raises(ValueError, match="available"):
        AffordanceSnapshot(
            available=(),
            candidates_for_turn=(
                AffordanceCandidate(
                    descriptor_name="orphan",
                    score=0.5,
                    rationale="",
                    expected_cost=AffordanceCost(
                        latency_class=AffordanceLatencyClass.FAST
                    ),
                ),
            ),
            selected=None,
        )


def test_snapshot_rejects_selected_not_in_candidates() -> None:
    registry = _populated_registry()
    snap = build_neutral_snapshot(registry)
    stray = AffordanceCandidate(
        descriptor_name="read_file",
        score=0.9,
        rationale="",
        expected_cost=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
    )
    # "read_file" IS in candidates, so this is valid.
    legal = AffordanceSnapshot(
        available=snap.available,
        candidates_for_turn=snap.candidates_for_turn,
        selected=stray,
    )
    assert legal.selected is not None
    # But a descriptor NOT in candidates is rejected.
    orphan = AffordanceCandidate(
        descriptor_name="not-in-registry",
        score=0.9,
        rationale="",
        expected_cost=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
    )
    with pytest.raises(ValueError, match="candidates_for_turn"):
        AffordanceSnapshot(
            available=snap.available,
            candidates_for_turn=snap.candidates_for_turn,
            selected=orphan,
        )
