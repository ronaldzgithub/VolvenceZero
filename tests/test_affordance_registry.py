"""AffordanceRegistry unit tests."""

from __future__ import annotations

import pytest

from lifeform_affordance import (
    AffordanceAlreadyRegisteredError,
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceRegistry,
    AffordanceRegistrySealedError,
    AffordanceSafety,
)


_HINT = (
    "This is a hint long enough to clear the 50-character minimum; "
    "production descriptors carry much richer copy."
)


def _descriptor(name: str, *, kind: AffordanceKind = AffordanceKind.TOOL) -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=kind,
        version="0.1.0",
        display_name=name.replace("_", " ").title(),
        description=f"Test descriptor {name}.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " (negative).",
        parameters_schema={"type": "object"},
        output_schema={"type": "object"},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
        safety_model=AffordanceSafety(),
        affordance_tags=("test",),
    )


def test_empty_registry_has_zero_length() -> None:
    registry = AffordanceRegistry()
    assert len(registry) == 0
    assert registry.names() == ()
    assert registry.all_descriptors() == ()


def test_register_inserts_in_order() -> None:
    registry = AffordanceRegistry()
    registry.register(_descriptor("alpha"))
    registry.register(_descriptor("bravo"))
    registry.register(_descriptor("charlie"))
    assert registry.names() == ("alpha", "bravo", "charlie")
    descriptors = registry.all_descriptors()
    assert [d.name for d in descriptors] == ["alpha", "bravo", "charlie"]


def test_register_rejects_duplicate_name() -> None:
    registry = AffordanceRegistry()
    registry.register(_descriptor("dup"))
    with pytest.raises(AffordanceAlreadyRegisteredError, match="dup"):
        registry.register(_descriptor("dup"))


def test_register_all_is_atomic_on_failure() -> None:
    """If any descriptor in a batch collides with an existing name,
    NONE of the batch is applied (atomic).
    """
    registry = AffordanceRegistry()
    registry.register(_descriptor("alpha"))
    with pytest.raises(AffordanceAlreadyRegisteredError, match="alpha"):
        registry.register_all(
            [
                _descriptor("bravo"),
                _descriptor("alpha"),  # collides
                _descriptor("charlie"),
            ]
        )
    # Pre-existing alpha is the only thing registered.
    assert registry.names() == ("alpha",)


def test_register_all_rejects_internal_duplicates() -> None:
    registry = AffordanceRegistry()
    with pytest.raises(AffordanceAlreadyRegisteredError, match="dup"):
        registry.register_all(
            [
                _descriptor("dup"),
                _descriptor("dup"),
            ]
        )
    assert registry.names() == ()


def test_get_returns_descriptor() -> None:
    registry = AffordanceRegistry()
    registry.register(_descriptor("read_file"))
    d = registry.get("read_file")
    assert d.name == "read_file"


def test_get_raises_on_unknown_name() -> None:
    registry = AffordanceRegistry()
    with pytest.raises(KeyError, match="unknown-tool"):
        registry.get("unknown-tool")


def test_try_get_returns_none_on_unknown_name() -> None:
    registry = AffordanceRegistry()
    assert registry.try_get("unknown") is None
    registry.register(_descriptor("known"))
    assert registry.try_get("known") is not None


def test_contains_reports_membership() -> None:
    registry = AffordanceRegistry()
    registry.register(_descriptor("alpha"))
    assert "alpha" in registry
    assert "bravo" not in registry
    assert 42 not in registry  # non-string keys return False


def test_by_kind_filters_correctly() -> None:
    registry = AffordanceRegistry()
    registry.register(_descriptor("t1", kind=AffordanceKind.TOOL))
    registry.register(_descriptor("a1", kind=AffordanceKind.ACTION))
    registry.register(_descriptor("t2", kind=AffordanceKind.TOOL))
    tools = registry.by_kind(AffordanceKind.TOOL)
    assert [d.name for d in tools] == ["t1", "t2"]
    actions = registry.by_kind(AffordanceKind.ACTION)
    assert [d.name for d in actions] == ["a1"]
    organs = registry.by_kind(AffordanceKind.ORGAN)
    assert organs == ()


def test_by_tag_filters_correctly() -> None:
    registry = AffordanceRegistry()
    registry.register_all(
        [
            AffordanceDescriptor(
                name="read_file",
                kind=AffordanceKind.TOOL,
                version="0.1.0",
                display_name="Read",
                description="read",
                when_to_use=_HINT,
                when_not_to_use=_HINT + " neg",
                parameters_schema={"type": "object"},
                output_schema={"type": "object"},
                cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
                safety_model=AffordanceSafety(),
                affordance_tags=("read", "code"),
            ),
            AffordanceDescriptor(
                name="grep",
                kind=AffordanceKind.TOOL,
                version="0.1.0",
                display_name="Grep",
                description="grep",
                when_to_use=_HINT,
                when_not_to_use=_HINT + " neg",
                parameters_schema={"type": "object"},
                output_schema={"type": "object"},
                cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
                safety_model=AffordanceSafety(),
                affordance_tags=("search", "code"),
            ),
        ]
    )
    code = registry.by_tag("code")
    assert [d.name for d in code] == ["read_file", "grep"]
    read = registry.by_tag("read")
    assert [d.name for d in read] == ["read_file"]


def test_seal_blocks_further_registration() -> None:
    registry = AffordanceRegistry()
    registry.register(_descriptor("alpha"))
    assert registry.sealed is False
    registry.seal()
    assert registry.sealed is True
    with pytest.raises(AffordanceRegistrySealedError):
        registry.register(_descriptor("bravo"))
    with pytest.raises(AffordanceRegistrySealedError):
        registry.register_all([_descriptor("charlie")])
    # Reads still work post-seal.
    assert registry.get("alpha").name == "alpha"


def test_lint_warnings_empty_in_slice_1() -> None:
    registry = AffordanceRegistry()
    registry.register(_descriptor("alpha"))
    # Slice 1 declares no soft lints (post_init raises handle hard
    # invariants). The accessor exists for slice 2 to grow into.
    assert registry.lint_warnings() == ()
