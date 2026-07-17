"""Packet C (long-horizon-closure) — AffordanceModule contract.

Three structural checks for the new ``AffordanceModule``:

1. Satisfies the ``RuntimeModule`` ABC (slot_name, owner, value_type,
   default_wiring_level, dependencies, async process()).
2. Default ``WiringLevel`` is ``SHADOW`` per the affordance spec —
   promotion to ACTIVE is intentionally a future-packet decision.
3. ``process(...)`` produces a valid, snapshot-shape-conforming
   ``AffordanceSnapshot`` even for the cold-start (no upstream)
   case.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceModule,
    AffordanceRegistry,
    AffordanceSafety,
    AffordanceSnapshot,
)
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


_HINT = (
    "Use only inside the AffordanceModule contract test to verify the "
    "module's RuntimeModule shape and snapshot publication."
)


def _descriptor(name: str, *, tag: str = "test") -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=AffordanceKind.TOOL,
        version="0.1.0",
        display_name=name.title(),
        description=f"Test affordance {name}.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Not for production use.",
        parameters_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        output_schema={"type": "object"},
        cost_model=AffordanceCost(
            latency_class=AffordanceLatencyClass.INSTANT,
        ),
        safety_model=AffordanceSafety(),
        affordance_tags=(tag,),
    )


def _registry_with_descriptors(*names: str) -> AffordanceRegistry:
    registry = AffordanceRegistry()
    for n in names:
        registry.register(_descriptor(n))
    return registry


def test_affordance_module_satisfies_runtime_module_contract() -> None:
    registry = _registry_with_descriptors("read_file", "write_file")
    module = AffordanceModule(registry=registry)
    # RuntimeModule sub-instance + all required class attrs present.
    assert isinstance(module, RuntimeModule)
    assert module.slot_name == "affordance"
    assert module.owner == "AffordanceModule"
    assert module.value_type is AffordanceSnapshot
    # CP-04 added the typed consent gate: boundary_consent joined the
    # declared dependency set alongside the metacontroller z_t source.
    assert module.dependencies == ("temporal_abstraction", "boundary_consent")
    # Default wiring level is ACTIVE post long-horizon-closure: the
    # module publishes its z_t-driven snapshot so downstream
    # consumers can pick it up without an explicit opt-in.
    assert module.default_wiring_level is WiringLevel.ACTIVE
    assert module.wiring_level is WiringLevel.ACTIVE


def test_affordance_module_explicit_shadow_wiring_level_overrides_default() -> None:
    """The class default is ACTIVE but callers can opt to SHADOW
    (or DISABLED) at construction time for benchmark ablations
    without subclassing.
    """
    registry = _registry_with_descriptors("read_file")
    shadow_module = AffordanceModule(
        registry=registry, wiring_level=WiringLevel.SHADOW
    )
    assert shadow_module.wiring_level is WiringLevel.SHADOW
    disabled_module = AffordanceModule(
        registry=registry, wiring_level=WiringLevel.DISABLED
    )
    assert disabled_module.wiring_level is WiringLevel.DISABLED


def test_affordance_module_cold_start_publishes_neutral_snapshot() -> None:
    """No upstream snapshots = empty z_t = neutral cold start. The
    module must still produce a valid AffordanceSnapshot (no crash,
    no negative scores).
    """
    registry = _registry_with_descriptors("read_file", "write_file", "grep")
    module = AffordanceModule(registry=registry)

    upstream: Mapping[str, Snapshot[Any]] = {}
    snapshot = asyncio.run(module.process(upstream))

    assert isinstance(snapshot, Snapshot)
    assert snapshot.slot_name == "affordance"
    value = snapshot.value
    assert isinstance(value, AffordanceSnapshot)
    # Cold start -> all neutral 0.5 -> no candidate clears the
    # 0.55 selection threshold; selected MUST be None.
    assert value.selected is None, (
        "Cold start (empty z_t) must not produce a selected affordance. "
        "Selection thresholds are gates against accidental confidence."
    )
    candidate_scores = {c.descriptor_name: c.score for c in value.candidates_for_turn}
    assert candidate_scores == {
        "read_file": 0.5,
        "write_file": 0.5,
        "grep": 0.5,
    }


def test_affordance_module_blocked_descriptor_not_selected() -> None:
    """A descriptor with ``excluded_from_runtime_selection=True``
    (or a regime-blocked descriptor when regime upstream is provided)
    must not surface as selected. The cold-start snapshot also must
    skip them from ``available``.
    """
    registry = AffordanceRegistry()
    registry.register(_descriptor("read_file"))
    excluded_descriptor = AffordanceDescriptor(
        name="excluded_tool",
        kind=AffordanceKind.TOOL,
        version="0.1.0",
        display_name="Excluded",
        description="Test excluded affordance.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Not for production use.",
        parameters_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        output_schema={"type": "object"},
        cost_model=AffordanceCost(
            latency_class=AffordanceLatencyClass.INSTANT,
        ),
        safety_model=AffordanceSafety(),
        excluded_from_runtime_selection=True,
    )
    registry.register(excluded_descriptor)

    module = AffordanceModule(registry=registry)
    snapshot = asyncio.run(module.process({}))
    available_names = {d.name for d in snapshot.value.available}
    candidate_names = {c.descriptor_name for c in snapshot.value.candidates_for_turn}
    # Excluded descriptors must not appear at all.
    assert "excluded_tool" not in available_names
    assert "excluded_tool" not in candidate_names
    assert "read_file" in available_names


def test_score_affordance_candidates_pure_function_invariants() -> None:
    """Pure function smoke: scores must be in [0, 1] for any z_t,
    and identical (z_t, name) inputs must yield identical scores.
    """
    from lifeform_affordance import score_affordance_candidates

    for z_t in [
        (),
        (0.0,),
        (1.0, -1.0, 0.5),
        (10.0, -10.0, 5.0, -5.0),  # large magnitudes -> no overflow
        (1e-9, 1e-9, 1e-9),  # near-zero magnitudes
    ]:
        scores = score_affordance_candidates(
            descriptor_names=("read_file", "write_file"),
            z_t=z_t,
        )
        for name, value in scores:
            assert 0.0 <= value <= 1.0, (
                f"score for {name!r} at z_t={z_t!r} is out of [0,1]: {value!r}"
            )


@pytest.mark.parametrize(
    "z_t_a, z_t_b",
    [
        ((0.5, -0.3, 0.4), (-0.5, 0.3, -0.4)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ],
)
def test_score_affordance_candidates_z_t_actually_drives_output(
    z_t_a: tuple[float, ...], z_t_b: tuple[float, ...]
) -> None:
    """Different z_t -> different (multi-name) score distributions.
    If this fails the projection is broken; z_t is supposed to be
    the actual driver of selection.
    """
    from lifeform_affordance import score_affordance_candidates

    a = dict(score_affordance_candidates(
        descriptor_names=("read_file", "write_file", "grep"),
        z_t=z_t_a,
    ))
    b = dict(score_affordance_candidates(
        descriptor_names=("read_file", "write_file", "grep"),
        z_t=z_t_b,
    ))
    assert a != b, (
        f"Distinct z_t ({z_t_a!r} vs {z_t_b!r}) produced identical "
        f"score distributions {a!r}; the projection is not z_t-driven."
    )
