"""Packet C (long-horizon-closure) — z_t-driven AffordanceModule selection.

End-to-end proof that ``AffordanceModule.process()`` selection
**actually swings on z_t** rather than on hard-coded descriptor names.
The test feeds two opposing z_t vectors through a ``ControllerState``
inside a real ``TemporalAbstractionSnapshot`` and asserts that the
top-scoring affordance switches.

Why this matters:

The affordance spec's ``selection-is-learned-not-hardcoded`` gate
demands that selection moves with metacontroller latent state. A
test that just calls ``score_affordance_candidates`` directly
proves the function works, but it doesn't prove the module wires
the public temporal snapshot's controller_state.code into the
projection. This test does — by going through ``AffordanceModule.process``
with real Snapshot wrappers.

This test does NOT exercise the live ``BrainSession``; that adds
substrate / regime / reflection / etc. dependencies that obscure
the z_t -> selection signal. The unit test pattern here is the
right scope.
"""

from __future__ import annotations

import asyncio
from typing import Any

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
from volvence_zero.runtime import Snapshot
from volvence_zero.runtime.kernel import utc_now_ms
from volvence_zero.temporal import (
    ControllerState,
    TemporalAbstractionSnapshot,
)


_HINT = (
    "Use only inside the metacontroller scoring e2e test to verify "
    "that affordance selection swings with z_t as required by the "
    "selection-is-learned-not-hardcoded acceptance gate."
)


def _descriptor(name: str) -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=AffordanceKind.TOOL,
        version="0.1.0",
        display_name=name.title(),
        description=f"Test descriptor {name} for the metacontroller scoring proof.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Not for any other test.",
        parameters_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        output_schema={"type": "object"},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.INSTANT),
        safety_model=AffordanceSafety(),
    )


def _wrap_temporal_snapshot(z_t: tuple[float, ...]) -> Snapshot[TemporalAbstractionSnapshot]:
    """Build a real ``TemporalAbstractionSnapshot`` -> ``Snapshot``
    wrapper carrying a controller_state with the supplied ``z_t``.
    """
    controller = ControllerState(
        code=z_t,
        code_dim=len(z_t),
        switch_gate=0.5,
        is_switching=False,
        steps_since_switch=1,
    )
    value = TemporalAbstractionSnapshot(
        controller_state=controller,
        active_abstract_action="test-action",
        controller_params_hash="test-hash",
        description=f"test temporal snapshot z_t={z_t!r}",
    )
    return Snapshot(
        slot_name="temporal_abstraction",
        owner="TestTemporalOwner",
        version=1,
        timestamp_ms=utc_now_ms(),
        value=value,
    )


def _find_z_t_that_picks(
    *,
    target_name: str,
    candidate_names: tuple[str, ...],
    other_required: tuple[str, ...] = (),
    z_dim: int = 4,
    max_attempts: int = 4096,
) -> tuple[float, ...]:
    """Search a small space of integer-tweaked z_t vectors until we
    find one where ``target_name`` is the strict top scorer (with the
    AffordanceModule's selection threshold satisfied).

    This avoids over-specifying a magic vector that could break if
    we ever change the projection / threshold; the test only needs
    SOME z_t to pick each target, not a specific one.
    """
    from itertools import product

    from lifeform_affordance import score_affordance_candidates

    # Try a simple grid of {-1, -0.5, 0, 0.5, 1} entries.
    grid = (-1.0, -0.5, 0.0, 0.5, 1.0)
    names = tuple(set(candidate_names) | set(other_required))
    attempts = 0
    for combo in product(grid, repeat=z_dim):
        attempts += 1
        if attempts > max_attempts:
            break
        scores = dict(score_affordance_candidates(
            descriptor_names=names,
            z_t=combo,
        ))
        target = scores.get(target_name, -1.0)
        others = [v for k, v in scores.items() if k != target_name]
        if not others:
            continue
        runner_up = max(others)
        # Match the AffordanceModule._SELECTION_MIN_SCORE / _MARGIN
        # exactly so the wrapper-level test will produce a non-None
        # selected and that selected.descriptor_name == target_name.
        if target >= 0.55 and (target - runner_up) >= 0.05:
            return combo
    raise RuntimeError(
        f"No suitable z_t found in grid for target={target_name!r} after "
        f"{attempts} attempts; widen the grid or change target."
    )


def _process(module: AffordanceModule, upstream: dict[str, Snapshot[Any]]) -> AffordanceSnapshot:
    snap = asyncio.run(module.process(upstream))
    assert isinstance(snap, Snapshot)
    return snap.value


def test_affordance_module_selection_switches_with_z_t() -> None:
    """The big one: same registry, different z_t -> different
    ``AffordanceSnapshot.selected.descriptor_name``. This is the
    direct demonstration that selection is z_t-driven, not
    hardcoded.
    """
    registry = AffordanceRegistry()
    for name in ("read_file", "write_file", "grep"):
        registry.register(_descriptor(name))
    module = AffordanceModule(registry=registry)

    z_for_read = _find_z_t_that_picks(
        target_name="read_file",
        candidate_names=("read_file", "write_file", "grep"),
    )
    z_for_grep = _find_z_t_that_picks(
        target_name="grep",
        candidate_names=("read_file", "write_file", "grep"),
    )

    snap_read = _process(
        module,
        {"temporal_abstraction": _wrap_temporal_snapshot(z_for_read)},
    )
    snap_grep = _process(
        module,
        {"temporal_abstraction": _wrap_temporal_snapshot(z_for_grep)},
    )

    assert snap_read.selected is not None
    assert snap_grep.selected is not None
    assert snap_read.selected.descriptor_name == "read_file", (
        f"Expected read_file selected for z_t={z_for_read!r}, "
        f"got {snap_read.selected.descriptor_name!r}."
    )
    assert snap_grep.selected.descriptor_name == "grep", (
        f"Expected grep selected for z_t={z_for_grep!r}, "
        f"got {snap_grep.selected.descriptor_name!r}."
    )
    # Sanity: the two chosen z_t vectors really are different.
    assert z_for_read != z_for_grep


def test_affordance_module_no_temporal_snapshot_keeps_all_neutral() -> None:
    """If no ``temporal_abstraction`` upstream is available (e.g.
    a turn before temporal has emitted anything, or temporal is
    DISABLED), AffordanceModule must keep every candidate at the
    neutral score and produce ``selected=None``.
    """
    registry = AffordanceRegistry()
    for name in ("read_file", "write_file"):
        registry.register(_descriptor(name))
    module = AffordanceModule(registry=registry)
    snap = _process(module, {})
    assert snap.selected is None
    for c in snap.candidates_for_turn:
        assert c.score == 0.5


def test_affordance_module_rationale_contains_z_t_signal_marker() -> None:
    """Each candidate's rationale must include a marker indicating
    the scoring source (``z_t_projection`` / ``neutral_cold_start`` /
    ``metacontroller``). Operators rely on this to debug why a
    selection happened.
    """
    registry = AffordanceRegistry()
    registry.register(_descriptor("read_file"))
    module = AffordanceModule(registry=registry)
    cold = _process(module, {})
    assert any(
        "neutral_cold_start" in c.rationale for c in cold.candidates_for_turn
    )

    z_t = (0.1, 0.2, 0.3, -0.1)
    warm = _process(
        module,
        {"temporal_abstraction": _wrap_temporal_snapshot(z_t)},
    )
    assert any(
        "z_t_projection" in c.rationale for c in warm.candidates_for_turn
    )


def test_affordance_module_independent_invocations_publish_different_versions() -> None:
    """Two consecutive ``process(...)`` calls must publish distinct
    snapshot versions (the kernel's version-tracking RuntimeModule
    base bumps it). This guards against accidentally caching a
    stale snapshot across turns.
    """
    registry = AffordanceRegistry()
    registry.register(_descriptor("read_file"))
    module = AffordanceModule(registry=registry)
    first = asyncio.run(module.process({}))
    second = asyncio.run(module.process({}))
    assert second.version > first.version
