"""Phase 1 W1.F matched-control gate for common_ground.

Same pattern as W1.A / W1.D:

* SHADOW config: snapshot lives in ``shadow_snapshots`` and the
  lifeform-side provider sees ``None``;
* ACTIVE config: snapshot lives in ``active_snapshots``; the typed
  payload is byte-for-byte identical to the SHADOW publication when
  the upstream input is identical;
* Drift between SHADOW and ACTIVE on unrelated slots must not exceed
  the run-to-run baseline plus the documented diagnostic surface
  (``response_assembly`` already counts ToM owner records and may
  surface a ``common_ground`` dyad count once the diagnostic catches
  up; for now we accept the same documented set as W1.D).

A second, planner-level test pins the typed rationale tag emission:
non-empty dyad atoms -> ``common_ground=observed(...)`` rationale tag
+ ``CONTINUITY_NOTE`` section addition. ``None`` and empty-atom
snapshots are no-ops.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from lifeform_expression.prompt_planner import PromptPlanner, SectionId
from volvence_zero.agent.response import ResponseContext
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    CommonGroundAtom,
    CommonGroundSnapshot,
    SocialScopeKind,
)
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


def _substrate() -> FeatureSurfaceSubstrateAdapter:
    return FeatureSurfaceSubstrateAdapter(
        model_id="common-ground-matched-control-model",
        feature_surface=(
            FeatureSignal(
                name="common_ground_matched_control_context",
                values=(0.5,),
                source="adapter",
            ),
        ),
    )


def _run_turn_with_wiring(level: WiringLevel) -> dict[str, dict[str, Snapshot[Any]]]:
    config = FinalRolloutConfig(common_ground=level)
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=_substrate(),
            user_input="hello there",
            session_id="common-ground-matched-control",
            wave_id="wave-1",
            turn_index=1,
        )
    )
    return {"active": result.active_snapshots, "shadow": result.shadow_snapshots}


def _drifted_slot_names(
    a: dict[str, Snapshot[Any]], b: dict[str, Snapshot[Any]]
) -> set[str]:
    shared = set(a) & set(b)
    return {name for name in shared if a[name].value != b[name].value}


_DOCUMENTED_COMMON_GROUND_DEPENDENT_SLOTS: frozenset[str] = frozenset(
    {
        # ResponseAssemblyModule reads common_ground via
        # _common_ground_snapshot_counts (existing diagnostic surface
        # in application/runtime.py). Promotion changes the published
        # count.
        "response_assembly",
        # Evaluation reads response_assembly downstream so its snapshot
        # legitimately drifts when the diagnostic count changes.
        "evaluation",
    }
)


def test_shadow_publishes_to_shadow_only_active_publishes_to_active_only() -> None:
    shadow_run = _run_turn_with_wiring(WiringLevel.SHADOW)
    active_run = _run_turn_with_wiring(WiringLevel.ACTIVE)
    assert "common_ground" in shadow_run["shadow"]
    assert "common_ground" not in shadow_run["active"]
    assert "common_ground" in active_run["active"]
    assert "common_ground" not in active_run["shadow"]


def test_active_promotion_does_not_change_published_state() -> None:
    shadow_value = _run_turn_with_wiring(WiringLevel.SHADOW)["shadow"][
        "common_ground"
    ].value
    active_value = _run_turn_with_wiring(WiringLevel.ACTIVE)["active"][
        "common_ground"
    ].value
    assert isinstance(shadow_value, CommonGroundSnapshot)
    assert isinstance(active_value, CommonGroundSnapshot)
    assert dataclasses.asdict(shadow_value) == dataclasses.asdict(active_value)


def test_active_promotion_drift_is_bounded_to_documented_dependents() -> None:
    baseline_a = _run_turn_with_wiring(WiringLevel.SHADOW)
    baseline_b = _run_turn_with_wiring(WiringLevel.SHADOW)
    baseline_drift = _drifted_slot_names(baseline_a["active"], baseline_b["active"])

    shadow_run = _run_turn_with_wiring(WiringLevel.SHADOW)
    active_run = _run_turn_with_wiring(WiringLevel.ACTIVE)
    promotion_drift = _drifted_slot_names(shadow_run["active"], active_run["active"])

    new_drift = (
        promotion_drift
        - baseline_drift
        - _DOCUMENTED_COMMON_GROUND_DEPENDENT_SLOTS
    )
    assert not new_drift, (
        "Promoting common_ground SHADOW->ACTIVE introduced new value "
        f"drift on undocumented slots: {sorted(new_drift)}. Either a "
        "kernel module silently took common_ground as a dependency, or "
        "the diagnostic-dependent slot list is stale. Both require a "
        "contract review."
    )


# ---------------------------------------------------------------------------
# Planner-level rationale-tag matched control
# ---------------------------------------------------------------------------


def _planner_context() -> ResponseContext:
    return ResponseContext(
        regime_id="acquaintance_building",
        regime_name="acquaintance_building",
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="idle",
        user_input="",
    )


def _make_dyad_atom(*, atom_id: str, confidence: float = 0.80) -> CommonGroundAtom:
    return CommonGroundAtom(
        atom_id=atom_id,
        scope_id="self+primary",
        scope_kind=SocialScopeKind.DYAD,
        summary="we discussed weekend plans",
        recursion_depth=1,
        confidence=confidence,
        accepted_by_ids=("self", "primary"),
        evidence=("weekend",),
    )


def test_planner_no_op_when_common_ground_snapshot_is_none() -> None:
    planner = PromptPlanner()
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        common_ground_snapshot=None,
    )
    assert not any(
        t.startswith("common_ground=") for t in plan.rationale_tags
    )


def test_planner_no_op_when_common_ground_dyad_atoms_empty() -> None:
    planner = PromptPlanner()
    snapshot = CommonGroundSnapshot(
        dyad_atoms=(),
        group_atoms=(),
        active_predictions=(),
        control_signal=0.0,
        description="empty",
    )
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        common_ground_snapshot=snapshot,
    )
    assert not any(
        t.startswith("common_ground=") for t in plan.rationale_tags
    )


def test_planner_emits_observed_tag_and_adds_continuity_note_for_dyad_atom() -> None:
    planner = PromptPlanner()
    snapshot = CommonGroundSnapshot(
        dyad_atoms=(
            _make_dyad_atom(atom_id="a1"),
            _make_dyad_atom(atom_id="a2", confidence=0.65),
        ),
        group_atoms=(),
        active_predictions=(),
        control_signal=0.72,
        description="non-empty",
    )
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        common_ground_snapshot=snapshot,
    )
    cg_tags = [t for t in plan.rationale_tags if t.startswith("common_ground")]
    assert any(t.startswith("common_ground=observed(dyads=2,") for t in cg_tags)
    # R14 (regime is not a prompt label): with no assembly snapshot the
    # planner falls back to a neutral ``TurnIntent.DIRECT_ANSWER`` rather
    # than mapping ``regime_id == "acquaintance_building"`` to
    # ``WARMTH_FIRST``. ``DIRECT_ANSWER`` does not include CONTINUITY_NOTE
    # in its default sections, so the common_ground modulation correctly
    # fires the ``common_ground_add=continuity_note`` rationale here.
    assert any(t.startswith("common_ground_add=continuity_note") for t in cg_tags)


def test_planner_adds_continuity_note_when_not_in_default_sections() -> None:
    planner = PromptPlanner()
    snapshot = CommonGroundSnapshot(
        dyad_atoms=(_make_dyad_atom(atom_id="a1"),),
        group_atoms=(),
        active_predictions=(),
        control_signal=0.7,
        description="non-empty",
    )
    no_continuity_context = dataclasses.replace(
        _planner_context(), regime_id="problem_solving", regime_name="problem_solving"
    )
    baseline = planner.plan(
        context=no_continuity_context,
        assembly=None,
    )
    assert SectionId.CONTINUITY_NOTE not in baseline.sections, (
        f"test setup: baseline must not include CONTINUITY_NOTE; got {baseline.sections!r}"
    )
    plan = planner.plan(
        context=no_continuity_context,
        assembly=None,
        common_ground_snapshot=snapshot,
    )
    assert SectionId.CONTINUITY_NOTE in plan.sections
    assert any(
        t.startswith("common_ground_add=continuity_note") for t in plan.rationale_tags
    )


def test_planner_filters_atoms_below_typed_confidence_floor() -> None:
    """Low-confidence atoms (below 0.50) must not trigger modulation."""
    planner = PromptPlanner()
    snapshot = CommonGroundSnapshot(
        dyad_atoms=(_make_dyad_atom(atom_id="a-low", confidence=0.30),),
        group_atoms=(),
        active_predictions=(),
        control_signal=0.30,
        description="low confidence",
    )
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        common_ground_snapshot=snapshot,
    )
    assert not any(
        t.startswith("common_ground=") for t in plan.rationale_tags
    )
