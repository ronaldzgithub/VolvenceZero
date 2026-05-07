"""Phase 2 W2.A matched-control gate for the three remaining
about-other ToM owners (``belief_about_other`` / ``intent_about_other``
/ ``preference_about_other``).

Same pattern as W1.D feeling: each owner exposes a typed snapshot
with ``records`` / ``control_signal`` / ``description`` fields. The
lifeform-side ``GroundedResponseSynthesizer`` reads each via its own
provider closure and the planner emits an observation-only typed
rationale tag (``framing=belief_observed`` /
``intent=expectation_observed`` / ``preference=style_observed``).

Two halves under test:

1. **Snapshot-level matched-control**: SHADOW vs ACTIVE produces the
   same typed payload. Drift on unrelated slots must not exceed
   baseline + the documented ``response_assembly`` /
   ``evaluation`` diagnostic surface (which already counts ToM
   records).
2. **Planner-level rationale tags**: typed records yield typed
   observation tags; ``None`` and below-floor confidence are no-ops.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from lifeform_expression.prompt_planner import PromptPlanner
from volvence_zero.agent.response import ResponseContext
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    IntentAboutOtherSnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceAboutOtherSnapshot,
)
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


def _substrate(model_id: str) -> FeatureSurfaceSubstrateAdapter:
    return FeatureSurfaceSubstrateAdapter(
        model_id=model_id,
        feature_surface=(
            FeatureSignal(
                name=f"{model_id}_context",
                values=(0.5,),
                source="adapter",
            ),
        ),
    )


def _config_for_slot(slot_name: str, level: WiringLevel) -> FinalRolloutConfig:
    return FinalRolloutConfig(**{slot_name: level})


def _run_turn_for_slot(
    *, slot_name: str, level: WiringLevel
) -> dict[str, dict[str, Snapshot[Any]]]:
    result = asyncio.run(
        run_final_wiring_turn(
            config=_config_for_slot(slot_name, level),
            substrate_adapter=_substrate(f"tom-mc-{slot_name}"),
            user_input=f"hi for {slot_name}",
            session_id=f"tom-mc-{slot_name}",
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


_SLOTS_AND_TYPES: tuple[tuple[str, type], ...] = (
    ("belief_about_other", BeliefAboutOtherSnapshot),
    ("intent_about_other", IntentAboutOtherSnapshot),
    ("preference_about_other", PreferenceAboutOtherSnapshot),
)
_DOCUMENTED_DIAGNOSTIC_SLOTS: frozenset[str] = frozenset(
    {"response_assembly", "evaluation"}
)


def test_each_active_publishes_to_active_only() -> None:
    for slot_name, _ in _SLOTS_AND_TYPES:
        active_run = _run_turn_for_slot(slot_name=slot_name, level=WiringLevel.ACTIVE)
        shadow_run = _run_turn_for_slot(slot_name=slot_name, level=WiringLevel.SHADOW)
        assert slot_name in active_run["active"], slot_name
        assert slot_name not in active_run["shadow"], slot_name
        assert slot_name in shadow_run["shadow"], slot_name
        assert slot_name not in shadow_run["active"], slot_name


def test_each_promotion_does_not_change_published_state() -> None:
    for slot_name, expected_type in _SLOTS_AND_TYPES:
        shadow_value = _run_turn_for_slot(
            slot_name=slot_name, level=WiringLevel.SHADOW
        )["shadow"][slot_name].value
        active_value = _run_turn_for_slot(
            slot_name=slot_name, level=WiringLevel.ACTIVE
        )["active"][slot_name].value
        assert isinstance(shadow_value, expected_type), slot_name
        assert isinstance(active_value, expected_type), slot_name
        assert dataclasses.asdict(shadow_value) == dataclasses.asdict(active_value), (
            slot_name
        )


def test_each_promotion_drift_is_bounded_to_documented_dependents() -> None:
    for slot_name, _ in _SLOTS_AND_TYPES:
        baseline_a = _run_turn_for_slot(slot_name=slot_name, level=WiringLevel.SHADOW)
        baseline_b = _run_turn_for_slot(slot_name=slot_name, level=WiringLevel.SHADOW)
        baseline_drift = _drifted_slot_names(
            baseline_a["active"], baseline_b["active"]
        )

        shadow_run = _run_turn_for_slot(slot_name=slot_name, level=WiringLevel.SHADOW)
        active_run = _run_turn_for_slot(slot_name=slot_name, level=WiringLevel.ACTIVE)
        promotion_drift = _drifted_slot_names(
            shadow_run["active"], active_run["active"]
        )

        new_drift = (
            promotion_drift - baseline_drift - _DOCUMENTED_DIAGNOSTIC_SLOTS
        )
        assert not new_drift, (
            f"Promoting {slot_name} SHADOW->ACTIVE introduced new drift "
            f"on undocumented slots: {sorted(new_drift)}"
        )


# ---------------------------------------------------------------------------
# Planner-level rationale-tag matched control
# ---------------------------------------------------------------------------


def _planner_context() -> ResponseContext:
    return ResponseContext(
        regime_id="problem_solving",
        regime_name="problem_solving",
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


def _record(
    *,
    record_id: str,
    kind: OtherMindRecordKind,
    summary: str,
    confidence: float,
) -> OtherMindRecord:
    return OtherMindRecord(
        record_id=record_id,
        interlocutor_id="primary",
        kind=kind,
        summary=summary,
        detail="detail",
        confidence=confidence,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=1,
        prediction_error_refs=(),
        evidence="evidence",
    )


def _build_belief() -> BeliefAboutOtherSnapshot:
    return BeliefAboutOtherSnapshot(
        records=(
            _record(
                record_id="b1",
                kind=OtherMindRecordKind.BELIEF,
                summary="user thinks meeting is tomorrow",
                confidence=0.78,
            ),
        ),
        active_predictions=(),
        control_signal=0.4,
        description="belief",
    )


def _build_intent() -> IntentAboutOtherSnapshot:
    return IntentAboutOtherSnapshot(
        records=(
            _record(
                record_id="i1",
                kind=OtherMindRecordKind.INTENT,
                summary="user wants brief reply",
                confidence=0.72,
            ),
        ),
        active_predictions=(),
        control_signal=0.30,
        description="intent",
    )


def _build_preference() -> PreferenceAboutOtherSnapshot:
    return PreferenceAboutOtherSnapshot(
        records=(
            _record(
                record_id="p1",
                kind=OtherMindRecordKind.PREFERENCE,
                summary="user prefers terse style",
                confidence=0.85,
            ),
        ),
        active_predictions=(),
        control_signal=0.20,
        description="preference",
    )


def test_planner_no_op_when_all_three_snapshots_are_none() -> None:
    planner = PromptPlanner()
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        belief_snapshot=None,
        intent_snapshot=None,
        preference_snapshot=None,
    )
    for prefix in ("framing=belief_observed", "intent=expectation_observed", "preference=style_observed"):
        assert not any(
            tag.startswith(prefix) for tag in plan.rationale_tags
        ), prefix


def test_planner_emits_three_typed_observation_tags_when_records_present() -> None:
    planner = PromptPlanner()
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        belief_snapshot=_build_belief(),
        intent_snapshot=_build_intent(),
        preference_snapshot=_build_preference(),
    )
    assert any(
        tag.startswith("framing=belief_observed(") for tag in plan.rationale_tags
    ), plan.rationale_tags
    assert any(
        tag.startswith("intent=expectation_observed(") for tag in plan.rationale_tags
    ), plan.rationale_tags
    assert any(
        tag.startswith("preference=style_observed(") for tag in plan.rationale_tags
    ), plan.rationale_tags


def test_planner_filters_records_below_typed_confidence_floor() -> None:
    planner = PromptPlanner()
    low_belief = BeliefAboutOtherSnapshot(
        records=(
            _record(
                record_id="b-low",
                kind=OtherMindRecordKind.BELIEF,
                summary="low confidence belief",
                confidence=0.30,
            ),
        ),
        active_predictions=(),
        control_signal=0.10,
        description="low belief",
    )
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        belief_snapshot=low_belief,
        intent_snapshot=None,
        preference_snapshot=None,
    )
    assert not any(
        tag.startswith("framing=belief_observed") for tag in plan.rationale_tags
    )


def test_planner_does_not_change_section_set_for_three_tom_owners() -> None:
    """Observation-only invariant: typed BELIEF / INTENT / PREFERENCE
    records influence rationale tags but MUST NOT mutate the section
    list (FEELING + InterlocutorState remain the dominant section
    gates). This pins the W2.A scope.
    """
    planner = PromptPlanner()
    baseline = planner.plan(context=_planner_context(), assembly=None)
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        belief_snapshot=_build_belief(),
        intent_snapshot=_build_intent(),
        preference_snapshot=_build_preference(),
    )
    assert plan.sections == baseline.sections
