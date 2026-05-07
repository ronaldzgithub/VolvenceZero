"""Phase 1 W1.D matched-control gate for feeling_about_other.

Two invariants under test:

1. **Snapshot-level matched-control**: same upstream input under
   ``feeling_about_other=SHADOW`` and ``feeling_about_other=ACTIVE``
   produces the SAME ``FeelingAboutOtherSnapshot`` payload. Promotion
   shifts the slot from ``shadow_snapshots`` to ``active_snapshots``
   without changing the snapshot's records / control_signal /
   description. SHADOW-vs-ACTIVE drift introduced by the wiring flip
   alone must not exceed the SHADOW-vs-SHADOW baseline drift caused by
   run-to-run non-determinism in unrelated owners.

2. **Planner-level rationale-tag observation**: when the kernel
   produces records (because an LLM-backed proposal runtime is
   wired) the lifeform-side ``GroundedResponseSynthesizer`` reads the
   ACTIVE snapshot via the per-session ``feeling_about_other_provider``
   and emits a typed ``feeling=observed(...)`` rationale tag. SHADOW
   wiring keeps the snapshot out of ``active_snapshots`` so the
   provider returns ``None`` and the rationale tag is absent.

Both halves of the matched-control gate use the same
``FeelingAboutOtherSnapshot`` schema; no user text is read.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from lifeform_expression.prompt_planner import PromptPlanner
from lifeform_expression.response_synthesizer import GroundedResponseSynthesizer

from volvence_zero.agent.response import ResponseContext
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state.llm_runtime import LLMSemanticProposalRuntime
from volvence_zero.social_cognition import (
    FeelingAboutOtherSnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    SocialPrediction,
)
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


_FEELING_PAYLOAD = (
    "["
    '{"target_slot": "feeling_about_other", '
    '"summary": "user feels overwhelmed", '
    '"detail": "user mentioned long day and tired", '
    '"evidence": "long day", '
    '"confidence": 0.78, '
    '"control_signal": 0.55}'
    "]"
)


class _FakeFeelingProvider:
    def generate(
        self, *, prompt: str, max_new_tokens: int = 16, temperature: float = 0.0
    ) -> str:
        del prompt, max_new_tokens, temperature
        return _FEELING_PAYLOAD


def _substrate() -> FeatureSurfaceSubstrateAdapter:
    return FeatureSurfaceSubstrateAdapter(
        model_id="feeling-matched-control-model",
        feature_surface=(
            FeatureSignal(
                name="feeling_matched_control_context",
                values=(0.5,),
                source="adapter",
            ),
        ),
    )


def _run_turn_with_wiring(level: WiringLevel) -> dict[str, dict[str, Snapshot[Any]]]:
    config = FinalRolloutConfig(feeling_about_other=level)
    semantic_runtime = LLMSemanticProposalRuntime(provider=_FakeFeelingProvider())
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=_substrate(),
            user_input="i had a long day and feel really tired",
            semantic_proposal_runtime=semantic_runtime,
            session_id="feeling-matched-control",
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


def test_shadow_publishes_to_shadow_only_active_publishes_to_active_only() -> None:
    shadow_run = _run_turn_with_wiring(WiringLevel.SHADOW)
    active_run = _run_turn_with_wiring(WiringLevel.ACTIVE)
    assert "feeling_about_other" in shadow_run["shadow"]
    assert "feeling_about_other" not in shadow_run["active"]
    assert "feeling_about_other" in active_run["active"]
    assert "feeling_about_other" not in active_run["shadow"]


def test_active_promotion_does_not_change_published_state() -> None:
    """Same upstream + same proposal runtime -> identical payload."""
    shadow_value = _run_turn_with_wiring(WiringLevel.SHADOW)["shadow"][
        "feeling_about_other"
    ].value
    active_value = _run_turn_with_wiring(WiringLevel.ACTIVE)["active"][
        "feeling_about_other"
    ].value
    assert isinstance(shadow_value, FeelingAboutOtherSnapshot)
    assert isinstance(active_value, FeelingAboutOtherSnapshot)
    assert dataclasses.asdict(shadow_value) == dataclasses.asdict(active_value)
    assert len(active_value.records) == 1
    assert active_value.records[0].kind is OtherMindRecordKind.FEELING


_DOCUMENTED_FEELING_DEPENDENT_SLOTS: frozenset[str] = frozenset(
    {
        # ResponseAssemblyModule declares feeling_about_other as a
        # dependency and surfaces a diagnostic count via
        # semantic_record_counts (see
        # docs/specs/social_cognition/02_theory_of_mind.md slice 5).
        # Promotion changes that count from 0 to N records, which is
        # the intentional diagnostic surface, not a hidden second
        # owner.
        "response_assembly",
        # EvaluationModule reads response_assembly downstream so its
        # snapshot legitimately drifts when the ToM count changes.
        "evaluation",
    }
)


def test_active_promotion_drift_is_bounded_to_documented_dependents() -> None:
    """Drift between SHADOW and ACTIVE must not extend beyond the
    documented ``response_assembly`` / ``evaluation`` diagnostic
    surface plus the run-to-run non-determinism baseline.

    If a NEW slot starts drifting after this wave lands, it means a
    kernel module silently took ``feeling_about_other`` as a
    dependency without updating ``_DOCUMENTED_FEELING_DEPENDENT_SLOTS``
    or the ToM-spec slice list. That requires a contract review.
    """
    baseline_a = _run_turn_with_wiring(WiringLevel.SHADOW)
    baseline_b = _run_turn_with_wiring(WiringLevel.SHADOW)
    baseline_drift = _drifted_slot_names(baseline_a["active"], baseline_b["active"])

    shadow_run = _run_turn_with_wiring(WiringLevel.SHADOW)
    active_run = _run_turn_with_wiring(WiringLevel.ACTIVE)
    promotion_drift = _drifted_slot_names(shadow_run["active"], active_run["active"])

    new_drift = promotion_drift - baseline_drift - _DOCUMENTED_FEELING_DEPENDENT_SLOTS
    assert not new_drift, (
        "Promoting feeling_about_other SHADOW->ACTIVE introduced new "
        f"value drift on undocumented slots: {sorted(new_drift)}. "
        "Either a kernel module silently took feeling_about_other as "
        "a dependency, or the diagnostic-dependent slot list is "
        "stale. Both require a contract review."
    )


# ---------------------------------------------------------------------------
# Planner-level rationale-tag matched control
# ---------------------------------------------------------------------------


def _build_feeling_snapshot(
    *, count: int = 1, max_confidence: float = 0.78, control_signal: float = 0.55
) -> FeelingAboutOtherSnapshot:
    records = tuple(
        OtherMindRecord(
            record_id=f"feeling-{index}",
            interlocutor_id="primary",
            kind=OtherMindRecordKind.FEELING,
            summary=f"feeling-summary-{index}",
            detail="feeling-detail",
            confidence=max_confidence,
            status=OtherMindRecordStatus.ACTIVE,
            source_turn=1,
            prediction_error_refs=(),
            evidence="feeling-evidence",
        )
        for index in range(count)
    )
    return FeelingAboutOtherSnapshot(
        records=records,
        active_predictions=(),
        control_signal=control_signal,
        description="test feeling snapshot",
    )


def _planner_context() -> ResponseContext:
    return ResponseContext(
        regime_id="emotional_support",
        regime_name="emotional_support",
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


def test_planner_emits_feeling_observed_tag_when_records_present() -> None:
    """ACTIVE-equivalent: provider returns a non-empty snapshot -> typed tag fires."""
    planner = PromptPlanner()
    snapshot = _build_feeling_snapshot()
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        feeling_snapshot=snapshot,
    )
    feeling_tags = [t for t in plan.rationale_tags if t.startswith("feeling=")]
    assert len(feeling_tags) == 1, (
        f"Expected exactly one feeling=observed tag; got {feeling_tags!r}"
    )
    assert feeling_tags[0].startswith("feeling=observed(count=1,")


def test_planner_no_op_when_feeling_snapshot_is_none() -> None:
    """SHADOW-equivalent: provider returns None -> rationale tag absent."""
    planner = PromptPlanner()
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        feeling_snapshot=None,
    )
    assert not any(
        t.startswith("feeling=") for t in plan.rationale_tags
    ), plan.rationale_tags


def test_planner_no_op_when_records_below_min_confidence() -> None:
    """Records-but-low-confidence is a no-op so noisy LLMs cannot push the planner."""
    planner = PromptPlanner()
    snapshot = _build_feeling_snapshot(max_confidence=0.30)
    plan = planner.plan(
        context=_planner_context(),
        assembly=None,
        feeling_snapshot=snapshot,
    )
    assert not any(
        t.startswith("feeling=") for t in plan.rationale_tags
    ), plan.rationale_tags


def _no_ack_context() -> ResponseContext:
    """Use a regime whose default sections do NOT include
    ACKNOWLEDGE_PRESSURE, so we can assert that high feeling
    control_signal ADDs the section.
    """
    return dataclasses.replace(_planner_context(), regime_id="casual_social", regime_name="casual_social")


def test_planner_adds_acknowledge_pressure_on_high_control_signal() -> None:
    """Typed control_signal threshold ADDs ACKNOWLEDGE_PRESSURE without
    requiring the 12-axis InterlocutorState to fire. Tested in a
    regime that does NOT carry ACKNOWLEDGE_PRESSURE in its default
    section list (``casual_social``); ``emotional_support`` /
    ``repair_and_deescalation`` already include the section so the
    feeling-driven add would be a no-op.
    """
    from lifeform_expression.prompt_planner import SectionId

    planner = PromptPlanner()
    snapshot = _build_feeling_snapshot(control_signal=0.55)
    baseline = planner.plan(
        context=_no_ack_context(),
        assembly=None,
    )
    assert SectionId.ACKNOWLEDGE_PRESSURE not in baseline.sections, (
        "test setup: baseline regime must not include ACKNOWLEDGE_PRESSURE "
        f"(got {baseline.sections!r})"
    )
    plan = planner.plan(
        context=_no_ack_context(),
        assembly=None,
        feeling_snapshot=snapshot,
    )
    assert SectionId.ACKNOWLEDGE_PRESSURE in plan.sections
    assert any(
        t.startswith("feeling_add=acknowledge_pressure(") for t in plan.rationale_tags
    ), plan.rationale_tags


def test_planner_no_section_change_when_control_signal_is_low() -> None:
    """Typed control_signal below threshold leaves sections unchanged."""

    planner = PromptPlanner()
    baseline = planner.plan(
        context=_no_ack_context(),
        assembly=None,
    )
    snapshot = _build_feeling_snapshot(control_signal=0.20)
    plan = planner.plan(
        context=_no_ack_context(),
        assembly=None,
        feeling_snapshot=snapshot,
    )
    assert plan.sections == baseline.sections
    feeling_tags = [t for t in plan.rationale_tags if t.startswith("feeling=")]
    assert len(feeling_tags) == 1
    assert "feeling_add=acknowledge_pressure" not in plan.rationale_tags
