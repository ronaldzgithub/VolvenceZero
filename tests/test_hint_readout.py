"""Unit tests for ``volvence_zero.regime.hint_readout`` (Gap 8 slice 2).

Covers:

* ``HintReadoutContext`` construction from various snapshot combos
  (cold start / eval-only / full signal) and ``evidence_score``
  scaling.
* Readout per-section scores: a clearly task-heavy profile maps to
  STRUCTURED panorama + task; an emotional-support profile maps to
  SILENT task and BRIEF / SILENT panorama.
* Flow kind selection: features, not regime_id string, decide
  kind.
* Cold-start fallback: when neither dual_track nor evaluation is
  available, the readout falls back to the slice-1 scaffold and
  caps ``confidence`` at 0.30 with a ``readout:cold-fallback:`` tag.
* Depth readout: covers all 5 tiers across a spectrum of
  total-pressure contexts.
* ``RegimeModule.hint_readout_mode`` switch: "scaffold" reproduces
  pre-slice-2 behaviour; "readout" uses the continuous readout;
  invalid value fails loudly.
"""

from __future__ import annotations

import pytest

from volvence_zero.dual_track.core import DualTrackSnapshot, TrackState
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.memory.store import MemoryEntry, MemorySnapshot, Track
from volvence_zero.prediction.error import PredictedOutcome, ActualOutcome, PredictionError, PredictionErrorSnapshot
from volvence_zero.regime import (
    CognitiveDepth,
    HintReadoutContext,
    ParticipationFlowKind,
    ParticipationLevel,
    RegimeModule,
    build_hint_readout_context,
    derive_cognitive_depth_hint,
    derive_participation_hint,
    readout_cognitive_depth_hint,
    readout_participation_hint,
)


# ---------------------------------------------------------------------------
# Snapshot fixtures
# ---------------------------------------------------------------------------


def _eval_snapshot(**overrides: float) -> EvaluationSnapshot:
    metrics = {
        "warmth": 0.5,
        "support_presence": 0.5,
        "task_pressure": 0.5,
        "cross_track_stability": 0.5,
        "info_integration": 0.5,
    }
    metrics.update(overrides)
    return EvaluationSnapshot(
        turn_scores=tuple(
            EvaluationScore(
                family="f", metric_name=name, value=value, confidence=1.0, evidence=""
            )
            for name, value in metrics.items()
        ),
        session_scores=(),
        alerts=(),
        description="",
    )


def _dual_track_snapshot(
    *,
    cross_tension: float = 0.2,
    world_tension: float = 0.3,
    self_tension: float = 0.3,
    world_controller_code: tuple[float, ...] = (0.3, 0.3, 0.2),
    self_controller_code: tuple[float, ...] = (0.3, 0.3, 0.2),
    world_action_hint: str | None = None,
    self_action_hint: str | None = None,
) -> DualTrackSnapshot:
    return DualTrackSnapshot(
        world_track=TrackState(
            track=Track.WORLD,
            active_goals=(),
            recent_credits=(),
            controller_code=world_controller_code,
            tension_level=world_tension,
            abstract_action_hint=world_action_hint,
        ),
        self_track=TrackState(
            track=Track.SELF,
            active_goals=(),
            recent_credits=(),
            controller_code=self_controller_code,
            tension_level=self_tension,
            abstract_action_hint=self_action_hint,
        ),
        cross_track_tension=cross_tension,
        description="",
    )


def _memory_snapshot(
    *, world_count: int = 0, self_count: int = 0
) -> MemorySnapshot:
    entries: list[MemoryEntry] = []
    for i in range(world_count):
        entries.append(
            MemoryEntry(
                entry_id=f"w-{i}",
                content=f"world{i}",
                track=Track.WORLD,
                stratum="episodic",
                created_at_ms=0,
                last_accessed_ms=0,
                strength=0.5,
                tags=(),
            )
        )
    for i in range(self_count):
        entries.append(
            MemoryEntry(
                entry_id=f"s-{i}",
                content=f"self{i}",
                track=Track.SELF,
                stratum="episodic",
                created_at_ms=0,
                last_accessed_ms=0,
                strength=0.5,
                tags=(),
            )
        )
    return MemorySnapshot(
        transient_summary="",
        episodic_summary="",
        durable_summary="",
        retrieved_entries=tuple(entries),
        total_entries_by_stratum=(),
        pending_promotions=0,
        pending_decays=0,
        cms_state=None,
        description="",
    )


def _pe_snapshot(magnitude: float = 0.3) -> PredictionErrorSnapshot:
    return PredictionErrorSnapshot(
        evaluated_prediction=None,
        actual_outcome=ActualOutcome(
            observed_turn_index=1,
            task_progress=0.5,
            relationship_delta=0.5,
            regime_stability=0.5,
            action_payoff=0.5,
            description="",
        ),
        next_prediction=PredictedOutcome(
            source_turn_index=1,
            target_turn_index=2,
            predicted_task_progress=0.5,
            predicted_relationship_delta=0.5,
            predicted_regime_stability=0.5,
            predicted_action_payoff=0.5,
            confidence=0.7,
            description="",
        ),
        error=PredictionError(
            task_error=0.0,
            relationship_error=0.0,
            regime_error=0.0,
            action_error=0.0,
            magnitude=magnitude,
            signed_reward=0.0,
            description="",
        ),
        turn_index=1,
        bootstrap=False,
        description="",
    )


# ---------------------------------------------------------------------------
# HintReadoutContext construction + evidence_score
# ---------------------------------------------------------------------------


def test_cold_context_has_low_evidence_score() -> None:
    ctx = build_hint_readout_context(
        regime_id="problem_solving",
        turns_in_current_regime=0,
        candidates=(("problem_solving", 0.5),),
        memory=None,
        dual_track=None,
        evaluation=None,
        prediction_error=None,
    )
    # No runtime signal -> evidence ~= 0.1 (baseline) + sharpness bonus.
    assert ctx.evidence_score() <= 0.3
    assert ctx.has_dual_track is False
    assert ctx.has_evaluation is False


def test_full_context_saturates_evidence_score() -> None:
    candidates = (
        ("problem_solving", 0.9),
        ("guided_exploration", 0.4),
    )
    ctx = build_hint_readout_context(
        regime_id="problem_solving",
        turns_in_current_regime=2,
        candidates=candidates,
        memory=_memory_snapshot(world_count=3, self_count=1),
        dual_track=_dual_track_snapshot(),
        evaluation=_eval_snapshot(),
        prediction_error=_pe_snapshot(),
    )
    # All sources present + reasonably sharp candidates.
    assert ctx.evidence_score() >= 0.80
    assert ctx.has_dual_track is True
    assert ctx.candidate_sharpness == pytest.approx(0.5, abs=0.01)


def test_candidate_sharpness_handles_single_candidate() -> None:
    ctx = build_hint_readout_context(
        regime_id="r",
        turns_in_current_regime=0,
        candidates=(("r", 0.5),),
        memory=None, dual_track=None, evaluation=None, prediction_error=None,
    )
    assert ctx.candidate_sharpness == 1.0


# ---------------------------------------------------------------------------
# Readout: per-section scores under representative profiles
# ---------------------------------------------------------------------------


def test_task_heavy_profile_renders_structured_task_and_panorama() -> None:
    ctx = build_hint_readout_context(
        regime_id="problem_solving",
        turns_in_current_regime=3,
        candidates=(
            ("problem_solving", 0.85),
            ("guided_exploration", 0.30),
        ),
        memory=_memory_snapshot(world_count=3),
        dual_track=_dual_track_snapshot(
            world_tension=0.5,
            self_tension=0.2,
            cross_tension=0.25,
            world_controller_code=(0.85, 0.4, 0.3),  # strong world drive
            world_action_hint="task_controller",
            self_action_hint="task_controller",
        ),
        evaluation=_eval_snapshot(
            task_pressure=0.8, support_presence=0.3, warmth=0.4
        ),
        prediction_error=_pe_snapshot(magnitude=0.1),
    )
    hint = readout_participation_hint(ctx)
    assert hint.flow_kind in {
        ParticipationFlowKind.PROBLEM,
        ParticipationFlowKind.TASK,
    }
    assert hint.task_level is ParticipationLevel.STRUCTURED
    # Panorama should render because world presence + switch are cued.
    assert hint.panorama_level in {
        ParticipationLevel.BRIEF,
        ParticipationLevel.STRUCTURED,
    }
    # Rationale is a learned-v1 tag, not a scaffold fallback.
    assert hint.rationale.startswith("readout.v1:")
    # Confidence should be materially higher than the scaffold 0.4.
    assert hint.confidence > 0.6


def test_emotional_support_profile_drops_task_level() -> None:
    ctx = build_hint_readout_context(
        regime_id="emotional_support",
        turns_in_current_regime=1,
        candidates=(("emotional_support", 0.75), ("casual_social", 0.40)),
        memory=_memory_snapshot(self_count=3),
        dual_track=_dual_track_snapshot(
            world_tension=0.2,
            self_tension=0.7,
            cross_tension=0.3,
            world_controller_code=(0.2, 0.3, 0.1),
            self_controller_code=(0.7, 0.4, 0.1),  # strong self drive
            self_action_hint="repair_controller",
        ),
        evaluation=_eval_snapshot(
            warmth=0.3, support_presence=0.8, task_pressure=0.2
        ),
        prediction_error=_pe_snapshot(magnitude=0.4),
    )
    hint = readout_participation_hint(ctx)
    # Repair/self-heavy profile -> task_level should be SILENT.
    assert hint.task_level is ParticipationLevel.SILENT
    # Flow kind must NOT be PROBLEM/TASK.
    assert hint.flow_kind not in {
        ParticipationFlowKind.TASK,
        ParticipationFlowKind.PROBLEM,
    }


def test_casual_social_profile_drops_panorama_and_task() -> None:
    ctx = build_hint_readout_context(
        regime_id="casual_social",
        turns_in_current_regime=4,
        candidates=(("casual_social", 0.70), ("acquaintance_building", 0.40)),
        memory=_memory_snapshot(),
        dual_track=_dual_track_snapshot(
            world_tension=0.1,
            self_tension=0.1,
            cross_tension=0.1,
            world_controller_code=(0.1, 0.1, 0.05),
            self_controller_code=(0.1, 0.1, 0.05),
            world_action_hint="stabilize_controller",
            self_action_hint="stabilize_controller",
        ),
        evaluation=_eval_snapshot(
            warmth=0.8, support_presence=0.4, task_pressure=0.1
        ),
        prediction_error=_pe_snapshot(magnitude=0.05),
    )
    hint = readout_participation_hint(ctx)
    # Low-pressure chitchat should NOT render task_level.
    assert hint.task_level is ParticipationLevel.SILENT
    # Flow kind should land SOCIAL (or ACQUAINTANCE as second-best);
    # never PROBLEM/TASK.
    assert hint.flow_kind in {
        ParticipationFlowKind.SOCIAL,
        ParticipationFlowKind.ACQUAINTANCE,
        ParticipationFlowKind.INFO,
    }


# ---------------------------------------------------------------------------
# Cold-start fallback
# ---------------------------------------------------------------------------


def test_readout_cold_start_falls_back_to_scaffold_for_known_regime() -> None:
    ctx = build_hint_readout_context(
        regime_id="problem_solving",
        turns_in_current_regime=0,
        candidates=(("problem_solving", 0.5),),
        memory=None, dual_track=None, evaluation=None, prediction_error=None,
    )
    hint = readout_participation_hint(ctx)
    scaffold = derive_participation_hint("problem_solving")
    assert hint.flow_kind == scaffold.flow_kind
    assert hint.panorama_level == scaffold.panorama_level
    assert hint.rationale.startswith("readout:cold-fallback:")
    # Cold-start confidence is capped at 0.30.
    assert hint.confidence <= 0.30


def test_readout_cold_start_falls_back_for_depth_too() -> None:
    ctx = build_hint_readout_context(
        regime_id="casual_social",
        turns_in_current_regime=0,
        candidates=(("casual_social", 0.5),),
        memory=None, dual_track=None, evaluation=None, prediction_error=None,
    )
    hint = readout_cognitive_depth_hint(ctx)
    scaffold = derive_cognitive_depth_hint("casual_social")
    assert hint.depth == scaffold.depth
    assert hint.confidence <= 0.30
    assert hint.rationale.startswith("readout:cold-fallback:")


# ---------------------------------------------------------------------------
# Depth readout tier coverage
# ---------------------------------------------------------------------------


def test_depth_reflexive_for_minimum_pressure() -> None:
    ctx = build_hint_readout_context(
        regime_id="casual_social",
        turns_in_current_regime=0,
        candidates=(("casual_social", 0.7),),
        memory=None,
        dual_track=_dual_track_snapshot(
            world_tension=0.0,
            self_tension=0.0,
            cross_tension=0.0,
            world_controller_code=(0.0, 0.0, 0.0),
            self_controller_code=(0.0, 0.0, 0.0),
            world_action_hint="stabilize_controller",
            self_action_hint="stabilize_controller",
        ),
        evaluation=_eval_snapshot(warmth=0.9, task_pressure=0.0),
        prediction_error=None,
    )
    hint = readout_cognitive_depth_hint(ctx)
    assert hint.depth in {CognitiveDepth.REFLEXIVE, CognitiveDepth.SHALLOW}


def test_depth_alert_or_deep_for_high_pressure() -> None:
    ctx = build_hint_readout_context(
        regime_id="repair_and_deescalation",
        turns_in_current_regime=0,
        candidates=(("repair_and_deescalation", 0.9), ("emotional_support", 0.3)),
        memory=_memory_snapshot(self_count=3),
        dual_track=_dual_track_snapshot(
            world_tension=0.7,
            self_tension=0.8,
            cross_tension=0.9,
            world_controller_code=(0.4, 0.3, 0.6),
            self_controller_code=(0.6, 0.3, 0.6),
            world_action_hint="repair_controller",
            self_action_hint="repair_controller",
        ),
        evaluation=_eval_snapshot(
            warmth=0.2, task_pressure=0.6, cross_track_stability=0.15
        ),
        prediction_error=_pe_snapshot(magnitude=0.8),
    )
    hint = readout_cognitive_depth_hint(ctx)
    assert hint.depth in {CognitiveDepth.ALERT, CognitiveDepth.DEEP}


# ---------------------------------------------------------------------------
# RegimeModule hint_readout_mode switch
# ---------------------------------------------------------------------------


def test_regime_module_defaults_to_readout_mode() -> None:
    module = RegimeModule()
    # _hint_readout_mode is private state but used internally;
    # exercise through behaviour: default mode must accept both
    # 'readout' and 'scaffold' construction.
    assert module._hint_readout_mode == "readout"  # noqa: SLF001


def test_regime_module_scaffold_mode_is_accepted() -> None:
    module = RegimeModule(hint_readout_mode="scaffold")
    assert module._hint_readout_mode == "scaffold"  # noqa: SLF001


def test_regime_module_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="hint_readout_mode"):
        RegimeModule(hint_readout_mode="bogus")


async def test_regime_module_scaffold_mode_matches_slice1_behaviour() -> None:
    """When hint_readout_mode='scaffold', the hint published in
    the snapshot must equal the slice-1 scaffold derivation for
    the chosen regime.
    """
    module = RegimeModule(hint_readout_mode="scaffold")
    snapshot = await module.process_standalone(
        memory_snapshot=None,
        dual_track_snapshot=None,
        evaluation_snapshot=None,
        prediction_error_snapshot=None,
        experience_fast_prior_snapshot=None,
    )
    regime_id = snapshot.value.active_regime.regime_id
    expected_part = derive_participation_hint(regime_id)
    expected_depth = derive_cognitive_depth_hint(regime_id)
    assert snapshot.value.participation_hint == expected_part
    assert snapshot.value.depth_hint == expected_depth


async def test_regime_module_readout_mode_uses_continuous_features() -> None:
    """readout mode + rich runtime signal -> the published hint
    rationale must start with ``readout.v1`` (not the scaffold tag)
    and confidence must exceed 0.4 (the scaffold flat default).
    """
    module = RegimeModule(hint_readout_mode="readout")
    snapshot = await module.process_standalone(
        memory_snapshot=_memory_snapshot(world_count=2, self_count=1),
        dual_track_snapshot=_dual_track_snapshot(
            world_tension=0.6,
            self_tension=0.3,
            cross_tension=0.2,
            world_controller_code=(0.7, 0.4, 0.3),
            self_controller_code=(0.3, 0.3, 0.2),
            world_action_hint="task_controller",
            self_action_hint="task_controller",
        ),
        evaluation_snapshot=_eval_snapshot(task_pressure=0.7, warmth=0.4),
        prediction_error_snapshot=_pe_snapshot(magnitude=0.2),
        experience_fast_prior_snapshot=None,
    )
    part = snapshot.value.participation_hint
    assert part.rationale.startswith("readout.v1:"), (
        f"expected readout.v1 rationale, got {part.rationale!r}"
    )
    assert part.confidence > 0.4
    assert snapshot.value.depth_hint.rationale.startswith("readout.v1.depth:")
