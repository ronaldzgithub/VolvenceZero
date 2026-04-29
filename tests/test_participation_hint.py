"""Unit tests for Gap 8: ParticipationHint + CognitiveDepthHint.

Covers:

* Enum completeness (ParticipationFlowKind / ParticipationLevel /
  CognitiveDepth).
* Hint construction invariants (confidence in [0, 1]).
* Scaffold derivation ``derive_participation_hint`` /
  ``derive_cognitive_depth_hint`` for every known regime plus
  a fallback for unknown regime_ids.
* ``RegimeSnapshot`` publishes the scaffold hints at standalone
  process time \u2014 confirms the RegimeModule integration.
"""

from __future__ import annotations

import asyncio

import pytest

from volvence_zero.regime import (
    CognitiveDepth,
    CognitiveDepthHint,
    ParticipationFlowKind,
    ParticipationHint,
    ParticipationLevel,
    RegimeModule,
    RegimeSnapshot,
    derive_cognitive_depth_hint,
    derive_participation_hint,
)


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


def test_participation_flow_kind_values_are_exhaustive() -> None:
    assert set(ParticipationFlowKind) == {
        ParticipationFlowKind.SOCIAL,
        ParticipationFlowKind.ACQUAINTANCE,
        ParticipationFlowKind.INFO,
        ParticipationFlowKind.PROBLEM,
        ParticipationFlowKind.TASK,
    }


def test_participation_level_values_are_exhaustive() -> None:
    assert set(ParticipationLevel) == {
        ParticipationLevel.SILENT,
        ParticipationLevel.BRIEF,
        ParticipationLevel.STRUCTURED,
    }


def test_cognitive_depth_values_are_exhaustive() -> None:
    assert set(CognitiveDepth) == {
        CognitiveDepth.REFLEXIVE,
        CognitiveDepth.SHALLOW,
        CognitiveDepth.FOCUSED,
        CognitiveDepth.ALERT,
        CognitiveDepth.DEEP,
    }


# ---------------------------------------------------------------------------
# Hint construction invariants
# ---------------------------------------------------------------------------


def test_participation_hint_default_is_all_structured_info() -> None:
    hint = ParticipationHint()
    assert hint.flow_kind is ParticipationFlowKind.INFO
    assert hint.panorama_level is ParticipationLevel.STRUCTURED
    assert hint.method_level is ParticipationLevel.STRUCTURED
    assert hint.task_level is ParticipationLevel.STRUCTURED
    assert 0.0 <= hint.confidence <= 1.0


def test_participation_hint_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        ParticipationHint(confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        ParticipationHint(confidence=-0.1)


def test_cognitive_depth_hint_default_is_focused() -> None:
    hint = CognitiveDepthHint()
    assert hint.depth is CognitiveDepth.FOCUSED
    assert 0.0 <= hint.confidence <= 1.0


def test_cognitive_depth_hint_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        CognitiveDepthHint(confidence=2.0)


# ---------------------------------------------------------------------------
# Scaffold derivation table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("regime_id", "expected_panorama", "expected_task"),
    [
        ("casual_social", ParticipationLevel.SILENT, ParticipationLevel.SILENT),
        ("acquaintance_building", ParticipationLevel.BRIEF, ParticipationLevel.SILENT),
        ("emotional_support", ParticipationLevel.BRIEF, ParticipationLevel.SILENT),
        ("guided_exploration", ParticipationLevel.BRIEF, ParticipationLevel.BRIEF),
        (
            "problem_solving",
            ParticipationLevel.STRUCTURED,
            ParticipationLevel.STRUCTURED,
        ),
        (
            "repair_and_deescalation",
            ParticipationLevel.BRIEF,
            ParticipationLevel.SILENT,
        ),
    ],
)
def test_derive_participation_hint_scaffold_table(
    regime_id: str,
    expected_panorama: ParticipationLevel,
    expected_task: ParticipationLevel,
) -> None:
    hint = derive_participation_hint(regime_id)
    assert hint.panorama_level is expected_panorama
    assert hint.task_level is expected_task
    # Every scaffold hint carries a non-empty rationale so operators
    # can see what was applied and why.
    assert hint.rationale.startswith("scaffold:")


def test_derive_participation_hint_fallback_for_unknown_regime() -> None:
    hint = derive_participation_hint("never-heard-of-this-regime")
    # Fallback returns a safe all-STRUCTURED baseline \u2014 don't drop
    # anything when we have no policy for this regime.
    assert hint.panorama_level is ParticipationLevel.STRUCTURED
    assert hint.method_level is ParticipationLevel.STRUCTURED
    assert hint.task_level is ParticipationLevel.STRUCTURED
    assert "scaffold:fallback" in hint.rationale


@pytest.mark.parametrize(
    ("regime_id", "expected_depth"),
    [
        ("casual_social", CognitiveDepth.SHALLOW),
        ("emotional_support", CognitiveDepth.FOCUSED),
        ("problem_solving", CognitiveDepth.ALERT),
    ],
)
def test_derive_cognitive_depth_hint_scaffold_table(
    regime_id: str, expected_depth: CognitiveDepth,
) -> None:
    hint = derive_cognitive_depth_hint(regime_id)
    assert hint.depth is expected_depth


def test_derive_cognitive_depth_hint_fallback_for_unknown_regime() -> None:
    hint = derive_cognitive_depth_hint("unknown-regime-id")
    assert hint.depth is CognitiveDepth.FOCUSED
    assert "scaffold:fallback" in hint.rationale


# ---------------------------------------------------------------------------
# RegimeModule publication
# ---------------------------------------------------------------------------


def test_regime_module_standalone_publishes_participation_and_depth_hints() -> None:
    module = RegimeModule()

    snapshot = asyncio.run(
        module.process_standalone(
            memory_snapshot=None,
            dual_track_snapshot=None,
            evaluation_snapshot=None,
            prediction_error_snapshot=None,
            experience_fast_prior_snapshot=None,
        )
    )
    assert isinstance(snapshot.value, RegimeSnapshot)
    # The hints are populated (not None) and come from the scaffold
    # derivation keyed on active_regime.regime_id. We don't assert
    # a specific regime here because process_standalone picks its own
    # default, but the hint objects are always well-formed.
    assert isinstance(snapshot.value.participation_hint, ParticipationHint)
    assert isinstance(snapshot.value.depth_hint, CognitiveDepthHint)
    assert snapshot.value.participation_hint.rationale != ""


def test_regime_snapshot_default_hints_preserve_backcompat() -> None:
    """A ``RegimeSnapshot`` constructed without providing hint fields
    (e.g. in a legacy test stub) must default to the neutral
    all-STRUCTURED baseline so pre-Gap-8 callers see no behaviour
    change.
    """
    from volvence_zero.regime import RegimeIdentity

    regime = RegimeIdentity(
        regime_id="custom",
        name="custom",
        embedding=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        entry_conditions="",
        exit_conditions="",
        historical_effectiveness=0.5,
    )
    snap = RegimeSnapshot(
        active_regime=regime,
        previous_regime=None,
        switch_reason="",
        candidate_regimes=(),
        turns_in_current_regime=0,
        description="legacy stub",
    )
    assert snap.participation_hint.panorama_level is ParticipationLevel.STRUCTURED
    assert snap.depth_hint.depth is CognitiveDepth.FOCUSED
