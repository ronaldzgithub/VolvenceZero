"""Wave T2 e2e — ExperientialReplayDriver runs an arc end-to-end.

Pins three load-bearing properties of the replay path:

1. Arc / character integrity check fires loudly when the wrong
   lifeform is used.
2. Running the demo arc against a 张无忌 lifeform produces non-zero
   PE on at least one scene — the "really lived through" floor.
3. Cross-scene accumulation moves at least one observable surface
   (regime sequence_payoff growth, drive drift, or total PE > 0).

These are disjunction-style pins: we don't lock to a specific surface
(future kernel evolution can change which surface advances) but the
driver must produce SOMETHING measurable, otherwise it's not driving
the kernel correctly.
"""

from __future__ import annotations

import pytest

from lifeform_domain_character import (
    ExperientialReplayDriver,
    build_zhang_wuji_demo_arc,
    build_zhang_wuji_lifeform,
    build_zhang_wuji_profile,
)
from lifeform_domain_character.lifeform_builder import build_character_lifeform


def test_replay_driver_run_arc_returns_report() -> None:
    bundle = build_zhang_wuji_lifeform()
    arc = build_zhang_wuji_demo_arc()
    driver = ExperientialReplayDriver()
    report = driver.run_arc(arc=arc, lifeform=bundle.lifeform)
    assert report.arc_id == arc.arc_id
    assert report.character_id == arc.character_id
    assert report.scenes_processed == len(arc.scenes)
    assert len(report.per_scene) == len(arc.scenes)


def test_replay_driver_produces_pe_on_at_least_one_scene() -> None:
    """The 'really lived through' floor: with synthetic substrate the
    PE owner still produces non-zero magnitude on at least one decision
    point. If this fails, the kernel is not being exercised.
    """
    bundle = build_zhang_wuji_lifeform()
    arc = build_zhang_wuji_demo_arc()
    driver = ExperientialReplayDriver()
    report = driver.run_arc(arc=arc, lifeform=bundle.lifeform)
    pe_per_scene = [scene.pe_magnitude for scene in report.per_scene]
    assert any(pe > 0.0 for pe in pe_per_scene), (
        f"every scene reported pe_magnitude=0; the kernel never "
        f"experienced any prediction error during replay. "
        f"per-scene={pe_per_scene}"
    )


def test_replay_driver_produces_some_cumulative_evidence() -> None:
    """Disjunction: at least one of (total_pe > 0, regime payoff
    growth > 0, max drive drift > 0.05) must hold.
    """
    bundle = build_zhang_wuji_lifeform()
    arc = build_zhang_wuji_demo_arc()
    driver = ExperientialReplayDriver()
    report = driver.run_arc(arc=arc, lifeform=bundle.lifeform)
    surfaces = {
        "total_pe_signal": report.total_pe_signal > 0.0,
        "regime_sequence_payoff_growth": (
            report.regime_sequence_payoff_growth > 0
        ),
        "drive_drift_above_0.05": (
            max((abs(delta) for _, delta in report.drive_drift), default=0.0)
            > 0.05
        ),
    }
    advanced = [name for name, moved in surfaces.items() if moved]
    assert advanced, (
        f"replay produced no measurable cross-scene evidence. "
        f"total_pe={report.total_pe_signal} "
        f"regime_growth={report.regime_sequence_payoff_growth} "
        f"drive_drift={report.drive_drift!r}"
    )


def test_replay_driver_per_scene_record_carries_canonical_action() -> None:
    bundle = build_zhang_wuji_lifeform()
    arc = build_zhang_wuji_demo_arc()
    driver = ExperientialReplayDriver()
    report = driver.run_arc(arc=arc, lifeform=bundle.lifeform)
    for record, scene in zip(report.per_scene, arc.scenes, strict=True):
        assert record.scene_id == scene.scene_id
        assert record.phase_label == scene.phase_label
        assert record.canonical_action == scene.canonical_action
        # outcome_kind is the typed enum's string value, picked from
        # the closed register-to-outcome mapping. We assert it is
        # one of the documented values.
        assert record.outcome_kind in {
            "helped",
            "felt_heard",
            "missed",
            "over_directive",
            "decision_clearer",
            "come_back",
            "unsafe",
            "abandoned",
        }


def test_replay_driver_rejects_wrong_character() -> None:
    """Integrity check: arc.character_id must match the lifeform's
    profile id. Running 张无忌 arc against a fictional 'test character'
    lifeform must fail loudly rather than silently absorb the wrong
    life into the wrong brain.
    """
    from lifeform_domain_character import (
        CharacterBoundaryPrior,
        CharacterDrivePrior,
        CharacterKnowledgeSeed,
        CharacterSignatureCase,
        CharacterSoulProfile,
        CharacterStrategyPrior,
    )

    other_profile = CharacterSoulProfile(
        profile_id="other-character",
        character_name="Other",
        source_title="Test Source",
        version="0.0.1",
        reviewed_by="test",
        source_uri="profile:other:test",
        description="A non-zhang-wuji test character.",
        knowledge_seeds=(
            CharacterKnowledgeSeed(
                seed_id="other-seed",
                domain="placeholder",
                title="placeholder title",
                summary="placeholder summary",
                snippet="placeholder snippet",
                evidence_locator="test:other",
                confidence=0.7,
            ),
        ),
        signature_cases=(
            CharacterSignatureCase(
                case_id="other-case",
                domain="placeholder",
                problem_pattern="placeholder",
                user_state_pattern="placeholder",
                risk_markers=("risk-low",),
                track_tags=("self",),
                regime_tags=("guided_exploration",),
                intervention_ordering=("step_one",),
                outcome_label="stable",
                description="placeholder",
                confidence=0.7,
            ),
        ),
        strategy_priors=(
            CharacterStrategyPrior(
                rule_id="other-rule",
                problem_pattern="placeholder",
                recommended_regime="guided_exploration",
                recommended_ordering=("step_one",),
                recommended_pacing="standard",
                avoid_patterns=("rushed",),
                applicability_scope=("risk-low",),
                confidence=0.7,
                description="placeholder",
            ),
        ),
        boundary_priors=(
            CharacterBoundaryPrior(
                boundary_id="other-boundary",
                regime_id=None,
                trigger_reasons=("test-trigger",),
                answer_depth_limit_hint="standard",
                clarification_required=False,
                refer_out_required=False,
                blocked_topics=(),
                required_disclaimers=(),
                confidence=0.7,
                description="placeholder",
            ),
        ),
        drive_priors=(
            CharacterDrivePrior(
                name="placeholder_drive",
                target=0.6,
                homeostatic_band=(0.4, 0.8),
                decay_per_tick=0.01,
                pe_weight=0.5,
                initial_level=0.5,
                recharge_per_turn=0.0,
            ),
        ),
    )
    bundle = build_character_lifeform(other_profile)
    arc = build_zhang_wuji_demo_arc()
    driver = ExperientialReplayDriver()
    with pytest.raises(ValueError, match="character_id"):
        driver.run_arc(arc=arc, lifeform=bundle.lifeform)
