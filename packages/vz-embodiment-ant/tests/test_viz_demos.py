"""Workstream G2/G3/G4 demonstration tests (fast, small parameters)."""

from __future__ import annotations

import json
from pathlib import Path

from volvence_ant.viz.bio_overlay import build_bio_overlays
from volvence_ant.viz.perturbation import run_perturbation_demo
from volvence_ant.viz.safety_demo import run_safety_demo


def test_g2_hardcoded_collapses_and_adaptive_recovers() -> None:
    report = run_perturbation_demo(n_ants=10, rounds=600, relocate_at=300, seed=0)
    # while assumptions hold, the hardcoded beeline is competent
    assert report.hardcoded.delivered_before > 0
    # after the unforeseen relocation it delivers nothing
    assert report.hardcoded_collapsed
    assert report.hardcoded.delivered_after == 0
    # the sensing/adaptive arm keeps finding food after the move
    assert report.emergent_recovered
    assert report.emergent.delivered_after > 0


def test_g3_overlay_reads_phase0_artifacts(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "phase0_homing.json").write_text(
        json.dumps(
            {
                "antbot_reference_ratio": 0.02,
                "passes_antbot_scale": True,
                "curve": [
                    {"journey_length": 4.0, "mean_normalized_error": 0.007},
                    {"journey_length": 8.0, "mean_normalized_error": 0.004},
                ],
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "phase0_route_learning.json").write_text(
        json.dumps(
            {
                "familiarity_improved": True,
                "first_exposure_novelty": 0.012,
                "last_exposure_novelty": 0.002,
                "novelty_by_exposure": [0.012, 0.009, 0.005, 0.003, 0.002],
            }
        ),
        encoding="utf-8",
    )
    report = build_bio_overlays(results_dir=results_dir, figures_dir=tmp_path / "figures")
    assert report.passes_antbot_scale
    assert report.familiarity_improved
    assert report.last_exposure_novelty < report.first_exposure_novelty


def test_g4_safety_reflex_is_one_vote_veto() -> None:
    report = run_safety_demo(n_ticks=150, alarm_probability=0.4, seed=1)
    assert report.n_alarmed > 0
    # every alarmed tick is the reflex, regardless of chaotic z_t
    assert report.all_alarmed_are_reflex
    # the reflex command is identical across all alarmed ticks (deterministic)
    assert report.reflex_deterministic
    assert report.reflex_ignores_code
    # the reflex is a straight full-speed flee, not a learned turn
    assert report.reflex_turn == 0.0
    assert report.reflex_step > 0.0
    # and the chaotic controller really could produce non-trivial turns when calm
    assert report.max_calm_turn_magnitude > 0.0
