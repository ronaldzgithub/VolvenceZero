"""Workstream G2/G3/G4 demonstration tests (fast, small parameters)."""

from __future__ import annotations

import json
from pathlib import Path

from volvence_ant.viz.bio_overlay import build_bio_overlays
from volvence_ant.viz.colony_theater import run_colony_theater
from volvence_ant.viz.dashboard import write_replay_dashboard
from volvence_ant.viz.perturbation import run_perturbation_demo
from volvence_ant.viz.safety_demo import run_safety_demo
from volvence_ant.env import AntWorld, AntWorldConfig
from volvence_ant.runtime import AntSession, AntSessionConfig


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


async def test_colony_theater_renders_both_arms_from_immutable_frames(
    tmp_path: Path,
) -> None:
    out = tmp_path / "theater.html"
    report = await run_colony_theater(
        n_ants=3, rounds=8, relocate_at=4, seed=0, out_path=out
    )
    # two arms: heuristic (hardcoded FSM) + digital-life (kernel controller)
    kinds = {arm.kind for arm in report.arms}
    assert kinds == {"heuristic", "digital-life"}
    # every frame carries one pose per body plus the shared trail heatmap grid
    for arm in report.arms:
        assert len(arm.frames) == 8
        first = arm.frames[0]
        assert len(first.ants) == 3
        assert len(first.trail) > 0 and len(first.trail[0]) > 0
    # the food relocation is visible: sources differ before vs after the move
    heuristic = next(arm for arm in report.arms if arm.kind == "heuristic")
    assert heuristic.frames[3].food != heuristic.frames[7].food
    # self-contained HTML embeds the data and the side-by-side canvas
    html = out.read_text(encoding="utf-8")
    assert "数字蚂蚁剧场" in html
    assert "__THEATER_DATA__" not in html  # placeholder was substituted
    assert "digital-life" in html


async def test_dashboard_is_generated_from_immutable_step_records(
    tmp_path: Path,
) -> None:
    session = AntSession(
        AntWorld(config=AntWorldConfig(seed=0)),
        config=AntSessionConfig(seed=0),
    )
    records = await session.run(2)
    path = write_replay_dashboard(
        tracks={"learned": records},
        out_path=tmp_path / "dashboard.html",
    )
    html = path.read_text(encoding="utf-8")
    assert "snapshot-driven live evidence" in html
    assert '"pe_magnitude"' in html
    assert '"backend_wiring"' in html
