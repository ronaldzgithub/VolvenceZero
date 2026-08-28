"""Directed tests for the Packet 3.6 episode-outcome gate machinery."""

from __future__ import annotations

import pathlib

import pytest

from lifeform_evolution.coding_lab_packet36 import (
    ARM_ALWAYS_ON,
    ARM_NOOP,
    ARM_RANDOM_GATE,
    ARM_TABLE_GATE,
    CertifiedCell,
    Packet36Config,
    Packet36EpisodeRow,
    arm_steers,
    derive_certified_cells,
    paired_gap_statistics,
)


def _cell(*, gain: float, category: str = "fix_bug") -> CertifiedCell:
    return CertifiedCell(
        state_key=f"{category}|reads=1|edited=0|tests=none",
        category=category,
        expert_action="edit",
        expert_itt_pass_rate=0.8 + gain,
        natural_control_pass_rate=0.8,
    )


def _config(**overrides: object) -> Packet36Config:
    defaults: dict[str, object] = {
        "run_id": "test",
        "output_root": pathlib.Path("artifacts/coding_lab"),
        "certified_cells": (_cell(gain=0.2),),
        "chains": 4,
        "episodes_per_chain": 4,
    }
    defaults.update(overrides)
    return Packet36Config(**defaults)


class TestArmPolicies:
    def test_noop_never_steers(self) -> None:
        config = _config()
        assert not arm_steers(config, arm=ARM_NOOP, cell=_cell(gain=0.2), chain_index=0, episode_index=0)

    def test_always_on_steers_at_opportunity_only(self) -> None:
        config = _config()
        assert arm_steers(config, arm=ARM_ALWAYS_ON, cell=_cell(gain=0.0), chain_index=0, episode_index=0)
        assert not arm_steers(config, arm=ARM_ALWAYS_ON, cell=None, chain_index=0, episode_index=0)

    def test_table_gate_thresholds_on_credited_gain(self) -> None:
        config = _config(table_gate_min_gain=0.05)
        assert arm_steers(config, arm=ARM_TABLE_GATE, cell=_cell(gain=0.2), chain_index=0, episode_index=0)
        assert not arm_steers(config, arm=ARM_TABLE_GATE, cell=_cell(gain=0.0), chain_index=0, episode_index=0)

    def test_random_gate_is_seed_deterministic(self) -> None:
        config = _config()
        draws = [
            arm_steers(config, arm=ARM_RANDOM_GATE, cell=_cell(gain=0.2), chain_index=c, episode_index=e)
            for c in range(4)
            for e in range(4)
        ]
        repeat = [
            arm_steers(config, arm=ARM_RANDOM_GATE, cell=_cell(gain=0.2), chain_index=c, episode_index=e)
            for c in range(4)
            for e in range(4)
        ]
        assert draws == repeat
        assert any(draws) and not all(draws)

    def test_unknown_arm_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="unknown arm"):
            arm_steers(_config(), arm="bogus", cell=_cell(gain=0.2), chain_index=0, episode_index=0)


class TestDeriveCertifiedCells:
    def _calibration(self) -> dict:
        return {
            "interventional_expert_actions": {
                "fix_bug|reads=1|edited=0|tests=none": "edit",
            },
            "interventional_table": {
                "fix_bug|reads=1|edited=0|tests=none": [
                    {"assigned_action": "edit", "trials": 17, "passes": 17, "pass_rate": 1.0},
                    {"assigned_action": "submit", "trials": 23, "passes": 0, "pass_rate": 0.0},
                ],
            },
            "observational_control_table": {
                "fix_bug|reads=1|edited=0|tests=none": [
                    {"action": "investigate", "trials": 16, "passes": 13, "pass_rate": 0.8125},
                    {"action": "test", "trials": 1, "passes": 1, "pass_rate": 1.0},
                ],
            },
        }

    def test_derives_gain_from_control_weighted_mean(self) -> None:
        cells = derive_certified_cells(self._calibration())
        assert len(cells) == 1
        cell = cells[0]
        assert cell.expert_action == "edit"
        assert cell.category == "fix_bug"
        assert cell.expert_itt_pass_rate == 1.0
        assert abs(cell.natural_control_pass_rate - 14 / 17) < 1e-9
        assert cell.credited_gain > 0.15

    def test_key_without_control_coverage_is_dropped(self) -> None:
        calibration = self._calibration()
        calibration["observational_control_table"] = {}
        with pytest.raises(ValueError, match="no certified steering cells"):
            derive_certified_cells(calibration)


def _row(arm: str, chain_index: int, passed: bool) -> Packet36EpisodeRow:
    return Packet36EpisodeRow(
        arm=arm,
        chain_index=chain_index,
        episode_index=0,
        task_id="t",
        category="fix_bug",
        passed=passed,
        opportunity=True,
        steer_decided=arm != ARM_NOOP,
        triggered=arm != ARM_NOOP,
        expert_action="edit" if arm != ARM_NOOP else None,
        submitted=True,
        steps_used=3,
        wall_seconds=1.0,
        prompt_tokens=10,
        completion_tokens=5,
        trajectory_sha256="0" * 64,
    )


class TestPairedGapStatistics:
    def test_positive_gap_and_deterministic_bootstrap(self) -> None:
        rows: list[Packet36EpisodeRow] = []
        for chain in range(6):
            rows.append(_row(ARM_TABLE_GATE, chain, passed=True))
            rows.append(_row(ARM_NOOP, chain, passed=chain % 2 == 0))
        first = paired_gap_statistics(
            rows, arm_a=ARM_TABLE_GATE, arm_b=ARM_NOOP, resamples=500, seed=7
        )
        second = paired_gap_statistics(
            rows, arm_a=ARM_TABLE_GATE, arm_b=ARM_NOOP, resamples=500, seed=7
        )
        assert first == second
        assert first["mean_gap"] == pytest.approx(0.5)
        assert first["bootstrap_ci_lower_5pct"] <= first["mean_gap"]

    def test_missing_arm_rows_fail_loudly(self) -> None:
        rows = [_row(ARM_TABLE_GATE, 0, passed=True)]
        with pytest.raises(ValueError, match="missing rows"):
            paired_gap_statistics(rows, arm_a=ARM_TABLE_GATE, arm_b=ARM_NOOP, resamples=10, seed=1)
