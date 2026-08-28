"""Directed tests for the Packet T2 directed-pairs assignment mode."""

from __future__ import annotations

import pathlib

import pytest

from lifeform_evolution.coding_lab_interventional import (
    InterventionalConfig,
    draw_assignment,
)

_CELLS = (
    ("fix_bug|reads=1|edited=0|tests=none", "edit"),
    ("fix_bug|reads=3|edited=0|tests=none", "submit"),
    ("refactor_alias|reads=2|edited=0|tests=none", "investigate"),
)


def _config(**overrides: object) -> InterventionalConfig:
    defaults: dict[str, object] = {
        "run_id": "test",
        "output_root": pathlib.Path("artifacts/coding_lab"),
        "target_state_keys": tuple(sorted({key for key, _ in _CELLS})),
        "directed_cells": _CELLS,
        "control_weight": 0.4,
    }
    defaults.update(overrides)
    return InterventionalConfig(**defaults)


class TestDirectedConfigValidation:
    def test_action_outside_protocol_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside protocol"):
            _config(
                directed_cells=(("fix_bug|reads=1|edited=0|tests=none", "reboot"),),
                target_state_keys=("fix_bug|reads=1|edited=0|tests=none",),
            )

    def test_keys_must_cover_targets_exactly(self) -> None:
        superset = tuple(sorted({key for key, _ in _CELLS})) + ("extra|x",)
        with pytest.raises(ValueError, match="exactly cover"):
            _config(target_state_keys=superset)

    def test_duplicate_cells_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            _config(
                directed_cells=(_CELLS[0], _CELLS[0]),
                target_state_keys=(_CELLS[0][0],),
            )


class TestDirectedDraw:
    def test_assignment_is_cell_action_or_control(self) -> None:
        config = _config()
        drawn_actions: set[str | None] = set()
        for chain in range(8):
            for episode in range(10):
                key, action = draw_assignment(config, chain, episode, "fix_bug")
                assert key is not None and key.startswith("fix_bug|")
                expected = {a for k, a in _CELLS if k == key}
                assert action is None or action in expected
                drawn_actions.add(action)
        assert None in drawn_actions  # control draws happen
        assert {"edit", "submit"} <= {a for a in drawn_actions if a}

    def test_category_without_cell_gets_no_assignment(self) -> None:
        assert draw_assignment(_config(), 0, 0, "add_helper") == (None, None)

    def test_draw_is_seed_deterministic(self) -> None:
        config = _config()
        first = [draw_assignment(config, c, e, "fix_bug") for c in range(4) for e in range(8)]
        second = [draw_assignment(config, c, e, "fix_bug") for c in range(4) for e in range(8)]
        assert first == second

    def test_full_menu_mode_unchanged_without_directed_cells(self) -> None:
        config = _config(directed_cells=(), target_state_keys=("fix_bug|reads=1|edited=0|tests=none",))
        key, _action = draw_assignment(config, 0, 0, "fix_bug")
        assert key == "fix_bug|reads=1|edited=0|tests=none"
