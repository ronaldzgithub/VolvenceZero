"""Small-budget P1 schema and fixed-gate smoke."""

from __future__ import annotations

import hashlib
import json

import pytest

from volvence_ant.experiments.ecology_p1 import (
    ECOLOGY_P1_ARM_NAMES,
    ECOLOGY_P1_GATE_NAMES,
    ECOLOGY_P1_SCHEMA_VERSION,
    EcologyP1Config,
    EcologyP1ProgressPaused,
    run_ecology_p1,
    run_ecology_p1_diagnostics,
)


async def test_p1_bounded_work_budget_pauses_on_committed_episode(
    tmp_path,
) -> None:
    config = EcologyP1Config(
        n_ants=1,
        temporal_latent_dim=4,
        training_rounds=1,
        evaluation_rounds=1,
        layouts_per_tier=1,
        seed=17,
    )
    progress_dir = tmp_path / "bounded"

    with pytest.raises(EcologyP1ProgressPaused) as first:
        await run_ecology_p1(
            config,
            progress_dir=progress_dir,
            max_new_work_items=1,
        )
    assert first.value.completed_work_items == 1
    state = json.loads(
        (progress_dir / "learned.json").read_text(encoding="utf-8")
    )
    assert state["completed_training_episodes"] == 1
    assert state["training_complete"] is False
    assert not (progress_dir / "no_optimize.json").exists()

    with pytest.raises(EcologyP1ProgressPaused):
        await run_ecology_p1(
            config,
            progress_dir=progress_dir,
            max_new_work_items=1,
        )
    state = json.loads(
        (progress_dir / "learned.json").read_text(encoding="utf-8")
    )
    assert state["completed_training_episodes"] == 2


def test_p1_diagnostics_are_checkpoint_free_and_structured() -> None:
    report = run_ecology_p1_diagnostics(
        EcologyP1Config(
            n_ants=1,
            temporal_latent_dim=4,
            training_rounds=1,
            evaluation_rounds=3,
            layouts_per_tier=1,
            seed=7,
        )
    )

    assert report.schema_version == "digital-ant-ecology-p1-diagnostics.v2"
    assert len(report.results) == 18
    assert len(report.oracle_success_by_capability) == 6


async def test_p1_uses_fixed_schedule_per_body_mastery(
    tmp_path,
) -> None:
    config = EcologyP1Config(
        n_ants=1,
        temporal_latent_dim=4,
        training_rounds=1,
        evaluation_rounds=3,
        layouts_per_tier=1,
        seed=7,
    )
    progress_dir = tmp_path / "progress"
    report = await run_ecology_p1(
        config,
        progress_dir=progress_dir,
    )

    assert report.schema_version == ECOLOGY_P1_SCHEMA_VERSION
    assert report.verdict in {"PASS", "BLOCK"}
    assert tuple(gate.name for gate in report.gates) == ECOLOGY_P1_GATE_NAMES
    assert {item.arm for item in report.layout_results} == set(
        ECOLOGY_P1_ARM_NAMES
    )
    assert len(report.layout_results) == len(ECOLOGY_P1_ARM_NAMES) * 6
    assert len(report.diagnostic_results) == 3 * 6
    assert {item.controller for item in report.diagnostic_results} == {
        "oracle_steering",
        "fixed_rule",
        "random",
    }
    assert all(item.required_bodies == 1 for item in report.layout_results)
    assert all(item.policy_fingerprint_stable for item in report.layout_results)
    assert report.schedule[:3] and all(
        item.tier.value == "near" for item in report.schedule[:3]
    )
    assert sum(item.forced_return for item in report.schedule) == 1
    for arm in ECOLOGY_P1_ARM_NAMES:
        state = json.loads(
            (progress_dir / f"{arm}.json").read_text(
                encoding="utf-8"
            )
        )
        assert state["training_complete"] is True
        assert state["completed_training_episodes"] == (
            0 if arm == "cold" else len(report.schedule)
        )
    evaluation_state = json.loads(
        (progress_dir / "evaluations.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(evaluation_state["layout_results"]) == (
        len(ECOLOGY_P1_ARM_NAMES) * 6
    )

    # Rewind one arm to its penultimate immutable episode archive. Resume
    # must execute only the missing suffix and converge to the same report.
    learned_state_path = progress_dir / "learned.json"
    learned_state = json.loads(
        learned_state_path.read_text(encoding="utf-8")
    )
    penultimate = progress_dir / (
        f"learned.slot-{(len(report.schedule) - 1) % 2}.vzac"
    )
    learned_state.update(
        {
            "completed_training_episodes": len(report.schedule) - 1,
            "training_complete": False,
            "checkpoint_archive": penultimate.name,
            "checkpoint_sha256": hashlib.sha256(
                penultimate.read_bytes()
            ).hexdigest(),
        }
    )
    learned_state_path.write_text(
        json.dumps(
            learned_state,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    resumed = await run_ecology_p1(
        config,
        progress_dir=progress_dir,
    )
    assert resumed.to_dict() == report.to_dict()

    with pytest.raises(ValueError, match="progress mismatch"):
        await run_ecology_p1(
            EcologyP1Config(
                n_ants=1,
                temporal_latent_dim=4,
                training_rounds=1,
                evaluation_rounds=3,
                layouts_per_tier=1,
                seed=8,
            ),
            progress_dir=progress_dir,
        )
