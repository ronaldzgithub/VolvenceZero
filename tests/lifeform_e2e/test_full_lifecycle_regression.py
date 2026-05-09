"""Wave T11 e2e regression — full character lifecycle.

Exercises the entire 11-wave pipeline in one test:

* ExperientialReplayDriver runs the demo arc.
* compute_drive_shape_evolution proposes deltas.
* apply_drive_evolution_through_gate routes them through OFFLINE.
* save_lifeform_template writes the lived state.
* give_birth recovers a fresh Lifeform from disk.
* The reborn lifeform answers a turn.

This is the deterministic synthetic-substrate version of
``examples/character_full_lifecycle_demo.py``; the demo script
itself is also imported and asserted to load (rot-guard pattern).
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib

import pytest

from lifeform_domain_character import (
    ExperientialReplayDriver,
    apply_drive_evolution_through_gate,
    build_zhang_wuji_demo_arc,
    build_zhang_wuji_lifeform,
    compute_drive_shape_evolution,
    give_birth,
    save_lifeform_template,
    vitals_drive_levels_from_session,
)
from volvence_zero.evaluation.types import EvaluationScore, EvaluationSnapshot
from volvence_zero.memory import build_default_memory_store


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEMO_PATH = _REPO_ROOT / "examples" / "character_full_lifecycle_demo.py"


def _healthy_evaluation() -> EvaluationSnapshot:
    score = lambda n, v: EvaluationScore(
        family="test",
        metric_name=n,
        value=v,
        confidence=1.0,
        evidence="regression test",
    )
    return EvaluationSnapshot(
        turn_scores=(
            score("contract_integrity", 1.0),
            score("rollback_resilience", 0.95),
            score("fallback_reliance", 0.10),
        ),
        session_scores=(),
        alerts=(),
        description="regression healthy",
    )


def test_full_lifecycle_pipeline_end_to_end(tmp_path: pathlib.Path) -> None:
    """Replay → evolution → save → give_birth → reborn turn — all green.
    """
    memory_store = build_default_memory_store()
    bundle = build_zhang_wuji_lifeform(memory_store=memory_store)
    arc = build_zhang_wuji_demo_arc()

    driver = ExperientialReplayDriver()
    report = driver.run_arc(arc=arc, lifeform=bundle.lifeform)
    assert report.scenes_processed == len(arc.scenes)

    evolution = compute_drive_shape_evolution(
        replay_report=report, base_profile=bundle.profile
    )
    apply_result = apply_drive_evolution_through_gate(
        evolution=evolution,
        base_profile=bundle.profile,
        evaluation_snapshot=_healthy_evaluation(),
        validation_delta=0.10,
        capacity_cost=0.10,
        rollback_evidence=f"base profile {bundle.profile.version}",
    )

    snapshot_session = bundle.lifeform.create_session(
        session_id="lifecycle-vitals-capture"
    )

    async def _capture():
        await snapshot_session.run_turn("准备保存。")

    asyncio.run(_capture())
    drive_levels = vitals_drive_levels_from_session(snapshot_session)

    save_result = save_lifeform_template(
        profile=bundle.profile,
        evolved_profile=(
            apply_result.evolved_profile
            if apply_result.allowed
            else None
        ),
        template_id="zhang-wuji-lifecycle",
        output_dir=tmp_path,
        memory_store=memory_store,
        vitals_drive_levels=drive_levels,
        replay_report=report,
        source_arc_id=arc.arc_id,
        replay_provenance=f"lifecycle-test-{len(apply_result.allowed)}",
    )
    assert save_result.template_path.exists()

    rebirth = give_birth(save_result.template_path)
    assert rebirth.profile.profile_id == "zhang-wuji"

    # The reborn lifeform should answer a turn without raising.
    reborn_session = rebirth.lifeform.create_session(
        session_id="lifecycle-reborn"
    )

    async def _go():
        return await reborn_session.run_turn(
            "你重生之后第一感觉是什么？"
        )

    result = asyncio.run(_go())
    assert result.response.text.strip(), "reborn lifeform produced empty response"
    assert result.active_regime, "reborn lifeform did not pick a regime"


def test_full_lifecycle_demo_module_loads_cleanly() -> None:
    """Rot-guard: importing the demo script must not raise.

    Catches accidental syntactic / import-graph regressions in
    ``examples/character_full_lifecycle_demo.py`` without paying the
    cost of running the multi-minute demo every test session.
    """
    spec = importlib.util.spec_from_file_location(
        "character_full_lifecycle_demo", _DEMO_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Sanity: ``main`` is callable; we don't actually call it here.
    assert callable(module.main)


def test_full_lifecycle_demo_path_exists() -> None:
    assert _DEMO_PATH.exists(), (
        f"Wave T11 demo missing at {_DEMO_PATH}"
    )
