"""End-to-end demo of the full Lifeform Template + Birth pipeline.

Walks the entire 11-wave milestone in one script:

1. Load the reviewed 张无忌 profile (Wave C4 / T0).
2. Build a base lifeform with shared MemoryStore (Wave C5).
3. Run the demo NarrativeArc through ExperientialReplayDriver
   (Wave T1 + T2).
4. Compute drive-shape evolution proposals (Wave T9).
5. Apply allowed deltas through the OFFLINE ModificationGate
   (Wave T10), producing an evolved profile.
6. Save the lived state as a LifeformTemplate JSON file (Wave T5).
7. give_birth from that template into a fresh Lifeform (Wave T6).
8. Run a couple of turns on the reborn lifeform to verify it
   actually responds.

Designed to run on the synthetic substrate (no torch / Qwen
required); finishes in ~2-3 minutes on a modern laptop.

Usage:

    python examples/character_full_lifecycle_demo.py

The on-disk template is written to
``artifacts/lifeform-templates/zhang-wuji-demo.json`` by default.
Set ``VZ_DEMO_OUTPUT_DIR`` to override.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from dataclasses import dataclass

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


def _demo_evaluation_snapshot() -> EvaluationSnapshot:
    """Synthetic 'healthy' evaluation snapshot for the gate.

    In production this comes from the kernel's evaluation pipeline
    after the replay; here we synthesize the three metric values
    the OFFLINE gate reads so the demo can run fully deterministic.
    """
    score = lambda name, value: EvaluationScore(
        family="demo",
        metric_name=name,
        value=value,
        confidence=1.0,
        evidence="demo synthetic",
    )
    return EvaluationSnapshot(
        turn_scores=(
            score("contract_integrity", 1.0),
            score("rollback_resilience", 0.95),
            score("fallback_reliance", 0.10),
        ),
        session_scores=(),
        alerts=(),
        description="demo healthy evaluation",
    )


@dataclass(frozen=True)
class DemoOutcome:
    template_path: pathlib.Path
    replay_pe_total: float
    deltas_proposed: int
    deltas_allowed: int
    deltas_blocked: int
    reborn_first_response: str
    reborn_second_response: str


async def _run_demo(*, output_dir: pathlib.Path) -> DemoOutcome:
    profile = None
    memory_store = build_default_memory_store()
    print("[demo] step 1/8 — building base 张无忌 lifeform")
    bundle = build_zhang_wuji_lifeform(memory_store=memory_store)
    profile = bundle.profile

    print("[demo] step 2/8 — loading demo NarrativeArc (10 scenes)")
    arc = build_zhang_wuji_demo_arc()

    print(
        "[demo] step 3/8 — running ExperientialReplayDriver across "
        f"{len(arc.scenes)} scenes (this takes the longest, ~1-2 min)"
    )
    driver = ExperientialReplayDriver()
    replay_report = await driver.run_arc_async(
        arc=arc, lifeform=bundle.lifeform
    )
    print(
        f"  -> total PE signal: {replay_report.total_pe_signal:.4f}; "
        f"regime sequence_payoff growth: "
        f"{replay_report.regime_sequence_payoff_growth}"
    )

    print("[demo] step 4/8 — computing drive shape evolution proposals")
    evolution = compute_drive_shape_evolution(
        replay_report=replay_report, base_profile=profile
    )
    print(
        f"  -> proposed deltas: {len(evolution.deltas)}; "
        f"drives observed: {len(evolution.drives_observed)}"
    )

    print(
        "[demo] step 5/8 — routing deltas through OFFLINE "
        "ModificationGate"
    )
    apply_result = apply_drive_evolution_through_gate(
        evolution=evolution,
        base_profile=profile,
        evaluation_snapshot=_demo_evaluation_snapshot(),
        validation_delta=0.10,
        capacity_cost=0.10,
        rollback_evidence=(
            f"base_profile.version={profile.version}; "
            f"arc_id={arc.arc_id}"
        ),
    )
    print(
        f"  -> allowed: {len(apply_result.allowed)}, "
        f"blocked: {len(apply_result.blocked)}"
    )

    print("[demo] step 6/8 — saving the lived state as a LifeformTemplate")
    # Capture vitals just before save.
    snapshot_session = bundle.lifeform.create_session(
        session_id="demo-vitals-capture"
    )
    await snapshot_session.run_turn("准备保存模板的最后一句话。")
    saved_levels = vitals_drive_levels_from_session(snapshot_session)
    save_result = save_lifeform_template(
        profile=profile,
        evolved_profile=(
            apply_result.evolved_profile
            if apply_result.allowed
            else None
        ),
        template_id="zhang-wuji-demo",
        output_dir=output_dir,
        memory_store=memory_store,
        vitals_drive_levels=saved_levels,
        replay_report=replay_report,
        source_arc_id=arc.arc_id,
        replay_provenance=(
            f"zhang-wuji-demo-arc-v0 + drive_evolution(allowed="
            f"{len(apply_result.allowed)},blocked="
            f"{len(apply_result.blocked)})"
        ),
        overwrite_existing=True,
    )
    print(f"  -> wrote {save_result.template_path}")

    print("[demo] step 7/8 — calling give_birth on the saved template")
    rebirth = give_birth(save_result.template_path)
    print(
        "  -> reborn lifeform profile_id="
        f"{rebirth.profile.profile_id} "
        f"(evolved={apply_result.evolved_profile is not None})"
    )

    print("[demo] step 8/8 — running 2 turns on the reborn lifeform")
    reborn_session = rebirth.lifeform.create_session(session_id="reborn")
    first = await reborn_session.run_turn(
        "重生之后，请你介绍一下自己。"
    )
    second = await reborn_session.run_turn(
        "如果对方已经投降，你会怎么做？"
    )

    return DemoOutcome(
        template_path=save_result.template_path,
        replay_pe_total=replay_report.total_pe_signal,
        deltas_proposed=len(evolution.deltas),
        deltas_allowed=len(apply_result.allowed),
        deltas_blocked=len(apply_result.blocked),
        reborn_first_response=first.response.text[:200],
        reborn_second_response=second.response.text[:200],
    )


def main() -> int:
    output_dir = pathlib.Path(
        os.environ.get(
            "VZ_DEMO_OUTPUT_DIR",
            "artifacts/lifeform-templates",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(" Lifeform Template + Birth Pipeline — End-to-End Demo")
    print(f" Output: {output_dir}")
    print("=" * 72)
    outcome = asyncio.run(_run_demo(output_dir=output_dir))
    print()
    print("=" * 72)
    print(" Demo summary")
    print("=" * 72)
    print(f"  template_path        = {outcome.template_path}")
    print(f"  replay PE total      = {outcome.replay_pe_total:.4f}")
    print(f"  deltas proposed      = {outcome.deltas_proposed}")
    print(f"  deltas allowed       = {outcome.deltas_allowed}")
    print(f"  deltas blocked       = {outcome.deltas_blocked}")
    print()
    print("  reborn turn 1 response:")
    print(f"    {outcome.reborn_first_response}")
    print()
    print("  reborn turn 2 response:")
    print(f"    {outcome.reborn_second_response}")
    print()
    print("Lifecycle PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
