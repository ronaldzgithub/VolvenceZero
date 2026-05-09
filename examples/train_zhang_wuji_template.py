"""Pre-train + save a 张无忌 ``LifeformTemplate`` for the browser-chat demo.

Distilled from :mod:`examples.character_full_lifecycle_demo` — keeps
only the parts that produce a template file:

1. Build a base 张无忌 lifeform (synthetic substrate; no torch needed).
2. Run the 10-scene demo NarrativeArc through ExperientialReplayDriver.
3. Compute drive-shape evolution and route allowed deltas through the
   OFFLINE ModificationGate.
4. Save the lived state as a JSON LifeformTemplate.

The template is written to::

    artifacts/lifeform-templates/zhang-wuji-demo.json

(or wherever ``VZ_DEMO_OUTPUT_DIR`` points) and is the canonical
artifact that ``start_browser_chat_zhang_wuji.{sh,ps1}`` expects via
``ZHANG_WUJI_TEMPLATE_PATH``.

Why this exists separately from the full demo:

The full demo also calls ``give_birth`` and runs reborn turns to
verify reincarnation works end-to-end. That is the right thing for
an integration smoke test but adds 30+ seconds and prints irrelevant
output for a user who only wants the artifact. This script stops at
"template written"; the browser-chat path is the consumer.

Usage::

    python examples/train_zhang_wuji_template.py
    python examples/train_zhang_wuji_template.py --force  # overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys

from lifeform_domain_character import (
    ExperientialReplayDriver,
    apply_drive_evolution_through_gate,
    build_zhang_wuji_demo_arc,
    build_zhang_wuji_lifeform,
    compute_drive_shape_evolution,
    save_lifeform_template,
    vitals_drive_levels_from_session,
)
from volvence_zero.evaluation.types import EvaluationScore, EvaluationSnapshot
from volvence_zero.memory import build_default_memory_store


def _healthy_evaluation_snapshot() -> EvaluationSnapshot:
    """Synthetic 'healthy' evaluation so the OFFLINE gate accepts deltas.

    In production the kernel's evaluation pipeline produces this from
    the replay session itself; we synthesize one here so the demo is
    fully deterministic and does not depend on a live evaluator.
    """
    score = lambda name, value: EvaluationScore(
        family="train",
        metric_name=name,
        value=value,
        confidence=1.0,
        evidence="train-template synthetic",
    )
    return EvaluationSnapshot(
        turn_scores=(
            score("contract_integrity", 1.0),
            score("rollback_resilience", 0.95),
            score("fallback_reliance", 0.10),
        ),
        session_scores=(),
        alerts=(),
        description="train-template healthy evaluation",
    )


async def _run(*, output_path: pathlib.Path, force: bool) -> int:
    if output_path.exists() and not force:
        print(
            f"[train] template already exists at {output_path}; "
            "pass --force to retrain."
        )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[train] step 1/4 — building base 张无忌 lifeform")
    memory_store = build_default_memory_store()
    bundle = build_zhang_wuji_lifeform(memory_store=memory_store)
    profile = bundle.profile

    print(
        "[train] step 2/4 — replaying the 10-scene demo arc "
        "(this takes the longest, ~1-2 min on a synthetic substrate)"
    )
    arc = build_zhang_wuji_demo_arc()
    driver = ExperientialReplayDriver()
    replay_report = await driver.run_arc_async(
        arc=arc, lifeform=bundle.lifeform
    )
    print(
        f"  -> total PE signal: {replay_report.total_pe_signal:.4f}"
    )

    print(
        "[train] step 3/4 — proposing drive-shape evolution and "
        "routing through OFFLINE ModificationGate"
    )
    evolution = compute_drive_shape_evolution(
        replay_report=replay_report, base_profile=profile
    )
    apply_result = apply_drive_evolution_through_gate(
        evolution=evolution,
        base_profile=profile,
        evaluation_snapshot=_healthy_evaluation_snapshot(),
        validation_delta=0.10,
        capacity_cost=0.10,
        rollback_evidence=(
            f"base_profile.version={profile.version}; "
            f"arc_id={arc.arc_id}"
        ),
    )
    print(
        f"  -> deltas proposed={len(evolution.deltas)} "
        f"allowed={len(apply_result.allowed)} "
        f"blocked={len(apply_result.blocked)}"
    )

    print("[train] step 4/4 — saving LifeformTemplate JSON")
    snapshot_session = bundle.lifeform.create_session(
        session_id="train-vitals-capture"
    )
    await snapshot_session.run_turn(
        "训练即将完成，请你说一句留给后世的话。"
    )
    saved_levels = vitals_drive_levels_from_session(snapshot_session)

    template_id = output_path.stem  # e.g. "zhang-wuji-demo"
    save_result = save_lifeform_template(
        profile=profile,
        evolved_profile=(
            apply_result.evolved_profile
            if apply_result.allowed
            else None
        ),
        template_id=template_id,
        output_dir=output_path.parent,
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
    if save_result.template_path != output_path:
        # Rare: ``save_lifeform_template`` chose a different filename
        # (e.g. when ``template_id`` differs from the path stem). The
        # browser-chat path keys off ``output_path`` so we report
        # whichever is canonical.
        print(
            f"  -> wrote {save_result.template_path} "
            f"(requested {output_path})"
        )
    else:
        print(f"  -> wrote {save_result.template_path}")
    print()
    print("Done. To use this template in the browser chat, set:")
    print(f"    VERTICAL=zhang_wuji")
    print(f"    ZHANG_WUJI_TEMPLATE_PATH={save_result.template_path}")
    print(
        "and run start_browser_chat_qwen.sh / .ps1 (or use the "
        "start_browser_chat_zhang_wuji.* convenience wrappers)."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help=(
            "Output JSON path (default: "
            "artifacts/lifeform-templates/zhang-wuji-demo.json or "
            "$VZ_DEMO_OUTPUT_DIR/zhang-wuji-demo.json)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if the template already exists.",
    )
    args = parser.parse_args()

    if args.output is not None:
        output_path = args.output
    else:
        base_dir = pathlib.Path(
            os.environ.get(
                "VZ_DEMO_OUTPUT_DIR",
                "artifacts/lifeform-templates",
            )
        )
        output_path = base_dir / "zhang-wuji-demo.json"

    print("=" * 72)
    print(" Train + Save — 张无忌 LifeformTemplate")
    print(f" Output: {output_path}")
    print("=" * 72)
    return asyncio.run(_run(output_path=output_path, force=args.force))


if __name__ == "__main__":
    sys.exit(main())
